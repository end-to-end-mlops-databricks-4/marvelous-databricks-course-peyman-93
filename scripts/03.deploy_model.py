"""Deploy Financial Complaints Feature Engineering Model to Serving Endpoint.

This script deploys the trained model to a serving endpoint with feature lookup capabilities.
It publishes feature tables to the online store and creates/updates the model serving endpoint.

Usage:
    python 03.deploy_model.py \
        --root_path /path/to/project \
        --env dev \
        --is_test 0
"""

import argparse
import time

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from financial_complaints.config import ProjectConfig
from financial_complaints.serving.fe_model_serving import FeatureLookupServing


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--is_test",
    action="store",
    default=0,
    type=int,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
is_test = args.is_test
config_path = f"{root_path}/files/project_config.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"complaints-model-serving-fe-{args.env}"

logger.info(f"Deploying model version: {model_version}")
logger.info(f"Catalog: {catalog_name}")
logger.info(f"Schema: {schema_name}")
logger.info(f"Endpoint: {endpoint_name}")

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Create online store if it doesn't exist
online_store_name = "financial-complaints-online-store"
online_store = None

try:
    online_store = fe.get_online_store(name=online_store_name)
    logger.info(f"Online store '{online_store_name}' found.")
except Exception as e:
    logger.info(f"Online store not found, will create it. Error: {e}")

if online_store is None:
    logger.info(f"Creating online store '{online_store_name}'...")
    try:
        fe.create_online_store(
            name=online_store_name,
            capacity="CU_1"
        )
        logger.info(f"Online store creation initiated.")
        # Wait a bit for creation
        time.sleep(10)
        online_store = fe.get_online_store(name=online_store_name)
        logger.info(f"Online store '{online_store_name}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create online store: {e}")
        raise

if online_store is None:
    raise RuntimeError(f"Online store '{online_store_name}' is None after creation attempt!")

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.complaint_fe_model",
    endpoint_name=endpoint_name,
    company_table=f"{catalog_name}.{schema_name}.company_features",
    state_table=f"{catalog_name}.{schema_name}.state_features",
)

# Publish company features to online store
logger.info("Publishing company_features to online store...")
feature_model_server.create_or_update_online_table(
    online_store=online_store,
    table_name=f"{catalog_name}.{schema_name}.company_features"
)
logger.info("Company features published.")

# Publish state features to online store
logger.info("Publishing state_features to online store...")
feature_model_server.create_or_update_online_table(
    online_store=online_store,
    table_name=f"{catalog_name}.{schema_name}.state_features"
)
logger.info("State features published.")

# CRITICAL FIX: Publish text features to online store
logger.info("Publishing text_features to online store...")
feature_model_server.create_or_update_online_table(
    online_store=online_store,
    table_name=f"{catalog_name}.{schema_name}.text_features"
)
logger.info("Text features published.")

# Wait for online tables to sync (give them time to be ready)
logger.info("Waiting 60 seconds for online tables to sync...")
time.sleep(300)
logger.info("Online tables should be ready now.")

# Deploy the model serving endpoint with feature lookup
logger.info("Deploying model serving endpoint...")
feature_model_server.deploy_or_update_serving_endpoint(version=model_version)
logger.info("Started deployment/update of the serving endpoint.")
logger.info(f"Endpoint name: {endpoint_name}")
logger.info("Note: Endpoint deployment is asynchronous. Check Databricks UI for status.")

# Delete endpoint if test
if is_test == 1:
    logger.info("Test mode: Cleaning up serving endpoint...")
    workspace = WorkspaceClient()
    workspace.serving_endpoints.delete(name=endpoint_name)
    logger.info("Serving endpoint deleted.")