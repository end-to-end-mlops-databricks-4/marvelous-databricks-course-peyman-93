"""Train and Register Financial Complaints Feature Engineering Model.

This script trains a machine learning model with feature engineering capabilities:
1. Creates feature tables (company, state, text features)
2. Defines feature functions for dynamic feature generation
3. Loads data from Unity Catalog
4. Creates training set with feature lookups and functions
5. Trains model and evaluates performance
6. Registers model to MLflow if performance improves

Usage:
    python 02.train_register_fe_model.py \
        --root_path /path/to/project \
        --env dev \
        --git_sha abc123 \
        --job_run_id 456 \
        --branch main \
        --is_test 0
"""

import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from financial_complaints.config import ProjectConfig, Tags
from financial_complaints.models.financial_complaints_feature_lookup_model import (
    FinancialComplaintsFeatureLookupModel,
)

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
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
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
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
fe_model = FinancialComplaintsFeatureLookupModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# Create feature tables (RUN ONCE - then comment out after first successful run)
# Uncomment these lines if you need to create the feature tables from scratch:
#fe_model.create_company_features_table()
#fe_model.create_state_features_table()
#fe_model.create_text_features_table()
#logger.info("Feature tables created.")

# Update feature tables with latest data (run every time)
fe_model.update_feature_tables()
logger.info("Feature tables updated.")

# Define feature functions
fe_model.define_feature_functions()

# Load data
fe_model.load_data()
logger.info("Data loaded.")

# Create training set with feature lookups
fe_model.create_training_set()

# Train the model
fe_model.train()
logger.info("Model training completed.")

# Evaluate model on test set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)

# Drop feature lookup columns that will be looked up from feature tables
company_features_to_drop = [
    'company_complaint_count', 
    'company_avg_processing_days',
    'company_median_processing_days', 
    'company_processing_std',
    'company_reliability_score'
]
state_features_to_drop = [
    'state_complaint_count',
    'state_avg_processing_days',
    'state_company_diversity',
    'state_regulatory_score'
]

cols_to_drop = [col for col in company_features_to_drop + state_features_to_drop
                if col in test_set.columns]

if cols_to_drop:
    test_set = test_set.drop(*cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} feature columns from test set")

model_improved = fe_model.model_improved(test_set=test_set)
logger.info(f"Model evaluation completed, model improved: {model_improved}")

is_test = args.is_test

# When running test, always register and deploy
if is_test == 1:
    model_improved = True

if model_improved:
    # Register the model
    latest_version = fe_model.register_model()
    logger.info(f"New model registered with version: {latest_version}")
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    # CRITICAL FIX: Use string "true" instead of integer 1
    dbutils.jobs.taskValues.set(key="model_updated", value="true")
else:
    # CRITICAL FIX: Use string "false" instead of integer 0
    dbutils.jobs.taskValues.set(key="model_updated", value="false")