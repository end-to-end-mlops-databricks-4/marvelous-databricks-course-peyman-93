# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Feature Engineering Model with Feature Lookups
# MAGIC Production-ready deployment for FE models with online feature serving

# COMMAND ----------

import os
import sys
import time
import json
import yaml
import tempfile
import requests
import mlflow
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from dotenv import load_dotenv

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.config import ProjectConfig
from financial_complaints.serving.fe_model_serving import FeatureLookupServing
from financial_complaints.utils import is_databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

if not is_databricks():
    load_dotenv()

# Setup workspace and tokens
w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# Load config
config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# FE Model configuration
model_name = f"{catalog_name}.{schema_name}.complaint_fe_model"
endpoint_name = "complaints-model-fe-serving"
company_table = f"{catalog_name}.{schema_name}.company_features"
state_table = f"{catalog_name}.{schema_name}.state_features"
online_store_name = "complaints-predictions"

print(f"FE Model: {model_name}")
print(f"Endpoint: {endpoint_name}")
print(f"Company features: {company_table}")
print(f"State features: {state_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify Model and Feature Tables

# COMMAND ----------

client = mlflow.MlflowClient()

# Verify FE model exists
try:
    model_version = client.get_model_version_by_alias(model_name, "latest-model")
    model_version_number = model_version.version
    print(f"✓ FE model found: {model_name} v{model_version_number}")
except:
    try:
        model_version = client.get_model_version_by_alias(model_name, "latest-fe-model")
        model_version_number = model_version.version
        print(f"✓ FE model found: {model_name} v{model_version_number}")
    except Exception as e:
        raise ValueError(f"FE model not found: {model_name}. Run train_register_fe_model.py first")

# Verify feature tables
for table_name, table_desc in [(company_table, "Company"), (state_table, "State")]:
    try:
        count = spark.table(table_name).count()
        print(f"✓ {table_desc} features table: {count:,} records")
    except:
        raise ValueError(f"{table_desc} features table not found: {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Extract Model Schema

# COMMAND ----------

def extract_fe_model_schema(model_name, version):
    """Extract schema from FE model, handling feature lookups."""
    model_uri = f"models:/{model_name}/{version}"
    
    # First try to get input example or signature
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmp_dir)
            
            # Check for input_example
            input_example_path = os.path.join(local_path, 'input_example.json')
            if os.path.exists(input_example_path):
                with open(input_example_path, 'r') as f:
                    example = json.load(f)
                    if 'columns' in example:
                        print(f"✓ Found input example with {len(example['columns'])} features")
                        return example['columns']
    except:
        pass
    
    # FE models often need fewer features (lookup keys + base features)
    # Return expected lookup keys based on config
    base_features = [
        "Company",
        "State", 
        "Product",
        "Sub_product",
        "Issue",
        "Sub_issue",
        "Submitted_via",
        "Consumer_consent_provided",
        "Timely_response",
        "consumer_disputed",
        "processing_days",
        "has_narrative",
        "Date_received",
        "Complaint_ID"
    ]
    
    print(f"✓ Using standard FE model input features: {len(base_features)} features")
    return base_features

expected_features = extract_fe_model_schema(model_name, model_version_number)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Setup Online Store

# COMMAND ----------

fe = FeatureEngineeringClient()

# Create or get online store
try:
    online_store = fe.get_online_store(name=online_store_name)
    print(f"✓ Online store exists: {online_store_name}")
except:
    print(f"Creating online store: {online_store_name}...")
    fe.create_online_store(
        name=online_store_name,
        capacity="CU_1"
    )
    online_store = fe.get_online_store(name=online_store_name)
    print(f"✓ Online store created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Publish Feature Tables to Online Store

# COMMAND ----------

feature_model_server = FeatureLookupServing(
    model_name=model_name,
    endpoint_name=endpoint_name,
    company_table=company_table,
    state_table=state_table
)

# Publish tables
for table_name in [company_table, state_table]:
    print(f"Publishing {table_name} to online store...")
    feature_model_server.create_or_update_online_table(
        online_store=online_store,
        table_name=table_name
    )
    print(f"✓ Published {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Deploy FE Model Endpoint

# COMMAND ----------

print(f"Deploying FE model endpoint: {endpoint_name}")
feature_model_server.deploy_or_update_serving_endpoint(
    version=model_version_number,
    workload_size="Small",
    scale_to_zero=True,
    wait=True
)
print("✓ FE model endpoint deployed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Prepare Test Data

# COMMAND ----------

def prepare_fe_test_data(spark, config, expected_features, limit=10):
    """Prepare test data for FE model (needs fewer features due to lookups)."""
    
    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(limit)
    test_df = test_set.toPandas()
    
    # Select only the features needed for FE model
    available_features = [f for f in expected_features if f in test_df.columns]
    test_data = test_df[available_features].copy()
    
    # Fill missing values appropriately
    for col in test_data.columns:
        if test_data[col].dtype == 'object':
            test_data[col] = test_data[col].fillna('MISSING')
        elif col in ['Date_received', 'Complaint_ID']:
            # Keep these as-is
            pass
        else:
            test_data[col] = pd.to_numeric(test_data[col], errors='coerce').fillna(0)
    
    print(f"✓ Prepared FE test data: {test_data.shape}")
    print(f"  Features: {list(test_data.columns)}")
    
    return test_data

test_df = prepare_fe_test_data(spark, config, expected_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test the FE Endpoint

# COMMAND ----------

def call_fe_endpoint(record, endpoint_name):
    """Call the FE model endpoint."""
    host = os.environ['DBR_HOST']
    if not host.startswith('http'):
        host = f"https://{host}"
    
    response = requests.post(
        f"{host}/serving-endpoints/{endpoint_name}/invocations",
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

# Create test payloads
test_records = test_df.to_dict(orient="records")
dataframe_records = [[record] for record in test_records]

print("Waiting for endpoint to be ready...")
time.sleep(30)

# Test single prediction
print("\n" + "="*60)
print("FE MODEL ENDPOINT TEST")
print("="*60)

status_code, response_text = call_fe_endpoint(dataframe_records[0], endpoint_name)
print(f"Status: {status_code}")

if status_code == 200:
    result = json.loads(response_text)
    print(f"✓ SUCCESS! Prediction: {result}")
else:
    print(f"❌ Error: {response_text[:500]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Batch Testing

# COMMAND ----------

print("Testing batch predictions with feature lookups...")
for i in range(min(3, len(dataframe_records))):
    status_code, response_text = call_fe_endpoint(dataframe_records[i], endpoint_name)
    
    if status_code == 200:
        print(f"Record {i+1}: ✓ Success")
    else:
        print(f"Record {i+1}: ❌ Error")
    
    time.sleep(0.2)

print("\n✓ FE model deployment completed!")
print(f"The endpoint enriches requests with:")
print(f"  - Company features from {company_table}")
print(f"  - State features from {state_table}")
print(f"  - Real-time lookups via online store")