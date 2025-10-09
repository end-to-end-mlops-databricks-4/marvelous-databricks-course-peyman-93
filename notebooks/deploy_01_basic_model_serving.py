# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Basic Model Serving Endpoint
# MAGIC Production-ready deployment script for registered ML models

# COMMAND ----------

# Install packages if needed
# %pip install -e ..
# %restart_python

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
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from dotenv import load_dotenv

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.config import ProjectConfig
from financial_complaints.serving.model_serving import ModelServing
from financial_complaints.utils import is_databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Setup environment
if not is_databricks():
    load_dotenv()

# Create workspace client and tokens
w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# Load project config
config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# Model configuration
model_name = f"{catalog_name}.{schema_name}.complaint_upheld_model_basic"
endpoint_name = "complaints-model-basic-serving"

print(f"Catalog: {catalog_name}")
print(f"Schema: {schema_name}")
print(f"Model: {model_name}")
print(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify Model Exists

# COMMAND ----------

client = mlflow.MlflowClient()

# Check if model exists and get version
try:
    model_version = client.get_model_version_by_alias(model_name, "latest-model")
    model_version_number = model_version.version
    print(f"✓ Model found: {model_name}")
    print(f"  Version: {model_version_number}")
    print(f"  Status: {model_version.status}")
except Exception as e:
    print(f"❌ Model not found with 'latest-model' alias")
    # Try to get version 1 as fallback
    try:
        model_version = client.get_model_version(model_name, "1")
        model_version_number = "1"
        print(f"✓ Using model version 1")
    except:
        raise ValueError(f"Model {model_name} not found. Please train and register the model first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Extract Model Schema

# COMMAND ----------

def extract_model_schema(model_name, version):
    """Extract the expected feature schema from the model signature."""
    model_uri = f"models:/{model_name}/{version}"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download model artifacts
        local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmp_dir)
        
        # Read MLmodel file for signature
        mlmodel_path = os.path.join(local_path, 'MLmodel')
        with open(mlmodel_path, 'r') as f:
            mlmodel_content = yaml.safe_load(f)
        
        # Extract feature names from signature
        if 'signature' in mlmodel_content:
            signature = mlmodel_content['signature']
            inputs = json.loads(signature['inputs']) if isinstance(signature['inputs'], str) else signature['inputs']
            
            feature_names = [inp['name'] for inp in inputs]
            feature_types = [inp['type'] for inp in inputs]
            
            print(f"✓ Extracted schema: {len(feature_names)} features")
            print(f"  First 5 features: {feature_names[:5]}")
            
            return feature_names, feature_types
        else:
            raise ValueError("Model signature not found")

# Get expected features
expected_features, feature_types = extract_model_schema(model_name, model_version_number)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Deploy Model Serving Endpoint

# COMMAND ----------

# Initialize model serving manager
model_serving = ModelServing(
    model_name=model_name,
    endpoint_name=endpoint_name
)

# Deploy or update the endpoint
print(f"Deploying endpoint '{endpoint_name}'...")
model_serving.deploy_or_update_serving_endpoint(
    version=model_version_number,
    workload_size="Small",
    scale_to_zero=True,
    wait=True
)
print("✓ Model serving endpoint deployed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Prepare Test Data

# COMMAND ----------

def prepare_test_data(spark, config, expected_features, limit=10):
    """Prepare test data with the exact schema the model expects."""
    
    # Load test set
    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(limit)
    test_df = test_set.toPandas()
    
    print(f"Loaded {len(test_df)} test records")
    print(f"Available columns: {len(test_df.columns)}")
    
    # Create DataFrame with exact features in correct order
    ordered_data = pd.DataFrame()
    missing_features = []
    
    for feat in expected_features:
        if feat in test_df.columns:
            ordered_data[feat] = test_df[feat]
        else:
            # Feature not in test set, use default value
            ordered_data[feat] = 0
            missing_features.append(feat)
    
    if missing_features:
        print(f"⚠️ {len(missing_features)} features not in test set (using defaults)")
    
    # Convert all columns to int64 (matching 'long' type expected by model)
    for col in ordered_data.columns:
        ordered_data[col] = pd.to_numeric(ordered_data[col], errors='coerce').fillna(0).astype('int64')
    
    print(f"✓ Prepared test data: {ordered_data.shape}")
    print(f"  All features present: {len(missing_features) == 0}")
    print(f"  All int64 types: {all(str(ordered_data[col].dtype) == 'int64' for col in ordered_data.columns)}")
    
    return ordered_data

# Prepare test data
test_df = prepare_test_data(spark, config, expected_features, limit=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test the Endpoint

# COMMAND ----------

def call_endpoint(record, endpoint_name):
    """Call the model serving endpoint."""
    host = os.environ['DBR_HOST']
    if not host.startswith('http'):
        host = f"https://{host}"
    
    serving_endpoint = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

# Create test payloads
test_records = test_df.to_dict(orient="records")
dataframe_records = [[record] for record in test_records]

print(f"Testing with {len(dataframe_records)} records")

# Wait for endpoint to be ready
print("Waiting for endpoint to be ready...")
time.sleep(30)

# Test single prediction
print("\n" + "="*60)
print("SINGLE PREDICTION TEST")
print("="*60)

status_code, response_text = call_endpoint(dataframe_records[0], endpoint_name)
print(f"Status: {status_code}")

if status_code == 200:
    result = json.loads(response_text)
    print(f"✓ SUCCESS! Prediction: {result['predictions'][0]}")
else:
    print(f"❌ Error: {response_text[:500]}")
    raise Exception("Endpoint test failed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Batch Testing

# COMMAND ----------

# Test multiple predictions
print("="*60)
print("BATCH PREDICTION TEST")
print("="*60)

predictions = []
for i in range(min(5, len(dataframe_records))):
    status_code, response_text = call_endpoint(dataframe_records[i], endpoint_name)
    
    if status_code == 200:
        result = json.loads(response_text)
        prediction = result['predictions'][0]
        predictions.append(prediction)
        print(f"Record {i+1}: Prediction = {prediction}")
    else:
        print(f"Record {i+1}: ERROR - {response_text[:100]}")
    
    time.sleep(0.2)  # Rate limiting

# Summary
if predictions:
    print(f"\n✓ Batch testing complete")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Class 0 (not upheld): {predictions.count(0.0)}")
    print(f"  Class 1 (upheld): {predictions.count(1.0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Endpoint Information

# COMMAND ----------

# Display endpoint details for external use
print("="*60)
print("ENDPOINT DEPLOYMENT SUMMARY")
print("="*60)
print(f"Model: {model_name}")
print(f"Version: {model_version_number}")
print(f"Endpoint: {endpoint_name}")
print(f"Status: ACTIVE")
print(f"\nEndpoint URL:")
print(f"{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations")
print(f"\nRequired features: {len(expected_features)}")
print(f"Feature order: {expected_features[:5]} ... (and {len(expected_features)-5} more)")

print("\n" + "="*60)
print("USAGE EXAMPLE")
print("="*60)
print("""
# Python example:
import requests

url = "<endpoint_url>"
headers = {"Authorization": "Bearer <your_token>"}
data = {"dataframe_records": [[<54_feature_values>]]}

response = requests.post(url, headers=headers, json=data)
prediction = response.json()['predictions'][0]
""")

print("\n✓ Model serving deployment completed successfully!")