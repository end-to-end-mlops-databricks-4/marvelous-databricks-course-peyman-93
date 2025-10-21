# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Feature Serving Endpoint
# MAGIC Serves pre-computed features/predictions without model inference

# COMMAND ----------

import os
import sys
import time
import json
import requests
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from dotenv import load_dotenv

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.config import ProjectConfig
from financial_complaints.serving.feature_serving import FeatureServing
from financial_complaints.utils import is_databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

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

# Feature serving configuration
feature_table_name = f"{catalog_name}.{schema_name}.complaint_predictions"
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"
endpoint_name = "complaints-feature-serving"
online_store_name = "complaints-predictions"

print(f"Feature table: {feature_table_name}")
print(f"Feature spec: {feature_spec_name}")
print(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create or Verify Feature Table
# MAGIC 
# MAGIC Note: This assumes you've already run batch predictions and stored them in the feature table.
# MAGIC If not, run this step to create sample predictions.

# COMMAND ----------

# Check if feature table exists
try:
    predictions_df = spark.table(feature_table_name)
    record_count = predictions_df.count()
    print(f"✓ Feature table exists: {record_count:,} records")
    
    # Show schema
    print("\nTable schema:")
    for field in predictions_df.schema.fields[:10]:
        print(f"  - {field.name}: {field.dataType}")
        
except Exception as e:
    print(f"⚠️ Feature table not found: {feature_table_name}")
    print("Creating sample predictions table...")
    
    # Create sample predictions from existing data
    import mlflow
    
    # Load a model for predictions
    model_name = f"{catalog_name}.{schema_name}.complaint_upheld_model_basic"
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}@latest-model")
        
        # Get test data
        test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").limit(1000)
        test_df = test_set.toPandas()
        
        # Prepare features (using numeric columns)
        numeric_cols = [col for col in test_df.columns 
                       if test_df[col].dtype in ['int64', 'float64'] 
                       and col not in ['Complaint_ID', 'complaint_upheld']]
        X = test_df[numeric_cols[:54]].fillna(0)  # Model expects 54 features
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Create predictions DataFrame
        predictions_df = test_df[['Complaint_ID', 'Company', 'State', 'processing_days']].copy()
        predictions_df['predicted_upheld'] = predictions
        predictions_df['predicted_probability'] = probabilities
        predictions_df['company_complaint_count'] = test_df.groupby('Company')['Company'].transform('count')
        predictions_df['state_complaint_count'] = test_df.groupby('State')['State'].transform('count')
        
        # Save as table
        spark_preds = spark.createDataFrame(predictions_df)
        spark_preds.write.mode("overwrite").saveAsTable(feature_table_name)
        
        # Enable Change Data Feed
        spark.sql(f"""
            ALTER TABLE {feature_table_name}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        
        print(f"✓ Created feature table with {len(predictions_df)} predictions")
        
    except Exception as e:
        print(f"Could not create predictions: {e}")
        raise ValueError("Please create the feature table first with batch predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Setup Online Store and Feature Spec

# COMMAND ----------

# Initialize feature serving manager
feature_serving = FeatureServing(
    feature_table_name=feature_table_name,
    feature_spec_name=feature_spec_name,
    endpoint_name=endpoint_name
)

# Create/update online table
print("Publishing to online store...")
feature_serving.create_or_update_online_table(online_store_name=online_store_name)
print("✓ Online table ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Feature Specification

# COMMAND ----------

# Get available features from the table
predictions_df = spark.table(feature_table_name)
available_columns = predictions_df.columns

# Select features to serve (exclude metadata)
features_to_serve = [
    col for col in available_columns 
    if col not in ['Complaint_ID', 'update_timestamp', 'update_timestamp_utc']
]

# Prioritize important features if too many
if len(features_to_serve) > 10:
    priority_features = [
        'predicted_upheld', 
        'predicted_probability',
        'company_complaint_count', 
        'state_complaint_count',
        'processing_days',
        'Company',
        'State'
    ]
    features_to_serve = [f for f in priority_features if f in features_to_serve]

print(f"Creating feature spec with {len(features_to_serve)} features:")
for f in features_to_serve:
    print(f"  - {f}")

# Create feature specification
feature_serving.create_feature_spec(feature_names=features_to_serve)
print("✓ Feature spec created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Deploy Feature Serving Endpoint

# COMMAND ----------

print(f"Deploying feature serving endpoint: {endpoint_name}")
feature_serving.deploy_or_update_serving_endpoint(
    workload_size="Small",
    scale_to_zero=True
)
print("✓ Feature serving endpoint deployed")

# Wait for deployment
print("Waiting for endpoint to be ready...")
time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test Feature Serving

# COMMAND ----------

def test_feature_endpoint(complaint_id, endpoint_name):
    """Test the feature serving endpoint."""
    host = os.environ['DBR_HOST']
    if not host.startswith('http'):
        host = f"https://{host}"
    
    response = requests.post(
        f"{host}/serving-endpoints/{endpoint_name}/invocations",
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": [{"Complaint_ID": str(complaint_id)}]},
    )
    return response

# Get sample IDs from the feature table
sample_ids = predictions_df.select("Complaint_ID").limit(5).collect()
sample_ids = [row['Complaint_ID'] for row in sample_ids]

print("="*60)
print("FEATURE SERVING TEST")
print("="*60)

if sample_ids:
    # Test single lookup
    response = test_feature_endpoint(sample_ids[0], endpoint_name)
    print(f"Test with ID {sample_ids[0]}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ SUCCESS! Retrieved features")
        print(f"Response: {json.dumps(result, indent=2)[:500]}...")
    else:
        print(f"❌ Error: {response.text[:500]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Batch Feature Lookups

# COMMAND ----------

print("Testing batch feature lookups...")
for i, complaint_id in enumerate(sample_ids[:3]):
    response = test_feature_endpoint(complaint_id, endpoint_name)
    
    if response.status_code == 200:
        print(f"ID {complaint_id}: ✓ Retrieved")
    else:
        print(f"ID {complaint_id}: ❌ Failed")
    
    time.sleep(0.2)

print("\n✓ Feature serving deployment completed!")
print(f"Endpoint serves pre-computed features from: {feature_table_name}")
print("Use this for low-latency feature lookups without model inference")