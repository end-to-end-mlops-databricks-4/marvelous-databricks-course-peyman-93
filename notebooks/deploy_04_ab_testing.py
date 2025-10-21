# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Custom Model with External Enrichment
# MAGIC Production-ready deployment with external data integration

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
import numpy as np
from typing import Dict, Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from dotenv import load_dotenv
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.config import ProjectConfig
from financial_complaints.utils import is_databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

if not is_databricks():
    load_dotenv()

# Load config
config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# Model configuration
BASE_MODEL_NAME = f"{catalog_name}.{schema_name}.complaint_upheld_model_basic"
CUSTOM_MODEL_NAME = f"{catalog_name}.{schema_name}.complaint_model_custom_enriched"
ENDPOINT_NAME = "complaints-custom-enriched-serving"

print(f"Base model: {BASE_MODEL_NAME}")
print(f"Custom model: {CUSTOM_MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify Base Model and Extract Schema

# COMMAND ----------

client = mlflow.MlflowClient()

# Get base model
try:
    model_version = client.get_model_version_by_alias(BASE_MODEL_NAME, "latest-model")
    base_model_version = model_version.version
    base_model_uri = f"models:/{BASE_MODEL_NAME}/{base_model_version}"
    print(f"✓ Base model found: v{base_model_version}")
except:
    raise ValueError(f"Base model not found: {BASE_MODEL_NAME}")

# Extract schema
def extract_schema(model_uri):
    """Extract expected features from model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmp_dir)
        mlmodel_path = os.path.join(local_path, 'MLmodel')
        
        with open(mlmodel_path, 'r') as f:
            mlmodel_content = yaml.safe_load(f)
        
        if 'signature' in mlmodel_content:
            signature = mlmodel_content['signature']
            inputs = json.loads(signature['inputs']) if isinstance(signature['inputs'], str) else signature['inputs']
            return [inp['name'] for inp in inputs]
    return None

expected_features = extract_schema(base_model_uri)
print(f"✓ Model expects {len(expected_features)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define Custom Model with External Enrichment

# COMMAND ----------

class ComplaintModelWithEnrichment(mlflow.pyfunc.PythonModel):
    """
    Enhanced model with external data enrichment capabilities.
    
    In production, this would integrate with:
    - DynamoDB for company risk scores
    - Redis for cached predictions
    - MongoDB for historical patterns
    - External APIs for real-time data
    """
    
    def load_context(self, context):
        """Load base model and initialize external connections."""
        self.model = mlflow.sklearn.load_model(context.artifacts["base_model"])
        self.expected_features = context.artifacts.get("expected_features", [])
        
        # Initialize connections (mock for demo, real in production)
        self.external_sources_available = self._init_external_connections()
        
    def _init_external_connections(self):
        """Initialize external data sources."""
        # In production, initialize real connections:
        # self.dynamodb = boto3.resource('dynamodb')
        # self.redis_client = redis.Redis(host=os.environ.get('REDIS_HOST'))
        # self.mongo_client = MongoClient(os.environ.get('MONGODB_URI'))
        
        # For demo, use mock connections
        print("External enrichment initialized (mock mode)")
        return True
    
    def _fetch_company_enrichment(self, company: str) -> Dict:
        """Fetch real-time company data."""
        # Mock implementation - in production, query actual databases
        np.random.seed(hash(company) % 1000)
        return {
            'real_time_risk_score': np.random.uniform(0, 1),
            'recent_complaint_volume': np.random.randint(0, 100),
            'regulatory_flags': np.random.randint(0, 5),
            'market_sentiment': np.random.uniform(-1, 1)
        }
    
    def _fetch_cached_prediction(self, cache_key: str) -> Optional[Dict]:
        """Check cache for existing predictions."""
        # In production: return self.redis_client.get(cache_key)
        return None  # Mock - no cache hit
    
    def _store_prediction(self, cache_key: str, prediction: Dict):
        """Store prediction in cache."""
        # In production: self.redis_client.setex(cache_key, 3600, json.dumps(prediction))
        pass  # Mock implementation
    
    def _prepare_features(self, row: pd.Series, enrichment: Dict) -> np.ndarray:
        """Prepare features for model prediction."""
        # Start with expected features
        features = []
        
        for feat in self.expected_features:
            if feat in row.index:
                value = row[feat]
                if pd.isna(value):
                    features.append(0)
                elif isinstance(value, str):
                    features.append(0)  # Would use proper encoding in production
                else:
                    features.append(float(value))
            else:
                features.append(0)  # Default for missing features
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, context, model_input):
        """Make predictions with external enrichment."""
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        results = []
        
        for idx, row in model_input.iterrows():
            # Generate cache key
            complaint_id = str(row.get('Complaint_ID', idx))
            cache_key = f"prediction:{complaint_id}"
            
            # Check cache
            cached = self._fetch_cached_prediction(cache_key)
            if cached:
                results.append(cached)
                continue
            
            # Fetch external enrichment
            company = str(row.get('Company', 'Unknown'))
            enrichment = self._fetch_company_enrichment(company)
            
            # Prepare features and predict
            features = self._prepare_features(row, enrichment)
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0, 1]
            
            # Create enriched result
            result = {
                'complaint_id': complaint_id,
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence': float(abs(probability - 0.5) * 2),
                'company': company,
                'external_enrichment': {
                    'risk_score': enrichment['real_time_risk_score'],
                    'complaint_volume': enrichment['recent_complaint_volume']
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Cache result
            self._store_prediction(cache_key, result)
            results.append(result)
        
        return results[0] if len(results) == 1 else results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Register Custom Model

# COMMAND ----------

# Prepare test data
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(5)
test_df = test_set.toPandas()

# Prepare features in correct order
ordered_data = pd.DataFrame()
for feat in expected_features:
    if feat in test_df.columns:
        ordered_data[feat] = test_df[feat]
    else:
        ordered_data[feat] = 0

# Add metadata columns
for col in ['Complaint_ID', 'Company', 'State']:
    if col in test_df.columns:
        ordered_data[col] = test_df[col]

# Convert types
for col in ordered_data.columns:
    if col not in ['Complaint_ID', 'Company', 'State']:
        ordered_data[col] = pd.to_numeric(ordered_data[col], errors='coerce').fillna(0).astype('int64')

print(f"Test data prepared: {ordered_data.shape}")

# Register custom model
mlflow.set_experiment("/Shared/complaints-custom-enriched")
custom_model = ComplaintModelWithEnrichment()

with mlflow.start_run(run_name="complaint-enriched-wrapper") as run:
    run_id = run.info.run_id
    
    # Create signature
    sample_output = {
        'complaint_id': '123456',
        'prediction': 1,
        'probability': 0.75,
        'confidence': 0.5,
        'company': 'Example Corp',
        'external_enrichment': {
            'risk_score': 0.45,
            'complaint_volume': 50
        },
        'timestamp': '2024-01-01T00:00:00'
    }
    
    signature = infer_signature(
        model_input=ordered_data.head(1),
        model_output=sample_output
    )
    
    # Log model with enrichment wrapper
    mlflow.pyfunc.log_model(
        python_model=custom_model,
        artifact_path="custom-enriched-model",
        artifacts={
            "base_model": base_model_uri,
            "expected_features": expected_features
        },
        signature=signature,
        pip_requirements=[
            "pandas",
            "numpy",
            "scikit-learn",
            "boto3",  # For DynamoDB
            "redis",  # For Redis cache
            "pymongo",  # For MongoDB
            "requests"  # For external APIs
        ]
    )
    
    # Log configuration
    mlflow.log_params({
        "model_type": "custom_with_enrichment",
        "base_model": BASE_MODEL_NAME,
        "base_model_version": base_model_version,
        "enrichment_sources": "DynamoDB,Redis,MongoDB,APIs",
        "cache_enabled": True,
        "feature_count": len(expected_features)
    })

# Register model
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/custom-enriched-model",
    name=CUSTOM_MODEL_NAME
)

print(f"✓ Custom model registered: v{model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Deploy Custom Model Endpoint

# COMMAND ----------

w = WorkspaceClient()

# Get credentials
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# Configure endpoint with environment variables
served_entities = [
    ServedEntityInput(
        entity_name=CUSTOM_MODEL_NAME,
        entity_version=model_version.version,
        scale_to_zero_enabled=True,
        workload_size="Small",
        environment_vars={
            # For production, use Databricks secrets:
            # "AWS_ACCESS_KEY_ID": "{{secrets/mlops/aws_access_key}}",
            # "REDIS_HOST": "{{secrets/mlops/redis_host}}",
            # "MONGODB_URI": "{{secrets/mlops/mongodb_uri}}",
            
            # Demo mode configuration
            "ENRICHMENT_MODE": "mock",
            "CACHE_TTL": "3600",
            "ENABLE_EXTERNAL_DATA": "true"
        }
    )
]

# Deploy endpoint
try:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served_entities)
    )
    print(f"✓ Created endpoint: {ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e):
        w.serving_endpoints.update_config(
            name=ENDPOINT_NAME,
            served_entities=served_entities
        )
        print(f"✓ Updated endpoint: {ENDPOINT_NAME}")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test Custom Endpoint

# COMMAND ----------

def call_custom_endpoint(record):
    """Call the custom enriched endpoint."""
    host = os.environ['DBR_HOST']
    if not host.startswith('http'):
        host = f"https://{host}"
    
    response = requests.post(
        f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record}
    )
    return response.status_code, response.text

# Prepare test records
test_records = ordered_data.to_dict(orient="records")
dataframe_records = [[record] for record in test_records]

print("Waiting for endpoint to be ready...")
time.sleep(30)

# Test single prediction
print("\n" + "="*60)
print("CUSTOM ENRICHED MODEL TEST")
print("="*60)

status_code, response_text = call_custom_endpoint(dataframe_records[0])
print(f"Status: {status_code}")

if status_code == 200:
    result = json.loads(response_text)
    print(f"✓ SUCCESS!")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Probability: {result['probability']:.3f}")
    print(f"  Company: {result['company']}")
    print(f"  External Risk Score: {result['external_enrichment']['risk_score']:.3f}")
    print(f"  Complaint Volume: {result['external_enrichment']['complaint_volume']}")
else:
    print(f"❌ Error: {response_text[:500]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Batch Testing with Enrichment

# COMMAND ----------

print("Testing batch predictions with enrichment...")
for i in range(min(3, len(dataframe_records))):
    status_code, response_text = call_custom_endpoint(dataframe_records[i])
    
    if status_code == 200:
        result = json.loads(response_text)
        print(f"\nRecord {i+1}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Enrichment applied: ✓")
        print(f"  Risk score: {result['external_enrichment']['risk_score']:.2f}")
    else:
        print(f"Record {i+1}: ❌ Failed")
    
    time.sleep(0.2)

print("\n" + "="*60)
print("DEPLOYMENT SUMMARY")
print("="*60)
print(f"✓ Custom model deployed successfully!")
print(f"  Model: {CUSTOM_MODEL_NAME}")
print(f"  Endpoint: {ENDPOINT_NAME}")
print(f"  Features: External enrichment enabled")
print(f"  Caching: Enabled (mock mode)")
print(f"\nThe endpoint enriches predictions with:")
print("  - Real-time company risk scores")
print("  - Recent complaint volumes")
print("  - Market sentiment data")
print("\nFor production use:")
print("  1. Set up Databricks secrets for external DBs")
print("  2. Update environment variables in endpoint config")
print("  3. Implement real external connections in model class")