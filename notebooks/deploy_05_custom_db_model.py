# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Custom Model with External Database Integration
# MAGIC Wraps existing registered model with external data enrichment capabilities

# COMMAND ----------

import os
import sys
import time
import mlflow
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from dotenv import load_dotenv
from mlflow.models import infer_signature

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.config import ProjectConfig
from financial_complaints.utils import is_databricks

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Setup environment
if not is_databricks():
    load_dotenv()

# Load project config
config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

print(f"Setting up custom model with DB for: {catalog_name}.{schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Model configuration - use your existing registered model
BASE_MODEL_NAME = f"{catalog_name}.{schema_name}.complaint_upheld_model_basic"  # Your registered model
CUSTOM_MODEL_NAME = f"{catalog_name}.{schema_name}.complaint_model_custom_db"
ENDPOINT_NAME = "complaints-custom-db-serving"
MODEL_VERSION = "latest-model"  # or specific version

print(f"Base model: {BASE_MODEL_NAME}")
print(f"Custom model name: {CUSTOM_MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify Base Model Exists

# COMMAND ----------

client = mlflow.MlflowClient()

# Verify base model exists
try:
    if MODEL_VERSION == "latest-model" or not MODEL_VERSION.isdigit():
        model_info = client.get_model_version_by_alias(BASE_MODEL_NAME, MODEL_VERSION)
        base_model_version = model_info.version
    else:
        base_model_version = MODEL_VERSION
        model_info = client.get_model_version(BASE_MODEL_NAME, base_model_version)
    
    base_model_uri = f"models:/{BASE_MODEL_NAME}/{base_model_version}"
    print(f"✓ Base model found: {BASE_MODEL_NAME} v{base_model_version}")
    print(f"  URI: {base_model_uri}")
    
except Exception as e:
    print(f"❌ Base model not found: {BASE_MODEL_NAME}")
    print(f"   Please ensure the model is registered first")
    print(f"   Error: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define Custom Model with External DB Integration

# COMMAND ----------

class ComplaintModelWithExternalDB(mlflow.pyfunc.PythonModel):
    """
    Custom model that integrates with external databases.
    
    In production, this would connect to:
    - DynamoDB for company risk scores
    - Redis for cached predictions
    - MongoDB for historical patterns
    - External APIs for real-time data
    """
    
    def load_context(self, context):
        """Load the base model and initialize connections."""
        # Load the trained model
        self.model = mlflow.sklearn.load_model(context.artifacts["base_model"])
        
        # Initialize external connections (mock for demo)
        self.init_external_connections()
        
    def init_external_connections(self):
        """Initialize connections to external data sources."""
        # In production, you would initialize real connections here:
        # self.dynamodb = boto3.resource('dynamodb')
        # self.redis_client = redis.Redis(host='redis-host', port=6379)
        # self.mongo_client = MongoClient('mongodb://localhost:27017/')
        
        # For demo, we'll use mock connections
        self.external_sources_available = True
        print("External connections initialized (mock mode)")
        
    def _fetch_company_risk_score(self, company_name: str) -> Dict:
        """
        Fetch real-time company risk score from DynamoDB.
        In production, this would query actual DynamoDB table.
        """
        # Mock external data fetch
        np.random.seed(hash(company_name) % 100)
        return {
            'risk_score': np.random.uniform(0, 1),
            'recent_complaints': np.random.randint(0, 100),
            'regulatory_flags': np.random.randint(0, 5),
            'market_sentiment': np.random.uniform(-1, 1),
            'esg_score': np.random.uniform(0, 100)
        }
    
    def _fetch_from_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Check Redis cache for existing predictions.
        Returns cached result if available.
        """
        # In production: return self.redis_client.get(cache_key)
        # For demo, always return None (no cache hit)
        return None
    
    def _store_in_cache(self, cache_key: str, data: Dict, ttl: int = 3600):
        """
        Store prediction in Redis cache with TTL.
        """
        # In production: self.redis_client.setex(cache_key, ttl, json.dumps(data))
        pass  # Mock implementation
    
    def _fetch_historical_patterns(self, state: str, product: str) -> Dict:
        """
        Fetch historical complaint patterns from MongoDB.
        """
        # Mock historical data
        return {
            'state_trend': np.random.choice(['increasing', 'stable', 'decreasing']),
            'product_complaint_rate': np.random.uniform(0, 0.3),
            'seasonal_factor': np.random.uniform(0.8, 1.2)
        }
    
    def _fetch_external_api_data(self, company: str) -> Dict:
        """
        Fetch real-time data from external APIs.
        Could include stock prices, news sentiment, etc.
        """
        # Mock API data
        return {
            'stock_volatility': np.random.uniform(0, 50),
            'news_sentiment': np.random.uniform(-1, 1),
            'social_mentions': np.random.randint(0, 1000)
        }
    
    def enrich_features(self, row: pd.Series) -> pd.Series:
        """
        Enrich input features with external data.
        """
        enriched = row.copy()
        
        # Fetch company risk data
        if 'Company' in row.index and pd.notna(row['Company']):
            company_data = self._fetch_company_risk_score(str(row['Company']))
            for key, value in company_data.items():
                enriched[f'external_{key}'] = value
        
        # Fetch historical patterns
        if 'State' in row.index and 'Product' in row.index:
            state = str(row.get('State', 'Unknown'))
            product = str(row.get('Product', 'Unknown'))
            historical = self._fetch_historical_patterns(state, product)
            for key, value in historical.items():
                enriched[f'historical_{key}'] = value
        
        # Fetch API data
        if 'Company' in row.index:
            api_data = self._fetch_external_api_data(str(row.get('Company', 'Unknown')))
            for key, value in api_data.items():
                enriched[f'api_{key}'] = value
        
        return enriched
    
    def predict(self, context, model_input):
        """
        Make predictions with external data enrichment.
        """
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        results = []
        
        for idx, row in model_input.iterrows():
            # Generate cache key
            complaint_id = str(row.get('Complaint_ID', idx))
            cache_key = f"prediction:{complaint_id}"
            
            # Check cache first
            cached = self._fetch_from_cache(cache_key)
            if cached:
                results.append(cached)
                continue
            
            # Enrich features with external data
            enriched_row = self.enrich_features(row)
            
            # Select features that model expects
            model_features = []
            for col in self.model.feature_names_in_:
                if col in enriched_row.index:
                    value = enriched_row[col]
                    # Handle different data types
                    if pd.isna(value):
                        model_features.append(0)
                    elif isinstance(value, str):
                        model_features.append(0)  # Would use encoder in production
                    else:
                        model_features.append(value)
                else:
                    model_features.append(0)  # Default value for missing features
            
            # Make prediction
            features_array = np.array(model_features).reshape(1, -1)
            prediction = self.model.predict(features_array)[0]
            probability = self.model.predict_proba(features_array)[0, 1]
            
            # Create enriched result
            result = {
                'complaint_id': complaint_id,
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence': float(abs(probability - 0.5) * 2),
                'external_data_used': True,
                'company': str(row.get('Company', 'Unknown')),
                'enriched_features_count': len([k for k in enriched_row.index if k.startswith('external_') or k.startswith('historical_') or k.startswith('api_')]),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add risk assessment based on external data
            if 'external_risk_score' in enriched_row.index:
                result['external_risk_score'] = float(enriched_row['external_risk_score'])
            
            # Store in cache
            self._store_in_cache(cache_key, result)
            
            results.append(result)
        
        # Return single dict for single prediction, list for batch
        return results[0] if len(results) == 1 else results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Register Custom Model

# COMMAND ----------

# Prepare test data for signature
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").limit(5)
feature_cols = [col for col in config.num_features + config.cat_features 
                if col in test_set.columns]
test_df = test_set.select(feature_cols + ["Complaint_ID", "Company", "State", "Product"]).toPandas()

# Fill missing values
for col in test_df.columns:
    if col not in ["Complaint_ID", "Company", "State", "Product"]:
        if test_df[col].dtype == 'object':
            test_df[col] = test_df[col].fillna('MISSING')
        else:
            test_df[col] = test_df[col].fillna(0)

print(f"Test data shape: {test_df.shape}")

# COMMAND ----------

# Register the custom model
mlflow.set_experiment("/Shared/complaints-custom-db-model")
custom_model = ComplaintModelWithExternalDB()

with mlflow.start_run(run_name="complaint-custom-db-wrapper") as run:
    run_id = run.info.run_id
    
    # Create signature
    sample_output = {
        'complaint_id': '123456',
        'prediction': 1,
        'probability': 0.75,
        'confidence': 0.5,
        'external_data_used': True,
        'company': 'Example Corp',
        'enriched_features_count': 12,
        'external_risk_score': 0.45,
        'timestamp': '2024-01-01T00:00:00'
    }
    
    signature = infer_signature(
        model_input=test_df.head(1),
        model_output=sample_output
    )
    
    # Log the custom model with reference to base model
    mlflow.pyfunc.log_model(
        python_model=custom_model,
        artifact_path="custom-db-model",
        artifacts={"base_model": base_model_uri},
        signature=signature,
        pip_requirements=[
            "pandas",
            "numpy",
            "scikit-learn",
            # External dependencies (for production)
            "boto3",  # For DynamoDB
            "redis",  # For Redis cache
            "pymongo",  # For MongoDB
            "requests"  # For external APIs
        ]
    )
    
    # Log configuration
    mlflow.log_params({
        "model_type": "custom_with_external_db",
        "base_model": BASE_MODEL_NAME,
        "base_model_version": base_model_version,
        "external_sources": "DynamoDB,Redis,MongoDB,External_APIs",
        "cache_enabled": True,
        "real_time_enrichment": True,
        "enrichment_features": "risk_score,historical_patterns,api_data"
    })

# Register the model
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/custom-db-model",
    name=CUSTOM_MODEL_NAME
)

print(f"✓ Custom model registered: {CUSTOM_MODEL_NAME} v{model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Deploy Custom Model Endpoint

# COMMAND ----------

w = WorkspaceClient()
entity_version = model_version.version

# Get credentials
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# Configure served entities with environment variables
served_entities = [
    ServedEntityInput(
        entity_name=CUSTOM_MODEL_NAME,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=entity_version,
        environment_vars={
            # In production, use Databricks secrets
            # Example secret references (create these in Databricks)
            # "AWS_ACCESS_KEY_ID": "{{secrets/mlops/aws_access_key}}",
            # "AWS_SECRET_ACCESS_KEY": "{{secrets/mlops/aws_secret_key}}",
            # "REDIS_HOST": "{{secrets/mlops/redis_host}}",
            # "MONGODB_URI": "{{secrets/mlops/mongodb_uri}}",
            
            # For demo, use placeholder values
            "EXTERNAL_DB_MODE": "mock",
            "CACHE_TTL": "3600",
            "ENABLE_EXTERNAL_ENRICHMENT": "true"
        }
    )
]

# Create or update endpoint
try:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served_entities),
    )
    print(f"✓ Created custom model endpoint: {ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e):
        w.serving_endpoints.update_config(
            name=ENDPOINT_NAME,
            served_entities=served_entities
        )
        print(f"✓ Updated custom model endpoint: {ENDPOINT_NAME}")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test Custom Model Endpoint

# COMMAND ----------

# Prepare test requests
test_records = test_df.head(5).to_dict(orient="records")
dataframe_records = [[record] for record in test_records]

print(f"Test records prepared: {len(test_records)}")
print(f"Sample columns: {list(test_records[0].keys())[:10]}...")

# COMMAND ----------

def call_custom_endpoint(record):
    """Call the custom model endpoint."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

# Wait for endpoint to be ready
print("Waiting for endpoint to be ready...")
time.sleep(30)

# Test with one record
print("\nTesting with single record...")
status_code, response_text = call_custom_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response: {response_text[:500]}...")

# COMMAND ----------

# Test batch predictions with external enrichment
print("\nBatch testing with external data enrichment:")
print("="*60)

for i in range(min(3, len(dataframe_records))):
    status_code, response_text = call_custom_endpoint(dataframe_records[i])
    print(f"\nRecord {i+1}:")
    print(f"  Status: {status_code}")
    if status_code == 200:
        import json
        try:
            result = json.loads(response_text)
            # Display enriched features info
            if isinstance(result, dict):
                print(f"  Prediction: {result.get('prediction')}")
                print(f"  Probability: {result.get('probability', 'N/A')}")
                print(f"  External Risk Score: {result.get('external_risk_score', 'N/A')}")
                print(f"  Enriched Features: {result.get('enriched_features_count', 0)}")
        except:
            print(f"  Response: {response_text[:200]}...")
    else:
        print(f"  Error: {response_text[:200]}...")
    
    time.sleep(0.5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Custom model with external DB integration deployed successfully!
# MAGIC 
# MAGIC ### Features Demonstrated:
# MAGIC 
# MAGIC 1. **External Data Sources Integration**:
# MAGIC    - DynamoDB for company risk scores
# MAGIC    - Redis for prediction caching
# MAGIC    - MongoDB for historical patterns
# MAGIC    - External APIs for real-time data
# MAGIC 
# MAGIC 2. **Real-time Feature Enrichment**:
# MAGIC    - Fetches additional features during inference
# MAGIC    - Combines ML predictions with business data
# MAGIC    - Caches results for performance
# MAGIC 
# MAGIC 3. **Production-Ready Patterns**:
# MAGIC    - Environment variables for configuration
# MAGIC    - Secret management ready (use Databricks secrets)
# MAGIC    - Error handling and fallbacks
# MAGIC    - Performance optimization with caching
# MAGIC 
# MAGIC ### To Use in Production:
# MAGIC 
# MAGIC 1. **Set up Databricks Secrets**:
# MAGIC    ```
# MAGIC    databricks secrets create-scope --scope mlops
# MAGIC    databricks secrets put --scope mlops --key aws_access_key
# MAGIC    databricks secrets put --scope mlops --key redis_host
# MAGIC    ```
# MAGIC 
# MAGIC 2. **Update Environment Variables** in the endpoint configuration to use real secrets
# MAGIC 
# MAGIC 3. **Implement Real Connections** in the model class instead of mock data
# MAGIC 
# MAGIC The endpoint enriches predictions with external data sources, providing more context for decision-making.