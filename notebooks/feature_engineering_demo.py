# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Demo for Financial Complaints
# MAGIC This notebook demonstrates feature tables, feature functions, and external database integration

# COMMAND ----------

import os
import sys
import json
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from mlflow.models import infer_signature
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from pyspark.errors import AnalysisException
from dotenv import load_dotenv

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.utils import is_databricks
from financial_complaints.config import ProjectConfig, Tags

# COMMAND ----------

# Setup environment
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
else:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

# Load configuration based on environment
if is_databricks():
    config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
else:
    config_path = "../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

# Load datasets
train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set")
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

print(f"Train set: {train_set.count()} rows")
print(f"Test set: {test_set.count()} rows")

# Check data types - IMPORTANT for UDF parameter types
print("\nData types of key columns:")
train_set.select("Date_received", "processing_days", "has_narrative", "consumer_disputed").printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Tables
# MAGIC We'll create feature tables for company and product features that can be looked up during model training and inference

# COMMAND ----------

# Feature table for company risk profiles
company_risk_table = f"{config.catalog_name}.{config.schema_name}.company_risk_features_demo"
lookup_features = ["company_complaint_count", "company_avg_processing_days", "company_risk_score"]

# COMMAND ----------

# Option 1: Using Feature Engineering Client
company_stats_df = spark.sql(f"""
SELECT 
    Company,
    COUNT(*) as company_complaint_count,
    AVG(processing_days) as company_avg_processing_days,
    AVG(CASE WHEN complaint_upheld = 1 THEN 1.0 ELSE 0.0 END) as company_upheld_rate,
    -- Calculate risk score
    (AVG(CASE WHEN complaint_upheld = 1 THEN 1.0 ELSE 0.0 END) * 0.5 + 
     CASE 
        WHEN AVG(processing_days) > 60 THEN 0.5
        WHEN AVG(processing_days) > 30 THEN 0.3
        ELSE 0.1
     END) as company_risk_score
FROM {config.catalog_name}.{config.schema_name}.train_set
WHERE Company IS NOT NULL
GROUP BY Company
HAVING COUNT(*) >= 10
""")

# Create the feature table
feature_table = fe.create_table(
    name=company_risk_table,
    primary_keys=["Company"],
    df=company_stats_df,
    description="Company risk profile features"
)

# Enable CDC for the table
spark.sql(f"ALTER TABLE {company_risk_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# Option 2: Using SQL for product features table
product_features_table = f"{config.catalog_name}.{config.schema_name}.product_features_demo"

spark.sql(f"""
CREATE OR REPLACE TABLE {product_features_table}
(Product STRING NOT NULL, 
 product_complaint_count INT,
 product_upheld_rate DOUBLE,
 product_avg_processing_days DOUBLE,
 product_complexity_score DOUBLE);
""")

# Add primary key constraint
try:
    spark.sql(f"ALTER TABLE {product_features_table} ADD CONSTRAINT product_pk_demo PRIMARY KEY(Product);")
except AnalysisException:
    pass

spark.sql(f"ALTER TABLE {product_features_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data
spark.sql(f"""
INSERT INTO {product_features_table}
SELECT 
    Product,
    COUNT(*) as product_complaint_count,
    AVG(CASE WHEN complaint_upheld = 1 THEN 1.0 ELSE 0.0 END) as product_upheld_rate,
    AVG(processing_days) as product_avg_processing_days,
    -- Complexity score based on various factors
    (COUNT(DISTINCT Issue) * 0.3 + 
     COUNT(DISTINCT Sub_issue) * 0.3 +
     AVG(CASE WHEN has_narrative = 1 THEN 0.4 ELSE 0.0 END)) as product_complexity_score
FROM {config.catalog_name}.{config.schema_name}.train_set
WHERE Product IS NOT NULL
GROUP BY Product
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Functions (Python UDFs with Correct Types)
# MAGIC All feature functions must use correct SQL types matching your data
# MAGIC - processing_days is BIGINT
# MAGIC - has_narrative is BIGINT
# MAGIC - Date_received is TIMESTAMP

# COMMAND ----------

# Function 1: Calculate complaint age in days (timezone-safe, TIMESTAMP input)
complaint_age_func = f"{config.catalog_name}.{config.schema_name}.calculate_complaint_age_demo"

spark.sql(f"""
CREATE OR REPLACE FUNCTION {complaint_age_func}(date_received TIMESTAMP)
RETURNS INT
LANGUAGE PYTHON AS
$$
from datetime import datetime

if date_received:
    # Remove timezone info to make both naive
    if hasattr(date_received, 'replace'):
        date_received_naive = date_received.replace(tzinfo=None)
    else:
        date_received_naive = date_received
    
    # Get current time without timezone
    now_naive = datetime.now()
    
    # Calculate difference in days
    return (now_naive - date_received_naive).days
else:
    return 0
$$
""")

# COMMAND ----------

# Function 2: Categorize complaint severity (BIGINT types to match your data)
severity_func = f"{config.catalog_name}.{config.schema_name}.categorize_severity_demo"

spark.sql(f"""
CREATE OR REPLACE FUNCTION {severity_func}(
    processing_days BIGINT,
    has_narrative BIGINT,
    consumer_disputed STRING
)
RETURNS STRING
LANGUAGE PYTHON AS
$$
# Convert to int for easier comparison
proc_days = int(processing_days) if processing_days is not None else 0
has_narr = int(has_narrative) if has_narrative is not None else 0

if proc_days > 60 and consumer_disputed == 'Yes':
    return 'Critical'
elif proc_days > 30 or has_narr == 1:
    return 'High'
elif proc_days > 15:
    return 'Medium'
else:
    return 'Low'
$$
""")

# COMMAND ----------

# Function 3: Response category (BIGINT type)
response_category_func = f"{config.catalog_name}.{config.schema_name}.response_category_demo"

spark.sql(f"""
CREATE OR REPLACE FUNCTION {response_category_func}(processing_days BIGINT)
RETURNS STRING
LANGUAGE PYTHON AS
$$
# Convert to int for easier comparison
proc_days = int(processing_days) if processing_days is not None else 0

if proc_days == 0:
    return 'Unknown'
elif proc_days <= 7:
    return 'Immediate'
elif proc_days <= 15:
    return 'Quick'
elif proc_days <= 30:
    return 'Standard'
elif proc_days <= 60:
    return 'Delayed'
else:
    return 'Severely_Delayed'
$$
""")

# COMMAND ----------

# Test all functions
print("Testing feature functions with correct types:")
spark.sql(f"""
SELECT 
    {complaint_age_func}(current_timestamp() - interval 30 days) as complaint_age,
    {severity_func}(CAST(45 AS BIGINT), CAST(1 AS BIGINT), 'Yes') as severity,
    {response_category_func}(CAST(25 AS BIGINT)) as response_category
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Training Set with Feature Lookups

# COMMAND ----------

# Check which columns exist in train_set
print("Available columns in train_set:")
print(train_set.columns)

# Only drop columns that actually exist and will be re-added via lookups
cols_to_drop = []
for col in ["company_complaint_count", "company_avg_processing_days", "company_risk_score"]:
    if col in train_set.columns:
        cols_to_drop.append(col)

print(f"Columns to drop: {cols_to_drop}")

if cols_to_drop:
    training_df = train_set.drop(*cols_to_drop)
else:
    training_df = train_set

# Create training set with feature lookups
training_set = fe.create_training_set(
    df=training_df.limit(5000),  # Using subset for demo
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=company_risk_table,
            feature_names=["company_complaint_count", "company_avg_processing_days", "company_risk_score"],
            lookup_key="Company"
        ),
        FeatureLookup(
            table_name=product_features_table,
            feature_names=["product_complaint_count", "product_upheld_rate", "product_complexity_score"],
            lookup_key="Product"
        ),
        FeatureFunction(
            udf_name=complaint_age_func,
            output_name="complaint_age_days",
            input_bindings={"date_received": "Date_received"}
        ),
        FeatureFunction(
            udf_name=severity_func,
            output_name="complaint_severity",
            input_bindings={
                "processing_days": "processing_days",
                "has_narrative": "has_narrative",
                "consumer_disputed": "consumer_disputed"
            }
        ),
        FeatureFunction(
            udf_name=response_category_func,
            output_name="response_category",
            input_bindings={"processing_days": "processing_days"}
        )
    ],
    exclude_columns=["update_timestamp", "Complaint_ID", "dataset_type", "stratification_key", "has_target"]
)

print("Training set created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and Register Model with Feature Engineering

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load training data
training_df = training_set.load_df().toPandas()
print(f"Training DataFrame shape: {training_df.shape}")
print(f"Available columns: {training_df.columns.tolist()[:20]}...")  # Show first 20 columns

# Prepare features - include only columns that exist
all_possible_features = config.num_features + config.cat_features
feature_cols = [col for col in all_possible_features if col in training_df.columns]

# Add new lookup features if they exist
lookup_cols = ['company_complaint_count', 'company_avg_processing_days', 'company_risk_score',
               'product_complaint_count', 'product_upheld_rate', 'product_complexity_score',
               'complaint_age_days', 'complaint_severity', 'response_category']

for col in lookup_cols:
    if col in training_df.columns and col not in feature_cols:
        feature_cols.append(col)

print(f"Using {len(feature_cols)} features for training")
print(f"Feature columns sample: {feature_cols[:10]}...")

# Store feature columns for later use
model_feature_cols = feature_cols.copy()

# Fill missing values before selecting features
X_train = training_df[feature_cols].fillna(0)
y_train = training_df[config.target]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Class distribution: {y_train.value_counts().to_dict()}")

# COMMAND ----------

# Build preprocessing pipeline
# Separate numeric and categorical features
numeric_features = []
categorical_features = []

for col in feature_cols:
    if col in config.num_features or col in ['company_complaint_count', 'company_avg_processing_days', 
                                              'company_risk_score', 'product_complaint_count', 
                                              'product_upheld_rate', 'product_complexity_score', 
                                              'complaint_age_days']:
        numeric_features.append(col)
    elif col in config.cat_features or col in ['complaint_severity', 'response_category']:
        categorical_features.append(col)

# Filter high cardinality features
low_card_cats = [col for col in categorical_features if col not in config.high_cardinality_features]

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features (low cardinality): {len(low_card_cats)}")

# Create preprocessor
transformers = []
if numeric_features:
    transformers.append(('num', StandardScaler(), numeric_features))
if low_card_cats:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cats))

if transformers:
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
else:
    preprocessor = 'passthrough'

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=config.parameters['random_state'],
        n_jobs=-1
    ))
])

# Train model
pipeline.fit(X_train, y_train)
print("Model training completed!")

# COMMAND ----------

# Log model with feature engineering
mlflow.set_experiment("/Shared/demo-complaints-fe")
with mlflow.start_run(run_name="demo-complaints-fe-run") as run:
    run_id = run.info.run_id
    
    mlflow.log_param("model_type", "RandomForest with Feature Engineering")
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})
    mlflow.log_param("num_features_used", len(numeric_features))
    mlflow.log_param("cat_features_used", len(low_card_cats))
    
    # Log some training metrics
    train_score = pipeline.score(X_train, y_train)
    mlflow.log_metric("train_accuracy", train_score)
    
    # Create signature
    signature = infer_signature(X_train, pipeline.predict(X_train))
    
    # Log model with feature engineering metadata
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="complaint-model-fe",
        training_set=training_set,
        signature=signature
    )
    
    print(f"Model logged with run_id: {run_id}")

# COMMAND ----------

# Register model
model_name = f"{config.catalog_name}.{config.schema_name}.complaint_model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/complaint-model-fe',
    name=model_name,
    tags={"feature_engineering": "enabled", "demo": "true"}
)

print(f"Model registered: {model_name} version {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Batch Scoring with Feature Lookups - Alternative Approach
# MAGIC 
# MAGIC Since the feature store is not automatically adding all required features,
# MAGIC we need to manually prepare the test data with all required features.

# COMMAND ----------

# Method 1: Manual feature preparation for batch scoring
print("Method 1: Manual feature preparation")

# First, let's get a test batch and apply the same feature engineering
test_batch_spark = test_set.limit(100)

# Apply feature functions manually
test_batch_with_features = test_batch_spark.selectExpr(
    "*",
    f"{complaint_age_func}(Date_received) as complaint_age_days",
    f"{severity_func}(processing_days, has_narrative, consumer_disputed) as complaint_severity", 
    f"{response_category_func}(processing_days) as response_category"
)

# Join with product features
test_batch_with_product = test_batch_with_features.join(
    spark.table(product_features_table),
    on="Product",
    how="left"
)

# Convert to pandas for prediction
test_batch_pandas = test_batch_with_product.toPandas()

# Ensure all model features are present
missing_cols = set(model_feature_cols) - set(test_batch_pandas.columns)
if missing_cols:
    print(f"Warning: Some features are still missing: {missing_cols}")
    # Add missing columns with default values
    for col in missing_cols:
        test_batch_pandas[col] = 0

# Select only the features the model was trained on
X_test_batch = test_batch_pandas[model_feature_cols].fillna(0)

# Make predictions using the pipeline directly
predictions_manual = pipeline.predict(X_test_batch)
print(f"Predictions shape: {predictions_manual.shape}")
print(f"Sample predictions: {predictions_manual[:10]}")

# COMMAND ----------

# Method 2: Try score_batch with proper DataFrame preparation
print("\nMethod 2: Using score_batch with properly prepared DataFrame")

try:
    # Prepare test batch with all required columns
    # First, ensure test_set has all the columns from train_set
    test_batch_prepared = test_set.limit(100)
    
    # Score batch - this should work if the model was logged with feature store
    predictions = fe.score_batch(
        model_uri=f"models:/{model_name}/{model_version.version}",
        df=test_batch_prepared
    )
    
    # Show predictions
    print("\nPredictions using score_batch:")
    predictions.select("prediction").show(10)
    print("Batch scoring completed successfully!")
    
except Exception as e:
    print(f"Score batch failed with error: {str(e)}")
    print("\nFalling back to manual prediction method...")
    
    # Create a DataFrame with predictions
    test_batch_pandas['prediction'] = predictions_manual
    predictions_df = spark.createDataFrame(test_batch_pandas[['Complaint_ID', 'prediction']])
    predictions_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Simple Model Without Feature Store
# MAGIC 
# MAGIC For production use, consider training a model without the feature store
# MAGIC if the feature store integration is causing issues.

# COMMAND ----------

# Train a simple model without feature store for comparison
print("Training simple model without feature store...")

# Use the original train and test sets
train_pandas = train_set.limit(5000).toPandas()
test_pandas = test_set.limit(100).toPandas()

# Select features that exist in both train and test
common_features = list(set(config.num_features + config.cat_features) & 
                       set(train_pandas.columns) & 
                       set(test_pandas.columns))

print(f"Common features found: {len(common_features)}")

X_train_simple = train_pandas[common_features].fillna(0)
y_train_simple = train_pandas[config.target]

X_test_simple = test_pandas[common_features].fillna(0)
y_test_simple = test_pandas[config.target] if config.target in test_pandas.columns else None

# Build simple pipeline
numeric_features_simple = [f for f in common_features if f in config.num_features]
cat_features_simple = [f for f in common_features if f in config.cat_features and f not in config.high_cardinality_features]

preprocessor_simple = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features_simple),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features_simple)
    ],
    remainder='drop'
)

pipeline_simple = Pipeline([
    ('preprocessor', preprocessor_simple),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# Train and predict
pipeline_simple.fit(X_train_simple, y_train_simple)
predictions_simple = pipeline_simple.predict(X_test_simple)

print(f"Simple model predictions: {predictions_simple[:10]}")

# Log simple model
with mlflow.start_run(run_name="simple-model-no-fe"):
    mlflow.sklearn.log_model(
        sk_model=pipeline_simple,
        artifact_path="model",
        signature=infer_signature(X_train_simple, predictions_simple)
    )
    mlflow.log_param("model_type", "Simple RandomForest without Feature Store")
    if y_test_simple is not None:
        accuracy = (predictions_simple == y_test_simple.values).mean()
        mlflow.log_metric("test_accuracy", accuracy)
        print(f"Simple model accuracy: {accuracy:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Recommendations
# MAGIC 
# MAGIC ### Issue Encountered:
# MAGIC The feature store's `score_batch` method is not properly adding all required features during inference.
# MAGIC 
# MAGIC ### Solutions Provided:
# MAGIC 
# MAGIC 1. **Manual Feature Engineering** (Method 1):
# MAGIC    - Apply feature functions manually using SQL
# MAGIC    - Join with feature tables
# MAGIC    - Convert to pandas and predict
# MAGIC 
# MAGIC 2. **Simple Model** (Recommended for Production):
# MAGIC    - Train model without feature store
# MAGIC    - Use only features available in both train and test sets
# MAGIC    - More reliable and easier to deploy
# MAGIC 
# MAGIC ### Key Learnings:
# MAGIC 
# MAGIC 1. **Type Matching**: UDF parameters must match DataFrame column types exactly
# MAGIC    - `processing_days`: BIGINT
# MAGIC    - `has_narrative`: BIGINT
# MAGIC    - `Date_received`: TIMESTAMP
# MAGIC 
# MAGIC 2. **Feature Store Limitations**:
# MAGIC    - May not automatically add all features during scoring
# MAGIC    - Requires careful management of feature columns
# MAGIC    - Consider simpler alternatives for production
# MAGIC 
# MAGIC 3. **Best Practices**:
# MAGIC    - Always verify feature availability in test data
# MAGIC    - Have fallback methods for prediction
# MAGIC    - Consider trade-offs between complexity and reliability

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom PyFunc Model with External Data Integration

# COMMAND ----------

import mlflow.pyfunc

class ComplaintModelWithExternalData(mlflow.pyfunc.PythonModel):
    """Model that can fetch features from external database during prediction."""
    
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns  # Store the exact features model was trained on
        
    def _fetch_external_features(self, companies):
        """Simulate fetching additional company data from external source.
        In production, this would connect to DynamoDB, Redis, MongoDB, etc.
        """
        external_features = {}
        for company in companies:
            # Simulate external data (in real scenario, fetch from database)
            external_features[company] = {
                'external_risk_rating': np.random.uniform(0.2, 0.8),
                'external_complaint_volume': np.random.randint(10, 1000),
                'external_resolution_rate': np.random.uniform(0.5, 0.95)
            }
        return external_features
    
    def predict(self, context, model_input):
        """Make predictions with external data enrichment."""
        if isinstance(model_input, pd.DataFrame):
            # Make a copy to avoid modifying original
            enriched_input = model_input.copy()
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(enriched_input.columns)
            for feat in missing_features:
                enriched_input[feat] = 0  # Add missing features with default values
            
            # Fetch external features if Company column exists
            if 'Company' in enriched_input.columns:
                companies = enriched_input['Company'].dropna().unique()
                external_data = self._fetch_external_features(companies)
                
                # Add external features to input
                enriched_input['external_risk_rating'] = enriched_input['Company'].map(
                    lambda x: external_data.get(x, {}).get('external_risk_rating', 0.5)
                )
            
            # Fill any missing values
            enriched_input = enriched_input.fillna(0)
            
            # Select only the features the model was trained on
            model_input_clean = enriched_input[self.feature_columns]
            
            # Make predictions
            predictions = self.model.predict(model_input_clean)
            
            # Return enriched predictions with confidence scores
            result = pd.DataFrame({
                'prediction': predictions,
                'confidence': np.random.uniform(0.6, 0.95, len(predictions)),  # Mock confidence
                'used_external_data': 'Company' in model_input.columns
            })
            
            return result
        else:
            raise ValueError("Input must be a pandas DataFrame")

# COMMAND ----------

# Create and test custom model
custom_model = ComplaintModelWithExternalData(pipeline_simple, common_features)

# Prepare test data
test_sample = test_set.limit(5).toPandas()
test_features = test_sample[common_features].fillna(0)

# Get predictions with external data
predictions_with_external = custom_model.predict(None, test_features)
print("Predictions with external data enrichment:")
print(predictions_with_external)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Custom Model with External Data Support

# COMMAND ----------

mlflow.set_experiment("/Shared/demo-complaints-external")
with mlflow.start_run(run_name="complaints-external-data-working") as run:
    custom_run_id = run.info.run_id
    
    # Create signature for custom model
    custom_signature = infer_signature(
        test_features,
        custom_model.predict(None, test_features)
    )
    
    # Log the custom model
    mlflow.pyfunc.log_model(
        python_model=custom_model,
        artifact_path="complaint-model-external",
        signature=custom_signature
    )
    
    mlflow.log_params({
        "model_type": "RandomForest with External Data",
        "external_source": "Mock_External_DB",
        "base_model": "RandomForestClassifier",
        "features_used": len(common_features)
    })
    
    print(f"Custom model with external data support logged: {custom_run_id}")