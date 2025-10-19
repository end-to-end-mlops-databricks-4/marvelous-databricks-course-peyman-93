# Databricks notebook source
# MAGIC %md
# MAGIC # Financial Complaints Model Monitoring
# MAGIC 
# MAGIC This notebook:
# MAGIC 1. Analyzes feature importance
# MAGIC 2. Generates synthetic data with drift
# MAGIC 3. Sends requests to the model endpoint
# MAGIC 4. Creates and refreshes monitoring tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import sys
sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/dev/marvelous-databricks-course-peyman-93/files/src')

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime
import itertools
import time
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, to_utc_timestamp
from databricks.sdk import WorkspaceClient
from databricks.connect import DatabricksSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from financial_complaints.config import ProjectConfig
from financial_complaints.data_processor import generate_synthetic_data
from financial_complaints.monitoring import create_or_refresh_monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration and Data

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

print(f"✓ Train set loaded: {train_set.shape}")
print(f"✓ Test set loaded: {test_set.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Analysis

# COMMAND ----------

def preprocess_data(df):
    """Encode categorical and datetime variables for feature importance analysis."""
    label_encoders = {}
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['object', 'datetime']).columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        label_encoders[col] = le
    return df_copy, label_encoders

# Encode data
train_set_encoded, label_encoders = preprocess_data(train_set)

# Define features and target
features = train_set_encoded.drop(columns=["complaint_upheld", "financial_relief", "has_target", "stratification_key"])
target = train_set_encoded["complaint_upheld"]

# Train Random Forest
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(features, target)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importances.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Data with Drift

# COMMAND ----------

# Generate synthetic data
inference_data_skewed = generate_synthetic_data(train_set, num_rows=200)
print(f"✓ Generated {len(inference_data_skewed)} synthetic records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Inference Data

# COMMAND ----------

def clean_column_names(df):
    """Clean DataFrame column names for Delta table compatibility."""
    df_cleaned = df.copy()
    df_cleaned.columns = [
        col.replace(' ', '_').replace('?', '').replace('-', '_').replace('/', '_')
           .replace('(', '').replace(')', '').replace('.', '_').replace(',', '')
           .replace(';', '').replace('{', '').replace('}', '').replace('\n', '')
           .replace('\t', '').replace('=', '')
        for col in df_cleaned.columns
    ]
    return df_cleaned

# Clean column names
inference_data_skewed_clean = clean_column_names(inference_data_skewed)

# Calculate processing_days if not present
if 'processing_days' not in inference_data_skewed_clean.columns:
    inference_data_skewed_clean['processing_days'] = (
        pd.to_datetime(inference_data_skewed_clean['Date_sent_to_company']) -
        pd.to_datetime(inference_data_skewed_clean['Date_received'])
    ).dt.days.astype('Int64')

# Generate temporal features
date_received = pd.to_datetime(inference_data_skewed_clean['Date_received'])
inference_data_skewed_clean['complaint_year'] = date_received.dt.year.astype('Int64')
inference_data_skewed_clean['complaint_month'] = date_received.dt.month.astype('Int64')
inference_data_skewed_clean['complaint_day'] = date_received.dt.day.astype('Int64')
inference_data_skewed_clean['complaint_day_of_week'] = date_received.dt.dayofweek.astype('Int64')
inference_data_skewed_clean['complaint_quarter'] = date_received.dt.quarter.astype('Int64')
inference_data_skewed_clean['complaint_week_of_year'] = date_received.dt.isocalendar().week.astype('Int64')
inference_data_skewed_clean['is_weekend'] = (date_received.dt.dayofweek >= 5).astype('Int64')
inference_data_skewed_clean['is_monthend'] = date_received.dt.is_month_end.astype('Int64')
inference_data_skewed_clean['is_monthstart'] = date_received.dt.is_month_start.astype('Int64')

# Calculate days_since_dataset_start
min_date = date_received.min()
inference_data_skewed_clean['days_since_dataset_start'] = (date_received - min_date).dt.days.astype('Int64')

# Generate region from State
state_to_region = {
    'ME': 'Northeast', 'NH': 'Northeast', 'VT': 'Northeast', 'MA': 'Northeast', 'RI': 'Northeast',
    'CT': 'Northeast', 'NY': 'Northeast', 'NJ': 'Northeast', 'PA': 'Northeast',
    'OH': 'Midwest', 'IN': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest', 'WI': 'Midwest',
    'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    'NE': 'Midwest', 'KS': 'Midwest',
    'DE': 'South', 'MD': 'South', 'VA': 'South', 'WV': 'South', 'NC': 'South', 'SC': 'South',
    'GA': 'South', 'FL': 'South', 'KY': 'South', 'TN': 'South', 'AL': 'South', 'MS': 'South',
    'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South',
    'MT': 'West', 'ID': 'West', 'WY': 'West', 'CO': 'West', 'NM': 'West', 'AZ': 'West',
    'UT': 'West', 'NV': 'West', 'WA': 'West', 'OR': 'West', 'CA': 'West', 'AK': 'West', 'HI': 'West'
}
inference_data_skewed_clean['region'] = inference_data_skewed_clean['State'].map(state_to_region).fillna('Unknown')

# Generate season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

inference_data_skewed_clean['season'] = inference_data_skewed_clean['complaint_month'].apply(get_season)
inference_data_skewed_clean['near_holiday'] = 'No'

# Generate processing_speed
def get_processing_speed(days):
    if pd.isna(days):
        return 'Unknown'
    elif days <= 3:
        return 'Fast'
    elif days <= 7:
        return 'Medium'
    else:
        return 'Slow'

inference_data_skewed_clean['processing_speed'] = inference_data_skewed_clean['processing_days'].apply(get_processing_speed)

# Drop unnecessary columns
columns_to_drop = ['Company_public_response', 'Consumer_complaint_narrative']
inference_data_skewed_clean = inference_data_skewed_clean.drop(
    columns=[col for col in columns_to_drop if col in inference_data_skewed_clean.columns],
    errors='ignore'
)

print(f"✓ Prepared {len(inference_data_skewed_clean)} records with {len(inference_data_skewed_clean.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Inference Data to Delta Table

# COMMAND ----------

# Convert to Spark DataFrame
inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed_clean).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

# Save to table
inference_data_skewed_spark.write \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

print(f"✓ Saved inference data table: {inference_data_skewed_spark.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update Company Features

# COMMAND ----------

spark.sql(f"""
    INSERT INTO {config.catalog_name}.{config.schema_name}.company_features
    SELECT DISTINCT
        Company,
        COUNT(*) as company_complaint_count,
        AVG(processing_days) as company_avg_processing_days,
        PERCENTILE(processing_days, 0.5) as company_median_processing_days,
        STDDEV(processing_days) as company_processing_std,
        0.0 as company_upheld_rate,
        0 as company_product_diversity,
        0 as company_state_coverage,
        0.5 as company_timely_response_rate,
        0.5 as company_reliability_score,
        current_timestamp() as update_timestamp
    FROM {config.catalog_name}.{config.schema_name}.inference_data_skewed
    WHERE Company IS NOT NULL
    GROUP BY Company
""")

print("✓ Company features updated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for Endpoint Requests

# COMMAND ----------

# Reload datasets
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set") \
                .withColumn("Complaint_ID", col("Complaint_ID").cast("string")) \
                .toPandas()

inference_data_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed") \
                        .withColumn("Complaint_ID", col("Complaint_ID").cast("string")) \
                        .toPandas()

# Convert datetime columns to strings
for df in [test_set, inference_data_skewed]:
    for col_name in df.select_dtypes(include=['datetime64']).columns:
        df[col_name] = df[col_name].astype(str)

# CRITICAL FIX: Exclude target variables AND internal columns
columns_to_exclude = [
    'complaint_upheld',      # Target variable
    'financial_relief',      # Secondary target
    'has_target',            # Internal flag used during training
    'stratification_key'     # Internal column for data splitting
]

# Get all columns except excluded ones
available_columns_test = [c for c in test_set.columns if c not in columns_to_exclude]
available_columns_skewed = [c for c in inference_data_skewed.columns if c not in columns_to_exclude]

print(f"✓ Using {len(available_columns_test)} columns from test_set")
print(f"✓ Using {len(available_columns_skewed)} columns from inference_data_skewed")
print(f"✓ Excluded columns: {columns_to_exclude}")

# Prepare clean data
test_set_clean = test_set[available_columns_test].copy()
inference_data_skewed_clean = inference_data_skewed[available_columns_skewed].copy()

# Fill NaN values
for df in [test_set_clean, inference_data_skewed_clean]:
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("")

# Create request records
test_set_records = test_set_clean.to_dict(orient="records")
sampled_skewed_records = inference_data_skewed_clean.to_dict(orient="records")

print(f"\n✓ Created {len(test_set_records)} test records")
print(f"✓ Created {len(sampled_skewed_records)} skewed records")

# Show sample columns being sent
if test_set_records:
    print(f"\n✓ Sample columns in first request: {list(test_set_records[0].keys())[:15]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Endpoint Connection

# COMMAND ----------

workspace = WorkspaceClient()
token = workspace.tokens.create(lifetime_seconds=1200).token_value
host = workspace.config.host

def send_request_https(dataframe_record):
    """Send request to model endpoint."""
    model_serving_endpoint = f"{host}/serving-endpoints/complaints-model-serving-fe-dev/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response

print(f"✓ Endpoint ready: {host}/serving-endpoints/complaints-model-serving-fe-dev")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Single Request First

# COMMAND ----------

print("="*60)
print("TESTING SINGLE REQUEST")
print("="*60)

test_response = send_request_https(test_set_records[0])
print(f"Status Code: {test_response.status_code}")

if test_response.status_code == 200:
    print(f"✓ SUCCESS!")
    print(f"Response: {test_response.json()}")
    print("\n✓ Ready to proceed with full request loops!")
else:
    print(f"✗ ERROR: {test_response.text[:500]}")
    print("\n⚠️  Fix the error above before proceeding!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Test Data Requests (10 minutes)
# MAGIC 
# MAGIC **Only run this cell if the single test request above succeeded!**

# COMMAND ----------

print("="*60)
print("SENDING TEST DATA REQUESTS FOR 10 MINUTES")
print("="*60)

end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
success_count = 0
error_count = 0

for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    
    try:
        response = send_request_https(record)
        if response.status_code == 200:
            success_count += 1
            if index % 50 == 0:
                print(f"✓ Request {index}: SUCCESS (Total: {success_count})")
        else:
            error_count += 1
            if error_count <= 3:
                print(f"✗ Request {index}: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        error_count += 1
        if error_count <= 3:
            print(f"✗ Request {index}: {str(e)[:200]}")
    
    time.sleep(0.2)

print("\n" + "="*60)
print(f"TEST DATA COMPLETE: {success_count} successful, {error_count} errors")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Skewed Data Requests (20 minutes)
# MAGIC 
# MAGIC **Only run this cell if test data requests succeeded!**

# COMMAND ----------

print("="*60)
print("SENDING SKEWED DATA REQUESTS FOR 20 MINUTES")
print("="*60)

end_time = datetime.datetime.now() + datetime.timedelta(minutes=20)
success_count = 0
error_count = 0

for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    
    try:
        response = send_request_https(record)
        if response.status_code == 200:
            success_count += 1
            if index % 50 == 0:
                print(f"✓ Request {index}: SUCCESS (Total: {success_count})")
        else:
            error_count += 1
            if error_count <= 3:
                print(f"✗ Request {index}: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        error_count += 1
        if error_count <= 3:
            print(f"✗ Request {index}: {str(e)[:200]}")
    
    time.sleep(0.2)

print("\n" + "="*60)
print(f"SKEWED DATA COMPLETE: {success_count} successful, {error_count} errors")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refresh Monitoring

# COMMAND ----------

print("="*60)
print("REFRESHING MONITORING TABLE")
print("="*60)

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)

print("\n✓ Monitoring refresh complete!")
print("✓ Check your monitoring dashboard in Databricks!")