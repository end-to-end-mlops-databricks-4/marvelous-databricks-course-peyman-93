# Databricks notebook source
# MAGIC %md
# MAGIC # Financial Complaints Data Processing Pipeline
# MAGIC This notebook processes financial complaints data and saves to Unity Catalog

# COMMAND ----------

# Install packages if needed
# %pip install -e ..
# %restart_python

# COMMAND ----------

from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger, Logger 
from logging import Logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit

# Add src to path
sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')

from financial_complaints.config import ProjectConfig
from financial_complaints.data_processor import DataProcessor

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Configuration

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(
    config_path="/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml", 
    env="dev"
)

logger.info("Configuration loaded:")
logger.info(f"Catalog: {config.catalog_name}")
logger.info(f"Schema: {config.schema_name}")
logger.info(f"Target: {config.target}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Spark Session

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("FinancialComplaintsProcessing").getOrCreate()
logger.info("Spark session initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Data

# COMMAND ----------

# Load the complaints dataset
data_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/data/Consumer_Complaints_Sample.csv"

if os.path.exists(data_path):
    logger.info(f"Loading data from: {data_path}")
    
    # Read only first 30k rows to avoid parsing errors
    df = pd.read_csv(data_path, 
                     nrows=30000,  # Limit to avoid parsing errors
                     low_memory=False)
    
    logger.info(f"Data loaded successfully: {len(df):,} records, {len(df.columns)} columns")
else:
    logger.error(f"Data file not found at: {data_path}")
    raise FileNotFoundError("Please ensure Consumer_Complaints_Sample.csv is committed and pushed to Git")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Preprocess Data

# COMMAND ----------

# Initialize data processor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
logger.info("Starting data preprocessing...")
data_processor.preprocess()
logger.info("Data preprocessing completed")
logger.info(f"Features created: {len(data_processor.df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Train/Test Splits
# MAGIC Note: Data type fixing for Spark compatibility is now handled automatically

# COMMAND ----------

# Split the data with automatic type fixing for Spark
logger.info("Creating train/test splits...")

train_df, test_df, temporal_test_df = data_processor.split_data(fix_types_for_spark=True)

logger.info(f"Training set shape: {train_df.shape}")
logger.info(f"Test set shape: {test_df.shape}")
logger.info(f"Temporal test set shape: {temporal_test_df.shape}")

# Log target distributions
logger.info(f"Train - {config.target} rate: {train_df[config.target].mean():.3f}")
logger.info(f"Test - {config.target} rate: {test_df[config.target].mean():.3f}")
logger.info(f"Temporal - {config.target} rate: {temporal_test_df[config.target].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save to Unity Catalog

# COMMAND ----------

# Function to save with error handling
def save_to_catalog_safe(
    df: pd.DataFrame, 
    table_name: str, 
    dataset_type: str, 
    config: ProjectConfig, 
    spark: SparkSession, 
    logger: Logger
) -> bool:
    """Save DataFrame to Unity Catalog with fallback handling."""
    logger.info(f"Saving {dataset_type} set...")
    
    try:
        # Create Spark DataFrame
        spark_df = spark.createDataFrame(df)
        
        # Add metadata columns
        spark_df = spark_df.withColumn("update_timestamp", current_timestamp()) \
                          .withColumn("dataset_type", lit(dataset_type))
        
        # Save to table
        spark_df.write \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(f"{config.catalog_name}.{config.schema_name}.{table_name}")
        
        logger.info(f"✓ Saved {table_name}: {len(df):,} records")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save {table_name}: {str(e)}")
        
        # Fallback: Convert all to strings
        logger.info(f"Attempting fallback save for {table_name} (converting to strings)...")
        try:
            df_str = df.astype(str)
            spark_df = spark.createDataFrame(df_str)
            spark_df = spark_df.withColumn("update_timestamp", current_timestamp()) \
                              .withColumn("dataset_type", lit(dataset_type))
            
            spark_df.write \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .saveAsTable(f"{config.catalog_name}.{config.schema_name}.{table_name}")
            
            logger.info(f"✓ Saved {table_name} using string conversion: {len(df):,} records")
            return True
        except Exception as e2:
            logger.error(f"Fallback also failed for {table_name}: {str(e2)}")
            return False

# COMMAND ----------

# Ensure catalog and schema exist
logger.info("Setting up catalog and schema...")
spark.sql(f"USE CATALOG {config.catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.schema_name}")
logger.info(f"Using: {config.catalog_name}.{config.schema_name}")

# COMMAND ----------

# Save all datasets
logger.info("Saving datasets to Unity Catalog...")

# Save training set
save_to_catalog_safe(train_df, "train_set", "train", config, spark, logger)

# Save test set
save_to_catalog_safe(test_df, "test_set", "test", config, spark, logger)

# Save temporal test set
save_to_catalog_safe(temporal_test_df, "temporal_test_set", "temporal_test", config, spark, logger)

# Save in-progress complaints if they exist (already type-fixed in split_data)
if hasattr(data_processor, 'in_progress_df') and len(data_processor.in_progress_df) > 0:
    logger.info("Processing in-progress complaints...")
    save_to_catalog_safe(data_processor.in_progress_df, "in_progress_set", "in_progress", config, spark, logger)
else:
    logger.info("No in-progress complaints to save")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Enable Change Data Feed

# COMMAND ----------

# Enable change data feed for tables
logger.info("Enabling Change Data Feed...")

tables = ['train_set', 'test_set', 'temporal_test_set']
if hasattr(data_processor, 'in_progress_df') and len(data_processor.in_progress_df) > 0:
    tables.append('in_progress_set')

for table in tables:
    try:
        spark.sql(f"""
            ALTER TABLE {config.catalog_name}.{config.schema_name}.{table} 
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"✓ Enabled CDF for {table}")
    except Exception as e:
        logger.warning(f"Could not enable CDF for {table}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verify Saved Data

# COMMAND ----------

# Verify the saved data
logger.info("Verifying saved data...")
logger.info("-" * 60)

for table in tables:
    try:
        count = spark.sql(f"""
            SELECT COUNT(*) as count 
            FROM {config.catalog_name}.{config.schema_name}.{table}
        """).collect()[0]['count']
        
        logger.info(f"✓ {table}: {count:,} records")
    except Exception as e:
        logger.error(f"✗ {table}: Could not verify - {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Display Sample Data

# COMMAND ----------

# Display a sample of the training data
logger.info("Displaying sample of training data...")
display(spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

# Print summary
print("=" * 80)
print("FINANCIAL COMPLAINTS DATA PROCESSING PIPELINE COMPLETE!")
print("=" * 80)
print(f"\nData successfully saved to Unity Catalog:")
print(f"  • Catalog: {config.catalog_name}")
print(f"  • Schema: {config.schema_name}")
print(f"\nTables created:")

for table in tables:
    try:
        count = spark.table(f"{config.catalog_name}.{config.schema_name}.{table}").count()
        print(f"  • {table}: {count:,} records")
    except Exception:
        print(f"  • {table}: Created")

print(f"\nTotal features: {len(train_df.columns)}")
print(f"\nNext steps:")
print(f"  1. Train models using the prepared datasets")
print(f"  2. Track experiments with MLflow")
print(f"  3. Register best models for deployment")
print("=" * 80)

# COMMAND ----------

# Get and display preprocessing report
preprocessing_report = data_processor.get_preprocessing_report()
print("\n" + preprocessing_report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick SQL Reference
# MAGIC 
# MAGIC ```sql
# MAGIC -- View your data
# MAGIC SELECT * FROM mlops_dev.peyman21.train_set LIMIT 100;
# MAGIC 
# MAGIC -- Check target balance
# MAGIC SELECT complaint_upheld, COUNT(*) as count
# MAGIC FROM mlops_dev.peyman21.train_set 
# MAGIC GROUP BY complaint_upheld;
# MAGIC 
# MAGIC -- Describe table
# MAGIC DESCRIBE TABLE mlops_dev.peyman21.train_set;
# MAGIC 
# MAGIC -- Show table history
# MAGIC DESCRIBE HISTORY mlops_dev.peyman21.train_set;
# MAGIC ```