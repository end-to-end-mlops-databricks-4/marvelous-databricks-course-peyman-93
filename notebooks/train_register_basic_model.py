# Databricks notebook source
# MAGIC %md
# MAGIC # Train and Register Basic Complaint Model
# MAGIC This notebook trains the complaint upheld model using the BasicComplaintModel class

# COMMAND ----------

import os
import sys
import mlflow
from dotenv import load_dotenv
from pyspark.sql import SparkSession

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.utils import is_databricks
from financial_complaints.config import ProjectConfig, Tags
from financial_complaints.models.basic_complaint_model import BasicComplaintModel

# COMMAND ----------

# Setup MLflow
if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
else:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

# Load configuration
if is_databricks():
    config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
else:
    config_path = "/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
spark = SparkSession.builder.getOrCreate()

# Create tags
tags = Tags(
    git_sha="abcd12345",
    branch="main",
    experiment_name=config.experiment_name_basic,
    model_name="complaint_upheld_basic"
)

print(f"Configuration loaded: {config.catalog_name}.{config.schema_name}")
print(f"Experiment: {config.experiment_name_basic}")

# COMMAND ----------

# Initialize model with the config
basic_model = BasicComplaintModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Load data from Unity Catalog
basic_model.load_data()

# COMMAND ----------

# Prepare features
basic_model.prepare_features()

# COMMAND ----------

# Train the model and log to MLflow
basic_model.train()
basic_model.log_model()

# COMMAND ----------

# Get the run ID
run_id = mlflow.search_runs(
    experiment_names=[config.experiment_name_basic],
    filter_string=f"tags.branch='main'"
).run_id[0]

print(f"Model trained and logged. Run ID: {run_id}")

# Load the model back
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
print("Model loaded successfully")

# COMMAND ----------

# Register model to Unity Catalog
basic_model.register_model()

# COMMAND ----------

# Test predictions on a sample
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)
X_test = test_set.toPandas().drop(columns=[config.target])

predictions = basic_model.load_latest_model_and_predict(X_test)
print(f"Sample predictions: {predictions[:5]}")

# COMMAND ----------

# Retrieve run metadata
metrics, params = basic_model.retrieve_current_run_metadata()
print(f"Model metrics: {metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Successfully Trained and Registered!
# MAGIC 
# MAGIC The model is now available at:
# MAGIC - **Model Name**: `{config.catalog_name}.{config.schema_name}.complaint_upheld_model_basic`
# MAGIC - **Alias**: `latest-model`