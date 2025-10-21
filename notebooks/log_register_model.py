# Databricks notebook source
# MAGIC %md
# MAGIC # Log and Register Models - Different Approaches
# MAGIC This notebook demonstrates different ways to log and register models in MLflow

# COMMAND ----------

import os
import sys
import mlflow
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.utils import is_databricks
from financial_complaints.config import ProjectConfig

# COMMAND ----------

# Setup
if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
else:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

# Load config
if is_databricks():
    config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
else:
    config_path = "/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

# COMMAND ----------

# Load data
spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Select available features
available_num = [col for col in config.num_features if col in train_set.columns][:10]  # Just first 10
available_cat = [col for col in config.cat_features if col in train_set.columns][:5]    # Just first 5

X_train = train_set[available_num + available_cat]
y_train = train_set[config.target]
X_test = test_set[available_num + available_cat]
y_test = test_set[config.target]

print(f"Using {len(available_num)} numerical and {len(available_cat)} categorical features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 1: Simple Model Logging

# COMMAND ----------

# Create a simple pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), available_cat),
                ("num", StandardScaler(), available_num)
            ]
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ))
    ]
)

# Train the pipeline
pipeline.fit(X_train, y_train)

# COMMAND ----------

# Simple model logging
mlflow.set_experiment(config.experiment_name_basic)

with mlflow.start_run(run_name="simple_model_logging") as run:
    run_id_simple = run.info.run_id
    
    # Log parameters
    mlflow.log_param("model_type", "RandomForest Pipeline")
    mlflow.log_param("n_features", len(available_num) + len(available_cat))
    
    # Log metrics
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)
    
    # Log model with signature
    signature = infer_signature(
        model_input=X_train,
        model_output=pipeline.predict(X_train)
    )
    
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature
    )
    
    print(f"Model logged with run_id: {run_id_simple}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 2: Model with Custom Artifacts

# COMMAND ----------

mlflow.set_experiment(config.experiment_name_basic)

with mlflow.start_run(run_name="model_with_artifacts") as run:
    run_id_artifacts = run.info.run_id
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train[available_num], y_train)  # Just numerical features
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': available_num,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    # Log config as artifact
    mlflow.log_dict(config.dict(), "config.json")
    
    # Log text descriptions
    mlflow.log_text(
        "This model uses only numerical features for simplicity",
        "model_description.txt"
    )
    
    # Clean up temp file
    os.remove("feature_importance.csv")
    
    print(f"Model with artifacts logged: {run_id_artifacts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 3: Model Registration

# COMMAND ----------

model_name = f"{config.catalog_name}.{config.schema_name}.complaint_model_demo"

# Register the model from the simple logging run
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id_simple}/model',
    name=model_name,
    tags={
        "experiment": "demo",
        "type": "sklearn_pipeline"
    }
)

print(f"Model registered: {model_name} v{model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 4: Custom PyFunc Model

# COMMAND ----------

class ComplaintModelWrapper(mlflow.pyfunc.PythonModel):
    """Custom model wrapper with preprocessing logic."""
    
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        """Custom prediction logic."""
        if isinstance(model_input, pd.DataFrame):
            # Apply custom logic (e.g., thresholding)
            predictions = self.model.predict(model_input)
            
            # Custom post-processing
            adjusted_predictions = np.where(
                self.model.predict_proba(model_input)[:, 1] > 0.6,  # Custom threshold
                1, 0
            )
            
            return adjusted_predictions
        else:
            raise ValueError("Input must be a pandas DataFrame")

# COMMAND ----------

# Train a model to wrap
base_model = RandomForestClassifier(n_estimators=50, random_state=42)
base_model.fit(X_train[available_num], y_train)

# Wrap the model
wrapped_model = ComplaintModelWrapper(base_model)

# Log the wrapped model
mlflow.set_experiment(config.experiment_name_custom)

with mlflow.start_run(run_name="custom_pyfunc_model") as run:
    run_id_pyfunc = run.info.run_id
    
    signature = infer_signature(
        model_input=X_train[available_num],
        model_output=wrapped_model.predict(None, X_train[available_num])
    )
    
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc_model",
        signature=signature
    )
    
    print(f"Custom PyFunc model logged: {run_id_pyfunc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Versioning and Aliases

# COMMAND ----------

client = MlflowClient()

# Set an alias for the model version
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=model_version.version
)

print(f"Alias 'champion' set for version {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Models - Different Methods

# COMMAND ----------

# Method 1: Load by run_id
model_by_run = mlflow.sklearn.load_model(f"runs:/{run_id_simple}/model")
print(f"✅ Loaded model by run_id")

# Method 2: Load by model name and version
model_by_version = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version.version}")
print(f"✅ Loaded model by version")

# Method 3: Load by alias
model_by_alias = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
print(f"✅ Loaded model by alias")

# Method 4: Load PyFunc model
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id_pyfunc}/pyfunc_model")
print(f"✅ Loaded custom PyFunc model")

# COMMAND ----------

# Test predictions
test_sample = X_test.head(5)

# Using the pipeline model
pred_pipeline = model_by_alias.predict(test_sample)
print(f"Pipeline predictions: {pred_pipeline}")

# Using the PyFunc model (numerical features only)
pred_pyfunc = pyfunc_model.predict(test_sample[available_num])
print(f"PyFunc predictions: {pred_pyfunc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search and Compare Models

# COMMAND ----------

# Search for all runs in our experiments
runs = mlflow.search_runs(
    experiment_names=[config.experiment_name_basic, config.experiment_name_custom],
    filter_string="metrics.test_accuracy > 0",
    order_by=["metrics.test_accuracy DESC"]
)

if not runs.empty:
    print("Model Comparison:")
    print(runs[['run_id', 'experiment_id', 'metrics.test_accuracy', 'tags.mlflow.runName']].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Up (Optional)

# COMMAND ----------

# List all versions of a model
versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")

# To delete a model version (uncomment if needed):
# client.delete_model_version(name=model_name, version="1")

# To delete entire registered model (uncomment if needed):
# client.delete_registered_model(name=model_name)

print("Demo complete!")