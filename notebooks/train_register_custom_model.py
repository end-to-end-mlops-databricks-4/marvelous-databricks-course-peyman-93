# Databricks notebook source
# MAGIC %md
# MAGIC # Train and Register Custom PyFunc Model
# MAGIC This notebook demonstrates custom model wrapping with business logic

# COMMAND ----------

import os
import sys
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

# Add src to path
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.utils import is_databricks
from financial_complaints.config import ProjectConfig, Tags
from financial_complaints.models.custom_complaint_model import CustomComplaintModel

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
    experiment_name=config.experiment_name_custom,
    model_name="complaint_upheld_custom"
)

print(f"Configuration loaded: {config.catalog_name}.{config.schema_name}")
print(f"Experiment: {config.experiment_name_custom}")

# COMMAND ----------

# Initialize custom model
custom_model = CustomComplaintModel(
    config=config,
    tags=tags,
    spark=spark
)

# COMMAND ----------

# Load and prepare data
custom_model.load_data()
custom_model.prepare_features()

# COMMAND ----------

# Train the base model
custom_model.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom PyFunc Wrapper Classes

# COMMAND ----------

class ComplaintPredictionWrapper(mlflow.pyfunc.PythonModel):
    """Custom model wrapper with business logic for complaint predictions."""
    
    def __init__(self, model, preprocessor=None, threshold=0.5):
        """Initialize with a trained model, preprocessor, and custom threshold.
        
        Args:
            model: Trained sklearn model
            preprocessor: Sklearn preprocessor (ColumnTransformer)
            threshold: Probability threshold for positive class
        """
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
        
    def _preprocess_input(self, model_input):
        """Apply preprocessing to input data."""
        # Fill missing values
        for col in model_input.columns:
            if model_input[col].dtype == 'object':
                model_input[col] = model_input[col].fillna('MISSING')
            else:
                model_input[col] = model_input[col].fillna(0)
        
        # Apply preprocessor if available
        if self.preprocessor is not None:
            processed = self.preprocessor.transform(model_input)
            # Convert to DataFrame and handle any remaining NaN/inf
            processed = pd.DataFrame(processed).fillna(0).replace([np.inf, -np.inf], 0)
            return processed.values
        else:
            # Just ensure no NaN/inf values
            return model_input.fillna(0).replace([np.inf, -np.inf], 0).values
        
    def predict(self, context, model_input):
        """Custom prediction logic with additional information.
        
        Returns both predictions and confidence scores.
        """
        if isinstance(model_input, pd.DataFrame):
            # Preprocess the input
            processed_input = self._preprocess_input(model_input.copy())
            
            # Get probabilities
            probabilities = self.model.predict_proba(processed_input)[:, 1]
            
            # Apply custom threshold
            predictions = (probabilities >= self.threshold).astype(int)
            
            # Create detailed output
            result = pd.DataFrame({
                'prediction': predictions,
                'upheld_probability': probabilities,
                'confidence': np.abs(probabilities - 0.5) * 2,  # Confidence score 0-1
                'risk_category': pd.cut(
                    probabilities,
                    bins=[0, 0.3, 0.5, 0.7, 1.0],
                    labels=['Low', 'Medium-Low', 'Medium-High', 'High']
                )
            })
            
            return result
        else:
            raise ValueError("Input must be a pandas DataFrame")

# COMMAND ----------

class ComplaintModelWithExplanation(mlflow.pyfunc.PythonModel):
    """Custom model that provides explanations for predictions."""
    
    def __init__(self, model, feature_importance, preprocessor=None, threshold=0.5):
        """Initialize with model, feature importance, and preprocessor."""
        self.model = model
        self.feature_importance = feature_importance
        self.preprocessor = preprocessor
        self.threshold = threshold
    
    def load_context(self, context):
        """Load any additional artifacts."""
        pass
        
    def _preprocess_input(self, model_input):
        """Apply preprocessing to input data."""
        # Fill missing values
        for col in model_input.columns:
            if model_input[col].dtype == 'object':
                model_input[col] = model_input[col].fillna('MISSING')
            else:
                model_input[col] = model_input[col].fillna(0)
        
        # Apply preprocessor if available
        if self.preprocessor is not None:
            processed = self.preprocessor.transform(model_input)
            processed = pd.DataFrame(processed).fillna(0).replace([np.inf, -np.inf], 0)
            return processed.values
        else:
            return model_input.fillna(0).replace([np.inf, -np.inf], 0).values
        
    def predict(self, context, model_input):
        """Predict with explanations."""
        if isinstance(model_input, pd.DataFrame):
            # Store original input for explanations
            original_input = model_input.copy()
            
            # Preprocess for prediction
            processed_input = self._preprocess_input(model_input.copy())
            
            # Get predictions
            probabilities = self.model.predict_proba(processed_input)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)
            
            # Get top features for each prediction
            explanations = []
            for idx in range(len(original_input)):
                # Get feature values for this instance
                instance = original_input.iloc[idx]
                
                # Find top contributing features
                feature_contributions = []
                for feat, importance in self.feature_importance.items():
                    if feat in instance.index:
                        value = instance[feat]
                        # Convert categorical to numeric for contribution calc
                        if pd.notna(value):
                            if isinstance(value, str):
                                contribution = importance  # Just use importance for categoricals
                            else:
                                contribution = value * importance
                        else:
                            contribution = 0
                        feature_contributions.append((feat, contribution))
                
                # Sort and get top 3
                top_features = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)[:3]
                explanation = "; ".join([f"{feat}: {contrib:.3f}" for feat, contrib in top_features])
                explanations.append(explanation)
            
            result = pd.DataFrame({
                'prediction': predictions,
                'probability': probabilities,
                'explanation': explanations
            })
            
            return result
        else:
            raise ValueError("Input must be a pandas DataFrame")

# COMMAND ----------

class ComplaintModelWithBusinessRules(mlflow.pyfunc.PythonModel):
    """Custom model that applies business rules on top of ML predictions."""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        
    def _preprocess_input(self, model_input):
        """Apply preprocessing to input data."""
        # Fill missing values
        for col in model_input.columns:
            if model_input[col].dtype == 'object':
                model_input[col] = model_input[col].fillna('MISSING')
            else:
                model_input[col] = model_input[col].fillna(0)
        
        # Apply preprocessor if available
        if self.preprocessor is not None:
            processed = self.preprocessor.transform(model_input)
            processed = pd.DataFrame(processed).fillna(0).replace([np.inf, -np.inf], 0)
            return processed.values
        else:
            return model_input.fillna(0).replace([np.inf, -np.inf], 0).values
        
    def predict(self, context, model_input):
        """Apply business rules to predictions."""
        if isinstance(model_input, pd.DataFrame):
            # Store original for business rules
            original_input = model_input.copy()
            
            # Preprocess for ML model
            processed_input = self._preprocess_input(model_input.copy())
            
            # Get base predictions
            base_predictions = self.model.predict(processed_input)
            base_probabilities = self.model.predict_proba(processed_input)[:, 1]
            
            # Apply business rules on original data
            final_predictions = base_predictions.copy()
            
            # Rule 1: If company has "Bank of America" and high probability, always uphold
            if 'Company' in original_input.columns:
                high_risk_companies = ['Bank of America', 'Wells Fargo']
                for idx, company in enumerate(original_input['Company']):
                    if company in high_risk_companies and base_probabilities[idx] > 0.4:
                        final_predictions[idx] = 1
            
            # Rule 2: If processing_days > 60, increase chance of upheld
            if 'processing_days' in original_input.columns:
                for idx, days in enumerate(original_input['processing_days']):
                    if pd.notna(days) and days > 60 and base_probabilities[idx] > 0.3:
                        final_predictions[idx] = 1
            
            # Rule 3: Priority states get extra scrutiny
            if 'State' in original_input.columns:
                priority_states = ['CA', 'TX', 'FL', 'NY']
                for idx, state in enumerate(original_input['State']):
                    if state in priority_states and base_probabilities[idx] > 0.45:
                        final_predictions[idx] = 1
            
            result = pd.DataFrame({
                'ml_prediction': base_predictions,
                'final_prediction': final_predictions,
                'probability': base_probabilities,
                'rules_applied': final_predictions != base_predictions
            })
            
            return result
        else:
            raise ValueError("Input must be a pandas DataFrame")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Custom Models

# COMMAND ----------

# Train and get the base model
model = custom_model.model
preprocessor = custom_model.preprocessor  # Get the preprocessor
feature_importance = dict(zip(custom_model.X_train.columns, model.feature_importances_))

# Sort feature importance
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])

# COMMAND ----------

# Log Model 1: With custom threshold and confidence scores
mlflow.set_experiment(config.experiment_name_custom)

# Fix: Use model_dump() instead of dict()
with mlflow.start_run(run_name="custom_model_with_threshold", tags=tags.model_dump()) as run:
    run_id_threshold = run.info.run_id
    
    # Create wrapped model with preprocessor
    wrapped_model = ComplaintPredictionWrapper(model, preprocessor, threshold=0.6)
    
    # Create signature - use the processed test data
    sample_input = custom_model.X_test.head(5)
    sample_output = wrapped_model.predict(None, sample_input)
    signature = infer_signature(sample_input, sample_output)
    
    # Log the custom model
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="custom_complaint_model",
        signature=signature
    )
    
    # Log metrics
    mlflow.log_metric("custom_threshold", 0.6)
    mlflow.log_params({"wrapper_type": "threshold_and_confidence"})
    
    print(f"Custom threshold model logged: {run_id_threshold}")

# COMMAND ----------

# Log Model 2: With explanations
with mlflow.start_run(run_name="custom_model_with_explanations", tags=tags.model_dump()) as run:
    run_id_explanation = run.info.run_id
    
    # Create wrapped model with explanations and preprocessor
    wrapped_model = ComplaintModelWithExplanation(
        model, 
        feature_importance, 
        preprocessor, 
        threshold=0.5
    )
    
    # Create signature
    sample_input = custom_model.X_test.head(5)
    sample_output = wrapped_model.predict(None, sample_input)
    signature = infer_signature(sample_input, sample_output)
    
    # Log the custom model
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="custom_complaint_model_explained",
        signature=signature
    )
    
    # Log feature importance as artifact
    import json
    with open("feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    mlflow.log_artifact("feature_importance.json")
    os.remove("feature_importance.json")
    
    mlflow.log_params({"wrapper_type": "with_explanations"})
    
    print(f"Explanation model logged: {run_id_explanation}")

# COMMAND ----------

# Log Model 3: With business rules
with mlflow.start_run(run_name="custom_model_with_business_rules", tags=tags.model_dump()) as run:
    run_id_rules = run.info.run_id
    
    # Create wrapped model with business rules and preprocessor
    wrapped_model = ComplaintModelWithBusinessRules(model, preprocessor)
    
    # Use the preprocessed features for signature
    sample_input = custom_model.X_test.head(5)
    sample_output = wrapped_model.predict(None, sample_input)
    signature = infer_signature(sample_input, sample_output)
    
    # Log the custom model
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="custom_complaint_model_rules",
        signature=signature
    )
    
    mlflow.log_params({"wrapper_type": "business_rules"})
    mlflow.log_text(
        "Rules: 1) High-risk companies, 2) Long processing days, 3) Priority states",
        "business_rules.txt"
    )
    
    print(f"Business rules model logged: {run_id_rules}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Custom Models

# COMMAND ----------

# Register the threshold model
model_name = f"{config.catalog_name}.{config.schema_name}.complaint_custom_threshold"
mlflow.register_model(
    model_uri=f"runs:/{run_id_threshold}/custom_complaint_model",
    name=model_name,
    tags={"type": "custom_threshold"}
)
print(f"Registered: {model_name}")

# Register the explanation model
model_name_explained = f"{config.catalog_name}.{config.schema_name}.complaint_custom_explained"
mlflow.register_model(
    model_uri=f"runs:/{run_id_explanation}/custom_complaint_model_explained",
    name=model_name_explained,
    tags={"type": "with_explanations"}
)
print(f"Registered: {model_name_explained}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Custom Models

# COMMAND ----------

# Load and test the threshold model
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id_threshold}/custom_complaint_model")

# Get test data
test_sample = custom_model.X_test.head(10)

# Make predictions with custom model
custom_predictions = loaded_model.predict(test_sample)
print("Custom Threshold Model Output:")
print(custom_predictions)
print(f"\nPredictions with threshold 0.6: {custom_predictions['prediction'].tolist()}")
print(f"Risk categories: {custom_predictions['risk_category'].value_counts().to_dict()}")

# COMMAND ----------

# Test explanation model
loaded_explanation_model = mlflow.pyfunc.load_model(f"runs:/{run_id_explanation}/custom_complaint_model_explained")

explanation_predictions = loaded_explanation_model.predict(test_sample)
print("\nModel with Explanations Output:")
for idx in range(min(3, len(explanation_predictions))):
    print(f"\nInstance {idx}:")
    print(f"  Prediction: {explanation_predictions.iloc[idx]['prediction']}")
    print(f"  Probability: {explanation_predictions.iloc[idx]['probability']:.3f}")
    print(f"  Explanation: {explanation_predictions.iloc[idx]['explanation']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

# Compare basic vs custom predictions
# Need to preprocess test sample for basic model
if preprocessor:
    test_sample_processed = preprocessor.transform(test_sample)
    test_sample_processed = pd.DataFrame(test_sample_processed).fillna(0).replace([np.inf, -np.inf], 0).values
    basic_predictions = model.predict(test_sample_processed)
else:
    test_sample_safe = test_sample.fillna(0).replace([np.inf, -np.inf], 0)
    basic_predictions = model.predict(test_sample_safe)

custom_with_threshold = loaded_model.predict(test_sample)['prediction'].values

comparison = pd.DataFrame({
    'basic_model': basic_predictions,
    'custom_threshold_0.6': custom_with_threshold,
    'difference': basic_predictions != custom_with_threshold
})

print(f"Predictions differ in {comparison['difference'].sum()} out of {len(comparison)} cases")
print(f"Basic model positive rate: {comparison['basic_model'].mean():.2%}")
print(f"Custom model positive rate: {comparison['custom_threshold_0.6'].mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC We've created three custom PyFunc models:
# MAGIC 
# MAGIC 1. **Threshold Model**: Adjustable probability threshold with confidence scores
# MAGIC 2. **Explanation Model**: Provides feature-based explanations for predictions
# MAGIC 3. **Business Rules Model**: Applies domain-specific rules on top of ML predictions
# MAGIC 
# MAGIC These custom models demonstrate how to:
# MAGIC - Wrap sklearn models with custom logic
# MAGIC - Include preprocessing in the wrapper
# MAGIC - Add business rules to ML predictions
# MAGIC - Provide interpretable outputs
# MAGIC - Control prediction thresholds
# MAGIC - Return structured outputs beyond simple predictions