"""Custom model implementation for Financial Complaints with PyFunc wrapper support.

"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score
)
from sklearn.preprocessing import OneHotEncoder

from financial_complaints.config import ProjectConfig, Tags


class CustomComplaintModel:
    """Custom model class for complaint prediction with PyFunc wrapper support."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the custom model with project configuration."""
        self.config = config
        self.spark = spark
        self.tags = tags.dict()

        # Extract settings from config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_custom
        self.model_name = f"{self.catalog_name}.{self.schema_name}.complaint_upheld_model_custom"
        
        # Model components
        self.model = None
        self.preprocessor = None
        self.mean_encoding_maps = {}

    def load_data(self) -> None:
        """Load training and testing data from Delta tables."""
        logger.info("📄 Loading data from Databricks tables...")
        
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.temporal_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.temporal_test_set").toPandas()
        
        # Store full dataframes for business rules
        self.test_df = self.test_set.copy()
        
        # Select features
        feature_cols = self.num_features + self.cat_features
        available_features = [col for col in feature_cols if col in self.train_set.columns]
        
        self.X_train = self.train_set[available_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[available_features]
        self.y_test = self.test_set[self.target]
        self.X_temporal = self.temporal_set[available_features]
        self.y_temporal = self.temporal_set[self.target]
        
        logger.info(f"✅ Data loaded - Train: {len(self.train_set)}, Test: {len(self.test_set)}")

    def prepare_features(self) -> None:
        """Prepare features with custom preprocessing."""
        logger.info("🔄 Preparing features for custom model...")
        
        # Fill missing values
        for col in self.X_train.columns:
            if self.X_train[col].dtype == 'object':
                self.X_train[col] = self.X_train[col].fillna('MISSING')
                self.X_test[col] = self.X_test[col].fillna('MISSING')
                self.X_temporal[col] = self.X_temporal[col].fillna('MISSING')
            else:
                self.X_train[col] = self.X_train[col].fillna(0)
                self.X_test[col] = self.X_test[col].fillna(0)
                self.X_temporal[col] = self.X_temporal[col].fillna(0)
        
        # Identify feature types
        actual_cat_features = [col for col in self.cat_features if col in self.X_train.columns]
        actual_num_features = [col for col in self.num_features if col in self.X_train.columns]
        
        # Handle high cardinality with mean encoding
        high_card_features = [col for col in self.config.high_cardinality_features if col in actual_cat_features]
        low_card_features = [col for col in actual_cat_features if col not in high_card_features]
        
        # Apply mean encoding
        if high_card_features:
            for col in high_card_features:
                if col in self.X_train.columns:
                    mean_map = self.train_set.groupby(col)[self.target].mean().to_dict()
                    default_value = self.y_train.mean()
                    self.mean_encoding_maps[col] = (mean_map, default_value)
                    
                    self.X_train[col] = self.X_train[col].map(mean_map).fillna(default_value)
                    self.X_test[col] = self.X_test[col].map(mean_map).fillna(default_value)
                    self.X_temporal[col] = self.X_temporal[col].map(mean_map).fillna(default_value)
        
        # One-hot encode low cardinality features
        if low_card_features:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_features)
                ],
                remainder="passthrough",
                sparse_threshold=0
            )
        else:
            self.preprocessor = None
        
        # Final cleanup
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        self.X_temporal = self.X_temporal.fillna(0)
        
        logger.info("✅ Custom features prepared")

    def train(self) -> None:
        """Train the model for custom wrapper."""
        logger.info("🚀 Training custom model...")
        
        # Model parameters
        model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_leaf': 10,
            'class_weight': 'balanced',
            'random_state': self.parameters['random_state'],
            'n_jobs': -1
        }
        
        self.model = RandomForestClassifier(**model_params)
        
        # Apply preprocessing if needed
        if self.preprocessor:
            X_train_processed = self.preprocessor.fit_transform(self.X_train)
            X_train_processed = pd.DataFrame(X_train_processed).fillna(0).values
            self.model.fit(X_train_processed, self.y_train)
        else:
            X_train_safe = self.X_train.fillna(0).replace([np.inf, -np.inf], 0)
            self.model.fit(X_train_safe, self.y_train)
        
        logger.info("✅ Custom model training complete")

    def log_base_model(self) -> str:
        """Log the base model before wrapping."""
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name="base_model_for_custom", tags=self.tags) as run:
            self.run_id = run.info.run_id
            
            # Make predictions for metrics
            if self.preprocessor:
                X_test_processed = self.preprocessor.transform(self.X_test)
                X_temporal_processed = self.preprocessor.transform(self.X_temporal)
                y_pred_test = self.model.predict(X_test_processed)
                y_pred_temporal = self.model.predict(X_temporal_processed)
                y_prob_test = self.model.predict_proba(X_test_processed)[:, 1]
                y_prob_temporal = self.model.predict_proba(X_temporal_processed)[:, 1]
            else:
                y_pred_test = self.model.predict(self.X_test)
                y_pred_temporal = self.model.predict(self.X_temporal)
                y_prob_test = self.model.predict_proba(self.X_test)[:, 1]
                y_prob_temporal = self.model.predict_proba(self.X_temporal)[:, 1]
            
            # Calculate and log metrics
            test_metrics = {
                'test_accuracy': accuracy_score(self.y_test, y_pred_test),
                'test_roc_auc': roc_auc_score(self.y_test, y_prob_test),
                'test_f1': f1_score(self.y_test, y_pred_test),
                'test_precision': precision_score(self.y_test, y_pred_test),
                'test_recall': recall_score(self.y_test, y_pred_test)
            }
            
            mlflow.log_metrics(test_metrics)
            mlflow.log_params(self.model.get_params())
            
            # Log base model
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="base_model"
            )
            
            logger.info(f"✅ Base model logged. Run ID: {self.run_id}")
            
        return self.run_id

    def create_custom_wrapper(self, wrapper_type: str = "threshold"):
        """Create a custom PyFunc wrapper for the model.
        
        Args:
            wrapper_type: Type of wrapper ('threshold', 'explanation', 'rules')
        """
        if wrapper_type == "threshold":
            from train_register_custom_model import ComplaintPredictionWrapper
            return ComplaintPredictionWrapper(self.model, threshold=0.6)
        
        elif wrapper_type == "explanation":
            from train_register_custom_model import ComplaintModelWithExplanation
            feature_importance = dict(zip(self.X_train.columns, self.model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])
            return ComplaintModelWithExplanation(self.model, feature_importance)
        
        elif wrapper_type == "rules":
            from train_register_custom_model import ComplaintModelWithBusinessRules
            return ComplaintModelWithBusinessRules(self.model)
        
        else:
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")

    def register_custom_model(self, model_uri: str, alias: str = "latest-custom") -> None:
        """Register custom model in Unity Catalog."""
        logger.info("📄 Registering custom model...")
        
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
            tags={**self.tags, "model_type": "custom_pyfunc"}
        )
        
        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias=alias,
            version=registered_model.version
        )
        
        logger.info(f"✅ Custom model registered as version {registered_model.version}")