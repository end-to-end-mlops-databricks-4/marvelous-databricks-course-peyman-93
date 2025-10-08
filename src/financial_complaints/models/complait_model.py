"""
Class-based model implementation for Financial Complaints prediction.
Place this in: src/financial_complaints/models/complaint_model.py

This is an alternative to basic_complaint_model.py with more features.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, average_precision_score
)
from category_encoders import TargetEncoder
from sklearn.preprocessing import RobustScaler

from financial_complaints.config import ProjectConfig
from financial_complaints.data_processor import DataProcessor


class ComplaintUpheldModel:
    """Advanced model class for predicting whether complaints will be upheld.
    
    This class provides a complete pipeline including:
    - Data loading from Unity Catalog
    - Feature engineering using DataProcessor
    - Model training with multiple algorithms
    - MLflow tracking and model registry
    - Prediction and evaluation methods
    """
    
    def __init__(self, config: ProjectConfig, spark: SparkSession, 
                 experiment_name: str = None, tags: Dict[str, str] = None):
        """Initialize the model with configuration.
        
        :param config: Project configuration from YAML
        :param spark: Active Spark session
        :param experiment_name: MLflow experiment name (optional)
        :param tags: Tags for MLflow tracking (optional)
        """
        self.config = config
        self.spark = spark
        self.tags = tags or {}
        
        # Model settings
        self.target = config.target
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        
        # MLflow settings
        self.experiment_name = experiment_name or config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.{config.target}_model"
        
        # Initialize placeholders
        self.train_df = None
        self.test_df = None
        self.temporal_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_temporal = None
        self.y_temporal = None
        self.model = None
        self.run_id = None
        self.data_processor = None
        self.selected_features = None
        
    def load_data(self) -> None:
        """Load data from Unity Catalog tables."""
        logger.info("Loading data from Unity Catalog...")
        
        # Load from Spark tables
        train_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        test_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        temporal_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.temporal_test_set")
        
        # Convert to Pandas
        self.train_df = train_spark.toPandas()
        self.test_df = test_spark.toPandas()
        self.temporal_df = temporal_spark.toPandas()
        
        # Store Spark DataFrame reference for MLflow logging
        self.train_spark_df = train_spark
        
        logger.info(f"✅ Data loaded - Train: {len(self.train_df)}, Test: {len(self.test_df)}, Temporal: {len(self.temporal_df)}")
        
    def prepare_features(self, feature_selection: bool = True, n_features: int = 100) -> None:
        """Prepare features using the DataProcessor.
        
        :param feature_selection: Whether to perform feature selection
        :param n_features: Number of top features to select
        """
        logger.info("Preparing features...")
        
        # Initialize data processor with config
        self.data_processor = DataProcessor(self.train_df, self.config, self.spark)
        
        # Create feature configuration
        feature_config = {
            'model_features': self.config.num_features + self.config.cat_features,
            'num_features': self.config.num_features,
            'cat_features': self.config.cat_features,
            'target': self.target
        }
        
        # Initialize feature engineering
        from financial_complaints.models.feature_engineering import FeatureEngineering
        fe = FeatureEngineering(feature_config)
        
        # Process features
        X_train, y_train = fe.prepare_features(self.train_df, fit=True)
        X_test, y_test = fe.prepare_features(self.test_df, fit=False)
        X_temporal, y_temporal = fe.prepare_features(self.temporal_df, fit=False)
        
        # Align columns
        common_cols = [col for col in X_train.columns if col in X_test.columns]
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        X_temporal = X_temporal[common_cols]
        
        if feature_selection:
            logger.info(f"Performing feature selection (top {n_features} features)...")
            
            # Quick feature importance using RandomForest
            rf_selector = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1
            )
            rf_selector.fit(X_train, y_train)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf_selector.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top features
            top_features = importance_df.head(min(n_features, len(importance_df)))['feature'].tolist()
            
            self.X_train = X_train[top_features]
            self.X_test = X_test[top_features]
            self.X_temporal = X_temporal[top_features]
            self.selected_features = top_features
            
            logger.info(f"✅ Selected {len(top_features)} features")
        else:
            self.X_train = X_train
            self.X_test = X_test
            self.X_temporal = X_temporal
            self.selected_features = common_cols
        
        self.y_train = y_train
        self.y_test = y_test
        self.y_temporal = y_temporal
        
        # Store feature engineering object
        self.feature_engineering = fe
        
        logger.info(f"✅ Features prepared - Shape: {self.X_train.shape}")
        
    def train(self, model_type: str = "RandomForest", model_params: Dict[str, Any] = None) -> None:
        """Train the model.
        
        :param model_type: Type of model to train ("RandomForest", "XGBoost", etc.)
        :param model_params: Model hyperparameters
        """
        logger.info(f"Training {model_type} model...")
        
        # Default parameters based on model type
        if model_params is None:
            if model_type == "RandomForest":
                model_params = {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_leaf': 10,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                }
            elif model_type == "XGBoost":
                import xgboost as xgb
                model_params = {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
        
        # Initialize model
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(**model_params)
        elif model_type == "XGBoost":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        self.model_type = model_type
        self.model_params = model_params
        
        logger.info("✅ Model training complete")
        
    def evaluate(self, dataset: str = "test") -> Dict[str, float]:
        """Evaluate the model on specified dataset.
        
        :param dataset: Which dataset to evaluate on ("test", "temporal", or "both")
        :return: Dictionary of metrics
        """
        metrics = {}
        
        if dataset in ["test", "both"]:
            y_pred = self.model.predict(self.X_test)
            y_prob = self.model.predict_proba(self.X_test)[:, 1]
            
            metrics.update({
                'test_accuracy': accuracy_score(self.y_test, y_pred),
                'test_roc_auc': roc_auc_score(self.y_test, y_prob),
                'test_f1': f1_score(self.y_test, y_pred),
                'test_precision': precision_score(self.y_test, y_pred),
                'test_recall': recall_score(self.y_test, y_pred),
                'test_avg_precision': average_precision_score(self.y_test, y_prob)
            })
        
        if dataset in ["temporal", "both"]:
            y_pred = self.model.predict(self.X_temporal)
            y_prob = self.model.predict_proba(self.X_temporal)[:, 1]
            
            metrics.update({
                'temporal_accuracy': accuracy_score(self.y_temporal, y_pred),
                'temporal_roc_auc': roc_auc_score(self.y_temporal, y_prob),
                'temporal_f1': f1_score(self.y_temporal, y_pred),
                'temporal_precision': precision_score(self.y_temporal, y_pred),
                'temporal_recall': recall_score(self.y_temporal, y_pred)
            })
        
        return metrics
        
    def log_model(self) -> str:
        """Log model to MLflow with comprehensive tracking.
        
        :return: MLflow run ID
        """
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            
            # Log tags
            mlflow.set_tags({
                "model_type": self.model_type,
                "training_date": datetime.now().isoformat(),
                **self.tags
            })
            
            # Evaluate model
            metrics = self.evaluate(dataset="both")
            
            # Log parameters
            mlflow.log_params(self.model_params)
            mlflow.log_param("n_features", len(self.selected_features))
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            signature = infer_signature(
                model_input=self.X_train,
                model_output=self.model.predict(self.X_train)
            )
            
            if self.model_type == "XGBoost":
                mlflow.xgboost.log_model(
                    xgb_model=self.model,
                    artifact_path="model",
                    signature=signature
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    signature=signature
                )
            
            # Log feature names
            mlflow.log_text('\n'.join(self.selected_features), "features.txt")
            
            # Log feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.selected_features,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_df.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
                import os
                os.remove("feature_importance.csv")
            
            # Log dataset info
            if self.train_spark_df is not None:
                dataset = mlflow.data.from_spark(
                    self.train_spark_df,
                    table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                    version="0"
                )
                mlflow.log_input(dataset, context="training")
            
            logger.info(f"✅ Model logged - Run ID: {self.run_id}")
            logger.info(f"   Test ROC-AUC: {metrics.get('test_roc_auc', 0):.4f}")
            logger.info(f"   Temporal ROC-AUC: {metrics.get('temporal_roc_auc', 0):.4f}")
            
        return self.run_id
        
    def register_model(self, model_alias: str = "champion") -> None:
        """Register model in Unity Catalog.
        
        :param model_alias: Alias to set for the model version
        """
        logger.info("Registering model...")
        
        # Register the model
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/model",
            name=self.model_name,
            tags=self.tags
        )
        
        # Set alias
        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias=model_alias,
            version=registered_model.version
        )
        
        logger.info(f"✅ Model registered as {self.model_name} v{registered_model.version}")
        logger.info(f"   Alias: {model_alias}")
        
    def load_model(self, version: str = "latest") -> Any:
        """Load a registered model.
        
        :param version: Model version or alias to load
        :return: Loaded model
        """
        if version == "latest":
            model_uri = f"models:/{self.model_name}@champion"
        else:
            model_uri = f"models:/{self.model_name}/{version}"
        
        if self.model_type == "XGBoost":
            model = mlflow.xgboost.load_model(model_uri)
        else:
            model = mlflow.sklearn.load_model(model_uri)
        
        return model
        
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        :param input_data: Input DataFrame
        :return: Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Apply same feature engineering
        X_input, _ = self.feature_engineering.prepare_features(input_data, fit=False)
        
        # Select same features
        X_input = X_input[self.selected_features]
        
        # Make predictions
        predictions = self.model.predict(X_input)
        
        return predictions
    
    def predict_proba(self, input_data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.
        
        :param input_data: Input DataFrame
        :return: Probability array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Apply same feature engineering
        X_input, _ = self.feature_engineering.prepare_features(input_data, fit=False)
        
        # Select same features
        X_input = X_input[self.selected_features]
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_input)
        
        return probabilities