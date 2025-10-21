"""Basic model implementation for Financial Complaints.

Place this in: src/financial_complaints/models/basic_complaint_model.py
"""

import mlflow
import mlflow.sklearn
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


class BasicComplaintModel:
    """Basic model class for complaint upheld prediction using Random Forest.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
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
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.complaint_upheld_model_basic"

    def load_data(self) -> None:
        """Load training and testing data from Delta tables."""
        logger.info("📄 Loading data from Databricks tables...")
        
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.temporal_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.temporal_test_set").toPandas()
        
        self.data_version = "0"  # You can get actual version from Delta history if needed
        
        # Select features
        feature_cols = self.num_features + self.cat_features
        available_features = [col for col in feature_cols if col in self.train_set.columns]
        
        self.X_train = self.train_set[available_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[available_features]
        self.y_test = self.test_set[self.target]
        self.X_temporal = self.temporal_set[available_features]
        self.y_temporal = self.temporal_set[self.target]
        
        logger.info(f"✅ Data loaded - Train: {len(self.train_set)}, Test: {len(self.test_set)}, Temporal: {len(self.temporal_set)}")

    def prepare_features(self) -> None:
        """Prepare features and define preprocessing pipeline."""
        logger.info("🔄 Preparing features...")
        
        # Fill missing values first
        for col in self.X_train.columns:
            if self.X_train[col].dtype == 'object':
                self.X_train[col] = self.X_train[col].fillna('MISSING')
                self.X_test[col] = self.X_test[col].fillna('MISSING')
                self.X_temporal[col] = self.X_temporal[col].fillna('MISSING')
            else:
                self.X_train[col] = self.X_train[col].fillna(0)
                self.X_test[col] = self.X_test[col].fillna(0)
                self.X_temporal[col] = self.X_temporal[col].fillna(0)
        
        # Identify categorical and numerical columns
        actual_cat_features = [col for col in self.cat_features if col in self.X_train.columns]
        actual_num_features = [col for col in self.num_features if col in self.X_train.columns]
        
        # Handle high cardinality features with simple mean encoding
        high_card_features = [col for col in self.config.high_cardinality_features if col in actual_cat_features]
        low_card_features = [col for col in actual_cat_features if col not in high_card_features]
        
        # Store mean encoding maps for later use
        self.mean_encoding_maps = {}
        
        # Apply mean encoding for high cardinality features
        if high_card_features:
            for col in high_card_features:
                if col in self.X_train.columns:
                    # Calculate mean target value for each category
                    mean_map = self.train_set.groupby(col)[self.target].mean().to_dict()
                    default_value = self.y_train.mean()
                    self.mean_encoding_maps[col] = (mean_map, default_value)
                    
                    # Apply to train, test, and temporal sets
                    self.X_train[col] = self.X_train[col].map(mean_map).fillna(default_value)
                    self.X_test[col] = self.X_test[col].map(mean_map).fillna(default_value)
                    self.X_temporal[col] = self.X_temporal[col].map(mean_map).fillna(default_value)
        
        # Create preprocessor for low cardinality features
        preprocessors = []
        
        if low_card_features:
            preprocessors.append(
                ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_features)
            )
        
        if preprocessors:
            self.preprocessor = ColumnTransformer(
                transformers=preprocessors,
                remainder="passthrough",  # Keep numerical features as-is
                sparse_threshold=0  # Force dense output
            )
        else:
            self.preprocessor = None
        
        # Final NaN check and fill
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        self.X_temporal = self.X_temporal.fillna(0)
        
        # Convert to numeric where possible
        for col in self.X_train.columns:
            if self.X_train[col].dtype == 'object':
                try:
                    self.X_train[col] = pd.to_numeric(self.X_train[col], errors='coerce').fillna(0)
                    self.X_test[col] = pd.to_numeric(self.X_test[col], errors='coerce').fillna(0)
                    self.X_temporal[col] = pd.to_numeric(self.X_temporal[col], errors='coerce').fillna(0)
                except:
                    pass
        
        # Define model parameters
        model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_leaf': 10,
            'class_weight': 'balanced',
            'random_state': self.parameters['random_state'],
            'n_jobs': -1
        }
        
        # Initialize model
        self.model = RandomForestClassifier(**model_params)
        
        logger.info("✅ Features prepared and model initialized")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        
        # Apply preprocessing if needed
        if self.preprocessor:
            X_train_processed = self.preprocessor.fit_transform(self.X_train)
            # Ensure no NaN values after preprocessing
            if hasattr(X_train_processed, 'toarray'):
                X_train_processed = X_train_processed.toarray()
            X_train_processed = pd.DataFrame(X_train_processed).fillna(0).values
            self.model.fit(X_train_processed, self.y_train)
        else:
            # Final safety check - ensure no NaN values
            X_train_safe = self.X_train.fillna(0)
            # Replace any remaining infinity values
            X_train_safe = X_train_safe.replace([np.inf, -np.inf], 0)
            self.model.fit(X_train_safe, self.y_train)
        
        logger.info("✅ Model training complete")

    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            
            # Make predictions
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
            
            # Calculate metrics
            test_metrics = {
                'test_accuracy': accuracy_score(self.y_test, y_pred_test),
                'test_roc_auc': roc_auc_score(self.y_test, y_prob_test),
                'test_f1': f1_score(self.y_test, y_pred_test),
                'test_precision': precision_score(self.y_test, y_pred_test),
                'test_recall': recall_score(self.y_test, y_pred_test)
            }
            
            temporal_metrics = {
                'temporal_accuracy': accuracy_score(self.y_temporal, y_pred_temporal),
                'temporal_roc_auc': roc_auc_score(self.y_temporal, y_prob_temporal),
                'temporal_f1': f1_score(self.y_temporal, y_pred_temporal)
            }
            
            # Log metrics
            for name, value in {**test_metrics, **temporal_metrics}.items():
                mlflow.log_metric(name, value)
                logger.info(f"📊 {name}: {value:.4f}")
            
            # Log parameters
            mlflow.log_param("model_type", "RandomForest with preprocessing")
            mlflow.log_params(self.model.get_params())
            
            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred_test)
            
            # Log dataset info
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                signature=signature
            )
            
            logger.info(f"✅ Model logged to MLflow. Run ID: {self.run_id}")

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("📄 Registering the model in UC...")
        
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/model",
            name=self.model_name,
            tags=self.tags,
        )
        
        logger.info(f"✅ Model registered as version {registered_model.version}")
        
        latest_version = registered_model.version
        
        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        
        logger.info(f"✅ Alias 'latest-model' set for version {latest_version}")

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata."""
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Metadata retrieved")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model and make predictions."""
        logger.info("📄 Loading model from MLflow...")
        
        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info("✅ Model loaded successfully")
        
        # Preprocess the input data the same way as training data
        # Select features
        feature_cols = self.num_features + self.cat_features
        available_features = [col for col in feature_cols if col in input_data.columns]
        X_input = input_data[available_features].copy()
        
        # Fill missing values
        for col in X_input.columns:
            if X_input[col].dtype == 'object':
                X_input[col] = X_input[col].fillna('MISSING')
            else:
                X_input[col] = X_input[col].fillna(0)
        
        # Apply mean encoding for high cardinality features if we have the maps
        if hasattr(self, 'mean_encoding_maps'):
            for col, (mean_map, default_value) in self.mean_encoding_maps.items():
                if col in X_input.columns:
                    X_input[col] = X_input[col].map(mean_map).fillna(default_value)
        
        # Apply preprocessor if it exists
        if self.preprocessor:
            input_processed = self.preprocessor.transform(X_input)
            if hasattr(input_processed, 'toarray'):
                input_processed = input_processed.toarray()
            predictions = model.predict(input_processed)
        else:
            # Ensure no NaN values
            X_input = X_input.fillna(0)
            X_input = X_input.replace([np.inf, -np.inf], 0)
            predictions = model.predict(X_input)
        
        return predictions