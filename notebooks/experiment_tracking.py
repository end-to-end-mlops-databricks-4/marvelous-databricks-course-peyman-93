# Databricks notebook source
"""
Financial Complaints Model Training with MLflow Experiment Tracking
Objective 1: Predicting Whether a Consumer's Complaint Will Be Upheld
"""

# COMMAND ----------

import json
import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
# Remove TargetEncoder import - we'll use simple mean encoding instead
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

# COMMAND ----------

# Add src to path - handle both Databricks and local environments
if os.path.exists('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src'):
    # Databricks path
    sys.path.append('/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/src')
else:
    # Local path
    sys.path.append('/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/src')

from financial_complaints.utils import is_databricks
from financial_complaints.config import ProjectConfig, Tags

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and MLflow Setup

# COMMAND ----------

# Load configuration FIRST - before using any config values
if is_databricks():
    config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
else:
    config_path = "/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
print(f"Config loaded successfully!")
print(f"Catalog: {config.catalog_name}, Schema: {config.schema_name}")
print(f"Experiment Name: {config.experiment_name_basic}")

# COMMAND ----------

# Set up MLflow tracking with proper environment handling
if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
else:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow Registry URI: {mlflow.get_registry_uri()}")

# Now set experiment using the loaded config
experiment = mlflow.set_experiment(
    experiment_name=config.experiment_name_basic  # This will be "/Shared/financial-complaints-basic"
)

mlflow.set_experiment_tags({
    "repository_name": "end-to-end-mlops-databricks-4/marvelous-databricks-course-peyman-93",
    "objective": "complaint_upheld_prediction",
    "team": "data_science",
    "project": "financial_complaints"
})

print(f"Experiment Name: {config.experiment_name_basic}")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Class

# COMMAND ----------

class FeatureEngineering:
    """Feature engineering pipeline for complaint upheld prediction."""
    
    def __init__(self, feature_config):
        self.feature_config = feature_config
        self.encoders = {}
        self.scaler = None
        self.selected_features = None
        self.high_card_cols = None
        self.low_card_cols = None
        self.numerical_cols = None
        self.encoded_columns = None
        self.low_card_categories = {}
        self.target_encoding_maps = {}
    
    def prepare_features(self, df, target_col='complaint_upheld', fit=True):
        """Prepare features for modeling."""
        # Filter to only features that exist in the dataframe
        available_features = [f for f in self.feature_config['model_features'] if f in df.columns]
        X = df[available_features].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('MISSING')
            else:
                X[col] = X[col].fillna(0)
        
        if fit:
            self.categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
            self.numerical_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
            self.high_card_cols = []
            self.low_card_cols = []
            
            for col in self.categorical_cols:
                if X[col].nunique() > 50:
                    self.high_card_cols.append(col)
                else:
                    self.low_card_cols.append(col)
        
        # Simple encoding for high cardinality features (using mean encoding)
        if self.high_card_cols and y is not None:
            if fit:
                for col in self.high_card_cols:
                    # Calculate mean target for each category
                    mean_map = df.groupby(col)[target_col].mean().to_dict()
                    self.target_encoding_maps[col] = mean_map
                    X[col] = X[col].map(mean_map).fillna(y.mean())
            else:
                for col in self.high_card_cols:
                    if col in self.target_encoding_maps:
                        X[col] = X[col].map(self.target_encoding_maps[col]).fillna(0.5)
        
        # One-hot encoding for low cardinality features
        if self.low_card_cols:
            if fit:
                self.low_card_categories = {c: X[c].unique().tolist() for c in self.low_card_cols}
                X = pd.get_dummies(X, columns=self.low_card_cols, drop_first=True, dtype=int)
                self.encoded_columns = X.columns.tolist()
            else:
                for col in self.low_card_cols:
                    if col in X.columns:
                        X[col] = X[col].apply(lambda x: x if x in self.low_card_categories.get(col, []) else 'MISSING')
                X = pd.get_dummies(X, columns=self.low_card_cols, drop_first=True, dtype=int)
                if self.encoded_columns:
                    for col in self.encoded_columns:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[self.encoded_columns]
        
        # Scale numerical features
        if self.numerical_cols:
            if fit:
                self.scaler = RobustScaler()
                X[self.numerical_cols] = self.scaler.fit_transform(X[self.numerical_cols])
            elif self.scaler:
                X[self.numerical_cols] = self.scaler.transform(X[self.numerical_cols])
        
        X = X.fillna(0)
        return X, y

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Evaluation Functions

# COMMAND ----------

def create_feature_importance_plot(model, feature_names, top_n=20):
    """Create and return feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    
    return fig

def create_roc_curve_plot(y_true, y_scores, model_name):
    """Create and return ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    return fig

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Create and return confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    return fig

# COMMAND ----------

def evaluate_model_with_mlflow(model, X_train, y_train, X_test, y_test, 
                              X_temporal, y_temporal, model_name, 
                              hyperparameters=None, parent_run_id=None):
    """Train model and log results to MLflow."""
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name}_training", 
                         nested=True if parent_run_id else False) as run:
        
        # Log tags
        mlflow.set_tags({
            "model_type": model_name,
            "training_date": datetime.now().isoformat(),
            "developer": "data_science_team"
        })
        
        # Log hyperparameters
        if hyperparameters:
            mlflow.log_params(hyperparameters)
        else:
            # Log model parameters
            mlflow.log_params(model.get_params())
        
        # Train model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_test = model.predict(X_test)
        y_pred_temporal = model.predict(X_temporal)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_prob_temporal = model.predict_proba(X_temporal)[:, 1]
        
        # Calculate metrics
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_prob_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_avg_precision': average_precision_score(y_test, y_prob_test)
        }
        
        temporal_metrics = {
            'temporal_accuracy': accuracy_score(y_temporal, y_pred_temporal),
            'temporal_roc_auc': roc_auc_score(y_temporal, y_prob_temporal),
            'temporal_f1': f1_score(y_temporal, y_pred_temporal),
            'temporal_precision': precision_score(y_temporal, y_pred_temporal),
            'temporal_recall': recall_score(y_temporal, y_pred_temporal),
            'temporal_avg_precision': average_precision_score(y_temporal, y_prob_temporal)
        }
        
        # Log metrics
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(temporal_metrics)
        
        # Log model
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Create and log plots
        if hasattr(model, 'feature_importances_'):
            fig_importance = create_feature_importance_plot(model, X_train.columns)
            mlflow.log_figure(fig_importance, "feature_importance.png")
            plt.close(fig_importance)
        
        fig_roc_test = create_roc_curve_plot(y_test, y_prob_test, f"{model_name} - Test Set")
        mlflow.log_figure(fig_roc_test, "roc_curve_test.png")
        plt.close(fig_roc_test)
        
        fig_roc_temporal = create_roc_curve_plot(y_temporal, y_prob_temporal, f"{model_name} - Temporal Set")
        mlflow.log_figure(fig_roc_temporal, "roc_curve_temporal.png")
        plt.close(fig_roc_temporal)
        
        fig_cm_test = create_confusion_matrix_plot(y_test, y_pred_test, f"{model_name} - Test Set")
        mlflow.log_figure(fig_cm_test, "confusion_matrix_test.png")
        plt.close(fig_cm_test)
        
        # Log classification report
        test_report = classification_report(y_test, y_pred_test, output_dict=True)
        mlflow.log_dict(test_report, "classification_report_test.json")
        
        temporal_report = classification_report(y_temporal, y_pred_temporal, output_dict=True)
        mlflow.log_dict(temporal_report, "classification_report_temporal.json")
        
        # Create results summary
        results = {
            'model': model_name,
            'run_id': run.info.run_id,
            **test_metrics,
            **temporal_metrics
        }
        
        print(f"Completed {model_name} - Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
        
        return results, model, run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Training Pipeline

# COMMAND ----------

def run_objective1_training_with_mlflow(config_path=None, use_spark_tables=True):
    """Run complete training pipeline with MLflow tracking.
    
    :param config_path: Path to the project configuration YAML file
    :param use_spark_tables: If True, load from Spark tables; if False, load from CSV files
    """
    
    # Load project configuration - handle both Databricks and local paths
    if config_path is None:
        if is_databricks():
            config_path = "/Workspace/Users/mopeyman93@gmail.com/.bundle/marvelous-databricks-course-peyman-93/dev/files/project_config.yml"
        else:
            config_path = "/Users/mohammadpeyman/Desktop/MLOps/marvelous-databricks-course-peyman-93/project_config.yml"
    
    config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
    
    # Create tags for tracking
    tags_dict = {
        "git_sha": "placeholder_sha",  # You can get actual git sha if needed
        "branch": "main",
        "experiment_name": config.experiment_name_basic,
        "model_name": "complaint_upheld_classifier"
    }
    tags = Tags(**tags_dict)
    
    # Start parent run for the entire training session
    with mlflow.start_run(run_name="complaint_upheld_model_comparison") as parent_run:
        
        # Log dataset information
        mlflow.log_params({
            "dataset": "financial_complaints",
            "objective": "complaint_upheld_prediction",
            "experiment_type": "model_comparison",
            "catalog": config.catalog_name,
            "schema": config.schema_name
        })
        
        # Load data
        print("Loading datasets...")
        
        if use_spark_tables:
            # Load from Databricks tables
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            train_df = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
            test_df = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
            temporal_df = spark.table(f"{config.catalog_name}.{config.schema_name}.temporal_test_set").toPandas()
        else:
            # Load from CSV files (fallback option)
            train_df = pd.read_csv('data/splits/train_set.csv')
            test_df = pd.read_csv('data/splits/test_set.csv')
            temporal_df = pd.read_csv('data/splits/temporal_test_set.csv')
        
        # Log dataset statistics
        mlflow.log_metrics({
            "train_size": len(train_df),
            "test_size": len(test_df),
            "temporal_size": len(temporal_df),
            "train_positive_rate": train_df[config.target].mean(),
            "test_positive_rate": test_df[config.target].mean(),
            "temporal_positive_rate": temporal_df[config.target].mean()
        })
        
        # Create feature configuration
        feature_config = {
            'model_features': config.num_features + config.cat_features,
            'num_features': config.num_features,
            'cat_features': config.cat_features,
            'target': config.target
        }
        
        mlflow.log_dict(feature_config, "feature_config.json")
        
        # Feature engineering
        print("Performing feature engineering...")
        fe = FeatureEngineering(feature_config)
        X_train, y_train = fe.prepare_features(train_df, fit=True)
        X_test, y_test = fe.prepare_features(test_df, fit=False)
        X_temporal, y_temporal = fe.prepare_features(temporal_df, fit=False)
        
        # Align columns
        common_cols = [col for col in X_train.columns if col in X_test.columns]
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        X_temporal = X_temporal[common_cols]
        
        # Log feature statistics
        mlflow.log_metrics({
            "n_features_total": len(common_cols),
            "n_numerical_features": len(fe.numerical_cols) if fe.numerical_cols else 0,
            "n_categorical_features": len(fe.low_card_cols) + len(fe.high_card_cols) if fe.low_card_cols and fe.high_card_cols else 0,
            "n_high_card_features": len(fe.high_card_cols) if fe.high_card_cols else 0,
            "n_low_card_features": len(fe.low_card_cols) if fe.low_card_cols else 0
        })
        
        # Feature selection using RandomForest
        print("Performing feature selection...")
        rf_quick = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_quick.fit(X_train, y_train)
        
        rf_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_quick.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        n_top_features = min(100, len(rf_importance))
        top_features = rf_importance.head(n_top_features)['feature'].tolist()
        
        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]
        X_temporal_selected = X_temporal[top_features]
        
        mlflow.log_metric("n_selected_features", len(top_features))
        mlflow.log_text('\n'.join(top_features), "selected_features.txt")
        
        # Save feature importance
        rf_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        # Define models to train
        models = {
            'Random Forest (Baseline)': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Random Forest (Balanced)': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Random Forest (Tuned)': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        
        # Train and evaluate models
        results_list = []
        trained_models = {}
        run_ids = {}
        
        for name, model in models.items():
            results, trained_model, run_id = evaluate_model_with_mlflow(
                model, X_train_selected, y_train,
                X_test_selected, y_test,
                X_temporal_selected, y_temporal,
                name, parent_run_id=parent_run.info.run_id
            )
            results_list.append(results)
            trained_models[name] = trained_model
            run_ids[name] = run_id
        
        # Create results comparison
        results_df = pd.DataFrame(results_list).sort_values('test_roc_auc', ascending=False)
        print("\nModel Comparison Results:")
        print(results_df[['model', 'test_roc_auc', 'test_f1', 'temporal_roc_auc']].to_string())
        
        # Log results comparison
        results_df.to_csv("model_comparison.csv", index=False)
        mlflow.log_artifact("model_comparison.csv")
        
        # Select best model
        best_model_name = results_df.iloc[0]['model']
        best_model = trained_models[best_model_name]
        best_run_id = run_ids[best_model_name]
        
        mlflow.log_metrics({
            "best_test_roc_auc": results_df.iloc[0]['test_roc_auc'],
            "best_test_f1": results_df.iloc[0]['test_f1'],
            "best_temporal_roc_auc": results_df.iloc[0]['temporal_roc_auc']
        })
        
        mlflow.set_tag("best_model", best_model_name)
        mlflow.set_tag("best_model_run_id", best_run_id)
        
        # Save best model artifacts
        output_dir = 'models/objective1_complaint_upheld'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        
        with open(os.path.join(output_dir, 'feature_engineering.pkl'), 'wb') as f:
            pickle.dump(fe, f)
        
        with open(os.path.join(output_dir, 'selected_features.pkl'), 'wb') as f:
            pickle.dump(top_features, f)
        
        # Log final artifacts
        mlflow.log_artifacts(output_dir, artifact_path="best_model_artifacts")
        
        # Create final configuration
        final_config = {
            'model_type': best_model_name,
            'mlflow_run_id': parent_run.info.run_id,
            'best_model_run_id': best_run_id,
            'performance_metrics': {
                'test_set': {
                    'roc_auc': float(results_df.iloc[0]['test_roc_auc']),
                    'f1_score': float(results_df.iloc[0]['test_f1']),
                    'precision': float(results_df.iloc[0]['test_precision']),
                    'recall': float(results_df.iloc[0]['test_recall']),
                    'accuracy': float(results_df.iloc[0]['test_accuracy'])
                },
                'temporal_set': {
                    'roc_auc': float(results_df.iloc[0]['temporal_roc_auc']),
                    'f1_score': float(results_df.iloc[0]['temporal_f1']),
                    'precision': float(results_df.iloc[0]['temporal_precision']),
                    'recall': float(results_df.iloc[0]['temporal_recall']),
                    'accuracy': float(results_df.iloc[0]['temporal_accuracy'])
                }
            },
            'training_date': datetime.now().isoformat(),
            'n_features': len(top_features),
            'selected_features': top_features[:20]  # Top 20 for reference
        }
        
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            json.dump(final_config, f, indent=2)
        
        mlflow.log_dict(final_config, "final_model_config.json")
        
        print(f"\n✅ Training complete! Best model: {best_model_name}")
        print(f"   Parent Run ID: {parent_run.info.run_id}")
        print(f"   Best Model Run ID: {best_run_id}")
        
        return final_config, parent_run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Training

# COMMAND ----------

if __name__ == "__main__":
    # No need to pass config_path since the function handles it internally
    # Run the training pipeline
    final_config, parent_run_id = run_objective1_training_with_mlflow(
        config_path=None,  # Will use default paths based on environment
        use_spark_tables=True  # Set to False if you want to use CSV files
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Model: {final_config['model_type']}")
    print(f"Test ROC-AUC: {final_config['performance_metrics']['test_set']['roc_auc']:.4f}")
    print(f"Temporal ROC-AUC: {final_config['performance_metrics']['temporal_set']['roc_auc']:.4f}")
    print(f"\nMLflow Parent Run ID: {parent_run_id}")
    print("\nView results in MLflow UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search and Retrieve Experiments

# COMMAND ----------

# Search for recent runs
# Get runs from the last 24 hours
time_24h_ago = int((datetime.now() - timedelta(hours=24)).timestamp()) * 1000

# Use the experiment name from config
recent_runs = mlflow.search_runs(
    experiment_names=[config.experiment_name_basic],  # Using "/Shared/financial-complaints-basic"
    filter_string=f"start_time>{time_24h_ago} AND tags.objective='complaint_upheld_prediction'",
    order_by=["metrics.test_roc_auc DESC"],
    max_results=10
)

print("Recent Experiment Runs:")
if not recent_runs.empty:
    print(recent_runs[['run_id', 'tags.model_type', 'metrics.test_roc_auc', 'metrics.test_f1', 'start_time']].to_string())
else:
    print("No recent runs found in the last 24 hours")

# COMMAND ----------

# Get best performing model across all runs
best_runs = mlflow.search_runs(
    experiment_names=[config.experiment_name_basic],  # Using "/Shared/financial-complaints-basic"
    filter_string="metrics.test_roc_auc > 0",
    order_by=["metrics.test_roc_auc DESC"],
    max_results=5
)

if not best_runs.empty:
    best_run_id = best_runs.iloc[0]['run_id']
    print(f"\nBest performing model run ID: {best_run_id}")
    print(f"ROC-AUC: {best_runs.iloc[0]['metrics.test_roc_auc']:.4f}")
    
    # Load the best model
    best_model_uri = f"runs:/{best_run_id}/model"
    print(f"\nModel URI for deployment: {best_model_uri}")
else:
    print("\nNo runs found with ROC-AUC metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry (Optional)

# COMMAND ----------

# Register the best model to MLflow Model Registry
def register_best_model(run_id, model_name="complaint_upheld_classifier"):
    """Register the best model to MLflow Model Registry."""
    
    model_uri = f"runs:/{run_id}/model"
    
    # Register model
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "objective": "complaint_upheld_prediction",
            "repository": "end-to-end-mlops-databricks-4/marvelous-databricks-course-peyman-93"
        }
    )
    
    # Transition to staging
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True
    )
    
    print(f"Model registered: {model_name} v{model_details.version}")
    print(f"Stage: Staging")
    
    return model_details

# Uncomment to register the best model
# if not best_runs.empty:
#     model_details = register_best_model(best_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up local artifacts (optional)

# COMMAND ----------

# Clean up temporary files
if os.path.exists("feature_importance.csv"):
    os.remove("feature_importance.csv")
if os.path.exists("model_comparison.csv"):
    os.remove("model_comparison.csv")
    
print("Temporary files cleaned up")