"""Train and Register Financial Complaints Feature Engineering Model.

This script trains a machine learning model with feature engineering capabilities:
1. Creates feature tables (company, state, text features)
2. Defines feature functions for dynamic feature generation
3. Loads data from Unity Catalog
4. Creates training set with feature lookups and functions
5. Trains model and evaluates performance
6. Registers model to MLflow if performance improves

Usage:
    python 02.train_register_fe_model.py \
        --root_path /path/to/project \
        --env dev \
        --git_sha abc123 \
        --job_run_id 456 \
        --branch main \
        --is_test 0
"""

import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from financial_complaints.config import ProjectConfig, Tags
from financial_complaints.models.financial_complaints_feature_lookup_model import (
    FinancialComplaintsFeatureLookupModel,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and register financial complaints model with feature engineering"
    )
    parser.add_argument(
        "--root_path",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Root path of the project",
    )
    parser.add_argument(
        "--env",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Environment: dev, acc, or prd",
    )
    parser.add_argument(
        "--git_sha",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Git commit SHA",
    )
    parser.add_argument(
        "--job_run_id",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Databricks job run ID",
    )
    parser.add_argument(
        "--branch",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Git branch name",
    )
    parser.add_argument(
        "--is_test",
        action="store",
        default=0,
        type=int,
        required=True,
        help="Whether running in test mode (1) or production mode (0)",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    root_path = args.root_path
    env = args.env
    is_test = args.is_test

    logger.info("=" * 70)
    logger.info("FINANCIAL COMPLAINTS MODEL TRAINING WITH FEATURE ENGINEERING")
    logger.info("=" * 70)
    logger.info(f"Root path: {root_path}")
    logger.info(f"Environment: {env}")
    logger.info(f"Git SHA: {args.git_sha}")
    logger.info(f"Branch: {args.branch}")
    logger.info(f"Job Run ID: {args.job_run_id}")
    logger.info(f"Test mode: {'Yes' if is_test == 1 else 'No'}")
    logger.info("=" * 70)

    # Load configuration
    config_path = f"{root_path}/files/project_config.yml"
    logger.info(f"Loading configuration from: {config_path}")
    config = ProjectConfig.from_yaml(config_path=config_path, env=env)

    # Initialize Spark and DBUtils
    spark = SparkSession.builder.getOrCreate()
    dbutils = DBUtils(spark)

    # Create tags for tracking
    tags_dict = {
        "git_sha": args.git_sha,
        "branch": args.branch,
        "experiment_name": config.experiment_name_fe,
        "model_name": "financial_complaints_fe_model",
    }
    tags = Tags(**tags_dict)

    logger.info("\nConfiguration loaded successfully")
    logger.info(f"Catalog: {config.catalog_name}")
    logger.info(f"Schema: {config.schema_name}")
    logger.info(f"Target: {config.target}")
    logger.info(f"Experiment: {config.experiment_name_fe}")

    # Initialize feature engineering model
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING FEATURE ENGINEERING MODEL")
    logger.info("=" * 70)
    fe_model = FinancialComplaintsFeatureLookupModel(config=config, tags=tags, spark=spark)
    logger.info("Model initialized successfully")

    # Create feature tables (ONLY RUN ONCE - Comment out after initial creation)
    # Uncomment these lines if you need to create feature tables from scratch
    # logger.info("\n" + "=" * 70)
    # logger.info("CREATING FEATURE TABLES")
    # logger.info("=" * 70)
    #
    # logger.info("Creating company features table...")
    # fe_model.create_company_features_table()
    #
    # logger.info("Creating state features table...")
    # fe_model.create_state_features_table()
    #
    # logger.info("Creating text features table...")
    # fe_model.create_text_features_table()
    #
    # logger.info("Feature tables created successfully")

    # Update feature tables with latest data (Run this on every training)
    logger.info("\n" + "=" * 70)
    logger.info("UPDATING FEATURE TABLES")
    logger.info("=" * 70)
    fe_model.update_feature_tables()
    logger.info("Feature tables updated successfully")

    # Define feature functions
    logger.info("\n" + "=" * 70)
    logger.info("DEFINING FEATURE FUNCTIONS")
    logger.info("=" * 70)
    fe_model.define_feature_functions()
    logger.info("Feature functions defined successfully")

    # Load data
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    fe_model.load_data()
    logger.info("Data loaded successfully")

    # Create training set with feature lookups
    logger.info("\n" + "=" * 70)
    logger.info("CREATING TRAINING SET WITH FEATURE LOOKUPS")
    logger.info("=" * 70)
    fe_model.create_training_set()
    logger.info("Training set created successfully")

    # Train the model
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODEL")
    logger.info("=" * 70)
    fe_model.train()
    logger.info("Model training completed successfully")

    # Evaluate model on test set
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 70)

    # Load test set from Delta table (limit for faster evaluation)
    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)

    # Drop feature lookup columns that will be looked up from feature tables
    # Note: These columns were already dropped in load_data, but we drop them again for safety
    company_features_to_drop = [
        'company_complaint_count', 'company_avg_processing_days',
        'company_median_processing_days', 'company_processing_std',
        'company_reliability_score'
    ]
    state_features_to_drop = [
        'state_complaint_count', 'state_avg_processing_days',
        'state_company_diversity', 'state_regulatory_score'
    ]

    cols_to_drop = [col for col in company_features_to_drop + state_features_to_drop
                   if col in test_set.columns]

    if cols_to_drop:
        test_set = test_set.drop(*cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} feature columns from test set")

    # Check if model improved (this method compares with latest registered model)
    try:
        model_improved = fe_model.model_improved(test_set=test_set)
        logger.info(f"Model evaluation completed. Model improved: {model_improved}")
    except Exception as e:
        logger.warning(f"Could not compare with existing model: {e}")
        model_improved = True  # Register anyway if no previous model exists
        logger.info("No previous model found, will register current model")

    # When running tests, always register and deploy
    if is_test == 1:
        logger.info("Running in test mode - forcing model registration")
        model_improved = True

    # Register model if improved
    if model_improved:
        logger.info("\n" + "=" * 70)
        logger.info("REGISTERING MODEL")
        logger.info("=" * 70)

        latest_version = fe_model.register_model()
        logger.info(f"Model registered successfully as version: {latest_version}")

        # Set task values for downstream tasks
        dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
        dbutils.jobs.taskValues.set(key="model_updated", value=1)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Model registered: {config.catalog_name}.{config.schema_name}.complaint_fe_model")
        logger.info(f"Model version: {latest_version}")
        logger.info(f"Experiment: {config.experiment_name_fe}")

    else:
        logger.info("\n" + "=" * 70)
        logger.info("MODEL NOT REGISTERED")
        logger.info("=" * 70)
        logger.info("Current model did not improve over existing model")
        logger.info("No new model version was registered")

        # Set task values to indicate no update
        dbutils.jobs.taskValues.set(key="model_updated", value=0)


if __name__ == "__main__":
    main()
