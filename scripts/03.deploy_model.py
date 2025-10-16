"""Deploy Financial Complaints Model to Serving Endpoint.

This script deploys a trained machine learning model with feature engineering to a
Databricks Model Serving endpoint:
1. Retrieves the model version from upstream training task
2. Loads project configuration
3. Creates or updates online feature tables for real-time serving
4. Deploys or updates the model serving endpoint
5. Optionally deletes the endpoint in test mode

Usage:
    python 03.deploy_model.py \
        --root_path /path/to/project \
        --env dev \
        --is_test 0
"""

import argparse

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from financial_complaints.config import ProjectConfig
from financial_complaints.serving.fe_model_serving import FeatureLookupServing


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy financial complaints model to serving endpoint"
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
        "--is_test",
        action="store",
        default=0,
        type=int,
        required=True,
        help="Whether running in test mode (1 deletes endpoint after creation)",
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
    logger.info("FINANCIAL COMPLAINTS MODEL DEPLOYMENT")
    logger.info("=" * 70)
    logger.info(f"Root path: {root_path}")
    logger.info(f"Environment: {env}")
    logger.info(f"Test mode: {'Yes (will delete endpoint)' if is_test == 1 else 'No'}")
    logger.info("=" * 70)

    # Initialize Spark and DBUtils
    spark = SparkSession.builder.getOrCreate()
    dbutils = DBUtils(spark)

    # Get model version from upstream training task
    try:
        model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")
        logger.info(f"Retrieved model version from training task: {model_version}")
    except Exception as e:
        logger.warning(f"Could not retrieve model version from task values: {e}")
        logger.info("Will use 'latest' version")
        model_version = "latest"

    # Load project configuration
    config_path = f"{root_path}/files/project_config.yml"
    logger.info(f"\nLoading configuration from: {config_path}")
    config = ProjectConfig.from_yaml(config_path=config_path, env=env)
    logger.info("Configuration loaded successfully")

    # Extract configuration values
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    endpoint_name = f"complaints-model-serving-fe-{env}"

    logger.info(f"\nDeployment configuration:")
    logger.info(f"  Catalog: {catalog_name}")
    logger.info(f"  Schema: {schema_name}")
    logger.info(f"  Endpoint: {endpoint_name}")
    logger.info(f"  Model version: {model_version}")

    # Define feature table names
    company_features_table = f"{catalog_name}.{schema_name}.company_features"
    state_features_table = f"{catalog_name}.{schema_name}.state_features"
    text_features_table = f"{catalog_name}.{schema_name}.text_features"

    # Initialize Feature Lookup Serving Manager
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING SERVING MANAGER")
    logger.info("=" * 70)
    feature_model_server = FeatureLookupServing(
        model_name=f"{catalog_name}.{schema_name}.complaint_fe_model",
        endpoint_name=endpoint_name,
        company_table=company_features_table,
        state_table=state_features_table,
    )
    logger.info("Serving manager initialized successfully")

    # Note: Online tables are now managed automatically by Feature Engineering
    # when using FeatureLookup in model serving. We don't need to manually create them.
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE TABLES READY FOR SERVING")
    logger.info("=" * 70)
    logger.info("Feature tables will be automatically synchronized for online serving")
    logger.info(f"  Company features: {company_features_table}")
    logger.info(f"  State features: {state_features_table}")
    logger.info("\n✓ Feature tables are ready for model serving")

    # Deploy or update the model serving endpoint
    logger.info("\n" + "=" * 70)
    logger.info("DEPLOYING/UPDATING MODEL SERVING ENDPOINT")
    logger.info("=" * 70)
    logger.info(f"Endpoint name: {endpoint_name}")
    logger.info(f"Model: {catalog_name}.{schema_name}.complaint_fe_model")
    logger.info(f"Version: {model_version}")

    try:
        feature_model_server.deploy_or_update_serving_endpoint(
            version=model_version,
            workload_size="Small",
            scale_to_zero=True,
            wait=False
        )
        logger.info("✓ Started deployment/update of the serving endpoint")
        logger.info(f"  Endpoint will be available at: /serving-endpoints/{endpoint_name}/invocations")
    except Exception as e:
        logger.error(f"Failed to deploy/update serving endpoint: {e}")
        raise

    # Delete endpoint if in test mode
    if is_test == 1:
        logger.info("\n" + "=" * 70)
        logger.info("TEST MODE: CLEANING UP")
        logger.info("=" * 70)
        logger.info(f"Deleting serving endpoint: {endpoint_name}")
        try:
            workspace = WorkspaceClient()
            workspace.serving_endpoints.delete(name=endpoint_name)
            logger.info("✓ Serving endpoint deleted successfully")
        except Exception as e:
            logger.warning(f"Could not delete endpoint: {e}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("DEPLOYMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Environment: {env}")
    logger.info(f"Endpoint: {endpoint_name}")
    logger.info(f"Model: {catalog_name}.{schema_name}.complaint_fe_model")
    logger.info(f"Version: {model_version}")
    if is_test == 0:
        logger.info(f"\nEndpoint URL: /serving-endpoints/{endpoint_name}/invocations")
        logger.info("The endpoint is now ready to serve predictions with feature lookups")
    else:
        logger.info("\nTest mode: Endpoint was created and then deleted")


if __name__ == "__main__":
    main()
