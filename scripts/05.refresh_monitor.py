"""Refresh Model Monitoring Table Script.

This script refreshes the monitoring table for the financial complaints model.
It processes inference logs from the model serving endpoint and updates
the quality monitoring metrics.

Usage:
    python 05.refresh_monitor.py \\
        --root_path /path/to/project \\
        --env dev
"""

import argparse

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from loguru import logger

from financial_complaints.config import ProjectConfig
from financial_complaints.monitoring import create_or_refresh_monitoring


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Refresh model monitoring table for financial complaints"
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
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    root_path = args.root_path
    config_path = f"{root_path}/files/project_config.yml"

    logger.info("=" * 70)
    logger.info("FINANCIAL COMPLAINTS MODEL MONITORING REFRESH")
    logger.info("=" * 70)
    logger.info(f"Root path: {root_path}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Config path: {config_path}")
    logger.info("=" * 70)

    # Load configuration
    logger.info("Loading configuration...")
    config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
    logger.info(f"Catalog: {config.catalog_name}")
    logger.info(f"Schema: {config.schema_name}")

    # Initialize Spark and Workspace
    logger.info("\nInitializing Spark session and Workspace client...")
    spark = DatabricksSession.builder.getOrCreate()
    workspace = WorkspaceClient()
    logger.info("Initialization complete")

    # Refresh monitoring
    logger.info("\n" + "=" * 70)
    logger.info("REFRESHING MONITORING TABLE")
    logger.info("=" * 70)
    create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)

    logger.info("\n" + "=" * 70)
    logger.info("MONITORING REFRESH COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
