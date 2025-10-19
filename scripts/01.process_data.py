"""Financial Complaints Data Processing Script.

This script processes financial complaints data through a complete ML pipeline:
1. Loads raw data from Unity Catalog volumes
2. Generates synthetic or test data (simulating new data arrival)
3. Preprocesses and engineers features
4. Splits data into train/test/temporal sets
5. Saves processed datasets to Unity Catalog tables

Usage:
    python 01.process_data.py --root_path /path/to/project --env dev --is_test 0
"""

import argparse

import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from financial_complaints.config import ProjectConfig
from financial_complaints.data_processor import (
    DataProcessor,
    generate_synthetic_data,
    generate_test_data,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process financial complaints data for ML pipeline"
    )
    parser.add_argument(
        "--root_path",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Root path of the project (e.g., /Workspace/Repos/...)",
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
        help="Whether to generate test data (1) or synthetic data (0)",
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
    logger.info("FINANCIAL COMPLAINTS DATA PROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Root path: {root_path}")
    logger.info(f"Environment: {env}")
    logger.info(f"Test mode: {'Yes' if is_test == 1 else 'No'}")
    logger.info("=" * 70)

    # Load configuration
    config_path = f"{root_path}/files/project_config.yml"
    logger.info(f"Loading configuration from: {config_path}")
    config = ProjectConfig.from_yaml(config_path=config_path, env=env)

    logger.info("\nConfiguration loaded:")
    logger.info(yaml.dump(config.model_dump(), default_flow_style=False))

    # Initialize Spark session
    logger.info("\nInitializing Spark session...")
    spark = SparkSession.builder.getOrCreate()
    logger.info("Spark session created successfully")

    # Load base complaints data from workspace files
    data_path = f"{root_path}/files/data/Consumer_Complaints_Sample.csv"
    logger.info(f"\nLoading base complaints data from: {data_path}")

    # Use pandas to read CSV with nrows limit to avoid parsing errors
    df = pd.read_csv(data_path, nrows=30000, low_memory=False)
    logger.info(f"Base data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Generate synthetic or test data
    if is_test == 0:
        # Generate synthetic data
        # This mimics new data arrival in production
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING SYNTHETIC DATA (Production Simulation)")
        logger.info("=" * 70)
        new_data = generate_synthetic_data(df, num_rows=1000)
        logger.info("Synthetic data generated successfully")
    else:
        # Generate test data
        # This is for integration testing with predictable results
        # Use more rows to ensure enough samples for stratified splitting
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING TEST DATA (Integration Testing)")
        logger.info("=" * 70)
        new_data = generate_test_data(df, num_rows=100)
        logger.info("Test data generated successfully")

    # Initialize DataProcessor
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING DATA PROCESSOR")
    logger.info("=" * 70)
    data_processor = DataProcessor(new_data, config, spark)
    logger.info("DataProcessor initialized")

    # Preprocess the data
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING DATA")
    logger.info("=" * 70)
    data_processor.preprocess()

    # Split the data
    logger.info("\n" + "=" * 70)
    logger.info("SPLITTING DATA")
    logger.info("=" * 70)
    train_df, test_df, temporal_test_df = data_processor.split_data()

    logger.info("\nDataset shapes:")
    logger.info(f"  Training set: {train_df.shape}")
    logger.info(f"  Test set: {test_df.shape}")
    logger.info(f"  Temporal test set: {temporal_test_df.shape}")

    # Save to Unity Catalog
    logger.info("\n" + "=" * 70)
    logger.info("SAVING TO UNITY CATALOG")
    logger.info("=" * 70)
    data_processor.save_to_catalog(train_df, test_df, temporal_test_df)

    # Generate and display preprocessing report
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING REPORT")
    logger.info("=" * 70)
    report = data_processor.get_preprocessing_report()
    logger.info(report)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Training data saved to: {config.catalog_name}.{config.schema_name}.train_set")
    logger.info(f"Test data saved to: {config.catalog_name}.{config.schema_name}.test_set")
    logger.info(
        f"Temporal test data saved to: {config.catalog_name}.{config.schema_name}.temporal_test_set"
    )


if __name__ == "__main__":
    main()
