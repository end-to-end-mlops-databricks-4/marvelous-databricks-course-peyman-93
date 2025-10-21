"""Model monitoring module for Financial Complaints."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

from financial_complaints.config import ProjectConfig


def create_or_refresh_monitoring(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create or refresh a monitoring table for model serving data.

    This function processes the inference data from a Delta table,
    parses the request and response JSON fields, joins with test sets,
    and writes the resulting DataFrame to a Delta table for monitoring purposes.

    :param config: Configuration object containing catalog and schema names.
    :param spark: Spark session used for executing SQL queries and transformations.
    :param workspace: Workspace object used for managing quality monitors.
    """
    # Get the payload table from model serving
    inf_table = spark.sql(
    f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`complaints-model-serving-fe-dev_payload`"
    )

    # Define request schema for financial complaints
    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("Complaint_ID", StringType(), True),
                            StructField("Product", StringType(), True),
                            StructField("Sub_product", StringType(), True),
                            StructField("Issue", StringType(), True),
                            StructField("Sub_issue", StringType(), True),
                            StructField("Company", StringType(), True),
                            StructField("State", StringType(), True),
                            StructField("ZIP_code", StringType(), True),
                            StructField("Submitted_via", StringType(), True),
                            StructField("Date_received", StringType(), True),
                            StructField("Date_sent_to_company", StringType(), True),
                            StructField("Company_response_to_consumer", StringType(), True),
                            StructField("processing_days", IntegerType(), True),
                            StructField("complaint_year", IntegerType(), True),
                            StructField("complaint_month", IntegerType(), True),
                            StructField("complaint_day", IntegerType(), True),
                            StructField("complaint_day_of_week", IntegerType(), True),
                            StructField("complaint_quarter", IntegerType(), True),
                            StructField("complaint_week_of_year", IntegerType(), True),
                            StructField("days_since_dataset_start", IntegerType(), True),
                            StructField("region", StringType(), True),
                            StructField("season", StringType(), True),
                            StructField("near_holiday", StringType(), True),
                            StructField("is_weekend", IntegerType(), True),
                            StructField("is_monthend", IntegerType(), True),
                            StructField("is_monthstart", IntegerType(), True),
                            StructField("processing_speed", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    # Define response schema
    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [
                        StructField("trace", StringType(), True),
                        StructField("databricks_request_id", StringType(), True),
                    ]
                ),
                True,
            ),
        ]
    )

    # Parse request and response JSON
    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))
    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    # Explode the array to get one row per request
    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    # Create final dataframe with key columns
    df_final = df_exploded.withColumn("timestamp_ms", (F.col("request_time").cast("long") * 1000)).select(
        F.col("request_time").alias("timestamp"),
        F.col("timestamp_ms"),
        "databricks_request_id",
        "execution_duration_ms",
        F.col("record.Complaint_ID").alias("Complaint_ID"),
        F.col("record.Product").alias("Product"),
        F.col("record.Company").alias("Company"),
        F.col("record.State").alias("State"),
        F.col("record.processing_days").alias("processing_days"),
        F.col("record.complaint_year").alias("complaint_year"),
        F.col("record.complaint_month").alias("complaint_month"),
        F.col("record.region").alias("region"),
        F.col("record.season").alias("season"),
        F.col("record.is_weekend").alias("is_weekend"),
        F.col("record.processing_speed").alias("processing_speed"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("complaints-model-fe").alias("model_name"),
    )

    # Join with test and temporal test sets to get ground truth labels
    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    temporal_test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.temporal_test_set")

    # Join with both test sets to get complaint_upheld labels
    df_final_with_labels = (
        df_final.join(
            test_set.select("Complaint_ID", "complaint_upheld"), on="Complaint_ID", how="left"
        )
        .withColumnRenamed("complaint_upheld", "complaint_upheld_test")
        .join(
            temporal_test_set.select("Complaint_ID", "complaint_upheld"), on="Complaint_ID", how="left"
        )
        .withColumnRenamed("complaint_upheld", "complaint_upheld_temporal")
        .select(
            "*",
            F.coalesce(
                F.col("complaint_upheld_test"), F.col("complaint_upheld_temporal")
            ).alias("complaint_upheld"),
        )
        .drop("complaint_upheld_test", "complaint_upheld_temporal")
        .withColumn("complaint_upheld", F.col("complaint_upheld").cast("double"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["complaint_upheld", "prediction"])
    )

    # Join with company features for additional context
    company_features = spark.table(f"{config.catalog_name}.{config.schema_name}.company_features")
    state_features = spark.table(f"{config.catalog_name}.{config.schema_name}.state_features")

    df_final_with_features = (
        df_final_with_labels.join(
            company_features.select(
                "Company",
                "company_complaint_count",
                "company_avg_processing_days",
                "company_reliability_score",
            ),
            on="Company",
            how="left",
        ).join(
            state_features.select("State", "state_complaint_count", "state_regulatory_score"),
            on="State",
            how="left",
        )
    )

    # Cast feature columns to double
    df_final_with_features = (
        df_final_with_features.withColumn(
            "company_avg_processing_days", F.col("company_avg_processing_days").cast("double")
        )
        .withColumn("company_reliability_score", F.col("company_reliability_score").cast("double"))
        .withColumn("state_regulatory_score", F.col("state_regulatory_score").cast("double"))
    )

    # Write to monitoring table
    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    # Create or refresh the quality monitor
    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exists, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table created.")


def create_monitoring_table(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create a new monitoring table for model monitoring.

    This function sets up a monitoring table using the provided configuration,
    SparkSession, and workspace. It also enables Change Data Feed for the table.

    :param config: Configuration object containing catalog and schema names
    :param spark: SparkSession object for executing SQL commands
    :param workspace: Workspace object for creating quality monitors
    """
    logger.info("Creating new monitoring table...")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="complaint_upheld",
        ),
    )

    # Important to enable Change Data Feed for monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
