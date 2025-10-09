"""Feature Lookup model implementation for Financial Complaints."""

from datetime import datetime
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from financial_complaints.config import ProjectConfig, Tags


class FinancialComplaintsFeatureLookupModel:
    """A class to manage Feature Lookup Model for Financial Complaints."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.
        
        :param config: Project configuration
        :param tags: Model tags for tracking
        :param spark: SparkSession instance
        """
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.binary_features = self.config.binary_features
        self.high_cardinality_features = self.config.high_cardinality_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function names
        self.company_features_table = f"{self.catalog_name}.{self.schema_name}.company_features"
        self.state_features_table = f"{self.catalog_name}.{self.schema_name}.state_features"
        self.text_features_table = f"{self.catalog_name}.{self.schema_name}.text_features"
        
        # Feature functions
        self.days_since_complaint_func = f"{self.catalog_name}.{self.schema_name}.calculate_days_since_complaint"
        self.response_urgency_func = f"{self.catalog_name}.{self.schema_name}.calculate_response_urgency"
        self.risk_score_func = f"{self.catalog_name}.{self.schema_name}.calculate_risk_score"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.model_dump()  # Using model_dump() instead of dict()
        
        # Model components
        self.model = None
        self.preprocessor = None
        self.training_set = None
        self.run_id = None

    def create_company_features_table(self) -> None:
        """Create or replace company-level feature table."""
        logger.info("ðŸ“Š Creating company features table...")
        
        # First, calculate company features from train and test sets
        # Filter out NULL companies in the query
        company_stats_df = self.spark.sql(f"""
        WITH company_stats AS (
            SELECT 
                Company,
                COUNT(*) as company_complaint_count,
                AVG(processing_days) as company_avg_processing_days,
                PERCENTILE(processing_days, 0.5) as company_median_processing_days,
                STDDEV(processing_days) as company_processing_std,
                AVG(CASE WHEN complaint_upheld = 1 THEN 1 ELSE 0 END) as company_upheld_rate,
                COUNT(DISTINCT Product) as company_product_diversity,
                COUNT(DISTINCT State) as company_state_coverage,
                AVG(CASE WHEN Timely_response = 'Yes' THEN 1 ELSE 0 END) as company_timely_response_rate
            FROM (
                SELECT * FROM {self.catalog_name}.{self.schema_name}.train_set
                UNION ALL
                SELECT * FROM {self.catalog_name}.{self.schema_name}.test_set
            )
            WHERE Company IS NOT NULL  -- Ensure no NULL companies
            GROUP BY Company
        )
        SELECT 
            Company,
            company_complaint_count,
            company_avg_processing_days,
            company_median_processing_days,
            company_processing_std,
            company_upheld_rate,
            company_product_diversity,
            company_state_coverage,
            company_timely_response_rate,
            -- Calculate reliability score
            (company_timely_response_rate * 0.4 + 
             (1 - company_upheld_rate) * 0.3 + 
             CASE 
                WHEN company_avg_processing_days < 30 THEN 0.3
                WHEN company_avg_processing_days < 60 THEN 0.2
                ELSE 0.1
             END) as company_reliability_score,
            current_timestamp() as update_timestamp
        FROM company_stats
        WHERE Company IS NOT NULL  -- Double-check no NULLs
        """)
        
        # Drop the table if it exists to avoid constraint conflicts
        self.spark.sql(f"DROP TABLE IF EXISTS {self.company_features_table}")
        
        # Create the table with NOT NULL constraint on Company
        self.spark.sql(f"""
        CREATE TABLE {self.company_features_table} (
            Company STRING NOT NULL,
            company_complaint_count BIGINT,
            company_avg_processing_days DOUBLE,
            company_median_processing_days DOUBLE,
            company_processing_std DOUBLE,
            company_upheld_rate DOUBLE,
            company_product_diversity BIGINT,
            company_state_coverage BIGINT,
            company_timely_response_rate DOUBLE,
            company_reliability_score DOUBLE,
            update_timestamp TIMESTAMP
        )
        """)
        
        # Add primary key constraint
        self.spark.sql(f"""
        ALTER TABLE {self.company_features_table} 
        ADD CONSTRAINT company_pk PRIMARY KEY(Company)
        """)
        
        # Enable Change Data Feed
        self.spark.sql(f"""
        ALTER TABLE {self.company_features_table} 
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        
        # Insert the data
        company_stats_df.write.mode("append").saveAsTable(self.company_features_table)
        
        logger.info("âœ… Company features table created.")

    def create_state_features_table(self) -> None:
        """Create or replace state-level feature table."""
        logger.info("ðŸ“Š Creating state features table...")
        
        # First create the dataframe with simpler calculation
        state_stats_df = self.spark.sql(f"""
        SELECT 
            State,
            COUNT(*) as state_complaint_count,
            AVG(processing_days) as state_avg_processing_days,
            COUNT(DISTINCT Company) as state_company_diversity,
            AVG(CASE WHEN complaint_upheld = 1 THEN 1.0 ELSE 0.0 END) as state_upheld_rate,
            COUNT(DISTINCT Product) as state_product_diversity,
            AVG(CASE WHEN financial_relief = 1 THEN 1.0 ELSE 0.0 END) as state_relief_rate,
            PERCENTILE(processing_days, 0.5) as state_median_processing_days
        FROM (
            SELECT * FROM {self.catalog_name}.{self.schema_name}.train_set
            UNION ALL
            SELECT * FROM {self.catalog_name}.{self.schema_name}.test_set
        )
        WHERE State IS NOT NULL
        GROUP BY State
        """)
        
        # Calculate regulatory score using DataFrame operations instead of SQL
        state_stats_with_score = state_stats_df.withColumn(
            "state_regulatory_score",
            (
                when(col("state_upheld_rate") > 0.5, lit(0.8))
                .when(col("state_upheld_rate") > 0.3, lit(0.6))
                .otherwise(lit(0.4))
            ) * lit(0.5) + 
            when(col("state_avg_processing_days") < 45, lit(0.5)).otherwise(lit(0.3))
        ).withColumn(
            "update_timestamp", 
            F.current_timestamp()
        )
        
        # Drop and recreate the table
        self.spark.sql(f"DROP TABLE IF EXISTS {self.state_features_table}")
        
        # Write directly as a Delta table - let Spark infer the schema
        state_stats_with_score.write.mode("overwrite").saveAsTable(self.state_features_table)
        
        # Add constraints after table creation
        try:
            # Add NOT NULL constraint on State
            self.spark.sql(f"""
            ALTER TABLE {self.state_features_table} 
            ALTER COLUMN State SET NOT NULL
            """)
            
            # Add primary key
            self.spark.sql(f"""
            ALTER TABLE {self.state_features_table} 
            ADD CONSTRAINT state_pk PRIMARY KEY(State)
            """)
            
            # Enable Change Data Feed
            self.spark.sql(f"""
            ALTER TABLE {self.state_features_table} 
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """)
        except Exception as e:
            logger.warning(f"Could not add constraints: {e}")
        
        logger.info("âœ… State features table created.")

    def create_text_features_table(self) -> None:
        """Create text features table for narrative analysis."""
        logger.info("ðŸ“Š Creating text features table...")
        
        # Check if Consumer_complaint_narrative column exists
        try:
            # Check if the column exists in train_set
            train_columns = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").columns
            
            if 'Consumer_complaint_narrative' not in train_columns:
                logger.warning("âš ï¸ Consumer_complaint_narrative column not found. Skipping text features table.")
                return
            
            # Calculate text features
            text_stats_df = self.spark.sql(f"""
            SELECT 
                Complaint_ID,
                LENGTH(COALESCE(Consumer_complaint_narrative, '')) as text_length,
                SIZE(SPLIT(COALESCE(Consumer_complaint_narrative, ''), ' ')) as text_word_count,
                SIZE(SPLIT(COALESCE(Consumer_complaint_narrative, ''), '.')) as text_sentence_count,
                CASE 
                    WHEN LENGTH(Consumer_complaint_narrative) > 1000 THEN 'long'
                    WHEN LENGTH(Consumer_complaint_narrative) > 500 THEN 'medium'
                    ELSE 'short'
                END as narrative_complexity,
                current_timestamp() as update_timestamp
            FROM (
                SELECT Complaint_ID, Consumer_complaint_narrative 
                FROM {self.catalog_name}.{self.schema_name}.train_set
                UNION ALL
                SELECT Complaint_ID, Consumer_complaint_narrative 
                FROM {self.catalog_name}.{self.schema_name}.test_set
            )
            WHERE Consumer_complaint_narrative IS NOT NULL
            AND Complaint_ID IS NOT NULL  -- Ensure no NULL IDs
            """)
            
            # Drop the table if it exists
            self.spark.sql(f"DROP TABLE IF EXISTS {self.text_features_table}")
            
            # Create the table with NOT NULL constraint on Complaint_ID
            self.spark.sql(f"""
            CREATE TABLE {self.text_features_table} (
                Complaint_ID STRING NOT NULL,
                text_length INT,
                text_word_count INT,
                text_sentence_count INT,
                narrative_complexity STRING,
                update_timestamp TIMESTAMP
            )
            """)
            
            # Add primary key
            self.spark.sql(f"""
            ALTER TABLE {self.text_features_table} 
            ADD CONSTRAINT text_pk PRIMARY KEY(Complaint_ID)
            """)
            
            # Enable Change Data Feed
            self.spark.sql(f"""
            ALTER TABLE {self.text_features_table} 
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """)
            
            # Insert the data
            text_stats_df.write.mode("append").saveAsTable(self.text_features_table)
            
            logger.info("âœ… Text features table created.")
            
        except Exception as e:
            logger.warning(f"âš ï¸ Could not create text features table: {str(e)}")

    def define_feature_functions(self) -> None:
        """Define feature functions for dynamic feature generation."""
        logger.info("ðŸ”§ Defining feature functions...")
        
        # Function 1: Calculate days since complaint (handle TIMESTAMP type)
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.days_since_complaint_func}(complaint_date TIMESTAMP)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime
        if complaint_date:
            # Remove timezone info if present
            if hasattr(complaint_date, 'replace'):
                complaint_date_naive = complaint_date.replace(tzinfo=None)
            else:
                complaint_date_naive = complaint_date
            return (datetime.now() - complaint_date_naive).days
        return 0
        $$
        """)
        
        # Function 2: Calculate response urgency based on processing days (BIGINT type)
        # FIXED: Converted from SQL UDF to Python UDF
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.response_urgency_func}(processing_days BIGINT)
        RETURNS STRING
        LANGUAGE PYTHON AS
        $$
        if processing_days is None:
            return 'unknown'
        
        proc_days = int(processing_days)
        if proc_days < 15:
            return 'high'
        elif proc_days < 30:
            return 'medium'
        elif proc_days < 60:
            return 'low'
        else:
            return 'very_low'
        $$
        """)
        
        # Function 3: Calculate risk score (handle BIGINT types)
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.risk_score_func}(
            processing_days BIGINT, 
            has_narrative BIGINT,
            timely_response STRING
        )
        RETURNS DOUBLE
        LANGUAGE PYTHON AS
        $$
        risk = 0.0
        
        # Processing time component
        if processing_days:
            proc_days = int(processing_days)
            if proc_days > 60:
                risk += 0.4
            elif proc_days > 30:
                risk += 0.2
            else:
                risk += 0.1
        
        # Narrative component
        if has_narrative and int(has_narrative) == 1:
            risk += 0.3
        
        # Timely response component
        if timely_response != 'Yes':
            risk += 0.3
        
        return min(risk, 1.0)
        $$
        """)
        
        logger.info("âœ… Feature functions defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables."""
        logger.info("ðŸ“‚ Loading data from Delta tables...")
        
        # Load base datasets
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        
        # Select features to drop (will be looked up from feature tables)
        company_features_to_drop = [
            'company_complaint_count', 'company_avg_processing_days', 
            'company_median_processing_days', 'company_processing_std',
            'company_reliability_score'
        ]
        
        state_features_to_drop = [
            'state_complaint_count', 'state_avg_processing_days',
            'state_company_diversity', 'state_regulatory_score'
        ]
        
        # Drop features that will be looked up
        existing_cols = self.train_set.columns
        cols_to_drop = [col for col in company_features_to_drop + state_features_to_drop if col in existing_cols]
        
        if cols_to_drop:
            self.train_set = self.train_set.drop(*cols_to_drop)
            
        logger.info(f"âœ… Data loaded - Train: {self.train_set.count()} rows")

    def create_training_set(self) -> None:
        """Create training set with feature lookups and functions."""
        logger.info("ðŸ”„ Creating training set with feature lookups...")
        
        # Define feature lookups
        feature_lookups = []
        
        # Company features lookup
        if 'Company' in self.train_set.columns:
            feature_lookups.append(
                FeatureLookup(
                    table_name=self.company_features_table,
                    feature_names=[
                        'company_complaint_count', 
                        'company_avg_processing_days',
                        'company_median_processing_days',
                        'company_reliability_score'
                    ],
                    lookup_key='Company'
                )
            )
        
        # State features lookup
        if 'State' in self.train_set.columns:
            feature_lookups.append(
                FeatureLookup(
                    table_name=self.state_features_table,
                    feature_names=[
                        'state_complaint_count',
                        'state_avg_processing_days',
                        'state_company_diversity',
                        'state_regulatory_score'
                    ],
                    lookup_key='State'
                )
            )
        
        # Add feature functions
        feature_functions = []
        
        if 'Date_received' in self.train_set.columns:
            feature_functions.append(
                FeatureFunction(
                    udf_name=self.days_since_complaint_func,
                    output_name='days_since_complaint',
                    input_bindings={'complaint_date': 'Date_received'}
                )
            )
        
        if 'processing_days' in self.train_set.columns:
            feature_functions.append(
                FeatureFunction(
                    udf_name=self.response_urgency_func,
                    output_name='response_urgency',
                    input_bindings={'processing_days': 'processing_days'}
                )
            )
        
        # Combine lookups and functions
        all_features = feature_lookups + feature_functions
        
        # Create the training set
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=all_features if all_features else None,
            exclude_columns=['update_timestamp_utc', 'update_timestamp', 'Complaint_ID']
        )
        
        logger.info("âœ… Training set created with feature lookups.")

    def train(self) -> None:
        """Train the model with feature engineering."""
        logger.info("ðŸš€ Starting model training...")
        
        # Load training data with feature lookups
        training_df = self.training_set.load_df().toPandas()
        
        # Prepare test set - remove columns that will be looked up from feature tables
        company_features_to_drop = [
            'company_complaint_count', 'company_avg_processing_days', 
            'company_median_processing_days', 'company_processing_std',
            'company_reliability_score', 'company_upheld_rate',
            'company_product_diversity', 'company_state_coverage',
            'company_timely_response_rate'
        ]
        
        state_features_to_drop = [
            'state_complaint_count', 'state_avg_processing_days',
            'state_company_diversity', 'state_regulatory_score',
            'state_upheld_rate', 'state_product_diversity',
            'state_relief_rate', 'state_median_processing_days'
        ]
        
        # Drop features from test set that will be looked up
        test_existing_cols = self.test_set.columns
        test_cols_to_drop = [col for col in company_features_to_drop + state_features_to_drop 
                            if col in test_existing_cols]
        
        test_set_cleaned = self.test_set
        if test_cols_to_drop:
            test_set_cleaned = self.test_set.drop(*test_cols_to_drop)
            logger.info(f"Dropped {len(test_cols_to_drop)} feature columns from test set: {test_cols_to_drop}")
        
        # Create a test training set to get feature lookups for test data
        test_feature_lookups = []
        
        # Company features lookup
        if 'Company' in test_set_cleaned.columns:
            test_feature_lookups.append(
                FeatureLookup(
                    table_name=self.company_features_table,
                    feature_names=[
                        'company_complaint_count', 
                        'company_avg_processing_days',
                        'company_median_processing_days',
                        'company_reliability_score'
                    ],
                    lookup_key='Company'
                )
            )
        
        # State features lookup
        if 'State' in test_set_cleaned.columns:
            test_feature_lookups.append(
                FeatureLookup(
                    table_name=self.state_features_table,
                    feature_names=[
                        'state_complaint_count',
                        'state_avg_processing_days',
                        'state_company_diversity',
                        'state_regulatory_score'
                    ],
                    lookup_key='State'
                )
            )
        
        # Add feature functions for test set
        test_feature_functions = []
        
        if 'Date_received' in test_set_cleaned.columns:
            test_feature_functions.append(
                FeatureFunction(
                    udf_name=self.days_since_complaint_func,
                    output_name='days_since_complaint',
                    input_bindings={'complaint_date': 'Date_received'}
                )
            )
        
        if 'processing_days' in test_set_cleaned.columns:
            test_feature_functions.append(
                FeatureFunction(
                    udf_name=self.response_urgency_func,
                    output_name='response_urgency',
                    input_bindings={'processing_days': 'processing_days'}
                )
            )
        
        # Combine lookups and functions for test set
        all_test_features = test_feature_lookups + test_feature_functions
        
        # Create test training set with feature lookups
        test_training_set = self.fe.create_training_set(
            df=test_set_cleaned,
            label=self.target,
            feature_lookups=all_test_features if all_test_features else None,
            exclude_columns=['update_timestamp_utc', 'update_timestamp', 'Complaint_ID']
        )
        
        # Load test data with feature lookups
        test_df = test_training_set.load_df().toPandas()
        
        # Prepare features - use columns that exist in training data
        feature_cols = [col for col in training_df.columns 
                    if col != self.target and col not in ['update_timestamp_utc', 'update_timestamp', 'Complaint_ID']]
        
        # Prepare X and y
        X_train = training_df[feature_cols]
        y_train = training_df[self.target]
        
        # Filter test set to only use features that exist in both train and test
        test_feature_cols = [col for col in feature_cols if col in test_df.columns]
        X_test = test_df[test_feature_cols]
        y_test = test_df[self.target]
        
        # Fill missing values
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                X_train[col] = X_train[col].fillna('MISSING')
                if col in X_test.columns:
                    X_test[col] = X_test[col].fillna('MISSING')
            else:
                X_train[col] = X_train[col].fillna(0)
                if col in X_test.columns:
                    X_test[col] = X_test[col].fillna(0)
        
        # Identify numeric and categorical features from actual data
        numeric_features = [col for col in feature_cols 
                        if col in X_train.columns and X_train[col].dtype in ['int64', 'float64']]
        categorical_features = [col for col in feature_cols 
                            if col in X_train.columns and X_train[col].dtype == 'object']
        
        # Handle high cardinality features separately (use only low cardinality for one-hot)
        low_card_features = [col for col in categorical_features 
                            if col not in self.high_cardinality_features]
        
        transformers = []
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        if low_card_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_features))
        
        if transformers:
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop'
            )
        else:
            preprocessor = 'passthrough'
        
        # Create and train pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=self.parameters['random_state'],
                n_jobs=-1
            ))
        ])
        
        # Set MLflow experiment
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            
            # Train the model
            pipeline.fit(X_train, y_train)
            self.model = pipeline
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'f1_score': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }
            
            # Log to MLflow
            mlflow.log_params({
                'model_type': 'RandomForest with Feature Engineering',
                'n_estimators': 200,
                'max_depth': 15,
                'feature_engineering': 'lookup_tables_and_functions'
            })
            mlflow.log_metrics(metrics)
            
            # Create signature
            signature = infer_signature(X_train, y_pred)
            
            # Log model with feature engineering
            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path='complaint_fe_model',
                training_set=self.training_set,
                signature=signature
            )
            
            logger.info(f"âœ… Model trained. Metrics: {metrics}")

    def register_model(self) -> str:
        """Register the trained model to MLflow registry."""
        model_name = f"{self.catalog_name}.{self.schema_name}.complaint_fe_model"
        
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/complaint_fe_model",
            name=model_name,
            tags=self.tags
        )
        
        # Set alias
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias='latest-fe-model',
            version=registered_model.version
        )
        
        logger.info(f"âœ… Model registered as {model_name} version {registered_model.version}")
        return registered_model.version

    def update_feature_tables(self) -> None:
        """Update feature tables with latest data."""
        logger.info("ðŸ”„ Updating feature tables...")
        
        # Update company features using MERGE
        self.spark.sql(f"""
        MERGE INTO {self.company_features_table} target
        USING (
            SELECT 
                Company,
                COUNT(*) as company_complaint_count,
                AVG(processing_days) as company_avg_processing_days,
                PERCENTILE(processing_days, 0.5) as company_median_processing_days,
                STDDEV(processing_days) as company_processing_std,
                AVG(CASE WHEN complaint_upheld = 1 THEN 1.0 ELSE 0.0 END) as company_upheld_rate,
                COUNT(DISTINCT Product) as company_product_diversity,
                COUNT(DISTINCT State) as company_state_coverage,
                AVG(CASE WHEN Timely_response = 'Yes' THEN 1.0 ELSE 0.0 END) as company_timely_response_rate,
                (AVG(CASE WHEN Timely_response = 'Yes' THEN 1.0 ELSE 0.0 END) * 0.4 + 
                 (1 - AVG(CASE WHEN complaint_upheld = 1 THEN 1.0 ELSE 0.0 END)) * 0.3 + 
                 CASE 
                    WHEN AVG(processing_days) < 30 THEN 0.3
                    WHEN AVG(processing_days) < 60 THEN 0.2
                    ELSE 0.1
                 END) as company_reliability_score,
                current_timestamp() as update_timestamp
            FROM {self.catalog_name}.{self.schema_name}.train_set
            WHERE Company IS NOT NULL
            GROUP BY Company
        ) source
        ON target.Company = source.Company
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """)
        
        logger.info("âœ… Feature tables updated.")

    def score_batch(self, df: DataFrame) -> DataFrame:
        """Score a batch of data using the registered model.
        
        :param df: Input DataFrame
        :return: DataFrame with predictions
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.complaint_fe_model@latest-fe-model"
        
        predictions = self.fe.score_batch(
            model_uri=model_uri,
            df=df
        )
        
        return predictions