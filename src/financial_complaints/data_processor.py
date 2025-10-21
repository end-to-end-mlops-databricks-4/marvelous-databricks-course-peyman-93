"""Data preprocessing module for Financial Complaints."""

import datetime
import json
import os
import re
import string
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp, col, lit
from sklearn.model_selection import train_test_split

from financial_complaints.config import ProjectConfig

warnings.filterwarnings('ignore')


class DataProcessor:
    """A class for preprocessing and managing Financial Complaints DataFrame operations.

    This class handles data preprocessing, feature engineering, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        """Initialize the DataProcessor with data, configuration, and Spark session.
        
        :param pandas_df: Input DataFrame containing raw financial complaints data
        :param config: Project configuration loaded from YAML
        :param spark: Active SparkSession for saving to Unity Catalog
        """
        self.df = pandas_df
        self.config = config
        self.spark = spark
        self.preprocessing_report = []
        self.feature_metadata = {}
        self.original_shape = pandas_df.shape

    def _log(self, message: str) -> None:
        """Log preprocessing steps for audit trail."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.preprocessing_report.append(log_entry)

    def fix_all_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all numeric types to Spark-compatible formats.
        
        This handles the Arrow type compatibility issues when converting pandas DataFrames
        to Spark DataFrames, particularly for int32 and float32 types.
        
        :param df: Input DataFrame with potentially incompatible numeric types
        :return: DataFrame with Spark-compatible numeric types
        """
        df = df.copy()
        
        # Debug: Log initial types
        self._log(f"Initial data types distribution: {df.dtypes.value_counts().to_dict()}")
        
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            
            # Handle integer types (including int32 which causes issues)
            if 'int' in dtype_str.lower():
                try:
                    # Try to convert to nullable Int64 first
                    df[col] = pd.array(df[col], dtype="Int64")
                except Exception:
                    # If that fails, convert to float64 (always works)
                    df[col] = df[col].astype('float64')
                    self._log(f"  Converted {col} from {dtype_str} to float64")
            
            # Handle float32 (convert to float64 for compatibility)
            elif dtype_str == 'float32':
                df[col] = df[col].astype('float64')
            
            # Handle object columns (convert to string)
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna('').astype(str)
            
            # Handle boolean columns
            elif df[col].dtype == 'bool':
                try:
                    df[col] = df[col].astype('boolean')
                except Exception:
                    df[col] = df[col].astype('float64')
        
        # Verify no int32 columns remain
        remaining_int32 = df.select_dtypes(include=['int32']).columns.tolist()
        if remaining_int32:
            self._log(f"  Converting remaining int32 columns to float64: {remaining_int32}")
            for col in remaining_int32:
                df[col] = df[col].astype('float64')
        
        self._log(f"Final data types distribution: {df.dtypes.value_counts().to_dict()}")
        
        return df

    def preprocess(self) -> None:
        """Execute the complete preprocessing pipeline for financial complaints data.
        
        This method orchestrates all preprocessing steps including:
        - Target variable creation
        - Temporal feature engineering
        - Missing value handling
        - Company-level feature engineering
        - Geographic feature engineering
        - Text feature engineering (if enabled)
        - Interaction feature creation
        """
        self._log("Starting Financial Complaints preprocessing pipeline...")
        
        # Step 1: Create target variables
        self._create_target_variables()
        
        # Step 2: Process temporal features
        self._process_temporal_features()
        
        # Step 3: Handle missing values
        self._handle_missing_values()
        
        # Step 4: Engineer company features
        self._engineer_company_features()
        
        # Step 5: Engineer geographic features
        self._engineer_geographic_features()
        
        # Step 6: Engineer text features (if enabled)
        if self.config.feature_engineering['enable_text_features']:
            self._engineer_text_features()
        
        # Step 7: Engineer temporal patterns
        if self.config.feature_engineering['enable_temporal_patterns']:
            self._engineer_temporal_patterns()
        
        # Step 8: Engineer interaction features
        if self.config.feature_engineering['enable_interaction_features']:
            self._engineer_interaction_features()
        
        # Step 9: Select and order features for modeling
        self._prepare_modeling_features()
        
        # Step 10: Clean column names for Unity Catalog compatibility
        self._clean_column_names()
        
        self._log(f"Preprocessing complete. Shape: {self.original_shape} -> {self.df.shape}")

    def _clean_column_names(self) -> None:
        """Clean column names for Unity Catalog compatibility."""
        self._log("Cleaning column names for Unity Catalog...")
        
        # Replace problematic characters
        self.df.columns = [
            col.replace(' ', '_')
               .replace('?', '')
               .replace('-', '_')
               .replace('/', '_')
               .replace('(', '')
               .replace(')', '')
               .replace('.', '_')
            for col in self.df.columns
        ]
        
        self._log(f"  Column names cleaned")

    def _create_target_variables(self) -> None:
        """Create target variables from Company response to consumer field."""
        self._log("Creating target variables...")
        
        # Define response categories
        upheld_responses = ['Closed with monetary relief', 'Closed with non-monetary relief', 'Closed with relief']
        not_upheld_responses = ['Closed with explanation', 'Closed without relief', 'Closed']
        excluded_responses = ['In progress', 'Untimely response']
        
        # Map responses to binary targets
        def map_complaint_outcome(response):
            if response in upheld_responses:
                return 1
            elif response in not_upheld_responses:
                return 0
            else:
                return np.nan
        
        def map_financial_relief(response):
            if response == 'Closed with monetary relief':
                return 1
            elif response in not_upheld_responses + ['Closed with non-monetary relief', 'Closed with relief']:
                return 0
            else:
                return np.nan
        
        self.df[self.config.target] = self.df['Company response to consumer'].apply(map_complaint_outcome)
        self.df[self.config.secondary_target] = self.df['Company response to consumer'].apply(map_financial_relief)
        
        # Track records with targets
        self.df['has_target'] = self.df[self.config.target].notna()
        
        # Create stratification key for balanced sampling
        self.df['stratification_key'] = (
            self.df[self.config.target].astype(str) + '_' + 
            self.df[self.config.secondary_target].astype(str)
        )
        
        # Log target distributions
        valid_records = self.df['has_target'].sum()
        self._log(f"  Valid records with targets: {valid_records:,} ({valid_records/len(self.df)*100:.1f}%)")
        self._log(f"  Complaint upheld rate: {self.df[self.config.target].mean():.3f}")
        self._log(f"  Financial relief rate: {self.df[self.config.secondary_target].mean():.3f}")

    def _process_temporal_features(self) -> None:
        """Process and engineer temporal features."""
        self._log("Processing temporal features...")
        
        # Convert dates
        date_columns = ['Date received', 'Date sent to company']
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Extract temporal components
        self.df['complaint_year'] = self.df['Date received'].dt.year
        self.df['complaint_month'] = self.df['Date received'].dt.month
        self.df['complaint_day'] = self.df['Date received'].dt.day
        self.df['complaint_day_of_week'] = self.df['Date received'].dt.dayofweek
        self.df['complaint_quarter'] = self.df['Date received'].dt.quarter
        
        # Week of year - handle compatibility
        try:
            self.df['complaint_week_of_year'] = self.df['Date received'].dt.isocalendar().week
        except AttributeError:
            self.df['complaint_week_of_year'] = self.df['Date received'].dt.week
        
        # Processing time
        self.df['processing_days'] = (self.df['Date sent to company'] - self.df['Date received']).dt.days
        self.df.loc[self.df['processing_days'] < 0, 'processing_days'] = 0
        
        # Fill missing with median
        median_processing = self.df['processing_days'].median()
        self.df['processing_days'] = self.df['processing_days'].fillna(median_processing)
        
        # Days since dataset start
        dataset_start = self.df['Date received'].min()
        self.df['days_since_dataset_start'] = (self.df['Date received'] - dataset_start).dt.days
        
        self._log(f"  Created temporal features for date range: {dataset_start.date()} to {self.df['Date received'].max().date()}")

    def _handle_missing_values(self) -> None:
        """Strategic missing value handling."""
        self._log("Handling missing values...")
        
        # Tags
        self.df['Tags'] = self.df['Tags'].fillna('No_tag')
        self.df['has_special_tag'] = (self.df['Tags'] != 'No_tag').astype(int)
        
        # Sub-categories
        self.df['Sub-product'] = self.df['Sub-product'].fillna('Not_specified')
        self.df['Sub-issue'] = self.df['Sub-issue'].fillna('Not_specified')
        
        # Consumer consent
        self.df['Consumer consent provided?'] = self.df['Consumer consent provided?'].fillna('Not_provided')
        
        # Geographic data
        self.df['State'] = self.df['State'].fillna('UNKNOWN')
        
        # Clean ZIP codes
        def clean_zip(zip_code):
            if pd.isna(zip_code) or zip_code == 'nan':
                return 'UNKNOWN'
            zip_str = str(zip_code).strip()
            match = re.match(r'^(\d{5})', zip_str)
            if match:
                return match.group(1)
            return 'UNKNOWN'
        
        self.df['ZIP code'] = self.df['ZIP code'].astype(str)
        self.df['ZIP_clean'] = self.df['ZIP code'].apply(clean_zip)
        self.df['ZIP_3digit'] = self.df['ZIP_clean'].apply(
            lambda x: x[:3] if x != 'UNKNOWN' and len(x) >= 3 else 'UNKNOWN'
        )
        
        # Binary indicators
        self.df['has_narrative'] = self.df['Consumer complaint narrative'].notna().astype(int)
        self.df['has_company_public_response'] = self.df['Company public response'].notna().astype(int)
        self.df['consumer_disputed'] = self.df['Consumer disputed?'].fillna('Not_recorded')
        
        self._log(f"  Handled missing values for {len(self.df.columns)} columns")

    def _engineer_company_features(self) -> None:
        """Engineer company-level aggregate features."""
        self._log("Engineering company features...")
        
        # Use only records with valid targets
        valid_mask = self.df['has_target'] == True
        
        # Calculate company metrics
        company_metrics = self.df[valid_mask].groupby('Company').agg({
            self.config.target: ['count', 'mean', 'std'],
            self.config.secondary_target: ['mean', 'sum'],
            'processing_days': ['mean', 'median', 'std'],
            'has_narrative': 'mean',
            'has_company_public_response': 'mean',
            'Timely response?': lambda x: (x == 'Yes').mean()
        }).round(4)
        
        # Flatten column names
        company_metrics.columns = [
            'company_complaint_count', 'company_success_rate', 'company_success_std',
            'company_relief_rate', 'company_relief_total',
            'company_avg_processing_days', 'company_median_processing_days', 'company_processing_std',
            'company_narrative_rate', 'company_response_rate', 'company_timely_rate'
        ]
        
        # Add reliability score
        company_metrics['company_reliability_score'] = np.minimum(
            company_metrics['company_complaint_count'] / 100, 1.0
        )
        
        # Company size categories - FIXED FOR SMALL DATASETS
        # Use fixed bins that work for any dataset size
        company_metrics['company_size'] = pd.cut(
            company_metrics['company_complaint_count'],
            bins=[0, 1, 5, 20, 100, float('inf')],
            labels=['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large'],
            include_lowest=True
        )
        
        # Performance rankings
        company_metrics['company_success_rank'] = company_metrics['company_success_rate'].rank(
            ascending=False, method='dense'
        )
        company_metrics['company_success_percentile'] = company_metrics['company_success_rate'].rank(
            pct=True, method='average'
        )
        
        # Risk scores
        avg_success_rate = self.df[valid_mask][self.config.target].mean()
        company_metrics['company_risk_score'] = (
            company_metrics['company_success_rate'] - avg_success_rate
        ) * company_metrics['company_reliability_score']
        
        # Merge back
        self.df = self.df.merge(company_metrics, left_on='Company', right_index=True, how='left')
        
        # Fill missing for new companies
        for col in company_metrics.columns:
            if col in self.df.columns and self.df[col].dtype != 'category':
                self.df[col] = self.df[col].fillna(self.df[col].median() if self.df[col].dtype in ['float64', 'int64'] else 'Unknown')
        
        self._log(f"  Created {len(company_metrics.columns)} company features for {len(company_metrics)} companies")

    def _engineer_geographic_features(self) -> None:
        """Engineer geographic features including state and regional aggregates."""
        self._log("Engineering geographic features...")
        
        # US Census regions
        region_mapping = {
            'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
            'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast', 'RI': 'Northeast',
            'VT': 'Northeast',
            'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest',
            'MI': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
            'ND': 'Midwest', 'OH': 'Midwest', 'SD': 'Midwest', 'WI': 'Midwest',
            'AL': 'South', 'AR': 'South', 'DE': 'South', 'FL': 'South', 'GA': 'South',
            'KY': 'South', 'LA': 'South', 'MD': 'South', 'MS': 'South', 'NC': 'South',
            'OK': 'South', 'SC': 'South', 'TN': 'South', 'TX': 'South', 'VA': 'South',
            'WV': 'South', 'DC': 'South',
            'AK': 'West', 'AZ': 'West', 'CA': 'West', 'CO': 'West', 'HI': 'West',
            'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West', 'OR': 'West',
            'UT': 'West', 'WA': 'West', 'WY': 'West'
        }
        
        self.df['region'] = self.df['State'].map(region_mapping).fillna('Other')
        
        # State-level metrics
        valid_mask = self.df['has_target'] == True
        state_metrics = self.df[valid_mask & (self.df['State'] != 'UNKNOWN')].groupby('State').agg({
            self.config.target: ['count', 'mean'],
            self.config.secondary_target: 'mean',
            'processing_days': 'mean',
            'Company': 'nunique'
        }).round(4)
        
        state_metrics.columns = [
            'state_complaint_count', 'state_success_rate',
            'state_relief_rate', 'state_avg_processing_days', 'state_company_diversity'
        ]
        
        # State regulatory score
        state_metrics['state_regulatory_score'] = (
            state_metrics['state_success_rate'] * 0.5 +
            state_metrics['state_relief_rate'] * 0.3 +
            (1 - state_metrics['state_avg_processing_days'] / state_metrics['state_avg_processing_days'].max()) * 0.2
        )
        
        # Merge state features
        self.df = self.df.merge(state_metrics, left_on='State', right_index=True, how='left')
        
        # Fill missing
        for col in state_metrics.columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # ZIP-3 level metrics
        zip3_metrics = self.df[valid_mask & (self.df['ZIP_3digit'] != 'UNKNOWN')].groupby('ZIP_3digit').agg({
            self.config.target: ['count', 'mean'],
            self.config.secondary_target: 'mean'
        }).round(4)
        
        zip3_metrics.columns = ['zip3_complaint_count', 'zip3_success_rate', 'zip3_relief_rate']
        
        # Privacy threshold
        min_zip_samples = self.config.parameters.get('min_samples_for_zip_metrics', 10)
        zip3_metrics = zip3_metrics[zip3_metrics['zip3_complaint_count'] >= min_zip_samples]
        
        # Merge ZIP features
        self.df = self.df.merge(zip3_metrics, left_on='ZIP_3digit', right_index=True, how='left')
        
        # Fill missing ZIP features with regional averages
        for col in ['zip3_success_rate', 'zip3_relief_rate']:
            if col in self.df.columns:
                regional_avg = self.df.groupby('region')[col].transform('mean')
                self.df[col] = self.df[col].fillna(regional_avg)
        
        self._log(f"  Created geographic features for {len(state_metrics)} states and {len(zip3_metrics)} ZIP areas")

    def _engineer_text_features(self) -> None:
        """Engineer text features from complaint narratives (simplified version)."""
        self._log("Engineering text features...")
        
        # Initialize text feature columns
        text_features = [
            'text_length', 'text_word_count', 'text_sentence_count',
            'text_avg_word_length', 'text_reading_ease', 'text_grade_level',
            'text_caps_ratio', 'text_punct_ratio', 'text_question_count',
            'text_exclamation_count', 'text_sentiment_polarity',
            'text_sentiment_subjectivity', 'text_unique_words'
        ]
        
        for feature in text_features:
            self.df[feature] = 0
        
        # Simple text feature extraction for narratives
        narrative_mask = self.df['has_narrative'] == 1
        
        if narrative_mask.sum() > 0:
            for idx in self.df[narrative_mask].index[:1000]:  # Process first 1000 for speed
                text = self.df.loc[idx, 'Consumer complaint narrative']
                if pd.notna(text):
                    text = str(text)
                    self.df.at[idx, 'text_length'] = len(text)
                    self.df.at[idx, 'text_word_count'] = len(text.split())
                    self.df.at[idx, 'text_sentence_count'] = text.count('.') + text.count('!') + text.count('?')
                    self.df.at[idx, 'text_question_count'] = text.count('?')
                    self.df.at[idx, 'text_exclamation_count'] = text.count('!')
                    self.df.at[idx, 'text_caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
                    self.df.at[idx, 'text_punct_ratio'] = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
        
        # Sentiment categories
        self.df['text_sentiment_category'] = pd.cut(
            self.df['text_sentiment_polarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        self._log(f"  Created {len(text_features)} text features")

    def _engineer_temporal_patterns(self) -> None:
        """Engineer temporal patterns and seasonality features."""
        self._log("Engineering temporal patterns...")
        
        # Seasons
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        self.df['season'] = self.df['complaint_month'].apply(get_season)
        
        # Holiday indicators
        def is_near_holiday(month, day):
            holidays = {
                (1, range(1, 8)): 'New_Year',
                (4, range(10, 20)): 'Tax_Season',
                (7, range(1, 8)): 'July_4th',
                (11, range(20, 30)): 'Thanksgiving',
                (12, range(15, 32)): 'Christmas'
            }
            
            for (h_month, h_days), holiday in holidays.items():
                if month == h_month and day in h_days:
                    return holiday
            return 'No_Holiday'
        
        self.df['near_holiday'] = self.df.apply(
            lambda row: is_near_holiday(row['complaint_month'], row['complaint_day']),
            axis=1
        )
        
        # Business day features
        self.df['is_weekend'] = (self.df['complaint_day_of_week'] >= 5).astype(int)
        self.df['is_monthend'] = (self.df['complaint_day'] >= 28).astype(int)
        self.df['is_monthstart'] = (self.df['complaint_day'] <= 3).astype(int)
        
        self._log("  Created temporal pattern features")

    def _engineer_interaction_features(self) -> None:
        """Create interaction features between key variables."""
        self._log("Engineering interaction features...")
        
        # Processing speed categories
        def categorize_processing_speed(days):
            if pd.isna(days):
                return 'Unknown'
            elif days <= 1:
                return 'Immediate'
            elif days <= 7:
                return 'Fast'
            elif days <= 30:
                return 'Standard'
            else:
                return 'Slow'
        
        self.df['processing_speed'] = self.df['processing_days'].apply(categorize_processing_speed)
        self.df['company_size_speed'] = self.df['company_size'].astype(str) + '_' + self.df['processing_speed']
        
        # Narrative complexity
        self.df['narrative_complexity'] = 'No_Narrative'
        has_narrative_mask = self.df['has_narrative'] == 1
        
        if has_narrative_mask.sum() > 0:
            # Simple complexity based on word count
            word_counts = self.df.loc[has_narrative_mask, 'text_word_count']
            if word_counts.sum() > 0:
                try:
                    complexity_bins = pd.qcut(
                        word_counts[word_counts > 0],
                        q=[0, 0.25, 0.5, 0.75, 1.0],
                        labels=['Simple', 'Basic', 'Moderate', 'Complex'],
                        duplicates='drop'
                    )
                    self.df.loc[word_counts.index[word_counts > 0], 'narrative_complexity'] = complexity_bins
                except:
                    pass
        
        self._log("  Created interaction features")

    def _prepare_modeling_features(self) -> None:
        """Prepare and select features for modeling."""
        self._log("Preparing modeling features...")
        
        # Extract features based on config
        all_features = self.config.num_features + self.config.cat_features + self.config.binary_features
        
        # Add any additional features created during preprocessing
        available_columns = self.df.columns.tolist()
        
        # Keep only configured features that exist in the dataframe
        modeling_features = [col for col in all_features if col in available_columns]
        
        # Add targets and metadata columns
        essential_columns = [
            'Complaint ID', 
            self.config.target, 
            self.config.secondary_target,
            'has_target',
            'stratification_key',
            'Date received',
            'Date sent to company',
            'Company response to consumer'
        ]
        
        # Select final columns
        final_columns = essential_columns + modeling_features
        final_columns = [col for col in final_columns if col in available_columns]
        
        # Reorder dataframe
        self.df = self.df[final_columns]
        
        # Store feature metadata
        self.feature_metadata['modeling_features'] = modeling_features
        self.feature_metadata['num_features'] = [f for f in modeling_features if f in self.config.num_features]
        self.feature_metadata['cat_features'] = [f for f in modeling_features if f in self.config.cat_features]
        self.feature_metadata['binary_features'] = [f for f in modeling_features if f in self.config.binary_features]
        
        self._log(f"  Selected {len(modeling_features)} features for modeling")

    def split_data(self, test_size: float = None, random_state: int = None, 
                  fix_types_for_spark: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame into training, test, and temporal test sets.
        
        :param test_size: The proportion of the dataset to include in the test split
        :param random_state: Controls the shuffling applied to the data before applying the split
        :param fix_types_for_spark: Whether to fix numeric types for Spark compatibility
        :return: A tuple containing training, test, and temporal test DataFrames
        """
        self._log("\nCreating train/test splits...")
        
        # Use config parameters if not provided
        if test_size is None:
            test_size = self.config.parameters['test_size']
        if random_state is None:
            random_state = self.config.parameters['random_state']
        
        # Separate resolved and in-progress complaints
        resolved_df = self.df[self.df['has_target'] == True].copy()
        in_progress_df = self.df[self.df['has_target'] == False].copy()
        
        self._log(f"  Resolved complaints: {len(resolved_df):,}")
        self._log(f"  In-progress complaints: {len(in_progress_df):,}")
        
        # Sort by date for temporal split - fix column name
        resolved_df = resolved_df.sort_values('Date_received').reset_index(drop=True)
        
        # Create temporal test set (last 10% of resolved data)
        temporal_test_size = self.config.parameters.get('temporal_test_size', 0.1)
        temporal_cutoff_idx = int(len(resolved_df) * (1 - temporal_test_size))
        temporal_cutoff_date = resolved_df.iloc[temporal_cutoff_idx]['Date_received']
        
        pre_temporal_df = resolved_df.iloc[:temporal_cutoff_idx]
        temporal_test_df = resolved_df.iloc[temporal_cutoff_idx:]
        
        self._log(f"  Temporal test set: {len(temporal_test_df):,} records (after {temporal_cutoff_date.date()})")
        
        # Create stratified train/test split on pre-temporal data
        # Check if we have enough samples for stratified split
        stratify_col = None
        if self.config.parameters.get('stratify', True):
            # Count samples per class
            class_counts = pre_temporal_df[self.config.target].value_counts()
            min_class_count = class_counts.min()

            # Need at least 2 samples per class for stratified split
            if min_class_count >= 2:
                stratify_col = pre_temporal_df[self.config.target]
                self._log(f"  Using stratified split (min class count: {min_class_count})")
            else:
                self._log(f"  WARNING: Not enough samples for stratified split (min class count: {min_class_count})")
                self._log(f"  Falling back to random split")

        train_df, test_df = train_test_split(
            pre_temporal_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        # Log split statistics
        self._log(f"\nSplit statistics:")
        self._log(f"  Training set: {len(train_df):,} records")
        self._log(f"  Test set: {len(test_df):,} records")
        self._log(f"  Temporal test set: {len(temporal_test_df):,} records")
        
        # Log target distributions
        self._log(f"\nTarget distributions:")
        self._log(f"  Train - {self.config.target}: {train_df[self.config.target].mean():.3f}")
        self._log(f"  Test - {self.config.target}: {test_df[self.config.target].mean():.3f}")
        self._log(f"  Temporal - {self.config.target}: {temporal_test_df[self.config.target].mean():.3f}")
        
        # Store in-progress for future predictions
        self.in_progress_df = in_progress_df
        
        # Fix data types for Spark compatibility if requested
        if fix_types_for_spark:
            self._log("\nFixing data types for Spark compatibility...")
            train_df = self.fix_all_numeric_types(train_df)
            test_df = self.fix_all_numeric_types(test_df)
            temporal_test_df = self.fix_all_numeric_types(temporal_test_df)
            
            if len(in_progress_df) > 0:
                self.in_progress_df = self.fix_all_numeric_types(in_progress_df)
            
            self._log("Data types fixed successfully")
        
        return train_df, test_df, temporal_test_df

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame,
                       temporal_test_set: pd.DataFrame = None) -> None:
        """Save datasets to Unity Catalog with proper table management.

        Creates or replaces training, test, and temporal test tables in the specified
        Unity Catalog schema. Uses PySpark for efficient data transfer.

        :param train_set: Training dataset
        :param test_set: Test dataset
        :param temporal_test_set: Temporal test dataset (optional)
        """
        self._log(f"\nSaving to Unity Catalog: {self.config.catalog_name}.{self.config.schema_name}")

        # Define table names
        train_table = f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        test_table = f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        temporal_test_table = f"{self.config.catalog_name}.{self.config.schema_name}.temporal_test_set"

        try:
            # Save training set
            self._log(f"  Saving training set to {train_table}...")
            train_spark_df = self.spark.createDataFrame(train_set)
            train_spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(train_table)
            self._log(f"    ✓ Training set saved: {len(train_set):,} records")

            # Save test set
            self._log(f"  Saving test set to {test_table}...")
            test_spark_df = self.spark.createDataFrame(test_set)
            test_spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(test_table)
            self._log(f"    ✓ Test set saved: {len(test_set):,} records")

            # Save temporal test set if provided
            if temporal_test_set is not None and len(temporal_test_set) > 0:
                self._log(f"  Saving temporal test set to {temporal_test_table}...")
                temporal_spark_df = self.spark.createDataFrame(temporal_test_set)
                temporal_spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(temporal_test_table)
                self._log(f"    ✓ Temporal test set saved: {len(temporal_test_set):,} records")

            self._log("\n✓ All datasets successfully saved to Unity Catalog")

        except Exception as e:
            self._log(f"\n✗ Error saving to Unity Catalog: {str(e)}")
            raise

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for all tables to track changes over time."""
        self._log("\nEnabling Change Data Feed...")
        self._log("Note: This will be done in the notebook after tables are created.")

    def get_preprocessing_report(self) -> str:
        """Generate a comprehensive preprocessing report.
        
        :return: Formatted preprocessing report as string
        """
        report = []
        report.append("=" * 70)
        report.append("FINANCIAL COMPLAINTS DATA PREPROCESSING REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.datetime.now()}")
        report.append(f"Environment: {self.config.catalog_name}")
        report.append(f"Schema: {self.config.schema_name}")
        report.append("=" * 70)
        report.append("\nPREPROCESSING STEPS:")
        report.extend(self.preprocessing_report)
        report.append("\n" + "=" * 70)
        report.append("FEATURE METADATA:")
        report.append(json.dumps(self.feature_metadata, indent=2, default=str))
        report.append("=" * 70)
        
        return "\n".join(report)


def generate_synthetic_data(reference_df: pd.DataFrame, num_rows: int = 10) -> pd.DataFrame:
    """Generate synthetic financial complaints data based on reference DataFrame schema.

    This mimics new data arrival in production. Generates realistic complaint data
    with random variations based on the reference data distributions.

    :param reference_df: Reference DataFrame to infer schema and distributions
    :param num_rows: Number of synthetic rows to generate
    :return: DataFrame with synthetic complaint records
    """
    np.random.seed(None)  # Random seed for variability

    # Sample companies, products, issues from reference data
    companies = reference_df['Company'].dropna().unique()
    products = reference_df['Product'].dropna().unique()
    states = reference_df['State'].dropna().unique()

    # Generate synthetic records
    synthetic_records = []

    for i in range(num_rows):
        # Random dates (recent complaints)
        days_ago = np.random.randint(1, 365)
        date_received = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
        date_sent = date_received + pd.Timedelta(days=np.random.randint(0, 5))

        # Random complaint outcome
        outcomes = [
            'Closed with monetary relief',
            'Closed with non-monetary relief',
            'Closed with explanation',
            'Closed without relief',
            'In progress'
        ]
        outcome_probs = [0.15, 0.20, 0.35, 0.20, 0.10]

        record = {
            'Complaint ID': f'SYN-{i+1000000}',
            'Date received': date_received.strftime('%Y-%m-%d'),
            'Product': np.random.choice(products),
            'Sub-product': np.random.choice(['Checking account', 'Credit card', 'Mortgage', None]),
            'Issue': np.random.choice(['Billing dispute', 'Unauthorized transaction', 'Incorrect information']),
            'Sub-issue': np.random.choice(['Account opening', 'Late fee', None]),
            'Company': np.random.choice(companies),
            'State': np.random.choice(states),
            'ZIP code': f'{np.random.randint(10000, 99999)}',
            'Submitted via': np.random.choice(['Web', 'Phone', 'Referral', 'Postal mail']),
            'Date sent to company': date_sent.strftime('%Y-%m-%d'),
            'Company response to consumer': np.random.choice(outcomes, p=outcome_probs),
            'Timely response?': np.random.choice(['Yes', 'No'], p=[0.95, 0.05]),
            'Consumer disputed?': np.random.choice(['Yes', 'No', None], p=[0.15, 0.70, 0.15]),
            'Tags': np.random.choice(['Servicemember', 'Older American', None], p=[0.1, 0.1, 0.8]),
            'Consumer consent provided?': np.random.choice(['Consent provided', 'Consent not provided', None]),
            'Company public response': None,
            'Consumer complaint narrative': None
        }

        synthetic_records.append(record)

    synthetic_df = pd.DataFrame(synthetic_records)
    print(f"Generated {num_rows} synthetic complaint records")

    return synthetic_df


def generate_test_data(reference_df: pd.DataFrame, num_rows: int = 10) -> pd.DataFrame:
    """Generate deterministic test data for integration testing.

    Unlike synthetic data, this generates consistent, predictable test cases
    suitable for validation and testing pipelines.

    :param reference_df: Reference DataFrame to infer schema
    :param num_rows: Number of test rows to generate
    :return: DataFrame with test complaint records
    """
    np.random.seed(42)  # Fixed seed for reproducibility

    # Use common, known values for testing
    test_records = []

    for i in range(num_rows):
        # Fixed date pattern for testing
        date_received = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
        date_sent = date_received + pd.Timedelta(days=2)

        # Alternating outcomes for balanced test data
        outcomes = [
            'Closed with monetary relief',
            'Closed with explanation',
            'Closed with non-monetary relief',
            'Closed without relief'
        ]

        record = {
            'Complaint ID': f'TEST-{i+1}',
            'Date received': date_received.strftime('%Y-%m-%d'),
            'Product': 'Credit card' if i % 2 == 0 else 'Checking or savings account',
            'Sub-product': 'General-purpose credit card or charge card' if i % 2 == 0 else 'Checking account',
            'Issue': 'Billing dispute' if i % 2 == 0 else 'Account opening, closing, or management',
            'Sub-issue': 'Credit card company isn\'t resolving a dispute' if i % 2 == 0 else 'Deposits and withdrawals',
            'Company': 'BANK OF AMERICA, NATIONAL ASSOCIATION' if i % 2 == 0 else 'WELLS FARGO & COMPANY',
            'State': 'CA' if i % 3 == 0 else ('NY' if i % 3 == 1 else 'TX'),
            'ZIP code': '90001' if i % 2 == 0 else '10001',
            'Submitted via': 'Web',
            'Date sent to company': date_sent.strftime('%Y-%m-%d'),
            'Company response to consumer': outcomes[i % len(outcomes)],
            'Timely response?': 'Yes',
            'Consumer disputed?': 'Yes' if i % 2 == 0 else 'No',
            'Tags': 'Servicemember' if i == 0 else None,
            'Consumer consent provided?': 'Consent provided',
            'Company public response': 'Company believes it acted appropriately',
            'Consumer complaint narrative': f'Test complaint narrative {i+1}. This is a test case for validation.'
        }

        test_records.append(record)

    test_df = pd.DataFrame(test_records)
    print(f"Generated {num_rows} deterministic test complaint records")

    return test_df