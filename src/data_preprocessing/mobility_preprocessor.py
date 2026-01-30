"""
TALEA - Civic Digital Twin: Mobility Preprocessing Module
File: src/data_preprocessing/mobility_preprocessor.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Unified interface for preprocessing all mobility datasets as MobilityPreprocessor.
And also include separate BicycleCounterPreprocessor, PedestrianFlowPreprocessor, and TrafficPreprocessor.

This module handles preprocessing for mobility datasets:
- Bicycle Counter Data
- Pedestrian Flow Data
- Traffic Monitor Data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class ColumnNames(Enum):
    """Standardized column names"""
    DATETIME = 'datetime'
    YEAR = 'year'
    MONTH = 'month'
    DAY = 'day'
    HOUR = 'hour'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'


@dataclass
class ProcessingConfig:
    """Configuration for preprocessing parameters"""
    # Missing value thresholds
    LOW_MISSING_THRESHOLD: float = 5.0
    MEDIUM_MISSING_THRESHOLD: float = 20.0
    
    # Outlier detection
    OUTLIER_PERCENTILE: float = 99.9
    
    # Time periods
    TIME_PERIOD_BINS: List[int] = None
    TIME_PERIOD_LABELS: List[str] = None
    
    # Peak hours
    MORNING_PEAK_START: int = 7
    MORNING_PEAK_END: int = 9
    EVENING_PEAK_START: int = 17
    EVENING_PEAK_END: int = 19
    
    # Window parameters (for sequence analysis)
    DEFAULT_WINDOW_LENGTH: int = 10
    MAX_AUTOCORR_LAG: int = 48  # For detecting periodicity
    
    def __post_init__(self):
        if self.TIME_PERIOD_BINS is None:
            self.TIME_PERIOD_BINS = [-1, 6, 10, 16, 20, 24]
        if self.TIME_PERIOD_LABELS is None:
            self.TIME_PERIOD_LABELS = ['night', 'morning_rush', 'midday', 
                                       'evening_rush', 'evening']


class MobilityPreprocessor:
    """
    Unified preprocessor for all mobility datasets.
    
    This class provides a simple interface for preprocessing bicycle, pedestrian,
    and traffic data with one class. It automatically detects the data type
    and applies appropriate preprocessing steps.
    
    Example:
        >>> prep = MobilityPreprocessor()
        >>> bicycle_clean = prep.preprocess_bicycle_data(bicycle_df)
        >>> pedestrian_clean = prep.preprocess_pedestrian_data(pedestrian_df)
        >>> traffic_clean = prep.preprocess_traffic_data(traffic_df)
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the mobility preprocessor
        
        Args:
            config: Optional processing configuration
        """
        self.config = config or ProcessingConfig()
        
        # Initialize specialized preprocessors
        self.bicycle_processor = BicycleCounterPreprocessor(self.config)
        self.pedestrian_processor = PedestrianFlowPreprocessor(self.config)
        self.traffic_processor = TrafficPreprocessor(self.config)
    
    # === BICYCLE DATA PREPROCESSING ===
    
    def preprocess_bicycle_data(self,
                               df: pd.DataFrame,
                               datetime_col: str = 'Data',
                               count_col: Optional[str] = None,
                               aggregate_hourly: bool = True,
                               add_features: bool = True,
                               normalize: bool = False) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for bicycle counter data
        
        Args:
            df: Raw bicycle dataframe
            datetime_col: Name of datetime column
            count_col: Name of count column (if None, uses all count columns)
            aggregate_hourly: Whether to aggregate to hourly frequency
            add_features: Whether to add derived features
            normalize: Whether to normalize count values
            
        Returns:
            Preprocessed bicycle dataframe
        """
        print(f"\n{'='*70}")
        print("PREPROCESSING BICYCLE DATA")
        print(f"{'='*70}\n")
        
        # Step 1: Clean data
        print("Step 1/5: Cleaning data...")
        df_clean = self.bicycle_processor.clean_data(df)
        print(f"  ✓ Cleaned: {len(df_clean):,} records")
        
        # Step 2: Aggregate temporal data
        if aggregate_hourly:
            print("\nStep 2/5: Aggregating to hourly frequency...")
            df_clean = self.bicycle_processor.aggregate_temporal(df_clean, frequency='h')
            print(f"  ✓ Aggregated: {len(df_clean):,} hourly records")
        else:
            print("\nStep 2/5: Skipping aggregation")
        
        # Step 3: Handle missing values
        print("\nStep 3/5: Handling missing values...")
        df_clean = self.bicycle_processor.interpolate_missing_values(df_clean)
        print(f"  ✓ Missing values handled")
        
        # Step 4: Add derived features
        if add_features:
            print("\nStep 4/5: Adding derived features...")
            df_clean = self.bicycle_processor.add_derived_features(df_clean)
            df_clean = self.bicycle_processor.add_temporal_features(df_clean, datetime_col='Data')
            print(f"  ✓ Added features (columns: {len(df_clean.columns)})")
        else:
            print("\nStep 4/5: Skipping feature engineering")
        
        # Step 5: Normalize (optional)
        if normalize and count_col:
            print("\nStep 5/5: Normalizing values...")
            df_clean = self.bicycle_processor.normalize_counts(
                df_clean, 
                count_col, 
                by_time_period=True
            )
            print(f"  ✓ Normalized {count_col}")
        else:
            print("\nStep 5/5: Skipping normalization")
        
        print(f"\n{'='*70}")
        print("✓ BICYCLE DATA PREPROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        return df_clean
    
    # === PEDESTRIAN DATA PREPROCESSING ===
    
    def preprocess_pedestrian_data(self,
                                  df: pd.DataFrame,
                                  datetime_col: str = 'Data',
                                  count_col: str = 'Numero di visitatori',
                                  add_features: bool = True,
                                  aggregate_zones: bool = False) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for pedestrian flow data
        
        Args:
            df: Raw pedestrian dataframe
            datetime_col: Name of datetime column
            count_col: Name of count column
            add_features: Whether to add derived features
            aggregate_zones: Whether to aggregate by zones
            
        Returns:
            Preprocessed pedestrian dataframe
        """
        print(f"\n{'='*70}")
        print("PREPROCESSING PEDESTRIAN DATA")
        print(f"{'='*70}\n")
        
        # Step 1: Clean data
        print("Step 1/4: Cleaning data...")
        df_clean = self.pedestrian_processor.clean_data(df)
        print(f"  ✓ Cleaned: {len(df_clean):,} records")
        
        # Step 2: Add temporal features
        if add_features:
            print("\nStep 2/4: Adding temporal features...")
            df_clean = self.pedestrian_processor.add_temporal_features(df_clean, datetime_col)
            print(f"  ✓ Added features (columns: {len(df_clean.columns)})")
        else:
            print("\nStep 2/4: Skipping feature engineering")
        
        # Step 3: Aggregate by zones (optional)
        if aggregate_zones:
            print("\nStep 3/4: Aggregating by zones...")
            df_clean = self.pedestrian_processor.aggregate_by_zone(df_clean)
            print(f"  ✓ Aggregated: {len(df_clean):,} zone records")
        else:
            print("\nStep 3/4: Skipping zone aggregation")
        
        # Step 4: Normalize counts (optional)
        if count_col in df_clean.columns:
            print(f"\nStep 4/4: Normalizing {count_col}...")
            df_clean = self.pedestrian_processor.normalize_counts(
                df_clean,
                count_col,
                by_time_period=True
            )
            print(f"  ✓ Normalized")
        else:
            print("\nStep 4/4: Skipping normalization (column not found)")
        
        print(f"\n{'='*70}")
        print("✓ PEDESTRIAN DATA PREPROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        return df_clean
    
    # === TRAFFIC DATA PREPROCESSING ===
    
    def preprocess_traffic_data(self,
                               df: pd.DataFrame,
                               datetime_col: str = 'data',
                               add_congestion_index: bool = True,
                               categorize_traffic: bool = True,
                               cap_outliers: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for traffic monitor data
        
        Args:
            df: Raw traffic dataframe
            datetime_col: Name of datetime column
            add_congestion_index: Whether to compute congestion index
            categorize_traffic: Whether to categorize traffic levels
            cap_outliers: Whether to cap extreme outliers
            
        Returns:
            Preprocessed traffic dataframe
        """
        print(f"\n{'='*70}")
        print("PREPROCESSING TRAFFIC DATA")
        print(f"{'='*70}\n")
        
        # Step 1: Clean and reshape data
        print("Step 1/4: Cleaning and reshaping data...")
        df_clean = self.traffic_processor.clean_data(df)
        print(f"  ✓ Cleaned: {len(df_clean):,} records")
        
        # Step 2: Cap outliers
        if cap_outliers:
            print("\nStep 2/4: Capping outliers...")
            df_clean = self.traffic_processor.cap_outliers(df_clean)
        else:
            print("\nStep 2/4: Skipping outlier capping")
        
        # Step 3: Categorize traffic levels
        if categorize_traffic:
            print("\nStep 3/4: Categorizing traffic levels...")
            df_clean = self.traffic_processor.categorize_vehicles(df_clean)
            print(f"  ✓ Traffic categories added")
        else:
            print("\nStep 3/4: Skipping categorization")
        
        # Step 4: Compute congestion index
        if add_congestion_index:
            print("\nStep 4/4: Computing congestion index...")
            df_clean = self.traffic_processor.compute_congestion_index(df_clean)
            print(f"  ✓ Congestion index computed")
        else:
            print("\nStep 4/4: Skipping congestion index")
        
        print(f"\n{'='*70}")
        print("✓ TRAFFIC DATA PREPROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        return df_clean
    
    # === BATCH PREPROCESSING ===
    
    def preprocess_all(self,
                      bicycle_df: Optional[pd.DataFrame] = None,
                      pedestrian_df: Optional[pd.DataFrame] = None,
                      traffic_df: Optional[pd.DataFrame] = None,
                      **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all provided mobility datasets at once
        
        Args:
            bicycle_df: Bicycle counter dataframe
            pedestrian_df: Pedestrian flow dataframe
            traffic_df: Traffic monitor dataframe
            **kwargs: Additional arguments passed to individual preprocessors
            
        Returns:
            Dictionary of preprocessed dataframes
        """
        results = {}
        
        print(f"\n{'='*70}")
        print("BATCH PREPROCESSING ALL MOBILITY DATA")
        print(f"{'='*70}\n")
        
        if bicycle_df is not None:
            results['bicycle'] = self.preprocess_bicycle_data(bicycle_df, **kwargs)
        
        if pedestrian_df is not None:
            results['pedestrian'] = self.preprocess_pedestrian_data(pedestrian_df, **kwargs)
        
        if traffic_df is not None:
            results['traffic'] = self.preprocess_traffic_data(traffic_df, **kwargs)
        
        print(f"\n{'='*70}")
        print(f"✓ BATCH PREPROCESSING COMPLETE ({len(results)} datasets)")
        print(f"{'='*70}\n")
        
        return results
    
    # === UTILITY METHODS ===
    
    def get_data_summary(self, df: pd.DataFrame, name: str = "Dataset") -> None:
        """
        Print a summary of the dataset
        
        Args:
            df: Dataframe to summarize
            name: Name of the dataset
        """
        print(f"\n{name} Summary:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Date range
        for col in ['datetime', 'Data', 'data']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"  Date range: {df[col].min()} to {df[col].max()}")
                    break
                except:
                    pass
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"  Missing values: {missing[missing > 0].to_dict()}")
    
    def export_processed(self,
                        df: pd.DataFrame,
                        output_path: str,
                        format: Literal['csv', 'parquet', 'feather'] = 'csv') -> None:
        """
        Export preprocessed data to file
        
        Args:
            df: Preprocessed dataframe
            output_path: Output file path
            format: Export format
        """
        if format == 'csv':
            df.to_csv(output_path, index=False, sep=';')
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'feather':
            df.to_feather(output_path)
        
        print(f"✓ Exported to {output_path}")


class BicycleCounterPreprocessor:
    """Preprocessor for bicycle counter data"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean bicycle counter data"""
        df = df.copy()
        
        # Convert date column to datetime with UTC timezone handling
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], utc=True)
            df['Data'] = df['Data'].dt.tz_localize(None)
        
        # Clean numeric columns
        count_columns = ['Direzione centro', 'Direzione periferia', 'Totale']
        for col in count_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicates
        key_columns = ['Data', 'Dispositivo conta-bici']
        existing_cols = [col for col in key_columns if col in df.columns]
        if existing_cols:
            df = df.drop_duplicates(subset=existing_cols)
        
        return df
    
    def aggregate_temporal(self, df: pd.DataFrame, frequency: str = 'h') -> pd.DataFrame:
        """Aggregate bicycle counts by time frequency"""
        df = df.copy()
        
        if 'Data' not in df.columns:
            raise ValueError("DataFrame must contain 'Data' column")
        
        df = df.set_index('Data')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        aggregated = df[numeric_cols].resample(frequency).sum()
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            aggregated[col] = df[col].resample(frequency).first()
        
        return aggregated.reset_index()
    
    def interpolate_missing_values(self, df: pd.DataFrame,
                                   location_col: str = 'Dispositivo conta-bici') -> pd.DataFrame:
        """Interpolate missing values in bicycle count data"""
        df = df.copy()
        
        count_columns = ['Direzione centro', 'Direzione periferia', 'Totale']
        existing_counts = [col for col in count_columns if col in df.columns]
        
        for col in existing_counts:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct == 0:
                continue
            
            if location_col in df.columns:
                df[col] = df.groupby(location_col)[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
            else:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for bicycle data"""
        df = df.copy()
        
        # Flow ratio
        if all(col in df.columns for col in ['Direzione centro', 'Direzione periferia']):
            df['flow_ratio'] = np.where(
                df['Direzione periferia'] > 0,
                df['Direzione centro'] / df['Direzione periferia'],
                0
            )
            
            df['net_flow'] = df['Direzione centro'] - df['Direzione periferia']
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'Data') -> pd.DataFrame:
        """Add temporal features"""
        df = df.copy()
        
        if datetime_col in df.columns:
            df['datetime'] = pd.to_datetime(df[datetime_col])
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['hour'] = df['datetime'].dt.hour
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            
            # Time periods
            df['time_period'] = pd.cut(
                df['hour'],
                bins=self.config.TIME_PERIOD_BINS,
                labels=self.config.TIME_PERIOD_LABELS
            )
            
            # Peak hours
            df['is_peak'] = (
                ((df['hour'] >= self.config.MORNING_PEAK_START) & 
                 (df['hour'] < self.config.MORNING_PEAK_END)) |
                ((df['hour'] >= self.config.EVENING_PEAK_START) & 
                 (df['hour'] < self.config.EVENING_PEAK_END))
            ).astype(int)
        
        return df
    
    def normalize_counts(self, df: pd.DataFrame, col: str, 
                        by_time_period: bool = False,
                        method: str = 'minmax') -> pd.DataFrame:
        """Normalize count values"""
        df = df.copy()
        
        if col not in df.columns:
            return df
        
        if by_time_period and 'hour' in df.columns:
            if 'hour' not in df.columns and 'Data' in df.columns:
                df['hour'] = pd.to_datetime(df['Data']).dt.hour
            
            def normalize_group(group):
                values = group[col]
                if method == 'minmax':
                    min_val = values.min()
                    max_val = values.max()
                    if max_val > min_val:
                        return (values - min_val) / (max_val - min_val)
                    return pd.Series(0.5, index=values.index)
                elif method == 'zscore':
                    return (values - values.mean()) / (values.std() + 1e-8)
                return values
            
            df[f'{col}_normalized'] = df.groupby('hour')[col].transform(normalize_group)
        else:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[f'{col}_normalized'] = (df[col] - mean_val) / std_val
        
        return df


class PedestrianFlowPreprocessor:
    """Preprocessor for pedestrian flow data"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean pedestrian flow data"""
        df = df.copy()
        
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], utc=True)
            df['Data'] = df['Data'].dt.tz_localize(None)
        
        if 'Numero di visitatori' in df.columns:
            df['Numero di visitatori'] = pd.to_numeric(
                df['Numero di visitatori'], 
                errors='coerce'
            )
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'Data') -> pd.DataFrame:
        """Add temporal features"""
        df = df.copy()
        
        if datetime_col in df.columns:
            df['datetime'] = pd.to_datetime(df[datetime_col])
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        return df
    
    def aggregate_by_zone(self, df: pd.DataFrame,
                         zone_cols: List[str] = None) -> pd.DataFrame:
        """Aggregate pedestrian flow by zones"""
        df = df.copy()
        
        if zone_cols is None:
            zone_cols = ['Area provenienza', 'Area Arrivo']
        
        existing_cols = [col for col in zone_cols if col in df.columns]
        if not existing_cols:
            return df
        
        if 'Numero di visitatori' in df.columns:
            aggregated = df.groupby(existing_cols).agg({
                'Numero di visitatori': ['sum', 'mean', 'count']
            }).reset_index()
            
            aggregated.columns = [
                '_'.join(col).strip('_') if col[1] else col[0] 
                for col in aggregated.columns.values
            ]
            
            return aggregated
        
        return df
    
    def normalize_counts(self, df: pd.DataFrame, col: str,
                        by_time_period: bool = False,
                        method: str = 'minmax') -> pd.DataFrame:
        """Normalize count values"""
        df = df.copy()
        
        if col not in df.columns:
            return df
        
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f'{col}_normalized'] = 0.5
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[f'{col}_normalized'] = (df[col] - mean_val) / (std_val + 1e-8)
        
        return df


class TrafficPreprocessor:
    """Preprocessor for traffic monitor data"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean traffic monitor data"""
        df = df.copy()
        
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], utc=True)
            df['data'] = df['data'].dt.tz_localize(None)
        
        # Identify and process time slot columns
        time_cols = [col for col in df.columns 
                    if ' - ' in str(col) and any(c.isdigit() for c in str(col))]
        
        if len(time_cols) > 0:
            df = self._reshape_wide_to_long(df, time_cols)
        
        if 'VIA_SPIRA' in df.columns:
            df['VIA_SPIRA'] = df['VIA_SPIRA'].str.strip()
        if 'DIREZIONE' in df.columns:
            df['DIREZIONE'] = df['DIREZIONE'].str.strip()
        
        return df
    
    def _reshape_wide_to_long(self, df: pd.DataFrame, 
                             time_cols: List[str]) -> pd.DataFrame:
        """Convert wide format to long format"""
        
        id_vars = [col for col in df.columns if col not in time_cols + ['tot']]
        
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=time_cols,
            var_name='time_slot',
            value_name='vehicle_count'
        )
        
        df_long['vehicle_count'] = pd.to_numeric(
            df_long['vehicle_count'], 
            errors='coerce'
        ).fillna(0)
        
        df_long['hour'] = df_long['time_slot'].str.extract(r'^(\d+)\.')[0].astype(int)
        df_long['minute'] = df_long['time_slot'].str.extract(r'^(\d+)\.(\d+)')[1].astype(int)
        
        if 'data' in df_long.columns:
            df_long['datetime'] = (
                df_long['data'] + 
                pd.to_timedelta(df_long['hour'], unit='h') +
                pd.to_timedelta(df_long['minute'], unit='m')
            )
        
        return df_long
    
    def categorize_vehicles(self, df: pd.DataFrame,
                           thresholds: Dict[str, int] = None) -> pd.DataFrame:
        """Categorize traffic volume levels"""
        df = df.copy()
        
        if 'vehicle_count' not in df.columns:
            return df
        
        if thresholds is None:
            thresholds = {'low': 50, 'medium': 200, 'high': 500}
        
        def categorize(count):
            if count < thresholds['low']:
                return 'low'
            elif count < thresholds['medium']:
                return 'medium'
            elif count < thresholds['high']:
                return 'high'
            else:
                return 'very_high'
        
        df['traffic_level'] = df['vehicle_count'].apply(categorize)
        
        return df
    
    def compute_congestion_index(self, df: pd.DataFrame,
                                 capacity_col: str = None) -> pd.DataFrame:
        """Compute congestion index for traffic data"""
        df = df.copy()
        
        if 'vehicle_count' not in df.columns:
            return df
        
        if capacity_col and capacity_col in df.columns:
            df['congestion_index'] = df['vehicle_count'] / df[capacity_col]
        else:
            df['congestion_index'] = df['vehicle_count'] / df['vehicle_count'].quantile(0.95)
        
        df['congestion_index'] = df['congestion_index'].clip(0, 1)
        
        return df
    
    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap extreme outliers in traffic data"""
        df = df.copy()
        
        if 'vehicle_count' in df.columns:
            cap_value = df['vehicle_count'].quantile(
                self.config.OUTLIER_PERCENTILE / 100
            )
            outliers = df['vehicle_count'] > cap_value
            
            if outliers.sum() > 0:
                print(f"  ⚠ Capped {outliers.sum()} outlier values at {cap_value:.0f}")
                df.loc[outliers, 'vehicle_count'] = cap_value
        
        return df