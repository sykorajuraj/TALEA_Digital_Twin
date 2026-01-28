"""
TALEA - Civic Digital Twin: Mobility Preprocessing Module
File: src/data_preprocessing/mobility_preprocessor.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

This module handles preprocessing for mobility datasets:
- Bicycle Counter Data
- Pedestrian Flow Data
- Traffic Monitor Data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
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


class BicycleCounterPreprocessor:
    """Preprocessor for bicycle counter data"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean bicycle counter data
        
        Args:
            df: Raw bicycle counter dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Convert date column to datetime with UTC timezone handling
        if 'Data' in df.columns:
            # First convert to datetime, handling mixed timezones by converting to UTC
            df['Data'] = pd.to_datetime(df['Data'], utc=True)
            # Then remove timezone info for consistency
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
    
    def aggregate_temporal(self, df: pd.DataFrame, 
                          frequency: str = '1H') -> pd.DataFrame:
        """
        Aggregate bicycle counts by time frequency
        
        Args:
            df: Bicycle counter dataframe
            frequency: Pandas frequency string (e.g., '1H', '1D')
            
        Returns:
            Aggregated dataframe
        """
        df = df.copy()
        
        if 'Data' not in df.columns:
            raise ValueError("DataFrame must contain 'Data' column")
        
        # Set datetime index
        df = df.set_index('Data')
        
        # Aggregate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        aggregated = df[numeric_cols].resample(frequency).sum()
        
        # Add back categorical columns (take first value)
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            aggregated[col] = df[col].resample(frequency).first()
        
        return aggregated.reset_index()
    
    def interpolate_missing_values(self, df: pd.DataFrame,
                                   location_col: str = 'Dispositivo conta-bici'
                                   ) -> pd.DataFrame:
        """
        Interpolate missing values in bicycle count data
        
        Args:
            df: Bicycle counter dataframe
            location_col: Column containing location information
            
        Returns:
            Dataframe with interpolated values
        """
        df = df.copy()
        
        count_columns = ['Direzione centro', 'Direzione periferia', 'Totale']
        existing_counts = [col for col in count_columns if col in df.columns]
        
        for col in existing_counts:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct == 0:
                continue
            
            if location_col in df.columns:
                # Interpolate by location group
                df[col] = df.groupby(location_col)[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
            else:
                # Simple interpolation
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # Fill remaining NaNs with median
            df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def compute_optimal_window_length(self,
                                     df: pd.DataFrame,
                                     value_col: str = 'Totale',
                                     max_lag: Optional[int] = None,
                                     correlation_threshold: float = 0.3) -> int:
        """
        Compute optimal window length based on autocorrelation analysis.
        Based on PDF 6: pick window length where correlation becomes negligible.
        
        Args:
            df: Bicycle counter dataframe
            value_col: Column to analyze
            max_lag: Maximum lag to test (defaults to config)
            correlation_threshold: Correlation threshold for window cutoff
            
        Returns:
            Optimal window length
        """
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found")
        
        if max_lag is None:
            max_lag = self.config.MAX_AUTOCORR_LAG
        
        series = df[value_col].dropna()
        
        if len(series) < max_lag + 1:
            max_lag = len(series) - 1
        
        # Compute autocorrelation
        acf_values = []
        for lag in range(1, max_lag + 1):
            corr = series.iloc[:-lag].corr(series.iloc[lag:])
            acf_values.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        # Find where correlation drops below threshold
        optimal_length = self.config.DEFAULT_WINDOW_LENGTH
        for i, acf in enumerate(acf_values):
            if acf < correlation_threshold:
                optimal_length = i + 1
                break
        
        print(f"  ✓ Optimal window length: {optimal_length} (correlation threshold: {correlation_threshold})")
        
        return optimal_length
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for bicycle data
        
        Args:
            df: Bicycle counter dataframe
            
        Returns:
            Dataframe with additional features
        """
        df = df.copy()
        
        # Flow ratio
        if all(col in df.columns for col in ['Direzione centro', 'Direzione periferia']):
            df['flow_ratio'] = np.where(
                df['Direzione periferia'] > 0,
                df['Direzione centro'] / df['Direzione periferia'],
                np.nan
            )
            
            # Dominant direction
            df['dominant_direction'] = np.where(
                df['Direzione centro'] > df['Direzione periferia'],
                'centro', 'periferia'
            )
        
        # Peak hour indicator
        if 'Data' in df.columns and 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['Data'])
        
        # Peak hour indicator
        hour_col = 'Data' if 'Data' in df.columns else 'datetime'
        if hour_col in df.columns:
            df['hour'] = pd.to_datetime(df[hour_col]).dt.hour
            df['is_peak_hour'] = df['hour'].apply(
                lambda x: 1 if (self.config.MORNING_PEAK_START <= x <= self.config.MORNING_PEAK_END or
                               self.config.EVENING_PEAK_START <= x <= self.config.EVENING_PEAK_END)
                else 0
            )
        
        return df
    
    def add_window_features(self,
                           df: pd.DataFrame,
                           value_col: str = 'Totale',
                           window_lengths: List[int] = None) -> pd.DataFrame:
        """
        Add rolling window features for capturing temporal patterns.
        Based on PDF concepts: window-based features capture short-term dependencies.
        
        Args:
            df: Bicycle counter dataframe
            value_col: Column to compute windows over
            window_lengths: List of window sizes (defaults to [3, 6, 12, 24])
            
        Returns:
            Dataframe with window features
        """
        df = df.copy()
        
        if value_col not in df.columns:
            return df
        
        if window_lengths is None:
            window_lengths = [3, 6, 12, 24]  # Hours for hourly data
        
        for window in window_lengths:
            # Rolling mean
            df[f'{value_col}_ma{window}'] = df[value_col].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Rolling std (volatility)
            df[f'{value_col}_std{window}'] = df[value_col].rolling(
                window=window, min_periods=1
            ).std()
            
            # Rolling max
            df[f'{value_col}_max{window}'] = df[value_col].rolling(
                window=window, min_periods=1
            ).max()
        
        print(f"  ✓ Added window features for {len(window_lengths)} window sizes")
        
        return df


class PedestrianFlowPreprocessor:
    """Preprocessor for pedestrian flow data"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean pedestrian flow data
        
        Args:
            df: Raw pedestrian flow dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Convert date with timezone handling
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], utc=True)
            df['Data'] = df['Data'].dt.tz_localize(None)
        
        # Clean count column
        if 'Numero di visitatori' in df.columns:
            df['Numero di visitatori'] = pd.to_numeric(
                df['Numero di visitatori'], 
                errors='coerce'
            )
        
        # Remove duplicates
        key_columns = ['Data', 'Area provenienza', 'Area Arrivo']
        existing_cols = [col for col in key_columns if col in df.columns]
        if existing_cols:
            df = df.drop_duplicates(subset=existing_cols)
        
        return df
    
    def normalize_flow_values(self, df: pd.DataFrame,
                              method: str = 'minmax',
                              by_time_of_day: bool = False) -> pd.DataFrame:
        """
        Normalize pedestrian flow values.
        Enhanced with time-dependent normalization option (inspired by PDF 7).
        
        Args:
            df: Pedestrian flow dataframe
            method: Normalization method ('minmax' or 'zscore')
            by_time_of_day: If True, normalize separately per time period
            
        Returns:
            Dataframe with normalized values
        """
        df = df.copy()
        
        if 'Numero di visitatori' not in df.columns:
            return df
        
        col = 'Numero di visitatori'
        
        if by_time_of_day and 'Data' in df.columns:
            # Time-dependent normalization
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
            # Global normalization
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[f'{col}_normalized'] = (df[col] - mean_val) / std_val
        
        return df
    
    def aggregate_by_zone(self, df: pd.DataFrame,
                         zone_cols: List[str] = None) -> pd.DataFrame:
        """
        Aggregate pedestrian flow by zones
        
        Args:
            df: Pedestrian flow dataframe
            zone_cols: Columns to group by (default: origin and destination)
            
        Returns:
            Aggregated dataframe
        """
        df = df.copy()
        
        if zone_cols is None:
            zone_cols = ['Area provenienza', 'Area Arrivo']
        
        # Check if columns exist
        existing_cols = [col for col in zone_cols if col in df.columns]
        if not existing_cols:
            return df
        
        # Aggregate by zones
        if 'Numero di visitatori' in df.columns:
            aggregated = df.groupby(existing_cols).agg({
                'Numero di visitatori': ['sum', 'mean', 'count']
            }).reset_index()
            
            # Flatten column names
            aggregated.columns = [
                '_'.join(col).strip('_') if col[1] else col[0] 
                for col in aggregated.columns.values
            ]
            
            return aggregated
        
        return df


class TrafficPreprocessor:
    """Preprocessor for traffic monitor data"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean traffic monitor data
        
        Args:
            df: Raw traffic dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Convert date column
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], utc=True)
            df['data'] = df['data'].dt.tz_localize(None)
        
        # Identify and process time slot columns
        time_cols = [col for col in df.columns 
                    if ' - ' in str(col) and any(c.isdigit() for c in str(col))]
        
        if len(time_cols) > 0:
            df = self._reshape_wide_to_long(df, time_cols)
        
        # Clean location columns
        if 'VIA_SPIRA' in df.columns:
            df['VIA_SPIRA'] = df['VIA_SPIRA'].str.strip()
        if 'DIREZIONE' in df.columns:
            df['DIREZIONE'] = df['DIREZIONE'].str.strip()
        
        return df
    
    def _reshape_wide_to_long(self, df: pd.DataFrame, 
                             time_cols: List[str]) -> pd.DataFrame:
        """Convert wide format to long format"""
        
        # Preserve metadata columns
        id_vars = [col for col in df.columns 
                  if col not in time_cols + ['tot']]
        
        # Melt to long format
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
        
        # Extract time components
        df_long['hour'] = df_long['time_slot'].str.extract(r'^(\d+)\.')[0].astype(int)
        df_long['minute'] = df_long['time_slot'].str.extract(r'^(\d+)\.(\d+)')[1].astype(int)
        
        # Create datetime if date column exists
        if 'data' in df_long.columns:
            df_long['datetime'] = (
                df_long['data'] + 
                pd.to_timedelta(df_long['hour'], unit='h') +
                pd.to_timedelta(df_long['minute'], unit='m')
            )
        
        return df_long
    
    def categorize_vehicles(self, df: pd.DataFrame,
                           thresholds: Dict[str, int] = None) -> pd.DataFrame:
        """
        Categorize traffic volume levels
        
        Args:
            df: Traffic dataframe
            thresholds: Dictionary with categorization thresholds
            
        Returns:
            Dataframe with traffic categories
        """
        df = df.copy()
        
        if 'vehicle_count' not in df.columns:
            return df
        
        if thresholds is None:
            thresholds = {
                'low': 50,
                'medium': 200,
                'high': 500
            }
        
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
        """
        Compute congestion index for traffic data
        
        Args:
            df: Traffic dataframe
            capacity_col: Column containing road capacity (optional)
            
        Returns:
            Dataframe with congestion index
        """
        df = df.copy()
        
        if 'vehicle_count' not in df.columns:
            return df
        
        if capacity_col and capacity_col in df.columns:
            # Congestion = volume / capacity
            df['congestion_index'] = df['vehicle_count'] / df[capacity_col]
        else:
            # Use percentile-based congestion
            df['congestion_index'] = df['vehicle_count'] / df['vehicle_count'].quantile(0.95)
        
        # Clip to [0, 1] range
        df['congestion_index'] = df['congestion_index'].clip(0, 1)
        
        return df
    
    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap extreme outliers in traffic data
        
        Args:
            df: Traffic dataframe
            
        Returns:
            Dataframe with capped outliers
        """
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