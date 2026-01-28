"""
TALEA - Civic Digital Twin: Weather Preprocessing Module
File: src/data_preprocessing/weather_preprocessor.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

This module handles preprocessing for weather datasets:
- Temperature data
- Precipitation data
- Air quality monitoring data
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class ColumnNames(Enum):
    """Standardized column names"""
    DATETIME = 'datetime'
    TEMPERATURE = 'temperature'
    TEMP_MAX = 'temp_max'
    TEMP_MIN = 'temp_min'
    PRECIPITATION = 'precipitation_mm'


@dataclass
class WeatherConfig:
    """Configuration for weather preprocessing"""
    # Temperature thresholds (°C)
    HOT_DAY_TEMP: float = 25.0
    VERY_HOT_DAY_TEMP: float = 30.0
    EXTREME_HEAT_TEMP: float = 35.0
    
    # Precipitation thresholds (mm)
    RAINY_DAY_MM: float = 0.1
    HEAVY_RAIN_MM: float = 10.0
    
    # Rolling window sizes
    PRECIPITATION_ROLLING_DAYS: int = 7
    TEMP_ROLLING_DAYS: int = 3
    
    # Binning configuration
    ADAPTIVE_BINNING: bool = True
    DEFAULT_BIN_SIZE: str = '1H'  # Default temporal bin size
    MIN_SAMPLES_PER_BIN: int = 1  # Minimum samples required per bin


class WeatherPreprocessor:
    """Unified preprocessor for all weather data"""
    
    def __init__(self, config: Optional[WeatherConfig] = None):
        self.config = config or WeatherConfig()
        self._distribution_stats = {}  # Store distribution statistics for correction
    
    def merge_weather_datasets(self,
                              temp_df: Optional[pd.DataFrame] = None,
                              precip_df: Optional[pd.DataFrame] = None,
                              aq_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge multiple weather datasets into unified dataframe
        
        Args:
            temp_df: Temperature dataframe
            precip_df: Precipitation dataframe
            aq_df: Air quality dataframe
            
        Returns:
            Merged weather dataframe
        """
        datasets = []
        
        # Process temperature data
        if temp_df is not None:
            temp_processed = self._process_temperature(temp_df)
            datasets.append(temp_processed)
        
        # Process precipitation data
        if precip_df is not None:
            precip_processed = self._process_precipitation(precip_df)
            datasets.append(precip_processed)
        
        # Process air quality data
        if aq_df is not None:
            aq_processed = self._process_air_quality(aq_df)
            datasets.append(aq_processed)
        
        if not datasets:
            raise ValueError("At least one weather dataset must be provided")
        
        # Merge all datasets
        merged = datasets[0]
        for df in datasets[1:]:
            merged = merged.merge(df, on=ColumnNames.DATETIME.value, how='outer')
        
        # Sort by datetime
        merged = merged.sort_values(ColumnNames.DATETIME.value).reset_index(drop=True)
        
        print(f"✓ Merged weather data: {len(merged):,} records")
        
        return merged
    
    def _process_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process temperature data"""
        df = df.copy()
        
        # Rename columns
        column_mapping = {
            'Data': ColumnNames.DATETIME.value,
            'Temperatura media': ColumnNames.TEMPERATURE.value,
            'Temperatura massima': ColumnNames.TEMP_MAX.value,
            'Temperatura minima': ColumnNames.TEMP_MIN.value
        }
        df = df.rename(columns=column_mapping)
        
        # Convert to datetime with timezone handling
        df[ColumnNames.DATETIME.value] = pd.to_datetime(df[ColumnNames.DATETIME.value], utc=True)
        df[ColumnNames.DATETIME.value] = df[ColumnNames.DATETIME.value].dt.tz_localize(None)
        
        # Convert to numeric
        for col in [ColumnNames.TEMPERATURE.value, ColumnNames.TEMP_MAX.value, 
                   ColumnNames.TEMP_MIN.value]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add derived features
        if ColumnNames.TEMP_MAX.value in df.columns and ColumnNames.TEMP_MIN.value in df.columns:
            df['temp_range'] = df[ColumnNames.TEMP_MAX.value] - df[ColumnNames.TEMP_MIN.value]
        
        if ColumnNames.TEMPERATURE.value in df.columns:
            df['is_hot_day'] = (df[ColumnNames.TEMPERATURE.value] >= 
                               self.config.HOT_DAY_TEMP).astype(int)
            df['is_very_hot_day'] = (df[ColumnNames.TEMPERATURE.value] >= 
                                    self.config.VERY_HOT_DAY_TEMP).astype(int)
            df['is_extreme_heat'] = (df[ColumnNames.TEMPERATURE.value] >= 
                                    self.config.EXTREME_HEAT_TEMP).astype(int)
        
        return df
    
    def _process_precipitation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process precipitation data"""
        df = df.copy()
        
        # Rename columns
        column_mapping = {
            'Data': ColumnNames.DATETIME.value,
            'Precipitazioni (mm)': ColumnNames.PRECIPITATION.value
        }
        df = df.rename(columns=column_mapping)
        
        # Convert to datetime with timezone handling
        df[ColumnNames.DATETIME.value] = pd.to_datetime(df[ColumnNames.DATETIME.value], utc=True)
        df[ColumnNames.DATETIME.value] = df[ColumnNames.DATETIME.value].dt.tz_localize(None)
        
        # Convert to numeric
        df[ColumnNames.PRECIPITATION.value] = pd.to_numeric(
            df[ColumnNames.PRECIPITATION.value], errors='coerce'
        )
        
        # Add derived features
        df['is_rainy_day'] = (df[ColumnNames.PRECIPITATION.value] > 
                             self.config.RAINY_DAY_MM).astype(int)
        df['is_heavy_rain'] = (df[ColumnNames.PRECIPITATION.value] > 
                              self.config.HEAVY_RAIN_MM).astype(int)
        
        # Rolling precipitation
        df = df.sort_values(ColumnNames.DATETIME.value)
        df[f'precipitation_{self.config.PRECIPITATION_ROLLING_DAYS}d'] = (
            df[ColumnNames.PRECIPITATION.value].rolling(
                window=self.config.PRECIPITATION_ROLLING_DAYS, 
                min_periods=1
            ).sum()
        )
        
        return df
    
    def _process_air_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process air quality data"""
        df = df.copy()
        
        # Convert reftime to datetime with timezone handling
        if 'reftime' in df.columns:
            df[ColumnNames.DATETIME.value] = pd.to_datetime(df['reftime'], utc=True)
            df[ColumnNames.DATETIME.value] = df[ColumnNames.DATETIME.value].dt.tz_localize(None)
        
        # Pivot to wide format (one column per pollutant)
        if 'agent atm' in df.columns and 'value' in df.columns:
            # Clean pollutant names
            df['pollutant'] = df['agent atm'].str.extract(r'([A-Z0-9]+)')[0]
            
            # Pivot
            df_pivot = df.pivot_table(
                index=ColumnNames.DATETIME.value,
                columns='pollutant',
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # Flatten column names
            df_pivot.columns.name = None
            
            return df_pivot
        
        return df
    
    def apply_adaptive_binning(self,
                              df: pd.DataFrame,
                              value_columns: List[str],
                              bin_size: Optional[str] = None,
                              aggregation_functions: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Apply adaptive binning to weather data with multiple aggregation functions.
        
        Args:
            df: Weather dataframe with datetime index
            value_columns: Columns to aggregate
            bin_size: Temporal bin size (e.g., '1H', '30min', '1D')
            aggregation_functions: Dict mapping columns to aggregation functions
                                  Default: ['mean', 'std', 'min', 'max'] for numeric
            
        Returns:
            Binned dataframe with aggregated features
        """
        if ColumnNames.DATETIME.value not in df.columns:
            raise ValueError("Dataframe must contain datetime column")
        
        df = df.copy()
        bin_size = bin_size or self.config.DEFAULT_BIN_SIZE
        
        # Ensure datetime is index
        if df.index.name != ColumnNames.DATETIME.value:
            df = df.set_index(ColumnNames.DATETIME.value)
        
        # Define default aggregation functions if not provided
        if aggregation_functions is None:
            aggregation_functions = {}
            for col in value_columns:
                if col in df.columns:
                    # For temperature and continuous metrics: mean, std, min, max
                    if any(x in col.lower() for x in ['temp', 'heat', 'index', 'apparent']):
                        aggregation_functions[col] = ['mean', 'std', 'min', 'max']
                    # For precipitation and counts: sum, mean, max
                    elif any(x in col.lower() for x in ['precip', 'rain', 'count']):
                        aggregation_functions[col] = ['sum', 'mean', 'max']
                    # For binary indicators: max (any occurrence)
                    elif col.startswith('is_'):
                        aggregation_functions[col] = ['max']
                    else:
                        aggregation_functions[col] = ['mean', 'std']
        
        # Apply binning with aggregation
        binned_data = df[value_columns].resample(bin_size).agg(aggregation_functions)
        
        # Flatten multi-level columns
        if isinstance(binned_data.columns, pd.MultiIndex):
            binned_data.columns = ['_'.join(col).strip() for col in binned_data.columns.values]
        
        binned_data = binned_data.reset_index()
        
        print(f"✓ Applied adaptive binning: {len(df)} → {len(binned_data)} samples "
              f"(bin_size={bin_size}, {len(binned_data.columns)-1} features)")
        
        return binned_data
    
    def compute_sample_weights_for_distribution_correction(self,
                                                          df: pd.DataFrame,
                                                          condition_column: str,
                                                          target_distribution: Optional[Dict] = None) -> pd.Series:
        """
        Compute sample weights to correct sampling bias in weather data.
        
        Example use case: Training data from summer months needs reweighting
        for year-round model generalization.
        
        Args:
            df: Weather dataframe
            condition_column: Column to balance (e.g., 'month', 'season', 'hour')
            target_distribution: Desired distribution as dict {value: probability}
                               If None, assumes uniform distribution
            
        Returns:
            Series of sample weights for importance sampling
        """
        if condition_column not in df.columns:
            raise ValueError(f"Column '{condition_column}' not found in dataframe")
        
        # Compute observed distribution (pi in the PDF notation)
        value_counts = df[condition_column].value_counts()
        observed_dist = value_counts / len(df)
        
        # Define target distribution (p*_i in the PDF notation)
        if target_distribution is None:
            # Assume uniform distribution
            unique_values = df[condition_column].unique()
            target_dist = pd.Series(
                1.0 / len(unique_values),
                index=unique_values
            )
        else:
            target_dist = pd.Series(target_distribution)
        
        # Compute weights: w_i = p*_i / p_i
        weights_map = {}
        for value in observed_dist.index:
            if value in target_dist.index:
                weights_map[value] = target_dist[value] / observed_dist[value]
            else:
                weights_map[value] = 1.0  # No adjustment if not in target
        
        # Map weights to each sample
        sample_weights = df[condition_column].map(weights_map)
        
        # Store statistics for reference
        self._distribution_stats[condition_column] = {
            'observed': observed_dist.to_dict(),
            'target': target_dist.to_dict(),
            'weights': weights_map
        }
        
        print(f"✓ Computed sample weights for '{condition_column}' distribution correction")
        print(f"  Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        
        return sample_weights
    
    def analyze_temporal_distribution(self,
                                    df: pd.DataFrame,
                                    value_column: str,
                                    groupby_column: str = 'hour') -> Dict:
        """
        Analyze conditional distributions of weather variables over time.
        
        Args:
            df: Weather dataframe with datetime
            value_column: Column to analyze
            groupby_column: Temporal grouping ('hour', 'weekday', 'month')
            
        Returns:
            Dictionary with distribution statistics
        """
        if ColumnNames.DATETIME.value not in df.columns:
            raise ValueError("Dataframe must contain datetime column")
        
        df = df.copy()
        
        # Extract temporal feature if not present
        if groupby_column not in df.columns:
            dt = pd.to_datetime(df[ColumnNames.DATETIME.value])
            if groupby_column == 'hour':
                df[groupby_column] = dt.dt.hour
            elif groupby_column == 'weekday':
                df[groupby_column] = dt.dt.weekday
            elif groupby_column == 'month':
                df[groupby_column] = dt.dt.month
        
        # Compute statistics per group
        grouped_stats = df.groupby(groupby_column)[value_column].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        
        # Compute coefficient of variation to assess predictive power
        grouped_stats['cv'] = grouped_stats['std'] / grouped_stats['mean']
        
        # Overall statistics
        overall_mean = df[value_column].mean()
        overall_std = df[value_column].std()
        
        # Compute variance ratio (between-group variance / total variance)
        # High ratio indicates strong temporal pattern
        between_var = ((grouped_stats['mean'] - overall_mean) ** 2 * grouped_stats['count']).sum()
        total_var = overall_std ** 2 * len(df)
        variance_ratio = between_var / total_var if total_var > 0 else 0
        
        results = {
            'groupby': groupby_column,
            'value_column': value_column,
            'group_statistics': grouped_stats.to_dict('index'),
            'overall_mean': float(overall_mean),
            'overall_std': float(overall_std),
            'variance_ratio': float(variance_ratio),
            'has_strong_pattern': variance_ratio > 0.1  # Rule of thumb threshold
        }
        
        print(f"✓ Temporal distribution analysis ({groupby_column}):")
        print(f"  Variance ratio: {variance_ratio:.3f} "
              f"({'strong' if variance_ratio > 0.1 else 'weak'} temporal pattern)")
        
        return results
    
    def fit_distribution_to_values(self,
                                   df: pd.DataFrame,
                                   value_column: str,
                                   distributions: Optional[List[str]] = None) -> Dict:
        """
        Fit statistical distributions to weather data.
        
        Args:
            df: Weather dataframe
            value_column: Column to analyze
            distributions: List of distribution names to test
                          Options: 'normal', 'poisson', 'exponential', 'gamma'
            
        Returns:
            Dictionary with best-fit distribution and parameters
        """
        if value_column not in df.columns:
            raise ValueError(f"Column '{value_column}' not found")
        
        data = df[value_column].dropna()
        
        if len(data) < 10:
            print(f"⚠ Insufficient data for distribution fitting ({len(data)} samples)")
            return {}
        
        if distributions is None:
            distributions = ['normal', 'exponential', 'gamma']
        
        results = {}
        
        for dist_name in distributions:
            try:
                if dist_name == 'normal':
                    params = stats.norm.fit(data)
                    dist = stats.norm(*params)
                    ks_stat, p_value = stats.kstest(data, dist.cdf)
                    results['normal'] = {
                        'params': {'mu': params[0], 'sigma': params[1]},
                        'ks_statistic': ks_stat,
                        'p_value': p_value
                    }
                
                elif dist_name == 'poisson' and (data >= 0).all() and (data == data.astype(int)).all():
                    # Poisson only for non-negative integer data
                    mu = data.mean()
                    dist = stats.poisson(mu)
                    # For discrete distributions, use different test
                    results['poisson'] = {
                        'params': {'lambda': mu},
                        'mean': mu,
                        'variance': data.var(),
                        'variance_to_mean_ratio': data.var() / mu if mu > 0 else None
                    }
                
                elif dist_name == 'exponential' and (data >= 0).all():
                    params = stats.expon.fit(data)
                    dist = stats.expon(*params)
                    ks_stat, p_value = stats.kstest(data, dist.cdf)
                    results['exponential'] = {
                        'params': {'loc': params[0], 'scale': params[1]},
                        'ks_statistic': ks_stat,
                        'p_value': p_value
                    }
                
                elif dist_name == 'gamma' and (data > 0).all():
                    params = stats.gamma.fit(data)
                    dist = stats.gamma(*params)
                    ks_stat, p_value = stats.kstest(data, dist.cdf)
                    results['gamma'] = {
                        'params': {'a': params[0], 'loc': params[1], 'scale': params[2]},
                        'ks_statistic': ks_stat,
                        'p_value': p_value
                    }
            
            except Exception as e:
                print(f"⚠ Could not fit {dist_name} distribution: {e}")
        
        # Determine best fit based on KS test (higher p-value is better)
        best_dist = None
        best_pvalue = 0
        
        for dist_name, result in results.items():
            if 'p_value' in result and result['p_value'] > best_pvalue:
                best_pvalue = result['p_value']
                best_dist = dist_name
        
        if best_dist:
            print(f"✓ Best-fit distribution: {best_dist} (p={best_pvalue:.4f})")
        
        return {
            'all_fits': results,
            'best_distribution': best_dist,
            'best_p_value': best_pvalue
        }
    
    def compute_heat_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute various heat stress indices
        
        Args:
            df: Weather dataframe with temperature and humidity
            
        Returns:
            Dataframe with heat indices
        """
        df = df.copy()
        
        if ColumnNames.TEMPERATURE.value not in df.columns:
            print("⚠ Temperature column required for heat indices")
            return df
        
        temp_col = ColumnNames.TEMPERATURE.value
        
        # Heat Index
        if 'humidity' in df.columns:
            df['heat_index'] = self._calculate_heat_index(
                df[temp_col], df['humidity']
            )
        
        # Universal Thermal Climate Index (UTCI)
        df['utci_category'] = pd.cut(
            df[temp_col],
            bins=[-np.inf, 9, 26, 32, 38, np.inf],
            labels=['cold', 'comfortable', 'moderate_heat', 
                   'strong_heat', 'extreme_heat']
        )
        
        # Apparent Temperature (feels like)
        if 'humidity' in df.columns and 'wind_speed' in df.columns:
            df['apparent_temp'] = self._calculate_apparent_temp(
                df[temp_col], df['humidity'], df['wind_speed']
            )
        
        return df
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate Heat Index (simplified Steadman formula)"""
        # Convert to Fahrenheit for calculation
        temp_f = temp * 9/5 + 32
        rh = humidity
        
        # Simplified Heat Index formula
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
        
        # Convert back to Celsius
        hi_c = (hi - 32) * 5/9
        
        return hi_c
    
    def _calculate_apparent_temp(self, temp: pd.Series, 
                                 humidity: pd.Series,
                                 wind_speed: pd.Series) -> pd.Series:
        """Calculate Apparent Temperature"""
        # Simplified apparent temperature formula
        vapor_pressure = (humidity / 100) * 6.105 * np.exp(
            17.27 * temp / (237.7 + temp)
        )
        
        apparent_temp = temp + 0.33 * vapor_pressure - 0.70 * wind_speed - 4.00
        
        return apparent_temp
    
    def align_temporal_resolution(self, df: pd.DataFrame,
                                  target_frequency: str = '1H') -> pd.DataFrame:
        """
        Align weather data to target temporal resolution
        
        Args:
            df: Weather dataframe
            target_frequency: Target frequency (e.g., '1H', '1D')
            
        Returns:
            Resampled dataframe
        """
        df = df.copy()
        
        if ColumnNames.DATETIME.value not in df.columns:
            raise ValueError("Dataframe must contain datetime column")
        
        # Ensure no timezone
        df[ColumnNames.DATETIME.value] = pd.to_datetime(df[ColumnNames.DATETIME.value])
        if df[ColumnNames.DATETIME.value].dt.tz is not None:
            df[ColumnNames.DATETIME.value] = df[ColumnNames.DATETIME.value].dt.tz_localize(None)
        
        # Set datetime as index
        df = df.set_index(ColumnNames.DATETIME.value)
        
        # Define aggregation methods
        agg_methods = {}
        for col in df.columns:
            if 'temp' in col.lower() or col in ['temperature', 'heat_index', 
                                                 'apparent_temp']:
                agg_methods[col] = 'mean'
            elif 'precip' in col.lower() or 'rain' in col.lower():
                agg_methods[col] = 'sum'
            elif any(x in col.lower() for x in ['pm', 'no2', 'o3', 'co']):
                agg_methods[col] = 'mean'  # Air quality
            elif 'is_' in col.lower() or 'category' in col.lower():
                agg_methods[col] = 'max'  # Binary/categorical
            else:
                agg_methods[col] = 'mean'  # Default
        
        # Resample
        df_resampled = df.resample(target_frequency).agg(agg_methods)
        
        return df_resampled.reset_index()