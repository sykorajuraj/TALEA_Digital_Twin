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
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


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


class WeatherPreprocessor:
    """Unified preprocessor for all weather data"""
    
    def __init__(self, config: Optional[WeatherConfig] = None):
        self.config = config or WeatherConfig()
    
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
        
        # Convert to datetime
        df[ColumnNames.DATETIME.value] = pd.to_datetime(df[ColumnNames.DATETIME.value])
        
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
        
        # Convert to datetime
        df[ColumnNames.DATETIME.value] = pd.to_datetime(df[ColumnNames.DATETIME.value])
        
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
        
        # Convert reftime to datetime
        if 'reftime' in df.columns:
            df[ColumnNames.DATETIME.value] = pd.to_datetime(df['reftime'])
        
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
    
    def interpolate_weather_grid(self, df: pd.DataFrame,
                                 spatial_resolution: float = 0.01) -> pd.DataFrame:
        """
        Interpolate weather data to spatial grid
        (Placeholder for spatial interpolation)
        
        Args:
            df: Weather dataframe
            spatial_resolution: Grid resolution in degrees
            
        Returns:
            Interpolated weather data
        """
        # This is a placeholder - full implementation would use
        # geostatistical methods like kriging
        print("⚠ Spatial interpolation not yet implemented")
        return df
    
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