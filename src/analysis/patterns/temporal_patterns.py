"""
TALEA - Civic Digital Twin: Temporal Pattern Analyzer
File: src/analysis/patterns/temporal_patterns.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Analyzes temporal patterns in mobility and weather data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple


class TemporalPatternAnalyzer:
    """Analyzes temporal patterns in time-series data"""
    
    def __init__(self):
        """Initialize temporal pattern analyzer"""
        pass
    
    def detect_daily_patterns(self,
                            df: pd.DataFrame,
                            groupby_cols: List[str],
                            value_col: Optional[str] = None) -> pd.DataFrame:
        """
        Detect daily patterns by grouping
        
        Args:
            df: DataFrame with datetime column
            groupby_cols: Columns to group by (e.g., ['hour', 'day_of_week'])
            value_col: Value column to aggregate (if None, counts occurrences)
            
        Returns:
            DataFrame with daily patterns
        """
        # Ensure required columns exist
        for col in groupby_cols:
            if col not in df.columns:
                # Try to create from datetime
                if 'datetime' in df.columns or 'Data' in df.columns:
                    df = self._add_temporal_features(df)
                else:
                    raise ValueError(f"Column '{col}' not found and cannot be derived")
        
        # Group and aggregate
        if value_col:
            patterns = df.groupby(groupby_cols)[value_col].agg(['mean', 'std', 'count'])
        else:
            patterns = df.groupby(groupby_cols).size().to_frame('count')
        
        patterns = patterns.reset_index()
        
        print(f"  ✓ Detected daily patterns across {len(patterns)} combinations")
        
        return patterns
    
    def detect_weekly_patterns(self,
                              df: pd.DataFrame,
                              value_col: Optional[str] = None) -> pd.DataFrame:
        """
        Detect weekly patterns (by day of week)
        
        Args:
            df: DataFrame with datetime
            value_col: Value column to aggregate
            
        Returns:
            DataFrame with weekly patterns
        """
        df = self._add_temporal_features(df)
        
        if 'day_of_week' not in df.columns:
            raise ValueError("Cannot derive day_of_week from data")
        
        if value_col:
            weekly = df.groupby('day_of_week')[value_col].agg(['mean', 'std', 'count'])
        else:
            weekly = df.groupby('day_of_week').size().to_frame('count')
        
        weekly = weekly.reset_index()
        
        # Add day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        weekly['day_name'] = weekly['day_of_week'].map(
            lambda x: day_names[int(x)] if 0 <= x < 7 else 'Unknown'
        )
        
        print(f"  ✓ Detected weekly patterns")
        
        return weekly
    
    def detect_seasonal_patterns(self,
                                df: pd.DataFrame,
                                value_col: Optional[str] = None) -> pd.DataFrame:
        """
        Detect seasonal patterns
        
        Args:
            df: DataFrame with datetime
            value_col: Value column to aggregate
            
        Returns:
            DataFrame with seasonal patterns
        """
        df = self._add_temporal_features(df)
        
        if 'season' not in df.columns:
            raise ValueError("Cannot derive season from data")
        
        if value_col:
            seasonal = df.groupby('season')[value_col].agg(['mean', 'std', 'count'])
        else:
            seasonal = df.groupby('season').size().to_frame('count')
        
        seasonal = seasonal.reset_index()
        
        print(f"  ✓ Detected seasonal patterns")
        
        return seasonal
    
    def identify_peak_hours(self,
                           df: pd.DataFrame,
                           value_col: str,
                           n_peaks: int = 3) -> Dict:
        """
        Identify peak hours in the data
        
        Args:
            df: DataFrame with hour information
            value_col: Column with values to find peaks
            n_peaks: Number of peak hours to identify
            
        Returns:
            Dictionary with peak hour information
        """
        df = self._add_temporal_features(df)
        
        if 'hour' not in df.columns:
            raise ValueError("Cannot derive hour from data")
        
        # Aggregate by hour
        hourly = df.groupby('hour')[value_col].mean().sort_values(ascending=False)
        
        # Get top n peaks
        top_peaks = hourly.head(n_peaks)
        
        peak_info = {
            'peak_hours': top_peaks.index.tolist(),
            'peak_values': top_peaks.values.tolist(),
            'hourly_pattern': hourly.to_dict()
        }
        
        print(f"  ✓ Identified {n_peaks} peak hours: {peak_info['peak_hours']}")
        
        return peak_info
    
    def compute_time_series_decomposition(self,
                                         df: pd.DataFrame,
                                         value_col: str,
                                         frequency: str = 'D') -> Dict:
        """
        Decompose time series into trend, seasonal, and residual
        
        Args:
            df: DataFrame with datetime index
            value_col: Value column to decompose
            frequency: Frequency for resampling ('H', 'D', 'W', 'M')
            
        Returns:
            Dictionary with decomposition components
        """
        # Ensure datetime index
        df_ts = df.copy()
        
        date_col = None
        for col in ['datetime', 'Data', 'data']:
            if col in df_ts.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("No datetime column found")
        
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.set_index(date_col)
        
        # Resample to regular frequency
        ts = df_ts[value_col].resample(frequency).mean()
        
        # Simple moving average for trend
        window = 7 if frequency == 'D' else 3
        trend = ts.rolling(window=window, center=True).mean()
        
        # Detrend
        detrended = ts - trend
        
        # Seasonal (simplified: average by period)
        if frequency == 'D':
            seasonal = detrended.groupby(detrended.index.dayofweek).transform('mean')
        else:
            seasonal = detrended.groupby(detrended.index.hour).transform('mean')
        
        # Residual
        residual = detrended - seasonal
        
        return {
            'original': ts,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from datetime column"""
        
        df = df.copy()
        
        # Find datetime column
        date_col = None
        for col in ['datetime', 'Data', 'data', 'reftime']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            return df
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Extract features
        if 'hour' not in df.columns:
            df['hour'] = df[date_col].dt.hour
        
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df[date_col].dt.dayofweek
        
        if 'day' not in df.columns:
            df['day'] = df[date_col].dt.day
        
        if 'month' not in df.columns:
            df['month'] = df[date_col].dt.month
        
        if 'year' not in df.columns:
            df['year'] = df[date_col].dt.year
        
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        if 'season' not in df.columns:
            df['season'] = df['month'].map(self._month_to_season)
        
        return df
    
    @staticmethod
    def _month_to_season(month: int) -> str:
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        else:
            return 'unknown'