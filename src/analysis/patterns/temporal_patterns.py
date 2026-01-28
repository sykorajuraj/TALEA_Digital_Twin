"""
TALEA - Civic Digital Twin: Temporal Pattern Analyzer
File: src/analysis/patterns/temporal_patterns.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Analyzes temporal patterns in mobility and weather data.

Enhanced with KDE-based anomaly detection and time-indexed models:
- Sliding window sequences for capturing temporal dependencies
- Time-dependent density estimators (one estimator per time period)
- Autocorrelation-based period detection
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


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
    
    def compute_autocorrelation(self,
                               df: pd.DataFrame,
                               value_col: str,
                               max_lag: int = 96) -> Dict:
        """
        Compute autocorrelation to detect periodicity in time series.
        Based on PDF guidance for period detection via autocorrelation.
        
        Args:
            df: DataFrame with time series data
            value_col: Column with values to analyze
            max_lag: Maximum lag to compute (e.g., 96 for 48 hours with 30-min intervals)
            
        Returns:
            Dictionary with autocorrelation values and detected periods
        """
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found")
        
        # Get series and remove NaNs
        series = df[value_col].dropna()
        
        if len(series) < max_lag + 1:
            max_lag = len(series) - 1
        
        # Compute autocorrelation for each lag
        acf_values = []
        lags = list(range(max_lag + 1))
        
        for lag in lags:
            if lag == 0:
                acf_values.append(1.0)
            else:
                # Pearson correlation between series and lagged series
                corr = series.iloc[:-lag].corr(series.iloc[lag:])
                acf_values.append(corr if not np.isnan(corr) else 0.0)
        
        # Find peaks in autocorrelation (potential periods)
        acf_array = np.array(acf_values)
        peaks = []
        
        # Skip lag 0 and find local maxima
        for i in range(2, len(acf_array) - 1):
            if acf_array[i] > acf_array[i-1] and acf_array[i] > acf_array[i+1]:
                if acf_array[i] > 0.3:  # Only significant correlations
                    peaks.append((i, acf_array[i]))
        
        # Sort by correlation strength
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        result = {
            'lags': lags,
            'acf_values': acf_values,
            'detected_periods': [p[0] for p in peaks[:3]],  # Top 3 periods
            'peak_correlations': [p[1] for p in peaks[:3]]
        }
        
        if peaks:
            print(f"  ✓ Detected primary period: {peaks[0][0]} steps (correlation: {peaks[0][1]:.3f})")
        
        return result
    
    def create_sliding_windows(self,
                              df: pd.DataFrame,
                              value_col: str,
                              window_length: int = 10,
                              normalize: bool = True) -> pd.DataFrame:
        """
        Create sliding windows for sequence-based analysis.
        Based on treats each sequence as a vector for multivariate KDE.
        
        Args:
            df: DataFrame with time series
            value_col: Column to window
            window_length: Number of timesteps per window
            normalize: Whether to normalize values (recommended for KDE)
            
        Returns:
            DataFrame where each row is a window sequence
        """
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found")
        
        series = df[value_col].values
        
        # Normalize if requested
        if normalize:
            scaler = MinMaxScaler()
            series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
        
        # Create windows
        windows = []
        indices = []
        
        for i in range(len(series) - window_length + 1):
            window = series[i:i + window_length]
            windows.append(window)
            indices.append(df.index[i + window_length - 1])  # Use last timestamp
        
        # Create DataFrame
        window_df = pd.DataFrame(
            windows,
            index=indices,
            columns=[f'lag_{j}' for j in range(window_length)]
        )
        
        print(f"  ✓ Created {len(window_df)} sliding windows of length {window_length}")
        
        return window_df
    
    def optimize_kde_bandwidth(self,
                              train_data: pd.DataFrame,
                              bandwidth_range: Tuple[float, float] = (0.01, 0.1),
                              n_bandwidths: int = 10,
                              cv_folds: int = 5) -> Dict:
        """
        Optimize KDE bandwidth using cross-validation.
        Pick bandwidth that maximizes likelihood on validation set.
        
        Args:
            train_data: Training data (can be univariate or multivariate)
            bandwidth_range: (min, max) bandwidth values to test
            n_bandwidths: Number of bandwidth values to try
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best bandwidth and fitted model
        """
        # Create bandwidth grid
        bandwidths = np.linspace(bandwidth_range[0], bandwidth_range[1], n_bandwidths)
        params = {'bandwidth': bandwidths}
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            params,
            cv=cv_folds,
            n_jobs=-1
        )
        
        grid_search.fit(train_data.values if isinstance(train_data, pd.DataFrame) else train_data)
        
        best_bandwidth = grid_search.best_params_['bandwidth']
        
        print(f"  ✓ Optimal bandwidth: {best_bandwidth:.6f}")
        
        return {
            'best_bandwidth': best_bandwidth,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
    
    def detect_anomalies_sequence_kde(self,
                                     df: pd.DataFrame,
                                     value_col: str,
                                     window_length: int = 10,
                                     train_end_idx: Optional[int] = None,
                                     threshold_percentile: float = 95.0) -> pd.Series:
        """
        Detect anomalies using sequence-based KDE (multivariate KDE on sliding windows).
        Based on capturing temporal dependencies through window sequences.
        
        Args:
            df: DataFrame with time series
            value_col: Column to analyze
            window_length: Length of sliding window
            train_end_idx: Index to split train/test (None = use first 70%)
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Series with anomaly scores (higher = more anomalous)
        """
        # Create sliding windows
        window_df = self.create_sliding_windows(df, value_col, window_length, normalize=True)
        
        # Split train/test
        if train_end_idx is None:
            train_end_idx = int(len(window_df) * 0.7)
        
        train_data = window_df.iloc[:train_end_idx]
        
        # Optimize bandwidth
        bandwidth_result = self.optimize_kde_bandwidth(
            train_data,
            bandwidth_range=(0.01, 0.2),
            n_bandwidths=15
        )
        
        kde = bandwidth_result['best_estimator']
        
        # Compute log-density (negative = anomaly score)
        log_density = kde.score_samples(window_df.values)
        anomaly_scores = -log_density  # Higher score = more anomalous
        
        # Create series with original index
        scores_series = pd.Series(anomaly_scores, index=window_df.index)
        
        # Determine threshold from training data
        train_scores = scores_series.iloc[:train_end_idx]
        threshold = train_scores.quantile(threshold_percentile / 100)
        
        n_anomalies = (scores_series > threshold).sum()
        print(f"  ✓ Detected {n_anomalies} anomalies ({n_anomalies/len(scores_series)*100:.1f}%)")
        
        return scores_series
    
    def detect_anomalies_time_indexed_kde(self,
                                         df: pd.DataFrame,
                                         value_col: str,
                                         window_length: int = 10,
                                         time_resolution: str = '30T',
                                         train_end_idx: Optional[int] = None,
                                         threshold_percentile: float = 95.0) -> pd.Series:
        """
        Detect anomalies using time-indexed KDE (one estimator per time period).
        Based on specialized estimators for each time value avoid learning 
        complex unnecessary distributions.
        
        Args:
            df: DataFrame with datetime index and time series
            value_col: Column to analyze
            window_length: Length of sliding window
            time_resolution: Time resolution for separate estimators (e.g., '30T', '1H')
            train_end_idx: Index to split train/test
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Series with anomaly scores
        """
        df = df.copy()
        df = self._add_temporal_features(df)
        
        # Create sliding windows
        window_df = self.create_sliding_windows(df, value_col, window_length, normalize=True)
        
        # Extract time-of-day (in fractional hours)
        date_col = self._find_datetime_column(df)
        if date_col:
            time_index = df.loc[window_df.index]
            day_hours = time_index['hour'] + time_index.get('minute', 0) / 60
        else:
            raise ValueError("Cannot extract time information")
        
        # Determine unique time slots
        unique_times = np.sort(day_hours.unique())
        
        # Split train/test
        if train_end_idx is None:
            train_end_idx = int(len(window_df) * 0.7)
        
        # Train one KDE per time slot
        kde_models = {}
        
        # Optimize single bandwidth on all training data
        train_data = window_df.iloc[:train_end_idx]
        bandwidth_result = self.optimize_kde_bandwidth(
            train_data,
            bandwidth_range=(0.01, 0.15),
            n_bandwidths=10,
            cv_folds=3
        )
        bandwidth = bandwidth_result['best_bandwidth']
        
        print(f"  ⋯ Training {len(unique_times)} time-specific KDE models...")
        
        for time_slot in unique_times:
            # Get training data for this time slot
            time_mask = (day_hours == time_slot) & (window_df.index <= window_df.index[train_end_idx])
            slot_train_data = window_df[time_mask]
            
            if len(slot_train_data) > 5:  # Need minimum samples
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
                kde.fit(slot_train_data.values)
                kde_models[time_slot] = kde
        
        # Compute anomaly scores using appropriate time-specific model
        anomaly_scores = []
        valid_indices = []
        
        for idx, time_val in zip(window_df.index, day_hours):
            # Find closest time slot with a model
            if time_val in kde_models:
                kde = kde_models[time_val]
            else:
                # Use nearest time slot
                distances = np.abs(np.array(list(kde_models.keys())) - time_val)
                nearest_time = list(kde_models.keys())[np.argmin(distances)]
                kde = kde_models[nearest_time]
            
            # Compute score
            window = window_df.loc[idx].values.reshape(1, -1)
            log_density = kde.score_samples(window)[0]
            anomaly_scores.append(-log_density)
            valid_indices.append(idx)
        
        scores_series = pd.Series(anomaly_scores, index=valid_indices)
        
        # Threshold
        train_scores = scores_series.iloc[:train_end_idx]
        threshold = train_scores.quantile(threshold_percentile / 100)
        
        n_anomalies = (scores_series > threshold).sum()
        print(f"  ✓ Detected {n_anomalies} anomalies using {len(kde_models)} time-indexed models")
        
        return scores_series
    
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
        date_col = self._find_datetime_column(df)
        
        if date_col is None:
            return df
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Extract features
        if 'hour' not in df.columns:
            df['hour'] = df[date_col].dt.hour
        
        if 'minute' not in df.columns:
            df['minute'] = df[date_col].dt.minute
        
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
    
    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find datetime column in dataframe"""
        for col in ['datetime', 'Data', 'data', 'reftime']:
            if col in df.columns:
                return col
        return None
    
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