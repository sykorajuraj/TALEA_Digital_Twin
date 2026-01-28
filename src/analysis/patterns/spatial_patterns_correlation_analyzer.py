"""
TALEA - Civic Digital Twin: Spatial Pattern and Correlation Analyzers
File: src/analysis/patterns/spatial_patterns_correlation_analyzer.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Combined module for spatial patterns and correlation analysis.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial import distance_matrix


class SpatialPatternAnalyzer:
    """Analyzes spatial patterns in geospatial data"""
    
    def __init__(self):
        """Initialize spatial pattern analyzer"""
        pass
    
    def compute_hotspots(self,
                        gdf: gpd.GeoDataFrame,
                        value_col: str,
                        method: str = 'kernel_density',
                        bandwidth: float = 1000) -> gpd.GeoDataFrame:
        """
        Compute spatial hotspots
        
        Args:
            gdf: GeoDataFrame with point geometries
            value_col: Column with values to analyze
            method: 'kernel_density' or 'percentile'
            bandwidth: Bandwidth for kernel density (meters)
            
        Returns:
            GeoDataFrame with hotspot scores
        """
        gdf = gdf.copy()
        
        if method == 'percentile':
            # Simple percentile-based approach
            threshold = gdf[value_col].quantile(0.75)
            gdf['is_hotspot'] = gdf[value_col] >= threshold
            gdf['hotspot_score'] = (gdf[value_col] - gdf[value_col].min()) / \
                                   (gdf[value_col].max() - gdf[value_col].min())
        
        elif method == 'kernel_density':
            # Kernel density estimation
            gdf = self._compute_kde_hotspots(gdf, value_col, bandwidth)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        hotspot_count = gdf['is_hotspot'].sum()
        print(f"  ✓ Identified {hotspot_count} hotspots")
        
        return gdf
    
    def _compute_kde_hotspots(self,
                             gdf: gpd.GeoDataFrame,
                             value_col: str,
                             bandwidth: float) -> gpd.GeoDataFrame:
        """Compute kernel density estimation hotspots"""
        
        # Get coordinates
        coords = np.array([[p.x, p.y] for p in gdf.geometry])
        values = gdf[value_col].values
        
        # Compute pairwise distances
        dists = distance_matrix(coords, coords)
        
        # Kernel density (Gaussian kernel)
        kernel_weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
        
        # Weighted density
        density = np.sum(kernel_weights * values[:, np.newaxis], axis=0)
        
        gdf['hotspot_score'] = density
        gdf['hotspot_score'] = (gdf['hotspot_score'] - gdf['hotspot_score'].min()) / \
                               (gdf['hotspot_score'].max() - gdf['hotspot_score'].min())
        
        # Threshold
        threshold = gdf['hotspot_score'].quantile(0.75)
        gdf['is_hotspot'] = gdf['hotspot_score'] >= threshold
        
        return gdf
    
    def compute_spatial_autocorrelation(self,
                                       gdf: gpd.GeoDataFrame,
                                       value_col: str,
                                       k_neighbors: int = 8) -> Dict:
        """
        Compute spatial autocorrelation (Moran's I)
        
        Args:
            gdf: GeoDataFrame
            value_col: Column to analyze
            k_neighbors: Number of nearest neighbors
            
        Returns:
            Dictionary with autocorrelation statistics
        """
        # Get coordinates
        coords = np.array([[p.x, p.y] for p in gdf.geometry])
        values = gdf[value_col].values
        
        # Compute distance matrix
        dists = distance_matrix(coords, coords)
        
        # Create spatial weights (k-nearest neighbors)
        weights = np.zeros_like(dists)
        for i in range(len(coords)):
            # Get k nearest neighbors (excluding self)
            nearest = np.argsort(dists[i])[1:k_neighbors+1]
            weights[i, nearest] = 1
        
        # Standardize values
        z = (values - values.mean()) / values.std()
        
        # Compute Moran's I
        n = len(values)
        W = weights.sum()
        
        numerator = np.sum(weights * np.outer(z, z))
        denominator = np.sum(z ** 2)
        
        moran_i = (n / W) * (numerator / denominator)
        
        # Expected value under null hypothesis
        expected_i = -1 / (n - 1)
        
        # Variance (simplified)
        variance_i = 1 / (n - 1)
        
        # Z-score
        z_score = (moran_i - expected_i) / np.sqrt(variance_i)
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        result = {
            'moran_i': float(moran_i),
            'expected_i': float(expected_i),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'interpretation': 'clustered' if moran_i > expected_i else 'dispersed'
        }
        
        print(f"  ✓ Moran's I = {moran_i:.3f} (p={p_value:.4f})")
        
        return result


class CorrelationAnalyzer:
    """Analyzes correlations between different datasets"""
    
    def __init__(self):
        """Initialize correlation analyzer"""
        self._correlation_history = {}  # Store time-varying correlations
    
    def compute_weather_mobility_correlations(self,
                                             mobility_df: pd.DataFrame,
                                             weather_df: pd.DataFrame,
                                             mobility_col: str = 'count',
                                             weather_cols: Optional[List[str]] = None,
                                             stratify_by: Optional[str] = None) -> pd.DataFrame:
        """
        Compute correlations between mobility and weather with optional stratification.
        
        Args:
            mobility_df: Mobility data
            weather_df: Weather data
            mobility_col: Mobility column to correlate
            weather_cols: Weather columns (if None, uses all numeric)
            stratify_by: Column to stratify analysis (e.g., 'hour', 'weekday')
            
        Returns:
            DataFrame with correlation results
        """
        # Merge datasets
        merged = self._merge_datasets(mobility_df, weather_df)
        
        if mobility_col not in merged.columns:
            raise ValueError(f"Mobility column '{mobility_col}' not found")
        
        # Select weather columns
        if weather_cols is None:
            weather_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
            weather_cols = [c for c in weather_cols if c != mobility_col]
        
        # If stratification requested, compute within each stratum
        if stratify_by is not None:
            return self._compute_stratified_correlations(
                merged, mobility_col, weather_cols, stratify_by
            )
        
        # Compute overall correlations
        correlations = []
        
        for weather_col in weather_cols:
            if weather_col in merged.columns:
                clean = merged[[mobility_col, weather_col]].dropna()
                
                if len(clean) > 2:
                    # Pearson correlation (linear)
                    pearson_r, pearson_p = stats.pearsonr(clean[mobility_col], clean[weather_col])
                    
                    # Spearman correlation (monotonic, more robust)
                    spearman_r, spearman_p = stats.spearmanr(clean[mobility_col], clean[weather_col])
                    
                    correlations.append({
                        'weather_variable': weather_col,
                        'pearson_correlation': pearson_r,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_r,
                        'spearman_p_value': spearman_p,
                        'is_significant': pearson_p < 0.05 or spearman_p < 0.05,
                        'n_observations': len(clean)
                    })
        
        result_df = pd.DataFrame(correlations).sort_values('pearson_correlation', 
                                                            key=abs, 
                                                            ascending=False)
        
        print(f"  ✓ Computed {len(result_df)} weather-mobility correlations")
        
        return result_df
    
    def _compute_stratified_correlations(self,
                                        merged: pd.DataFrame,
                                        mobility_col: str,
                                        weather_cols: List[str],
                                        stratify_by: str) -> pd.DataFrame:
        """
        Compute correlations within temporal strata.
        """
        # Extract stratification variable
        if stratify_by not in merged.columns:
            # Try to extract from datetime
            if 'datetime' in merged.columns or 'Data' in merged.columns:
                date_col = 'datetime' if 'datetime' in merged.columns else 'Data'
                dt = pd.to_datetime(merged[date_col])
                
                if stratify_by == 'hour':
                    merged[stratify_by] = dt.dt.hour
                elif stratify_by == 'weekday':
                    merged[stratify_by] = dt.dt.weekday
                elif stratify_by == 'month':
                    merged[stratify_by] = dt.dt.month
        
        stratified_results = []
        
        for stratum_value, group in merged.groupby(stratify_by):
            for weather_col in weather_cols:
                if weather_col in group.columns:
                    clean = group[[mobility_col, weather_col]].dropna()
                    
                    if len(clean) > 5:  # Need more samples per stratum
                        corr, p_value = stats.pearsonr(clean[mobility_col], clean[weather_col])
                        
                        stratified_results.append({
                            'weather_variable': weather_col,
                            f'{stratify_by}': stratum_value,
                            'correlation': corr,
                            'p_value': p_value,
                            'is_significant': p_value < 0.05,
                            'n_observations': len(clean)
                        })
        
        result_df = pd.DataFrame(stratified_results)
        
        print(f"  ✓ Computed stratified correlations by {stratify_by}")
        
        return result_df
    
    def compute_time_lagged_correlations(self,
                                        series1: pd.Series,
                                        series2: pd.Series,
                                        max_lag: int = 24) -> pd.DataFrame:
        """
        Compute correlations at different time lags.
        
        Args:
            series1: First time series
            series2: Second time series
            max_lag: Maximum lag to test (in time steps)
            
        Returns:
            DataFrame with correlation at each lag
        """
        results = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # series1 leads series2
                s1 = series1.iloc[:lag]
                s2 = series2.iloc[-lag:]
            elif lag > 0:
                # series2 leads series1
                s1 = series1.iloc[lag:]
                s2 = series2.iloc[:-lag]
            else:
                # No lag
                s1 = series1
                s2 = series2
            
            # Align indices
            common_idx = s1.index.intersection(s2.index)
            if len(common_idx) > 10:
                s1_aligned = s1.loc[common_idx]
                s2_aligned = s2.loc[common_idx]
                
                corr, p_value = stats.pearsonr(s1_aligned, s2_aligned)
                
                results.append({
                    'lag': lag,
                    'correlation': corr,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05,
                    'n_observations': len(common_idx)
                })
        
        result_df = pd.DataFrame(results)
        
        # Find optimal lag
        if len(result_df) > 0:
            optimal_lag = result_df.loc[result_df['correlation'].abs().idxmax(), 'lag']
            optimal_corr = result_df.loc[result_df['correlation'].abs().idxmax(), 'correlation']
            
            print(f"  ✓ Optimal lag: {optimal_lag} (correlation={optimal_corr:.3f})")
        
        return result_df
    
    def compute_weighted_correlation(self,
                                   df: pd.DataFrame,
                                   col1: str,
                                   col2: str,
                                   weight_col: str,
                                   method: str = 'pearson') -> Dict:
        """
        Compute weighted correlation coefficient.

        Args:
            df: DataFrame with data
            col1: First column
            col2: Second column
            weight_col: Column with sample weights
            method: 'pearson' or 'spearman'
            
        Returns:
            Dictionary with correlation results
        """
        clean = df[[col1, col2, weight_col]].dropna()
        
        if len(clean) < 3:
            return {'correlation': np.nan, 'n_observations': len(clean)}
        
        x = clean[col1].values
        y = clean[col2].values
        w = clean[weight_col].values
        
        # Normalize weights
        w = w / w.sum()
        
        if method == 'pearson':
            # Weighted Pearson correlation
            # Formula: cov_w(X,Y) / (std_w(X) * std_w(Y))
            
            x_mean = np.sum(w * x)
            y_mean = np.sum(w * y)
            
            cov_xy = np.sum(w * (x - x_mean) * (y - y_mean))
            
            var_x = np.sum(w * (x - x_mean) ** 2)
            var_y = np.sum(w * (y - y_mean) ** 2)
            
            if var_x > 0 and var_y > 0:
                corr = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))
            else:
                corr = np.nan
            
        elif method == 'spearman':
            # For Spearman, apply weights to ranks
            # This is an approximation
            x_ranks = stats.rankdata(x)
            y_ranks = stats.rankdata(y)
            
            x_mean = np.sum(w * x_ranks)
            y_mean = np.sum(w * y_ranks)
            
            cov_xy = np.sum(w * (x_ranks - x_mean) * (y_ranks - y_mean))
            
            var_x = np.sum(w * (x_ranks - x_mean) ** 2)
            var_y = np.sum(w * (y_ranks - y_mean) ** 2)
            
            if var_x > 0 and var_y > 0:
                corr = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))
            else:
                corr = np.nan
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result = {
            'correlation': float(corr),
            'method': method,
            'weighted': True,
            'n_observations': len(clean),
            'effective_n': 1 / np.sum(w ** 2)  # Effective sample size
        }
        
        print(f"  ✓ Weighted {method} correlation: {corr:.3f} "
              f"(effective n={result['effective_n']:.0f})")
        
        return result
    
    def analyze_cross_modal_patterns(self,
                                    bicycle_df: pd.DataFrame,
                                    pedestrian_df: pd.DataFrame,
                                    transit_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze patterns across different mobility modes
        
        Args:
            bicycle_df: Bicycle data
            pedestrian_df: Pedestrian data
            transit_df: Transit data (optional)
            
        Returns:
            Dictionary with cross-modal analysis
        """
        results = {}
        
        # Bicycle vs Pedestrian
        bike_ped = self._compute_mode_correlation(
            bicycle_df, pedestrian_df,
            'bicycle', 'pedestrian'
        )
        results['bicycle_pedestrian'] = bike_ped
        
        # Add transit if available
        if transit_df is not None:
            bike_transit = self._compute_mode_correlation(
                bicycle_df, transit_df,
                'bicycle', 'transit'
            )
            results['bicycle_transit'] = bike_transit
            
            ped_transit = self._compute_mode_correlation(
                pedestrian_df, transit_df,
                'pedestrian', 'transit'
            )
            results['pedestrian_transit'] = ped_transit
        
        print(f"  ✓ Analyzed {len(results)} cross-modal patterns")
        
        return results
    
    def _compute_mode_correlation(self,
                                 df1: pd.DataFrame,
                                 df2: pd.DataFrame,
                                 mode1: str,
                                 mode2: str) -> Dict:
        """Compute correlation between two modes"""
        
        # Find count columns
        count_col1 = self._find_count_column(df1, mode1)
        count_col2 = self._find_count_column(df2, mode2)
        
        # Merge
        merged = self._merge_datasets(df1, df2)
        
        # Compute correlation
        clean = merged[[count_col1, count_col2]].dropna()
        
        if len(clean) > 2:
            corr, p_value = stats.pearsonr(clean[count_col1], clean[count_col2])
        else:
            corr, p_value = np.nan, np.nan
        
        return {
            'mode1': mode1,
            'mode2': mode2,
            'correlation': float(corr) if not np.isnan(corr) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'is_significant': p_value < 0.05 if not np.isnan(p_value) else False,
            'n_observations': len(clean)
        }
    
    def identify_substitution_effects(self,
                                     df: pd.DataFrame,
                                     heat_events: pd.Series,
                                     mode_cols: List[str]) -> pd.DataFrame:
        """
        Identify modal substitution during heat events
        
        Args:
            df: Multi-modal mobility data
            heat_events: Boolean series indicating heat events
            mode_cols: List of columns for different modes
            
        Returns:
            DataFrame with substitution analysis
        """
        # Split by heat condition
        normal = df[~heat_events][mode_cols]
        heat = df[heat_events][mode_cols]
        
        # Compute averages
        normal_avg = normal.mean()
        heat_avg = heat.mean()
        
        # Changes
        changes = heat_avg - normal_avg
        pct_changes = ((heat_avg - normal_avg) / normal_avg * 100).fillna(0)
        
        # Statistical significance test (t-test)
        results_list = []
        for col in mode_cols:
            if col in normal.columns and col in heat.columns:
                t_stat, t_pvalue = stats.ttest_ind(
                    normal[col].dropna(),
                    heat[col].dropna(),
                    equal_var=False  # Welch's t-test
                )
                
                results_list.append({
                    'mode': col,
                    'normal_avg': normal_avg[col],
                    'heat_avg': heat_avg[col],
                    'change': changes[col],
                    'pct_change': pct_changes[col],
                    't_statistic': t_stat,
                    'p_value': t_pvalue,
                    'is_significant': t_pvalue < 0.05,
                    'effect': 'increase' if pct_changes[col] > 5 else (
                        'decrease' if pct_changes[col] < -5 else 'stable'
                    )
                })
        
        result = pd.DataFrame(results_list)
        
        print(f"  ✓ Analyzed substitution effects for {len(mode_cols)} modes")
        
        return result
    
    @staticmethod
    def _find_count_column(df: pd.DataFrame, mode: str) -> str:
        """Find the count column for a mode"""
        
        # Mapping of modes to likely column names
        column_map = {
            'bicycle': ['Totale', 'count', 'bicycle_count'],
            'pedestrian': ['Numero di visitatori', 'count', 'pedestrian_count'],
            'transit': ['vehicle_count', 'count', 'transit_count']
        }
        
        candidates = column_map.get(mode, ['count'])
        
        for col in candidates:
            if col in df.columns:
                return col
        
        # Fallback to first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        raise ValueError(f"Cannot find count column for mode: {mode}")
    
    @staticmethod
    def _merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Merge two datasets on datetime"""
        
        # Find datetime columns
        date_col1 = None
        for col in ['datetime', 'Data', 'data']:
            if col in df1.columns:
                date_col1 = col
                break
        
        date_col2 = None
        for col in ['datetime', 'Data', 'data']:
            if col in df2.columns:
                date_col2 = col
                break
        
        if date_col1 is None or date_col2 is None:
            raise ValueError("Both datasets must have datetime column")
        
        # Merge on date
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        
        df1_copy['merge_date'] = pd.to_datetime(df1_copy[date_col1]).dt.date
        df2_copy['merge_date'] = pd.to_datetime(df2_copy[date_col2]).dt.date
        
        merged = df1_copy.merge(df2_copy, on='merge_date', how='inner', 
                               suffixes=('_1', '_2'))
        
        return merged