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
        pass
    
    def compute_weather_mobility_correlations(self,
                                             mobility_df: pd.DataFrame,
                                             weather_df: pd.DataFrame,
                                             mobility_col: str = 'count',
                                             weather_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute correlations between mobility and weather
        
        Args:
            mobility_df: Mobility data
            weather_df: Weather data
            mobility_col: Mobility column to correlate
            weather_cols: Weather columns (if None, uses all numeric)
            
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
        
        # Compute correlations
        correlations = []
        
        for weather_col in weather_cols:
            if weather_col in merged.columns:
                clean = merged[[mobility_col, weather_col]].dropna()
                
                if len(clean) > 2:
                    corr, p_value = stats.pearsonr(clean[mobility_col], clean[weather_col])
                    
                    correlations.append({
                        'weather_variable': weather_col,
                        'correlation': corr,
                        'p_value': p_value,
                        'is_significant': p_value < 0.05,
                        'n_observations': len(clean)
                    })
        
        result_df = pd.DataFrame(correlations).sort_values('correlation', 
                                                            key=abs, 
                                                            ascending=False)
        
        print(f"  ✓ Computed {len(result_df)} weather-mobility correlations")
        
        return result_df
    
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
        
        # Create result
        result = pd.DataFrame({
            'mode': mode_cols,
            'normal_avg': normal_avg.values,
            'heat_avg': heat_avg.values,
            'change': changes.values,
            'pct_change': pct_changes.values
        })
        
        result['effect'] = result['pct_change'].apply(
            lambda x: 'increase' if x > 5 else ('decrease' if x < -5 else 'stable')
        )
        
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