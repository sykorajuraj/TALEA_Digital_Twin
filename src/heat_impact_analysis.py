"""
TALEA - Civic Digital Twin: Pattern Analysis and Heat Impact Module
File: src/heat_impact_analysis.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

This module analyzes mobility patterns, detects anomalies, and studies
the impact of heat stress on bicycle mobility in Bologna:
- Multi-modal mobility patterns (bicycle, pedestrian, vehicle)
- Origin-destination flows
- Heat impact across all modes
- Vulnerable routes and POIs
- ZTL effectiveness
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION

@dataclass
class AnalysisConfig:
    """Configuration for pattern and heat analysis"""
    
    # Heat thresholds
    HIGH_TEMP_THRESHOLD: float = 30.0
    MODERATE_TEMP_THRESHOLD: float = 25.0
    
    # Vulnerability analysis
    VULNERABLE_PERCENTILE: float = 25.0
    
    # Anomaly detection
    ANOMALY_CONTAMINATION: float = 0.05
    ANOMALY_RANDOM_STATE: int = 42
    
    # Statistical tests
    SIGNIFICANCE_LEVEL: float = 0.05
    
    # Spatial analysis (ZTL proximity)
    ZTL_PROXIMITY_METERS: float = 500.0  # 500m radius
    
    # Preprocessing config reference
    preprocessing_config: Optional['ProcessingConfig'] = None


class MobilityMode(Enum):
    """Standardized mobility mode identifiers"""
    BICYCLE = 'bicycle'
    PEDESTRIAN = 'pedestrian'
    TRAFFIC = 'traffic'

# VALIDATION

class DataValidator:
    """Validates datasets before analysis"""
    
    REQUIRED_COLUMNS = {
        MobilityMode.BICYCLE: {
            'required': ['Totale', 'hour', 'season', 'is_weekend'],
            'optional': ['temperature', 'is_hot_day', 'is_very_hot_day', 
                        'latitude', 'longitude', 'Dispositivo conta-bici']
        },
        MobilityMode.PEDESTRIAN: {
            'required': ['Numero di visitatori', 'hour', 'season', 'is_weekend'],
            'optional': ['temperature', 'is_hot_day', 'Area provenienza', 'Area Arrivo', 'route']
        },
        MobilityMode.TRAFFIC: {
            'required': ['vehicle_count', 'hour', 'season', 'is_weekend'],
            'optional': ['temperature', 'is_hot_day', 'VIA_SPIRA', 'latitude', 'longitude']
        }
    }
    
    @classmethod
    def validate_dataset(cls, df: pd.DataFrame, mode: MobilityMode) -> Tuple[bool, List[str]]:
        """
        Validate dataset has required columns.
        
        Returns:
            (is_valid, missing_columns)
        """
        if df is None or len(df) == 0:
            return False, ['Dataset is empty or None']
        
        schema = cls.REQUIRED_COLUMNS.get(mode)
        if not schema:
            return False, [f'Unknown mode: {mode}']
        
        missing = [col for col in schema['required'] if col not in df.columns]
        
        return len(missing) == 0, missing
    
    @classmethod
    def validate_all_datasets(cls, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate all datasets and return validation report"""
        validation_report = {}
        
        for dataset_name, df in datasets.items():
            if dataset_name in ['bicycle', 'pedestrian', 'traffic']:
                mode = MobilityMode(dataset_name)
                is_valid, missing = cls.validate_dataset(df, mode)
                validation_report[dataset_name] = (is_valid, missing)
        
        return validation_report
    
    @classmethod
    def check_weather_integration(cls, df: pd.DataFrame) -> str:
        """
        Determine weather integration status.
        
        Returns:
            'preprocessed_flags' | 'temperature_column' | 'season_proxy' | 'none'
        """
        if 'is_very_hot_day' in df.columns and 'is_hot_day' in df.columns:
            return 'preprocessed_flags'
        elif 'temperature' in df.columns:
            return 'temperature_column'
        elif 'season' in df.columns:
            return 'season_proxy'
        else:
            return 'none'


# PATTERN ANALYZER

class PatternAnalyzer:
    """Analyzes temporal and behavioral mobility patterns"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def analyze_temporal_patterns(self, df: pd.DataFrame, 
                                  count_col: str, 
                                  mode_name: str) -> Dict:
        """
        Comprehensive temporal pattern analysis.
        
        Returns dictionary with hourly, daily, seasonal patterns.
        """
        patterns = {}
        
        # Hourly patterns
        if 'hour' in df.columns:
            try:
                hourly = df.groupby('hour')[count_col].agg(['mean', 'std', 'count'])
                patterns['hourly'] = hourly
                peak_hour = hourly['mean'].idxmax()
                patterns['peak_hour'] = int(peak_hour)
                patterns['peak_hour_avg'] = float(hourly.loc[peak_hour, 'mean'])
            except Exception as e:
                print(f"  ⚠ Hourly analysis failed: {e}")
        
        # Daily patterns
        if 'day_of_week' in df.columns:
            try:
                daily = df.groupby('day_of_week')[count_col].agg(['mean', 'std'])
                patterns['daily'] = daily
                busiest_day = daily['mean'].idxmax()
                patterns['busiest_day'] = int(busiest_day)
                patterns['busiest_day_avg'] = float(daily.loc[busiest_day, 'mean'])
            except Exception as e:
                print(f"  ⚠ Daily analysis failed: {e}")
        
        # Seasonal patterns
        if 'season' in df.columns:
            try:
                seasonal = df.groupby('season')[count_col].agg(['mean', 'std', 'count'])
                patterns['seasonal'] = seasonal
                
                if len(seasonal) > 1:
                    variation = seasonal['mean'].max() / seasonal['mean'].min()
                    patterns['seasonal_variation'] = float(variation)
            except Exception as e:
                print(f"  ⚠ Seasonal analysis failed: {e}")
        
        # Weekend vs weekday
        if 'is_weekend' in df.columns:
            try:
                weekend_avg = df[df['is_weekend'] == 1][count_col].mean()
                weekday_avg = df[df['is_weekend'] == 0][count_col].mean()
                
                if weekday_avg > 0:
                    patterns['weekend_ratio'] = float(weekend_avg / weekday_avg)
                    patterns['weekend_avg'] = float(weekend_avg)
                    patterns['weekday_avg'] = float(weekday_avg)
            except Exception as e:
                print(f"  ⚠ Weekend analysis failed: {e}")
        
        return patterns
    
    def print_pattern_summary(self, patterns: Dict, mode_name: str):
        """Print formatted pattern summary"""
        print(f"\n{mode_name.upper()} PATTERNS:")
        
        if 'peak_hour' in patterns:
            print(f"   Peak hour: {patterns['peak_hour']}:00 "
                  f"({patterns['peak_hour_avg']:.0f} avg)")
        
        if 'busiest_day' in patterns:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_name = day_names[patterns['busiest_day']]
            print(f"   Busiest day: {day_name} ({patterns['busiest_day_avg']:.0f} avg)")
        
        if 'seasonal_variation' in patterns:
            print(f"   Seasonal variation: {patterns['seasonal_variation']:.2f}x")
        
        if 'weekend_ratio' in patterns:
            print(f"   Weekend/Weekday ratio: {patterns['weekend_ratio']:.2f}")


# HEAT IMPACT ANALYZER

class HeatImpactAnalyzer:
    """Analyzes heat stress impacts on mobility"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def create_heat_mask(self, df: pd.DataFrame, 
                        threshold: Optional[float] = None) -> Tuple[pd.Series, str]:
        """
        Create heat condition mask based on available data.
        
        Priority:
        1. Use preprocessed flags (is_very_hot_day, is_hot_day)
        2. Use temperature column with threshold
        3. Fall back to season proxy (summer)
        
        Returns:
            (heat_mask, label_description)
        """
        threshold = threshold or self.config.HIGH_TEMP_THRESHOLD
        
        weather_status = DataValidator.check_weather_integration(df)
        
        if weather_status == 'preprocessed_flags':
            # Use pre-computed flags from preprocessing
            heat_mask = df['is_very_hot_day'] == 1
            label = f'Very Hot Days (≥{self.config.preprocessing_config.VERY_HOT_DAY_TEMP if self.config.preprocessing_config else 30}°C)'
            
        elif weather_status == 'temperature_column':
            # Use temperature column with threshold
            heat_mask = df['temperature'] >= threshold
            label = f'≥{threshold}°C'
            
        elif weather_status == 'season_proxy':
            # Fall back to season proxy
            heat_mask = df['season'] == 'summer'
            label = 'Summer (season proxy)'
            
        else:
            raise ValueError("No weather data available for heat analysis")
        
        return heat_mask, label
    
    def compare_heat_vs_normal(self, df: pd.DataFrame, 
                               count_col: str,
                               heat_mask: pd.Series) -> Dict:
        """
        Compare mobility during heat vs normal conditions.
        
        Returns comprehensive impact metrics.
        """
        # Calculate averages
        heat_avg = df[heat_mask][count_col].mean()
        normal_avg = df[~heat_mask][count_col].mean()
        
        # Impact ratio
        impact_ratio = heat_avg / normal_avg if normal_avg > 0 else np.nan
        pct_change = ((heat_avg - normal_avg) / normal_avg * 100) if normal_avg > 0 else np.nan
        
        # Statistical test (fixed tuple unpacking)
        heat_values = df[heat_mask][count_col].dropna()
        normal_values = df[~heat_mask][count_col].dropna()
        
        if len(heat_values) > 0 and len(normal_values) > 0:
            t_stat, p_value = stats.ttest_ind(heat_values, normal_values, nan_policy='omit')
            is_significant = p_value < self.config.SIGNIFICANCE_LEVEL
        else:
            t_stat, p_value, is_significant = np.nan, np.nan, False
        
        return {
            'heat_avg': float(heat_avg) if not np.isnan(heat_avg) else None,
            'normal_avg': float(normal_avg) if not np.isnan(normal_avg) else None,
            'impact_ratio': float(impact_ratio) if not np.isnan(impact_ratio) else None,
            'pct_change': float(pct_change) if not np.isnan(pct_change) else None,
            'heat_observations': int(heat_mask.sum()),
            'normal_observations': int((~heat_mask).sum()),
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'is_significant': is_significant
        }
    
    def analyze_heat_impact_by_time_period(self, df: pd.DataFrame,
                                           count_col: str,
                                           heat_mask: pd.Series) -> Dict:
        """Analyze heat impact broken down by time period"""
        
        if 'time_period' not in df.columns:
            return {}
        
        period_impacts = {}
        
        for period in df['time_period'].unique():
            if pd.isna(period):
                continue
            
            period_mask = df['time_period'] == period
            heat_period = df[heat_mask & period_mask][count_col].mean()
            normal_period = df[~heat_mask & period_mask][count_col].mean()
            
            if not np.isnan(heat_period) and not np.isnan(normal_period) and normal_period > 0:
                ratio = heat_period / normal_period
                period_impacts[period] = {
                    'heat_avg': float(heat_period),
                    'normal_avg': float(normal_period),
                    'ratio': float(ratio)
                }
        
        return period_impacts
    
    def identify_vulnerable_routes(self, df: pd.DataFrame,
                                   location_col: str,
                                   count_col: str,
                                   mode_name: str,
                                   percentile: Optional[float] = None) -> pd.DataFrame:
        """
        Identify routes most vulnerable to heat stress.
        
        Vulnerability = largest decrease in mobility during heat.
        """
        percentile = percentile or self.config.VULNERABLE_PERCENTILE
        
        # Create heat mask
        heat_mask, heat_label = self.create_heat_mask(df)
        
        route_impacts = []
        
        for location in df[location_col].unique():
            if pd.isna(location):
                continue
            
            loc_data = df[df[location_col] == location]
            
            heat_avg = loc_data[heat_mask][count_col].mean()
            normal_avg = loc_data[~heat_mask][count_col].mean()
            
            if not np.isnan(heat_avg) and not np.isnan(normal_avg) and normal_avg > 0:
                impact_ratio = heat_avg / normal_avg
                pct_change = ((heat_avg - normal_avg) / normal_avg) * 100
                
                route_impacts.append({
                    'location': location,
                    'heat_avg': heat_avg,
                    'normal_avg': normal_avg,
                    'impact_ratio': impact_ratio,
                    'pct_change': pct_change,
                    'mode': mode_name,
                    'heat_observations': int((loc_data.index.isin(df[heat_mask].index)).sum()),
                    'normal_observations': int((loc_data.index.isin(df[~heat_mask].index)).sum())
                })
        
        if not route_impacts:
            return pd.DataFrame()
        
        impact_df = pd.DataFrame(route_impacts).sort_values('impact_ratio')
        
        # Identify vulnerable routes (bottom percentile)
        threshold = impact_df['impact_ratio'].quantile(percentile / 100)
        impact_df['is_vulnerable'] = impact_df['impact_ratio'] <= threshold
        
        return impact_df


# SPATIAL ANALYZER

class SpatialAnalyzer:
    """Spatial analysis for ZTL proximity and route vulnerability"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Calculate Haversine distance between two points in meters.
        """
        R = 6371000  # Earth radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def find_nearest_ztl(self, traffic_df: pd.DataFrame,
                        ztl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find nearest ZTL zone for each traffic location.
        
        Requires latitude/longitude in both datasets.
        """
        if not all(col in traffic_df.columns for col in ['latitude', 'longitude']):
            raise ValueError("Traffic data missing coordinates")
        
        if not all(col in ztl_df.columns for col in ['latitude', 'longitude']):
            raise ValueError("ZTL data missing coordinates")
        
        traffic_df = traffic_df.copy()
        
        # Calculate minimum distance to any ZTL zone
        distances = []
        for _, traffic_row in traffic_df.iterrows():
            if pd.isna(traffic_row['latitude']) or pd.isna(traffic_row['longitude']):
                distances.append(np.inf)
                continue
            
            min_dist = np.inf
            for _, ztl_row in ztl_df.iterrows():
                if pd.isna(ztl_row['latitude']) or pd.isna(ztl_row['longitude']):
                    continue
                
                dist = self.calculate_distance(
                    traffic_row['latitude'], traffic_row['longitude'],
                    ztl_row['latitude'], ztl_row['longitude']
                )
                min_dist = min(min_dist, dist)
            
            distances.append(min_dist)
        
        traffic_df['distance_to_ztl'] = distances
        traffic_df['near_ztl'] = traffic_df['distance_to_ztl'] <= self.config.ZTL_PROXIMITY_METERS
        
        return traffic_df
    
    def analyze_ztl_proximity_impact(self, traffic_df: pd.DataFrame,
                                     ztl_df: pd.DataFrame,
                                     count_col: str = 'vehicle_count') -> Dict:
        """
        Compare traffic patterns near vs far from ZTL zones.
        """
        # Add proximity information
        traffic_with_ztl = self.find_nearest_ztl(traffic_df, ztl_df)
        
        # Compare near vs far
        near_avg = traffic_with_ztl[traffic_with_ztl['near_ztl']][count_col].mean()
        far_avg = traffic_with_ztl[~traffic_with_ztl['near_ztl']][count_col].mean()
        
        near_count = traffic_with_ztl['near_ztl'].sum()
        far_count = (~traffic_with_ztl['near_ztl']).sum()
        
        return {
            'near_ztl_avg': float(near_avg) if not np.isnan(near_avg) else None,
            'far_from_ztl_avg': float(far_avg) if not np.isnan(far_avg) else None,
            'ratio': float(near_avg / far_avg) if far_avg > 0 else None,
            'near_ztl_locations': int(near_count),
            'far_from_ztl_locations': int(far_count),
            'proximity_threshold_meters': self.config.ZTL_PROXIMITY_METERS
        }


# ANOMALY DETECTOR

class AnomalyDetector:
    """Detects anomalies in mobility data"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def detect_isolation_forest(self, df: pd.DataFrame, 
                                count_col: str,
                                contamination: Optional[float] = None) -> pd.DataFrame:
        """Isolation Forest anomaly detection"""
        
        contamination = contamination or self.config.ANOMALY_CONTAMINATION
        
        # Prepare features
        feature_cols = [count_col, 'hour', 'day_of_week', 'is_weekend']
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            raise ValueError("No features available for anomaly detection")
        
        df_result = df.copy()
        X = df_result[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Detect anomalies
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.config.ANOMALY_RANDOM_STATE,
            n_estimators=100
        )
        
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        df_result['anomaly'] = (anomaly_labels == -1).astype(int)
        df_result['anomaly_score'] = iso_forest.score_samples(X_scaled)
        
        return df_result
    
    def detect_zscore(self, df: pd.DataFrame, count_col: str, 
                     threshold: float = 3.0) -> pd.DataFrame:
        """Z-score based anomaly detection"""
        
        df_result = df.copy()
        z_scores = np.abs(stats.zscore(df_result[count_col].fillna(df_result[count_col].mean())))
        
        df_result['anomaly'] = (z_scores > threshold).astype(int)
        df_result['anomaly_score'] = -z_scores
        
        return df_result
    
    def detect_iqr(self, df: pd.DataFrame, count_col: str,
                  multiplier: float = 1.5) -> pd.DataFrame:
        """IQR-based anomaly detection"""
        
        df_result = df.copy()
        
        Q1 = df_result[count_col].quantile(0.25)
        Q3 = df_result[count_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_result['anomaly'] = (
            (df_result[count_col] < lower_bound) | 
            (df_result[count_col] > upper_bound)
        ).astype(int)
        df_result['anomaly_score'] = df_result[count_col]
        
        return df_result
    
    def characterize_anomalies(self, df: pd.DataFrame, count_col: str) -> Dict:
        """Analyze characteristics of detected anomalies"""
        
        if 'anomaly' not in df.columns:
            return {}
        
        anomalies = df[df['anomaly'] == 1]
        
        if len(anomalies) == 0:
            return {'count': 0}
        
        characteristics = {
            'count': len(anomalies),
            'percentage': (len(anomalies) / len(df)) * 100,
            'avg_value': float(anomalies[count_col].mean())
        }
        
        if 'hour' in anomalies.columns:
            characteristics['most_common_hour'] = int(anomalies['hour'].mode().values[0])
        
        if 'day_name' in anomalies.columns:
            characteristics['most_common_day'] = str(anomalies['day_name'].mode().values[0])
        
        if 'is_weekend' in anomalies.columns:
            weekend_count = anomalies['is_weekend'].sum()
            characteristics['weekend_anomalies'] = int(weekend_count)
            characteristics['weekend_pct'] = float((weekend_count / len(anomalies)) * 100)
        
        return characteristics


# MAIN ANALYZER (Orchestrator)

class TALEAPatternAnalyzer:
    """
    Main orchestrator for multi-modal mobility pattern and heat impact analysis.
    Delegates to specialized analyzers for modularity.
    """
    
    # Mode-specific column mappings
    MODE_CONFIG = {
        MobilityMode.BICYCLE: {
            'count_col': 'Totale',
            'location_col': 'Dispositivo conta-bici'
        },
        MobilityMode.PEDESTRIAN: {
            'count_col': 'Numero di visitatori',
            'location_col': 'route'
        },
        MobilityMode.TRAFFIC: {
            'count_col': 'vehicle_count',
            'location_col': 'VIA_SPIRA'
        }
    }
    
    def __init__(self, processed_data: Dict[str, pd.DataFrame],
                 config: Optional[AnalysisConfig] = None):
        """
        Initialize with validated processed datasets.
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary of processed DataFrames from preprocessing
        config : AnalysisConfig, optional
            Analysis configuration
        """
        if not processed_data:
            raise ValueError("processed_data dictionary cannot be empty")
        
        # Validate datasets
        validation_report = DataValidator.validate_all_datasets(processed_data)
        
        print("=" * 70)
        print("TALEA PATTERN ANALYZER - INITIALIZATION")
        print("=" * 70)
        
        failed = []
        for dataset_name, (is_valid, missing) in validation_report.items():
            if is_valid:
                print(f"✓ {dataset_name}: Valid")
            else:
                print(f"✗ {dataset_name}: Missing columns {missing}")
                failed.append(dataset_name)
        
        if failed:
            raise ValueError(f"Validation failed for datasets: {failed}")
        
        self.data = processed_data
        self.config = config or AnalysisConfig()
        
        # Initialize specialized analyzers
        self.pattern_analyzer = PatternAnalyzer(self.config)
        self.heat_analyzer = HeatImpactAnalyzer(self.config)
        self.spatial_analyzer = SpatialAnalyzer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        
        self.results = {}
        
        print(f"\nLoaded {len(processed_data)} datasets")
        print("=" * 70 + "\n")
    
    # PUBLIC API - Multi-Modal Analysis
    
    def analyze_multimodal_patterns(self) -> Dict:
        """Analyze patterns across all mobility modes"""
        
        print("=" * 70)
        print("MULTI-MODAL PATTERN ANALYSIS")
        print("=" * 70)
        
        results = {}
        
        for mode in [MobilityMode.BICYCLE, MobilityMode.PEDESTRIAN, MobilityMode.TRAFFIC]:
            dataset_name = mode.value
            
            if dataset_name not in self.data:
                continue
            
            config = self.MODE_CONFIG[mode]
            df = self.data[dataset_name]
            
            patterns = self.pattern_analyzer.analyze_temporal_patterns(
                df, config['count_col'], dataset_name
            )
            
            self.pattern_analyzer.print_pattern_summary(patterns, dataset_name)
            results[dataset_name] = patterns
        
        self.results['multimodal_patterns'] = results
        return results
    
    def analyze_heat_impact_multimodal(self, 
                                       high_temp_threshold: Optional[float] = None) -> Dict:
        """Analyze heat impact across all mobility modes"""
        
        print("\n" + "=" * 70)
        print("MULTI-MODAL HEAT IMPACT ANALYSIS")
        print("=" * 70)
        
        results = {}
        
        for mode in [MobilityMode.BICYCLE, MobilityMode.PEDESTRIAN, MobilityMode.TRAFFIC]:
            dataset_name = mode.value
            
            if dataset_name not in self.data:
                continue
            
            print(f"\n{dataset_name.upper()} Heat Impact:")
            
            config = self.MODE_CONFIG[mode]
            df = self.data[dataset_name]
            
            try:
                # Create heat mask
                heat_mask, heat_label = self.heat_analyzer.create_heat_mask(
                    df, high_temp_threshold
                )
                
                print(f"  Condition: {heat_label}")
                
                # Overall impact
                impact = self.heat_analyzer.compare_heat_vs_normal(
                    df, config['count_col'], heat_mask
                )
                
                print(f"  Heat avg: {impact['heat_avg']:.0f}")
                print(f"  Normal avg: {impact['normal_avg']:.0f}")
                print(f"  Impact ratio: {impact['impact_ratio']:.2f}x")
                print(f"  Change: {impact['pct_change']:+.1f}%")
                print(f"  Significance: p={impact['p_value']:.4f} "
                      f"{'✓ Significant' if impact['is_significant'] else '✗ Not significant'}")
                
                # Impact by time period
                period_impacts = self.heat_analyzer.analyze_heat_impact_by_time_period(
                    df, config['count_col'], heat_mask
                )
                
                if period_impacts:
                    print(f"\n  Impact by Time Period:")
                    for period, metrics in period_impacts.items():
                        print(f"    {period:15s}: {metrics['ratio']:.2f}x")
                
                results[dataset_name] = {
                    'overall': impact,
                    'by_period': period_impacts,
                    'heat_label': heat_label
                }
                
            except Exception as e:
                print(f"  ⚠ Analysis failed: {e}")
                results[dataset_name] = {'error': str(e)}
        
        self.results['heat_impact'] = results
        return results
    
    def identify_vulnerable_routes_multimodal(self,
                                              threshold_percentile: Optional[float] = None) -> Dict:
        """Identify vulnerable routes across all modes"""
        
        print("\n" + "=" * 70)
        print("VULNERABLE ROUTES IDENTIFICATION (MULTI-MODAL)")
        print("=" * 70)
        
        results = {}
        
        for mode in [MobilityMode.BICYCLE, MobilityMode.PEDESTRIAN, MobilityMode.TRAFFIC]:
            dataset_name = mode.value
            
            if dataset_name not in self.data:
                continue
            
            config = self.MODE_CONFIG[mode]
            df = self.data[dataset_name]
            
            if config['location_col'] not in df.columns:
                print(f"\n{dataset_name.upper()}: No location column")
                continue
            
            print(f"\n{dataset_name.upper()} Vulnerable Routes:")
            
            try:
                vulnerable_df = self.heat_analyzer.identify_vulnerable_routes(
                    df, config['location_col'], config['count_col'],
                    dataset_name, threshold_percentile
                )
                
                if len(vulnerable_df) > 0:
                    vulnerable = vulnerable_df[vulnerable_df['is_vulnerable']]
                    print(f"  Total routes: {len(vulnerable_df)}")
                    print(f"  Vulnerable: {len(vulnerable)}")
                    print(f"\n  Top 5 Most Vulnerable:")
                    for idx, row in vulnerable.head(5).iterrows():
                        print(f"    {row['location'][:40]:40s}: {row['pct_change']:+6.1f}%")
                    
                    results[dataset_name] = vulnerable_df
                else:
                    print(f"  No routes found")
                    
            except Exception as e:
                print(f"  ⚠ Analysis failed: {e}")
        
        self.results['vulnerable_routes'] = results
        return results
    
    def analyze_origin_destination_flows(self) -> Optional[pd.DataFrame]:
        """Analyze pedestrian OD flows"""
        
        if 'pedestrian' not in self.data:
            print("⚠ No pedestrian data available")
            return None
        
        print("\n" + "=" * 70)
        print("ORIGIN-DESTINATION FLOW ANALYSIS")
        print("=" * 70)
        
        df = self.data['pedestrian']
        
        if not all(col in df.columns for col in ['Area provenienza', 'Area Arrivo']):
            print("⚠ Missing OD columns")
            return None
        
        # Create flow matrix
        flow_matrix = df.groupby(['Area provenienza', 'Area Arrivo']).agg({
            'Numero di visitatori': ['sum', 'mean', 'count']
        }).reset_index()
        
        flow_matrix.columns = ['Origin', 'Destination', 'Total_Flow', 'Avg_Flow', 'Observations']
        flow_matrix = flow_matrix.sort_values('Total_Flow', ascending=False)
        
        print(f"\nFlow Statistics:")
        print(f"  Total OD pairs: {len(flow_matrix)}")
        print(f"  Total pedestrians: {flow_matrix['Total_Flow'].sum():,.0f}")
        
        print(f"\nTop 10 Busiest Routes:")
        for idx, row in flow_matrix.head(10).iterrows():
            print(f"  {row['Origin'][:30]:30s} → {row['Destination'][:30]:30s}: {row['Total_Flow']:8,.0f}")
        
        self.results['flow_matrix'] = flow_matrix
        return flow_matrix
    
    def analyze_ztl_effectiveness(self) -> Optional[Dict]:
        """Analyze ZTL effectiveness with spatial analysis"""
        
        if 'ztl' not in self.data:
            print("⚠ No ZTL data available")
            return None
        
        print("\n" + "=" * 70)
        print("ZTL ZONE EFFECTIVENESS ANALYSIS")
        print("=" * 70)
        
        results = {}
        
        # Pedestrian activity in ZTL zones
        if 'pedestrian' in self.data:
            print("\n1. Pedestrian Activity:")
            
            ped_data = self.data['pedestrian']
            
            if 'season' in ped_data.columns:
                seasonal = ped_data.groupby('season')['Numero di visitatori'].mean()
                results['seasonal_pedestrian'] = seasonal.to_dict()
                
                print(f"\n  Seasonal Pedestrian Flow:")
                for season, avg in seasonal.items():
                    print(f"    {season.capitalize():10s}: {avg:8,.0f} avg")
            
            if 'is_weekend' in ped_data.columns:
                weekend_avg = ped_data[ped_data['is_weekend']==1]['Numero di visitatori'].mean()
                weekday_avg = ped_data[ped_data['is_weekend']==0]['Numero di visitatori'].mean()
                
                print(f"\n  Weekend vs Weekday:")
                print(f"    Weekend: {weekend_avg:,.0f}")
                print(f"    Weekday: {weekday_avg:,.0f}")
                print(f"    Ratio: {weekend_avg/weekday_avg:.2f}x")
                
                results['weekend_ratio'] = weekend_avg / weekday_avg
        
        # Spatial analysis of traffic near ZTL
        if 'traffic' in self.data and all(col in self.data['traffic'].columns for col in ['latitude', 'longitude']):
            print("\n2. Traffic Near ZTL Zones (Spatial Analysis):")
            
            try:
                proximity_analysis = self.spatial_analyzer.analyze_ztl_proximity_impact(
                    self.data['traffic'], self.data['ztl']
                )
                
                print(f"\n  Proximity Threshold: {proximity_analysis['proximity_threshold_meters']}m")
                print(f"  Near ZTL avg: {proximity_analysis['near_ztl_avg']:,.0f} vehicles")
                print(f"  Far from ZTL avg: {proximity_analysis['far_from_ztl_avg']:,.0f} vehicles")
                print(f"  Ratio: {proximity_analysis['ratio']:.2f}x")
                
                results['traffic_proximity'] = proximity_analysis
                
            except Exception as e:
                print(f"  ⚠ Spatial analysis failed: {e}")
        
        # ZTL characteristics
        print("\n3. ZTL Zone Characteristics:")
        ztl = self.data['ztl']
        
        if 'TipoZTL' in ztl.columns:
            print(f"  Zone types: {ztl['TipoZTL'].nunique()}")
            for zone_type in ztl['TipoZTL'].unique():
                count = (ztl['TipoZTL'] == zone_type).sum()
                print(f"    {zone_type}: {count} zones")
        
        if 'area' in ztl.columns:
            total_area = ztl['area'].sum()
            print(f"  Total ZTL area: {total_area:,.0f} m²")
        
        self.results['ztl_effectiveness'] = results
        return results
    
    def detect_anomalies(self, dataset_name: str = 'bicycle',
                        method: str = 'isolation_forest',
                        contamination: Optional[float] = None) -> pd.DataFrame:
        """
        Detect anomalies in mobility data.
        
        Parameters:
        -----------
        dataset_name : str
            'bicycle', 'pedestrian', or 'traffic'
        method : str
            'isolation_forest', 'zscore', or 'iqr'
        contamination : float, optional
            Proportion of outliers (for isolation_forest)
        """
        print("\n" + "=" * 70)
        print(f"ANOMALY DETECTION: {dataset_name.upper()}")
        print("=" * 70)
        
        if dataset_name not in self.data:
            raise ValueError(f"Dataset '{dataset_name}' not available")
        
        try:
            mode = MobilityMode(dataset_name)
        except ValueError:
            raise ValueError(f"Unknown mode: {dataset_name}")
        
        config = self.MODE_CONFIG[mode]
        df = self.data[dataset_name]
        
        # Detect anomalies
        if method == 'isolation_forest':
            df_with_anomalies = self.anomaly_detector.detect_isolation_forest(
                df, config['count_col'], contamination
            )
        elif method == 'zscore':
            df_with_anomalies = self.anomaly_detector.detect_zscore(
                df, config['count_col']
            )
        elif method == 'iqr':
            df_with_anomalies = self.anomaly_detector.detect_iqr(
                df, config['count_col']
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Characterize anomalies
        characteristics = self.anomaly_detector.characterize_anomalies(
            df_with_anomalies, config['count_col']
        )
        
        print(f"\nMethod: {method}")
        print(f"Anomalies: {characteristics.get('count', 0):,} ({characteristics.get('percentage', 0):.2f}%)")
        
        if characteristics.get('count', 0) > 0:
            print(f"\nAnomaly Characteristics:")
            print(f"  Avg value: {characteristics.get('avg_value', 'N/A'):.0f}")
            if 'most_common_hour' in characteristics:
                print(f"  Most common hour: {characteristics['most_common_hour']}")
            if 'most_common_day' in characteristics:
                print(f"  Most common day: {characteristics['most_common_day']}")
            if 'weekend_pct' in characteristics:
                print(f"  Weekend anomalies: {characteristics['weekend_pct']:.1f}%")
        
        self.results[f'{dataset_name}_anomalies'] = df_with_anomalies[df_with_anomalies['anomaly'] == 1]
        return df_with_anomalies
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        
        print("\n" + "=" * 70)
        print("TALEA PATTERN ANALYSIS - SUMMARY REPORT")
        print("=" * 70)
        
        summary = {}
        
        for dataset_name, df in self.data.items():
            if dataset_name not in ['bicycle', 'pedestrian', 'traffic']:
                continue
            
            mode = MobilityMode(dataset_name)
            config = self.MODE_CONFIG[mode]
            count_col = config['count_col']
            
            print(f"\n{dataset_name.upper()} Summary:")
            print(f"  Total observations: {len(df):,}")
            
            # Date range
            date_cols = ['Data', 'datetime', 'data']
            date_col = next((col for col in date_cols if col in df.columns), None)
            if date_col:
                print(f"  Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
            
            # Statistics
            if count_col in df.columns:
                print(f"  Average: {df[count_col].mean():.0f}")
                print(f"  Median: {df[count_col].median():.0f}")
                print(f"  Total: {df[count_col].sum():,.0f}")
            
            # Weather integration status
            weather_status = DataValidator.check_weather_integration(df)
            print(f"  Weather integration: {weather_status}")
            
            summary[dataset_name] = {
                'observations': len(df),
                'avg': float(df[count_col].mean()) if count_col in df.columns else None,
                'total': float(df[count_col].sum()) if count_col in df.columns else None,
                'weather_status': weather_status
            }
        
        print("\n" + "=" * 70)
        return summary


# USAGE GUIDE

if __name__ == "__main__":
    print("TALEA Heat-Stress Impact Analysis Module")
    print("\nExample workflow:")
    print("1. config = AnalysisConfig(HIGH_TEMP_THRESHOLD=32.0)")
    print("2. analyzer = TALEAPatternAnalyzer(processed_data, config)")
    print("3. analyzer.analyze_multimodal_patterns()")
    print("4. analyzer.analyze_heat_impact_multimodal()")
    print("5. analyzer.identify_vulnerable_routes_multimodal()")
    print("6. analyzer.analyze_ztl_effectiveness()  # with spatial analysis")
    print("7. analyzer.detect_anomalies('bicycle', method='isolation_forest')")
    print("8. analyzer.generate_summary_report()")