"""
TALEA - Civic Digital Twin: Mobility Impact Analyzer
File: src/analysis/heat_impact/mobility_impact_analyzer.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Analyzes heat impact on bicycle and pedestrian mobility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from dataclasses import dataclass


@dataclass
class MobilityImpactConfig:
    """Configuration for mobility impact analysis"""
    
    # Heat thresholds
    HIGH_TEMP: float = 30.0
    MODERATE_TEMP: float = 25.0
    
    # Statistical significance
    SIGNIFICANCE_LEVEL: float = 0.05
    
    # Demand reduction factors
    HEAT_REDUCTION_FACTOR: float = 0.15  # 15% reduction in extreme heat
    
    # Time-dependent analysis
    TIME_BINS: int = 48  # For 30-minute intervals in 24 hours


class MobilityImpactAnalyzer:
    """Analyzes heat stress impact on mobility patterns"""
    
    def __init__(self, config: Optional[MobilityImpactConfig] = None):
        """
        Initialize analyzer
        
        Args:
            config: Configuration for analysis
        """
        self.config = config or MobilityImpactConfig()
    
    def analyze_bicycle_heat_correlation(self,
                                        bicycle_df: pd.DataFrame,
                                        weather_df: pd.DataFrame,
                                        count_col: str = 'Totale',
                                        temp_col: str = 'temperature') -> Dict:
        """
        Analyze correlation between bicycle usage and heat
        
        Args:
            bicycle_df: Bicycle counter data (must have datetime)
            weather_df: Weather data with temperature
            count_col: Bicycle count column
            temp_col: Temperature column
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Merge datasets
        merged = self._merge_mobility_weather(bicycle_df, weather_df)
        
        if temp_col not in merged.columns or count_col not in merged.columns:
            raise ValueError(f"Missing required columns: {temp_col}, {count_col}")
        
        # Remove NaN values
        clean = merged[[temp_col, count_col]].dropna()
        
        # Compute correlation
        correlation, p_value = stats.pearsonr(clean[temp_col], clean[count_col])
        
        # Heat categories
        merged['temp_category'] = pd.cut(
            merged[temp_col],
            bins=[-np.inf, self.config.MODERATE_TEMP, 
                  self.config.HIGH_TEMP, np.inf],
            labels=['comfortable', 'moderate_heat', 'high_heat']
        )
        
        # Average usage by heat category
        usage_by_temp = merged.groupby('temp_category')[count_col].agg([
            'mean', 'std', 'count'
        ])
        
        # Compare high heat vs comfortable
        comfortable_avg = usage_by_temp.loc['comfortable', 'mean']
        high_heat_avg = usage_by_temp.loc['high_heat', 'mean']
        
        impact_ratio = high_heat_avg / comfortable_avg if comfortable_avg > 0 else np.nan
        pct_change = ((high_heat_avg - comfortable_avg) / comfortable_avg * 100) if comfortable_avg > 0 else np.nan
        
        # T-test
        comfortable_data = merged[merged['temp_category'] == 'comfortable'][count_col].dropna()
        high_heat_data = merged[merged['temp_category'] == 'high_heat'][count_col].dropna()
        
        if len(comfortable_data) > 0 and len(high_heat_data) > 0:
            t_stat, t_pvalue = stats.ttest_ind(comfortable_data, high_heat_data)
        else:
            t_stat, t_pvalue = np.nan, np.nan
        
        results = {
            'correlation': float(correlation),
            'correlation_pvalue': float(p_value),
            'is_significant': p_value < self.config.SIGNIFICANCE_LEVEL,
            'comfortable_avg': float(comfortable_avg),
            'high_heat_avg': float(high_heat_avg),
            'impact_ratio': float(impact_ratio) if not np.isnan(impact_ratio) else None,
            'pct_change': float(pct_change) if not np.isnan(pct_change) else None,
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            't_pvalue': float(t_pvalue) if not np.isnan(t_pvalue) else None,
            'usage_by_temp_category': usage_by_temp.to_dict()
        }
        
        print(f"  ✓ Bicycle-heat correlation: r={correlation:.3f}, p={p_value:.4f}")
        print(f"    Impact: {pct_change:+.1f}% in high heat")
        
        return results
    
    def analyze_time_dependent_correlation(self,
                                          mobility_df: pd.DataFrame,
                                          weather_df: pd.DataFrame,
                                          count_col: str = 'Totale',
                                          temp_col: str = 'temperature') -> Dict:
        """
        Analyze correlation accounting for time-of-day effects.
        Based on learning conditional probability f(mobility|temp,time) rather than f(mobility,temp).
        
        This avoids spurious correlations where both mobility and temperature 
        are influenced by time of day.
        
        Args:
            mobility_df: Mobility data with datetime
            weather_df: Weather data with temperature
            count_col: Mobility count column
            temp_col: Temperature column
            
        Returns:
            Dictionary with time-dependent correlation analysis
        """
        # Merge datasets
        merged = self._merge_mobility_weather(mobility_df, weather_df)
        
        if temp_col not in merged.columns or count_col not in merged.columns:
            raise ValueError(f"Missing required columns")
        
        # Extract time of day (fractional hours)
        merged = self._add_time_features(merged)
        
        # Analyze correlation within each time slot
        time_slot_correlations = []
        
        # Group by hour to control for time-of-day effects
        for hour in sorted(merged['hour'].unique()):
            hour_data = merged[merged['hour'] == hour]
            
            clean = hour_data[[temp_col, count_col]].dropna()
            
            if len(clean) > 10:  # Need sufficient samples
                corr, p_val = stats.pearsonr(clean[temp_col], clean[count_col])
                
                time_slot_correlations.append({
                    'hour': int(hour),
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n_samples': len(clean),
                    'is_significant': p_val < self.config.SIGNIFICANCE_LEVEL
                })
        
        # Overall time-controlled correlation (average of within-slot correlations)
        if time_slot_correlations:
            avg_correlation = np.mean([tc['correlation'] for tc in time_slot_correlations])
            significant_slots = sum(1 for tc in time_slot_correlations if tc['is_significant'])
        else:
            avg_correlation = np.nan
            significant_slots = 0
        
        # Compare with naive correlation (not controlling for time)
        clean_all = merged[[temp_col, count_col]].dropna()
        naive_corr, naive_p = stats.pearsonr(clean_all[temp_col], clean_all[count_col])
        
        results = {
            'time_controlled_correlation': float(avg_correlation) if not np.isnan(avg_correlation) else None,
            'naive_correlation': float(naive_corr),
            'naive_p_value': float(naive_p),
            'significant_time_slots': significant_slots,
            'total_time_slots': len(time_slot_correlations),
            'by_time_slot': time_slot_correlations,
            'interpretation': self._interpret_time_dependent_results(avg_correlation, naive_corr)
        }
        
        print(f"  ✓ Time-controlled correlation: {avg_correlation:.3f}")
        print(f"    Naive correlation: {naive_corr:.3f}")
        print(f"    Significant slots: {significant_slots}/{len(time_slot_correlations)}")
        
        return results
    
    def _interpret_time_dependent_results(self, 
                                         controlled_corr: float, 
                                         naive_corr: float) -> str:
        """Interpret the difference between controlled and naive correlations"""
        
        if np.isnan(controlled_corr) or np.isnan(naive_corr):
            return "insufficient_data"
        
        diff = abs(controlled_corr - naive_corr)
        
        if diff < 0.1:
            return "time_effect_minimal"
        elif diff < 0.3:
            return "moderate_time_confounding"
        else:
            return "strong_time_confounding"
    
    def analyze_pedestrian_heat_correlation(self,
                                           pedestrian_df: pd.DataFrame,
                                           weather_df: pd.DataFrame,
                                           count_col: str = 'Numero di visitatori',
                                           temp_col: str = 'temperature') -> Dict:
        """
        Analyze correlation between pedestrian flow and heat
        
        Args:
            pedestrian_df: Pedestrian flow data
            weather_df: Weather data with temperature
            count_col: Pedestrian count column
            temp_col: Temperature column
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Merge datasets
        merged = self._merge_mobility_weather(pedestrian_df, weather_df)
        
        if temp_col not in merged.columns or count_col not in merged.columns:
            raise ValueError(f"Missing required columns: {temp_col}, {count_col}")
        
        # Remove NaN values
        clean = merged[[temp_col, count_col]].dropna()
        
        # Compute correlation
        correlation, p_value = stats.pearsonr(clean[temp_col], clean[count_col])
        
        # Heat categories
        merged['temp_category'] = pd.cut(
            merged[temp_col],
            bins=[-np.inf, self.config.MODERATE_TEMP, 
                  self.config.HIGH_TEMP, np.inf],
            labels=['comfortable', 'moderate_heat', 'high_heat']
        )
        
        # Average usage by heat category
        usage_by_temp = merged.groupby('temp_category')[count_col].agg([
            'mean', 'std', 'count'
        ])
        
        # Compare high heat vs comfortable
        comfortable_avg = usage_by_temp.loc['comfortable', 'mean']
        high_heat_avg = usage_by_temp.loc['high_heat', 'mean']
        
        impact_ratio = high_heat_avg / comfortable_avg if comfortable_avg > 0 else np.nan
        pct_change = ((high_heat_avg - comfortable_avg) / comfortable_avg * 100) if comfortable_avg > 0 else np.nan
        
        results = {
            'correlation': float(correlation),
            'correlation_pvalue': float(p_value),
            'is_significant': p_value < self.config.SIGNIFICANCE_LEVEL,
            'comfortable_avg': float(comfortable_avg),
            'high_heat_avg': float(high_heat_avg),
            'impact_ratio': float(impact_ratio) if not np.isnan(impact_ratio) else None,
            'pct_change': float(pct_change) if not np.isnan(pct_change) else None,
            'usage_by_temp_category': usage_by_temp.to_dict()
        }
        
        print(f"  ✓ Pedestrian-heat correlation: r={correlation:.3f}, p={p_value:.4f}")
        print(f"    Impact: {pct_change:+.1f}% in high heat")
        
        return results
    
    def compute_modal_shift_probabilities(self,
                                         mobility_df: pd.DataFrame,
                                         heat_stress_levels: pd.Series,
                                         mode_col: str = 'mode',
                                         count_col: str = 'count') -> pd.DataFrame:
        """
        Compute probabilities of modal shift under heat stress
        
        Args:
            mobility_df: Multi-modal mobility data
            heat_stress_levels: Heat stress level classifications
            mode_col: Column indicating transport mode
            count_col: Count/volume column
            
        Returns:
            DataFrame with modal shift probabilities
        """
        df = mobility_df.copy()
        df['heat_level'] = heat_stress_levels
        
        # Group by mode and heat level
        modal_by_heat = df.groupby(['heat_level', mode_col])[count_col].sum().unstack(fill_value=0)
        
        # Calculate proportions
        modal_proportions = modal_by_heat.div(modal_by_heat.sum(axis=1), axis=0)
        
        # Calculate shift probabilities (change from comfortable conditions)
        if 'comfortable' in modal_proportions.index:
            baseline = modal_proportions.loc['comfortable']
            
            shift_probs = {}
            for heat_level in modal_proportions.index:
                if heat_level != 'comfortable':
                    shift = modal_proportions.loc[heat_level] - baseline
                    shift_probs[heat_level] = shift.to_dict()
        else:
            shift_probs = {}
        
        print(f"  ✓ Modal shift probabilities computed for {len(shift_probs)} heat levels")
        
        return modal_proportions, shift_probs
    
    def estimate_demand_reduction(self,
                                  base_demand: Union[float, pd.Series],
                                  heat_stress_level: Union[str, pd.Series],
                                  mode: str = 'bicycle') -> Union[float, pd.Series]:
        """
        Estimate demand reduction due to heat stress
        
        Args:
            base_demand: Baseline demand values
            heat_stress_level: Heat stress level classification
            mode: Transport mode ('bicycle', 'pedestrian', 'transit')
            
        Returns:
            Adjusted demand values
        """
        # Reduction factors by heat level and mode
        reduction_factors = {
            'bicycle': {
                'low': 1.0,
                'moderate': 0.95,
                'high': 0.85,
                'extreme': 0.70
            },
            'pedestrian': {
                'low': 1.0,
                'moderate': 0.90,
                'high': 0.75,
                'extreme': 0.60
            },
            'transit': {
                'low': 1.0,
                'moderate': 1.05,  # Slight increase as people avoid active modes
                'high': 1.10,
                'extreme': 1.15
            }
        }
        
        mode_factors = reduction_factors.get(mode, reduction_factors['bicycle'])
        
        # Apply reduction
        if isinstance(heat_stress_level, pd.Series):
            factors = heat_stress_level.map(mode_factors).fillna(1.0)
            adjusted_demand = base_demand * factors
        else:
            factor = mode_factors.get(heat_stress_level, 1.0)
            adjusted_demand = base_demand * factor
        
        return adjusted_demand
    
    def analyze_heat_impact_by_location(self,
                                       mobility_df: pd.DataFrame,
                                       location_col: str,
                                       count_col: str,
                                       temp_col: str = 'temperature') -> pd.DataFrame:
        """
        Analyze heat impact by location/route
        
        Args:
            mobility_df: Mobility data with location and temperature
            location_col: Location identifier column
            count_col: Count column
            temp_col: Temperature column
            
        Returns:
            DataFrame with location-specific heat impacts
        """
        if temp_col not in mobility_df.columns:
            raise ValueError(f"Missing temperature column: {temp_col}")
        
        # Create heat categories
        df = mobility_df.copy()
        df['is_high_heat'] = df[temp_col] >= self.config.HIGH_TEMP
        
        location_impacts = []
        
        for location in df[location_col].unique():
            if pd.isna(location):
                continue
            
            loc_data = df[df[location_col] == location]
            
            # Normal and high heat averages
            normal_avg = loc_data[~loc_data['is_high_heat']][count_col].mean()
            heat_avg = loc_data[loc_data['is_high_heat']][count_col].mean()
            
            if not np.isnan(normal_avg) and not np.isnan(heat_avg) and normal_avg > 0:
                impact_ratio = heat_avg / normal_avg
                pct_change = ((heat_avg - normal_avg) / normal_avg) * 100
                
                location_impacts.append({
                    'location': location,
                    'normal_avg': normal_avg,
                    'heat_avg': heat_avg,
                    'impact_ratio': impact_ratio,
                    'pct_change': pct_change,
                    'observations': len(loc_data),
                    'heat_observations': loc_data['is_high_heat'].sum()
                })
        
        impact_df = pd.DataFrame(location_impacts).sort_values('impact_ratio')
        
        print(f"  ✓ Analyzed {len(impact_df)} locations")
        
        return impact_df
    
    def analyze_window_based_impact(self,
                                   mobility_series: pd.Series,
                                   temperature_series: pd.Series,
                                   window_length: int = 24) -> Dict:
        """
        Analyze heat impact using rolling windows to capture delayed effects.
        Based on PDF concepts: windows capture temporal dependencies.
        
        Args:
            mobility_series: Time series of mobility counts
            temperature_series: Time series of temperature
            window_length: Rolling window size (e.g., 24 hours)
            
        Returns:
            Dictionary with window-based analysis
        """
        # Align series
        common_idx = mobility_series.index.intersection(temperature_series.index)
        mobility = mobility_series.loc[common_idx]
        temperature = temperature_series.loc[common_idx]
        
        # Compute rolling features
        mobility_ma = mobility.rolling(window=window_length, min_periods=1).mean()
        temp_ma = temperature.rolling(window=window_length, min_periods=1).mean()
        
        # Correlation between smoothed series
        clean_data = pd.DataFrame({
            'mobility_ma': mobility_ma,
            'temp_ma': temp_ma
        }).dropna()
        
        if len(clean_data) > 10:
            corr, p_val = stats.pearsonr(clean_data['mobility_ma'], clean_data['temp_ma'])
        else:
            corr, p_val = np.nan, np.nan
        
        # Identify heat waves (sustained high temperature)
        heat_wave_threshold = temperature.quantile(0.9)
        heat_waves = (temp_ma > heat_wave_threshold)
        
        # Mobility during heat waves
        if heat_waves.sum() > 0:
            mobility_normal = mobility[~heat_waves].mean()
            mobility_heatwave = mobility[heat_waves].mean()
            heatwave_impact = ((mobility_heatwave - mobility_normal) / mobility_normal * 100) if mobility_normal > 0 else np.nan
        else:
            mobility_normal = mobility.mean()
            mobility_heatwave = np.nan
            heatwave_impact = np.nan
        
        results = {
            'window_length': window_length,
            'smoothed_correlation': float(corr) if not np.isnan(corr) else None,
            'smoothed_p_value': float(p_val) if not np.isnan(p_val) else None,
            'heat_wave_days': int(heat_waves.sum()),
            'mobility_normal': float(mobility_normal),
            'mobility_heatwave': float(mobility_heatwave) if not np.isnan(mobility_heatwave) else None,
            'heatwave_impact_pct': float(heatwave_impact) if not np.isnan(heatwave_impact) else None
        }
        
        print(f"  ✓ Window-based analysis (window={window_length})")
        print(f"    Smoothed correlation: {corr:.3f}")
        print(f"    Heat wave impact: {heatwave_impact:+.1f}%" if not np.isnan(heatwave_impact) else "    No heat waves detected")
        
        return results
    
    def _merge_mobility_weather(self,
                               mobility_df: pd.DataFrame,
                               weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge mobility and weather data on datetime"""
        
        # Identify datetime columns
        mob_date_col = None
        for col in ['datetime', 'Data', 'data']:
            if col in mobility_df.columns:
                mob_date_col = col
                break
        
        weather_date_col = None
        for col in ['datetime', 'Data', 'data']:
            if col in weather_df.columns:
                weather_date_col = col
                break
        
        if mob_date_col is None or weather_date_col is None:
            raise ValueError("Both dataframes must have datetime column")
        
        # Ensure datetime format
        mobility = mobility_df.copy()
        weather = weather_df.copy()
        
        mobility[mob_date_col] = (
            pd.to_datetime(mobility[mob_date_col], utc=True)
            .dt.tz_convert(None)
        )

        weather[weather_date_col] = (
            pd.to_datetime(weather[weather_date_col], utc=True)
            .dt.tz_convert(None)
        )
        
        # Create merge key (date only for daily data, datetime for hourly)
        mobility['merge_key'] = mobility[mob_date_col].dt.date
        weather['merge_key'] = weather[weather_date_col].dt.date
        
        # Merge
        merged = mobility.merge(weather, on='merge_key', how='left', suffixes=('', '_weather'))
        
        return merged
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-of-day features"""
        
        df = df.copy()
        
        # Find datetime column
        date_col = None
        for col in ['datetime', 'Data', 'data']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            if 'hour' not in df.columns:
                df['hour'] = df[date_col].dt.hour
            if 'minute' not in df.columns:
                df['minute'] = df[date_col].dt.minute
        
        return df