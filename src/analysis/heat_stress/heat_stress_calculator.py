"""
TALEA - Civic Digital Twin: Heat Stress Calculator
File: src/analysis/heat_stress/heat_stress_calculator.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Computes various heat stress indices and classification levels.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class HeatStressThresholds:
    """Thresholds for heat stress classification"""
    
    # Heat Index thresholds (°C)
    HI_CAUTION: float = 27.0
    HI_EXTREME_CAUTION: float = 32.0
    HI_DANGER: float = 39.0
    HI_EXTREME_DANGER: float = 51.0
    
    # WBGT thresholds (°C)
    WBGT_LOW: float = 18.0
    WBGT_MODERATE: float = 23.0
    WBGT_HIGH: float = 28.0
    WBGT_EXTREME: float = 32.0
    
    # UTCI thresholds (°C)
    UTCI_COLD: float = 9.0
    UTCI_COMFORTABLE: float = 26.0
    UTCI_MODERATE_HEAT: float = 32.0
    UTCI_STRONG_HEAT: float = 38.0


class HeatStressCalculator:
    """Calculates heat stress indices and classification levels"""
    
    def __init__(self, thresholds: Optional[HeatStressThresholds] = None):
        """
        Initialize calculator
        
        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or HeatStressThresholds()
        self._uncertainty_estimates = {}  # Store uncertainty for predictions
    
    def compute_heat_index(self, 
                          temperature: Union[float, pd.Series], 
                          humidity: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Compute Heat Index using Steadman formula
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity (0-100)
            
        Returns:
            Heat index in Celsius
        """
        # Convert to Fahrenheit for calculation
        temp_f = temperature * 9/5 + 32
        rh = humidity
        
        # Simplified Heat Index formula (Steadman)
        hi_f = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
        
        # For higher temperatures, use Rothfusz regression
        if isinstance(temp_f, pd.Series):
            high_temp_mask = temp_f >= 80
            if high_temp_mask.any():
                hi_f_complex = self._rothfusz_regression(temp_f[high_temp_mask], rh[high_temp_mask])
                hi_f.loc[high_temp_mask] = hi_f_complex
        elif temp_f >= 80:
            hi_f = self._rothfusz_regression(temp_f, rh)
        
        # Convert back to Celsius
        hi_c = (hi_f - 32) * 5/9
        
        return hi_c
    
    def compute_heat_index_with_uncertainty(self,
                                           temperature: pd.Series,
                                           humidity: pd.Series,
                                           temp_uncertainty: Optional[pd.Series] = None,
                                           humidity_uncertainty: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
        """
        Compute Heat Index with uncertainty propagation.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity (0-100)
            temp_uncertainty: Standard deviation of temperature measurements
            humidity_uncertainty: Standard deviation of humidity measurements
            
        Returns:
            Tuple of (heat_index_mean, heat_index_std)
        """
        # Compute nominal heat index
        hi_mean = self.compute_heat_index(temperature, humidity)
        
        # If no uncertainty provided, estimate from typical sensor errors
        if temp_uncertainty is None:
            temp_uncertainty = pd.Series(0.5, index=temperature.index)  # ±0.5°C typical
        
        if humidity_uncertainty is None:
            humidity_uncertainty = pd.Series(3.0, index=humidity.index)  # ±3% typical
        
        # Propagate uncertainty using linear approximation (first-order Taylor)
        # ΔHI ≈ (∂HI/∂T) * ΔT + (∂HI/∂H) * ΔH
        
        # Numerical derivatives
        delta = 0.1
        
        # Partial derivative w.r.t. temperature
        hi_temp_plus = self.compute_heat_index(temperature + delta, humidity)
        dhi_dtemp = (hi_temp_plus - hi_mean) / delta
        
        # Partial derivative w.r.t. humidity
        hi_humid_plus = self.compute_heat_index(temperature, humidity + delta)
        dhi_dhumid = (hi_humid_plus - hi_mean) / delta
        
        # Combined uncertainty (assuming independent errors)
        hi_variance = (dhi_dtemp * temp_uncertainty) ** 2 + (dhi_dhumid * humidity_uncertainty) ** 2
        hi_std = np.sqrt(hi_variance)
        
        # Store uncertainty estimates
        self._uncertainty_estimates['heat_index'] = {
            'mean': hi_mean,
            'std': hi_std,
            'temperature_contribution': np.abs(dhi_dtemp * temp_uncertainty),
            'humidity_contribution': np.abs(dhi_dhumid * humidity_uncertainty)
        }
        
        return hi_mean, hi_std
    
    def estimate_heat_wave_survival_function(self,
                                            temperature_series: pd.Series,
                                            threshold: float = 30.0) -> pd.DataFrame:
        """
        Estimate survival function for heat wave duration using survival analysis.
        
        Args:
            temperature_series: Time series of temperature values
            threshold: Temperature threshold defining a heat wave (°C)
            
        Returns:
            DataFrame with survival function and hazard rate
        """
        # Identify heat wave periods (consecutive days above threshold)
        is_heat_wave = temperature_series >= threshold
        
        # Find heat wave episodes
        heat_wave_episodes = []
        current_duration = 0
        
        for is_hot in is_heat_wave:
            if is_hot:
                current_duration += 1
            else:
                if current_duration > 0:
                    heat_wave_episodes.append(current_duration)
                current_duration = 0
        
        # Add final episode if ongoing
        if current_duration > 0:
            heat_wave_episodes.append(current_duration)
        
        if len(heat_wave_episodes) == 0:
            print("⚠ No heat wave episodes found")
            return pd.DataFrame()
        
        max_duration = max(heat_wave_episodes)
        
        # Compute empirical survival function: S(t) = P(Duration > t)
        # And hazard function: λ(t) = P(end at t | survived to t)
        
        survival = []
        hazard = []
        
        for t in range(1, max_duration + 1):
            # Number still surviving at time t
            n_surviving = sum(1 for d in heat_wave_episodes if d >= t)
            # Number that ended at exactly time t
            n_ending = sum(1 for d in heat_wave_episodes if d == t)
            
            # Survival probability
            s_t = n_surviving / len(heat_wave_episodes)
            
            # Hazard rate (conditional probability of ending)
            # λ(t) = P(T = t | T >= t) = n_ending / n_surviving
            if n_surviving > 0:
                lambda_t = n_ending / n_surviving
            else:
                lambda_t = 0
            
            survival.append(s_t)
            hazard.append(lambda_t)
        
        results = pd.DataFrame({
            'duration': range(1, max_duration + 1),
            'survival_probability': survival,
            'hazard_rate': hazard
        })
        
        # Add cumulative hazard
        results['cumulative_hazard'] = results['hazard_rate'].cumsum()
        
        print(f"✓ Heat wave survival analysis:")
        print(f"  Total episodes: {len(heat_wave_episodes)}")
        print(f"  Mean duration: {np.mean(heat_wave_episodes):.1f} days")
        print(f"  Max duration: {max_duration} days")
        print(f"  P(duration > 3 days): {results.loc[results['duration']==3, 'survival_probability'].values[0]:.2%}"
              if len(results) >= 3 else "")
        
        return results
    
    def compute_heat_stress_with_confidence_intervals(self,
                                                     temperature: pd.Series,
                                                     humidity: pd.Series,
                                                     confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Compute heat stress indices with confidence intervals.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity (0-100)
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            DataFrame with heat indices and confidence intervals
        """
        # Compute mean and uncertainty
        hi_mean, hi_std = self.compute_heat_index_with_uncertainty(temperature, humidity)
        
        # Compute confidence intervals (assuming normal distribution)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        results = pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'heat_index_mean': hi_mean,
            'heat_index_std': hi_std,
            'heat_index_lower': hi_mean - z_score * hi_std,
            'heat_index_upper': hi_mean + z_score * hi_std
        })
        
        # Classify based on mean
        results['stress_level'] = self.classify_stress_levels(hi_mean, 'heat_index')
        
        # Add probability of exceeding danger threshold
        if isinstance(hi_mean, pd.Series):
            danger_threshold = self.thresholds.HI_DANGER
            # P(HI > threshold) using normal CDF
            z = (danger_threshold - hi_mean) / hi_std
            results['prob_danger'] = 1 - stats.norm.cdf(z)
        
        return results
    
    def _rothfusz_regression(self, temp_f: Union[float, pd.Series], 
                            rh: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """Rothfusz regression for high temperature Heat Index"""
        
        hi = (-42.379 + 
              2.04901523 * temp_f + 
              10.14333127 * rh - 
              0.22475541 * temp_f * rh - 
              0.00683783 * temp_f**2 - 
              0.05481717 * rh**2 + 
              0.00122874 * temp_f**2 * rh + 
              0.00085282 * temp_f * rh**2 - 
              0.00000199 * temp_f**2 * rh**2)
        
        return hi
    
    def compute_wbgt(self, 
                     temperature: Union[float, pd.Series],
                     humidity: Union[float, pd.Series],
                     solar_radiation: Optional[Union[float, pd.Series]] = None) -> Union[float, pd.Series]:
        """
        Compute Wet Bulb Globe Temperature (WBGT)
        
        Simplified estimation without actual wet-bulb measurement
        
        Args:
            temperature: Air temperature in Celsius
            humidity: Relative humidity (0-100)
            solar_radiation: Solar radiation in W/m² (optional, assumes outdoor if None)
            
        Returns:
            WBGT in Celsius
        """
        # Estimate wet-bulb temperature from temp and humidity
        wet_bulb = temperature * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + \
                   np.arctan(temperature + humidity) - \
                   np.arctan(humidity - 1.676331) + \
                   0.00391838 * humidity**(3/2) * np.arctan(0.023101 * humidity) - \
                   4.686035
        
        # Simplified WBGT for outdoor conditions (with solar radiation)
        if solar_radiation is None:
            # Assume moderate solar load
            wbgt = 0.7 * wet_bulb + 0.2 * temperature + 0.1 * temperature
        else:
            # Adjust for solar radiation
            solar_factor = 1 + (solar_radiation - 500) / 1000 * 0.1
            wbgt = 0.7 * wet_bulb + 0.2 * temperature * solar_factor + 0.1 * temperature
        
        return wbgt
    
    def compute_apparent_temperature(self,
                                     temperature: Union[float, pd.Series],
                                     humidity: Union[float, pd.Series],
                                     wind_speed: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Compute Apparent Temperature (Australian model)
        
        Args:
            temperature: Air temperature in Celsius
            humidity: Relative humidity (0-100)
            wind_speed: Wind speed in m/s
            
        Returns:
            Apparent temperature in Celsius
        """
        # Vapor pressure
        es = 6.112 * np.exp((17.62 * temperature) / (243.12 + temperature))
        vp = humidity / 100 * es
        
        # Apparent temperature formula
        at = temperature + 0.33 * vp - 0.70 * wind_speed - 4.00
        
        return at
    
    def classify_stress_levels(self, 
                               heat_index: Union[float, pd.Series],
                               index_type: str = 'heat_index') -> Union[str, pd.Series]:
        """
        Classify heat stress levels based on index values
        
        Args:
            heat_index: Heat index values
            index_type: Type of index ('heat_index', 'wbgt', 'utci')
            
        Returns:
            Stress level classification
        """
        if index_type == 'heat_index':
            return self._classify_heat_index(heat_index)
        elif index_type == 'wbgt':
            return self._classify_wbgt(heat_index)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def _classify_heat_index(self, hi: Union[float, pd.Series]) -> Union[str, pd.Series]:
        """Classify Heat Index levels"""
        
        if isinstance(hi, pd.Series):
            conditions = [
                hi < self.thresholds.HI_CAUTION,
                (hi >= self.thresholds.HI_CAUTION) & (hi < self.thresholds.HI_EXTREME_CAUTION),
                (hi >= self.thresholds.HI_EXTREME_CAUTION) & (hi < self.thresholds.HI_DANGER),
                (hi >= self.thresholds.HI_DANGER) & (hi < self.thresholds.HI_EXTREME_DANGER),
                hi >= self.thresholds.HI_EXTREME_DANGER
            ]
            choices = ['low', 'caution', 'extreme_caution', 'danger', 'extreme_danger']
            return pd.Series(np.select(conditions, choices, default='unknown'), index=hi.index)
        else:
            if hi < self.thresholds.HI_CAUTION:
                return 'low'
            elif hi < self.thresholds.HI_EXTREME_CAUTION:
                return 'caution'
            elif hi < self.thresholds.HI_DANGER:
                return 'extreme_caution'
            elif hi < self.thresholds.HI_EXTREME_DANGER:
                return 'danger'
            else:
                return 'extreme_danger'
    
    def _classify_wbgt(self, wbgt: Union[float, pd.Series]) -> Union[str, pd.Series]:
        """Classify WBGT levels"""
        
        if isinstance(wbgt, pd.Series):
            conditions = [
                wbgt < self.thresholds.WBGT_LOW,
                (wbgt >= self.thresholds.WBGT_LOW) & (wbgt < self.thresholds.WBGT_MODERATE),
                (wbgt >= self.thresholds.WBGT_MODERATE) & (wbgt < self.thresholds.WBGT_HIGH),
                (wbgt >= self.thresholds.WBGT_HIGH) & (wbgt < self.thresholds.WBGT_EXTREME),
                wbgt >= self.thresholds.WBGT_EXTREME
            ]
            choices = ['low', 'moderate', 'high', 'very_high', 'extreme']
            return pd.Series(np.select(conditions, choices, default='unknown'), index=wbgt.index)
        else:
            if wbgt < self.thresholds.WBGT_LOW:
                return 'low'
            elif wbgt < self.thresholds.WBGT_MODERATE:
                return 'moderate'
            elif wbgt < self.thresholds.WBGT_HIGH:
                return 'high'
            elif wbgt < self.thresholds.WBGT_EXTREME:
                return 'very_high'
            else:
                return 'extreme'
    
    def compute_all_indices(self,
                           temperature: pd.Series,
                           humidity: pd.Series,
                           wind_speed: Optional[pd.Series] = None,
                           solar_radiation: Optional[pd.Series] = None,
                           include_uncertainty: bool = False) -> pd.DataFrame:
        """
        Compute all heat stress indices at once
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity (0-100)
            wind_speed: Wind speed in m/s (optional)
            solar_radiation: Solar radiation in W/m² (optional)
            include_uncertainty: Whether to include uncertainty estimates
            
        Returns:
            DataFrame with all indices and classifications
        """
        results = pd.DataFrame(index=temperature.index)
        
        if include_uncertainty:
            # Probabilistic approach with confidence intervals
            hi_mean, hi_std = self.compute_heat_index_with_uncertainty(temperature, humidity)
            results['heat_index'] = hi_mean
            results['heat_index_std'] = hi_std
        else:
            # Deterministic approach
            results['heat_index'] = self.compute_heat_index(temperature, humidity)
        
        results['heat_index_level'] = self.classify_stress_levels(
            results['heat_index'], 'heat_index'
        )
        
        # WBGT
        results['wbgt'] = self.compute_wbgt(temperature, humidity, solar_radiation)
        results['wbgt_level'] = self.classify_stress_levels(
            results['wbgt'], 'wbgt'
        )
        
        # Apparent Temperature (requires wind speed)
        if wind_speed is not None:
            results['apparent_temp'] = self.compute_apparent_temperature(
                temperature, humidity, wind_speed
            )
        
        return results