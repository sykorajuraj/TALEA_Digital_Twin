"""
TALEA - Civic Digital Twin: Simulation API
File: src/simulation/api/simulation_api.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

High-level API for creating and running multi-modal mobility simulations under heat stress.

Pipeline Architecture:
    Data Quality Check → Data Preprocessor → Heat Stress/Impact/Pattern Analysis → 
    Simulation API → Notebook (Custom Simulations & Analysis)

6 Mobility Datasets:
    1. Bicycle Counter Data
    2. Pedestrian Flow Data  
    3. Street Network Data
    4. Traffic Monitor Data
    5. Pedestrian Zones (ZTL)
    6. Points of Interest (POI)

3 Weather Datasets:
    1. Temperature in Bologna
    2. Precipitation in Bologna
    3. Air Quality Monitoring in Bologna
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
import json
import warnings

# Core simulation modules
from src.simulation.models.environment import UrbanEnvironment, WeatherConditions
from src.simulation.engine.simulation_engine import (
    SimulationEngine, SimulationConfig, ScenarioManager, SimulationMetrics
)

# Analysis modules for integration
from src.analysis.heat_stress.heat_stress_calculator import HeatStressCalculator
from src.analysis.heat_stress.vulnerability_mapper import VulnerabilityMapper
from src.analysis.heat_impact.mobility_impact_analyzer import MobilityImpactAnalyzer
from src.analysis.heat_impact.route_impact_analyzer import RouteImpactAnalyzer


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SimulationAPIConfig:
    """Configuration for Simulation API"""
    
    # Output settings
    output_dir: Path = Path('simulation_outputs')
    auto_export: bool = True
    export_format: str = 'parquet'  # 'parquet', 'csv', 'json'
    
    # Simulation defaults
    default_time_step_minutes: int = 15
    default_num_agents: int = 1000
    random_seed: int = 42
    
    # Validation
    validate_inputs: bool = True
    require_heat_indices: bool = True
    
    # Analysis integration
    compute_heat_exposure: bool = True
    compute_vulnerability: bool = True
    track_interventions: bool = True


# ============================================================================
# EXCEPTIONS
# ============================================================================

class SimulationAPIError(Exception):
    """Base exception for Simulation API"""
    pass


class EnvironmentNotLoadedError(SimulationAPIError):
    """Raised when environment is not loaded before simulation"""
    pass


class SimulationNotFoundError(SimulationAPIError):
    """Raised when simulation ID doesn't exist"""
    pass


class InvalidConfigurationError(SimulationAPIError):
    """Raised when configuration is invalid"""
    pass


# ============================================================================
# DATA VALIDATORS
# ============================================================================

class DataValidator:
    """Validates input data for simulations"""
    
    @staticmethod
    def validate_street_network(network: gpd.GeoDataFrame) -> Tuple[bool, List[str]]:
        """Validate street network data"""
        errors = []
        
        if network.empty:
            errors.append("Street network is empty")
            return False, errors
        
        required_cols = ['geometry']
        missing = [col for col in required_cols if col not in network.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        if not all(network.geometry.type.isin(['LineString', 'MultiLineString'])):
            errors.append("Network must contain only LineString geometries")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_weather_data(weather: pd.DataFrame, 
                            require_heat_indices: bool = True) -> Tuple[bool, List[str]]:
        """Validate weather data"""
        errors = []
        
        if weather.empty:
            errors.append("Weather data is empty")
            return False, errors
        
        # Check for datetime
        has_datetime = False
        for col in ['datetime', 'Data', 'data']:
            if col in weather.columns:
                has_datetime = True
                break
        
        if not has_datetime:
            errors.append("Weather data must have datetime column")
        
        # Check for temperature
        if 'temperature' not in weather.columns:
            errors.append("Weather data must have 'temperature' column")
        
        # Check for heat indices if required
        if require_heat_indices:
            if 'heat_index' not in weather.columns:
                errors.append("Weather data must have 'heat_index' column (use HeatStressCalculator)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_calibration_data(calibration: Dict) -> Tuple[bool, List[str]]:
        """Validate calibration data"""
        errors = []
        
        if not isinstance(calibration, dict):
            errors.append("Calibration data must be a dictionary")
            return False, errors
        
        for mode in ['bicycle', 'pedestrian']:
            if mode not in calibration:
                continue
            
            data = calibration[mode]
            if not isinstance(data, pd.DataFrame):
                errors.append(f"{mode} data must be a DataFrame")
            
            # Check for datetime
            has_datetime = any(col in data.columns for col in ['datetime', 'Data', 'data'])
            if not has_datetime:
                errors.append(f"{mode} data must have datetime column")
        
        return len(errors) == 0, errors


# ============================================================================
# MAIN SIMULATION API
# ============================================================================

class SimulationAPI:
    """
    High-level API for TALEA multi-modal mobility simulations under heat stress.
    
    Usage:
        # 1. Initialize API
        api = SimulationAPI()
        
        # 2. Load environment (from preprocessed data)
        api.load_environment(
            street_network=network_gdf,
            weather_data=weather_df,
            poi_data=poi_gdf,
            ztl_zones=ztl_gdf,
            spatial_grid=grid_gdf
        )
        
        # 3. Optionally add calibration data
        api.add_calibration_data(bicycle_df, pedestrian_df)
        
        # 4. Create and run simulation
        sim_id = api.create_simulation(
            start_time='2024-06-01',
            end_time='2024-06-07',
            scenario_name='summer_heatwave'
        )
        
        results = api.run_simulation(sim_id)
        
        # 5. Analyze and export
        api.export_results(sim_id)
    """
    
    def __init__(self, config: Optional[SimulationAPIConfig] = None):
        """
        Initialize Simulation API
        
        Args:
            config: API configuration (uses defaults if None)
        """
        self.config = config or SimulationAPIConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.environment: Optional[UrbanEnvironment] = None
        self.scenario_manager: Optional[ScenarioManager] = None
        self.calibration_data: Optional[Dict[str, pd.DataFrame]] = None
        
        # Analysis tools
        self.heat_calculator = HeatStressCalculator()
        self.vulnerability_mapper = VulnerabilityMapper()
        self.mobility_analyzer = MobilityImpactAnalyzer()
        self.route_analyzer = RouteImpactAnalyzer()
        
        # State
        self.simulations: Dict[str, SimulationEngine] = {}
        self.is_initialized = False
        
        print(f"✓ Simulation API initialized")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"  Default agents: {self.config.default_num_agents}")
        print(f"  Time step: {self.config.default_time_step_minutes} minutes")
    
    # ========================================================================
    # ENVIRONMENT SETUP
    # ========================================================================
    
    def load_environment(self,
                        street_network: gpd.GeoDataFrame,
                        weather_data: pd.DataFrame,
                        poi_data: Optional[gpd.GeoDataFrame] = None,
                        ztl_zones: Optional[gpd.GeoDataFrame] = None,
                        spatial_grid: Optional[gpd.GeoDataFrame] = None,
                        validate: Optional[bool] = None) -> 'SimulationAPI':
        """
        Load urban environment data
        
        Args:
            street_network: Street network (must have geometry, optionally heat_exposure)
            weather_data: Weather data with temperature and heat indices
            poi_data: Points of interest
            ztl_zones: Pedestrian zones
            spatial_grid: Spatial grid with vulnerability data
            validate: Validate inputs (uses config default if None)
        
        Returns:
            Self for method chaining
        
        Raises:
            InvalidConfigurationError: If validation fails
        """
        print(f"\n{'='*70}")
        print("LOADING ENVIRONMENT")
        print(f"{'='*70}\n")
        
        validate = validate if validate is not None else self.config.validate_inputs
        
        # Validate inputs
        if validate:
            self._validate_environment_data(street_network, weather_data)
        
        # Create environment
        self.environment = UrbanEnvironment(
            street_network=street_network,
            weather_data=self._prepare_weather_data(weather_data),
            poi_data=poi_data,
            ztl_zones=ztl_zones,
            spatial_grid=spatial_grid
        )
        
        # Initialize scenario manager
        self.scenario_manager = ScenarioManager(
            self.environment,
            calibration_data=self.calibration_data
        )
        
        self.is_initialized = True
        
        # Print summary
        self._print_environment_summary()
        
        return self
    
    def add_calibration_data(self,
                           bicycle_data: Optional[pd.DataFrame] = None,
                           pedestrian_data: Optional[pd.DataFrame] = None,
                           validate: Optional[bool] = None) -> 'SimulationAPI':
        """
        Add calibration data for realistic agent generation
        
        Args:
            bicycle_data: Historical bicycle counter data
            pedestrian_data: Historical pedestrian flow data
            validate: Validate inputs
        
        Returns:
            Self for method chaining
        """
        print("\nAdding calibration data...")
        
        self.calibration_data = {}
        
        if bicycle_data is not None:
            self.calibration_data['bicycle'] = bicycle_data
            print(f"  ✓ Bicycle data: {len(bicycle_data):,} records")
        
        if pedestrian_data is not None:
            self.calibration_data['pedestrian'] = pedestrian_data
            print(f"  ✓ Pedestrian data: {len(pedestrian_data):,} records")
        
        # Validate if requested
        validate = validate if validate is not None else self.config.validate_inputs
        if validate and self.calibration_data:
            valid, errors = DataValidator.validate_calibration_data(self.calibration_data)
            if not valid:
                warnings.warn(f"Calibration data validation warnings: {errors}")
        
        # Update scenario manager if exists
        if self.scenario_manager is not None:
            self.scenario_manager.calibration_data = self.calibration_data
        
        return self
    
    # ========================================================================
    # SIMULATION CREATION & EXECUTION
    # ========================================================================
    
    def create_simulation(self,
                         start_time: Union[str, pd.Timestamp],
                         end_time: Union[str, pd.Timestamp],
                         scenario_name: str = "baseline",
                         num_agents: Optional[int] = None,
                         time_step_minutes: Optional[int] = None,
                         interventions: Optional[List[Dict]] = None,
                         **kwargs) -> str:
        """
        Create a new simulation
        
        Args:
            start_time: Simulation start time
            end_time: Simulation end time
            scenario_name: Descriptive name for scenario
            num_agents: Number of agents (uses default if None)
            time_step_minutes: Time step in minutes (uses default if None)
            interventions: List of intervention dictionaries
            **kwargs: Additional SimulationConfig parameters
        
        Returns:
            Simulation ID (unique identifier)
        
        Raises:
            EnvironmentNotLoadedError: If environment not loaded
        """
        if not self.is_initialized or self.environment is None:
            raise EnvironmentNotLoadedError(
                "Environment not loaded. Call load_environment() first."
            )
        
        # Parse times
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        
        # Use defaults
        num_agents = num_agents or self.config.default_num_agents
        time_step_minutes = time_step_minutes or self.config.default_time_step_minutes
        
        # Create configuration
        config = SimulationConfig(
            scenario_name=scenario_name,
            start_time=start_time,
            end_time=end_time,
            time_step=timedelta(minutes=time_step_minutes),
            interventions=interventions or [],
            use_calibrated_generation=self.calibration_data is not None,
            random_seed=self.config.random_seed,
            **kwargs
        )
        
        # Create engine
        engine = SimulationEngine(
            self.environment,
            config,
            calibration_data=self.calibration_data
        )
        
        # Generate unique ID
        sim_id = f"{scenario_name}_{uuid.uuid4().hex[:8]}"
        self.simulations[sim_id] = engine
        
        print(f"\n{'='*70}")
        print(f"SIMULATION CREATED: {sim_id}")
        print(f"{'='*70}")
        print(f"  Scenario: {scenario_name}")
        print(f"  Period: {start_time.date()} to {end_time.date()}")
        print(f"  Duration: {(end_time - start_time).days} days")
        print(f"  Time step: {time_step_minutes} minutes")
        print(f"  Calibrated generation: {config.use_calibrated_generation}")
        if interventions:
            print(f"  Interventions: {len(interventions)}")
        
        return sim_id
    
    def run_simulation(self,
                      simulation_id: str,
                      verbose: bool = True,
                      export_results: Optional[bool] = None) -> Dict:
        """
        Run a simulation
        
        Args:
            simulation_id: ID of simulation to run
            verbose: Print progress updates
            export_results: Auto-export results (uses config default if None)
        
        Returns:
            Summary dictionary with results
        
        Raises:
            SimulationNotFoundError: If simulation ID doesn't exist
        """
        if simulation_id not in self.simulations:
            raise SimulationNotFoundError(
                f"Simulation '{simulation_id}' not found. "
                f"Available: {list(self.simulations.keys())}"
            )
        
        engine = self.simulations[simulation_id]
        
        # Run simulation
        print(f"\n{'='*70}")
        print(f"RUNNING SIMULATION: {simulation_id}")
        print(f"{'='*70}\n")
        
        results = engine.run(verbose=verbose)
        
        # Export if requested
        export_results = export_results if export_results is not None else self.config.auto_export
        if export_results:
            output_path = self.config.output_dir / simulation_id
            self.export_results(simulation_id, output_path)
        
        return results
    
    def run_multiple_simulations(self,
                                simulation_ids: List[str],
                                verbose: bool = False) -> Dict[str, Dict]:
        """
        Run multiple simulations in sequence
        
        Args:
            simulation_ids: List of simulation IDs
            verbose: Print detailed progress
        
        Returns:
            Dictionary mapping simulation IDs to results
        """
        print(f"\n{'='*70}")
        print(f"RUNNING {len(simulation_ids)} SIMULATIONS")
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for i, sim_id in enumerate(simulation_ids, 1):
            print(f"\n[{i}/{len(simulation_ids)}] Running: {sim_id}")
            try:
                results = self.run_simulation(sim_id, verbose=verbose)
                all_results[sim_id] = results
                print(f"  ✓ Completed")
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                all_results[sim_id] = {'error': str(e)}
        
        print(f"\n{'='*70}")
        print(f"BATCH COMPLETE: {len(all_results)} / {len(simulation_ids)} successful")
        print(f"{'='*70}\n")
        
        return all_results
    
    # ========================================================================
    # RESULTS & ANALYSIS
    # ========================================================================
    
    def get_results(self, simulation_id: str) -> Dict:
        """
        Get detailed results from completed simulation
        
        Args:
            simulation_id: Simulation ID
        
        Returns:
            Dictionary with summary, metrics DataFrames, and analysis
        """
        engine = self._get_engine(simulation_id)
        
        # Export metrics to DataFrames
        timestep_df, agent_df = engine.metrics.export_to_dataframe()
        summary = engine.metrics.get_summary()
        
        results = {
            'simulation_id': simulation_id,
            'scenario_name': engine.config.scenario_name,
            'summary': summary,
            'timestep_metrics': timestep_df,
            'agent_trips': agent_df,
            'config': engine.config
        }
        
        # Add intervention stats if available
        if self.environment.interventions:
            results['intervention_stats'] = self.environment.get_intervention_statistics()
        
        return results
    
    def compare_scenarios(self,
                         simulation_ids: List[str],
                         metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare results across multiple scenarios
        
        Args:
            simulation_ids: List of simulation IDs to compare
            metrics: Specific metrics to include (all if None)
        
        Returns:
            DataFrame with comparison across scenarios
        """
        comparison_data = []
        
        for sim_id in simulation_ids:
            if sim_id not in self.simulations:
                warnings.warn(f"Simulation '{sim_id}' not found, skipping")
                continue
            
            engine = self.simulations[sim_id]
            summary = engine.metrics.get_summary()
            
            row = {
                'simulation_id': sim_id,
                'scenario_name': engine.config.scenario_name,
                'start_time': engine.config.start_time,
                'end_time': engine.config.end_time,
                **summary
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Filter to specific metrics if requested
        if metrics:
            available_metrics = [m for m in metrics if m in comparison_df.columns]
            comparison_df = comparison_df[['simulation_id', 'scenario_name'] + available_metrics]
        
        return comparison_df
    
    def export_results(self,
                      simulation_id: str,
                      output_path: Optional[Path] = None,
                      format: Optional[str] = None) -> Path:
        """
        Export simulation results to files
        
        Args:
            simulation_id: Simulation to export
            output_path: Output directory (default: output_dir/sim_id)
            format: Export format (uses config default if None)
        
        Returns:
            Path to output directory
        """
        engine = self._get_engine(simulation_id)
        
        # Determine output path
        if output_path is None:
            output_path = self.config.output_dir / simulation_id
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        format = format or self.config.export_format
        
        print(f"\nExporting results to {output_path}/")
        
        # Export metrics
        timestep_df, agent_df = engine.metrics.export_to_dataframe()
        
        if format == 'parquet':
            timestep_df.to_parquet(output_path / 'timestep_metrics.parquet')
            agent_df.to_parquet(output_path / 'agent_trips.parquet')
        elif format == 'csv':
            timestep_df.to_csv(output_path / 'timestep_metrics.csv', index=False)
            agent_df.to_csv(output_path / 'agent_trips.csv', index=False)
        
        # Export summary as JSON
        summary = engine.metrics.get_summary()
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Export configuration
        config_dict = {
            'scenario_name': engine.config.scenario_name,
            'start_time': str(engine.config.start_time),
            'end_time': str(engine.config.end_time),
            'time_step_minutes': engine.config.time_step.total_seconds() / 60,
            'use_calibrated_generation': engine.config.use_calibrated_generation,
            'num_interventions': len(engine.config.interventions)
        }
        with open(output_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"  ✓ timestep_metrics.{format}")
        print(f"  ✓ agent_trips.{format}")
        print(f"  ✓ summary.json")
        print(f"  ✓ config.json")
        
        return output_path
    
    # ========================================================================
    # SCENARIO HELPERS
    # ========================================================================
    
    def create_baseline_scenario(self,
                                 start_time: Union[str, pd.Timestamp],
                                 end_time: Union[str, pd.Timestamp],
                                 **kwargs) -> str:
        """
        Create baseline scenario (normal conditions, no interventions)
        
        Returns:
            Simulation ID
        """
        return self.create_simulation(
            start_time=start_time,
            end_time=end_time,
            scenario_name="baseline",
            interventions=[],
            **kwargs
        )
    
    def create_heatwave_scenario(self,
                                start_time: Union[str, pd.Timestamp],
                                end_time: Union[str, pd.Timestamp],
                                intensity: str = 'moderate',
                                **kwargs) -> str:
        """
        Create heat wave scenario
        
        Args:
            intensity: 'moderate', 'severe', or 'extreme'
        
        Returns:
            Simulation ID
        
        Note:
            Requires weather data to already contain heat wave conditions
        """
        return self.create_simulation(
            start_time=start_time,
            end_time=end_time,
            scenario_name=f"heatwave_{intensity}",
            **kwargs
        )
    
    def create_intervention_scenario(self,
                                    start_time: Union[str, pd.Timestamp],
                                    end_time: Union[str, pd.Timestamp],
                                    interventions: List[Dict],
                                    scenario_name: Optional[str] = None,
                                    **kwargs) -> str:
        """
        Create scenario with cooling interventions
        
        Args:
            interventions: List of intervention dictionaries with:
                - type: 'cooling_station', 'shade_structure', 'tree_planting'
                - location: (lon, lat) tuple
                - parameters: Dict with intervention-specific params
                    - For cooling_station: {'capacity': int, 'temperature_reduction': float}
                    - For shade_structure: {'temperature_reduction': float, 'radius': float}
                    - For tree_planting: {'temperature_reduction': float, 'radius': float}
            scenario_name: Custom name (auto-generated if None)
        
        Returns:
            Simulation ID
        
        Example:
            interventions = [
                {
                    'type': 'cooling_station',
                    'location': (11.34, 44.49),
                    'parameters': {'capacity': 50, 'temperature_reduction': 3.0}
                },
                {
                    'type': 'tree_planting',
                    'location': (11.35, 44.50),
                    'parameters': {'temperature_reduction': 2.0, 'radius': 100.0}
                }
            ]
        """
        if scenario_name is None:
            scenario_name = f"intervention_{len(interventions)}_measures"
        
        return self.create_simulation(
            start_time=start_time,
            end_time=end_time,
            scenario_name=scenario_name,
            interventions=interventions,
            **kwargs
        )
    
    # ========================================================================
    # ANALYSIS INTEGRATION
    # ========================================================================
    
    def analyze_mobility_heat_impact(self,
                                    simulation_id: str) -> Dict:
        """
        Analyze heat impact on mobility from simulation results
        
        Args:
            simulation_id: Simulation to analyze
        
        Returns:
            Dictionary with impact analysis
        """
        results = self.get_results(simulation_id)
        agent_df = results['agent_trips']
        
        if agent_df.empty:
            return {'error': 'No trip data available'}
        
        analysis = {
            'mode_distribution': agent_df['mode'].value_counts().to_dict(),
            'avg_travel_time_by_mode': agent_df.groupby('mode')['travel_time'].mean().to_dict(),
            'avg_heat_exposure_by_mode': agent_df.groupby('mode')['heat_exposure'].mean().to_dict()
        }
        
        # Add heat stress impact if age/health data available
        if 'age_group' in agent_df.columns:
            analysis['trips_by_age_group'] = agent_df['age_group'].value_counts().to_dict()
        
        if 'health_status' in agent_df.columns:
            analysis['trips_by_health_status'] = agent_df['health_status'].value_counts().to_dict()
        
        return analysis
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def get_simulation_status(self, simulation_id: str) -> Dict:
        """Get current status of a simulation"""
        engine = self._get_engine(simulation_id)
        
        return {
            'simulation_id': simulation_id,
            'scenario_name': engine.config.scenario_name,
            'is_initialized': engine.is_initialized,
            'is_running': engine.is_running,
            'current_time': str(engine.current_time) if engine.current_time else None,
            'step_count': engine.step_count,
            'start_time': str(engine.config.start_time),
            'end_time': str(engine.config.end_time)
        }
    
    def list_simulations(self) -> List[Dict]:
        """List all created simulations"""
        return [
            {
                'simulation_id': sim_id,
                'scenario_name': engine.config.scenario_name,
                'is_initialized': engine.is_initialized,
                'is_running': engine.is_running
            }
            for sim_id, engine in self.simulations.items()
        ]
    
    def delete_simulation(self, simulation_id: str):
        """Delete a simulation from memory"""
        if simulation_id in self.simulations:
            del self.simulations[simulation_id]
            print(f"✓ Deleted simulation: {simulation_id}")
        else:
            raise SimulationNotFoundError(f"Simulation '{simulation_id}' not found")
    
    def clear_all_simulations(self):
        """Clear all simulations from memory"""
        count = len(self.simulations)
        self.simulations.clear()
        print(f"✓ Cleared {count} simulations")
    
    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================
    
    def _get_engine(self, simulation_id: str) -> SimulationEngine:
        """Get simulation engine, raising error if not found"""
        if simulation_id not in self.simulations:
            raise SimulationNotFoundError(
                f"Simulation '{simulation_id}' not found. "
                f"Available: {list(self.simulations.keys())}"
            )
        return self.simulations[simulation_id]
    
    def _validate_environment_data(self,
                                  network: gpd.GeoDataFrame,
                                  weather: pd.DataFrame):
        """Validate environment input data"""
        
        # Validate street network
        valid, errors = DataValidator.validate_street_network(network)
        if not valid:
            raise InvalidConfigurationError(
                f"Street network validation failed: {errors}"
            )
        
        # Validate weather data
        valid, errors = DataValidator.validate_weather_data(
            weather,
            require_heat_indices=self.config.require_heat_indices
        )
        if not valid:
            raise InvalidConfigurationError(
                f"Weather data validation failed: {errors}"
            )
    
    def _prepare_weather_data(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Prepare weather data for simulation"""
        weather = weather.copy()
        
        # Ensure datetime index
        for col in ['datetime', 'Data', 'data']:
            if col in weather.columns:
                weather['datetime'] = pd.to_datetime(weather[col])
                weather.set_index('datetime', inplace=True)
                break
        
        # Ensure required columns
        if 'temperature' not in weather.columns:
            raise InvalidConfigurationError("Weather data must have 'temperature' column")
        
        # Add heat_index if missing (use temperature as fallback)
        if 'heat_index' not in weather.columns:
            warnings.warn(
                "Weather data missing 'heat_index' - using temperature as fallback. "
                "Consider using HeatStressCalculator for accurate heat indices."
            )
            weather['heat_index'] = weather['temperature']
        
        # Add heat_index_level if missing
        if 'heat_index_level' not in weather.columns:
            weather['heat_index_level'] = self.heat_calculator.classify_stress_levels(
                weather['heat_index'],
                index_type='heat_index'
            )
        
        return weather
    
    def _print_environment_summary(self):
        """Print summary of loaded environment"""
        stats = self.environment.get_summary_statistics()
        
        print(f"\n{'='*70}")
        print("ENVIRONMENT LOADED")
        print(f"{'='*70}\n")
        
        # Network
        print("Network:")
        print(f"  Nodes: {stats['network']['num_nodes']:,}")
        print(f"  Edges: {stats['network']['num_edges']:,}")
        print(f"  Total length: {stats['network']['total_length_km']:.1f} km")
        
        # Weather
        if 'weather' in stats:
            print("\nWeather:")
            print(f"  Period: {stats['weather']['time_range']}")
            print(f"  Avg temperature: {stats['weather']['avg_temperature']:.1f}°C")
            print(f"  Max heat index: {stats['weather']['max_heat_index']:.1f}")
        
        # POI
        if 'poi' in stats:
            print(f"\nPOI: {stats['poi']['total_poi']} locations")
        
        print(f"\n{'='*70}\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_api_from_preprocessed_data(
    street_network: gpd.GeoDataFrame,
    weather_data: pd.DataFrame,
    bicycle_data: Optional[pd.DataFrame] = None,
    pedestrian_data: Optional[pd.DataFrame] = None,
    poi_data: Optional[gpd.GeoDataFrame] = None,
    ztl_zones: Optional[gpd.GeoDataFrame] = None,
    spatial_grid: Optional[gpd.GeoDataFrame] = None,
    config: Optional[SimulationAPIConfig] = None
) -> SimulationAPI:
    """
    Convenience function to create and initialize API in one step
    
    Args:
        street_network: Processed street network with heat exposure
        weather_data: Processed weather data with heat indices
        bicycle_data: Calibration data for bicycles (optional)
        pedestrian_data: Calibration data for pedestrians (optional)
        poi_data: Points of interest (optional)
        ztl_zones: Pedestrian zones (optional)
        spatial_grid: Spatial grid with vulnerability (optional)
        config: API configuration (optional)
    
    Returns:
        Initialized SimulationAPI ready for creating simulations
    
    Example:
        api = create_api_from_preprocessed_data(
            street_network=network_gdf,
            weather_data=weather_df,
            bicycle_data=bicycle_df,
            pedestrian_data=pedestrian_df
        )
        
        sim_id = api.create_baseline_scenario('2024-06-01', '2024-06-07')
        results = api.run_simulation(sim_id)
    """
    api = SimulationAPI(config=config)
    
    # Load environment
    api.load_environment(
        street_network=street_network,
        weather_data=weather_data,
        poi_data=poi_data,
        ztl_zones=ztl_zones,
        spatial_grid=spatial_grid
    )
    
    # Add calibration data if provided
    if bicycle_data is not None or pedestrian_data is not None:
        api.add_calibration_data(
            bicycle_data=bicycle_data,
            pedestrian_data=pedestrian_data
        )
    
    return api