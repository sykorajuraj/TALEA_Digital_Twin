"""
TALEA - Civic Digital Twin: Simulation Engine
File: src/simulation/engines/simulation_engine.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Core simulation engine for multi-modal mobility under heat stress.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.simulation.models.agent import MobilityAgent, AgentPopulation, TransportMode
from src.simulation.models.environment import UrbanEnvironment, WeatherConditions


@dataclass
class SimulationConfig:
    """Configuration for simulation run"""
    
    # Temporal parameters
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    time_step: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    
    # Agent parameters
    use_calibrated_generation: bool = True
    trip_purposes: List[str] = field(default_factory=lambda: [
        'commute', 'shopping', 'leisure', 'other'
    ])
    
    # Simulation parameters
    random_seed: Optional[int] = 42
    collect_detailed_metrics: bool = True
    output_frequency: int = 4 # Save every N steps
    
    # Scenario parameters
    scenario_name: str = "baseline"
    interventions: List[Dict] = field(default_factory=list)


class SimulationMetrics:
    """Collects and stores simulation metrics"""
    
    def __init__(self):
        self.timestep_metrics = []
        self.agent_metrics = []
        self.mode_share_history = []
        self.heat_exposure_history = []
        self.edge_flow_history = []  # Track flow on edges
        self.intervention_usage = []  # Track intervention usage
    
    def record_timestep(self, timestamp: pd.Timestamp, metrics: Dict):
        """Record metrics for a timestep"""
        metrics['timestamp'] = timestamp
        self.timestep_metrics.append(metrics)
    
    def record_agent_trip(self, agent):  # agent: MobilityAgent type
        """Record completed agent trip with detailed path"""
        if agent.trip_completed:
            self.agent_metrics.append({
                'agent_id': agent.agent_id,
                'origin': agent.origin,
                'destination': agent.destination,
                'mode': agent.actual_mode.value if agent.actual_mode else None,
                'travel_time': agent.actual_travel_time,
                'heat_exposure': agent.heat_exposure,
                'trip_purpose': agent.trip_purpose,
                'age_group': agent.sensitivity_profile.age_group.value,
                'health_status': agent.sensitivity_profile.health_status.value,
                # Path tracking
                'path_length': len(agent.path_history),
                'edges_used': len(agent.edge_history)
            })
    
    def record_edge_flows(self, timestamp: pd.Timestamp, edge_flows: Dict):
        """Record flow on network edges"""
        self.edge_flow_history.append({
            'timestamp': timestamp,
            'flows': edge_flows.copy()
        })
    
    def record_intervention_usage(self, timestamp: pd.Timestamp, 
                                 usage_stats: Dict):
        """Record intervention usage statistics"""
        usage_stats['timestamp'] = timestamp
        self.intervention_usage.append(usage_stats)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.agent_metrics:
            return {}
        
        agent_df = pd.DataFrame(self.agent_metrics)
        
        summary = {
            'total_trips': len(agent_df),
            'avg_travel_time': agent_df['travel_time'].mean() if 'travel_time' in agent_df else 0,
            'avg_heat_exposure': agent_df['heat_exposure'].mean() if 'heat_exposure' in agent_df else 0,
            'mode_shares': agent_df['mode'].value_counts(normalize=True).to_dict() if 'mode' in agent_df else {},
            'trips_by_purpose': agent_df['trip_purpose'].value_counts().to_dict() if 'trip_purpose' in agent_df else {}
        }
        
        return summary
    
    def export_to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Export metrics to DataFrames"""
        timestep_df = pd.DataFrame(self.timestep_metrics)
        agent_df = pd.DataFrame(self.agent_metrics)
        return timestep_df, agent_df


class SimulationEngine:
    """Main simulation engine"""
    
    def __init__(self, environment,  # UrbanEnvironment type
                 config: SimulationConfig,
                 calibration_data: Optional[Dict] = None):
        """
        Initialize simulation engine
        
        Args:
            environment: Urban environment
            config: Simulation configuration
            calibration_data: Dict with 'bicycle' and 'pedestrian' DataFrames
        """
        self.environment = environment
        self.config = config
        
        self.calibration_data = calibration_data
        self.population = AgentPopulation(calibration_data=calibration_data)
        # self.population = None # placeholder
        
        self.metrics = SimulationMetrics()
        
        # State
        self.current_time: Optional[pd.Timestamp] = None
        self.step_count: int = 0
        self.is_initialized: bool = False
        self.is_running: bool = False
        
        # Set random seed
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def initialize_simulation(self):
        """Initialize simulation"""
        print(f"\n{'='*70}")
        print(f"INITIALIZING SIMULATION: {self.config.scenario_name}")
        print(f"{'='*70}")
        
        if self.config.interventions:
            print(f"\nApplying {len(self.config.interventions)} interventions...")
            self._apply_interventions()
        
        # Initialize population (will be generated dynamically during simulation)
        print(f"\nUsing calibrated agent generation: {self.config.use_calibrated_generation}")
        
        # Initialize time
        self.current_time = self.config.start_time
        self.environment.update_weather(self.current_time)
        
        self.is_initialized = True
        print(f"\n✓ Simulation initialized")
        print(f"  Start time: {self.config.start_time}")
        print(f"  End time: {self.config.end_time}")
        print(f"  Time step: {self.config.time_step}")
    
    def step(self) -> Dict:
        """
        Execute single simulation time step with REALISTIC MOVEMENT
        
        Returns:
            Dict with step metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized")
        
        # Update environment
        self.environment.update_weather(self.current_time)
        conditions = self.environment.current_weather
        
        # Generate new agents if using calibrated generation
        if self.config.use_calibrated_generation:
            self._generate_agents_for_timestep()
        
        # Get active agents
        active_agents = self.population.get_active_agents(self.current_time)
        
        step_metrics = {
            'active_agents': len(active_agents),
            'temperature': conditions.temperature,
            'heat_index': conditions.heat_index,
            'heat_stress_level': conditions.heat_stress_level,
            'completed_trips': 0,
            'cancelled_trips': 0,
            'mode_choices': {},
            'agents_in_motion': 0
        }
        
        # Track edge flows
        edge_flows = {}
        
        # Process each active agent
        for agent in active_agents:
            env_conditions = self.environment.get_conditions_at(
                agent.current_location, self.current_time
            )
            
            if not agent.trip_completed and not agent.trip_cancelled:
                # Just departed?
                if agent.departure_time == self.current_time:
                    # Decide on trip execution
                    if not agent.decide_trip_execution(env_conditions):
                        step_metrics['cancelled_trips'] += 1
                        continue
                    
                    # Choose mode
                    available_modes_str = self.environment.get_available_modes_at(
                        agent.origin, self.current_time
                    )
                    # Convert to enum
                    from enum import Enum
                    # available_modes = [TransportMode[m.upper()] for m in available_modes_str
                    #                   if m.upper() in TransportMode.__members__]
                    # For now, simplified
                    available_modes = []  # Would use proper conversion
                    
                    chosen_mode = agent.choose_mode(available_modes, env_conditions)
                    
                    # Track mode choice
                    mode_str = chosen_mode.value
                    step_metrics['mode_choices'][mode_str] = \
                        step_metrics['mode_choices'].get(mode_str, 0) + 1
                    
                    # Find route
                    route = self.environment.network.find_shortest_path(
                        agent.origin, agent.destination, mode_str
                    )
                    
                    if route:
                        agent.set_route(route)
                
                # Move agent along network (EDGE-BY-EDGE)
                if agent.current_route:
                    time_step_minutes = self.config.time_step.total_seconds() / 60
                    
                    completed = agent.move_along_network(
                        time_step_minutes,
                        self.environment.network,
                        env_conditions
                    )
                    
                    if completed:
                        step_metrics['completed_trips'] += 1
                        agent.complete_trip(
                            (self.current_time - agent.departure_time).total_seconds() / 60
                        )
                        self.metrics.record_agent_trip(agent)
                    else:
                        step_metrics['agents_in_motion'] += 1
                    
                    # Track edge flows
                    for edge_id in agent.edge_history:
                        mode_key = f"{edge_id}_{agent.current_mode.value if agent.current_mode else 'walk'}"
                        edge_flows[mode_key] = edge_flows.get(mode_key, 0) + 1
        
        # Record metrics
        self.metrics.record_timestep(self.current_time, step_metrics)
        self.metrics.record_edge_flows(self.current_time, edge_flows)
        
        # Record intervention usage
        if self.environment.interventions:
            intervention_stats = self.environment.get_intervention_statistics()
            self.metrics.record_intervention_usage(self.current_time, intervention_stats)
        
        # Advance time
        self.current_time += self.config.time_step
        self.step_count += 1
        
        return step_metrics
    
    def run(self, verbose: bool = True) -> Dict:
        """Run complete simulation"""
        if not self.is_initialized:
            self.initialize_simulation()
        
        self.is_running = True
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RUNNING SIMULATION")
            print(f"{'='*70}\n")
        
        total_steps = int(
            (self.config.end_time - self.current_time) / self.config.time_step
        )
        
        while self.current_time < self.config.end_time:
            step_metrics = self.step()
            
            if verbose and self.step_count % self.config.output_frequency == 0:
                progress = (self.step_count / total_steps) * 100
                print(f"Step {self.step_count}/{total_steps} ({progress:.1f}%) - "
                      f"Active: {step_metrics['active_agents']}, "
                      f"Completed: {step_metrics['completed_trips']}, "
                      f"Temp: {step_metrics['temperature']:.1f}°C, "
                      f"HI: {step_metrics['heat_index']:.1f}°C")
        
        self.is_running = False
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIMULATION COMPLETE")
            print(f"{'='*70}\n")
        
        summary = self._generate_summary()
        
        if verbose:
            self._print_summary(summary)
        
        return summary
    
    def _generate_agents_for_timestep(self):
        """Generate agents for current timestep using calibrated rates"""
        locations = self._get_typical_locations()
        
        # Generate using calibrated rates
        time_step_hours = self.config.time_step.total_seconds() / 3600
        
        agents = self.population.generate_agents_calibrated(
            timestamp=self.current_time,
            duration_hours=time_step_hours,
            origins=locations,
            destinations=locations,
            trip_purposes=self.config.trip_purposes
        )
    
    def _get_typical_locations(self) -> List[Tuple[float, float]]:
        """Get typical origin/destination locations"""
        # Bologna city center approximate bounds
        min_lon, max_lon = 11.32, 11.36
        min_lat, max_lat = 44.48, 44.51
        
        locations = []
        for _ in range(50):
            lon = np.random.uniform(min_lon, max_lon)
            lat = np.random.uniform(min_lat, max_lat)
            locations.append((lon, lat))
        
        return locations
    
    def _apply_interventions(self):
        """Apply scenario interventions to environment"""
        for intervention in self.config.interventions:
            intervention_type = intervention.get('type')
            location = tuple(intervention.get('location'))
            parameters = intervention.get('parameters', {})
            
            self.environment.add_intervention(
                intervention_type, location, parameters
            )
    
    def _generate_summary(self) -> Dict:
        """Generate simulation summary"""
        pop_stats = self.population.get_statistics()
        metrics_summary = self.metrics.get_summary()
        
        summary = {
            'scenario_name': self.config.scenario_name,
            'total_steps': self.step_count,
            'simulation_period': f"{self.config.start_time} to {self.config.end_time}",
            **pop_stats,
            **metrics_summary
        }
        
        # Add intervention statistics
        if self.environment.interventions:
            summary['interventions'] = self.environment.get_intervention_statistics()
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print simulation summary"""
        print("Simulation Summary:")
        print(f"  Scenario: {summary.get('scenario_name')}")
        print(f"  Total agents: {summary.get('total_agents', 0)}")
        print(f"  Completed trips: {summary.get('completed_trips', 0)}")
        print(f"  Cancelled trips: {summary.get('cancelled_trips', 0)}")
        print(f"  Completion rate: {summary.get('completion_rate', 0):.1%}")
        print(f"  Average travel time: {summary.get('avg_travel_time', 0):.1f} min")
        print(f"  Average heat exposure: {summary.get('avg_heat_exposure', 0):.1f}")
        
        if 'mode_shares' in summary:
            print("\n  Mode shares:")
            for mode, share in summary['mode_shares'].items():
                print(f"    {mode}: {share:.1%}")
        
        if 'interventions' in summary:
            print(f"\n  Interventions: {summary['interventions'].get('total_interventions', 0)}")
            for itype, count in summary['interventions'].get('by_type', {}).items():
                print(f"    {itype}: {count}")
    
    def export_results(self, output_dir: Path, format: str = 'parquet'):
        """Export simulation results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestep_df, agent_df = self.metrics.export_to_dataframe()
        
        if format == 'parquet':
            timestep_df.to_parquet(output_dir / 'timestep_metrics.parquet')
            agent_df.to_parquet(output_dir / 'agent_trips.parquet')
        elif format == 'csv':
            timestep_df.to_csv(output_dir / 'timestep_metrics.csv', index=False)
            agent_df.to_csv(output_dir / 'agent_trips.csv', index=False)
        
        # Export summary
        summary = self._generate_summary()
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✓ Results exported to {output_dir}/")


class ScenarioManager:
    """Manages different simulation scenarios"""
    
    def __init__(self, environment, calibration_data: Optional[Dict] = None):
        self.environment = environment
        self.calibration_data = calibration_data
        self.scenarios = {}
        self.results = {}
    
    def create_baseline_scenario(self, start_time: pd.Timestamp,
                                 end_time: pd.Timestamp) -> SimulationConfig:
        """Create baseline scenario (normal conditions)"""
        config = SimulationConfig(
            scenario_name="baseline",
            start_time=start_time,
            end_time=end_time,
            use_calibrated_generation=True,
            interventions=[]
        )
        
        self.scenarios['baseline'] = config
        return config
    
    def create_intervention_scenario(self, base_scenario: str,
                                    interventions: List[Dict],
                                    scenario_name: Optional[str] = None) -> SimulationConfig:
        """Create scenario with interventions"""
        if base_scenario not in self.scenarios:
            raise ValueError(f"Base scenario '{base_scenario}' not found")
        
        base_config = self.scenarios[base_scenario]
        
        if scenario_name is None:
            scenario_name = f"{base_scenario}_with_interventions"
        
        config = SimulationConfig(
            scenario_name=scenario_name,
            start_time=base_config.start_time,
            end_time=base_config.end_time,
            use_calibrated_generation=True,
            interventions=interventions
        )
        
        self.scenarios[scenario_name] = config
        return config
    
    def run_scenario(self, scenario_name: str, verbose: bool = True) -> Dict:
        """Run a specific scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        config = self.scenarios[scenario_name]
        
        engine = SimulationEngine(
            self.environment, 
            config,
            calibration_data=self.calibration_data
        )
        results = engine.run(verbose=verbose)
        
        self.results[scenario_name] = {
            'config': config,
            'summary': results,
            'metrics': engine.metrics
        }
        
        return results
    
    def compare_scenarios(self, scenario_names: List[str]) -> pd.DataFrame:
        """Compare results across scenarios"""
        comparison_data = []
        
        for name in scenario_names:
            if name not in self.results:
                print(f"Warning: '{name}' not yet run")
                continue
            
            summary = self.results[name]['summary']
            
            comparison_data.append({
                'scenario': name,
                'total_trips': summary.get('total_trips', 0),
                'completion_rate': summary.get('completion_rate', 0),
                'cancellation_rate': summary.get('cancellation_rate', 0),
                'avg_travel_time': summary.get('avg_travel_time', 0),
                'avg_heat_exposure': summary.get('avg_heat_exposure', 0),
                **summary.get('mode_shares', {})
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df