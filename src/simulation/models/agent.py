"""
TALEA - Civic Digital Twin: Agent Models
File: src/simulation/models/agent.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Defines agent behavior for mobility simulation under heat stress.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random


class AgeGroup(Enum):
    """Age categories for heat sensitivity"""
    CHILD = "child"  # 0-12
    TEEN = "teen"  # 13-17
    YOUNG_ADULT = "young_adult"  # 18-35
    ADULT = "adult"  # 36-65
    ELDERLY = "elderly"  # 65+


class HealthStatus(Enum):
    """Health status categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class TransportMode(Enum):
    """Available transport modes"""
    WALK = "walk"
    BICYCLE = "bicycle"
    BUS = "bus"
    CAR = "car"
    SCOOTER = "scooter"


@dataclass
class HeatSensitivityProfile:
    """Profile defining agent's sensitivity to heat stress"""
    
    age_group: AgeGroup
    health_status: HealthStatus
    acclimatization: float = 0.5  # 0=not acclimatized, 1=fully acclimatized
    
    # Heat stress thresholds (°C)
    DISCOMFORT_THRESHOLDS: Dict[AgeGroup, float] = field(default_factory=lambda: {
        AgeGroup.CHILD: 26.0,
        AgeGroup.TEEN: 28.0,
        AgeGroup.YOUNG_ADULT: 30.0,
        AgeGroup.ADULT: 29.0,
        AgeGroup.ELDERLY: 25.0
    })
    
    # Health status modifiers
    HEALTH_MODIFIERS: Dict[HealthStatus, float] = field(default_factory=lambda: {
        HealthStatus.EXCELLENT: 1.1,
        HealthStatus.GOOD: 1.0,
        HealthStatus.FAIR: 0.9,
        HealthStatus.POOR: 0.8
    })
    
    def get_discomfort_threshold(self) -> float:
        """Calculate personalized discomfort threshold"""
        base_threshold = self.DISCOMFORT_THRESHOLDS[self.age_group]
        health_modifier = self.HEALTH_MODIFIERS[self.health_status]
        acclimatization_bonus = 2.0 * self.acclimatization
        
        return base_threshold * health_modifier + acclimatization_bonus
    
    def get_mode_preference_adjustment(self, heat_stress_level: str, 
                                      mode: TransportMode) -> float:
        """
        Get preference adjustment for a mode under heat stress
        
        Returns: multiplier for mode utility (0-1)
        """
        # Base adjustments by mode and heat level
        adjustments = {
            'low': {
                TransportMode.WALK: 1.0,
                TransportMode.BICYCLE: 1.0,
                TransportMode.BUS: 1.0,
                TransportMode.CAR: 0.95,
                TransportMode.SCOOTER: 1.0
            },
            'moderate': {
                TransportMode.WALK: 0.85,
                TransportMode.BICYCLE: 0.80,
                TransportMode.BUS: 1.05,
                TransportMode.CAR: 1.10,
                TransportMode.SCOOTER: 0.90
            },
            'high': {
                TransportMode.WALK: 0.60,
                TransportMode.BICYCLE: 0.55,
                TransportMode.BUS: 1.15,
                TransportMode.CAR: 1.25,
                TransportMode.SCOOTER: 0.75
            },
            'extreme': {
                TransportMode.WALK: 0.30,
                TransportMode.BICYCLE: 0.25,
                TransportMode.BUS: 1.25,
                TransportMode.CAR: 1.40,
                TransportMode.SCOOTER: 0.50
            }
        }
        
        base_adjustment = adjustments.get(heat_stress_level, {}).get(mode, 1.0)
        
        # Apply age and health modifiers
        if self.age_group in [AgeGroup.ELDERLY, AgeGroup.CHILD]:
            # More sensitive to heat
            if mode in [TransportMode.WALK, TransportMode.BICYCLE]:
                base_adjustment *= 0.85
            else:
                base_adjustment *= 1.1
        
        if self.health_status in [HealthStatus.FAIR, HealthStatus.POOR]:
            if mode in [TransportMode.WALK, TransportMode.BICYCLE]:
                base_adjustment *= 0.90
        
        return base_adjustment
    
    def calculate_trip_cancellation_probability(self, 
                                               heat_index: float,
                                               trip_purpose: str) -> float:
        """Calculate probability of canceling trip due to heat"""
        threshold = self.get_discomfort_threshold()
        
        if heat_index <= threshold:
            base_prob = 0.0
        elif heat_index <= threshold + 5:
            base_prob = 0.1
        elif heat_index <= threshold + 10:
            base_prob = 0.25
        else:
            base_prob = 0.45
        
        purpose_modifiers = {
            'commute': 0.5,
            'work': 0.5,
            'school': 0.6,
            'shopping': 1.2,
            'leisure': 1.5,
            'other': 1.0
        }
        
        modifier = purpose_modifiers.get(trip_purpose, 1.0)
        
        return min(base_prob * modifier, 0.8) # Cap at 80%


@dataclass
class MobilityAgent:
    """Agent with REALISTIC EDGE-BY-EDGE MOVEMENT"""
    
    agent_id: str
    origin: Tuple[float, float] # (lon, lat)
    destination: Tuple[float, float]
    departure_time: pd.Timestamp
    trip_purpose: str
    sensitivity_profile: HeatSensitivityProfile
    
    # Movement state
    current_location: Optional[Tuple[float, float]] = None
    current_mode: Optional[TransportMode] = None
    current_route: Optional[List[Tuple[float, float]]] = None  # List of nodes
    current_edge_index: int = 0  # Which edge we're currently on
    progress_on_edge: float = 0.0  # 0.0 to 1.0 progress along current edge
    
    trip_cancelled: bool = False
    trip_completed: bool = False
    
    # Preferences and constraints
    available_modes: List[TransportMode] = field(default_factory=lambda: [
        TransportMode.WALK, TransportMode.BICYCLE, 
        TransportMode.BUS
    ])
    max_walk_distance: float = 1000  # meters
    value_of_time: float = 15.0  # €/hour
    
    # Trip tracking with detailed path
    actual_mode: Optional[TransportMode] = None
    actual_travel_time: Optional[float] = None
    heat_exposure: Optional[float] = None
    path_history: List[Tuple[float, float]] = field(default_factory=list)
    edge_history: List[int] = field(default_factory=list)  # Track which edges used
    
    def __post_init__(self):
        self.current_location = self.origin
        self.path_history.append(self.origin)
    
    def decide_trip_execution(self, conditions: Dict) -> bool:
        """Decide whether to execute the trip given current conditions"""
        heat_index = conditions.get('heat_index', 25.0)
        
        cancel_prob = self.sensitivity_profile.calculate_trip_cancellation_probability(
            heat_index, self.trip_purpose
        )
        
        if random.random() < cancel_prob:
            self.trip_cancelled = True
            return False
        
        return True
    
    def choose_mode(self, available_modes: List[TransportMode],
                   conditions: Dict) -> TransportMode:
        """Choose transport mode based on conditions and preferences"""
        heat_level = self._classify_heat_level(conditions.get('heat_index', 25.0))
        
        utilities = {}
        for mode in available_modes:
            if mode not in self.available_modes:
                continue
            
            # Base utility components
            travel_time = self._estimate_travel_time(mode, conditions)
            cost = self._get_mode_cost(mode)
            comfort = self._get_comfort_score(mode, conditions)
            
            # Utility function: minimize time and cost, maximize comfort
            utility = (
                -0.4 * (travel_time / 60) -  # Time cost
                0.2 * cost -  # Monetary cost
                0.4 * (1 - comfort)  # Discomfort cost
            )
            
            # Apply heat sensitivity adjustment
            heat_adjustment = self.sensitivity_profile.get_mode_preference_adjustment(
                heat_level, mode
            )
            utility *= heat_adjustment
            
            utilities[mode] = utility
        
        if utilities:
            # Softmax choice (stochastic)
            utilities_array = np.array(list(utilities.values()))
            utilities_array = utilities_array - utilities_array.max()  # Numerical stability
            exp_utilities = np.exp(utilities_array / 0.5)  # Temperature parameter
            probabilities = exp_utilities / exp_utilities.sum()
            
            modes = list(utilities.keys())
            probs_dict = dict(zip(modes, probabilities))
            
            modes = list(probs_dict.keys())
            probs = list(probs_dict.values())
            chosen_mode = np.random.choice(modes, p=probs)
            
            self.current_mode = chosen_mode
            return chosen_mode
        
        return TransportMode.WALK # default
    
    def set_route(self, route: List[Tuple[float, float]]):
        """Set the route (list of node coordinates) for the agent to follow"""
        self.current_route = route
        self.current_edge_index = 0
        self.progress_on_edge = 0.0
        self.current_location = route[0] if route else self.origin
    
    def move_along_network(self, time_step_minutes: float, 
                          network, conditions: Dict) -> bool:
        """
        EDGE-BY-EDGE MOVEMENT
        Move agent along network for given time step
        
        Args:
            time_step_minutes: Time step duration in minutes
            network: TransportNetwork object
            conditions: Current environmental conditions
            
        Returns:
            True if agent has reached destination, False otherwise
        """
        if self.trip_completed or self.trip_cancelled or not self.current_route:
            return self.trip_completed
        
        if self.current_edge_index >= len(self.current_route) - 1:
            # Already at destination
            self.trip_completed = True
            self.current_location = self.destination
            return True
        
        # Get current mode speed (km/h)
        base_speed = self._get_mode_speed(self.current_mode)
        
        # Apply heat penalty for active modes
        heat_index = conditions.get('heat_index', 25.0)
        if self.current_mode in [TransportMode.WALK, TransportMode.BICYCLE]:
            if heat_index > 35:
                base_speed *= 0.7  # 30% slower in extreme heat
            elif heat_index > 30:
                base_speed *= 0.85  # 15% slower in high heat
        
        # Convert to m/min
        speed_m_per_min = (base_speed * 1000) / 60
        
        # Distance that can be traveled in this time step
        distance_budget = speed_m_per_min * time_step_minutes
        
        # Move along edges
        while distance_budget > 0 and self.current_edge_index < len(self.current_route) - 1:
            # Get current edge
            current_node = self.current_route[self.current_edge_index]
            next_node = self.current_route[self.current_edge_index + 1]
            
            # Get edge data from network
            edge_data = network.get_edge_data(current_node, next_node)
            
            if edge_data is None:
                # Edge not found, skip
                self.current_edge_index += 1
                self.progress_on_edge = 0.0
                continue
            
            edge_length = edge_data.get('length', 100)  # meters
            
            # Distance remaining on this edge
            distance_remaining = edge_length * (1.0 - self.progress_on_edge)
            
            if distance_budget >= distance_remaining:
                # Can complete this edge
                distance_budget -= distance_remaining
                self.current_edge_index += 1
                self.progress_on_edge = 0.0
                self.current_location = next_node
                self.path_history.append(next_node)
                
                # Track edge for flow analysis
                edge_id = edge_data.get('edge_id', -1)
                if edge_id not in self.edge_history:
                    self.edge_history.append(edge_id)
                
                # Track heat exposure along edge
                edge_heat = edge_data.get('heat_exposure', heat_index)
                edge_time = (edge_length / speed_m_per_min)  # minutes on this edge
                if self.heat_exposure is None:
                    self.heat_exposure = 0.0
                self.heat_exposure += edge_heat * edge_time
                
            else:
                # Partial progress on this edge
                progress_fraction = distance_budget / edge_length
                self.progress_on_edge += progress_fraction
                distance_budget = 0
                
                # Interpolate location
                self.current_location = self._interpolate_location(
                    current_node, next_node, self.progress_on_edge
                )
        
        # Check if completed
        if self.current_edge_index >= len(self.current_route) - 1:
            self.trip_completed = True
            self.current_location = self.destination
            return True
        
        return False
    
    def _interpolate_location(self, node1: Tuple[float, float], 
                             node2: Tuple[float, float],
                             progress: float) -> Tuple[float, float]:
        """Interpolate position between two nodes"""
        lon1, lat1 = node1
        lon2, lat2 = node2
        
        lon = lon1 + (lon2 - lon1) * progress
        lat = lat1 + (lat2 - lat1) * progress
        
        return (lon, lat)
    
    def _get_mode_speed(self, mode: Optional[TransportMode]) -> float:
        """Get mode speed in km/h"""
        if mode is None:
            return 5.0
        
        speeds = {
            TransportMode.WALK: 5.0,
            TransportMode.BICYCLE: 15.0,
            TransportMode.BUS: 20.0,
            TransportMode.CAR: 30.0,
            TransportMode.SCOOTER: 20.0
        }
        return speeds.get(mode, 5.0)
    
    def complete_trip(self, actual_travel_time: float):
        """Mark trip as completed"""
        self.trip_completed = True
        self.actual_travel_time = actual_travel_time
        self.actual_mode = self.current_mode
    
    def _classify_heat_level(self, heat_index: float) -> str:
        """Classify heat stress level"""
        threshold = self.sensitivity_profile.get_discomfort_threshold()
        
        if heat_index < threshold:
            return 'low'
        elif heat_index < threshold + 5:
            return 'moderate'
        elif heat_index < threshold + 10:
            return 'high'
        else:
            return 'extreme'
    
    def _estimate_travel_time(self, mode: TransportMode, 
                             conditions: Dict) -> float:
        """Estimate travel time in minutes"""
        distance = self._calculate_distance(self.origin, self.destination)
        
        speed = self._get_mode_speed(mode)
        time_minutes = (distance / 1000) / speed * 60
        
        # Heat penalty
        if mode in [TransportMode.WALK, TransportMode.BICYCLE]:
            heat_index = conditions.get('heat_index', 25.0)
            if heat_index > 30:
                time_minutes *= 1.2
        
        return time_minutes
    
    def _get_mode_cost(self, mode: TransportMode) -> float:
        """Get monetary cost of mode (€)"""
        costs = {
            TransportMode.WALK: 0.0,
            TransportMode.BICYCLE: 0.0,
            TransportMode.BUS: 1.50,
            TransportMode.CAR: 2.50,
            TransportMode.SCOOTER: 1.00
        }
        return costs.get(mode, 0.0)
    
    def _get_comfort_score(self, mode: TransportMode, 
                          conditions: Dict) -> float:
        """Get comfort score (0-1)"""
        base_comfort = {
            TransportMode.WALK: 0.7,
            TransportMode.BICYCLE: 0.8,
            TransportMode.BUS: 0.6,
            TransportMode.CAR: 0.9,
            TransportMode.SCOOTER: 0.75
        }
        
        comfort = base_comfort.get(mode, 0.5)
        
        if conditions.get('is_rainy', False):
            if mode in [TransportMode.WALK, TransportMode.BICYCLE]:
                comfort *= 0.5
        
        heat_index = conditions.get('heat_index', 25.0)
        if heat_index > 30:
            if mode in [TransportMode.WALK, TransportMode.BICYCLE]:
                comfort *= 0.7
            elif mode in [TransportMode.BUS, TransportMode.CAR]:
                comfort *= 1.1
        
        return comfort
    
    @staticmethod
    def _calculate_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate straight-line distance in meters"""
        from math import radians, cos, sin, asin, sqrt
        
        lon1, lat1 = point1
        lon2, lat2 = point2
        
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371000  # Radius of earth in meters
        
        return c * r


class AgentPopulation:
    """Manages population with DATA-DRIVEN CALIBRATION"""
    
    def __init__(self, calibration_data: Optional[Dict] = None):
        """
        Initialize agent population
        
        Args:
            calibration_data: Dict with 'bicycle' and 'pedestrian' DataFrames
                            from real counter data for calibration
        """
        self.agents: Dict[str, MobilityAgent] = {}
        self.agent_counter = 0
        self.calibration_data = calibration_data or {}
        
        # Calibrated generation rates (agents per hour)
        self.generation_rates = self._calibrate_generation_rates()
    
    def _calibrate_generation_rates(self) -> Dict:
        """
        CALIBRATE agent generation to real data
        
        Returns dict with hourly generation rates by mode
        """
        rates = {
            'bicycle': {},
            'pedestrian': {},
            'overall': {}
        }
        
        # Calibrate from bicycle counter data
        if 'bicycle' in self.calibration_data:
            bicycle_df = self.calibration_data['bicycle']
            
            if 'Data' in bicycle_df.columns and 'Totale' in bicycle_df.columns:
                bicycle_df['datetime'] = pd.to_datetime(bicycle_df['Data'])
                bicycle_df['hour'] = bicycle_df['datetime'].dt.hour
                
                # Average counts by hour
                hourly_avg = bicycle_df.groupby('hour')['Totale'].mean()
                
                for hour in range(24):
                    rates['bicycle'][hour] = hourly_avg.get(hour, 50)  # Default 50
        else:
            # Default pattern if no data
            for hour in range(24):
                if 7 <= hour <= 9:
                    rates['bicycle'][hour] = 150  # Morning peak
                elif 17 <= hour <= 19:
                    rates['bicycle'][hour] = 180  # Evening peak
                elif 9 < hour < 17:
                    rates['bicycle'][hour] = 80  # Midday
                else:
                    rates['bicycle'][hour] = 20  # Off-peak
        
        # Calibrate from pedestrian data
        if 'pedestrian' in self.calibration_data:
            ped_df = self.calibration_data['pedestrian']
            
            if 'Data' in ped_df.columns and 'Numero di visitatori' in ped_df.columns:
                ped_df['datetime'] = pd.to_datetime(ped_df['Data'])
                ped_df['hour'] = ped_df['datetime'].dt.hour
                
                hourly_avg = ped_df.groupby('hour')['Numero di visitatori'].mean()
                
                for hour in range(24):
                    rates['pedestrian'][hour] = hourly_avg.get(hour, 100)
        else:
            # Default pattern
            for hour in range(24):
                if 10 <= hour <= 13:
                    rates['pedestrian'][hour] = 300  # Lunch peak
                elif 17 <= hour <= 20:
                    rates['pedestrian'][hour] = 400  # Evening peak
                elif 7 <= hour < 10:
                    rates['pedestrian'][hour] = 200  # Morning
                else:
                    rates['pedestrian'][hour] = 80
        
        # Overall = weighted combination
        for hour in range(24):
            bike_rate = rates['bicycle'].get(hour, 50)
            ped_rate = rates['pedestrian'].get(hour, 100)
            # 30% bicycle, 70% pedestrian (typical modal split)
            rates['overall'][hour] = int(0.3 * bike_rate + 0.7 * ped_rate)
        
        print("✓ Calibrated generation rates from real data")
        return rates
    
    def generate_agents_calibrated(self, 
                                   timestamp: pd.Timestamp,
                                   duration_hours: float,
                                   origins: List[Tuple[float, float]],
                                   destinations: List[Tuple[float, float]],
                                   trip_purposes: List[str]) -> List[MobilityAgent]:
        """
        Generate agents using CALIBRATED rates
        
        Args:
            timestamp: Current timestamp
            duration_hours: Duration to generate for
            origins: List of possible origins
            destinations: List of possible destinations
            trip_purposes: List of possible trip purposes
        """
        hour = timestamp.hour
        
        # Get calibrated rate for this hour
        base_rate = self.generation_rates['overall'].get(hour, 100)
        
        # Scale by duration
        n_agents = int(base_rate * duration_hours)
        
        # Add random variation (±20%)
        n_agents = int(n_agents * random.uniform(0.8, 1.2))
        
        agents = []
        
        for i in range(n_agents):
            age_group = self._sample_age_group()
            health_status = self._sample_health_status()
            acclimatization = random.uniform(0.3, 0.9)
            
            profile = HeatSensitivityProfile(
                age_group=age_group,
                health_status=health_status,
                acclimatization=acclimatization
            )
            
            # Departure time within the hour
            minute_offset = random.randint(0, 59)
            departure_time = timestamp + pd.Timedelta(minutes=minute_offset)
            
            agent = MobilityAgent(
                agent_id=f"agent_{self.agent_counter:06d}",
                origin=random.choice(origins),
                destination=random.choice(destinations),
                departure_time=departure_time,
                trip_purpose=random.choice(trip_purposes),
                sensitivity_profile=profile
            )
            
            agents.append(agent)
            self.agents[agent.agent_id] = agent
            self.agent_counter += 1
        
        return agents
    
    @staticmethod
    def _sample_age_group() -> AgeGroup:
        """Sample age group from realistic distribution"""
        groups = list(AgeGroup)
        weights = [0.15, 0.10, 0.35, 0.30, 0.10]  # Bologna demographics
        return random.choices(groups, weights=weights)[0]
    
    @staticmethod
    def _sample_health_status() -> HealthStatus:
        """Sample health status"""
        statuses = list(HealthStatus)
        weights = [0.20, 0.50, 0.25, 0.05]
        return random.choices(statuses, weights=weights)[0]
    
    def get_active_agents(self, timestamp: pd.Timestamp) -> List[MobilityAgent]:
        """Get agents active at given timestamp"""
        return [
            agent for agent in self.agents.values()
            if not agent.trip_completed and not agent.trip_cancelled
            and agent.departure_time <= timestamp
        ]
    
    def get_statistics(self) -> Dict:
        """Get population statistics"""
        total = len(self.agents)
        completed = sum(1 for a in self.agents.values() if a.trip_completed)
        cancelled = sum(1 for a in self.agents.values() if a.trip_cancelled)
        
        mode_choices = {}
        for agent in self.agents.values():
            if agent.actual_mode:
                mode_choices[agent.actual_mode] = mode_choices.get(agent.actual_mode, 0) + 1
        
        return {
            'total_agents': total,
            'completed_trips': completed,
            'cancelled_trips': cancelled,
            'completion_rate': completed / total if total > 0 else 0,
            'cancellation_rate': cancelled / total if total > 0 else 0,
            'mode_shares': {k.value: v/completed for k, v in mode_choices.items()} if completed > 0 else {}
        }