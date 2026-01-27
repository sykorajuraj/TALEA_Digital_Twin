"""
TALEA - Civic Digital Twin: Environment Models
File: src/simulation/models/environment.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Defines the urban environment for mobility simulation.

Working interventions:
 - cooling stations
 - shade
 - tree planting
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
from shapely.geometry import Point, LineString
from dataclasses import dataclass, field
from enum import Enum


class ZoneType(Enum):
    """Urban zone types"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    ZTL = "ztl" # Pedestrian zone
    PARK = "park"


@dataclass
class WeatherConditions:
    """Weather conditions at a point in time"""
    timestamp: pd.Timestamp
    temperature: float
    heat_index: float
    humidity: Optional[float] = None
    precipitation: float = 0.0
    wind_speed: Optional[float] = None
    is_rainy: bool = False
    heat_stress_level: str = 'low'
    
    @classmethod
    def from_weather_data(cls, weather_row: pd.Series, timestamp: pd.Timestamp):
        """Create from weather dataframe row"""
        return cls(
            timestamp=timestamp,
            temperature=weather_row.get('temperature', 25.0),
            heat_index=weather_row.get('heat_index', weather_row.get('temperature', 25.0)),
            humidity=weather_row.get('humidity'),
            precipitation=weather_row.get('precipitation_mm', 0.0),
            wind_speed=weather_row.get('wind_speed'),
            is_rainy=weather_row.get('precipitation_mm', 0.0) > 0.1,
            heat_stress_level=weather_row.get('heat_index_level', 'low')
        )


@dataclass
class Intervention:
    """Represents an urban intervention"""
    intervention_id: str
    intervention_type: str  # 'cooling_station', 'tree_planting', 'shade_structure'
    location: Tuple[float, float]
    parameters: Dict
    active: bool = True
    
    # Effects
    temperature_reduction: float = 0.0  # °C reduction in vicinity
    heat_index_reduction: float = 0.0  # Heat index reduction
    radius_of_effect: float = 100.0  # meters
    capacity: Optional[int] = None  # For cooling stations


class TransportNetwork:
    """Transport network with intervention support"""
    
    def __init__(self, street_network: gpd.GeoDataFrame,
                 capacity_constraints: Optional[Dict] = None):
        """Initialize transport network"""
        self.street_network = street_network.copy()
        self.capacity_constraints = capacity_constraints or {}
        self.graph = self._build_graph()
        
        # Dynamic state
        self.current_flows = {} # edge_id -> {mode: flow}
        self.congestion_levels = {} # edge_id -> congestion_factor
        
        # Interventions affecting network
        self.interventions: Dict[str, Intervention] = {}
        self.affected_edges: Dict[str, List[int]] = {}  # intervention_id -> edge_ids
    
    def _build_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph from street network"""
        G = nx.MultiDiGraph()
        
        for idx, row in self.street_network.iterrows():
            if row.geometry.type == 'LineString':
                coords = list(row.geometry.coords)
                start = coords[0]
                end = coords[-1]
                
                # Edge attributes
                attrs = {
                    'edge_id': idx,
                    'geometry': row.geometry,
                    'length': row.geometry.length,
                    'heat_exposure': row.get('heat_exposure', 25.0),
                    'cooling_corridor': row.get('is_cooling_corridor', False),
                    # Track intervention effects
                    'heat_reduction': 0.0,
                    'has_shade': False,
                    'has_trees': False
                }
                
                G.add_edge(start, end, **attrs)
                G.add_edge(end, start, **attrs)
        
        return G
    
    def get_edge_data(self, node1: Tuple[float, float], 
                     node2: Tuple[float, float]) -> Optional[Dict]:
        """Get edge data between two nodes"""
        if self.graph.has_edge(node1, node2):
            edge_data = self.graph[node1][node2][0]
            return edge_data
        return None
    
    def apply_intervention_to_edges(self, intervention: Intervention):
        """
        Apply intervention effects to nearby edges
        """
        intervention_location = Point(intervention.location)
        affected_edges = []
        
        # Find edges within radius
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_geom = data.get('geometry')
            if edge_geom is None:
                continue
            
            # Check distance from intervention to edge
            distance = edge_geom.distance(intervention_location)
            # Convert to meters (approximate)
            distance_m = distance * 111000
            
            if distance_m <= intervention.radius_of_effect:
                edge_id = data.get('edge_id')
                affected_edges.append(edge_id)
                
                # Apply intervention effects based on type
                if intervention.intervention_type == 'tree_planting':
                    # Trees reduce heat exposure
                    data['has_trees'] = True
                    data['heat_reduction'] = intervention.temperature_reduction
                    data['heat_exposure'] = max(
                        data['heat_exposure'] - intervention.temperature_reduction,
                        15.0  # Min temperature
                    )
                    # Improve as cooling corridor
                    data['cooling_corridor'] = True
                
                elif intervention.intervention_type == 'shade_structure':
                    # Shade structures reduce heat
                    data['has_shade'] = True
                    data['heat_reduction'] = intervention.temperature_reduction
                    data['heat_exposure'] = max(
                        data['heat_exposure'] - intervention.temperature_reduction,
                        15.0
                    )
                
                elif intervention.intervention_type == 'cooling_station':
                    # Cooling stations are POI, but mark nearby edges as cooler
                    data['heat_reduction'] = intervention.temperature_reduction * 0.5
                    data['heat_exposure'] = max(
                        data['heat_exposure'] - intervention.temperature_reduction * 0.5,
                        15.0
                    )
        
        # Store affected edges
        self.affected_edges[intervention.intervention_id] = affected_edges
        
        print(f"  → Intervention {intervention.intervention_id} affects {len(affected_edges)} edges")
    
    def remove_intervention_from_edges(self, intervention_id: str):
        """Remove intervention effects from edges"""
        if intervention_id not in self.affected_edges:
            return
        
        affected = self.affected_edges[intervention_id]
        
        # Revert changes (simplified - assumes single intervention per edge)
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_id = data.get('edge_id')
            if edge_id in affected:
                data['has_trees'] = False
                data['has_shade'] = False
                data['heat_reduction'] = 0.0
                # Would need to restore original heat_exposure
        
        del self.affected_edges[intervention_id]
    
    def find_shortest_path(self, origin: Tuple[float, float],
                          destination: Tuple[float, float],
                          mode: str = 'walk',
                          preferences: Optional[Dict] = None) -> Optional[List]:
        """Find shortest path considering mode and preferences"""
        preferences = preferences or {'minimize': 'time'}
        
        origin_node = self._find_nearest_node(origin)
        dest_node = self._find_nearest_node(destination)
        
        if origin_node is None or dest_node is None:
            return None
        
        weight_attr = self._get_weight_attribute(mode, preferences)
        
        try:
            path = nx.shortest_path(
                self.graph, 
                origin_node, 
                dest_node,
                weight=weight_attr
            )
            return path
        except nx.NetworkXNoPath:
            return None
    
    def _get_weight_attribute(self, mode: str, preferences: Dict) -> str:
        """Determine edge weight attribute"""
        minimize = preferences.get('minimize', 'time')
        
        if minimize == 'time':
            return 'length'  # Simplified - length proxy for time
        elif minimize == 'heat':
            return 'heat_exposure'
        elif minimize == 'balanced':
            # Would need to compute balanced weight
            return 'length'
        else:
            return 'length'
    
    def _find_nearest_node(self, location: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find nearest node to location"""
        min_dist = float('inf')
        nearest = None
        
        target = Point(location)
        
        for node in self.graph.nodes():
            node_point = Point(node)
            dist = target.distance(node_point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def get_available_modes_at(self, location: Tuple[float, float],
                              zone_type: Optional[str] = None) -> List[str]:
        """Get available transport modes at location"""
        modes = ['walk']
        
        if zone_type == 'ztl':
            modes.extend(['bicycle', 'bus'])
        else:
            modes.extend(['bicycle', 'bus', 'car', 'scooter'])
        
        return modes


class UrbanEnvironment:
    """Urban environment with WORKING INTERVENTIONS"""
    
    def __init__(self, street_network: gpd.GeoDataFrame,
                 weather_data: pd.DataFrame,
                 poi_data: Optional[gpd.GeoDataFrame] = None,
                 ztl_zones: Optional[gpd.GeoDataFrame] = None,
                 spatial_grid: Optional[gpd.GeoDataFrame] = None):
        """Initialize urban environment"""
        self.network = TransportNetwork(street_network)
        self.weather_data = weather_data.copy()
        self.poi_data = poi_data.copy() if poi_data is not None else None
        self.ztl_zones = ztl_zones
        self.spatial_grid = spatial_grid.copy() if spatial_grid is not None else None
        
        # Ensure datetime index for weather
        if 'datetime' in self.weather_data.columns:
            self.weather_data = self.weather_data.set_index('datetime')
        
        # Current state
        self.current_time: Optional[pd.Timestamp] = None
        self.current_weather: Optional[WeatherConditions] = None
        
        # Interventions
        self.interventions: Dict[str, Intervention] = {}
        self.intervention_counter = 0
        
        # Track cooling station usage
        self.cooling_station_usage: Dict[str, int] = {}
    
    def add_intervention(self, intervention_type: str, 
                        location: Tuple[float, float],
                        parameters: Optional[Dict] = None):
        """
        Add intervention to environment - ACTUALLY WORKS NOW
        
        Args:
            intervention_type: 'cooling_station', 'tree_planting', 'shade_structure'
            location: (lon, lat)
            parameters: Additional parameters
        """
        parameters = parameters or {}
        
        # Create intervention with appropriate effects
        if intervention_type == 'cooling_station':
            temp_reduction = parameters.get('temperature_reduction', 5.0)
            radius = parameters.get('radius', 200.0)
            capacity = parameters.get('capacity', 100)
            
            intervention = Intervention(
                intervention_id=f"cooling_{self.intervention_counter}",
                intervention_type=intervention_type,
                location=location,
                parameters=parameters,
                temperature_reduction=temp_reduction,
                heat_index_reduction=temp_reduction * 1.2,  # Heat index reduces more
                radius_of_effect=radius,
                capacity=capacity
            )
            
            # Add to POI
            if self.poi_data is not None:
                new_poi = gpd.GeoDataFrame({
                    'poi_category': ['cooling_station'],
                    'name': [f'Cooling Station {self.intervention_counter}'],
                    'capacity': [capacity],
                    'geometry': [Point(location)]
                }, crs=self.poi_data.crs)
                
                self.poi_data = pd.concat([self.poi_data, new_poi], ignore_index=True)
            
            self.cooling_station_usage[intervention.intervention_id] = 0
        
        elif intervention_type == 'tree_planting':
            temp_reduction = parameters.get('temperature_reduction', 3.0)
            radius = parameters.get('radius', 150.0)
            
            intervention = Intervention(
                intervention_id=f"trees_{self.intervention_counter}",
                intervention_type=intervention_type,
                location=location,
                parameters=parameters,
                temperature_reduction=temp_reduction,
                heat_index_reduction=temp_reduction * 1.1,
                radius_of_effect=radius
            )
        
        elif intervention_type == 'shade_structure':
            temp_reduction = parameters.get('temperature_reduction', 4.0)
            radius = parameters.get('radius', 100.0)
            
            intervention = Intervention(
                intervention_id=f"shade_{self.intervention_counter}",
                intervention_type=intervention_type,
                location=location,
                parameters=parameters,
                temperature_reduction=temp_reduction,
                heat_index_reduction=temp_reduction * 1.15,
                radius_of_effect=radius
            )
        
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        
        # Store intervention
        self.interventions[intervention.intervention_id] = intervention
        self.network.interventions[intervention.intervention_id] = intervention
        
        # Apply to network edges
        self.network.apply_intervention_to_edges(intervention)
        
        # Apply to spatial grid
        if self.spatial_grid is not None:
            self._apply_intervention_to_grid(intervention)
        
        self.intervention_counter += 1
        
        print(f"✓ Added {intervention_type} at {location}")
        print(f"  Temperature reduction: {intervention.temperature_reduction}°C")
        print(f"  Radius of effect: {intervention.radius_of_effect}m")
    
    def _apply_intervention_to_grid(self, intervention: Intervention):
        """Apply intervention effects to spatial grid"""
        intervention_location = Point(intervention.location)
        
        for idx, cell in self.spatial_grid.iterrows():
            # Check if cell is within radius
            distance = cell.geometry.distance(intervention_location)
            distance_m = distance * 111000  # Approximate
            
            if distance_m <= intervention.radius_of_effect:
                # Reduce heat metrics
                if 'avg_temp' in self.spatial_grid.columns:
                    reduction = intervention.temperature_reduction * (
                        1.0 - distance_m / intervention.radius_of_effect
                    )
                    self.spatial_grid.at[idx, 'avg_temp'] = max(
                        cell['avg_temp'] - reduction,
                        15.0
                    )
                
                if 'heat_island' in self.spatial_grid.columns:
                    # May no longer be a heat island
                    if self.spatial_grid.at[idx, 'avg_temp'] < self.spatial_grid['avg_temp'].quantile(0.75):
                        self.spatial_grid.at[idx, 'is_heat_island'] = False
    
    def remove_intervention(self, intervention_id: str):
        """Remove intervention from environment"""
        if intervention_id not in self.interventions:
            return
        
        intervention = self.interventions[intervention_id]
        
        # Remove from network
        self.network.remove_intervention_from_edges(intervention_id)
        del self.network.interventions[intervention_id]
        
        # Remove from POI (if cooling station)
        if intervention.intervention_type == 'cooling_station' and self.poi_data is not None:
            # Find and remove
            poi_location = Point(intervention.location)
            distances = self.poi_data.geometry.distance(poi_location)
            if distances.min() < 0.0001:  # Very close
                idx_to_remove = distances.idxmin()
                self.poi_data = self.poi_data.drop(idx_to_remove)
        
        # Remove from interventions dict
        del self.interventions[intervention_id]
        
        print(f"✓ Removed intervention {intervention_id}")
    
    def get_conditions_at(self, location: Tuple[float, float],
                         timestamp: pd.Timestamp) -> Dict:
        """
        Get environmental conditions - NOW INCLUDES INTERVENTION EFFECTS
        """
        conditions = {}
        
        # Base weather
        weather = self._get_weather_at_time(timestamp)
        conditions['temperature'] = weather.temperature
        conditions['heat_index'] = weather.heat_index
        conditions['humidity'] = weather.humidity
        conditions['precipitation'] = weather.precipitation
        conditions['is_rainy'] = weather.is_rainy
        conditions['heat_stress_level'] = weather.heat_stress_level
        
        # Apply intervention effects
        intervention_reduction = self._get_intervention_effect_at(location)
        conditions['temperature'] -= intervention_reduction['temperature']
        conditions['heat_index'] -= intervention_reduction['heat_index']
        
        # Spatial conditions
        if self.spatial_grid is not None:
            spatial_data = self._get_grid_cell_data(location)
            conditions.update(spatial_data)
        
        # Zone type
        conditions['zone_type'] = self._get_zone_type(location)
        conditions['in_ztl'] = self._is_in_ztl(location)
        
        # Nearby cooling stations
        conditions['nearby_cooling_stations'] = self._get_nearby_cooling_stations(location)
        
        return conditions
    
    def _get_intervention_effect_at(self, location: Tuple[float, float]) -> Dict:
        """Calculate cumulative intervention effects at location"""
        total_temp_reduction = 0.0
        total_hi_reduction = 0.0
        
        location_point = Point(location)
        
        for intervention in self.interventions.values():
            if not intervention.active:
                continue
            
            intervention_point = Point(intervention.location)
            distance = location_point.distance(intervention_point) * 111000  # to meters
            
            if distance <= intervention.radius_of_effect:
                # Effect decreases with distance
                distance_factor = 1.0 - (distance / intervention.radius_of_effect)
                
                total_temp_reduction += intervention.temperature_reduction * distance_factor
                total_hi_reduction += intervention.heat_index_reduction * distance_factor
        
        return {
            'temperature': total_temp_reduction,
            'heat_index': total_hi_reduction
        }
    
    def _get_nearby_cooling_stations(self, location: Tuple[float, float],
                                    radius: float = 500.0) -> List[Dict]:
        """Get nearby cooling stations"""
        stations = []
        
        location_point = Point(location)
        
        for intervention_id, intervention in self.interventions.items():
            if intervention.intervention_type != 'cooling_station':
                continue
            
            intervention_point = Point(intervention.location)
            distance = location_point.distance(intervention_point) * 111000
            
            if distance <= radius:
                current_usage = self.cooling_station_usage.get(intervention_id, 0)
                capacity = intervention.capacity or 100
                
                stations.append({
                    'id': intervention_id,
                    'location': intervention.location,
                    'distance': distance,
                    'capacity': capacity,
                    'current_usage': current_usage,
                    'available_capacity': max(0, capacity - current_usage)
                })
        
        # Sort by distance
        stations.sort(key=lambda x: x['distance'])
        
        return stations
    
    def use_cooling_station(self, station_id: str) -> bool:
        """
        Register cooling station usage
        
        Returns True if capacity available, False if full
        """
        if station_id not in self.interventions:
            return False
        
        intervention = self.interventions[station_id]
        
        if intervention.intervention_type != 'cooling_station':
            return False
        
        current_usage = self.cooling_station_usage.get(station_id, 0)
        capacity = intervention.capacity or 100
        
        if current_usage < capacity:
            self.cooling_station_usage[station_id] = current_usage + 1
            return True
        
        return False
    
    def reset_cooling_station_usage(self):
        """Reset cooling station usage counters"""
        for station_id in self.cooling_station_usage.keys():
            self.cooling_station_usage[station_id] = 0
    
    def get_available_modes_at(self, location: Tuple[float, float],
                              timestamp: pd.Timestamp) -> List[str]:
        """Get available transport modes at location"""
        zone_type = self._get_zone_type(location)
        return self.network.get_available_modes_at(location, zone_type)
    
    def update_weather(self, timestamp: pd.Timestamp):
        """Update current weather conditions"""
        self.current_time = timestamp
        self.current_weather = self._get_weather_at_time(timestamp)
    
    def _get_weather_at_time(self, timestamp: pd.Timestamp) -> WeatherConditions:
        """Get weather conditions at specific time"""
        if timestamp in self.weather_data.index:
            weather_row = self.weather_data.loc[timestamp]
        else:
            nearest_idx = self.weather_data.index.get_indexer([timestamp], method='nearest')[0]
            weather_row = self.weather_data.iloc[nearest_idx]
        
        return WeatherConditions.from_weather_data(weather_row, timestamp)
    
    def _get_grid_cell_data(self, location: Tuple[float, float]) -> Dict:
        """Get data from spatial grid cell containing location"""
        point = Point(location)
        
        cell = self.spatial_grid[self.spatial_grid.contains(point)]
        
        if len(cell) > 0:
            cell_data = cell.iloc[0]
            return {
                'grid_id': cell_data.get('grid_id'),
                'vulnerability_index': cell_data.get('vulnerability_index', 0.0),
                'heat_island': cell_data.get('is_heat_island', False),
                'poi_density': cell_data.get('poi_density', 0.0)
            }
        
        return {}
    
    def _get_zone_type(self, location: Tuple[float, float]) -> Optional[str]:
        """Determine zone type at location"""
        if self._is_in_ztl(location):
            return 'ztl'
        return 'residential'
    
    def _is_in_ztl(self, location: Tuple[float, float]) -> bool:
        """Check if location is in ZTL zone"""
        if self.ztl_zones is None:
            return False
        
        point = Point(location)
        in_ztl = self.ztl_zones.contains(point).any()
        return in_ztl
    
    def get_intervention_statistics(self) -> Dict:
        """Get statistics about interventions"""
        stats = {
            'total_interventions': len(self.interventions),
            'by_type': {},
            'total_affected_edges': 0,
            'cooling_station_usage': {}
        }
        
        # Count by type
        for intervention in self.interventions.values():
            itype = intervention.intervention_type
            stats['by_type'][itype] = stats['by_type'].get(itype, 0) + 1
        
        # Affected edges
        for edges in self.network.affected_edges.values():
            stats['total_affected_edges'] += len(edges)
        
        # Cooling station usage
        for station_id, usage in self.cooling_station_usage.items():
            if station_id in self.interventions:
                capacity = self.interventions[station_id].capacity or 100
                stats['cooling_station_usage'][station_id] = {
                    'usage': usage,
                    'capacity': capacity,
                    'utilization_rate': usage / capacity if capacity > 0 else 0
                }
        
        return stats
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of environment"""
        stats = {
            'network': {
                'num_nodes': self.network.graph.number_of_nodes(),
                'num_edges': self.network.graph.number_of_edges(),
                'total_length_km': sum(
                    data['length'] for u, v, data in self.network.graph.edges(data=True)
                ) / 1000
            }
        }
        
        if self.weather_data is not None:
            stats['weather'] = {
                'time_range': f"{self.weather_data.index.min()} to {self.weather_data.index.max()}",
                'avg_temperature': self.weather_data.get('temperature', pd.Series()).mean(),
                'max_heat_index': self.weather_data.get('heat_index', pd.Series()).max()
            }
        
        if self.poi_data is not None:
            stats['poi'] = {
                'total_poi': len(self.poi_data),
                'categories': self.poi_data.get('poi_category', pd.Series()).nunique()
            }
        stats['interventions'] = self.get_intervention_statistics()
        
        return stats