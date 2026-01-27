"""
TALEA - Civic Digital Twin: Route Impact Analyzer
File: src/analysis/heat_impact/route_impact_analyzer.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Analyzes heat exposure by route and identifies cooling corridors.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Optional, Tuple
from shapely.geometry import LineString, Point
import networkx as nx


class RouteImpactAnalyzer:
    """Analyzes heat exposure on routes and identifies cooling corridors"""
    
    def __init__(self):
        """Initialize route impact analyzer"""
        pass
    
    def compute_heat_exposure_by_route(self,
                                      network: gpd.GeoDataFrame,
                                      heat_grid: gpd.GeoDataFrame,
                                      heat_col: str = 'avg_temp') -> gpd.GeoDataFrame:
        """
        Compute heat exposure for each route segment
        
        Args:
            network: Street network GeoDataFrame
            heat_grid: Spatial grid with heat data
            heat_col: Column with heat values
            
        Returns:
            Network with heat exposure scores
        """
        network = network.copy()
        
        if heat_col not in heat_grid.columns:
            raise ValueError(f"Heat column '{heat_col}' not found in heat_grid")
        
        # Ensure same CRS
        if network.crs != heat_grid.crs:
            heat_grid = heat_grid.to_crs(network.crs)
        
        # Spatial join to find heat exposure
        network_with_heat = gpd.sjoin(
            network,
            heat_grid[['geometry', heat_col, 'grid_id']],
            how='left',
            predicate='intersects'
        )
        
        # Aggregate if multiple grid cells intersect
        if len(network_with_heat) > len(network):
            heat_by_segment = network_with_heat.groupby(network_with_heat.index)[heat_col].mean()
            network['heat_exposure'] = heat_by_segment
        else:
            network['heat_exposure'] = network_with_heat[heat_col]
        
        # Fill missing values with median
        network['heat_exposure'] = network['heat_exposure'].fillna(
            network['heat_exposure'].median()
        )
        
        # Classify exposure levels
        network['exposure_level'] = pd.cut(
            network['heat_exposure'],
            bins=[network['heat_exposure'].min(),
                  network['heat_exposure'].quantile(0.33),
                  network['heat_exposure'].quantile(0.67),
                  network['heat_exposure'].max()],
            labels=['low', 'moderate', 'high'],
            include_lowest=True
        )
        
        print(f"  ✓ Computed heat exposure for {len(network)} route segments")
        
        return network
    
    def identify_cooling_corridors(self,
                                  network: gpd.GeoDataFrame,
                                  tree_coverage: Optional[gpd.GeoDataFrame] = None,
                                  shade_data: Optional[gpd.GeoDataFrame] = None,
                                  heat_threshold: Optional[float] = None) -> gpd.GeoDataFrame:
        """
        Identify cooling corridors with low heat exposure
        
        Args:
            network: Street network with heat exposure
            tree_coverage: Tree/vegetation coverage data (optional)
            shade_data: Shade data (optional)
            heat_threshold: Maximum heat for cooling corridor
            
        Returns:
            Network with cooling corridor flags
        """
        network = network.copy()
        
        # Determine heat threshold (bottom 25% if not specified)
        if heat_threshold is None and 'heat_exposure' in network.columns:
            heat_threshold = network['heat_exposure'].quantile(0.25)
        
        # Base criteria: low heat exposure
        if 'heat_exposure' in network.columns:
            network['is_cooling_corridor'] = network['heat_exposure'] <= heat_threshold
        else:
            network['is_cooling_corridor'] = False
        
        # Compute cooling score (0-1)
        network['cooling_score'] = self._compute_cooling_score(network)
        
        cooling_count = network['is_cooling_corridor'].sum()
        cooling_pct = (cooling_count / len(network)) * 100
        
        print(f"  ✓ Identified {cooling_count} cooling corridor segments ({cooling_pct:.1f}%)")
        
        return network
    
    def _compute_cooling_score(self, network: gpd.GeoDataFrame) -> pd.Series:
        """Compute composite cooling score"""
        
        components = []
        weights = []
        
        # Low heat exposure
        if 'heat_exposure' in network.columns:
            # Invert and normalize (lower heat = higher score)
            max_heat = network['heat_exposure'].max()
            min_heat = network['heat_exposure'].min()
            if max_heat > min_heat:
                heat_score = 1 - (network['heat_exposure'] - min_heat) / (max_heat - min_heat)
            else:
                heat_score = pd.Series(0.5, index=network.index)
            components.append(heat_score)
            weights.append(0.5)
        
        if not components:
            return pd.Series(0, index=network.index)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted average
        cooling_score = sum(comp * weight for comp, weight in zip(components, weights))
        
        return cooling_score
    
    def suggest_alternative_routes(self,
                                  origin: Tuple[float, float],
                                  destination: Tuple[float, float],
                                  network: gpd.GeoDataFrame,
                                  heat_grid: gpd.GeoDataFrame,
                                  n_routes: int = 3) -> List[Dict]:
        """
        Suggest alternative routes optimizing for cooling
        
        Args:
            origin: (lon, lat) of origin
            destination: (lon, lat) of destination
            network: Street network with heat exposure
            heat_grid: Spatial grid with heat data
            n_routes: Number of alternative routes to suggest
            
        Returns:
            List of route dictionaries with paths and metrics
        """
        # Convert network to graph
        G = self._network_to_graph(network)
        
        # Find nearest nodes
        origin_node = self._find_nearest_node(origin, network)
        dest_node = self._find_nearest_node(destination, network)
        
        if origin_node is None or dest_node is None:
            print("  ⚠ Could not find route endpoints")
            return []
        
        # Find multiple paths
        routes = []
        
        try:
            # Shortest path (distance)
            shortest_path = nx.shortest_path(G, origin_node, dest_node, weight='length')
            routes.append({
                'type': 'shortest',
                'path': shortest_path,
                'metrics': self._compute_route_metrics(shortest_path, network)
            })
        except nx.NetworkXNoPath:
            print("  ⚠ No path found")
            return []
        
        # Coolest path (minimize heat exposure)
        if 'heat_exposure' in network.columns:
            try:
                coolest_path = nx.shortest_path(G, origin_node, dest_node, weight='heat_weight')
                routes.append({
                    'type': 'coolest',
                    'path': coolest_path,
                    'metrics': self._compute_route_metrics(coolest_path, network)
                })
            except nx.NetworkXNoPath:
                pass
        
        # Balanced path (compromise between distance and heat)
        if 'heat_exposure' in network.columns:
            try:
                balanced_path = nx.shortest_path(G, origin_node, dest_node, weight='balanced_weight')
                routes.append({
                    'type': 'balanced',
                    'path': balanced_path,
                    'metrics': self._compute_route_metrics(balanced_path, network)
                })
            except nx.NetworkXNoPath:
                pass
        
        print(f"  ✓ Generated {len(routes)} alternative routes")
        
        return routes[:n_routes]
    
    def _network_to_graph(self, network: gpd.GeoDataFrame) -> nx.Graph:
        """Convert network to NetworkX graph"""
        
        G = nx.Graph()
        
        for idx, row in network.iterrows():
            if row.geometry.type == 'LineString':
                coords = list(row.geometry.coords)
                start = coords[0]
                end = coords[-1]
                
                # Edge attributes
                length = row.geometry.length if hasattr(row.geometry, 'length') else 1
                
                edge_attrs = {'length': length}
                
                if 'heat_exposure' in row.index:
                    heat = row['heat_exposure']
                    edge_attrs['heat_exposure'] = heat
                    edge_attrs['heat_weight'] = heat * length  # Heat-weighted distance
                    edge_attrs['balanced_weight'] = 0.5 * length + 0.5 * heat * length
                
                G.add_edge(start, end, **edge_attrs)
        
        return G
    
    def _find_nearest_node(self,
                          point: Tuple[float, float],
                          network: gpd.GeoDataFrame) -> Optional[Tuple[float, float]]:
        """Find nearest network node to a point"""
        
        target = Point(point)
        min_dist = float('inf')
        nearest_node = None
        
        for idx, row in network.iterrows():
            if row.geometry.type == 'LineString':
                coords = list(row.geometry.coords)
                
                for coord in [coords[0], coords[-1]]:
                    dist = target.distance(Point(coord))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = coord
        
        return nearest_node
    
    def _compute_route_metrics(self,
                               path: List[Tuple[float, float]],
                               network: gpd.GeoDataFrame) -> Dict:
        """Compute metrics for a route path"""
        
        total_distance = 0
        total_heat_exposure = 0
        segment_count = 0
        
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # Find matching network segment
            for idx, row in network.iterrows():
                if row.geometry.type == 'LineString':
                    coords = list(row.geometry.coords)
                    if (coords[0] == start and coords[-1] == end) or \
                       (coords[0] == end and coords[-1] == start):
                        total_distance += row.geometry.length
                        if 'heat_exposure' in row.index:
                            total_heat_exposure += row['heat_exposure']
                        segment_count += 1
                        break
        
        avg_heat = total_heat_exposure / segment_count if segment_count > 0 else 0
        
        return {
            'total_distance_m': total_distance,
            'avg_heat_exposure': avg_heat,
            'segment_count': segment_count
        }