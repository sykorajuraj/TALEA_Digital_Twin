"""
TALEA - Civic Digital Twin: Geospatial Preprocessing Module
File: src/data_preprocessing/geospatial_preprocessor.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

This module handles preprocessing for geospatial datasets:
- Street Network Data
- Points of Interest (POI)
- Pedestrian Zones (ZTL)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from shapely.geometry import Point, LineString, Polygon
import warnings
warnings.filterwarnings('ignore')


@dataclass
class GeospatialConfig:
    """Configuration for geospatial preprocessing"""
    
    # Coordinate reference systems
    WGS84_EPSG: int = 4326  # WGS84 - lat/lon
    PROJECTED_EPSG: int = 32632  # UTM Zone 32N (for Bologna)
    
    # Spatial grid parameters
    DEFAULT_GRID_RESOLUTION: float = 0.005  # ~500m in degrees
    
    # Network analysis parameters
    ACCESSIBILITY_RADIUS: float = 500  # meters
    WALKABLE_SPEED: float = 1.4  # m/s (5 km/h)
    
    # POI categories of interest
    POI_CATEGORIES: List[str] = None
    
    def __post_init__(self):
        if self.POI_CATEGORIES is None:
            self.POI_CATEGORIES = [
                'healthcare', 'education', 'transport',
                'recreation', 'commercial', 'public_service'
            ]


class GeospatialPreprocessor:
    """Preprocessor for geospatial data"""
    
    def __init__(self, config: Optional[GeospatialConfig] = None):
        self.config = config or GeospatialConfig()
    
    # STREET NETWORK PROCESSING
    
    def clean_street_network(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Clean street network data
        
        Args:
            df: Raw street network dataframe
            
        Returns:
            GeoDataFrame with cleaned street network
        """
        gdf = self._ensure_geodataframe(df)
        
        # Convert to projected CRS for accurate measurements
        if gdf.crs is None:
            gdf = gdf.set_crs(f"EPSG:{self.config.WGS84_EPSG}")
        
        gdf_projected = gdf.to_crs(f"EPSG:{self.config.PROJECTED_EPSG}")
        
        # Clean geometry
        gdf_projected = gdf_projected[gdf_projected.geometry.notna()]
        gdf_projected = gdf_projected[gdf_projected.geometry.is_valid]
        
        # Remove duplicate geometries
        gdf_projected = gdf_projected.drop_duplicates(subset='geometry')
        
        # Add basic attributes
        if 'length_m' not in gdf_projected.columns:
            gdf_projected['length_m'] = gdf_projected.geometry.length
        
        # Convert back to WGS84 for consistency
        gdf = gdf_projected.to_crs(f"EPSG:{self.config.WGS84_EPSG}")
        
        print(f"  ✓ Cleaned street network: {len(gdf):,} segments")
        return gdf
    
    def compute_network_metrics(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute network topology metrics
        
        Args:
            gdf: Street network GeoDataFrame
            
        Returns:
            GeoDataFrame with network metrics
        """
        gdf = gdf.copy()
        
        # Ensure projected CRS for calculations
        original_crs = gdf.crs
        gdf_proj = gdf.to_crs(f"EPSG:{self.config.PROJECTED_EPSG}")
        
        # Street density (simplified - segments per area)
        if 'segment_density' not in gdf_proj.columns:
            # Create buffer around each segment
            buffer_area = gdf_proj.geometry.buffer(50).area  # 50m buffer
            gdf_proj['segment_density'] = 1 / (buffer_area / 1000000)  # per km²
        
        # Connectivity (count nearby segments)
        gdf_proj['connectivity'] = self._compute_connectivity(gdf_proj)
        
        # Convert back to original CRS
        gdf = gdf_proj.to_crs(original_crs)
        
        print(f"  ✓ Network metrics computed")
        return gdf
    
    def _compute_connectivity(self, gdf: gpd.GeoDataFrame, 
                            distance: float = 100) -> pd.Series:
        """Compute connectivity as number of segments within distance"""
        
        connectivity = []
        for idx, row in gdf.iterrows():
            # Buffer around segment
            buffer = row.geometry.buffer(distance)
            # Count intersecting segments
            count = gdf.intersects(buffer).sum() - 1  # exclude self
            connectivity.append(count)
        
        return pd.Series(connectivity, index=gdf.index)
    
    # POI PROCESSING
    
    def process_poi(self, df: pd.DataFrame,
                   category_column: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Process Points of Interest
        
        Args:
            df: Raw POI dataframe
            category_column: Column containing POI categories
            
        Returns:
            Processed POI GeoDataFrame
        """
        gdf = self._ensure_geodataframe(df)
        
        # Ensure WGS84 CRS
        if gdf.crs is None:
            gdf = gdf.set_crs(f"EPSG:{self.config.WGS84_EPSG}")
        
        # Clean geometry
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[gdf.geometry.is_valid]
        
        # Ensure all geometries are Points
        gdf = gdf[gdf.geometry.type == 'Point']
        
        # Categorize POIs if category column provided
        if category_column and category_column in gdf.columns:
            gdf['poi_category'] = gdf[category_column].str.lower().str.strip()
        
        # Remove duplicates
        gdf = gdf.drop_duplicates(subset='geometry')
        
        print(f"  ✓ Processed POI: {len(gdf):,} points")
        
        if 'poi_category' in gdf.columns:
            print(f"    Categories: {gdf['poi_category'].nunique()}")
        
        return gdf
    
    # SPATIAL GRID
    
    def create_spatial_grid(self, bounds: Tuple[float, float, float, float],
                          resolution: float = None) -> gpd.GeoDataFrame:
        """
        Create spatial grid over study area
        
        Args:
            bounds: (min_lon, max_lon, min_lat, max_lat)
            resolution: Grid cell size in degrees
            
        Returns:
            GeoDataFrame with grid cells
        """
        if resolution is None:
            resolution = self.config.DEFAULT_GRID_RESOLUTION
        
        min_lon, max_lon, min_lat, max_lat = bounds
        
        # Create grid
        cols = np.arange(min_lon, max_lon, resolution)
        rows = np.arange(min_lat, max_lat, resolution)
        
        polygons = []
        grid_ids = []
        
        for i, lon in enumerate(cols):
            for j, lat in enumerate(rows):
                # Create grid cell polygon
                polygon = Polygon([
                    (lon, lat),
                    (lon + resolution, lat),
                    (lon + resolution, lat + resolution),
                    (lon, lat + resolution)
                ])
                polygons.append(polygon)
                grid_ids.append(f"grid_{i}_{j}")
        
        # Create GeoDataFrame
        grid = gpd.GeoDataFrame({
            'grid_id': grid_ids,
            'geometry': polygons
        }, crs=f"EPSG:{self.config.WGS84_EPSG}")
        
        # Add centroid coordinates
        grid['centroid_lon'] = grid.geometry.centroid.x
        grid['centroid_lat'] = grid.geometry.centroid.y
        
        # Add area in km²
        grid_proj = grid.to_crs(f"EPSG:{self.config.PROJECTED_EPSG}")
        grid['area_km2'] = grid_proj.geometry.area / 1000000
        
        print(f"  ✓ Created grid: {len(grid):,} cells")
        print(f"    Resolution: ~{resolution * 111:.0f}m × {resolution * 111:.0f}m")
        
        return grid
    
    def join_poi_to_grid(self, grid_gdf: gpd.GeoDataFrame,
                        poi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Join POI counts to grid cells
        
        Args:
            grid_gdf: Grid GeoDataFrame
            poi_gdf: POI GeoDataFrame
            
        Returns:
            Grid with POI counts
        """
        grid = grid_gdf.copy()
        
        # Spatial join
        joined = gpd.sjoin(poi_gdf, grid, how='inner', predicate='within')
        
        # Count POIs per grid cell
        poi_counts = joined.groupby('grid_id').size().reset_index(name='poi_count')
        
        # Merge back to grid
        grid = grid.merge(poi_counts, on='grid_id', how='left')
        grid['poi_count'] = grid['poi_count'].fillna(0).astype(int)
        
        # If POI has categories, count by category
        if 'poi_category' in poi_gdf.columns:
            for category in poi_gdf['poi_category'].unique():
                if pd.notna(category):
                    cat_poi = poi_gdf[poi_gdf['poi_category'] == category]
                    cat_joined = gpd.sjoin(cat_poi, grid, how='inner', predicate='within')
                    cat_counts = cat_joined.groupby('grid_id').size().reset_index(
                        name=f'poi_{category}_count'
                    )
                    grid = grid.merge(cat_counts, on='grid_id', how='left')
                    grid[f'poi_{category}_count'] = grid[f'poi_{category}_count'].fillna(0).astype(int)
        
        # Compute POI density (per km²)
        grid['poi_density'] = grid['poi_count'] / grid['area_km2']
        
        print(f"  ✓ Joined {poi_counts['poi_count'].sum():.0f} POIs to grid")
        
        return grid
    
    # ACCESSIBILITY ANALYSIS
    
    def compute_accessibility_scores(self, network_gdf: gpd.GeoDataFrame,
                                    poi_gdf: gpd.GeoDataFrame,
                                    radius: float = None) -> gpd.GeoDataFrame:
        """
        Compute accessibility scores based on POI proximity
        
        Args:
            network_gdf: Street network GeoDataFrame
            poi_gdf: POI GeoDataFrame
            radius: Search radius in meters
            
        Returns:
            GeoDataFrame with accessibility scores
        """
        if radius is None:
            radius = self.config.ACCESSIBILITY_RADIUS
        
        # Ensure projected CRS
        network = network_gdf.to_crs(f"EPSG:{self.config.PROJECTED_EPSG}")
        poi = poi_gdf.to_crs(f"EPSG:{self.config.PROJECTED_EPSG}")
        
        # Create points along network segments
        network_points = []
        network_ids = []
        
        for idx, row in network.iterrows():
            # Sample points along line (every 50m)
            if row.geometry.type == 'LineString':
                length = row.geometry.length
                num_points = max(2, int(length / 50))
                
                for i in range(num_points):
                    point = row.geometry.interpolate(i / num_points, normalized=True)
                    network_points.append(point)
                    network_ids.append(idx)
        
        network_point_gdf = gpd.GeoDataFrame({
            'network_id': network_ids,
            'geometry': network_points
        }, crs=network.crs)
        
        # Count POIs within radius for each network point
        accessibility_scores = []
        
        for idx, point_row in network_point_gdf.iterrows():
            buffer = point_row.geometry.buffer(radius)
            poi_count = poi.intersects(buffer).sum()
            
            # Simple accessibility score: number of POIs
            accessibility_scores.append(poi_count)
        
        network_point_gdf['accessibility_score'] = accessibility_scores
        
        # Aggregate back to network segments
        segment_scores = network_point_gdf.groupby('network_id')['accessibility_score'].mean()
        
        network['accessibility_score'] = network.index.map(segment_scores).fillna(0)
        
        # Normalize to 0-1 scale
        max_score = network['accessibility_score'].max()
        if max_score > 0:
            network['accessibility_normalized'] = network['accessibility_score'] / max_score
        else:
            network['accessibility_normalized'] = 0
        
        # Convert back to WGS84
        network = network.to_crs(f"EPSG:{self.config.WGS84_EPSG}")
        
        print(f"  ✓ Accessibility scores computed (radius: {radius}m)")
        print(f"    Mean score: {network['accessibility_score'].mean():.2f}")
        
        return network
    
    def compute_walkability_index(self, grid_gdf: gpd.GeoDataFrame,
                                  network_gdf: gpd.GeoDataFrame,
                                  poi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute composite walkability index
        
        Components:
        - Street connectivity
        - POI density
        - Network density
        
        Args:
            grid_gdf: Spatial grid
            network_gdf: Street network
            poi_gdf: Points of Interest
            
        Returns:
            Grid with walkability scores
        """
        grid = grid_gdf.copy()
        
        # Join POI to grid
        grid = self.join_poi_to_grid(grid, poi_gdf)
        
        # Compute network density per grid cell
        network_proj = network_gdf.to_crs(f"EPSG:{self.config.PROJECTED_EPSG}")
        grid_proj = grid.to_crs(f"EPSG:{self.config.PROJECTED_EPSG}")
        
        network_in_grid = gpd.sjoin(network_proj, grid_proj, how='inner', predicate='intersects')
        network_length = network_in_grid.groupby('grid_id')['length_m'].sum()
        
        grid['network_length_m'] = grid['grid_id'].map(network_length).fillna(0)
        grid['network_density_km_km2'] = grid['network_length_m'] / 1000 / grid['area_km2']
        
        # Normalize components
        grid['poi_density_norm'] = self._normalize_column(grid['poi_density'])
        grid['network_density_norm'] = self._normalize_column(grid['network_density_km_km2'])
        
        # Compute walkability index (weighted average)
        grid['walkability_index'] = (
            0.5 * grid['poi_density_norm'] +
            0.5 * grid['network_density_norm']
        )
        
        print(f"  ✓ Walkability index computed")
        print(f"    Mean walkability: {grid['walkability_index'].mean():.3f}")
        
        return grid
    
    # UTILITIES
    
    def _ensure_geodataframe(self, df: pd.DataFrame,
                            coord_column: str = None) -> gpd.GeoDataFrame:
        """Convert DataFrame to GeoDataFrame if needed"""
        
        if isinstance(df, gpd.GeoDataFrame):
            return df
        
        # Try to find coordinate column
        if coord_column is None:
            for col in ['Geo Point', 'geopoint', 'coordinates', 'geometry']:
                if col in df.columns:
                    coord_column = col
                    break
        
        if coord_column and coord_column in df.columns:
            # Parse coordinates
            coords = df[coord_column].str.split(',', expand=True)
            
            if coords.shape[1] >= 2:
                df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
                df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
                
                # Create geometry
                geometry = [
                    Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
                    for lon, lat in zip(df['longitude'], df['latitude'])
                ]
                
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=geometry,
                    crs=f"EPSG:{self.config.WGS84_EPSG}"
                )
                
                return gdf[gdf.geometry.notna()]
        
        # If no coordinate column found, raise error
        raise ValueError(
            f"Cannot convert to GeoDataFrame. No coordinate column found. "
            f"Available columns: {list(df.columns)}"
        )
    
    def convert_to_geodataframe(self, df: pd.DataFrame,
                               coord_column: str = None) -> gpd.GeoDataFrame:
        """Public method to convert DataFrame to GeoDataFrame"""
        return self._ensure_geodataframe(df, coord_column)
    
    @staticmethod
    def _normalize_column(series: pd.Series) -> pd.Series:
        """Normalize series to 0-1 range"""
        min_val = series.min()
        max_val = series.max()
        
        if max_val - min_val == 0:
            return pd.Series(0.5, index=series.index)
        
        return (series - min_val) / (max_val - min_val)
    
    def validate_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate and fix geometries"""
        
        gdf = gdf.copy()
        
        # Remove null geometries
        gdf = gdf[gdf.geometry.notna()]
        
        # Fix invalid geometries
        invalid = ~gdf.geometry.is_valid
        if invalid.sum() > 0:
            print(f"  ⚠ Fixing {invalid.sum()} invalid geometries")
            gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].buffer(0)
        
        return gdf