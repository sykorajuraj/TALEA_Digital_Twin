"""
TALEA - Civic Digital Twin: Data Integration Module
File: src/data_preprocessing/integrator.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

This module handles integration of processed datasets:
- Spatiotemporal joins
- Dataset merging
- Unified dataset creation
- Export utilities
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataIntegrator:
    """Handles integration of processed datasets"""
    
    def __init__(self, csv_separator: str = ';'):
        """
        Initialize DataIntegrator
        
        Args:
            csv_separator: Separator to use for CSV exports (default: ';')
        """
        self.csv_separator = csv_separator
    
    # TEMPORAL INTEGRATION
    
    def spatiotemporal_join(self,
                           mobility_df: pd.DataFrame,
                           weather_df: pd.DataFrame,
                           mobility_date_col: str = 'Data',
                           weather_date_col: str = 'datetime') -> pd.DataFrame:
        """
        Join weather data to mobility data based on temporal proximity
        
        Args:
            mobility_df: Mobility dataset
            weather_df: Weather dataset
            mobility_date_col: Date column in mobility data
            weather_date_col: Date column in weather data
            
        Returns:
            Merged dataframe
        """
        mobility = mobility_df.copy()
        weather = weather_df.copy()
        
        # Ensure datetime format
        mobility[mobility_date_col] = pd.to_datetime(mobility[mobility_date_col])
        weather[weather_date_col] = pd.to_datetime(weather[weather_date_col])
        
        # Determine temporal resolution
        weather_resolution = self._detect_temporal_resolution(weather[weather_date_col])
        mobility_resolution = self._detect_temporal_resolution(mobility[mobility_date_col])
        
        print(f"  Mobility resolution: {mobility_resolution}")
        print(f"  Weather resolution: {weather_resolution}")
        
        # Create merge keys based on resolution
        if weather_resolution == 'hourly':
            mobility['merge_key'] = mobility[mobility_date_col].dt.floor('h')
            weather['merge_key'] = weather[weather_date_col].dt.floor('h')
        elif weather_resolution == 'daily':
            mobility['merge_key'] = mobility[mobility_date_col].dt.date
            weather['merge_key'] = weather[weather_date_col].dt.date
        else:
            # Default to daily
            mobility['merge_key'] = mobility[mobility_date_col].dt.date
            weather['merge_key'] = weather[weather_date_col].dt.date
        
        # Select weather columns to merge (exclude datetime columns)
        weather_cols = [col for col in weather.columns 
                       if col not in [weather_date_col, 'merge_key']]
        
        # Perform merge
        merged = mobility.merge(
            weather[['merge_key'] + weather_cols],
            on='merge_key',
            how='left'
        )
        
        # Drop merge key
        merged = merged.drop('merge_key', axis=1)
        
        # Report merge statistics
        weather_coverage = (merged[weather_cols[0]].notna().sum() / len(merged)) * 100
        print(f"  ✓ Weather data coverage: {weather_coverage:.1f}%")
        
        return merged
    
    def _detect_temporal_resolution(self, datetime_series: pd.Series) -> str:
        """Detect temporal resolution of datetime series"""
        
        if len(datetime_series) < 2:
            return 'unknown'
        
        # Calculate median time difference
        time_diff = datetime_series.sort_values().diff().median()
        
        if pd.isna(time_diff):
            return 'unknown'
        
        # Determine resolution
        if time_diff <= pd.Timedelta(hours=1):
            return 'hourly'
        elif time_diff <= pd.Timedelta(days=1):
            return 'daily'
        elif time_diff <= pd.Timedelta(weeks=1):
            return 'weekly'
        else:
            return 'monthly'
    
    # SPATIAL INTEGRATION
    
    def spatial_join_to_grid(self,
                            mobility_df: pd.DataFrame,
                            grid_gdf: gpd.GeoDataFrame,
                            lat_col: str = 'latitude',
                            lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Join mobility data to spatial grid
        
        Args:
            mobility_df: Mobility dataset with coordinates
            grid_gdf: Spatial grid GeoDataFrame
            lat_col: Latitude column name
            lon_col: Longitude column name
            
        Returns:
            Mobility data with grid assignments
        """
        from shapely.geometry import Point
        
        mobility = mobility_df.copy()
        
        # Create GeoDataFrame from mobility data
        geometry = [
            Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
            for lon, lat in zip(mobility[lon_col], mobility[lat_col])
        ]
        
        mobility_gdf = gpd.GeoDataFrame(
            mobility,
            geometry=geometry,
            crs=grid_gdf.crs
        )
        
        # Remove rows with null geometry
        mobility_gdf = mobility_gdf[mobility_gdf.geometry.notna()]
        
        # Spatial join
        joined = gpd.sjoin(
            mobility_gdf,
            grid_gdf[['grid_id', 'geometry']],
            how='left',
            predicate='within'
        )
        
        # Drop geometry column to return regular DataFrame
        joined = pd.DataFrame(joined.drop(columns='geometry'))
        
        coverage = (joined['grid_id'].notna().sum() / len(joined)) * 100
        print(f"  ✓ Grid assignment coverage: {coverage:.1f}%")
        
        return joined
    
    # UNIFIED DATASET CREATION
    
    def create_unified_dataset(self,
                              datasets: Dict[str, pd.DataFrame],
                              join_on: List[str] = None,
                              join_type: str = 'outer') -> pd.DataFrame:
        """
        Create unified dataset from multiple datasets
        
        Args:
            datasets: Dictionary of {name: dataframe}
            join_on: Columns to join on (default: datetime-based)
            join_type: Type of join ('inner', 'outer', 'left', 'right')
            
        Returns:
            Unified dataframe
        """
        if not datasets:
            raise ValueError("No datasets provided")
        
        # Auto-detect join columns if not specified
        if join_on is None:
            join_on = self._detect_common_columns(datasets)
            print(f"  Auto-detected join columns: {join_on}")
        
        # Start with first dataset
        dataset_names = list(datasets.keys())
        unified = datasets[dataset_names[0]].copy()
        
        # Add source column
        unified['source'] = dataset_names[0]
        
        # Sequentially merge remaining datasets
        for name in dataset_names[1:]:
            df = datasets[name].copy()
            df['source'] = name
            
            # Add prefix to overlapping columns (except join columns)
            overlap_cols = set(unified.columns) & set(df.columns) - set(join_on) - {'source'}
            rename_dict = {col: f"{name}_{col}" for col in overlap_cols}
            df = df.rename(columns=rename_dict)
            
            # Merge
            unified = unified.merge(
                df,
                on=join_on,
                how=join_type,
                suffixes=('', f'_{name}')
            )
            
            print(f"  ✓ Merged {name}: {len(unified):,} records")
        
        print(f"\n  Final unified dataset: {len(unified):,} records × {len(unified.columns)} columns")
        
        return unified
    
    def _detect_common_columns(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Detect common columns across datasets for joining"""
        
        if not datasets:
            return []
        
        # Get column sets
        column_sets = [set(df.columns) for df in datasets.values()]
        
        # Find intersection
        common_cols = set.intersection(*column_sets)
        
        # Prioritize datetime-related columns
        datetime_candidates = ['datetime', 'Data', 'data', 'date', 'timestamp']
        
        for col in datetime_candidates:
            if col in common_cols:
                return [col]
        
        # Prioritize spatial columns
        spatial_candidates = ['grid_id', 'location', 'zone']
        
        for col in spatial_candidates:
            if col in common_cols:
                return [col]
        
        # Return all common columns if no priority match
        return list(common_cols)
    
    # AGGREGATION
    
    def aggregate_by_time(self,
                         df: pd.DataFrame,
                         datetime_col: str,
                         frequency: str,
                         agg_dict: Dict[str, Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Aggregate dataset by time frequency
        
        Args:
            df: Input dataframe
            datetime_col: Datetime column name
            frequency: Pandas frequency string ('1H', '1D', '1W', etc.)
            agg_dict: Dictionary of {column: aggregation_function}
            
        Returns:
            Aggregated dataframe
        """
        data = df.copy()
        
        # Ensure datetime
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        
        # Set datetime index
        data = data.set_index(datetime_col)
        
        # Default aggregation if not specified
        if agg_dict is None:
            agg_dict = self._create_default_agg_dict(data)
        
        # Aggregate
        aggregated = data.resample(frequency).agg(agg_dict)
        
        # Flatten multi-level columns if present
        if isinstance(aggregated.columns, pd.MultiIndex):
            aggregated.columns = ['_'.join(col).strip('_') 
                                 for col in aggregated.columns.values]
        
        aggregated = aggregated.reset_index()
        
        print(f"  ✓ Aggregated to {frequency}: {len(aggregated):,} records")
        
        return aggregated
    
    def _create_default_agg_dict(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create default aggregation dictionary based on column types"""
        
        agg_dict = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Numeric columns: use mean
                agg_dict[col] = 'mean'
            elif df[col].dtype == 'object':
                # Categorical columns: use mode (most common)
                agg_dict[col] = lambda x: x.mode()[0] if len(x.mode()) > 0 else None
            else:
                # Other types: use first value
                agg_dict[col] = 'first'
        
        return agg_dict
    
    def aggregate_by_zone(self,
                         df: pd.DataFrame,
                         zone_col: str,
                         agg_dict: Dict[str, Union[str, List[str]]]) -> pd.DataFrame:
        """
        Aggregate dataset by spatial zones
        
        Args:
            df: Input dataframe
            zone_col: Zone/grid identifier column
            agg_dict: Dictionary of {column: aggregation_function}
            
        Returns:
            Aggregated dataframe
        """
        data = df.copy()
        
        # Group by zone
        aggregated = data.groupby(zone_col).agg(agg_dict).reset_index()
        
        # Flatten multi-level columns if present
        if isinstance(aggregated.columns, pd.MultiIndex):
            aggregated.columns = ['_'.join(col).strip('_') 
                                 for col in aggregated.columns.values]
        
        print(f"  ✓ Aggregated by zone: {len(aggregated):,} zones")
        
        return aggregated
    
    # EXPORT
    
    def export_processed_data(self,
                             df: Union[pd.DataFrame, gpd.GeoDataFrame],
                             output_path: Union[str, Path],
                             format: str = 'csv',
                             compression: Optional[str] = None) -> None:
        """
        Export processed data to file
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            format: Export format ('csv', 'parquet', 'feather', 'geojson', 'shapefile')
            compression: Compression type ('gzip', 'bz2', 'xz')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle GeoDataFrame
        if isinstance(df, gpd.GeoDataFrame):
            if format == 'geojson':
                df.to_file(output_path, driver='GeoJSON')
            elif format == 'shapefile':
                df.to_file(output_path, driver='ESRI Shapefile')
            elif format == 'parquet':
                df.to_parquet(output_path, compression=compression)
            else:
                # Convert to regular DataFrame for CSV
                df_export = pd.DataFrame(df.drop(columns='geometry'))
                df_export.to_csv(output_path, index=False, sep=self.csv_separator, 
                                compression=compression)
        
        # Handle regular DataFrame
        else:
            if format == 'csv':
                df.to_csv(output_path, index=False, sep=self.csv_separator, 
                         compression=compression)
            elif format == 'parquet':
                df.to_parquet(output_path, compression=compression, index=False)
            elif format == 'feather':
                df.to_feather(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def export_multiple(self,
                       datasets: Dict[str, pd.DataFrame],
                       output_dir: Union[str, Path],
                       format: str = 'csv',
                       compression: Optional[str] = None) -> None:
        """
        Export multiple datasets
        
        Args:
            datasets: Dictionary of {name: dataframe}
            output_dir: Output directory
            format: Export format
            compression: Compression type
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting {len(datasets)} datasets to {output_dir}/")
        
        for name, df in datasets.items():
            output_path = output_dir / f"{name}.{format}"
            
            try:
                self.export_processed_data(
                    df,
                    output_path,
                    format=format,
                    compression=compression
                )
                print(f"  ✓ {output_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to export {name}: {str(e)}")
    
    # VALIDATION
    
    def validate_temporal_alignment(self,
                                   df1: pd.DataFrame,
                                   df2: pd.DataFrame,
                                   datetime_col1: str,
                                   datetime_col2: str) -> Dict:
        """
        Validate temporal alignment between two datasets
        
        Returns:
            Dictionary with alignment statistics
        """
        df1_dates = pd.to_datetime(df1[datetime_col1])
        df2_dates = pd.to_datetime(df2[datetime_col2])
        
        # Find overlap
        df1_range = (df1_dates.min(), df1_dates.max())
        df2_range = (df2_dates.min(), df2_dates.max())
        
        overlap_start = max(df1_range[0], df2_range[0])
        overlap_end = min(df1_range[1], df2_range[1])
        
        has_overlap = overlap_start <= overlap_end
        
        return {
            'df1_range': df1_range,
            'df2_range': df2_range,
            'overlap_start': overlap_start if has_overlap else None,
            'overlap_end': overlap_end if has_overlap else None,
            'has_overlap': has_overlap,
            'overlap_days': (overlap_end - overlap_start).days if has_overlap else 0
        }
    
    def generate_integration_report(self,
                                   datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate report on dataset integration
        
        Returns:
            DataFrame with integration statistics
        """
        report_data = []
        
        for name, df in datasets.items():
            # Basic statistics
            stats = {
                'dataset': name,
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
                'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            }
            
            # Temporal info
            for col in ['datetime', 'Data', 'data', 'date']:
                if col in df.columns:
                    try:
                        dates = pd.to_datetime(df[col])
                        stats['date_range_start'] = dates.min()
                        stats['date_range_end'] = dates.max()
                        stats['date_range_days'] = (dates.max() - dates.min()).days
                        break
                    except:
                        pass
            
            report_data.append(stats)
        
        report = pd.DataFrame(report_data)
        
        return report