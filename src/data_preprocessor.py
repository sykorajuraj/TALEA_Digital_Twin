"""
TALEA - Civic Digital Twin: Main Data Preprocessing Orchestrator
File: src/data_preprocessor.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

This module orchestrates data cleaning, feature engineering, and preparation
for mobility pattern analysis in Bologna's civic digital twin.

Delegates specialized preprocessing to:
- mobility_preprocessor.py (bicycle, pedestrian, traffic)
- weather_preprocessor.py (temperature, precipitation, air quality)
- geospatial_preprocessor.py (street networks, POI, ZTL)
- integrator.py (spatiotemporal joins, unified datasets)

6 mobility datasets:
1. Bicycle Counter Data
2. Pedestrian Flow Data
3. Street Network Data
4. Traffic Monitor Data
5. Pedestrian Zones (ZTL)
6. Points of Interest (POI)

3 weather datasets:
1. Temperature in Bologna
2. Precipitation in Bologna
3. Air quality monitoring in Bologna
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import specialized preprocessors
from src.data_preprocessing.mobility_preprocessor import (
    BicycleCounterPreprocessor,
    PedestrianFlowPreprocessor,
    TrafficPreprocessor,
    ProcessingConfig as MobilityConfig
)
from src.data_preprocessing.weather_preprocessor import (
    WeatherPreprocessor,
    WeatherConfig
)
from src.data_preprocessing.geospatial_preprocessor import (
    GeospatialPreprocessor,
    GeospatialConfig
)
from src.data_preprocessing.integrator import DataIntegrator


# EXCEPTIONS

class DataProcessingError(Exception):
    """Base exception for preprocessing errors"""
    pass


class MissingDatasetError(DataProcessingError):
    """Raised when required dataset is missing"""
    pass


class InvalidDataError(DataProcessingError):
    """Raised when data validation fails"""
    pass


# CONFIGURATION

@dataclass
class TALEAConfig:
    """Master configuration for TALEA preprocessing pipeline"""
    
    # Sub-configurations
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    geospatial: GeospatialConfig = field(default_factory=GeospatialConfig)
    
    # Geographic bounds (Bologna)
    BOLOGNA_LAT_RANGE: Tuple[float, float] = (44.4, 44.6)
    BOLOGNA_LON_RANGE: Tuple[float, float] = (11.2, 11.5)
    
    # Output settings
    EXPORT_FORMAT: str = 'csv'  # 'csv', 'parquet', 'feather'
    COMPRESSION: Optional[str] = None  # 'gzip', 'bz2', 'xz'


class ColumnNames(Enum):
    """Standardized column names across all datasets"""
    # Temporal
    DATETIME = 'datetime'
    YEAR = 'year'
    MONTH = 'month'
    DAY = 'day'
    HOUR = 'hour'
    
    # Spatial
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'
    
    # Weather
    TEMPERATURE = 'temperature'
    TEMP_MAX = 'temp_max'
    TEMP_MIN = 'temp_min'
    PRECIPITATION = 'precipitation_mm'


@dataclass
class DatasetConfig:
    """Configuration for each dataset type"""
    name: str
    date_column: str
    count_columns: List[str] = field(default_factory=list)
    location_column: Optional[str] = None
    coordinate_column: Optional[str] = None
    key_columns: List[str] = field(default_factory=list)


# DATA LOADER

class DataLoader:
    """Handles loading and initial validation of datasets"""
    
    DATASETS = {
        'bicycle': DatasetConfig(
            name='mobility_data/colonnine-conta-bici.csv',
            date_column='Data',
            count_columns=['Direzione centro', 'Direzione periferia', 'Totale'],
            location_column='Dispositivo conta-bici',
            coordinate_column='Geo Point',
            key_columns=['Data', 'Dispositivo conta-bici']
        ),
        'pedestrian': DatasetConfig(
            name='mobility_data/bologna-daily-mobility.csv',
            date_column='Data',
            count_columns=['Numero di visitatori'],
            location_column='route',
            key_columns=['Data', 'route']
        ),
        'traffic': DatasetConfig(
            name='mobility_data/traffico-viali.csv',
            date_column='data',
            count_columns=['vehicle_count'],
            location_column='VIA_SPIRA',
            key_columns=['data', 'VIA_SPIRA']
        ),
        'street': DatasetConfig(
            name='mobility_data/rifter_arcstra_li.csv',
            date_column='DATA_ISTIT',
            coordinate_column='Geo Point'
        ),
        'ztl': DatasetConfig(
            name='mobility_data/zona-pedonale-centro-storico.csv',
            date_column=None,
            coordinate_column='Geo Point'
        ),
        'poi': DatasetConfig(
            name='mobility_data/musei_gallerie_luoghi_e_teatri_storici.csv',
            date_column=None,
            coordinate_column='geopoint'
        ),
        'temperature': DatasetConfig(
            name='weather_data/temperature_bologna.csv',
            date_column='Data',
            count_columns=['Temperatura media', 'Temperatura massima', 'Temperatura minima']
        ),
        'precipitation': DatasetConfig(
            name='weather_data/precipitazioni_bologna.csv',
            date_column='Data',
            count_columns=['Precipitazioni (mm)']
        ),
        'air_quality': DatasetConfig(
            name='weather_data/centraline-qualita-aria.csv',
            date_column='reftime',
            count_columns=['value']
        )
    }

    def __init__(self, data_dir: Union[str, Path] = 'dataset'):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load(self, dataset_type: str, 
             path: Optional[Union[str, Path]] = None,
             df: Optional[pd.DataFrame] = None,
             separator: str = ';') -> pd.DataFrame:
        """Load a single dataset with validation"""
        
        if dataset_type not in self.DATASETS:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        try:
            if df is not None:
                data = df.copy()
            elif path:
                data = pd.read_csv(path, sep=separator)
            else:
                config = self.DATASETS[dataset_type]
                default_path = self.data_dir / config.name
                if not default_path.exists():
                    raise FileNotFoundError(f"Dataset file not found: {default_path}")
                data = pd.read_csv(default_path, sep=separator)
            
            if data.empty:
                raise InvalidDataError(f"Dataset {dataset_type} is empty")
            
            print(f"✓ Loaded {dataset_type}: {data.shape[0]:,} rows × {data.shape[1]} cols")
            self.datasets[dataset_type] = data
            return data
            
        except Exception as e:
            raise DataProcessingError(f"Failed to load {dataset_type}: {str(e)}")
    
    def load_all(self, paths: Dict[str, Union[str, Path]] = None,
                 dataframes: Dict[str, pd.DataFrame] = None,
                 separator: str = ';',
                 skip_missing: bool = True) -> Dict[str, pd.DataFrame]:
        """Load multiple datasets at once
        
        Args:
            paths: Optional dict mapping dataset_type to custom paths
            dataframes: Optional dict mapping dataset_type to DataFrames
            separator: CSV separator
            skip_missing: If True, skip missing files; if False, raise error
        """
        
        print("=" * 70)
        print("LOADING DATASETS")
        print("=" * 70 + "\n")
        
        paths = paths or {}
        dataframes = dataframes or {}
        
        for dataset_type in self.DATASETS.keys():
            try:
                if dataset_type in dataframes:
                    self.load(dataset_type, df=dataframes[dataset_type])
                elif dataset_type in paths:
                    self.load(dataset_type, path=paths[dataset_type], separator=separator)
                else:
                    # Try loading from default location
                    self.load(dataset_type, separator=separator)
            except Exception as e:
                if skip_missing:
                    print(f"⚠ Skipped {dataset_type}: {str(e)}")
                else:
                    raise
        
        print(f"\n✓ Loaded {len(self.datasets)} datasets\n")
        return self.datasets
    
# MAIN PREPROCESSOR

class TALEADataPreprocessor:
    """
    Main preprocessor - orchestrates all preprocessing operations
    Delegates to specialized preprocessors for each data type
    """
    
    def __init__(self, data_dir: Union[str, Path] = 'dataset',
                 config: Optional[TALEAConfig] = None):
        self.config = config or TALEAConfig()
        self.loader = DataLoader(data_dir)
        
        # Initialize specialized preprocessors
        self.bicycle_processor = BicycleCounterPreprocessor(self.config.mobility)
        self.pedestrian_processor = PedestrianFlowPreprocessor(self.config.mobility)
        self.traffic_processor = TrafficPreprocessor(self.config.mobility)
        self.weather_processor = WeatherPreprocessor(self.config.weather)
        self.geospatial_processor = GeospatialPreprocessor(self.config.geospatial)
        self.integrator = DataIntegrator()
        
        self.datasets = {}
        self.processed_datasets = {}
        self.weather_data = None
    
    def load_datasets(self, **kwargs) -> 'TALEADataPreprocessor':
        """Load datasets using DataLoader"""
        self.datasets = self.loader.load_all(**kwargs)
        return self
    
    # MOBILITY DATA PROCESSING
    
    def process_bicycle(self) -> pd.DataFrame:
        """Process bicycle counter data using BicycleCounterPreprocessor"""
        
        if 'bicycle' not in self.datasets:
            raise MissingDatasetError("Bicycle dataset not loaded")
        
        print(f"\n{'='*70}")
        print("PROCESSING BICYCLE COUNTER DATA")
        print(f"{'='*70}")
        
        df = self.datasets['bicycle'].copy()
        
        # Clean data
        df = self.bicycle_processor.clean_data(df)
        print("✓ Data cleaned")
        
        # Interpolate missing values
        df = self.bicycle_processor.interpolate_missing_values(df)
        print("✓ Missing values handled")
        
        # Add derived features
        df = self.bicycle_processor.add_derived_features(df)
        print("✓ Derived features added")
        
        self._print_summary(df, 'bicycle')
        self.processed_datasets['bicycle'] = df
        return df
    
    def process_pedestrian(self) -> pd.DataFrame:
        """Process pedestrian flow data using PedestrianFlowPreprocessor"""
        
        if 'pedestrian' not in self.datasets:
            raise MissingDatasetError("Pedestrian dataset not loaded")
        
        print(f"\n{'='*70}")
        print("PROCESSING PEDESTRIAN FLOW DATA")
        print(f"{'='*70}")
        
        df = self.datasets['pedestrian'].copy()
        
        # Clean data
        df = self.pedestrian_processor.clean_data(df)
        print("✓ Data cleaned")
        
        # Normalize flow values
        df = self.pedestrian_processor.normalize_flow_values(df, method='minmax')
        print("✓ Flow values normalized")
        
        self._print_summary(df, 'pedestrian')
        self.processed_datasets['pedestrian'] = df
        return df
    
    def process_traffic(self) -> pd.DataFrame:
        """Process traffic monitor data using TrafficPreprocessor"""
        
        if 'traffic' not in self.datasets:
            raise MissingDatasetError("Traffic dataset not loaded")
        
        print(f"\n{'='*70}")
        print("PROCESSING TRAFFIC MONITOR DATA")
        print(f"{'='*70}")
        
        df = self.datasets['traffic'].copy()
        
        # Clean data (handles wide-to-long conversion)
        df = self.traffic_processor.clean_data(df)
        print("✓ Data cleaned and reshaped")
        
        # Categorize traffic levels
        df = self.traffic_processor.categorize_vehicles(df)
        print("✓ Traffic levels categorized")
        
        # Compute congestion index
        df = self.traffic_processor.compute_congestion_index(df)
        print("✓ Congestion index computed")
        
        # Cap outliers
        df = self.traffic_processor.cap_outliers(df)
        print("✓ Outliers capped")
        
        self._print_summary(df, 'traffic')
        self.processed_datasets['traffic'] = df
        return df
    
    # WEATHER DATA PROCESSING
    
    def process_weather(self) -> pd.DataFrame:
        """Process and merge all weather datasets using WeatherPreprocessor"""
        
        print(f"\n{'='*70}")
        print("PROCESSING WEATHER DATA")
        print(f"{'='*70}")
        
        temp_df = self.datasets.get('temperature')
        precip_df = self.datasets.get('precipitation')
        aq_df = self.datasets.get('air_quality')
        
        if not any([temp_df is not None, precip_df is not None, aq_df is not None]):
            raise MissingDatasetError("No weather datasets loaded")
        
        # Merge all weather datasets
        self.weather_data = self.weather_processor.merge_weather_datasets(
            temp_df=temp_df,
            precip_df=precip_df,
            aq_df=aq_df
        )
        
        # Compute heat indices
        if ColumnNames.TEMPERATURE.value in self.weather_data.columns:
            self.weather_data = self.weather_processor.compute_heat_indices(self.weather_data)
            print("✓ Heat indices computed")
        
        self._print_summary(self.weather_data, 'weather')
        self.processed_datasets['weather'] = self.weather_data
        return self.weather_data
    
    def align_weather_temporal_resolution(self, target_frequency: str = '1H') -> pd.DataFrame:
        """Align weather data to target temporal resolution"""
        
        if self.weather_data is None:
            raise MissingDatasetError("Weather data not processed yet")
        
        print(f"\nAligning weather data to {target_frequency} resolution...")
        self.weather_data = self.weather_processor.align_temporal_resolution(
            self.weather_data, 
            target_frequency
        )
        print("✓ Weather data aligned")
        
        self.processed_datasets['weather'] = self.weather_data
        return self.weather_data
    
    # GEOSPATIAL DATA PROCESSING
    
    def process_street_network(self) -> 'GeoDataFrame':
        """Process street network data using GeospatialPreprocessor"""
        
        if 'street' not in self.datasets:
            raise MissingDatasetError("Street network dataset not loaded")
        
        print(f"\n{'='*70}")
        print("PROCESSING STREET NETWORK DATA")
        print(f"{'='*70}")
        
        df = self.datasets['street'].copy()
        
        # Clean street network
        gdf = self.geospatial_processor.clean_street_network(df)
        print("✓ Street network cleaned")
        
        # Compute network metrics
        gdf = self.geospatial_processor.compute_network_metrics(gdf)
        print("✓ Network metrics computed")
        
        self._print_summary(gdf, 'street')
        self.processed_datasets['street'] = gdf
        return gdf
    
    def process_poi(self) -> 'GeoDataFrame':
        """Process Points of Interest using GeospatialPreprocessor"""
        
        if 'poi' not in self.datasets:
            raise MissingDatasetError("POI dataset not loaded")
        
        print(f"\n{'='*70}")
        print("PROCESSING POINTS OF INTEREST")
        print(f"{'='*70}")
        
        df = self.datasets['poi'].copy()
        
        # Convert to GeoDataFrame
        gdf = self.geospatial_processor.convert_to_geodataframe(df)
        print("✓ Converted to GeoDataFrame")
        
        self._print_summary(gdf, 'poi')
        self.processed_datasets['poi'] = gdf
        return gdf
    
    def process_ztl(self) -> 'GeoDataFrame':
        """Process Pedestrian Zones (ZTL) using GeospatialPreprocessor"""
        
        if 'ztl' not in self.datasets:
            raise MissingDatasetError("ZTL dataset not loaded")
        
        print(f"\n{'='*70}")
        print("PROCESSING PEDESTRIAN ZONES (ZTL)")
        print(f"{'='*70}")
        
        df = self.datasets['ztl'].copy()
        
        # Convert to GeoDataFrame
        gdf = self.geospatial_processor.convert_to_geodataframe(df)
        print("✓ Converted to GeoDataFrame")
        
        self._print_summary(gdf, 'ztl')
        self.processed_datasets['ztl'] = gdf
        return gdf
    
    def create_spatial_grid(self, resolution: float = 0.005) -> 'GeoDataFrame':
        """Create spatial grid for Bologna using GeospatialPreprocessor"""
        
        print(f"\n{'='*70}")
        print("CREATING SPATIAL GRID")
        print(f"{'='*70}")
        
        bounds = (*self.config.BOLOGNA_LON_RANGE, *self.config.BOLOGNA_LAT_RANGE)
        grid = self.geospatial_processor.create_spatial_grid(bounds, resolution)
        
        print(f"✓ Created grid with {len(grid)} cells")
        self.processed_datasets['grid'] = grid
        return grid
    
    def compute_accessibility_scores(self) -> 'GeoDataFrame':
        """Compute accessibility scores using street network and POI"""
        
        if 'street' not in self.processed_datasets:
            raise MissingDatasetError("Street network not processed yet")
        if 'poi' not in self.processed_datasets:
            raise MissingDatasetError("POI not processed yet")
        
        print(f"\n{'='*70}")
        print("COMPUTING ACCESSIBILITY SCORES")
        print(f"{'='*70}")
        
        network = self.processed_datasets['street']
        poi = self.processed_datasets['poi']
        
        accessibility = self.geospatial_processor.compute_accessibility_scores(network, poi)
        
        print("✓ Accessibility scores computed")
        self.processed_datasets['accessibility'] = accessibility
        return accessibility
    
    # DATA INTEGRATION
    
    def integrate_weather_with_mobility(self) -> Dict[str, pd.DataFrame]:
        """Integrate weather data with all mobility datasets"""
        
        if self.weather_data is None:
            raise MissingDatasetError("Weather data not processed yet")
        
        print(f"\n{'='*70}")
        print("INTEGRATING WEATHER WITH MOBILITY DATA")
        print(f"{'='*70}")
        
        for mobility_type in ['bicycle', 'pedestrian', 'traffic']:
            if mobility_type not in self.processed_datasets:
                continue
            
            mobility_df = self.processed_datasets[mobility_type]
            config = DataLoader.DATASETS[mobility_type]
            
            # Determine date column
            if 'datetime' not in mobility_df.columns:
                date_col = config.date_column
                if date_col and date_col in mobility_df.columns:
                    # Create standardized datetime column
                    mobility_df['datetime'] = pd.to_datetime(mobility_df[date_col])
                else:
                    print(f"⚠ Warning: No valid date column for {mobility_type}, skipping integration")
                    continue
            
            # Perform integration
            integrated = self.integrator.spatiotemporal_join(
                mobility_df=mobility_df,
                weather_df=self.weather_data,
                mobility_date_col='datetime',
                weather_date_col='datetime'
            )
            
            self.processed_datasets[mobility_type] = integrated
            print(f"✓ Weather integrated with {mobility_type}")
        
        return self.processed_datasets
    
    def create_unified_dataset(self, dataset_types: List[str] = None) -> pd.DataFrame:
        """Create unified dataset from multiple processed datasets"""
        
        if dataset_types is None:
            dataset_types = ['bicycle', 'pedestrian', 'traffic']
        
        print(f"\n{'='*70}")
        print("CREATING UNIFIED DATASET")
        print(f"{'='*70}")
        
        datasets_to_merge = {
            name: self.processed_datasets[name] 
            for name in dataset_types 
            if name in self.processed_datasets
        }
        
        if not datasets_to_merge:
            raise MissingDatasetError("No processed datasets available for unification")
        
        unified = self.integrator.create_unified_dataset(datasets_to_merge)
        
        print(f"✓ Unified dataset created: {len(unified):,} records")
        self.processed_datasets['unified'] = unified
        return unified
    
    # BATCH PROCESSING
    
    def process_all_mobility(self) -> Dict[str, pd.DataFrame]:
        """Process all mobility datasets"""
        
        print(f"\n{'='*70}")
        print("PROCESSING ALL MOBILITY DATASETS")
        print(f"{'='*70}")
        
        processors = {
            'bicycle': self.process_bicycle,
            'pedestrian': self.process_pedestrian,
            'traffic': self.process_traffic
        }
        
        for dataset_type, processor in processors.items():
            if dataset_type in self.datasets:
                try:
                    processor()
                except Exception as e:
                    print(f"⚠ Failed to process {dataset_type}: {str(e)}")
        
        return self.processed_datasets
    
    def process_all_geospatial(self) -> Dict[str, 'GeoDataFrame']:
        """Process all geospatial datasets"""
        
        print(f"\n{'='*70}")
        print("PROCESSING ALL GEOSPATIAL DATASETS")
        print(f"{'='*70}")
        
        processors = {
            'street': self.process_street_network,
            'poi': self.process_poi,
            'ztl': self.process_ztl
        }
        
        for dataset_type, processor in processors.items():
            if dataset_type in self.datasets:
                try:
                    processor()
                except Exception as e:
                    print(f"⚠ Failed to process {dataset_type}: {str(e)}")
        
        return self.processed_datasets
    
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """Process all loaded datasets"""
        
        print(f"\n{'='*70}")
        print("PROCESSING ALL DATASETS")
        print(f"{'='*70}")
        
        # Process mobility
        self.process_all_mobility()
        
        # Process weather
        try:
            self.process_weather()
        except Exception as e:
            print(f"⚠ Failed to process weather: {str(e)}")
        
        # Process geospatial
        self.process_all_geospatial()
        
        # Integration
        if self.weather_data is not None:
            try:
                self.integrate_weather_with_mobility()
            except Exception as e:
                print(f"⚠ Failed to integrate weather: {str(e)}")
        
        print(f"\n{'='*70}")
        print("✓ ALL PROCESSING COMPLETE")
        print(f"{'='*70}")
        
        for name, df in self.processed_datasets.items():
            print(f"  • {name:15s}: {len(df):8,} rows × {len(df.columns):3d} cols")
        
        return self.processed_datasets
    
    # EXPORT
    
    def save_processed(self, output_dir: Union[str, Path] = '../dataset/processed_data'):
        """Save all processed datasets"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}/")
        
        for name, df in self.processed_datasets.items():
            try:
                self.integrator.export_processed_data(
                    df, 
                    output_dir / f'talea_{name}_processed.{self.config.EXPORT_FORMAT}',
                    format=self.config.EXPORT_FORMAT,
                    compression=self.config.COMPRESSION
                )
                print(f"  ✓ talea_{name}_processed.{self.config.EXPORT_FORMAT}")
            except Exception as e:
                print(f"  ✗ Failed to save {name}: {str(e)}")
        
        print("\n✓ All files saved")
    
    # UTILITIES
    
    def _print_summary(self, df: pd.DataFrame, name: str):
        """Print processing summary"""
        
        print(f"\n✓ Processed {name}: {len(df):,} records")
        
        # Date range if datetime column exists
        for date_col in ['datetime', 'Data', 'data', 'reftime']:
            if date_col in df.columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    print(f"  Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
                    break
                except:
                    pass
    
    def get_processing_report(self) -> Dict:
        """Generate comprehensive processing report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets_loaded': list(self.datasets.keys()),
            'datasets_processed': list(self.processed_datasets.keys()),
            'dataset_details': {}
        }
        
        for name, df in self.processed_datasets.items():
            report['dataset_details'][name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'column_list': list(df.columns)
            }
        
        return report