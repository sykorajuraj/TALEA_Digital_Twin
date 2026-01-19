"""
TALEA - Civic Digital Twin: Data Preprocessing Module
File: src/data_preprocessor.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

This module handles data cleaning, feature engineering, and preparation
for mobility pattern analysis in Bologna's civic digital twin.

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
from typing import Dict, List, Optional, Union, Tuple, Callable
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# EXECEPTIONS

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
class ProcessingConfig:
    """Centralized configuration for all preprocessing parameters"""
    
    # Geographic bounds (Bologna)
    BOLOGNA_LAT_RANGE: Tuple[float, float] = (44.4, 44.6)
    BOLOGNA_LON_RANGE: Tuple[float, float] = (11.2, 11.5)
    
    # Weather thresholds
    HOT_DAY_TEMP: float = 25.0
    VERY_HOT_DAY_TEMP: float = 30.0
    RAINY_DAY_MM: float = 0.1
    PRECIPITATION_ROLLING_DAYS: int = 7
    
    # Time periods
    TIME_PERIOD_BINS: List[int] = field(default_factory=lambda: [-1, 6, 10, 16, 20, 24])
    TIME_PERIOD_LABELS: List[str] = field(default_factory=lambda: 
        ['night', 'morning_rush', 'midday', 'evening_rush', 'evening'])
    
    # Peak hours
    MORNING_PEAK_START: int = 7
    MORNING_PEAK_END: int = 9
    EVENING_PEAK_START: int = 17
    EVENING_PEAK_END: int = 19
    RUSH_HOURS: List[int] = field(default_factory=lambda: [7, 8, 9, 17, 18, 19])
    
    # Missing value thresholds
    LOW_MISSING_THRESHOLD: float = 5.0  # Use interpolation
    MEDIUM_MISSING_THRESHOLD: float = 20.0  # Use group median
    
    # Outlier detection
    OUTLIER_PERCENTILE: float = 99.9
    
    # Season mapping
    SEASON_MAP: Dict[int, str] = field(default_factory=lambda: {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })


class ColumnNames(Enum):
    """Standardized column names"""
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
    key_columns: List[str] = field(default_factory=list)  # For duplicate detection

# DATA LOADER

class DataLoader:
    """Handles loading and initial validation of datasets"""
    
    DATASETS = {
        'bicycle': DatasetConfig(
            name='bicycle',
            date_column='Data',
            count_columns=['Direzione centro', 'Direzione periferia', 'Totale'],
            location_column='Dispositivo conta-bici',
            coordinate_column='Geo Point',
            key_columns=['Data', 'Dispositivo conta-bici']
        ),
        'pedestrian': DatasetConfig(
            name='pedestrian',
            date_column='Data',
            count_columns=['Numero di visitatori'],
            location_column='route',
            key_columns=['Data', 'route']
        ),
        'traffic': DatasetConfig(
            name='traffic',
            date_column='data',
            count_columns=['vehicle_count'],
            location_column='VIA_SPIRA',
            key_columns=['data', 'VIA_SPIRA']
        ),
        'street': DatasetConfig(
            name='street',
            date_column='DATA_ISTIT',
            coordinate_column='Geo Point'
        ),
        'ztl': DatasetConfig(
            name='ztl',
            coordinate_column='Geo Point'
        ),
        'poi': DatasetConfig(
            name='poi',
            coordinate_column='geopoint'
        )
    }

    def __init__(self, data_dir: Union[str, Path] = 'dataset/raw_data'):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load(self, dataset_type: str, 
             path: Optional[Union[str, Path]] = None,
             df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Load a single dataset with validation"""
        
        if dataset_type not in self.DATASETS:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        try:
            if df is not None:
                data = df.copy()
            elif path:
                data = pd.read_csv(path, sep=';')
            else:
                raise ValueError(f"Provide either 'path' or 'df' for {dataset_type}")
            
            if data.empty:
                raise InvalidDataError(f"Dataset {dataset_type} is empty")
            
            print(f"✓ Loaded {dataset_type}: {data.shape[0]:,} rows × {data.shape[1]} cols")
            self.datasets[dataset_type] = data
            return data
            
        except Exception as e:
            raise DataProcessingError(f"Failed to load {dataset_type}: {str(e)}")
    
    def load_all(self, paths: Dict[str, Union[str, Path]] = None,
                 dataframes: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """Load multiple datasets at once"""
        
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
                    self.load(dataset_type, path=paths[dataset_type])
            except Exception as e:
                print(f"⚠ Skipped {dataset_type}: {str(e)}")
        
        print(f"\n✓ Loaded {len(self.datasets)} datasets\n")
        return self.datasets
    
# SPECIALIZED PREPROCESSORS

class TemporalProcessor:
    """Handles all temporal feature engineering"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def add_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add comprehensive temporal features"""
        
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise InvalidDataError(f"Failed to parse dates in {date_col}: {str(e)}")
        
        # Basic temporal features
        df[ColumnNames.YEAR.value] = df[date_col].dt.year
        df[ColumnNames.MONTH.value] = df[date_col].dt.month
        df[ColumnNames.DAY.value] = df[date_col].dt.day
        df[ColumnNames.HOUR.value] = df[date_col].dt.hour
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['day_name'] = df[date_col].dt.day_name()
        
        # Derived features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = df[ColumnNames.MONTH.value].map(self.config.SEASON_MAP)
        
        # Time periods
        df['time_period'] = pd.cut(
            df[ColumnNames.HOUR.value],
            bins=self.config.TIME_PERIOD_BINS,
            labels=self.config.TIME_PERIOD_LABELS
        )
        
        # Peak hours
        df['is_morning_peak'] = df[ColumnNames.HOUR.value].between(
            self.config.MORNING_PEAK_START, 
            self.config.MORNING_PEAK_END
        ).astype(int)
        
        df['is_evening_peak'] = df[ColumnNames.HOUR.value].between(
            self.config.EVENING_PEAK_START, 
            self.config.EVENING_PEAK_END
        ).astype(int)
        
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        return df


class CoordinateProcessor:
    """Handles coordinate extraction and validation"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def extract_coordinates(self, df: pd.DataFrame, coord_col: str,
                          lat_name: str = None,
                          lon_name: str = None) -> pd.DataFrame:
        """Extract lat/lon from Geo Point column"""
        
        lat_name = lat_name or ColumnNames.LATITUDE.value
        lon_name = lon_name or ColumnNames.LONGITUDE.value
        
        if coord_col not in df.columns:
            return df
        
        try:
            coords = df[coord_col].str.split(',', expand=True)
            df[lat_name] = pd.to_numeric(coords[0], errors='coerce')
            df[lon_name] = pd.to_numeric(coords[1], errors='coerce')
        except Exception as e:
            raise InvalidDataError(f"Failed to extract coordinates: {str(e)}")
        
        return df
    
    def validate_bologna_coordinates(self, df: pd.DataFrame,
                                     lat_col: str = None,
                                     lon_col: str = None) -> pd.DataFrame:
        """Validate coordinates are within Bologna bounds"""
        
        lat_col = lat_col or ColumnNames.LATITUDE.value
        lon_col = lon_col or ColumnNames.LONGITUDE.value
        
        lat_min, lat_max = self.config.BOLOGNA_LAT_RANGE
        lon_min, lon_max = self.config.BOLOGNA_LON_RANGE
        
        if lat_col in df.columns and lon_col in df.columns:
            invalid = ((df[lat_col] < lat_min) | (df[lat_col] > lat_max) |
                      (df[lon_col] < lon_min) | (df[lon_col] > lon_max))
            
            if invalid.sum() > 0:
                print(f"  ⚠ {invalid.sum()} invalid coordinates (outside Bologna)")
                df.loc[invalid, [lat_col, lon_col]] = np.nan
        
        return df

class MissingValueHandler:
    """Handles missing value imputation strategies"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def analyze(self, df: pd.DataFrame, name: str = 'dataset') -> pd.DataFrame:
        """Analyze missing values"""
        
        print(f"\nMissing Values in {name.upper()}:")
        
        missing = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
        })
        
        missing = missing[missing['missing_count'] > 0].sort_values(
            'missing_pct', ascending=False
        )
        
        if len(missing) == 0:
            print("  ✓ No missing values")
        else:
            print(missing.to_string(index=False))
        
        return missing
    
    def impute_smart(self, df: pd.DataFrame, columns: List[str],
                    group_col: Optional[str] = None) -> pd.DataFrame:
        """Smart imputation based on missing data percentage"""
        
        for col in columns:
            if col not in df.columns:
                continue
                
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct == 0:
                continue
            
            if missing_pct < self.config.LOW_MISSING_THRESHOLD:
                # Low missing: interpolation
                df = self._interpolate(df, col, group_col)
            
            elif missing_pct < self.config.MEDIUM_MISSING_THRESHOLD:
                # Medium missing: group median
                df = self._group_median(df, col, group_col)
            
            else:
                # High missing: forward/backward fill + median
                df = self._fill_forward_backward(df, col)
        
        return df
    
    def _interpolate(self, df: pd.DataFrame, col: str, 
                    group_col: Optional[str]) -> pd.DataFrame:
        """Interpolate missing values"""
        if group_col and group_col in df.columns:
            df[col] = df.groupby(group_col)[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        else:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
        return df
    
    def _group_median(self, df: pd.DataFrame, col: str, 
                     group_col: Optional[str]) -> pd.DataFrame:
        """Fill with group median"""
        if group_col and group_col in df.columns:
            df[col] = df.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            df[col] = df[col].fillna(df[col].median())
        return df
    
    def _fill_forward_backward(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Forward/backward fill then median"""
        df[col] = df[col].ffill().bfill().fillna(df[col].median())
        return df

class WeatherProcessor:
    """Handles weather data processing and integration"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def process_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process temperature data"""
        
        df['Data'] = pd.to_datetime(df['Data'])
        
        # Convert to numeric
        temp_cols = ['Temperatura media', 'Temperatura massima', 'Temperatura minima']
        for col in temp_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename
        df.rename(columns={
            'Data': ColumnNames.DATETIME.value,
            'Temperatura media': ColumnNames.TEMPERATURE.value,
            'Temperatura massima': ColumnNames.TEMP_MAX.value,
            'Temperatura minima': ColumnNames.TEMP_MIN.value
        }, inplace=True)
        
        # Derived features
        if ColumnNames.TEMP_MAX.value in df.columns and ColumnNames.TEMP_MIN.value in df.columns:
            df['temp_range'] = df[ColumnNames.TEMP_MAX.value] - df[ColumnNames.TEMP_MIN.value]
        
        if ColumnNames.TEMPERATURE.value in df.columns:
            df['is_hot_day'] = (df[ColumnNames.TEMPERATURE.value] >= self.config.HOT_DAY_TEMP).astype(int)
            df['is_very_hot_day'] = (df[ColumnNames.TEMPERATURE.value] >= self.config.VERY_HOT_DAY_TEMP).astype(int)
        
        return df
    
    def process_precipitation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process precipitation data"""
        
        df['Data'] = pd.to_datetime(df['Data'])
        df['Precipitazioni (mm)'] = pd.to_numeric(df['Precipitazioni (mm)'], errors='coerce')
        
        df.rename(columns={
            'Data': ColumnNames.DATETIME.value,
            'Precipitazioni (mm)': ColumnNames.PRECIPITATION.value
        }, inplace=True)
        
        df['is_rainy_day'] = (df[ColumnNames.PRECIPITATION.value] > self.config.RAINY_DAY_MM).astype(int)
        
        # Rolling precipitation
        df = df.sort_values(ColumnNames.DATETIME.value)
        df['precipitation_7d'] = df[ColumnNames.PRECIPITATION.value].rolling(
            window=self.config.PRECIPITATION_ROLLING_DAYS, 
            min_periods=1
        ).sum()
        
        return df
    
    def merge_weather(self, mobility_df: pd.DataFrame,
                     weather_df: pd.DataFrame,
                     mobility_date_col: str = 'Data') -> pd.DataFrame:
        """Merge weather data with mobility data"""
        
        # Determine resolution
        time_diff = weather_df[ColumnNames.DATETIME.value].diff().median()
        is_hourly = time_diff <= pd.Timedelta(hours=1)
        
        # Create merge keys
        if is_hourly:
            mobility_df['merge_key'] = mobility_df[mobility_date_col].dt.floor('H')
            weather_df['merge_key'] = weather_df[ColumnNames.DATETIME.value].dt.floor('H')
        else:
            mobility_df['merge_key'] = mobility_df[mobility_date_col].dt.date
            weather_df['merge_key'] = weather_df[ColumnNames.DATETIME.value].dt.date
        
        # Select weather columns
        weather_cols = [col for col in weather_df.columns 
                       if col not in [ColumnNames.DATETIME.value, 'merge_key']]
        
        # Merge
        merged = mobility_df.merge(
            weather_df[['merge_key'] + weather_cols],
            on='merge_key',
            how='left'
        ).drop('merge_key', axis=1)
        
        return merged

# DATASET-SPECIFIC PROCESSORS

class BicycleProcessor:
    """Process bicycle counter data"""
    
    @staticmethod
    def process(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Add bicycle-specific features"""
        
        # Clean numeric columns
        for col in config.count_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Flow ratio and dominant direction
        if all(col in df.columns for col in ['Direzione centro', 'Direzione periferia']):
            df['flow_ratio'] = np.where(
                df['Direzione periferia'] > 0,
                df['Direzione centro'] / df['Direzione periferia'],
                np.nan
            )
            df['dominant_direction'] = np.where(
                df['Direzione centro'] > df['Direzione periferia'],
                'centro', 'periferia'
            )
        
        return df


class PedestrianProcessor:
    """Process pedestrian flow data"""
    
    @staticmethod
    def process(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Add pedestrian-specific features"""
        
        # Clean counts
        df['Numero di visitatori'] = pd.to_numeric(
            df['Numero di visitatori'], errors='coerce'
        ).fillna(0)
        
        # Create route identifier
        if 'Area provenienza' in df.columns and 'Area Arrivo' in df.columns:
            df['route'] = df['Area provenienza'] + ' → ' + df['Area Arrivo']
        
        return df


class TrafficProcessor:
    """Process traffic monitor data"""
    
    @staticmethod
    def process(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Process traffic data - handle wide format conversion"""
        
        # Identify time slot columns
        time_cols = [col for col in df.columns 
                    if ' - ' in str(col) and any(c.isdigit() for c in str(col))]
        
        print(f"  Found {len(time_cols)} time slot columns")
        
        if len(time_cols) > 0:
            df = TrafficProcessor._reshape_wide_to_long(df, config, time_cols)
            df = TrafficProcessor._add_traffic_features(df)
        
        return df
    
    @staticmethod
    def _reshape_wide_to_long(df: pd.DataFrame, config: DatasetConfig, 
                             time_cols: List[str]) -> pd.DataFrame:
        """Convert wide format to long format"""
        
        # Preserve metadata columns
        id_vars = [col for col in df.columns 
                  if col not in time_cols + ['tot']]
        
        # Melt to long format
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=time_cols,
            var_name='time_slot',
            value_name='vehicle_count'
        )
        
        df_long['vehicle_count'] = pd.to_numeric(
            df_long['vehicle_count'], 
            errors='coerce'
        ).fillna(0)
        
        # Extract time components
        df_long[ColumnNames.HOUR.value] = df_long['time_slot'].str.extract(r'^(\d+)\.')[0].astype(int)
        df_long['minute'] = df_long['time_slot'].str.extract(r'^(\d+)\.(\d+)')[1].astype(int)
        
        # Create datetime
        df_long[ColumnNames.DATETIME.value] = (
            df_long[config.date_column] + 
            pd.to_timedelta(df_long[ColumnNames.HOUR.value], unit='h') +
            pd.to_timedelta(df_long['minute'], unit='m')
        )
        
        return df_long
    
    @staticmethod
    def _add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add traffic-specific features"""
        
        # Clean location columns
        if 'VIA_SPIRA' in df.columns:
            df['VIA_SPIRA'] = df['VIA_SPIRA'].str.strip()
        if 'DIREZIONE' in df.columns:
            df['DIREZIONE'] = df['DIREZIONE'].str.strip()
        
        return df
    
    @staticmethod
    def cap_outliers(df: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
        """Remove extreme outliers"""
        
        if 'vehicle_count' in df.columns:
            cap_value = df['vehicle_count'].quantile(config.OUTLIER_PERCENTILE / 100)
            outliers = df['vehicle_count'] > cap_value
            
            if outliers.sum() > 0:
                print(f"  ⚠ Capped {outliers.sum()} outlier values at {cap_value:.0f}")
                df.loc[outliers, 'vehicle_count'] = cap_value
        
        return df

# MAIN PREPROCESSOR

class TALEADataPreprocessor:
    """
    Main preprocessor - orchestrates all preprocessing operations
    """
    
    def __init__(self, data_dir: Union[str, Path] = 'dataset/raw_data',
                 config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.loader = DataLoader(data_dir)
        
        # Initialize processors
        self.temporal = TemporalProcessor(self.config)
        self.coords = CoordinateProcessor(self.config)
        self.missing = MissingValueHandler(self.config)
        self.weather_proc = WeatherProcessor(self.config)
        
        self.datasets = {}
        self.weather_data = None
    
    def load_datasets(self, **kwargs) -> 'TALEADataPreprocessor':
        """Load datasets"""
        self.datasets = self.loader.load_all(**kwargs)
        return self
    
    def _process_dataset(self, dataset_type: str,
                        custom_processor: Optional[Callable] = None) -> pd.DataFrame:
        """Generic dataset processing template"""
        
        if dataset_type not in self.datasets:
            raise MissingDatasetError(f"Dataset '{dataset_type}' not loaded")
        
        print(f"\n{'='*70}")
        print(f"PROCESSING {dataset_type.upper()} DATA")
        print(f"{'='*70}")
        
        df = self.datasets[dataset_type].copy()
        config = DataLoader.DATASETS[dataset_type]
        
        # Standard temporal processing
        if config.date_column and config.date_column in df.columns:
            df = self.temporal.add_features(df, config.date_column)
        
        # Standard coordinate processing
        if config.coordinate_column and config.coordinate_column in df.columns:
            df = self.coords.extract_coordinates(df, config.coordinate_column)
            df = self.coords.validate_bologna_coordinates(df)
        
        # Custom processing
        if custom_processor:
            df = custom_processor(df, config)
        
        # Remove duplicates
        if config.key_columns:
            existing_cols = [col for col in config.key_columns if col in df.columns]
            if existing_cols:
                df = df.drop_duplicates(subset=existing_cols)
        
        # Handle missing values
        self.missing.analyze(df, dataset_type)
        if config.count_columns:
            existing_counts = [col for col in config.count_columns if col in df.columns]
            if existing_counts:
                df = self.missing.impute_smart(
                    df, 
                    existing_counts,
                    group_col=config.location_column
                )
        
        # Summary
        self._print_summary(df, config)
        
        self.datasets[dataset_type] = df
        return df
    
    def _print_summary(self, df: pd.DataFrame, config: DatasetConfig):
        """Print processing summary"""
        
        print(f"\n✓ Processed: {df.shape[0]:,} records")
        
        date_col = ColumnNames.DATETIME.value if ColumnNames.DATETIME.value in df.columns else config.date_column
        if date_col in df.columns:
            print(f"  Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
        
        if config.location_column and config.location_column in df.columns:
            print(f"  Locations: {df[config.location_column].nunique()}")
    
    def process_bicycle(self) -> pd.DataFrame:
        """Process bicycle counter data"""
        return self._process_dataset('bicycle', BicycleProcessor.process)
    
    def process_pedestrian(self) -> pd.DataFrame:
        """Process pedestrian flow data"""
        return self._process_dataset('pedestrian', PedestrianProcessor.process)
    
    def process_traffic(self) -> pd.DataFrame:
        """Process traffic monitor data"""
        df = self._process_dataset('traffic', TrafficProcessor.process)
        
        # Add temporal features for long format
        if ColumnNames.DATETIME.value in df.columns:
            df = self.temporal.add_features(df, ColumnNames.DATETIME.value)
        
        # Cap outliers
        df = TrafficProcessor.cap_outliers(df, self.config)
        
        self.datasets['traffic'] = df
        return df
    
    def load_weather(self, temp_path: str = None, precip_path: str = None,
                    temp_df: pd.DataFrame = None, precip_df: pd.DataFrame = None) -> pd.DataFrame:
        """Load and process weather data"""
        
        print(f"\n{'='*70}")
        print("PROCESSING WEATHER DATA")
        print(f"{'='*70}")
        
        weather_components = []
        
        # Temperature
        if temp_df is not None or temp_path:
            temp = temp_df if temp_df is not None else pd.read_csv(temp_path, sep=';')
            temp = self.weather_proc.process_temperature(temp)
            weather_components.append(temp)
            print("✓ Temperature data processed")
        
        # Precipitation
        if precip_df is not None or precip_path:
            precip = precip_df if precip_df is not None else pd.read_csv(precip_path, sep=';')
            precip = self.weather_proc.process_precipitation(precip)
            weather_components.append(precip)
            print("✓ Precipitation data processed")
        
        # Merge components
        if len(weather_components) > 1:
            self.weather_data = weather_components[0].merge(
                weather_components[1],
                on=ColumnNames.DATETIME.value,
                how='outer'
            )
        elif len(weather_components) == 1:
            self.weather_data = weather_components[0]
        
        if self.weather_data is not None:
            print(f"\n✓ Weather data ready: {len(self.weather_data):,} records")
        
        return self.weather_data
    
    def integrate_weather(self) -> 'TALEADataPreprocessor':
        """Integrate weather with mobility datasets"""
        
        if self.weather_data is None:
            print("⚠ No weather data loaded")
            return self
        
        print(f"\n{'='*70}")
        print("INTEGRATING WEATHER DATA")
        print(f"{'='*70}")
        
        for dataset_name in ['bicycle', 'pedestrian', 'traffic']:
            if dataset_name not in self.datasets:
                continue
                
            config = DataLoader.DATASETS[dataset_name]
            date_col = (ColumnNames.DATETIME.value if ColumnNames.DATETIME.value in self.datasets[dataset_name].columns 
                       else config.date_column)
            
            try:
                self.datasets[dataset_name] = self.weather_proc.merge_weather(
                    self.datasets[dataset_name],
                    self.weather_data,
                    mobility_date_col=date_col
                )
                print(f"✓ Weather integrated with {dataset_name}")
            except Exception as e:
                print(f"⚠ Failed to integrate weather with {dataset_name}: {str(e)}")
        
        return self
    
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """Process all loaded datasets"""
        
        print(f"\n{'='*70}")
        print("PROCESSING ALL DATASETS")
        print(f"{'='*70}")
        
        # Process each dataset type
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
        
        print(f"\n{'='*70}")
        print("✓ ALL PROCESSING COMPLETE")
        print(f"{'='*70}")
        
        for name, df in self.datasets.items():
            print(f"  • {name:12s}: {df.shape[0]:8,} rows × {df.shape[1]:3d} cols")
        
        return self.datasets
    
    def save_processed(self, output_dir: Union[str, Path] = 'dataset/processed_data'):
        """Save all processed datasets"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}/")
        
        for name, df in self.datasets.items():
            try:
                filepath = output_dir / f'talea_{name}_processed.csv'
                df.to_csv(filepath, index=False)
                print(f"  ✓ {filepath.name}")
            except Exception as e:
                print(f"  ✗ Failed to save {name}: {str(e)}")
        
        if self.weather_data is not None:
            try:
                weather_path = output_dir / 'talea_weather_processed.csv'
                self.weather_data.to_csv(weather_path, index=False)
                print(f"  ✓ {weather_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to save weather data: {str(e)}")
        
        print("\n✓ All files saved")