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
import warnings
from typing import Dict, List, Optional, Union
warnings.filterwarnings('ignore')


@dataclass
class DatasetConfig:
    """Configuration for each dataset type"""
    name: str
    date_column: str
    count_columns: List[str] = field(default_factory=list)
    location_column: Optional[str] = None
    coordinate_column: Optional[str] = None
    

class DataLoader:
    """Handles loading and initial validation of datasets"""
    
    DATASETS = {
        'bicycle': DatasetConfig(
            name='bicycle',
            date_column='Data',
            count_columns=['Direzione centro', 'Direzione periferia', 'Totale'],
            location_column='Dispositivo conta-bici',
            coordinate_column='Geo Point'
        ),
        'pedestrian': DatasetConfig(
            name='pedestrian',
            date_column='Data',
            count_columns=['Numero di visitatori'],
            location_column='route'
        ),
        'traffic': DatasetConfig(
            name='traffic',
            date_column='data',
            count_columns=['vehicle_count'],
            location_column='VIA_SPIRA'
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
        """Load a single dataset"""
        
        if df is not None:
            data = df.copy()
        elif path:
            data = pd.read_csv(path, sep=';')
        else:
            raise ValueError(f"Provide either 'path' or 'df' for {dataset_type}")
        
        print(f"✓ Loaded {dataset_type}: {data.shape[0]:,} rows × {data.shape[1]} cols")
        self.datasets[dataset_type] = data
        return data
    
    def load_all(self, paths: Dict[str, Union[str, Path]] = None,
                 dataframes: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """Load multiple datasets at once"""
        
        print("=" * 70)
        print("LOADING DATASETS")
        print("=" * 70 + "\n")
        
        paths = paths or {}
        dataframes = dataframes or {}
        
        for dataset_type in self.DATASETS.keys():
            if dataset_type in dataframes:
                self.load(dataset_type, df=dataframes[dataset_type])
            elif dataset_type in paths:
                self.load(dataset_type, path=paths[dataset_type])
        
        print(f"\n✓ Loaded {len(self.datasets)} datasets\n")
        return self.datasets


class TemporalProcessor:
    """Handles all temporal feature engineering"""
    
    @staticmethod
    def add_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add comprehensive temporal features"""
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['hour'] = df[date_col].dt.hour
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['day_name'] = df[date_col].dt.day_name()
        
        # Derived features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # Time periods
        df['time_period'] = pd.cut(
            df['hour'],
            bins=[-1, 6, 10, 16, 20, 24],
            labels=['night', 'morning_rush', 'midday', 'evening_rush', 'evening']
        )
        
        df['is_morning_peak'] = df['hour'].between(7, 9).astype(int)
        df['is_evening_peak'] = df['hour'].between(17, 19).astype(int)
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        return df


class CoordinateProcessor:
    """Handles coordinate extraction and validation"""
    
    @staticmethod
    def extract_coordinates(df: pd.DataFrame, coord_col: str,
                          lat_name: str = 'latitude',
                          lon_name: str = 'longitude') -> pd.DataFrame:
        """Extract lat/lon from Geo Point column"""
        
        if coord_col not in df.columns:
            return df
        
        coords = df[coord_col].str.split(',', expand=True)
        df[lat_name] = pd.to_numeric(coords[0], errors='coerce')
        df[lon_name] = pd.to_numeric(coords[1], errors='coerce')
        
        return df
    
    @staticmethod
    def validate_bologna_coordinates(df: pd.DataFrame,
                                     lat_col: str = 'latitude',
                                     lon_col: str = 'longitude') -> pd.DataFrame:
        """Validate coordinates are within Bologna bounds"""
        
        # Bologna bounding box
        LAT_MIN, LAT_MAX = 44.4, 44.6
        LON_MIN, LON_MAX = 11.2, 11.5
        
        if lat_col in df.columns and lon_col in df.columns:
            invalid = ((df[lat_col] < LAT_MIN) | (df[lat_col] > LAT_MAX) |
                      (df[lon_col] < LON_MIN) | (df[lon_col] > LON_MAX))
            
            if invalid.sum() > 0:
                print(f"  ⚠ {invalid.sum()} invalid coordinates (outside Bologna)")
                df.loc[invalid, [lat_col, lon_col]] = np.nan
        
        return df


class MissingValueHandler:
    """Handles missing value imputation strategies"""
    
    @staticmethod
    def analyze(df: pd.DataFrame, name: str = 'dataset') -> pd.DataFrame:
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
    
    @staticmethod
    def impute_smart(df: pd.DataFrame, columns: List[str],
                    group_col: Optional[str] = None) -> pd.DataFrame:
        """Smart imputation based on data characteristics"""
        
        for col in columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct == 0:
                continue
            
            if missing_pct < 5:
                # Low missing: interpolation
                if group_col and group_col in df.columns:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda x: x.interpolate(method='linear', limit_direction='both')
                    )
                else:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            elif missing_pct < 20:
                # Medium missing: group mean or median
                if group_col and group_col in df.columns:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                else:
                    df[col] = df[col].fillna(df[col].median())
            
            else:
                # High missing: forward/backward fill + median
                df[col] = df[col].ffill().bfill().fillna(df[col].median())
        
        return df


class WeatherProcessor:
    """Handles weather data processing and integration"""
    
    @staticmethod
    def process_temperature(df: pd.DataFrame) -> pd.DataFrame:
        """Process temperature data"""
        
        df['Data'] = pd.to_datetime(df['Data'])
        
        temp_cols = ['Temperatura media', 'Temperatura massima', 'Temperatura minima']
        for col in temp_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.rename(columns={
            'Data': 'datetime',
            'Temperatura media': 'temperature',
            'Temperatura massima': 'temp_max',
            'Temperatura minima': 'temp_min'
        }, inplace=True)
        
        # Derived features
        if 'temp_max' in df.columns and 'temp_min' in df.columns:
            df['temp_range'] = df['temp_max'] - df['temp_min']
        
        if 'temperature' in df.columns:
            df['is_hot_day'] = (df['temperature'] >= 25).astype(int)
            df['is_very_hot_day'] = (df['temperature'] >= 30).astype(int)
        
        return df
    
    @staticmethod
    def process_precipitation(df: pd.DataFrame) -> pd.DataFrame:
        """Process precipitation data"""
        
        df['Data'] = pd.to_datetime(df['Data'])
        df['Precipitazioni (mm)'] = pd.to_numeric(df['Precipitazioni (mm)'], errors='coerce')
        
        df.rename(columns={
            'Data': 'datetime',
            'Precipitazioni (mm)': 'precipitation_mm'
        }, inplace=True)
        
        df['is_rainy_day'] = (df['precipitation_mm'] > 0.1).astype(int)
        
        # Rolling 7-day precipitation
        df = df.sort_values('datetime')
        df['precipitation_7d'] = df['precipitation_mm'].rolling(window=7, min_periods=1).sum()
        
        return df
    
    @staticmethod
    def merge_weather(mobility_df: pd.DataFrame,
                     weather_df: pd.DataFrame,
                     mobility_date_col: str = 'Data') -> pd.DataFrame:
        """Merge weather data with mobility data"""
        
        # Determine weather resolution
        time_diff = weather_df['datetime'].diff().median()
        is_hourly = time_diff <= pd.Timedelta(hours=1)
        
        if is_hourly:
            mobility_df['merge_key'] = mobility_df[mobility_date_col].dt.floor('H')
            weather_df['merge_key'] = weather_df['datetime'].dt.floor('H')
        else:
            mobility_df['merge_key'] = mobility_df[mobility_date_col].dt.date
            weather_df['merge_key'] = weather_df['datetime'].dt.date
        
        # Select weather columns
        weather_cols = [col for col in weather_df.columns 
                       if col not in ['datetime', 'merge_key']]
        
        merged = mobility_df.merge(
            weather_df[['merge_key'] + weather_cols],
            on='merge_key',
            how='left'
        ).drop('merge_key', axis=1)
        
        return merged


class TALEADataPreprocessor:
    """
    Main preprocessor class - orchestrates all preprocessing operations
    """
    
    def __init__(self, data_dir: Union[str, Path] = 'dataset/raw_data'):
        self.loader = DataLoader(data_dir)
        self.temporal = TemporalProcessor()
        self.coords = CoordinateProcessor()
        self.missing = MissingValueHandler()
        self.weather_proc = WeatherProcessor()
        
        self.datasets = {}
        self.weather_data = None
    
    def load_datasets(self, **kwargs) -> 'TALEADataPreprocessor':
        """Load datasets (paths or dataframes)"""
        self.datasets = self.loader.load_all(**kwargs)
        return self
    
    def process_bicycle(self) -> pd.DataFrame:
        """Process bicycle counter data"""
        
        print("\n" + "=" * 70)
        print("PROCESSING BICYCLE DATA")
        print("=" * 70)
        
        df = self.datasets['bicycle'].copy()
        config = DataLoader.DATASETS['bicycle']
        
        # Parse datetime
        df = self.temporal.add_features(df, config.date_column)
        
        # Extract coordinates
        if config.coordinate_column:
            df = self.coords.extract_coordinates(df, config.coordinate_column)
            df = self.coords.validate_bologna_coordinates(df)
        
        # Clean numeric columns
        for col in config.count_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Feature engineering
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
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[config.date_column, config.location_column])
        
        # Handle missing values
        self.missing.analyze(df, 'bicycle')
        df = self.missing.impute_smart(
            df, 
            config.count_columns,
            group_col=config.location_column
        )
        
        print(f"\n✓ Processed: {df.shape[0]:,} records")
        print(f"  Date range: {df[config.date_column].min().date()} to {df[config.date_column].max().date()}")
        print(f"  Counters: {df[config.location_column].nunique()}")
        
        self.datasets['bicycle'] = df
        return df
    
    def process_pedestrian(self) -> pd.DataFrame:
        """Process pedestrian flow data"""
        
        print("\n" + "=" * 70)
        print("PROCESSING PEDESTRIAN DATA")
        print("=" * 70)
        
        df = self.datasets['pedestrian'].copy()
        config = DataLoader.DATASETS['pedestrian']
        
        # Parse datetime
        df = self.temporal.add_features(df, config.date_column)
        
        # Clean counts
        df['Numero di visitatori'] = pd.to_numeric(
            df['Numero di visitatori'], errors='coerce'
        ).fillna(0)
        
        # Create route identifier
        df['route'] = df['Area provenienza'] + ' → ' + df['Area Arrivo']
        
        # Handle missing values
        self.missing.analyze(df, 'pedestrian')
        df = self.missing.impute_smart(df, config.count_columns, group_col='route')
        
        print(f"\n✓ Processed: {df.shape[0]:,} records")
        print(f"  Routes: {df['route'].nunique()}")
        
        self.datasets['pedestrian'] = df
        return df
    
    def process_traffic(self) -> pd.DataFrame:
        """Process traffic monitor data"""

        print("\n" + "=" * 70)
        print("PROCESSING TRAFFIC DATA")
        print("=" * 70)

        df = self.datasets['traffic'].copy()
        config = DataLoader.DATASETS['traffic']

        df[config.date_column] = pd.to_datetime(df[config.date_column])

        # Identify time slot columns (format: "0.00 - 0.30", "0.30 - 1.00", etc.)
        time_cols = [col for col in df.columns 
                     if ' - ' in str(col) and any(c.isdigit() for c in str(col))]

        print(f"  Found {len(time_cols)} time slot columns")

        if len(time_cols) > 0:
            # Preserve metadata columns
            id_vars = [col for col in df.columns 
                       if col not in time_cols + ['tot']]

            # Reshape from wide to long format
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

            # Extract hour from time slot (e.g., "7.30 - 8.00" -> 7)
            df_long['hour'] = df_long['time_slot'].str.extract(r'^(\d+)\.')[0].astype(int)

            # Extract minutes for more precise timing
            df_long['minute'] = df_long['time_slot'].str.extract(r'^(\d+)\.(\d+)')[1].astype(int)

            # Create precise datetime
            df_long['datetime'] = (
                df_long[config.date_column] + 
                pd.to_timedelta(df_long['hour'], unit='h') +
                pd.to_timedelta(df_long['minute'], unit='m')
            )

            # Add temporal features
            df_long = self.temporal.add_features(df_long, 'datetime')

            # Add traffic-specific features
            df_long['is_rush_hour'] = df_long['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

            # Clean location/direction columns
            if 'VIA_SPIRA' in df_long.columns:
                df_long['VIA_SPIRA'] = df_long['VIA_SPIRA'].str.strip()
            if 'DIREZIONE' in df_long.columns:
                df_long['DIREZIONE'] = df_long['DIREZIONE'].str.strip()

            df = df_long
        else:
            # If already in long format, just add temporal features
            df = self.temporal.add_features(df, config.date_column)

        # Handle missing values
        self.missing.analyze(df, 'traffic')

        # Impute missing vehicle counts by location and hour
        if 'vehicle_count' in df.columns and 'VIA_SPIRA' in df.columns:
            df = self.missing.impute_smart(
                df, 
                ['vehicle_count'],
                group_col='VIA_SPIRA'
            )

        # Remove obvious outliers (cap at 99.9th percentile)
        if 'vehicle_count' in df.columns:
            cap_value = df['vehicle_count'].quantile(0.999)
            outliers = df['vehicle_count'] > cap_value
            if outliers.sum() > 0:
                print(f"  ⚠ Capped {outliers.sum()} outlier values at {cap_value:.0f}")
                df.loc[outliers, 'vehicle_count'] = cap_value

        print(f"\n✓ Processed: {df.shape[0]:,} records")
        if 'datetime' in df.columns:
            print(f"  Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        if 'VIA_SPIRA' in df.columns:
            print(f"  Locations: {df['VIA_SPIRA'].nunique()}")

        self.datasets['traffic'] = df
        return df
    
    def load_weather(self, temp_path: str = None, precip_path: str = None,
                    temp_df: pd.DataFrame = None, precip_df: pd.DataFrame = None) -> pd.DataFrame:
        """Load and process weather data"""
        
        print("\n" + "=" * 70)
        print("PROCESSING WEATHER DATA")
        print("=" * 70)
        
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
        
        # Merge weather components
        if len(weather_components) > 1:
            self.weather_data = weather_components[0].merge(
                weather_components[1],
                on='datetime',
                how='outer'
            )
        elif len(weather_components) == 1:
            self.weather_data = weather_components[0]
        
        print(f"\n✓ Weather data ready: {len(self.weather_data):,} records")
        return self.weather_data
    
    def integrate_weather(self) -> 'TALEADataPreprocessor':
        """Integrate weather with mobility datasets"""
        
        if self.weather_data is None:
            print("⚠ No weather data loaded")
            return self
        
        print("\n" + "=" * 70)
        print("INTEGRATING WEATHER DATA")
        print("=" * 70)
        
        for dataset_name in ['bicycle', 'pedestrian', 'traffic']:
            if dataset_name in self.datasets:
                config = DataLoader.DATASETS[dataset_name]
                date_col = 'datetime' if 'datetime' in self.datasets[dataset_name].columns else config.date_column
                
                self.datasets[dataset_name] = self.weather_proc.merge_weather(
                    self.datasets[dataset_name],
                    self.weather_data,
                    mobility_date_col=date_col
                )
                print(f"✓ Weather integrated with {dataset_name}")
        
        return self
    
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """Process all loaded datasets"""
        
        print("\n" + "=" * 70)
        print("PROCESSING ALL DATASETS")
        print("=" * 70)
        
        if 'bicycle' in self.datasets:
            self.process_bicycle()
        
        if 'pedestrian' in self.datasets:
            self.process_pedestrian()
        
        if 'traffic' in self.datasets:
            self.process_traffic()
        
        print("\n" + "=" * 70)
        print("✓ ALL PROCESSING COMPLETE")
        print("=" * 70)
        
        for name, df in self.datasets.items():
            print(f"  • {name:12s}: {df.shape[0]:8,} rows × {df.shape[1]:3d} cols")
        
        return self.datasets
    
    def save_processed(self, output_dir: Union[str, Path] = 'dataset/processed_data'):
        """Save all processed datasets"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}/")
        
        for name, df in self.datasets.items():
            filepath = output_dir / f'talea_{name}_processed.csv'
            df.to_csv(filepath, index=False)
            print(f"  ✓ {filepath.name}")
        
        if self.weather_data is not None:
            weather_path = output_dir / 'talea_weather_processed.csv'
            self.weather_data.to_csv(weather_path, index=False)
            print(f"  ✓ {weather_path.name}")


# USAGE
if __name__ == "__main__":
    # Simple usage example
    preprocessor = TALEADataPreprocessor()
    
    # Load data
    preprocessor.load_datasets(paths={
        'bicycle': 'dataset/raw_data/colonnine-conta-bici.csv',
        'pedestrian': 'dataset/raw_data/bologna-daily-mobility.csv',
        'traffic': 'dataset/raw_data/traffico-viali.csv'
    })
    
    # Process all
    processed_data = preprocessor.process_all()
    
    # Load and integrate weather
    preprocessor.load_weather(
        temp_path='dataset/weather_data/temperature_bologna.csv',
        precip_path='dataset/weather_data/precipitazioni_bologna.csv'
    )
    preprocessor.integrate_weather()
    
    # Save results
    preprocessor.save_processed()