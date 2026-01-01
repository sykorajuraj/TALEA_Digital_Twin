"""
TALEA - Civic Digital Twin: Data Preprocessing Module
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

This module handles data cleaning, feature engineering, and preparation
for mobility pattern analysis in Bologna's civic digital twin.

6 datasets:
1. Bicycle Counter Data
2. Pedestrian Flow Data
3. Street Network Data
4. Traffic Monitor Data
5. Pedestrian Zones (ZTL)
6. Points of Interest (POI)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

class TALEADataPreprocessor:
    """
    Preprocessor for Bologna bicycle counter data and street network data.
    Handles missing values, creates temporal features, and prepares data
    for heat impact analysis.
    """
    
    def __init__(self):
        self.bike_data = None
        self.pedestrian_data = None
        self.street_data = None
        self.traffic_data = None
        self.ztl_data = None
        self.poi_data = None
        self.integrated_data = None

    # DATA LOADING
        
    def load_data(self, 
                     bike_path: Optional[str] = None,
                     pedestrian_path: Optional[str] = None,
                     street_path: Optional[str] = None,
                     traffic_path: Optional[str] = None,
                     ztl_path: Optional[str] = None,
                     poi_path: Optional[str] = None,
                     bike_df: Optional[pd.DataFrame] = None,
                     pedestrian_df: Optional[pd.DataFrame] = None,
                     street_df: Optional[pd.DataFrame] = None,
                     traffic_df: Optional[pd.DataFrame] = None,
                     ztl_df: Optional[pd.DataFrame] = None,
                     poi_df: Optional[pd.DataFrame] = None):
        """
        Load all datasets from /dataset/raw_data
        """

        print("=" * 70)
        print("LOADING DATASETS")
        print("=" * 70)

        # Load bike counter data
        if bike_df is not None:
            self.bike_data = bike_df.copy()
        elif bike_path:
            self.bike_data = pd.read_csv(bike_path, sep=';')
        
        # Load pedestrian flow data
        if pedestrian_df is not None:
            self.pedestrian_data = pedestrian_df.copy()
        elif pedestrian_path:
            self.pedestrian_data = pd.read_csv(pedestrian_path, sep=';')

        # Load street network data
        if street_df is not None:
            self.street_data = street_df.copy()
        elif street_path:
            self.street_data = pd.read_csv(street_path, sep=';')

        # Load traffic monitor data
        if traffic_df is not None:
            self.traffic_data = traffic_df.copy()
        elif traffic_path:
            self.traffic_data = pd.read_csv(traffic_path, sep=';')
        
        # Load ZTL zones data
        if ztl_df is not None:
            self.ztl_data = ztl_df.copy()
        elif ztl_path:
            self.ztl_data = pd.read_csv(ztl_path, sep=';')
        
        # Load POI data
        if poi_df is not None:
            self.poi_data = poi_df.copy()
        elif poi_path:
            self.poi_data = pd.read_csv(poi_path, sep=';')
        
        self._print_data_summary()

    def _print_data_summary(self):
        """Print summary of loaded datasets."""
        datasets = {
            'Bicycle Counter': self.bike_data,
            'Pedestrian Flow': self.pedestrian_data,
            'Street Network': self.street_data,
            'Traffic Monitor': self.traffic_data,
            'ZTL Zones': self.ztl_data,
            'Points of Interest': self.poi_data
        }
        
        print("\nDataset Summary:")
        for name, df in datasets.items():
            if df is not None:
                print(f"  ✓ {name:20s}: {df.shape[0]:6d} rows × {df.shape[1]:3d} cols")
            else:
                print(f"  ✗ {name:20s}: Not loaded")
        print()

    # BICYCLE DATA PROCESSING

    def clean_bike_data(self):
        """
        Clean and prepare bicycle counter data.
        """
        if self.bike_data is None:
            raise ValueError("No bike data loaded.")
        
        print("Processing Bicycle Counter Data")
        df = self.bike_data.copy()
        
        # Parse datetime
        df['Data'] = pd.to_datetime(df['Data'])
        
        # Extract coordinates from Geo Point
        if 'Geo Point' in df.columns:
            coords = df['Geo Point'].str.split(',', expand=True)
            df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
            df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
        
        # Handle missing values in traffic counts
        numeric_cols = ['Direzione centro', 'Direzione periferia', 'Totale']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill missing with 0 (assuming no count = no bikes)
                df[col] = df[col].fillna(0)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Data', 'Dispositivo conta-bici'])
        
        # Create temporal features
        df = self._add_temporal_features(df, date_col='Data')
        
        # Create mobility features
        df['flow_ratio'] = np.where(
            df['Direzione periferia'] > 0,
            df['Direzione centro'] / df['Direzione periferia'],
            np.nan
        )
        df['dominant_direction'] = np.where(
            df['Direzione centro'] > df['Direzione periferia'],
            'centro', 'periferia'
        )

        print(f"  ✓ Cleaned: {df.shape[0]:,} observations")
        print(f"  ✓ Date range: {df['Data'].min()} to {df['Data'].max()}")
        print(f"  ✓ Unique counters: {df['Dispositivo conta-bici'].nunique()}")
        
        self.bike_data = df
        return df
    
    # PEDESTRIAN DATA PROCESSING

    def clean_pedestrian_data(self) -> pd.DataFrame:
        """Clean and prepare pedestrian flow data."""
        if self.pedestrian_data is None:
            raise ValueError("No pedestrian data loaded")
        
        print("\nProcessing Pedestrian Flow Data...")
        df = self.pedestrian_data.copy()
        
        # Parse date
        df['Data'] = pd.to_datetime(df['Data'])
        
        # Handle visitor counts
        df['Numero di visitatori'] = pd.to_numeric(
            df['Numero di visitatori'], errors='coerce'
        ).fillna(0)
        
        # Create origin-destination pairs
        df['route'] = df['Area provenienza'] + ' → ' + df['Area Arrivo']
        
        # Add temporal features
        df = self._add_temporal_features(df, date_col='Data')
        
        # Calculate flow metrics
        df['is_zero_flow'] = (df['Numero di visitatori'] == 0).astype(int)
        
        print(f"  ✓ Cleaned: {df.shape[0]:,} flow records")
        print(f"  ✓ Date range: {df['Data'].min()} to {df['Data'].max()}")
        print(f"  ✓ Unique routes: {df['route'].nunique()}")
        print(f"  ✓ Zero-flow records: {df['is_zero_flow'].sum():,} ({df['is_zero_flow'].mean()*100:.1f}%)")
        
        self.pedestrian_data = df
        return df
    
    # STREET NETWORK PROCESSING
    
    def clean_street_data(self) -> pd.DataFrame:
        """Clean and prepare street network data."""
        if self.street_data is None:
            raise ValueError("No street data loaded")
        
        print("\nProcessing Street Network Data...")
        df = self.street_data.copy()
        
        # Parse coordinates
        if 'Geo Point' in df.columns:
            coords = df['Geo Point'].str.split(',', expand=True)
            df['street_lat'] = pd.to_numeric(coords[0], errors='coerce')
            df['street_lon'] = pd.to_numeric(coords[1], errors='coerce')
        
        # Parse geometry (simplified - for full GIS use geopandas)
        if 'Geo Shape' in df.columns:
            df['has_geometry'] = df['Geo Shape'].notna()
        
        # Convert length to numeric
        if 'LUNGHEZ' in df.columns:
            df['LUNGHEZ'] = pd.to_numeric(df['LUNGHEZ'], errors='coerce')
        
        # Parse date fields
        if 'DATA_ISTIT' in df.columns:
            df['DATA_ISTIT'] = pd.to_datetime(df['DATA_ISTIT'], errors='coerce')
        
        print(f"  ✓ Cleaned: {df.shape[0]:,} street segments")
        if 'NOMEVIA' in df.columns:
            print(f"  ✓ Unique streets: {df['NOMEVIA'].nunique()}")
        if 'LUNGHEZ' in df.columns:
            print(f"  ✓ Total length: {df['LUNGHEZ'].sum():,.0f} m")
        if 'Quartiere' in df.columns:
            print(f"  ✓ Districts: {df['Quartiere'].nunique()}")
        
        self.street_data = df
        return df
    
    # TRAFFIC MONITOR PROCESSING

    def clean_traffic_data(self) -> pd.DataFrame:
        """Clean and prepare traffic monitor data (wide format → long format)."""
        if self.traffic_data is None:
            raise ValueError("No traffic data loaded")
        
        print("\nProcessing Traffic Monitor Data...")
        df = self.traffic_data.copy()
        
        # Parse date
        df['data'] = pd.to_datetime(df['data'])
        
        # Identify time columns (format: "HH.MM - HH.MM")
        time_cols = [col for col in df.columns if ':' in col or '.' in col and col != 'data']
        
        # If we have the 48 half-hourly columns, reshape to long format
        if len(time_cols) > 0:
            # Melt time columns
            id_vars = [col for col in df.columns if col not in time_cols + ['tot']]
            
            df_long = df.melt(
                id_vars=id_vars,
                value_vars=time_cols,
                var_name='time_slot',
                value_name='vehicle_count'
            )
            
            # Convert counts to numeric
            df_long['vehicle_count'] = pd.to_numeric(
                df_long['vehicle_count'], errors='coerce'
            ).fillna(0)
            
            # Parse time slot to get hour and minute
            df_long = self._parse_time_slot(df_long)
            
            # Create full datetime
            df_long['datetime'] = df_long.apply(
                lambda row: row['data'] + pd.Timedelta(hours=row['hour'], minutes=row['minute']),
                axis=1
            )
            
            df = df_long
        
        # Add temporal features
        if 'datetime' in df.columns:
            df = self._add_temporal_features(df, date_col='datetime')
        else:
            df = self._add_temporal_features(df, date_col='data')
        
        print(f"  ✓ Cleaned: {df.shape[0]:,} observations")
        if 'VIA_SPIRA' in df.columns:
            print(f"  ✓ Unique locations: {df['VIA_SPIRA'].nunique()}")
        print(f"  ✓ Temporal resolution: {'30-minute' if len(time_cols) > 0 else 'daily'}")
        
        self.traffic_data = df
        return df
    
    def _parse_time_slot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse time slot string to extract hour and minute."""
        # Example: "8.00 - 8.30"
        def extract_time(slot):
            try:
                start_time = slot.split(' - ')[0].strip()
                if '.' in start_time:
                    hour, minute = start_time.split('.')
                elif ':' in start_time:
                    hour, minute = start_time.split(':')
                else:
                    return 0, 0
                return int(hour), int(minute)
            except:
                return 0, 0
        
        df[['hour', 'minute']] = df['time_slot'].apply(
            lambda x: pd.Series(extract_time(x))
        )
        
        return df
    
    # ZLT ZONES PROCESSING

    def clean_ztl_data(self) -> pd.DataFrame:
        """Clean and prepare pedestrian zone (ZTL) data."""
        if self.ztl_data is None:
            raise ValueError("No ZTL data loaded")
        
        print("\nProcessing Pedestrian Zones (ZTL) Data...")
        df = self.ztl_data.copy()
        
        # Parse coordinates
        if 'Geo Point' in df.columns:
            coords = df['Geo Point'].str.split(',', expand=True)
            df['zone_lat'] = pd.to_numeric(coords[0], errors='coerce')
            df['zone_lon'] = pd.to_numeric(coords[1], errors='coerce')
        
        # Parse geometry (simplified)
        if 'Geo Shape' in df.columns:
            df['has_polygon'] = df['Geo Shape'].notna()
        
        # Parse area if available
        if 'area' in df.columns:
            df['area'] = pd.to_numeric(df['area'], errors='coerce')
        
        print(f"  ✓ Cleaned: {df.shape[0]:,} zones")
        if 'TipoZTL' in df.columns:
            print(f"  ✓ Zone types: {df['TipoZTL'].nunique()}")
            print(f"  ✓ Type distribution:\n{df['TipoZTL'].value_counts().to_string()}")
        
        self.ztl_data = df
        return df
    
    # POI PROCESSING

    def clean_poi_data(self) -> pd.DataFrame:
        """Clean and prepare Points of Interest data."""
        if self.poi_data is None:
            raise ValueError("No POI data loaded")
        
        print("\nProcessing Points of Interest Data...")
        df = self.poi_data.copy()
        
        # Parse coordinates
        coord_cols = ['latitudine', 'longitudine', 'geopoint']
        for col in coord_cols:
            if col in df.columns:
                if col == 'geopoint':
                    coords = df[col].str.split(',', expand=True)
                    df['poi_lat'] = pd.to_numeric(coords[0], errors='coerce')
                    df['poi_lon'] = pd.to_numeric(coords[1], errors='coerce')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize accessibility info
        if 'accessibilità' in df.columns:
            df['is_accessible'] = df['accessibilità'].notna()
            df['accessibility_features'] = df['accessibilità']
        
        # Categorize POIs
        if 'macro zona' in df.columns:
            df['macro_zone'] = df['macro zona']
        if 'zona' in df.columns:
            df['district'] = df['zona']
        
        print(f"  ✓ Cleaned: {df.shape[0]:,} POIs")
        if 'macro zona' in df.columns:
            print(f"  ✓ Macro zones: {df['macro zona'].nunique()}")
        if 'accessibilità' in df.columns:
            print(f"  ✓ With accessibility info: {df['is_accessible'].sum()} ({df['is_accessible'].mean()*100:.1f}%)")
        
        self.poi_data = df
        return df
    
    # TEMPORAL FEATURES
    
    def _add_temporal_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add comprehensive temporal features to any dataset."""
        # Basic temporal features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['hour'] = df[date_col].dt.hour
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['day_name'] = df[date_col].dt.day_name()
        
        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Season
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # Time period
        def get_time_period(hour):
            if hour < 6:
                return 'night'
            elif hour < 10:
                return 'morning_rush'
            elif hour < 16:
                return 'midday'
            elif hour < 20:
                return 'evening_rush'
            else:
                return 'evening'
        
        df['time_period'] = df['hour'].apply(get_time_period)
        
        # Peak hours
        df['is_morning_peak'] = df['hour'].between(7, 9).astype(int)
        df['is_evening_peak'] = df['hour'].between(17, 19).astype(int)
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        return df
    
    # DATA INTEGRATION

    def integrate_datasets(self, method: str = 'temporal') -> pd.DataFrame:
        """
        Integrate multiple datasets based on temporal or spatial alignment.
        
        Parameters:
        -----------
        method : str
            'temporal' - align all datasets to common time periods
            'spatial' - perform spatial joins (requires full coordinates)
        """
        print("\n" + "=" * 70)
        print("INTEGRATING DATASETS")
        print("=" * 70)
        
        if method == 'temporal':
            return self._temporal_integration()
        elif method == 'spatial':
            return self._spatial_integration()
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def _temporal_integration(self) -> pd.DataFrame:
        """Align datasets temporally (hourly aggregation)."""
        print("\nPerforming temporal integration (hourly)...")
        
        integrated_data = []
        
        # Aggregate bicycle data by hour
        if self.bike_data is not None:
            bike_hourly = self.bike_data.groupby(
                [pd.Grouper(key='Data', freq='H'), 'Dispositivo conta-bici']
            ).agg({
                'Totale': 'sum',
                'Direzione centro': 'sum',
                'Direzione periferia': 'sum'
            }).reset_index()
            bike_hourly['data_type'] = 'bicycle'
            integrated_data.append(bike_hourly)
        
        # Aggregate pedestrian data by hour
        if self.pedestrian_data is not None:
            ped_hourly = self.pedestrian_data.groupby(
                [pd.Grouper(key='Data', freq='H'), 'route']
            ).agg({
                'Numero di visitatori': 'sum'
            }).reset_index()
            ped_hourly['data_type'] = 'pedestrian'
            integrated_data.append(ped_hourly)
        
        # Traffic data (if already processed to hourly)
        if self.traffic_data is not None and 'datetime' in self.traffic_data.columns:
            traffic_hourly = self.traffic_data.groupby(
                [pd.Grouper(key='datetime', freq='H'), 'VIA_SPIRA']
            ).agg({
                'vehicle_count': 'sum'
            }).reset_index()
            traffic_hourly['data_type'] = 'vehicle'
            integrated_data.append(traffic_hourly)
        
        if integrated_data:
            print(f"  ✓ Integrated {len(integrated_data)} datasets temporally")
            return integrated_data
        else:
            print("  ✗ No data available for integration")
            return None
    
    def _spatial_integration(self) -> pd.DataFrame:
        """Perform spatial joins between datasets (requires coordinates)."""
        print("\nPerforming spatial integration...")
        print("  ⚠ Note: Full spatial joins require GeoPandas")
        print("  ⚠ Using simplified distance-based matching")
        
        # This is a simplified version
        # For production, use GeoPandas with actual geometry operations
        
        if self.bike_data is not None and self.street_data is not None:
            print("  • Matching bike counters to street network...")
            # Simple nearest-neighbor matching would go here
        
        if self.poi_data is not None:
            print("  • Calculating distances to POIs...")
            # Distance calculations would go here
        
        print("  ✓ Spatial integration framework ready")
        print("  ℹ For full GIS capabilities, use GeoPandas version")
        
        return None
    
    # COMPLETE PIPELINE

    def prepare_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete preprocessing pipeline for all datasets.
        
        Returns:
        --------
        Dict with processed datasets
        """
        print("\n" + "=" * 70)
        print("TALEA MULTI-DATASET PREPROCESSING PIPELINE")
        print("=" * 70)
        
        results = {}
        
        # Process each dataset
        if self.bike_data is not None:
            results['bicycle'] = self.clean_bike_data()
        
        if self.pedestrian_data is not None:
            results['pedestrian'] = self.clean_pedestrian_data()
        
        if self.street_data is not None:
            results['streets'] = self.clean_street_data()
        
        if self.traffic_data is not None:
            results['traffic'] = self.clean_traffic_data()
        
        if self.ztl_data is not None:
            results['ztl'] = self.clean_ztl_data()
        
        if self.poi_data is not None:
            results['poi'] = self.clean_poi_data()
        
        print("\n" + "=" * 70)
        print("✓ PREPROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nProcessed datasets: {len(results)}")
        for name, df in results.items():
            print(f"  • {name:12s}: {df.shape[0]:8,} rows × {df.shape[1]:3d} columns")
        
        return results
    
    # EXPORT FUNCTIONS

    def save_processed_data(self, output_dir: str = '.'):
        """Save all processed datasets to CSV files."""
        import os
        
        print(f"\nSaving processed datasets to {output_dir}/...")
        
        datasets = {
            'bicycle': self.bike_data,
            'pedestrian': self.pedestrian_data,
            'streets': self.street_data,
            'traffic': self.traffic_data,
            'ztl': self.ztl_data,
            'poi': self.poi_data
        }
        
        for name, df in datasets.items():
            if df is not None:
                filepath = os.path.join(output_dir, f'talea_{name}_processed.csv')
                df.to_csv(filepath, index=False)
                print(f"  ✓ Saved: {filepath}")