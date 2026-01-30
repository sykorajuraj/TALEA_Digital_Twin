"""
TALEA - Civic Digital Twin: Data Loader
File: src/data_loader.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Simple, modular data loading utilities for the TALEA project.
For internal loading of datasets is TALEADataLoader in data_preprocessor.py
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Metadata for a dataset"""
    name: str
    path: str
    type: str  # 'mobility', 'weather', 'geospatial'
    description: Optional[str] = None


class DataLoader:
    """Simple data loader for TALEA datasets"""
    
    # Default dataset configurations
    DATASETS = {
        'bicycle': DatasetInfo(
            name='bicycle',
            path='mobility_data/colonnine-conta-bici.csv',
            type='mobility',
            description='Bicycle counter data'
        ),
        'pedestrian': DatasetInfo(
            name='pedestrian',
            path='mobility_data/bologna-daily-mobility.csv',
            type='mobility',
            description='Pedestrian flow data'
        ),
        'traffic': DatasetInfo(
            name='traffic',
            path='mobility_data/traffico-viali.csv',
            type='mobility',
            description='Traffic monitor data'
        ),
        'street': DatasetInfo(
            name='street',
            path='mobility_data/rifter_arcstra_li.csv',
            type='geospatial',
            description='Street network'
        ),
        'poi': DatasetInfo(
            name='poi',
            path='mobility_data/musei_gallerie_luoghi_e_teatri_storici.csv',
            type='geospatial',
            description='Points of Interest'
        ),
        'ztl': DatasetInfo(
            name='ztl',
            path='mobility_data/zona-pedonale-centro-storico.csv',
            type='geospatial',
            description='Pedestrian zones'
        ),
        'temperature': DatasetInfo(
            name='temperature',
            path='weather_data/temperature_bologna.csv',
            type='weather',
            description='Temperature data'
        ),
        'precipitation': DatasetInfo(
            name='precipitation',
            path='weather_data/precipitazioni_bologna.csv',
            type='weather',
            description='Precipitation data'
        ),
        'air_quality': DatasetInfo(
            name='air_quality',
            path='weather_data/centraline-qualita-aria.csv',
            type='weather',
            description='Air quality monitoring'
        )
    }
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize data loader
        
        Args:
            data_dir: Base directory for data files (relative or absolute path)
        """
        self.data_dir = Path(data_dir)
        self.loaded_datasets = {}
        
        # Validate that the data directory exists
        if not self.data_dir.exists():
            print(f"⚠ Warning: Data directory does not exist: {self.data_dir.resolve()}")
            print(f"   Current working directory: {Path.cwd()}")
    
    def load(self,
            dataset_name: str,
            path: Optional[Union[str, Path]] = None,
            separator: str = ';',
            **kwargs) -> pd.DataFrame:
        """
        Load a single dataset
        
        Args:
            dataset_name: Name of dataset to load
            path: Custom path (overrides default)
            separator: CSV separator
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Determine path
        if path is None:
            dataset_info = self.DATASETS[dataset_name]
            file_path = self.data_dir / dataset_info.path
        else:
            file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path.resolve()}\n"
                f"Data directory: {self.data_dir.resolve()}\n"
                f"Current working directory: {Path.cwd()}"
            )
        
        # Load data
        try:
            df = pd.read_csv(file_path, sep=separator, **kwargs)
            
            # Cache
            self.loaded_datasets[dataset_name] = df
            
            print(f"✓ Loaded {dataset_name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            return df
            
        except Exception as e:
            raise IOError(f"Failed to load {dataset_name}: {str(e)}")
    
    def load_multiple(self,
                     dataset_names: List[str],
                     skip_missing: bool = True,
                     **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load multiple datasets
        
        Args:
            dataset_names: List of dataset names to load
            skip_missing: Skip files that don't exist
            **kwargs: Additional arguments for load()
            
        Returns:
            Dictionary mapping names to DataFrames
        """
        results = {}
        
        print("=" * 70)
        print("LOADING DATASETS")
        print("=" * 70)
        print(f"Data directory: {self.data_dir.resolve()}")
        print(f"Working directory: {Path.cwd()}\n")
        
        for name in dataset_names:
            try:
                df = self.load(name, **kwargs)
                results[name] = df
            except Exception as e:
                if skip_missing:
                    print(f"⚠ Skipped {name}: {str(e).split(chr(10))[0]}")
                else:
                    raise
        
        print(f"\n✓ Loaded {len(results)}/{len(dataset_names)} datasets\n")
        
        return results
    
    def load_all(self,
                skip_missing: bool = True,
                **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets
        
        Args:
            skip_missing: Skip files that don't exist
            **kwargs: Additional arguments for load()
            
        Returns:
            Dictionary mapping names to DataFrames
        """
        return self.load_multiple(
            list(self.DATASETS.keys()),
            skip_missing=skip_missing,
            **kwargs
        )
    
    def load_by_type(self,
                    data_type: str,
                    skip_missing: bool = True,
                    **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets of a specific type
        
        Args:
            data_type: Type of data ('mobility', 'weather', 'geospatial')
            skip_missing: Skip files that don't exist
            **kwargs: Additional arguments for load()
            
        Returns:
            Dictionary mapping names to DataFrames
        """
        dataset_names = [
            name for name, info in self.DATASETS.items()
            if info.type == data_type
        ]
        
        return self.load_multiple(dataset_names, skip_missing=skip_missing, **kwargs)
    
    def get_dataset_info(self, dataset_name: Optional[str] = None) -> Union[DatasetInfo, Dict[str, DatasetInfo]]:
        """
        Get information about dataset(s)
        
        Args:
            dataset_name: Specific dataset (if None, returns all)
            
        Returns:
            DatasetInfo or dictionary of DatasetInfo
        """
        if dataset_name is None:
            return self.DATASETS
        
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return self.DATASETS[dataset_name]
    
    def list_datasets(self, data_type: Optional[str] = None) -> List[str]:
        """
        List available datasets
        
        Args:
            data_type: Filter by type (optional)
            
        Returns:
            List of dataset names
        """
        if data_type is None:
            return list(self.DATASETS.keys())
        
        return [
            name for name, info in self.DATASETS.items()
            if info.type == data_type
        ]
    
    def clear_cache(self):
        """Clear loaded dataset cache"""
        self.loaded_datasets.clear()
        print("✓ Cache cleared")
    
    def get_cached(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Get cached dataset
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Cached DataFrame or None
        """
        return self.loaded_datasets.get(dataset_name)


# Convenience functions

def load_dataset(dataset_name: str,
                data_dir: Union[str, Path],
                **kwargs) -> pd.DataFrame:
    """
    Quick load a single dataset
    
    Args:
        dataset_name: Name of dataset
        data_dir: Base data directory
        **kwargs: Additional arguments for DataLoader.load()
        
    Returns:
        DataFrame
    """
    loader = DataLoader(data_dir)
    return loader.load(dataset_name, **kwargs)


def load_mobility_data(data_dir: Union[str, Path],
                      **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Load all mobility datasets
    
    Args:
        data_dir: Base data directory
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of mobility DataFrames
    """
    loader = DataLoader(data_dir)
    return loader.load_by_type('mobility', **kwargs)


def load_weather_data(data_dir: Union[str, Path],
                     **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Load all weather datasets
    
    Args:
        data_dir: Base data directory
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of weather DataFrames
    """
    loader = DataLoader(data_dir)
    return loader.load_by_type('weather', **kwargs)


def load_geospatial_data(data_dir: Union[str, Path],
                        **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Load all geospatial datasets
    
    Args:
        data_dir: Base data directory
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of geospatial DataFrames
    """
    loader = DataLoader(data_dir)
    return loader.load_by_type('geospatial', **kwargs)