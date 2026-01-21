"""
TALEA - Civic Digital Twin: Data Quality Check Module
File: src/data_quality.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Data validation and quality checks module before analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA QUALITY CHECKER
# ============================================================================

class TALEADataQualityChecker:
    """Data validator before analysis"""    
    def __init__(self, config: Optional['ProcessingConfig'] = None):
        """
        Initialize the data quality checker.
        
        Parameters:
        -----------
        config : ProcessingConfig, optional
            Configuration object containing validation parameters
            (e.g., geographic bounds for Bologna)
        """
        self.config = config or ProcessingConfig()
        self.quality_reports = {}
    
    def run_complete_quality_check(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run comprehensive quality checks on all datasets.
        
        Parameters:
        -----------
        datasets : dict
            Dictionary of dataset name -> DataFrame
        
        Returns:
        --------
        quality_report : dict
            Complete quality report for all datasets
        """
        print("=" * 70)
        print("DATA QUALITY CHECK")
        print("=" * 70)
        
        quality_report = {}
        
        for name, df in datasets.items():
            print(f"\n{'='*70}")
            print(f"Checking: {name.upper()}")
            print(f"{'='*70}")
            
            try:
                dataset_report = self._check_single_dataset(df, name)
                quality_report[name] = dataset_report
                self._print_dataset_summary(name, dataset_report)
            except Exception as e:
                print(f"⚠ Quality check failed for {name}: {str(e)}")
                quality_report[name] = {'error': str(e)}
        
        self.quality_reports = quality_report
        return quality_report
    
    def _check_single_dataset(self, df: pd.DataFrame, name: str) -> Dict:
        """
        Check a single dataset for quality issues.
        """
        dataset_report = {}
        
        # 1. Completeness
        dataset_report['completeness'] = self._check_completeness(df)
        
        # 2. Temporal continuity
        date_cols = ['Data', 'datetime', 'data']
        date_col = next((col for col in date_cols if col in df.columns), None)
        if date_col:
            dataset_report['temporal'] = self._check_temporal_continuity(df, date_col)
        
        # 3. Outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            dataset_report['outliers'] = self._check_outliers(df, numeric_cols)
        
        # 4. Duplicates
        dataset_report['duplicates'] = self._check_duplicates(df)
        
        # 5. Data ranges
        dataset_report['ranges'] = self._check_data_ranges(df, name)
        
        return dataset_report
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """
        Check data completeness.
        
        Calculates:
        - Total rows and columns
        - Missing values count and percentage
        - Complete rows (rows with no missing values)
        """
        completeness = {
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'complete_rows': df.notna().all(axis=1).sum(),
            'complete_rows_pct': (df.notna().all(axis=1).sum() / len(df)) * 100
        }
        
        print(f"\n1. Completeness:")
        print(f"   Total records: {completeness['total_rows']:,}")
        print(f"   Complete records: {completeness['complete_rows']:,} ({completeness['complete_rows_pct']:.1f}%)")
        print(f"   Missing values: {completeness['missing_values']:,} ({completeness['missing_pct']:.2f}%)")
        
        return completeness
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """
        Check for duplicate records.
        """
        duplicates = {
            'total_duplicates': df.duplicated().sum(),
            'duplicate_pct': (df.duplicated().sum() / len(df)) * 100
        }
        
        print(f"\n2. Duplicates:")
        print(f"   Duplicate rows: {duplicates['total_duplicates']:,} ({duplicates['duplicate_pct']:.2f}%)")
        
        return duplicates
    
    def _check_data_ranges(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Validate data ranges for dataset-specific columns.
        
        Checks:
        - Count columns for negative values
        - Geographic coordinates within Bologna bounds
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to check
        dataset_name : str
            Name of dataset for specific validation rules
        
        Returns:
        --------
        ranges : dict
            Range validation results
        """
        ranges = {}
        print(f"\n3. Data Ranges:")
        
        # Count columns - should not be negative
        count_cols = {
            'bicycle': ['Totale', 'Direzione centro', 'Direzione periferia'],
            'pedestrian': ['Numero di visitatori'],
            'traffic': ['vehicle_count']
        }
        
        if dataset_name in count_cols:
            for col in count_cols[dataset_name]:
                if col in df.columns:
                    negative = (df[col] < 0).sum()
                    ranges[col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'negative_values': int(negative)
                    }
                    
                    if negative > 0:
                        print(f"   ⚠ {col}: {negative} negative values")
                    else:
                        print(f"   ✓ {col}: range [{df[col].min():.0f}, {df[col].max():.0f}]")
        
        # Coordinate validation - Bologna specific bounds
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_min, lat_max = self.config.BOLOGNA_LAT_RANGE
            lon_min, lon_max = self.config.BOLOGNA_LON_RANGE
            
            out_of_range = (
                (df['latitude'] < lat_min) | (df['latitude'] > lat_max) |
                (df['longitude'] < lon_min) | (df['longitude'] > lon_max)
            ).sum()
            
            if out_of_range > 0:
                print(f"   ⚠ Coordinates: {out_of_range} values outside Bologna")
            else:
                print(f"   ✓ Coordinates: all within Bologna bounds")
        
        return ranges
    
    def _check_temporal_continuity(self, df: pd.DataFrame, date_col: str) -> Dict:
        """
        Check temporal continuity in time series data.
        
        Automatically detects expected frequency (hourly or daily) based on
        median time difference, then identifies gaps larger than expected.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with temporal data
        date_col : str
            Name of the datetime column
        
        Returns:
        --------
        continuity : dict
            Temporal continuity statistics including gaps found
        """
        print(f"\n4. Temporal Continuity:")
        
        df_sorted = df.sort_values(date_col).copy()
        time_diffs = df_sorted[date_col].diff()
        
        # Detect expected frequency automatically
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(hours=1):
            expected_diff = pd.Timedelta(hours=1)
            freq_name = "hourly"
        else:
            expected_diff = pd.Timedelta(days=1)
            freq_name = "daily"
        
        # Find gaps (more than 1.5x expected difference)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        
        continuity = {
            'expected_frequency': freq_name,
            'total_gaps': len(gaps),
            'largest_gap': str(gaps.max()) if len(gaps) > 0 else "None",
            'total_gap_time': str(gaps.sum()) if len(gaps) > 0 else "None"
        }
        
        if len(gaps) > 0:
            print(f"   ⚠ Found {len(gaps)} gaps (expected {freq_name})")
            print(f"   Largest gap: {gaps.max()}")
        else:
            print(f"   ✓ No significant gaps ({freq_name})")
        
        return continuity
    
    def _check_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """
        Detect outliers using IQR (Interquartile Range) method.
        
        Uses 3×IQR rule: values beyond Q1 - 3×IQR or Q3 + 3×IQR are flagged.
        This method is more robust than z-score for non-normal distributions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to check
        numeric_cols : list
            List of numeric column names to check
        
        Returns:
        --------
        outliers : dict
            Outlier statistics for each numeric column
        """
        print(f"\n5. Outlier Detection (IQR method):")
        
        outliers = {}
        
        # Skip certain columns that shouldn't be checked for outliers
        skip_cols = ['year', 'month', 'day', 'hour', 'latitude', 'longitude', 
                    'day_of_week', 'week_of_year']
        
        for col in numeric_cols:
            if col in skip_cols:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            # Calculate IQR bounds
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Count outliers
            outlier_count = ((values < lower_bound) | (values > upper_bound)).sum()
            outlier_pct = (outlier_count / len(values)) * 100
            
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': float(outlier_pct),
                'bounds': (float(lower_bound), float(upper_bound))
            }
            
            if outlier_count > 0:
                print(f"   ⚠ {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
            else:
                print(f"   ✓ {col}: no extreme outliers")
        
        return outliers
    
    def _print_dataset_summary(self, name: str, report: Dict):
        """
        Print quality summary for a dataset.
        """
        if 'error' in report:
            return
        
        print(f"\n{'='*70}")
        print(f"QUALITY SUMMARY: {name.upper()}")
        print(f"{'='*70}")
        
        scores = []
        
        # Completeness score
        if 'completeness' in report:
            completeness_score = report['completeness']['complete_rows_pct']
            scores.append(completeness_score)
            print(f"Completeness: {completeness_score:.1f}%")
        
        # Uniqueness score (100% - duplicate percentage)
        if 'duplicates' in report:
            uniqueness_score = 100 - report['duplicates']['duplicate_pct']
            scores.append(uniqueness_score)
            print(f"Uniqueness: {uniqueness_score:.1f}%")
        
        # Overall quality score
        if scores:
            overall = np.mean(scores)
            print(f"\nOverall Quality: {overall:.1f}%")
            
            if overall >= 95:
                print("Status: ✓ EXCELLENT")
            elif overall >= 85:
                print("Status: ✓ GOOD")
            elif overall >= 70:
                print("Status: ⚠ ACCEPTABLE")
            else:
                print("Status: ✗ NEEDS IMPROVEMENT")