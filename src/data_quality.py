"""
TALEA - Civic Digital Twin: Data Quality Check Module
File: src/data_quality.py
Author: Juraj Sýkora
Organization: Alma Mater Studiorum - Università di Bologna

Data validation and quality checks module before analysis
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TALEADataQualityChecker:
    """Validate data quality before analysis"""

    def __init__(self):
        self.quality_reports = {}
    
    def run_complete_quality_check(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run quality checks on all datasets.
        
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
            
            dataset_report = {}
            
            # 1. Completeness
            dataset_report['completeness'] = self._check_completeness_detailed(df, name)
            
            # 2. Temporal continuity (for time-series data)
            if any(col in df.columns for col in ['Data', 'datetime', 'data']):
                date_col = next(col for col in ['Data', 'datetime', 'data'] if col in df.columns)
                dataset_report['temporal'] = self.check_temporal_continuity(df, date_col)
            
            # 3. Outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                dataset_report['outliers'] = self.check_outliers(df, numeric_cols)
            
            # 4. Duplicates
            dataset_report['duplicates'] = self._check_duplicates(df, name)
            
            # 5. Data ranges
            dataset_report['ranges'] = self._check_data_ranges(df, name)
            
            quality_report[name] = dataset_report
            
            # Print summary
            self._print_dataset_quality_summary(name, dataset_report)
        
        self.quality_reports = quality_report
        return quality_report
    
    def _check_completeness_detailed(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Detailed completeness check"""
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
    
    def _check_duplicates(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check for duplicate records"""
        duplicates = {
            'total_duplicates': df.duplicated().sum(),
            'duplicate_pct': (df.duplicated().sum() / len(df)) * 100
        }
        
        print(f"\n2. Duplicates:")
        print(f"   Duplicate rows: {duplicates['total_duplicates']:,} ({duplicates['duplicate_pct']:.2f}%)")
        
        return duplicates
    
    def _check_data_ranges(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check if numeric data is within expected ranges"""
        ranges = {}
        
        print(f"\n3. Data Ranges:")
        
        # Dataset-specific range checks
        if dataset_name == 'bicycle':
            for col in ['Totale', 'Direzione centro', 'Direzione periferia']:
                if col in df.columns:
                    negative = (df[col] < 0).sum()
                    ranges[col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'negative_values': negative
                    }
                    if negative > 0:
                        print(f"   ⚠ {col}: {negative} negative values found")
                    else:
                        print(f"   ✓ {col}: range [{df[col].min():.0f}, {df[col].max():.0f}]")
        
        elif dataset_name == 'pedestrian':
            if 'Numero di visitatori' in df.columns:
                col = 'Numero di visitatori'
                negative = (df[col] < 0).sum()
                ranges[col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'negative_values': negative
                }
                print(f"   ✓ {col}: range [{df[col].min():.0f}, {df[col].max():.0f}]")
        
        # Check coordinates if present
        coord_checks = {
            'latitude': (-90, 90),
            'longitude': (-180, 180),
            'street_lat': (44.4, 44.6),  # Bologna specific
            'street_lon': (11.2, 11.5)   # Bologna specific
        }
        
        for col, (min_val, max_val) in coord_checks.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    print(f"   ⚠ {col}: {out_of_range} values out of range [{min_val}, {max_val}]")
                else:
                    print(f"   ✓ {col}: all values in valid range")
        
        return ranges
    
    def _print_dataset_quality_summary(self, name: str, report: Dict):
        """Print summary for a dataset"""
        print(f"\n{'='*70}")
        print(f"QUALITY SUMMARY: {name.upper()}")
        print(f"{'='*70}")
        
        # Overall quality score
        scores = []
        
        # Completeness score
        if 'completeness' in report:
            completeness_score = report['completeness']['complete_rows_pct']
            scores.append(completeness_score)
            print(f"Completeness Score: {completeness_score:.1f}%")
        
        # Duplicate score
        if 'duplicates' in report:
            duplicate_score = 100 - report['duplicates']['duplicate_pct']
            scores.append(duplicate_score)
            print(f"Uniqueness Score: {duplicate_score:.1f}%")
        
        # Overall score
        if scores:
            overall = np.mean(scores)
            print(f"\nOverall Quality Score: {overall:.1f}%")
            
            if overall >= 95:
                print("Status: ✓ EXCELLENT")
            elif overall >= 85:
                print("Status: ✓ GOOD")
            elif overall >= 70:
                print("Status: ⚠ ACCEPTABLE")
            else:
                print("Status: ✗ NEEDS IMPROVEMENT")

    def check_completeness(self, df, required_cols):
        """Check for required columns and coverage"""
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Check data coverage
        coverage = (1 - df.isnull().sum() / len(df)) * 100
        low_coverage = coverage[coverage < 80]
        if len(low_coverage) > 0:
            warnings.warn(f"Low coverage columns:\n{low_coverage}")
    
    def check_temporal_continuity(self, df: pd.DataFrame, date_col: str, freq='H'):
        """Identify gaps in time series"""
        print(f"\n4. Temporal Continuity ({freq}):")
        
        df_sorted = df.sort_values(date_col).copy()
        time_diffs = df_sorted[date_col].diff()
        
        # Expected difference
        freq_map = {'H': 1, 'D': 24, '30T': 0.5}
        expected_hours = freq_map.get(freq, 1)
        expected_diff = pd.Timedelta(hours=expected_hours)
        
        # Find gaps
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        
        continuity = {
            'total_gaps': len(gaps),
            'largest_gap': gaps.max() if len(gaps) > 0 else pd.Timedelta(0),
            'total_gap_time': gaps.sum() if len(gaps) > 0 else pd.Timedelta(0)
        }
        
        if len(gaps) > 0:
            print(f"   ⚠ Found {len(gaps)} temporal gaps")
            print(f"   Largest gap: {gaps.max()}")
            print(f"   Total gap time: {gaps.sum()}")
        else:
            print(f"   ✓ No significant gaps found")
        
        return continuity
    
    def check_outliers(self, df: pd.DataFrame, numeric_cols: List[str], z_threshold=4):
        """Flag extreme outliers using z-score method"""
        print(f"\n5. Outlier Detection (z-score > {z_threshold}):")
        
        outliers = {}
        for col in numeric_cols:
            if col in ['year', 'month', 'day', 'hour', 'latitude', 'longitude']:
                continue  # Skip these columns
            
            values = df[col].dropna()
            if len(values) > 0:
                z_scores = np.abs(stats.zscore(values))
                outlier_count = (z_scores > z_threshold).sum()
                outlier_pct = (outlier_count / len(values)) * 100
                
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': outlier_pct
                }
                
                if outlier_count > 0:
                    print(f"   ⚠ {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
                else:
                    print(f"   ✓ {col}: no extreme outliers")
        
        return outliers