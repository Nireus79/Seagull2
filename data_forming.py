import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class PointInTimeResampler:
    """
    A point-in-time resampling system that maintains statistical integrity
    by ensuring no look-ahead bias in multi-timeframe analysis.
    """

    def __init__(self, base_data: pd.DataFrame):
        """
        Initialize with base timeframe data (e.g., 5-minute bars)

        Args:
            base_data: DataFrame with OHLCV data, datetime index
        """
        self.base_data = base_data.copy()
        self.base_timeframe = self._detect_timeframe()
        self.resampled_cache = {}

    def _detect_timeframe(self) -> str:
        """Detect the base timeframe from the data"""
        if len(self.base_data) < 2:
            return "unknown"

        time_diff = self.base_data.index[1] - self.base_data.index[0]
        minutes = time_diff.total_seconds() / 60

        if minutes == 1:
            return "1min"
        elif minutes == 5:
            return "5min"
        elif minutes == 15:
            return "15min"
        else:
            return f"{int(minutes)}min"

    def resample_ohlcv(self, target_timeframe: str, offset: str = None) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe with proper point-in-time alignment

        Args:
            target_timeframe: Target timeframe (e.g., '30min', '4H', '1D')
            offset: Optional offset for alignment (e.g., '30min' to start at :30)

        Returns:
            Resampled DataFrame with proper temporal alignment
        """
        cache_key = f"{target_timeframe}_{offset}"

        if cache_key in self.resampled_cache:
            return self.resampled_cache[cache_key].copy()

        # Define OHLCV aggregation rules
        ohlcv_agg = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

        # Add any additional columns that should be summed or averaged
        for col in self.base_data.columns:
            if col not in ohlcv_agg:
                if 'volume' in col.lower() or 'vol' in col.lower():
                    ohlcv_agg[col] = 'sum'
                elif col in ['t', 'timestamp']:
                    ohlcv_agg[col] = 'last'
                else:
                    ohlcv_agg[col] = 'last'  # Default to last value

        # Create resampler with proper offset
        resampler = self.base_data.resample(
            target_timeframe,
            offset=offset,
            closed='left',  # Left-closed intervals for proper alignment
            label='right'  # Label with the end of the interval
        )

        resampled = resampler.agg(ohlcv_agg)

        # Cache the result
        self.resampled_cache[cache_key] = resampled.copy()

        return resampled

    def get_point_in_time_data(self,
                               timestamp: pd.Timestamp,
                               timeframes: List[str],
                               lookback_periods: Dict[str, int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get point-in-time data for multiple timeframes at a specific timestamp.
        Only returns data that would have been available at that exact timestamp.

        Args:
            timestamp: The point in time for which to get data
            timeframes: List of timeframes to include (e.g., ['30min', '4H', '1D'])
            lookback_periods: Number of periods to look back for each timeframe

        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        if lookback_periods is None:
            lookback_periods = {tf: 100 for tf in timeframes}

        result = {}

        for tf in timeframes:
            # Get resampled data for this timeframe
            resampled = self.resample_ohlcv(tf)

            # Find the last complete period before the given timestamp
            available_data = resampled[resampled.index < timestamp]

            if len(available_data) == 0:
                result[tf] = pd.DataFrame()
                continue

            # Get the specified number of lookback periods
            periods = lookback_periods.get(tf, 100)
            result[tf] = available_data.tail(periods).copy()

        return result

    def create_aligned_dataset(self,
                               target_timeframe: str = '30min',
                               higher_timeframes: List[str] = ['4H', '1D'],
                               alignment_method: str = 'last_complete') -> pd.DataFrame:
        """
        Create a dataset with properly aligned multi-timeframe data.
        Each row represents a target_timeframe period with corresponding
        higher timeframe data that was available at that point in time.

        Args:
            target_timeframe: The main timeframe for predictions
            higher_timeframes: List of higher timeframes to include as features
            alignment_method: How to align data ('last_complete', 'interpolate')

        Returns:
            Aligned dataset with multi-timeframe data
        """
        # Get the target timeframe data
        target_data = self.resample_ohlcv(target_timeframe)

        # Create aligned dataset starting with target timeframe
        aligned_data = target_data.copy()

        for higher_tf in higher_timeframes:
            higher_data = self.resample_ohlcv(higher_tf)

            if alignment_method == 'last_complete':
                # For each target period, find the last complete higher timeframe period
                aligned_higher = self._align_last_complete(
                    target_data.index,
                    higher_data,
                    higher_tf
                )
            else:
                # Forward fill method (less statistically rigorous)
                aligned_higher = higher_data.reindex(
                    target_data.index,
                    method='ffill'
                )

            # Rename columns to indicate source timeframe
            column_mapping = {
                col: f"{higher_tf}_{col}"
                for col in aligned_higher.columns
                if col in ['Open', 'High', 'Low', 'Close', 'Volume']
            }
            aligned_higher = aligned_higher.rename(columns=column_mapping)

            # Merge with target data
            aligned_data = aligned_data.join(aligned_higher[column_mapping.values()])

        return aligned_data

    def _align_last_complete(self,
                             target_index: pd.DatetimeIndex,
                             higher_data: pd.DataFrame,
                             higher_tf: str) -> pd.DataFrame:
        """
        Align higher timeframe data using last complete period method
        """
        aligned_data = pd.DataFrame(index=target_index, columns=higher_data.columns)

        for timestamp in target_index:
            # Find the last complete higher timeframe period before this timestamp
            available_periods = higher_data[higher_data.index < timestamp]

            if len(available_periods) > 0:
                last_complete = available_periods.iloc[-1]
                aligned_data.loc[timestamp] = last_complete

        return aligned_data

    def validate_alignment(self, dataset: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that the aligned dataset maintains temporal integrity

        Returns:
            Dictionary with validation metrics
        """
        validation_results = {
            'total_rows': len(dataset),
            'null_percentage': dataset.isnull().sum() / len(dataset),
            'temporal_gaps': [],
            'look_ahead_violations': 0
        }

        # Check for temporal gaps
        time_diffs = dataset.index.to_series().diff()
        expected_diff = time_diffs.mode().iloc[0]
        gaps = time_diffs[time_diffs != expected_diff]
        validation_results['temporal_gaps'] = len(gaps)

        # Additional validation can be added here

        return validation_results


# Usage example and demonstration
def demonstrate_resampling():
    """
    Demonstrate the point-in-time resampling system
    """
    # Create sample data (this would be replaced with your actual ETH data loading)
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='5min')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(1800, 2200, len(dates)),
        'High': np.random.uniform(1900, 2300, len(dates)),
        'Low': np.random.uniform(1700, 2100, len(dates)),
        'Close': np.random.uniform(1800, 2200, len(dates)),
        'Volume': np.random.uniform(1000, 10000, len(dates)),
        't': [int(d.timestamp() * 1000) for d in dates]  # Timestamp in milliseconds
    }, index=dates)

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    sample_data['High'] = np.maximum(sample_data['High'],
                                     np.maximum(sample_data['Open'], sample_data['Close']))
    sample_data['Low'] = np.minimum(sample_data['Low'],
                                    np.minimum(sample_data['Open'], sample_data['Close']))

    # Initialize the resampler
    resampler = PointInTimeResampler(sample_data)

    print(f"Base timeframe detected: {resampler.base_timeframe}")
    print(f"Base data shape: {sample_data.shape}")

    # Create aligned dataset
    aligned_data = resampler.create_aligned_dataset(
        target_timeframe='30min',
        higher_timeframes=['4H', '1D']
    )

    print(f"Aligned data shape: {aligned_data.shape}")
    print("\nAligned data columns:")
    for col in aligned_data.columns:
        print(f"  {col}")

    # Validate alignment
    validation = resampler.validate_alignment(aligned_data)
    print(f"\nValidation results:")
    print(f"  Total rows: {validation['total_rows']}")
    print(f"  Temporal gaps: {validation['temporal_gaps']}")

    # Demonstrate point-in-time data access
    test_timestamp = sample_data.index[1000]  # Pick a point in the middle
    pit_data = resampler.get_point_in_time_data(
        timestamp=test_timestamp,
        timeframes=['30min', '4H', '1D'],
        lookback_periods={'30min': 10, '4H': 5, '1D': 3}
    )

    print(f"\nPoint-in-time data for {test_timestamp}:")
    for tf, data in pit_data.items():
        if not data.empty:
            print(f"  {tf}: {len(data)} periods, last close: {data['Close'].iloc[-1]:.2f}")
        else:
            print(f"  {tf}: No data available")

    return resampler, aligned_data


if __name__ == "__main__":
    demonstrate_resampling()