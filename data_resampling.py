import pandas as pd
from typing import Dict, List, Optional, Union
import warnings
import numpy as np

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class FlexiblePointInTimeResampler:
    """
    Flexible point-in-time resampling system that can work with any base timeframe
    and create any combination of higher timeframes with configurable granularity.
    """

    def __init__(self, base_data: pd.DataFrame, base_timeframe: str = None):
        """
        Initialize with base timeframe data

        Args:
            base_data: DataFrame with OHLCV data, datetime index
            base_timeframe: Optional explicit base timeframe (e.g., '1min', '5min')
                          If None, will auto-detect from data
        """
        self.base_data = base_data.copy()
        self.base_timeframe = base_timeframe or self._detect_timeframe()
        self.resampled_cache = {}

        print(f"Initialized with base timeframe: {self.base_timeframe}")
        print(f"Data shape: {base_data.shape}")
        print(f"Date range: {base_data.index[0]} to {base_data.index[-1]}")

    def _detect_timeframe(self) -> str:
        """Detect the base timeframe from the data"""
        if len(self.base_data) < 2:
            return "unknown"

        time_diff = self.base_data.index[1] - self.base_data.index[0]
        total_seconds = time_diff.total_seconds()

        if total_seconds < 60:
            return f"{int(total_seconds)}s"

        minutes = total_seconds / 60
        if minutes < 60:
            if minutes == int(minutes):
                return f"{int(minutes)}min"
            else:
                return f"{minutes:.1f}min"

        hours = minutes / 60
        if hours < 24:
            if hours == int(hours):
                return f"{int(hours)}H"
            else:
                return f"{hours:.1f}H"

        days = hours / 24
        return f"{int(days)}D"

    def resample_ohlcv(self, target_timeframe: str, offset: str = None) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe with proper point-in-time alignment

        Args:
            target_timeframe: Target timeframe (e.g., '1min', '5min', '30min', '4H', '1D')
            offset: Optional offset for alignment
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

        # Handle additional columns intelligently
        for col in self.base_data.columns:
            if col not in ohlcv_agg:
                col_lower = col.lower()
                if any(x in col_lower for x in ['volume', 'vol', 'amount', 'turnover']):
                    ohlcv_agg[col] = 'sum'
                elif any(x in col_lower for x in ['count', 'trades', 'ticks']):
                    ohlcv_agg[col] = 'sum'
                elif col in ['t', 'timestamp', 'time']:
                    ohlcv_agg[col] = 'last'
                elif any(x in col_lower for x in ['price', 'close', 'open', 'high', 'low']):
                    ohlcv_agg[col] = 'last'
                else:
                    ohlcv_agg[col] = 'last'  # Default behavior

        # Create resampler
        resampler = self.base_data.resample(
            target_timeframe,
            offset=offset,
            closed='left',
            label='right'
        )

        resampled = resampler.agg(ohlcv_agg)
        self.resampled_cache[cache_key] = resampled.copy()
        return resampled

    def create_enhanced_dataset(self,
                                output_timeframe: str = None,
                                higher_timeframes: List[str] = None,
                                include_features: bool = True,
                                feature_types: List[str] = None) -> pd.DataFrame:
        """
        Create dataset with configurable output timeframe and higher timeframe context

        Args:
            output_timeframe: Timeframe for output rows (e.g., '1min', '5min', '30min')
                            If None, uses base timeframe
            higher_timeframes: List of higher timeframes to include as features (REQUIRED)
            include_features: Whether to add derived features
            feature_types: Types of features to include
                          ['basic', 'returns', 'distances', 'ranges', 'positions', 'momentum']

        Returns:
            Enhanced dataset with specified granularity + higher TF context
        """
        if output_timeframe is None:
            output_timeframe = self.base_timeframe

        if higher_timeframes is None:
            raise ValueError("higher_timeframes must be specified (e.g., ['30min', '4H', '1D'])")

        if feature_types is None:
            feature_types = ['basic', 'returns', 'distances', 'ranges']

        print(f"Creating dataset with:")
        print(f"  Output timeframe: {output_timeframe}")
        print(f"  Higher timeframes: {higher_timeframes}")
        print(f"  Feature types: {feature_types}")

        # Get base data at output timeframe
        if output_timeframe == self.base_timeframe:
            enhanced_data = self.base_data.copy()
        else:
            enhanced_data = self.resample_ohlcv(output_timeframe)

        print(f"  Base data shape: {enhanced_data.shape}")

        # Add higher timeframe context
        for higher_tf in higher_timeframes:
            print(f"  Processing {higher_tf}...")

            higher_data = self.resample_ohlcv(higher_tf)
            higher_features = self._create_higher_tf_features(
                enhanced_data.index,
                higher_data,
                higher_tf,
                feature_types
            )

            enhanced_data = enhanced_data.join(higher_features)

        if include_features and 'basic' not in feature_types:
            enhanced_data = self._add_derived_features(
                enhanced_data,
                higher_timeframes,
                feature_types
            )

        print(f"  Final dataset shape: {enhanced_data.shape}")
        return enhanced_data

    def _get_smart_higher_timeframes(self, output_timeframe: str) -> List[str]:
        """
        Generate smart default higher timeframes based on output timeframe
        """
        # Define timeframe hierarchy in minutes
        tf_minutes = {
            '1min': 1, '5min': 5, '15min': 15, '30min': 30,
            '1H': 60, '2H': 120, '4H': 240, '6H': 360, '8H': 480, '12H': 720,
            '1D': 1440, '3D': 4320, '1W': 10080
        }

        output_mins = tf_minutes.get(output_timeframe, 5)  # Default to 5 if unknown

        # Select higher timeframes that are meaningful multiples
        candidates = []
        for tf, mins in tf_minutes.items():
            if mins > output_mins:
                # Include timeframes that are reasonable multiples
                ratio = mins / output_mins
                if ratio >= 2 and (ratio <= 50 or tf in ['1D', '1W']):
                    candidates.append((tf, mins))

        # Sort by minutes and select up to 4 meaningful timeframes
        candidates.sort(key=lambda x: x[1])

        # Smart selection logic
        selected = []
        if output_mins <= 5:  # 1min, 5min
            selected = ['15min', '1H', '4H', '1D']
        elif output_mins <= 30:  # 15min, 30min
            selected = ['1H', '4H', '1D']
        elif output_mins <= 240:  # 1H, 2H, 4H
            selected = ['1D', '3D', '1W']
        else:  # 1D and above
            selected = ['3D', '1W']

        # Filter to only include available candidates
        available = [tf for tf, _ in candidates]
        selected = [tf for tf in selected if tf in available]

        return selected[:4]  # Limit to 4 higher timeframes

    def _create_higher_tf_features(self,
                                   base_index: pd.DatetimeIndex,
                                   higher_data: pd.DataFrame,
                                   timeframe: str,
                                   feature_types: List[str]) -> pd.DataFrame:
        """
        Create higher timeframe features for each base timeframe period
        """
        feature_columns = []

        # Basic OHLCV features
        if 'basic' in feature_types:
            basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            for feature in basic_features:
                if feature in higher_data.columns:
                    feature_columns.append(f"{timeframe}_{feature}")

        # Initialize feature dataframe
        features_df = pd.DataFrame(
            index=base_index,
            columns=feature_columns
        )

        # Populate features using point-in-time alignment
        for timestamp in base_index:
            available_periods = higher_data[higher_data.index < timestamp]

            if len(available_periods) > 0:
                last_complete = available_periods.iloc[-1]

                # Add basic features
                if 'basic' in feature_types:
                    for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if feature in higher_data.columns:
                            col_name = f"{timeframe}_{feature}"
                            if col_name in features_df.columns:
                                features_df.loc[timestamp, col_name] = last_complete[feature]

        return features_df

    def _add_derived_features(self,
                              data: pd.DataFrame,
                              timeframes: List[str],
                              feature_types: List[str]) -> pd.DataFrame:
        """
        Add derived features based on specified types
        """
        enhanced_data = data.copy()

        for tf in timeframes:
            tf_close = f"{tf}_Close"
            tf_open = f"{tf}_Open"
            tf_high = f"{tf}_High"
            tf_low = f"{tf}_Low"

            # Returns
            if 'returns' in feature_types and tf_close in enhanced_data.columns and tf_open in enhanced_data.columns:
                enhanced_data[f"{tf}_Return"] = (
                                                        enhanced_data[tf_close] / enhanced_data[tf_open] - 1
                                                ) * 100

            # Distance from higher TF levels
            if 'distances' in feature_types and tf_close in enhanced_data.columns and 'Close' in enhanced_data.columns:
                enhanced_data[f"Distance_from_{tf}"] = (
                                                               enhanced_data['Close'] / enhanced_data[tf_close] - 1
                                                       ) * 100

            # Ranges and volatility
            if 'ranges' in feature_types and tf_high in enhanced_data.columns and tf_low in enhanced_data.columns:
                enhanced_data[f"{tf}_Range_Pct"] = (
                                                           (enhanced_data[tf_high] - enhanced_data[tf_low]) /
                                                           enhanced_data[tf_close]
                                                   ) * 100

            # Position within range
            if 'positions' in feature_types and all(
                    col in enhanced_data.columns for col in [tf_high, tf_low, tf_close, 'Close']):
                range_size = enhanced_data[tf_high] - enhanced_data[tf_low]
                position = enhanced_data['Close'] - enhanced_data[tf_low]
                enhanced_data[f"Position_in_{tf}_Range"] = (position / range_size) * 100

            # Momentum features
            if 'momentum' in feature_types and tf_close in enhanced_data.columns and 'Close' in enhanced_data.columns:
                # Current vs higher TF momentum alignment
                enhanced_data[f"{tf}_Momentum_Align"] = np.where(
                    (enhanced_data['Close'] > enhanced_data['Close'].shift(1)) &
                    (enhanced_data[tf_close] > enhanced_data[tf_close].shift(1)), 1,
                    np.where(
                        (enhanced_data['Close'] < enhanced_data['Close'].shift(1)) &
                        (enhanced_data[tf_close] < enhanced_data[tf_close].shift(1)), -1, 0
                    )
                )

        return enhanced_data

    def get_timeframe_info(self) -> Dict:
        """
        Get basic information about the resampler
        """
        return {
            'base_timeframe': self.base_timeframe,
            'base_data_shape': self.base_data.shape,
            'date_range': (self.base_data.index[0], self.base_data.index[-1]),
            'available_feature_types': ['basic', 'returns', 'distances', 'ranges', 'positions', 'momentum']
        }

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        tf_map = {
            '1min': 1, '5min': 5, '15min': 15, '30min': 30,
            '1H': 60, '2H': 120, '4H': 240, '6H': 360, '8H': 480, '12H': 720,
            '1D': 1440, '3D': 4320, '1W': 10080
        }
        return tf_map.get(timeframe, 5)  # Default to 5 minutes

    def compare_configurations(self, configs: List[Dict]) -> None:
        """
        Compare different configuration setups

        Args:
            configs: List of configuration dictionaries with keys:
                    {'output_timeframe', 'higher_timeframes', 'name'}
        """
        print("=== CONFIGURATION COMPARISON ===\n")

        results = []
        for config in configs:
            name = config.get('name', 'Unnamed')
            output_tf = config.get('output_timeframe', self.base_timeframe)
            higher_tfs = config.get('higher_timeframes', self._get_smart_higher_timeframes(output_tf))

            dataset = self.create_enhanced_dataset(
                output_timeframe=output_tf,
                higher_timeframes=higher_tfs,
                include_features=False  # Skip features for comparison speed
            )

            results.append({
                'name': name,
                'output_timeframe': output_tf,
                'rows': len(dataset),
                'columns': len(dataset.columns),
                'higher_timeframes': higher_tfs
            })

        # Display comparison
        for result in results:
            print(f"{result['name']}:")
            print(f"  Output timeframe: {result['output_timeframe']}")
            print(f"  Higher timeframes: {result['higher_timeframes']}")
            print(f"  Dataset size: {result['rows']} rows × {result['columns']} columns")
            print(f"  Data points: {result['rows']:,}")
            print()


# Usage with your actual ETH data
def process_data(raw_data, frequency, frames):
    """
    Process your actual ETH 5-minute data with flexible configurations
    """
    try:
        # Load your ETH data
        raw = raw_data

        # Clean up the data
        raw['t'] = raw.time
        raw.time = pd.to_datetime(raw.time, unit='ms')
        raw.set_index('time', inplace=True)

        if 'Unnamed: 0' in raw.columns:
            raw.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

        print(f"Loaded data: {raw.shape}")

        # Initialize flexible resampler
        resampler = FlexiblePointInTimeResampler(raw, base_timeframe='5min')

        # Show basic info
        info = resampler.get_timeframe_info()
        print(f"Base timeframe: {info['base_timeframe']}")
        print(f"Data shape: {info['base_data_shape']}")

        # Create enhanced dataset - YOU specify exactly what you want
        enhanced_data = resampler.create_enhanced_dataset(
            output_timeframe=frequency,  # Change to whatever you want
            higher_timeframes=time_frames,  # YOU specify these
            include_features=True,
            feature_types=['basic', 'returns', 'distances', 'ranges']
        )

        print(f"\nEnhanced dataset: {enhanced_data.shape}")
        print(f"Sample data (last 3 rows):")
        print(enhanced_data[['Close', '30min_Close', '4H_Close', '1D_Close']].tail(3))

        return resampler, enhanced_data

    except FileNotFoundError:
        print("Error: Could not find data file at specified path")
        print("Please check the file path")
        return None, None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None


raw_data = pd.read_csv('D:/Seagull_data/historical_data/time/ETHEUR/ETHEUR_5m.csv')
freq = '5min'
time_frames = ['30min', '4H', '1D']
# Run with your ETH data
resampler, enhanced_data = process_data(raw_data, freq, time_frames)
print(resampler)
print(enhanced_data)
enhanced_data.to_csv('resampled5mEE.csv', index=False)

