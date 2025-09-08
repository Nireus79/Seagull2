import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
import ta
from itertools import combinations
# from data_resampling import resampled

warnings.filterwarnings('ignore')

resampled = pd.read_csv('resampled5mEE.csv')[:10000]
# pd.set_option('display.max_columns', None)


class FlexibleTechnicalIndicators:
    """
    Modified flexible technical indicators system implementing Elder's methodology:
    - Base indicators calculated on 30min actual data (sampled every 30min)
    - Results presented at 5min intervals (forward-filled)
    - Includes CUSUM and Bollinger crossing events
    - Enhanced with comprehensive co-event detection
    """

    def __init__(self, resampled_data: pd.DataFrame, config: Dict = None, low_timeframe: str = "30min"):
        """
        Initialize with resampled multi-timeframe data

        Args:
            resampled_data: DataFrame from FlexiblePointInTimeResampler with higher TF columns
            config: Configuration for indicators and events
            low_timeframe: The timeframe to use for "base" indicators (e.g., "30min")
        """
        self.data = resampled_data.copy()
        self.data['t'] = pd.to_datetime(self.data['t'], unit='ms')
        self.data = self.data.set_index('t')
        self.config = config or self._default_config()
        self.low_timeframe = low_timeframe  # This is what Elder calls the "intermediate" timeframe

        # Detect available timeframes
        self.base_timeframe = self._detect_base_timeframe()  # Actual index frequency (5min)
        self.higher_timeframes = self._detect_higher_timeframes()

        print(f"Detected base timeframe (index): {self.base_timeframe}")
        print(f"Low timeframe for indicators: {self.low_timeframe}")
        print(f"Available higher timeframes: {self.higher_timeframes}")

        # Validate required columns
        self._validate_data()

    def _default_config(self) -> Dict:
        """Default configuration for indicators and events"""
        config = {
            # Standard indicator periods (applied to low_timeframe, e.g., 30min)
            'ema_periods': {'fast': 12, 'medium': 26, 'slow': 50},
            'sma_periods': {'short': 20, 'medium': 50, 'long': 200},
            'rsi_period': 14,
            'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,

            # Multi-timeframe periods (applied to actual TF data)
            'daily_periods': {'short': 5, 'medium': 20, 'long': 50},
            'hourly_periods': {'short': 6, 'medium': 24, 'long': 72},
            'intraday_periods': {'short': 12, 'medium': 26, 'long': 50},

            # CUSUM parameters
            'cusum_config': {
                'threshold': 2.0,  # Threshold for CUSUM detection
                'drift': 0.5,  # Expected drift
                'reset_period': 100  # Period to reset CUSUM if no events
            },

            # Bollinger Band crossing parameters
            'bb_crossing_config': {
                'lookback_periods': 3,  # How many periods to confirm crossing
                'min_distance_pct': 1.0  # Minimum distance from band for valid crossing
            },

            # Normalization settings
            'volatility_lookback': 50,
            'correlation_window': 20,

            # Event detection settings
            'event_config': {
                'vpd_price_lookback': 20,
                'vpd_volume_lookback': 20,
                'vpd_divergence_threshold': 0.7,
                'vol_breakout_threshold': 2.0,
                'return_lookback': 100,
                'outlier_threshold': 2.5,
                'momentum_regime_threshold': 0.3,
            }
        }

        # Add co-event configuration
        config.update(self._default_co_event_config())
        return config

    def _default_co_event_config(self) -> Dict:
        """Default configuration for co-event detection"""
        return {
            'co_event_config': {
                'enable_2way': True,
                'enable_3way': True,
                'enable_combined': True,
                'time_window': 1,  # Events within N periods (0 = exact same time, >0 = within window)
                'min_strength_threshold': 0.0,  # Minimum individual event strength (if available)
                'create_named_combinations': True,  # Create specific named combinations
                'create_generic_combinations': True,  # Create generic "any_2_events" style columns
                'max_combinations': 50,  # Limit number of combinations to prevent explosion

                # Specific meaningful combinations (financial logic)
                'priority_2way_combinations': [
                    ('CUSUM_event', 'BB_upper_cross'),
                    ('CUSUM_event', 'BB_lower_cross'),
                    ('vpd_volatility_event', 'momentum_regime_event'),
                    ('outlier_event', 'BB_expansion'),
                    ('momentum_regime_event', 'BB_squeeze'),
                    ('vpd_volatility_event', 'outlier_event'),
                    ('BB_upper_cross', 'momentum_regime_event'),
                    ('BB_lower_cross', 'momentum_regime_event'),
                ],

                'priority_3way_combinations': [
                    ('CUSUM_event', 'vpd_volatility_event', 'momentum_regime_event'),
                    ('BB_upper_cross', 'outlier_event', 'momentum_regime_event'),
                    ('BB_lower_cross', 'outlier_event', 'momentum_regime_event'),
                    ('vpd_volatility_event', 'outlier_event', 'momentum_regime_event'),
                    ('CUSUM_event', 'BB_expansion', 'momentum_regime_event'),
                ]
            }
        }

    def _detect_base_timeframe(self) -> str:
        """Detect base timeframe from index frequency"""
        if len(self.data) < 2:
            return "unknown"

        try:
            if not isinstance(self.data.index, pd.DatetimeIndex):
                print("Warning: Index is not DatetimeIndex, attempting conversion...")
                self.data.index = pd.to_datetime(self.data.index)

            time_diff = self.data.index[1] - self.data.index[0]

            if hasattr(time_diff, 'total_seconds'):
                total_seconds = time_diff.total_seconds()
            elif isinstance(time_diff, (int, float)):
                total_seconds = time_diff / 1e9
            else:
                print(f"Warning: Unexpected time_diff type: {type(time_diff)}")
                return "unknown"

            if total_seconds < 60:
                return f"{int(total_seconds)}s"

            minutes = total_seconds / 60
            if minutes < 60:
                return f"{int(minutes)}min" if minutes == int(minutes) else f"{minutes:.1f}min"

            hours = minutes / 60
            if hours < 24:
                return f"{int(hours)}H" if hours == int(hours) else f"{hours:.1f}H"

            days = hours / 24
            return f"{int(days)}D"

        except Exception as e:
            print(f"Error detecting timeframe: {e}")
            return "unknown"

    def _detect_higher_timeframes(self) -> List[str]:
        """Detect available higher timeframes from column names"""
        timeframes = []
        tf_patterns = ['30min', '1H', '2H', '4H', '6H', '8H', '12H', '1D', '3D', '1W']

        for tf in tf_patterns:
            if f"{tf}_Close" in self.data.columns:
                timeframes.append(tf)

        return sorted(timeframes, key=lambda x: self._timeframe_to_minutes(x))

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        tf_map = {
            '1min': 1, '5min': 5, '15min': 15, '30min': 30,
            '1H': 60, '2H': 120, '4H': 240, '6H': 360, '8H': 480, '12H': 720,
            '1D': 1440, '3D': 4320, '1W': 10080
        }
        return tf_map.get(timeframe, 5)

    def _validate_data(self):
        """Validate required columns are present"""
        # Check for base timeframe columns
        required_base = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_base if col not in self.data.columns]

        # Check for low timeframe columns (e.g., 30min)
        required_low_tf = [f"{self.low_timeframe}_{col}" for col in required_base]
        missing_low_tf = [col for col in required_low_tf if col not in self.data.columns]

        if missing and missing_low_tf:
            raise ValueError(
                f"Missing required columns. Need either base columns {missing} or low TF columns {missing_low_tf}")

    def _extract_low_timeframe_data(self) -> pd.DataFrame:
        """
        Extract actual low timeframe data (e.g., 30min) for indicator calculations
        This implements Elder's methodology by using actual 30min bars
        """
        # Extract low timeframe columns
        low_tf_cols = [col for col in self.data.columns if col.startswith(f"{self.low_timeframe}_")]
        low_tf_data = self.data[low_tf_cols].copy()

        # Get unique periods (where the low timeframe close actually changes)
        close_col = f"{self.low_timeframe}_Close"
        if close_col in low_tf_data.columns:
            # Keep only rows where the low TF close price changes (actual new bars)
            mask = (low_tf_data[close_col] != low_tf_data[close_col].shift(1)) | (
                    low_tf_data.index == low_tf_data.index[0])
            unique_periods = low_tf_data[mask].copy()
        else:
            unique_periods = low_tf_data.copy()

        # Rename columns for ta library compatibility
        column_mapping = {
            f"{self.low_timeframe}_Open": "Open",
            f"{self.low_timeframe}_High": "High",
            f"{self.low_timeframe}_Low": "Low",
            f"{self.low_timeframe}_Close": "Close",
            f"{self.low_timeframe}_Volume": "Volume"
        }

        available_mapping = {k: v for k, v in column_mapping.items() if k in unique_periods.columns}
        unique_periods = unique_periods.rename(columns=available_mapping)

        print(f"Extracted {len(unique_periods)} unique {self.low_timeframe} periods for indicator calculation")
        return unique_periods

    def add_base_indicators(self) -> pd.DataFrame:
        """
        Add standard technical indicators calculated on low timeframe data (Elder's methodology)
        Indicators are calculated every 30min but presented every 5min via forward-fill
        """
        data = self.data.copy()

        # Extract actual low timeframe data for calculations
        low_tf_data = self._extract_low_timeframe_data()

        if len(low_tf_data) < 20:
            print(f"Warning: Insufficient {self.low_timeframe} data for indicators")
            return data

        # Calculate indicators on low timeframe data using ta library
        indicators = self._calculate_low_tf_indicators(low_tf_data)

        # Forward-fill indicators to base timeframe (5min) resolution
        aligned_indicators = self._align_indicators_to_base(indicators, data.index)

        # Add to main dataset
        for col in aligned_indicators.columns:
            data[col] = aligned_indicators[col]

        return data

    def _calculate_low_tf_indicators(self, low_tf_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators on actual low timeframe data"""
        if 'Close' not in low_tf_data.columns or len(low_tf_data) < 10:
            return pd.DataFrame(index=low_tf_data.index)

        indicators = pd.DataFrame(index=low_tf_data.index)
        high, low, close, volume = (low_tf_data.get(col, pd.Series())
                                    for col in ['High', 'Low', 'Close', 'Volume'])

        # Use ta library for all standard indicators on actual low TF data
        # EMAs
        if 'ema_periods' in self.config:
            for name, period in self.config['ema_periods'].items():
                if len(close) >= period:
                    indicators[f'EMA_{name}_{period}'] = ta.trend.EMAIndicator(close, window=period).ema_indicator()

        # SMAs
        if 'sma_periods' in self.config:
            for name, period in self.config['sma_periods'].items():
                if len(close) >= period:
                    indicators[f'SMA_{name}_{period}'] = ta.trend.SMAIndicator(close, window=period).sma_indicator()

        # MACD
        if 'macd_params' in self.config and len(close) >= 26:
            macd_params = self.config['macd_params']
            macd = ta.trend.MACD(
                close,
                window_slow=macd_params.get('slow', 26),
                window_fast=macd_params.get('fast', 12),
                window_sign=macd_params.get('signal', 9)
            )
            indicators['MACD'] = macd.macd()
            indicators['MACD_signal'] = macd.macd_signal()
            indicators['MACD_histogram'] = macd.macd_diff()

        # RSI
        rsi_period = self.config.get('rsi_period', 14)
        if len(close) >= rsi_period:
            indicators['RSI'] = ta.momentum.RSIIndicator(close, window=rsi_period).rsi()

        # Bollinger Bands
        bb_period = self.config.get('bb_period', 20)
        bb_std = self.config.get('bb_std', 2)
        if len(close) >= bb_period:
            bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
            indicators['BB_upper'] = bb.bollinger_hband()
            indicators['BB_middle'] = bb.bollinger_mavg()
            indicators['BB_lower'] = bb.bollinger_lband()

        # ATR
        atr_period = self.config.get('atr_period', 14)
        if len(close) >= atr_period and len(high) >= atr_period and len(low) >= atr_period:
            indicators['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()

        # Volume indicators (if volume data available)
        if len(volume) > 0 and not volume.isna().all():
            indicators['Volume_SMA'] = ta.trend.SMAIndicator(volume, window=min(20, len(volume) // 2)).sma_indicator()
            volume_sma = indicators['Volume_SMA']
            indicators['Volume_ratio'] = volume / volume_sma.replace(0, np.nan)

        return indicators

    def add_cusum_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add CUSUM (Cumulative Sum) change detection on low timeframe data
        """
        # Extract low timeframe returns for CUSUM calculation
        low_tf_close_col = f"{self.low_timeframe}_Close"
        if low_tf_close_col not in data.columns:
            print(f"Warning: {low_tf_close_col} not found, using base Close for CUSUM")
            price_series = data['Close']
        else:
            price_series = data[low_tf_close_col]

        # Calculate returns on low timeframe data
        returns = price_series.pct_change().fillna(0)

        # CUSUM parameters
        cusum_config = self.config.get('cusum_config', {})
        threshold = cusum_config.get('threshold', 2.0)
        drift = cusum_config.get('drift', 0.5)
        reset_period = cusum_config.get('reset_period', 100)

        # Initialize CUSUM arrays
        cusum_pos = np.zeros(len(returns))
        cusum_neg = np.zeros(len(returns))
        cusum_events = np.zeros(len(returns), dtype=bool)
        cusum_direction = np.zeros(len(returns))  # 1 for positive, -1 for negative

        # Calculate CUSUM
        for i in range(1, len(returns)):
            # Positive CUSUM (detect upward changes)
            cusum_pos[i] = max(0, cusum_pos[i - 1] + returns.iloc[i] - drift / 100)

            # Negative CUSUM (detect downward changes)
            cusum_neg[i] = min(0, cusum_neg[i - 1] + returns.iloc[i] + drift / 100)

            # Check for events
            if cusum_pos[i] > threshold / 100:
                cusum_events[i] = True
                cusum_direction[i] = 1
                cusum_pos[i] = 0  # Reset after detection

            elif cusum_neg[i] < -threshold / 100:
                cusum_events[i] = True
                cusum_direction[i] = -1
                cusum_neg[i] = 0  # Reset after detection

            # Periodic reset to avoid drift
            if i % reset_period == 0:
                cusum_pos[i] = 0
                cusum_neg[i] = 0

        # Add CUSUM features to dataset
        data['CUSUM_pos'] = cusum_pos
        data['CUSUM_neg'] = cusum_neg
        data['CUSUM_event'] = cusum_events
        data['CUSUM_direction'] = cusum_direction  # 1=up, -1=down, 0=none

        return data

    def add_bollinger_crossing_events(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Band crossing events detection
        """
        if not all(col in data.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
            print("Warning: Bollinger Bands not found, skipping crossing events")
            return data

        bb_config = self.config.get('bb_crossing_config', {})
        lookback = bb_config.get('lookback_periods', 3)
        min_distance_pct = bb_config.get('min_distance_pct', 1.0)

        # Use the current close price for crossing detection (real-time)
        close_price = data[f"{self.low_timeframe}_Close"] if f"{self.low_timeframe}_Close" in data.columns else data[
            'Close']

        # Calculate distances from bands
        upper_distance = (close_price - data['BB_upper']) / close_price * 100
        lower_distance = (data['BB_lower'] - close_price) / close_price * 100
        middle_distance = close_price - data['BB_middle']

        # Initialize crossing event columns
        data['BB_upper_cross'] = False
        data['BB_lower_cross'] = False
        data['BB_middle_cross_up'] = False
        data['BB_middle_cross_down'] = False
        data['BB_squeeze'] = False
        data['BB_expansion'] = False

        # Upper band crossing (price breaks above upper band)
        for i in range(lookback, len(data)):
            # Check if price was below upper band and now above
            was_below = all(data['BB_upper'].iloc[i - j] > close_price.iloc[i - j] for j in range(1, lookback + 1))
            is_above = close_price.iloc[i] > data['BB_upper'].iloc[i]
            sufficient_distance = upper_distance.iloc[i] > min_distance_pct

            if was_below and is_above and sufficient_distance:
                data.loc[data.index[i], 'BB_upper_cross'] = True

        # Lower band crossing (price breaks below lower band)
        for i in range(lookback, len(data)):
            was_above = all(data['BB_lower'].iloc[i - j] < close_price.iloc[i - j] for j in range(1, lookback + 1))
            is_below = close_price.iloc[i] < data['BB_lower'].iloc[i]
            sufficient_distance = lower_distance.iloc[i] > min_distance_pct

            if was_above and is_below and sufficient_distance:
                data.loc[data.index[i], 'BB_lower_cross'] = True

        # Middle line crossings
        for i in range(1, len(data)):
            # Upward cross of middle line
            if (middle_distance.iloc[i - 1] <= 0 and middle_distance.iloc[i] > 0):
                data.loc[data.index[i], 'BB_middle_cross_up'] = True

            # Downward cross of middle line
            elif (middle_distance.iloc[i - 1] >= 0 and middle_distance.iloc[i] < 0):
                data.loc[data.index[i], 'BB_middle_cross_down'] = True

        # Bollinger Band squeeze and expansion detection
        bb_width = (data['BB_upper'] - data['BB_lower']) / data['BB_middle'] * 100
        bb_width_ma = bb_width.rolling(20).mean()
        bb_width_std = bb_width.rolling(20).std()

        # Squeeze: width below moving average minus 1 std
        data['BB_squeeze'] = bb_width < (bb_width_ma - bb_width_std)

        # Expansion: width above moving average plus 1 std
        data['BB_expansion'] = bb_width > (bb_width_ma + bb_width_std)

        # Combined BB events
        data['BB_any_cross'] = (data['BB_upper_cross'] | data['BB_lower_cross'] |
                                data['BB_middle_cross_up'] | data['BB_middle_cross_down'])

        return data

    def add_normalized_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add López de Prado style normalized/relative indicators"""

        # Use low timeframe data for returns calculation
        close_col = f"{self.low_timeframe}_Close" if f"{self.low_timeframe}_Close" in data.columns else 'Close'

        # Base volatility measures
        data['returns'] = data[close_col].pct_change()
        data['vol_realized'] = data['returns'].rolling(20).std() * np.sqrt(
            24 * 60 / self._timeframe_to_minutes(self.low_timeframe))
        data['ATR_pct'] = data['ATR'] / data[close_col] * 100

        # Normalized trend indicators (distances from moving averages)
        if 'ema_periods' in self.config:
            for name, period in self.config['ema_periods'].items():
                ema_col = f'EMA_{name}_{period}'
                if ema_col in data.columns:
                    # Distance normalized by ATR (López de Prado style)
                    data[f'EMA_{name}_distance_atr'] = (data[close_col] - data[ema_col]) / data['ATR']
                    # Percentage distance
                    data[f'EMA_{name}_distance_pct'] = (data[close_col] / data[ema_col] - 1) * 100

        # Volatility-normalized MACD (more stable)
        if 'MACD' in data.columns and 'ATR' in data.columns:
            data['MACD_normalized'] = data['MACD'] / data['ATR']
            data['MACD_efficiency'] = data['MACD'] / data['vol_realized'].replace(0, np.nan)

        # RSI momentum (Elder style - RSI direction * momentum alignment)
        if 'RSI' in data.columns:
            momentum_direction = np.sign(data[close_col].pct_change(5))
            data['RSI_momentum'] = (data['RSI'] - 50) * momentum_direction

        # Bollinger Bands position and efficiency
        if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
            bb_range = data['BB_upper'] - data['BB_lower']
            data['BB_position'] = ((data[close_col] - data['BB_lower']) / bb_range * 100).fillna(50)
            data['BB_width_pct'] = (bb_range / data[close_col] * 100).fillna(0)

        return data

    def add_higher_timeframe_context(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add higher timeframe indicators using ACTUAL timeframe data"""

        for tf in self.higher_timeframes:
            if tf == self.low_timeframe:  # Skip if it's the same as our base indicator timeframe
                continue

            tf_close_col = f"{tf}_Close"

            if tf_close_col not in data.columns:
                continue

            print(f"Processing {tf} timeframe context...")

            # Extract actual timeframe data
            tf_data = self._extract_actual_timeframe_data(data, tf)

            if len(tf_data) < 20:
                print(f"Insufficient {tf} data ({len(tf_data)} periods), skipping...")
                continue

            # Get appropriate periods for this timeframe
            periods = self._get_timeframe_periods(tf)

            # Calculate indicators on actual TF data using ta library
            tf_indicators = self._calculate_timeframe_indicators(tf_data, periods, tf)

            # Align back to base timeframe
            aligned_indicators = self._align_indicators_to_base(tf_indicators, data.index)

            # Merge indicators
            for col in aligned_indicators.columns:
                data[col] = aligned_indicators[col]

            # Add cross-timeframe relationships
            data = self._add_cross_timeframe_features(data, tf)

        return data

    def _extract_actual_timeframe_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Extract unique periods from higher timeframe data"""
        tf_cols = [col for col in data.columns if col.startswith(f"{timeframe}_")]
        tf_data = data[tf_cols].dropna()

        # Remove consecutive duplicates to get actual TF periods
        close_col = f"{timeframe}_Close"
        if close_col in tf_data.columns:
            mask = (tf_data[close_col] != tf_data[close_col].shift(1)) | (tf_data.index == tf_data.index[0])
            tf_data = tf_data[mask].copy()

        # Rename columns for ta library compatibility
        column_mapping = {
            f"{timeframe}_Open": "Open",
            f"{timeframe}_High": "High",
            f"{timeframe}_Low": "Low",
            f"{timeframe}_Close": "Close",
            f"{timeframe}_Volume": "Volume"
        }

        available_mapping = {k: v for k, v in column_mapping.items() if k in tf_data.columns}
        tf_data = tf_data.rename(columns=available_mapping)

        return tf_data

    def _get_timeframe_periods(self, timeframe: str) -> Dict:
        """Get appropriate periods based on timeframe type"""
        tf_minutes = self._timeframe_to_minutes(timeframe)

        if tf_minutes >= 1440:  # Daily or higher
            return self.config['daily_periods']
        elif tf_minutes >= 60:  # Hourly
            return self.config['hourly_periods']
        else:
            return self.config['intraday_periods']

    def _calculate_timeframe_indicators(self, tf_data: pd.DataFrame, periods: Dict, tf: str) -> pd.DataFrame:
        """Calculate indicators on actual timeframe data using ta library"""

        if 'Close' not in tf_data.columns or len(tf_data) < 10:
            return pd.DataFrame(index=tf_data.index)

        indicators = pd.DataFrame(index=tf_data.index)
        close = tf_data['Close']

        # EMAs using ta library
        for name, period in periods.items():
            if len(close) >= period:
                indicators[f'{tf}_EMA_{name}'] = ta.trend.EMAIndicator(close, window=period).ema_indicator()

        # RSI
        if len(close) >= 14:
            indicators[f'{tf}_RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

        # Returns and momentum
        indicators[f'{tf}_returns'] = close.pct_change()

        if 'short' in periods and len(close) >= periods['short']:
            indicators[f'{tf}_momentum_short'] = close.pct_change(periods['short'])
        if 'medium' in periods and len(close) >= periods['medium']:
            indicators[f'{tf}_momentum_medium'] = close.pct_change(periods['medium'])

        # Volatility
        indicators[f'{tf}_volatility'] = indicators[f'{tf}_returns'].rolling(min(20, len(close) // 2)).std()

        # Volume indicators if available
        if 'Volume' in tf_data.columns and not tf_data['Volume'].isna().all():
            vol_window = min(20, len(tf_data) // 2)
            indicators[f'{tf}_volume_sma'] = ta.trend.SMAIndicator(
                tf_data['Volume'], window=vol_window
            ).sma_indicator()

        return indicators

    def _align_indicators_to_base(self, tf_indicators: pd.DataFrame, base_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Align higher TF indicators to base timeframe using forward fill"""
        aligned = pd.DataFrame(index=base_index)

        for col in tf_indicators.columns:
            aligned[col] = tf_indicators[col].reindex(base_index, method='ffill')

        return aligned

    def _add_cross_timeframe_features(self, data: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Add Elder-style cross-timeframe relationship features"""

        tf_close = f"{tf}_Close"
        if tf_close not in data.columns:
            return data

        # Use low timeframe close for comparison
        close_col = f"{self.low_timeframe}_Close" if f"{self.low_timeframe}_Close" in data.columns else 'Close'

        # Price position relative to higher TF
        data[f'price_vs_{tf}_pct'] = (data[close_col] / data[tf_close] - 1) * 100

        # Momentum alignment (Elder's triple screen concept)
        base_momentum = data[close_col].pct_change(5)

        if f'{tf}_momentum_short' in data.columns:
            tf_momentum = data[f'{tf}_momentum_short']
            # Momentum alignment: +1 same direction, -1 opposite, 0 mixed
            data[f'momentum_align_{tf}'] = np.sign(base_momentum) * np.sign(tf_momentum)

            # Momentum strength ratio
            base_momentum_abs = abs(base_momentum)
            tf_momentum_abs = abs(tf_momentum)
            data[f'momentum_ratio_{tf}'] = (base_momentum_abs / tf_momentum_abs.replace(0, np.nan)).fillna(1)

        return data

    def add_event_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add specific features needed for event detection"""

        # Use low timeframe data for event detection
        close_col = f"{self.low_timeframe}_Close" if f"{self.low_timeframe}_Close" in data.columns else 'Close'
        volume_col = f"{self.low_timeframe}_Volume" if f"{self.low_timeframe}_Volume" in data.columns else 'Volume'

        config = self.config.get('event_config', {})

        # Price momentum normalized by volatility
        price_lookback = config.get('vpd_price_lookback', 20)
        price_momentum = data[close_col].pct_change(price_lookback)
        data['price_momentum_norm'] = price_momentum / data['ATR_pct'] * 100

        # Volume momentum (z-score)
        if volume_col in data.columns:
            vol_lookback = config.get('vpd_volume_lookback', 20)
            vol_ma = data[volume_col].rolling(vol_lookback).mean()
            vol_std = data[volume_col].rolling(vol_lookback).std()
            data['volume_zscore'] = (data[volume_col] - vol_ma) / vol_std.replace(0, np.nan)

            # Price-Volume correlation for divergence detection
            data['price_vol_corr'] = data['price_momentum_norm'].rolling(price_lookback).corr(
                data['volume_zscore']
            )

        # Volatility breakout score
        vol_lookback_long = self.config.get('volatility_lookback', 50)
        vol_ma_long = data['vol_realized'].rolling(vol_lookback_long).mean()
        vol_std_long = data['vol_realized'].rolling(vol_lookback_long).std()
        data['vol_breakout_score'] = (data['vol_realized'] - vol_ma_long) / vol_std_long.replace(0, np.nan)

        # Return outlier features
        returns = data['returns']
        return_lookback = config.get('return_lookback', 100)
        data['return_zscore_rolling'] = returns.rolling(return_lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0,
            raw=False
        )

        # Multi-timeframe momentum for regime detection
        data['momentum_short'] = data[close_col].pct_change(6)
        data['momentum_medium'] = data[close_col].pct_change(16)
        data['momentum_long'] = data[close_col].pct_change(48)

        # Momentum regime score
        momentum_alignment = (np.sign(data['momentum_short']) *
                              np.sign(data['momentum_medium']) *
                              np.sign(data['momentum_long']))
        data['momentum_regime_score'] = momentum_alignment.rolling(4).mean()

        return data

    def detect_events(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect events and add event columns including CUSUM and Bollinger crossing events"""

        config = self.config.get('event_config', {})

        # Initialize event columns
        data['vpd_volatility_event'] = False
        data['outlier_event'] = False
        data['momentum_regime_event'] = False

        # Volume-Price Divergence + Volatility events
        if all(col in data.columns for col in ['price_vol_corr', 'vol_breakout_score']):
            divergence_threshold = config.get('vpd_divergence_threshold', 0.7)
            vol_threshold = config.get('vol_breakout_threshold', 2.0)

            if 'volume_zscore' in data.columns:
                vpd_condition = abs(data['price_vol_corr']) < divergence_threshold
                vol_condition = abs(data['vol_breakout_score']) > vol_threshold
                strength_condition = (abs(data['volume_zscore']) > 1.5) | (abs(data['price_momentum_norm']) > 1.5)
                data['vpd_volatility_event'] = vpd_condition & vol_condition & strength_condition

        # Return Distribution Outlier events
        if 'return_zscore_rolling' in data.columns:
            outlier_threshold = config.get('outlier_threshold', 2.5)
            data['outlier_event'] = abs(data['return_zscore_rolling']) > outlier_threshold

        # Momentum Regime Change events
        if 'momentum_regime_score' in data.columns:
            regime_threshold = config.get('momentum_regime_threshold', 0.3)
            regime_changes = abs(data['momentum_regime_score'].diff()) > regime_threshold
            data['momentum_regime_event'] = regime_changes

        # Combined traditional events
        data['traditional_event'] = (data['vpd_volatility_event'] |
                                     data['outlier_event'] |
                                     data['momentum_regime_event'])

        # Include CUSUM and Bollinger crossing events in combined events
        cusum_event = data.get('CUSUM_event', False)
        bb_any_cross = data.get('BB_any_cross', False)
        bb_squeeze = data.get('BB_squeeze', False)
        bb_expansion = data.get('BB_expansion', False)

        # Combined event indicators (including new event types)
        data['any_event'] = (data['traditional_event'] |
                             cusum_event |
                             bb_any_cross |
                             bb_squeeze |
                             bb_expansion)

        # Enhanced event type encoding
        data['event_type'] = 0
        data.loc[data['vpd_volatility_event'], 'event_type'] = 1
        data.loc[data['outlier_event'], 'event_type'] = 2
        data.loc[data['momentum_regime_event'], 'event_type'] = 3
        data.loc[cusum_event, 'event_type'] = 4
        data.loc[bb_any_cross, 'event_type'] = 5
        data.loc[bb_squeeze, 'event_type'] = 6
        data.loc[bb_expansion, 'event_type'] = 7

        # Handle multiple simultaneous events
        event_cols = ['vpd_volatility_event', 'outlier_event', 'momentum_regime_event']
        if 'CUSUM_event' in data.columns:
            event_cols.append('CUSUM_event')
        if 'BB_any_cross' in data.columns:
            event_cols.append('BB_any_cross')

        multiple_events = data[event_cols].sum(axis=1) > 1
        data.loc[multiple_events, 'event_type'] = 8 + data.loc[multiple_events, event_cols].sum(axis=1) - 2

        return data

    # Co-Event Detection Methods
    def add_co_event_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive co-event detection with 2-way, 3-way, and combined combinations
        """

        # Define base event columns (existing individual events)
        base_event_columns = [
            'vpd_volatility_event',
            'outlier_event',
            'momentum_regime_event',
            'CUSUM_event',
            'BB_upper_cross',
            'BB_lower_cross',
            'BB_middle_cross_up',
            'BB_middle_cross_down',
            'BB_squeeze',
            'BB_expansion'
        ]

        # Filter to only existing columns
        existing_events = [col for col in base_event_columns if col in data.columns]

        print(f"Detected {len(existing_events)} base event types for co-event analysis")

        # Get co-event configuration
        co_event_config = self.config.get('co_event_config', {})

        # Add 2-way co-events
        if co_event_config.get('enable_2way', True):
            data = self._add_2way_co_events(data, existing_events, co_event_config)

        # Add 3-way co-events
        if co_event_config.get('enable_3way', True):
            data = self._add_3way_co_events(data, existing_events, co_event_config)

        # Add combined 2-way and 3-way summary features
        if co_event_config.get('enable_combined', True):
            data = self._add_combined_co_events(data, co_event_config)

        return data

    def _add_2way_co_events(self, data: pd.DataFrame, event_columns: List[str], config: Dict) -> pd.DataFrame:
        """Add 2-way co-event combinations"""

        time_window = config.get('time_window', 1)
        create_named = config.get('create_named_combinations', True)
        create_generic = config.get('create_generic_combinations', True)
        max_combinations = config.get('max_combinations', 50)

        # Priority combinations first (meaningful financial combinations)
        priority_combinations = config.get('priority_2way_combinations', [])
        combinations_created = 0

        if create_named:
            print("Creating priority 2-way co-events...")
            for event1, event2 in priority_combinations:
                if event1 in event_columns and event2 in event_columns and combinations_created < max_combinations:
                    col_name = f"covent_2way_{event1.replace('_event', '')}_{event2.replace('_event', '')}"
                    data[col_name] = self._detect_co_event_within_window(
                        data, [event1, event2], time_window
                    )
                    combinations_created += 1

        # Generic combinations if requested and within limits
        if create_generic and combinations_created < max_combinations:
            print("Creating additional 2-way co-events...")
            remaining_combinations = max_combinations - combinations_created

            all_2way_combinations = list(combinations(event_columns, 2))
            # Remove already created priority combinations
            priority_set = set(priority_combinations)
            additional_combinations = [combo for combo in all_2way_combinations
                                       if combo not in priority_set and (combo[1], combo[0]) not in priority_set]

            # Limit to remaining budget
            additional_combinations = additional_combinations[:remaining_combinations]

            for event1, event2 in additional_combinations:
                col_name = f"covent_2way_{event1.replace('_event', '')}_{event2.replace('_event', '')}"
                data[col_name] = self._detect_co_event_within_window(
                    data, [event1, event2], time_window
                )
                combinations_created += 1

        # Summary 2-way features
        data['any_2way_covent'] = data[[col for col in data.columns if 'covent_2way_' in col]].any(axis=1)
        data['count_2way_covent'] = data[[col for col in data.columns if 'covent_2way_' in col]].sum(axis=1)

        print(f"Created {combinations_created} 2-way co-events")
        return data

    def _add_3way_co_events(self, data: pd.DataFrame, event_columns: List[str], config: Dict) -> pd.DataFrame:
        """Add 3-way co-event combinations"""

        time_window = config.get('time_window', 1)
        create_named = config.get('create_named_combinations', True)
        create_generic = config.get('create_generic_combinations', True)
        max_combinations = config.get('max_combinations', 50)

        combinations_created = 0

        # Priority 3-way combinations first
        priority_combinations = config.get('priority_3way_combinations', [])

        if create_named:
            print("Creating priority 3-way co-events...")
            for event1, event2, event3 in priority_combinations:
                if (event1 in event_columns and event2 in event_columns and
                        event3 in event_columns and combinations_created < max_combinations):
                    col_name = (f"covent_3way_{event1.replace('_event', '')}_{event2.replace('_event', '')}_"
                                f"{event3.replace('_event', '')}")
                    data[col_name] = self._detect_co_event_within_window(
                        data, [event1, event2, event3], time_window
                    )
                    combinations_created += 1

        # Additional generic 3-way combinations if requested and space allows
        if create_generic and combinations_created < max_combinations:
            print("Creating additional 3-way co-events...")
            remaining_combinations = max_combinations - combinations_created

            # Limit to most important events for 3-way (avoid explosion)
            important_events = event_columns[:6]  # Take top 6 events to limit combinations

            all_3way_combinations = list(combinations(important_events, 3))
            priority_set = set(priority_combinations)
            additional_combinations = [combo for combo in all_3way_combinations
                                       if combo not in priority_set]

            # Limit to remaining budget
            additional_combinations = additional_combinations[:remaining_combinations]

            for event1, event2, event3 in additional_combinations:
                col_name = (f"covent_3way_{event1.replace('_event', '')}_{event2.replace('_event', '')}_"
                            f"{event3.replace('_event', '')}")
                data[col_name] = self._detect_co_event_within_window(
                    data, [event1, event2, event3], time_window
                )
                combinations_created += 1

        # Summary 3-way features
        data['any_3way_covent'] = data[[col for col in data.columns if 'covent_3way_' in col]].any(axis=1)
        data['count_3way_covent'] = data[[col for col in data.columns if 'covent_3way_' in col]].sum(axis=1)

        print(f"Created {combinations_created} 3-way co-events")
        return data

    def _add_combined_co_events(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Add combined co-event features and meta-analysis"""

        # Overall co-event summary
        data['any_covent'] = (data.get('any_2way_covent', False) | data.get('any_3way_covent', False))
        data['total_covent_count'] = (data.get('count_2way_covent', 0) + data.get('count_3way_covent', 0))

        # Co-event intensity (how many simultaneous co-events)
        data['covent_intensity'] = data['total_covent_count']

        # High-intensity co-event periods (multiple co-events happening)
        data['high_intensity_covent'] = data['covent_intensity'] >= 2
        data['very_high_intensity_covent'] = data['covent_intensity'] >= 3

        # Co-event persistence (co-events lasting multiple periods)
        if 'any_covent' in data.columns:
            data['covent_persistence'] = data['any_covent'].rolling(window=3, min_periods=1).sum()
            data['sustained_covent'] = data['covent_persistence'] >= 2

        # Event escalation (single events evolving into co-events)
        individual_event_cols = [col for col in data.columns if '_event' in col and 'covent' not in col]
        if individual_event_cols:
            data['individual_event_count'] = data[individual_event_cols].sum(axis=1)
            data['event_escalation'] = (data['individual_event_count'] >= 1) & (data['any_covent'])

        # Special confluence patterns
        data = self._add_confluence_patterns(data)

        print("Added combined co-event features")
        return data

    def _detect_co_event_within_window(self, data: pd.DataFrame, event_columns: List[str], window: int) -> pd.Series:
        """
        Detect co-events within a time window

        Args:
            data: DataFrame containing event columns
            event_columns: List of event column names to check
            window: Time window (0 = exact same time, >0 = within N periods)

        Returns:
            Boolean series indicating co-event occurrence
        """

        if window == 0:
            # Exact simultaneous events
            return data[event_columns].all(axis=1)

        else:
            # Events within window
            co_events = pd.Series(False, index=data.index)

            for i in range(len(data)):
                # Check if all events occur within window around current time
                start_idx = max(0, i - window)
                end_idx = min(len(data), i + window + 1)

                window_data = data[event_columns].iloc[start_idx:end_idx]

                # Check if all event types have at least one occurrence in window
                events_in_window = window_data.any(axis=0)
                co_events.iloc[i] = events_in_window.all()

            return co_events

    def _add_confluence_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add special confluence pattern detection"""

        # Trend reversal confluence (CUSUM + Bollinger + momentum)
        if all(col in data.columns for col in ['CUSUM_event', 'momentum_regime_event']):
            bb_cross_cols = [col for col in data.columns if 'BB_' in col and 'cross' in col]
            if bb_cross_cols:
                bb_any_cross = data[bb_cross_cols].any(axis=1)
                data['trend_reversal_confluence'] = (data['CUSUM_event'] &
                                                     data['momentum_regime_event'] &
                                                     bb_any_cross)

        # Breakout confluence (volatility + price + volume)
        if all(col in data.columns for col in ['vpd_volatility_event', 'outlier_event', 'BB_expansion']):
            data['breakout_confluence'] = (data['vpd_volatility_event'] &
                                           data['outlier_event'] &
                                           data['BB_expansion'])

        # Squeeze release confluence
        if all(col in data.columns for col in ['BB_squeeze', 'BB_expansion', 'momentum_regime_event']):
            # Squeeze followed by expansion with momentum change
            squeeze_to_expansion = (data['BB_squeeze'].shift(1) | data['BB_squeeze'].shift(2)) & data['BB_expansion']
            data['squeeze_release_confluence'] = (squeeze_to_expansion & data['momentum_regime_event'])

        return data

    def detect_events_enhanced(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced event detection including co-events
        This replaces the original detect_events method
        """

        # First run original event detection
        data = self.detect_events(data)  # This calls the original method

        # Then add co-event detection
        data = self.add_co_event_detection(data)

        # Update overall event indicators to include co-events
        covent_cols = [col for col in data.columns if
                       'covent' in col and col not in ['any_covent', 'total_covent_count']]

        if covent_cols:
            # Update 'any_event' to include co-events
            data['any_event_with_covent'] = (data['any_event'] | data.get('any_covent', False))

            # Enhanced event type encoding
            data['enhanced_event_type'] = data['event_type'].copy()

            # Add co-event type encoding (starting from 20 to avoid conflicts)
            data.loc[data.get('any_2way_covent', False) & ~data['any_event'], 'enhanced_event_type'] = 20
            data.loc[data.get('any_3way_covent', False) & ~data['any_event'], 'enhanced_event_type'] = 21
            data.loc[data.get('high_intensity_covent', False), 'enhanced_event_type'] = 22
            data.loc[data.get('trend_reversal_confluence', False), 'enhanced_event_type'] = 23
            data.loc[data.get('breakout_confluence', False), 'enhanced_event_type'] = 24
            data.loc[data.get('squeeze_release_confluence', False), 'enhanced_event_type'] = 25

        return data

    def create_complete_dataset(self) -> pd.DataFrame:
        """
        Enhanced version of create_complete_dataset that includes co-events
        """

        print("Step 1/8: Adding base indicators calculated on low timeframe data...")
        data = self.add_base_indicators()

        print("Step 2/8: Adding CUSUM detection...")
        data = self.add_cusum_detection(data)

        print("Step 3/8: Adding Bollinger Band crossing events...")
        data = self.add_bollinger_crossing_events(data)

        print("Step 4/8: Adding normalized/relative indicators...")
        data = self.add_normalized_indicators(data)

        print("Step 5/8: Adding higher timeframe context...")
        data = self.add_higher_timeframe_context(data)

        print("Step 6/8: Adding event detection features...")
        data = self.add_event_detection_features(data)

        print("Step 7/8: Detecting individual events...")
        data = self.detect_events(data)

        print("Step 8/8: Adding co-event detection...")
        data = self.add_co_event_detection(data)

        print(f"Complete dataset with co-events shape: {data.shape}")
        return data

    def get_summary_statistics(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive summary of the enhanced dataset"""

        summary = {
            'dataset_info': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'date_range': (str(data.index.min()), str(data.index.max())),
                'base_timeframe': self.base_timeframe,
                'low_timeframe': self.low_timeframe,
                'higher_timeframes': self.higher_timeframes
            },

            'indicator_counts': {
                'base_indicators': len(
                    [col for col in data.columns if any(x in col for x in ['EMA_', 'SMA_', 'RSI', 'MACD', 'BB_'])]),
                'normalized_indicators': len([col for col in data.columns if any(
                    x in col for x in ['_distance_', '_pct', '_normalized', '_efficiency'])]),
                'higher_tf_indicators': len(
                    [col for col in data.columns if any(tf in col for tf in self.higher_timeframes)]),
                'event_features': len([col for col in data.columns if 'event' in col.lower() or any(
                    x in col for x in ['zscore', 'breakout', 'regime', 'CUSUM', 'BB_'])]),
                'cusum_features': len([col for col in data.columns if 'CUSUM' in col]),
                'bollinger_features': len([col for col in data.columns if 'BB_' in col and any(
                    x in col for x in ['cross', 'squeeze', 'expansion'])]),
                'co_event_features': len([col for col in data.columns if 'covent' in col])
            }
        }

        # Event statistics
        if 'any_event' in data.columns:
            event_stats = {
                'total_events': int(data['any_event'].sum()),
                'event_rate_pct': round(data['any_event'].mean() * 100, 2),
                'event_types': data['event_type'].value_counts().to_dict()
            }
            summary['events'] = event_stats

            # Enhanced event type meanings
            summary['event_type_legend'] = {
                0: "No event",
                1: "Volume-Price Divergence + Volatility",
                2: "Return Distribution Outlier",
                3: "Momentum Regime Change",
                4: "CUSUM Change Detection",
                5: "Bollinger Band Crossing",
                6: "Bollinger Band Squeeze",
                7: "Bollinger Band Expansion",
                8: "Multiple events (2 types)",
                9: "Multiple events (3 types)",
                10: "Multiple events (4+ types)"
            }

        # Co-event statistics
        if 'any_covent' in data.columns:
            covent_stats = {
                'total_co_events': int(data['any_covent'].sum()),
                'co_event_rate_pct': round(data['any_covent'].mean() * 100, 2),
                'avg_co_event_intensity': round(data['covent_intensity'].mean(), 2),
                'high_intensity_periods': int(data.get('high_intensity_covent', pd.Series(False)).sum())
            }
            summary['co_events'] = covent_stats

        # CUSUM statistics
        if 'CUSUM_event' in data.columns:
            cusum_stats = {
                'total_cusum_events': int(data['CUSUM_event'].sum()),
                'cusum_up_events': int((data['CUSUM_direction'] == 1).sum()),
                'cusum_down_events': int((data['CUSUM_direction'] == -1).sum()),
            }
            summary['cusum_events'] = cusum_stats

        # Bollinger Band crossing statistics
        bb_cols = [col for col in data.columns if 'BB_' in col and 'cross' in col]
        if bb_cols:
            bb_stats = {}
            for col in bb_cols:
                bb_stats[col] = int(data[col].sum())
            summary['bollinger_crossings'] = bb_stats

        return summary


def create_enhanced_dataset(resampled_data: pd.DataFrame,
                            custom_config: Dict = None,
                            low_timeframe: str = "30min") -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to create enhanced dataset with Elder's methodology and co-events

    Args:
        resampled_data: DataFrame from FlexiblePointInTimeResampler
        custom_config: Optional custom configuration
        low_timeframe: The timeframe to use for base indicators (e.g., "30min")

    Returns:
        Tuple of (enhanced_dataset, summary_statistics)
    """

    # Initialize system with Elder's methodology
    indicator_system = FlexibleTechnicalIndicators(resampled_data, custom_config, low_timeframe)

    # Create complete dataset
    enhanced_data = indicator_system.create_complete_dataset()

    # Generate summary
    summary = indicator_system.get_summary_statistics(enhanced_data)

    return enhanced_data, summary


# Example usage with Elder's methodology and co-events
def example_usage():
    """
    Example of how to use with Elder's approach and co-event detection
    """

    # Configuration for Elder's approach with 30min base indicators and co-events
    elder_config = {
        # Base indicators calculated on 30min data (Elder's intermediate timeframe)
        'ema_periods': {'fast': 12, 'medium': 26, 'slow': 50},
        'sma_periods': {'short': 20, 'medium': 50, 'long': 200},
        'rsi_period': 14,
        'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
        'bb_period': 20,
        'bb_std': 2,

        # Higher timeframe periods (for trend analysis)
        'daily_periods': {'short': 5, 'medium': 20, 'long': 50},
        'hourly_periods': {'short': 6, 'medium': 24, 'long': 72},

        # CUSUM parameters for change detection
        'cusum_config': {
            'threshold': 1.5,  # More sensitive for 30min data
            'drift': 0.3,
            'reset_period': 48  # Reset every 24 hours (48 * 30min)
        },

        # Bollinger Band crossing parameters
        'bb_crossing_config': {
            'lookback_periods': 2,  # 1 hour confirmation for 30min bars
            'min_distance_pct': 0.5  # More sensitive
        },

        # Enhanced event detection
        'event_config': {
            'vol_breakout_threshold': 1.5,  # More sensitive
            'outlier_threshold': 2.0,
            'momentum_regime_threshold': 0.25,
        },

        # Co-event configuration
        'co_event_config': {
            'enable_2way': True,
            'enable_3way': True,
            'enable_combined': True,
            'time_window': 2,  # Allow events within 2 periods (1-hour window)
            'max_combinations': 30,  # Limit for performance
            'create_named_combinations': True,
            'create_generic_combinations': False,  # Only create meaningful combinations

            'priority_2way_combinations': [
                ('CUSUM_event', 'BB_upper_cross'),
                ('CUSUM_event', 'BB_lower_cross'),
                ('vpd_volatility_event', 'momentum_regime_event'),
                ('outlier_event', 'BB_expansion'),
                ('momentum_regime_event', 'BB_squeeze'),
            ],

            'priority_3way_combinations': [
                ('CUSUM_event', 'vpd_volatility_event', 'momentum_regime_event'),
                ('BB_upper_cross', 'outlier_event', 'momentum_regime_event'),
                ('vpd_volatility_event', 'outlier_event', 'momentum_regime_event'),
            ]
        }
    }

    print("Enhanced Technical Indicators System with Elder's Methodology & Co-Events")
    print("=========================================================================")
    print("✓")
    return elder_config


conf = example_usage()
indicated, summ = create_enhanced_dataset(resampled, conf)
