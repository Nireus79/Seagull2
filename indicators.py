import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings

# Technical Analysis library
import ta

warnings.filterwarnings('ignore')


class FlexibleTechnicalIndicators:
    """
    Flexible technical indicators system using ta library for standard indicators,
    focused on multi-timeframe alignment, normalization, and event detection.
    """

    def __init__(self, resampled_data: pd.DataFrame, config: Dict = None):
        """
        Initialize with resampled multi-timeframe data

        Args:
            resampled_data: DataFrame from FlexiblePointInTimeResampler with higher TF columns
            config: Configuration for indicators and events
        """
        self.data = resampled_data.copy()
        self.config = config or self._default_config()

        # Detect available timeframes
        self.base_timeframe = self._detect_base_timeframe()
        self.higher_timeframes = self._detect_higher_timeframes()

        print(f"Detected base timeframe: {self.base_timeframe}")
        print(f"Available higher timeframes: {self.higher_timeframes}")

        # Validate required columns
        self._validate_data()

    def _default_config(self) -> Dict:
        """Default configuration for indicators and events"""
        return {
            # Standard indicator periods
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

    def _detect_base_timeframe(self) -> str:
        """Detect base timeframe from index frequency"""
        if len(self.data) < 2:
            return "unknown"

        try:
            # Ensure index is datetime
            if not isinstance(self.data.index, pd.DatetimeIndex):
                print("Warning: Index is not DatetimeIndex, attempting conversion...")
                self.data.index = pd.to_datetime(self.data.index)

            time_diff = self.data.index[1] - self.data.index[0]

            # Handle different time_diff types
            if hasattr(time_diff, 'total_seconds'):
                total_seconds = time_diff.total_seconds()
            elif isinstance(time_diff, (int, float)):
                # Assume nanoseconds and convert to seconds
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
            print(f"Index type: {type(self.data.index)}")
            print(f"Index sample: {self.data.index[:3]}")
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
        required_base = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_base if col not in self.data.columns]

        if missing:
            raise ValueError(f"Missing required base columns: {missing}")

    def add_base_indicators(self) -> pd.DataFrame:
        """Add standard technical indicators using ta library"""
        data = self.data.copy()

        # Use ta library for all standard indicators
        high, low, close, volume = data['High'], data['Low'], data['Close'], data['Volume']

        # Trend indicators - EMAs
        if 'ema_periods' in self.config:
            for name, period in self.config['ema_periods'].items():
                data[f'EMA_{name}_{period}'] = ta.trend.EMAIndicator(close, window=period).ema_indicator()

        # SMAs with safety check
        if 'sma_periods' in self.config:
            for name, period in self.config['sma_periods'].items():
                data[f'SMA_{name}_{period}'] = ta.trend.SMAIndicator(close, window=period).sma_indicator()

        # MACD with safety check
        if 'macd_params' in self.config:
            macd_params = self.config['macd_params']
            macd = ta.trend.MACD(
                close,
                window_slow=macd_params.get('slow', 26),
                window_fast=macd_params.get('fast', 12),
                window_sign=macd_params.get('signal', 9)
            )
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_histogram'] = macd.macd_diff()

        # RSI with safety check
        rsi_period = self.config.get('rsi_period', 14)
        data['RSI'] = ta.momentum.RSIIndicator(close, window=rsi_period).rsi()

        # Bollinger Bands with safety check
        bb_period = self.config.get('bb_period', 20)
        bb_std = self.config.get('bb_std', 2)
        bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_middle'] = bb.bollinger_mavg()
        data['BB_lower'] = bb.bollinger_lband()

        # ATR with safety check
        atr_period = self.config.get('atr_period', 14)
        data['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()

        # Volume indicators
        data['Volume_SMA'] = ta.trend.SMAIndicator(volume, window=20).sma_indicator()
        data['Volume_ratio'] = volume / data['Volume_SMA']

        return data

    def add_normalized_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add López de Prado style normalized/relative indicators"""

        # Base volatility measures
        data['returns'] = data['Close'].pct_change()
        data['vol_realized'] = data['returns'].rolling(20).std() * np.sqrt(24 * 60 / 30)  # Annualized
        data['ATR_pct'] = data['ATR'] / data['Close'] * 100  # ATR as % of price

        # Normalized trend indicators (distances from moving averages)
        if 'ema_periods' in self.config:
            for name, period in self.config['ema_periods'].items():
                ema_col = f'EMA_{name}_{period}'
                if ema_col in data.columns:
                    # Distance normalized by ATR (López de Prado style)
                    data[f'EMA_{name}_distance_atr'] = (data['Close'] - data[ema_col]) / data['ATR']
                    # Percentage distance
                    data[f'EMA_{name}_distance_pct'] = (data['Close'] / data[ema_col] - 1) * 100

        # Volatility-normalized MACD (more stable)
        if 'MACD' in data.columns and 'ATR' in data.columns:
            data['MACD_normalized'] = data['MACD'] / data['ATR']
            data['MACD_efficiency'] = data['MACD'] / data['vol_realized'].replace(0, np.nan)

        # RSI momentum (Elder style - RSI direction * momentum alignment)
        if 'RSI' in data.columns:
            momentum_direction = np.sign(data['Close'].pct_change(5))
            data['RSI_momentum'] = (data['RSI'] - 50) * momentum_direction

        # Bollinger Bands position and efficiency
        if all(col in data.columns for col in ['BB_upper', 'BB_lower', 'Close']):
            bb_range = data['BB_upper'] - data['BB_lower']
            data['BB_position'] = ((data['Close'] - data['BB_lower']) / bb_range * 100).fillna(50)
            data['BB_width_pct'] = (bb_range / data['Close'] * 100).fillna(0)

        # Volume-price relationship (microstructure)
        corr_window = self.config.get('correlation_window', 20)
        data['volume_price_corr'] = data['Volume'].rolling(corr_window).corr(data['Close'].pct_change())

        # Price impact efficiency
        price_change_abs = abs(data['Close'].pct_change())
        volume_normalized = data['Volume_ratio'].replace(0, np.nan)
        data['price_impact_efficiency'] = price_change_abs / volume_normalized

        return data

    def add_higher_timeframe_context(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add higher timeframe indicators using ACTUAL timeframe data"""

        for tf in self.higher_timeframes:
            tf_close_col = f"{tf}_Close"

            if tf_close_col not in data.columns:
                continue

            print(f"Processing {tf} timeframe context...")

            # Extract actual timeframe data
            tf_data = self._extract_actual_timeframe_data(data, tf)

            if len(tf_data) < 20:  # Minimum data check
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
            # Keep only rows where the TF close price changes
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

        # Price position relative to higher TF
        data[f'price_vs_{tf}_pct'] = (data['Close'] / data[tf_close] - 1) * 100

        # Momentum alignment (Elder's triple screen concept)
        base_momentum = data['Close'].pct_change(5)

        if f'{tf}_momentum_short' in data.columns:
            tf_momentum = data[f'{tf}_momentum_short']
            # Momentum alignment: +1 same direction, -1 opposite, 0 mixed
            data[f'momentum_align_{tf}'] = np.sign(base_momentum) * np.sign(tf_momentum)

            # Momentum strength ratio
            base_momentum_abs = abs(base_momentum)
            tf_momentum_abs = abs(tf_momentum)
            data[f'momentum_ratio_{tf}'] = (base_momentum_abs / tf_momentum_abs.replace(0, np.nan)).fillna(1)

        # Volatility cascade (volatility regime comparison)
        if f'{tf}_volatility' in data.columns:
            base_vol = data['vol_realized']
            tf_vol = data[f'{tf}_volatility']
            data[f'vol_regime_{tf}'] = (base_vol / tf_vol.replace(0, np.nan)).fillna(1)

        # Trend strength comparison
        if f'{tf}_EMA_short' in data.columns:
            # Higher TF trend strength
            tf_trend_strength = abs(data[tf_close] - data[f'{tf}_EMA_short']) / data[tf_close] * 100
            # Base TF trend strength
            base_trend_strength = abs(data['Close'] - data['EMA_fast_12']) / data['Close'] * 100
            data[f'trend_strength_vs_{tf}'] = base_trend_strength - tf_trend_strength

        return data

    def add_event_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add specific features needed for event detection"""

        # Volume-Price Divergence features
        config = self.config.get('event_config', {})

        # Price momentum normalized by volatility
        price_lookback = config.get('vpd_price_lookback', 20)
        price_momentum = data['Close'].pct_change(price_lookback)
        data['price_momentum_norm'] = price_momentum / data['ATR_pct'] * 100

        # Volume momentum (z-score)
        vol_lookback = config.get('vpd_volume_lookback', 20)
        vol_ma = data['Volume'].rolling(vol_lookback).mean()
        vol_std = data['Volume'].rolling(vol_lookback).std()
        data['volume_zscore'] = (data['Volume'] - vol_ma) / vol_std.replace(0, np.nan)

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
        data['momentum_short'] = data['Close'].pct_change(6)  # 3 hours for 30min base
        data['momentum_medium'] = data['Close'].pct_change(16)  # 8 hours
        data['momentum_long'] = data['Close'].pct_change(48)  # 24 hours

        # Momentum regime score
        momentum_alignment = (np.sign(data['momentum_short']) *
                              np.sign(data['momentum_medium']) *
                              np.sign(data['momentum_long']))
        data['momentum_regime_score'] = momentum_alignment.rolling(4).mean()

        return data

    def detect_events(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect events and add event columns"""

        config = self.config.get('event_config', {})

        # Initialize event columns
        data['vpd_volatility_event'] = False
        data['outlier_event'] = False
        data['momentum_regime_event'] = False

        # Volume-Price Divergence + Volatility events
        if all(col in data.columns for col in ['price_vol_corr', 'vol_breakout_score', 'volume_zscore']):
            divergence_threshold = config.get('vpd_divergence_threshold', 0.7)
            vol_threshold = config.get('vol_breakout_threshold', 2.0)

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

        # Combined event indicators
        data['any_event'] = (data['vpd_volatility_event'] |
                             data['outlier_event'] |
                             data['momentum_regime_event'])

        # Event type encoding
        data['event_type'] = 0
        data.loc[data['vpd_volatility_event'], 'event_type'] = 1
        data.loc[data['outlier_event'], 'event_type'] = 2
        data.loc[data['momentum_regime_event'], 'event_type'] = 3

        # Handle multiple simultaneous events
        event_cols = ['vpd_volatility_event', 'outlier_event', 'momentum_regime_event']
        multiple_events = data[event_cols].sum(axis=1) > 1
        data.loc[multiple_events, 'event_type'] = 4 + data.loc[multiple_events, event_cols].sum(axis=1) - 2

        return data

    def create_complete_dataset(self) -> pd.DataFrame:
        """Create complete dataset with all indicators and events"""

        print("Step 1/5: Adding base indicators using ta library...")
        data = self.add_base_indicators()

        print("Step 2/5: Adding normalized/relative indicators...")
        data = self.add_normalized_indicators(data)

        print("Step 3/5: Adding higher timeframe context...")
        data = self.add_higher_timeframe_context(data)

        print("Step 4/5: Adding event detection features...")
        data = self.add_event_detection_features(data)

        print("Step 5/5: Detecting events...")
        data = self.detect_events(data)

        print(f"Complete dataset shape: {data.shape}")
        return data

    def get_summary_statistics(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive summary of the enhanced dataset"""

        summary = {
            'dataset_info': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'date_range': (str(data.index.min()), str(data.index.max())),
                'base_timeframe': self.base_timeframe,
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
                    x in col for x in ['zscore', 'breakout', 'regime'])])
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

            # Event type meanings
            summary['event_type_legend'] = {
                0: "No event",
                1: "Volume-Price Divergence + Volatility",
                2: "Return Distribution Outlier",
                3: "Momentum Regime Change",
                4: "Multiple events (2 types)",
                5: "Multiple events (3 types)"
            }

        return summary


def create_enhanced_dataset(resampled_data: pd.DataFrame,
                            custom_config: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to create enhanced dataset with all indicators and events

    Args:
        resampled_data: DataFrame from FlexiblePointInTimeResampler
        custom_config: Optional custom configuration

    Returns:
        Tuple of (enhanced_dataset, summary_statistics)
    """

    # Initialize system
    indicator_system = FlexibleTechnicalIndicators(resampled_data, custom_config)

    # Create complete dataset
    enhanced_data = indicator_system.create_complete_dataset()

    # Generate summary
    summary = indicator_system.get_summary_statistics(enhanced_data)

    return enhanced_data, summary


# Example usage and configuration
def example_usage(resampled_data):
    """
    Example of how to use with different timeframe configurations
    """
    config_30min_4h_24h = {
        # 30min base indicators
        'ema_periods': {'fast': 12, 'medium': 26, 'slow': 50},
        'rsi_period': 14,
        'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},

        # 4H indicators (uses actual 4H closes)
        'hourly_periods': {
            'short': 6,  # 6 * 4H periods
            'medium': 24,  # 24 * 4H periods
            'long': 72  # 72 * 4H periods
        },

        # 24H/Daily indicators (uses actual daily closes)
        'daily_periods': {
            'short': 5,  # 5 actual days
            'medium': 20,  # 20 actual days
            'long': 50  # 50 actual days
        }
    }
    # Custom configuration example
    custom_config = {
        # For 2H instead of 4H, modify your resampler first, then:
        'hourly_periods': {'short': 12, 'medium': 48, 'long': 168},  # 2H-based periods

        # Custom daily periods (uses actual daily closes)
        'daily_periods': {'short': 7, 'medium': 21, 'long': 50},  # 7, 21, 50 DAYS

        # More sensitive event detection
        'event_config': {
            'vol_breakout_threshold': 2.0,  # More sensitive
            'outlier_threshold': 2.0,  # More sensitive
            'momentum_regime_threshold': 0.2,  # More sensitive
        }
    }

    print("Enhanced Technical Indicators System")
    print("====================================")
    print("Uses 'ta' library for standard indicators")
    print("Focuses on multi-timeframe alignment and normalization")
    print("Includes López de Prado style relative indicators")
    print("Integrates Elder's triple screen methodology")
    print("Provides comprehensive event detection")

    # Usage:
    enhanced_data, summary = create_enhanced_dataset(resampled_data, config_30min_4h_24h)

    return enhanced_data, summary


data = pd.read_csv('resampled5mEE.csv')
indicated, summ = example_usage(data)
print(indicated)
print(summ)
