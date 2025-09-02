import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')


class AdvancedEventDetector:
    """
    Advanced event detection system implementing three sophisticated event types:
    1. Volume-Price Divergence + Volatility Breakout
    2. Return Distribution Outliers
    3. Multi-timeframe Momentum Regime Changes
    """

    def __init__(self, data: pd.DataFrame, config: Dict = None):
        """
        Initialize the event detector with aligned multi-timeframe data

        Args:
            data: DataFrame with 30min target + higher timeframe data from PointInTimeResampler
            config: Configuration parameters for event detection
        """
        self.data = data.copy()
        self.config = config or self._default_config()

        # Ensure we have required columns
        self._validate_data()

        # Pre-compute base features
        self._compute_base_features()

    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            # Volume-Price Divergence settings
            'vpd_price_lookback': 20,
            'vpd_volume_lookback': 20,
            'vpd_divergence_threshold': 0.7,  # Correlation threshold
            'vpd_strength_threshold': 1.5,  # Z-score threshold

            # Volatility Breakout settings
            'vol_lookback': 50,
            'vol_breakout_threshold': 2.0,  # Standard deviations
            'vol_persistence_periods': 3,  # Minimum periods for breakout

            # Return Distribution Outlier settings
            'return_lookback': 100,
            'outlier_threshold': 2.5,  # Z-score threshold
            'outlier_min_significance': 0.01,  # P-value threshold
            'distribution_test_window': 250,  # Window for normality testing

            # Multi-timeframe Momentum settings
            'momentum_short_window': 6,  # 30min periods (3 hours)
            'momentum_medium_window': 16,  # 30min periods (8 hours)
            'momentum_long_window': 48,  # 30min periods (24 hours)
            'momentum_regime_threshold': 0.3,  # Threshold for regime change
            'momentum_persistence': 4,  # Minimum periods for regime

            # General settings
            'min_data_periods': 100,
        }

    def _validate_data(self):
        """Validate that required columns are present"""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        higher_tf_cols = ['4H_Close', '4H_Volume', '1D_Close']

        missing_cols = []
        for col in required_cols + higher_tf_cols:
            if col not in self.data.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _compute_base_features(self):
        """Compute base features needed for all event types"""
        data = self.data.copy()

        # Ensure numeric data types and handle missing values
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(method='ffill')

        # Handle higher timeframe columns
        for col in ['4H_Close', '4H_Volume', '1D_Close', '1D_Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(method='ffill')

        # Returns and volatility with proper error handling
        try:
            # Ensure Close prices are positive before taking log
            close_prices = data['Close'].copy()
            close_prices = close_prices[close_prices > 0]  # Remove non-positive values

            if len(close_prices) > 1:
                data['returns'] = np.log(close_prices).diff()
                data['vol_realized'] = data['returns'].rolling(20).std() * np.sqrt(24 * 60 / 30)  # Annualized
            else:
                data['returns'] = 0.0
                data['vol_realized'] = 0.0

        except Exception as e:
            print(f"Warning: Error computing returns: {e}")
            data['returns'] = data['Close'].pct_change()  # Fallback to simple returns
            data['vol_realized'] = data['returns'].rolling(20).std() * np.sqrt(24 * 60 / 30)

        # ATR with error handling
        try:
            data['atr'] = self._compute_atr(data['High'], data['Low'], data['Close'], 14)
        except Exception as e:
            print(f"Warning: Error computing ATR: {e}")
            data['atr'] = (data['High'] - data['Low']).rolling(14).mean()

        # Volume features with error handling
        try:
            data['volume_ma'] = data['Volume'].rolling(self.config['vpd_volume_lookback']).mean()
            data['volume_std'] = data['Volume'].rolling(self.config['vpd_volume_lookback']).std()

            # Avoid division by zero
            volume_std_safe = data['volume_std'].replace(0, np.nan)
            data['volume_zscore'] = (data['Volume'] - data['volume_ma']) / volume_std_safe
            data['volume_zscore'] = data['volume_zscore'].fillna(0)

        except Exception as e:
            print(f"Warning: Error computing volume features: {e}")
            data['volume_ma'] = data['Volume'].rolling(20).mean()
            data['volume_std'] = data['Volume'].rolling(20).std()
            data['volume_zscore'] = 0.0

        # Price momentum features
        try:
            data['price_momentum_short'] = data['Close'].pct_change(self.config['momentum_short_window'])
            data['price_momentum_medium'] = data['Close'].pct_change(self.config['momentum_medium_window'])
            data['price_momentum_long'] = data['Close'].pct_change(self.config['momentum_long_window'])
        except Exception as e:
            print(f"Warning: Error computing price momentum: {e}")
            data['price_momentum_short'] = 0.0
            data['price_momentum_medium'] = 0.0
            data['price_momentum_long'] = 0.0

        # Higher timeframe features with error handling
        if '4H_Close' in data.columns:
            try:
                h4_close = data['4H_Close'].dropna()
                if len(h4_close[h4_close > 0]) > 1:
                    data['4H_returns'] = np.log(h4_close[h4_close > 0]).diff()
                else:
                    data['4H_returns'] = 0.0

                data['4H_momentum'] = data['4H_Close'].pct_change(6)  # 24H momentum in 4H bars
            except Exception as e:
                print(f"Warning: Error computing 4H features: {e}")
                data['4H_returns'] = 0.0
                data['4H_momentum'] = 0.0

        if '1D_Close' in data.columns:
            try:
                d1_close = data['1D_Close'].dropna()
                if len(d1_close[d1_close > 0]) > 1:
                    data['1D_returns'] = np.log(d1_close[d1_close > 0]).diff()
                else:
                    data['1D_returns'] = 0.0

                data['1D_momentum'] = data['1D_Close'].pct_change(7)  # 7D momentum
            except Exception as e:
                print(f"Warning: Error computing 1D features: {e}")
                data['1D_returns'] = 0.0
                data['1D_momentum'] = 0.0

        # Fill any remaining NaN values
        numeric_feature_columns = [col for col in data.columns if
                                   data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        data[numeric_feature_columns] = data[numeric_feature_columns].fillna(0)

        self.data = data

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """Compute Average True Range with error handling"""
        try:
            # Ensure all series are numeric
            high = pd.to_numeric(high, errors='coerce').fillna(method='ffill')
            low = pd.to_numeric(low, errors='coerce').fillna(method='ffill')
            close = pd.to_numeric(close, errors='coerce').fillna(method='ffill')

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            # Handle any remaining NaN values
            tr1 = tr1.fillna(0)
            tr2 = tr2.fillna(0)
            tr3 = tr3.fillna(0)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window, min_periods=1).mean()

            return atr.fillna(0)

        except Exception as e:
            print(f"Warning: Error in ATR calculation: {e}")
            # Fallback to simple high-low range
            return (high - low).rolling(window, min_periods=1).mean().fillna(0)

    def detect_volume_price_divergence_volatility_events(self) -> pd.Series:
        """
        Detect Volume-Price Divergence combined with Volatility Breakout events

        Logic:
        1. Identify periods where price and volume momentum diverge significantly
        2. Confirm with volatility breakout
        3. Require minimum strength and persistence
        """
        data = self.data
        config = self.config

        try:
            # Price momentum (normalized by volatility) with safety checks
            atr_safe = data['atr'].replace(0, data['atr'].mean())  # Avoid division by zero
            price_momentum = data['Close'].pct_change(config['vpd_price_lookback']) / atr_safe
            price_momentum = price_momentum.fillna(0)

            # Volume momentum with safety checks
            volume_momentum = data['volume_zscore'].fillna(0)

            # Rolling correlation between price and volume momentum
            corr_window = config['vpd_price_lookback']

            # Compute correlation with minimum periods to handle sparse data
            price_vol_corr = price_momentum.rolling(
                corr_window, min_periods=max(5, corr_window // 4)
            ).corr(volume_momentum)
            price_vol_corr = price_vol_corr.fillna(0)

            # Volatility breakout detection with safety checks
            vol_ma = data['vol_realized'].rolling(
                config['vol_lookback'], min_periods=max(10, config['vol_lookback'] // 5)
            ).mean()
            vol_std = data['vol_realized'].rolling(
                config['vol_lookback'], min_periods=max(10, config['vol_lookback'] // 5)
            ).std()

            vol_std_safe = vol_std.replace(0, vol_std.mean())  # Avoid division by zero
            vol_breakout = (data['vol_realized'] - vol_ma) / vol_std_safe
            vol_breakout = vol_breakout.fillna(0)

            # Store indicator columns in the data for later use
            data['price_momentum_vpd'] = price_momentum
            data['volume_momentum_vpd'] = volume_momentum
            data['price_vol_correlation'] = price_vol_corr
            data['vol_breakout_score'] = vol_breakout

            # Event conditions with safety checks - convert to boolean explicitly
            divergence_condition = (abs(price_vol_corr) < config['vpd_divergence_threshold']).astype(bool).fillna(False)
            volatility_condition = (abs(vol_breakout) > config['vol_breakout_threshold']).astype(bool).fillna(False)
            strength_condition = (
                    (abs(price_momentum) > config['vpd_strength_threshold']) |
                    (abs(volume_momentum) > config['vpd_strength_threshold'])
            ).astype(bool).fillna(False)

            # Combine conditions
            events = divergence_condition & volatility_condition & strength_condition
            events = events.astype(bool).fillna(False)

            # Add persistence filter
            events_filtered = self._apply_persistence_filter(
                events, config['vol_persistence_periods']
            )

            return events_filtered.fillna(False)

        except Exception as e:
            print(f"Warning: Error in VPD event detection: {e}")
            return pd.Series(False, index=data.index)

    def detect_return_distribution_outliers(self) -> pd.Series:
        """
        Detect Return Distribution Outlier events

        Logic:
        1. Use rolling window to estimate return distribution parameters
        2. Identify returns that are statistical outliers
        3. Adaptive thresholds based on current market regime
        4. Include significance testing
        """
        data = self.data
        config = self.config

        try:
            returns = data['returns'].dropna()
            events = pd.Series(False, index=data.index)

            lookback = config['return_lookback']
            threshold = config['outlier_threshold']

            # Store intermediate calculations for transparency
            z_scores = pd.Series(np.nan, index=data.index)
            adaptive_thresholds = pd.Series(np.nan, index=data.index)
            p_values = pd.Series(np.nan, index=data.index)

            for i in range(lookback, len(returns)):
                # Get rolling window of returns
                window_returns = returns.iloc[i - lookback:i]
                current_return = returns.iloc[i]
                current_idx = returns.index[i]

                if pd.isna(current_return) or len(window_returns.dropna()) < lookback * 0.8:
                    continue

                # Compute rolling statistics
                window_mean = window_returns.mean()
                window_std = window_returns.std()

                if window_std == 0:
                    continue

                # Z-score based detection
                z_score = abs((current_return - window_mean) / window_std)
                z_scores.loc[current_idx] = z_score

                # Adaptive threshold based on volatility regime
                vol_regime_multiplier = min(2.0, data.loc[current_idx, 'vol_realized'] /
                                            data['vol_realized'].iloc[i - lookback:i].median())
                adaptive_threshold = threshold * vol_regime_multiplier
                adaptive_thresholds.loc[current_idx] = adaptive_threshold

                # Statistical significance test
                p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed test
                p_values.loc[current_idx] = p_value

                # Event conditions
                outlier_condition = z_score > adaptive_threshold
                significance_condition = p_value < config['outlier_min_significance']

                if outlier_condition and significance_condition:
                    events.loc[current_idx] = True

            # Store indicator columns
            data['return_zscore'] = z_scores
            data['outlier_threshold_adaptive'] = adaptive_thresholds
            data['outlier_pvalue'] = p_values

            return events.astype(bool).fillna(False)

        except Exception as e:
            print(f"Warning: Error in outlier event detection: {e}")
            return pd.Series(False, index=data.index)

    def detect_momentum_regime_changes(self) -> pd.Series:
        """
        Detect Multi-timeframe Momentum Regime Change events

        Logic:
        1. Compare momentum across multiple timeframes
        2. Identify regime changes when momentum alignment shifts
        3. Consider both cross-timeframe and temporal regime changes
        """
        data = self.data
        config = self.config

        try:
            # Momentum indicators across timeframes
            momentum_30m_short = data['price_momentum_short'].fillna(0)
            momentum_30m_medium = data['price_momentum_medium'].fillna(0)
            momentum_30m_long = data['price_momentum_long'].fillna(0)

            # Cross-timeframe momentum (if available)
            momentum_4h = data.get('4H_momentum', momentum_30m_medium).fillna(0)
            momentum_1d = data.get('1D_momentum', momentum_30m_long).fillna(0)

            # Momentum regime indicators
            # 1. Intra-timeframe regime change (30m timeframe momentum alignment)
            momentum_alignment_30m = (
                    np.sign(momentum_30m_short) *
                    np.sign(momentum_30m_medium) *
                    np.sign(momentum_30m_long)
            )
            momentum_regime_30m = momentum_alignment_30m.rolling(
                config['momentum_persistence'], min_periods=1
            ).mean()

            # 2. Cross-timeframe regime change
            cross_tf_alignment = (
                    np.sign(momentum_30m_medium) *
                    np.sign(momentum_4h) *
                    np.sign(momentum_1d)
            )
            cross_tf_regime = cross_tf_alignment.rolling(
                config['momentum_persistence'], min_periods=1
            ).mean()

            # 3. Momentum strength regime
            momentum_strength = (
                                        abs(momentum_30m_short) +
                                        abs(momentum_30m_medium) +
                                        abs(momentum_30m_long)
                                ) / 3
            momentum_strength_ma = momentum_strength.rolling(
                config['momentum_long_window'], min_periods=max(5, config['momentum_long_window'] // 5)
            ).mean()

            # Avoid division by zero
            momentum_strength_ma_safe = momentum_strength_ma.replace(0, momentum_strength_ma.mean())
            momentum_strength_regime = momentum_strength / momentum_strength_ma_safe

            # Store indicator columns
            data['momentum_alignment_30m'] = momentum_alignment_30m
            data['momentum_regime_30m'] = momentum_regime_30m
            data['cross_tf_alignment'] = cross_tf_alignment
            data['cross_tf_regime'] = cross_tf_regime
            data['momentum_strength'] = momentum_strength
            data['momentum_strength_regime'] = momentum_strength_regime

            # Detect regime changes - convert to boolean explicitly
            regime_change_intra = (
                    abs(momentum_regime_30m.diff()) > config['momentum_regime_threshold']
            ).astype(bool).fillna(False)

            regime_change_cross = (
                    abs(cross_tf_regime.diff()) > config['momentum_regime_threshold']
            ).astype(bool).fillna(False)

            regime_change_strength = (
                    abs(momentum_strength_regime.diff()) > config['momentum_regime_threshold']
            ).astype(bool).fillna(False)

            # Combine regime change signals
            events = regime_change_intra | regime_change_cross | regime_change_strength
            events = events.astype(bool).fillna(False)

            # Add persistence filter
            events_filtered = self._apply_persistence_filter(
                events, config['momentum_persistence']
            )

            return events_filtered.fillna(False)

        except Exception as e:
            print(f"Warning: Error in momentum regime event detection: {e}")
            return pd.Series(False, index=data.index)

    def _apply_persistence_filter(self, events: pd.Series, min_periods: int) -> pd.Series:
        """Apply persistence filter to reduce noise"""
        try:
            # Ensure events is boolean
            events = events.astype(bool).fillna(False)

            # Create rolling sum of events
            rolling_events = events.rolling(min_periods, min_periods=1).sum()

            # Keep events that persist for minimum periods
            persistent_events = rolling_events >= 1
            persistent_events = persistent_events.astype(bool).fillna(False)

            # Only trigger at the start of persistent periods
            event_starts = persistent_events & (~persistent_events.shift(1).fillna(False))

            return event_starts.astype(bool).fillna(False)

        except Exception as e:
            print(f"Warning: Error in persistence filter: {e}")
            return pd.Series(False, index=events.index)

    def detect_all_events(self) -> pd.DataFrame:
        """
        Detect all event types and add event columns to the original dataframe

        Returns:
            Original DataFrame with added event indicator columns
        """
        if len(self.data) < self.config['min_data_periods']:
            raise ValueError(f"Insufficient data. Need at least {self.config['min_data_periods']} periods")

        print("Detecting Volume-Price Divergence + Volatility events...")
        vpd_vol_events = self.detect_volume_price_divergence_volatility_events()

        print("Detecting Return Distribution Outlier events...")
        outlier_events = self.detect_return_distribution_outliers()

        print("Detecting Momentum Regime Change events...")
        momentum_events = self.detect_momentum_regime_changes()

        # Add event columns to original data
        result_df = self.data.copy()

        # Add event indicators
        result_df['vpd_volatility_event'] = vpd_vol_events.fillna(False)
        result_df['outlier_event'] = outlier_events.fillna(False)
        result_df['momentum_regime_event'] = momentum_events.fillna(False)

        # Create combined event indicator
        result_df['any_event'] = (
                result_df['vpd_volatility_event'] |
                result_df['outlier_event'] |
                result_df['momentum_regime_event']
        )

        # Add numeric event type column (useful for ML)
        result_df['event_type'] = 0  # 0 = no event
        result_df.loc[result_df['vpd_volatility_event'], 'event_type'] = 1
        result_df.loc[result_df['outlier_event'], 'event_type'] = 2
        result_df.loc[result_df['momentum_regime_event'], 'event_type'] = 3
        # If multiple events occur simultaneously, use highest priority
        result_df.loc[result_df['vpd_volatility_event'] & result_df['outlier_event'], 'event_type'] = 4
        result_df.loc[result_df['vpd_volatility_event'] & result_df['momentum_regime_event'], 'event_type'] = 5
        result_df.loc[result_df['outlier_event'] & result_df['momentum_regime_event'], 'event_type'] = 6
        result_df.loc[result_df['vpd_volatility_event'] & result_df['outlier_event'] & result_df[
            'momentum_regime_event'], 'event_type'] = 7

        return result_df

    def _add_event_metadata(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata about events (deprecated - now included in main dataframe)"""
        # This method is kept for backwards compatibility but is no longer used
        # since events are now added directly to the main dataframe
        return events_df

    def get_event_statistics(self, data_with_events: pd.DataFrame) -> Dict:
        """Generate statistics about detected events from dataframe with event columns"""
        stats_dict = {}

        event_columns = {
            'vpd_volatility_event': 'Volume-Price Divergence + Volatility Events',
            'outlier_event': 'Return Distribution Outlier Events',
            'momentum_regime_event': 'Momentum Regime Change Events',
            'any_event': 'Any Event'
        }

        for event_col, event_name in event_columns.items():
            if event_col in data_with_events.columns:
                events = data_with_events[event_col]
                total_events = events.sum()
                event_rate = total_events / len(events) * 100

                stats_dict[event_name] = {
                    'total_events': int(total_events),
                    'event_rate_pct': round(event_rate, 2),
                    'avg_days_between_events': round(len(events) / max(1, total_events) * 0.5, 1),  # 30min -> days
                }

                # Add performance statistics if we have price data
                if total_events > 0 and 'returns' in data_with_events.columns:
                    event_returns = data_with_events.loc[events, 'returns'].dropna()
                    if len(event_returns) > 0:
                        stats_dict[event_name].update({
                            'avg_event_return_pct': round(event_returns.mean() * 100, 3),
                            'event_return_std_pct': round(event_returns.std() * 100, 3),
                            'event_sharpe': round(event_returns.mean() / max(0.001, event_returns.std()), 2),
                        })

        return stats_dict


def add_events_to_resampled_data(resampled_data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """
    Standalone function to add event detection columns to any resampled dataframe

    Args:
        resampled_data: DataFrame with 30min OHLCV + higher timeframe data
        config: Optional configuration for event detection

    Returns:
        Original dataframe with added event columns
    """
    detector = AdvancedEventDetector(resampled_data, config)
    return detector.detect_all_events()


def process_events():
    """
    Process ETH data and detect events, returning the full dataframe with event columns
    """
    try:
        # Import and use the resampler from the previous script
        from data_resampling import PointInTimeResampler

        # Load ETH data
        eth5m = pd.read_csv('D:/Seagull_data/historical_data/time/ETHEUR/ETHEUR_5m.csv')
        eth5m['t'] = eth5m.time
        eth5m.time = pd.to_datetime(eth5m.time, unit='ms')
        eth5m.set_index('time', inplace=True)

        if 'Unnamed: 0' in eth5m.columns:
            eth5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

        print(f"Loaded ETH data: {eth5m.shape}")
        print(f"Date range: {eth5m.index.min()} to {eth5m.index.max()}")

        # Create aligned multi-timeframe data
        resampler = PointInTimeResampler(eth5m)
        aligned_data = resampler.create_aligned_dataset(
            target_timeframe='30min',
            higher_timeframes=['4H', '1D']
        )

        print(f"Aligned data shape: {aligned_data.shape}")

        # Add event detection columns
        data_with_events = add_events_to_resampled_data(aligned_data)

        print(f"Data with events shape: {data_with_events.shape}")

        # Get statistics
        detector = AdvancedEventDetector(aligned_data)
        event_stats = detector.get_event_statistics(data_with_events)

        print("\n" + "=" * 50)
        print("EVENT DETECTION RESULTS")
        print("=" * 50)

        for event_type, stats in event_stats.items():
            print(f"\n{event_type}:")
            for stat_name, stat_value in stats.items():
                print(f"  {stat_name}: {stat_value}")

        # Show sample of events
        event_rows = data_with_events[data_with_events['any_event']]
        if len(event_rows) > 0:
            print(f"\nSample of detected events:")
            sample_cols = ['Close', 'Volume', 'vpd_volatility_event', 'outlier_event',
                           'momentum_regime_event', 'event_type', 'returns', 'vol_realized']
            available_cols = [col for col in sample_cols if col in event_rows.columns]
            print(event_rows[available_cols].head(10).round(4))

        print(f"\nEvent type encoding:")
        print("  0: No event")
        print("  1: Volume-Price Divergence + Volatility event only")
        print("  2: Return Distribution Outlier event only")
        print("  3: Momentum Regime Change event only")
        print("  4: VPD + Outlier events")
        print("  5: VPD + Momentum events")
        print("  6: Outlier + Momentum events")
        print("  7: All three event types")

        print(f"\nEvent type distribution:")
        event_type_counts = data_with_events['event_type'].value_counts().sort_index()
        for event_type, count in event_type_counts.items():
            pct = count / len(data_with_events) * 100
            print(f"  Type {event_type}: {count} occurrences ({pct:.2f}%)")
        print(data_with_events.head())
        print(data_with_events.columns)
        print(event_stats)
        return data_with_events, event_stats

    except ImportError:
        print("Error: Could not import PointInTimeResampler. Make sure the previous script is available.")
        return None, None
    except FileNotFoundError:
        print("Error: Could not find 'csv/tb/ETHEUR_5m.csv'")
        return None, None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None



process_events()
