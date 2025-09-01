import pandas as pd
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')
"""Key Features:
1. True Point-in-Time Alignment

get_point_in_time_data(): Only returns data that would have been available at any given timestamp
No look-ahead bias - higher timeframe data is strictly from completed periods

2. Last Complete Period Method

Instead of forward-filling, uses the last complete higher timeframe period
Example: At 2:30 PM, uses 4H data from the period that ended at 12:00 PM, not the current incomplete one

3. Proper Temporal Labeling

Uses closed='left', label='right' for statistically correct interval handling
Each bar represents data from its time period, labeled at the end

4. Flexible Alignment Options

Can create aligned datasets with multiple higher timeframes
Maintains cache for efficiency when processing large datasets

5. Built-in Validation

Validates temporal integrity and checks for alignment issues
Provides metrics to ensure data quality

How to Use With Your Data:
python# Load your ETH data
eth5m = pd.read_csv('csv/tb/ETHEUR_5m.csv')
eth5m.time = pd.to_datetime(eth5m.time, unit='ms')
eth5m.set_index('time', inplace=True)

# Initialize the resampler
resampler = PointInTimeResampler(eth5m)

# Create properly aligned multi-timeframe dataset
aligned_data = resampler.create_aligned_dataset(
    target_timeframe='30min',
    higher_timeframes=['4H', '1D']
)
This eliminates the statistical issues from your original approach while maintaining all the multi-timeframe information you need. Ready to build indicators on this foundation?"""

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


# Usage example with actual ETH data
def process_data():
    """
    Process actual ETH 5-minute data using the point-in-time resampling system
    """
    # Load actual ETH data
    try:
        raw = pd.read_csv('D:/Seagull_data/historical_data/time/ETHEUR/ETHEUR_5m.csv')

        # Clean up the data as in your original script
        raw['t'] = raw.time
        raw.time = pd.to_datetime(raw.time, unit='ms')
        raw.set_index('time', inplace=True)

        # Drop unnamed columns if they exist
        if 'Unnamed: 0' in raw.columns:
            raw.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

        print(f"Loaded raw data: {raw.shape}")
        print(f"Date range: {raw.index.min()} to {raw.index.max()}")
        print(f"Columns: {list(raw.columns)}")

        # Initialize the resampler
        resampler = PointInTimeResampler(raw)

        print(f"Base timeframe detected: {resampler.base_timeframe}")

        # Create aligned dataset with proper point-in-time alignment
        aligned_data = resampler.create_aligned_dataset(
            target_timeframe='30min',
            higher_timeframes=['4H', '1D']
        )

        print(f"Aligned data shape: {aligned_data.shape}")
        print("\nAligned data columns:")
        for col in aligned_data.columns:
            print(f"  {col}")

        # Show sample of aligned data
        print(f"\nSample of aligned data (last 5 rows):")
        print(aligned_data.tail())

        # Validate alignment
        validation = resampler.validate_alignment(aligned_data)
        print(f"\nValidation results:")
        print(f"  Total rows: {validation['total_rows']}")
        print(f"  Temporal gaps: {validation['temporal_gaps']}")

        # Show null percentages for higher timeframe data
        print(f"\nNull percentages by column:")
        for col, null_pct in validation['null_percentage'].items():
            if null_pct > 0:
                print(f"  {col}: {null_pct:.2%}")

        # Demonstrate point-in-time data access for a recent timestamp
        if len(aligned_data) > 100:
            test_timestamp = aligned_data.index[-50]  # Pick a point near the end but not the very end
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
        print(resampler)
        print(aligned_data)
        return resampler, aligned_data

    except FileNotFoundError:
        print("Error: Could not find 'csv/tb/ETHEUR_5m.csv'")
        print("Please ensure the file path is correct.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


process_data()
# EVENTS ---------------------------------------------------------------------------------------------------------------
"""Volume-Price Events
1. Volume-Price Divergence Events

Price makes new highs/lows but volume doesn't confirm
More reliable than pure price events because volume shows conviction
Less noisy than Bollinger bands in sideways markets

2. Volume Shock Events

Volume spikes beyond 2-3 standard deviations of rolling average
Often precedes significant price moves
Can be combined with price acceleration for confirmation

Volatility Regime Events
3. Volatility Breakout Events

When realized volatility exceeds implied volatility by significant margin
Or when volatility moves beyond historical percentiles (95th/5th)
More fundamental than technical indicators

4. Volatility Clustering Events

Periods where volatility stays elevated for multiple periods
Based on GARCH-type modeling rather than simple rolling windows

Market Structure Events
5. Microstructure Breakdowns

When bid-ask spreads widen significantly (if you have L2 data)
Or when price gaps exceed normal distributions
Indicates liquidity stress or information asymmetry

6. Momentum Regime Changes

When short-term momentum (1-6 hours) diverges from medium-term (1-7 days)
Uses return autocorrelation changes rather than price levels
More stable across different market conditions

Statistical Events
7. Return Distribution Outliers

Events when returns exceed historical z-scores by 2+ standard deviations
Can be applied to log returns, volatility-adjusted returns, or residuals from factor models
More theoretically grounded than arbitrary technical levels

8. Correlation Breakdown Events

When ETH/BTC correlation breaks down significantly
Or when cross-timeframe correlations (5min vs 4H) diverge
Indicates regime changes or unique ETH-specific events

Multi-Asset Events
9. Cross-Asset Momentum Divergence

When ETH momentum diverges from broader crypto market
Compare ETH performance vs Bitcoin, major altcoins, or crypto index
Captures ETH-specific alpha opportunities

10. Funding Rate Events (if available)

When perp funding rates reach extreme levels
Indicates positioning imbalances that often reverse
More fundamental than pure technical analysis

My Top 3 Recommendations:

Volume-Price Divergence + Volatility Breakout: Combines price action with volume confirmation and volatility context
Return Distribution Outliers: Statistically robust, adapts to changing market conditions automatically
Multi-timeframe Momentum Regime Changes: Captures structural shifts rather than noise

The key advantage of these over cusum-bollinger is they're:

Less prone to false signals in choppy markets
More adaptable to changing market regimes
Theoretically grounded in market microstructure principles
Less curve-fitted to historical data

Which of these resonates with your trading philosophy? I can implement whichever approach interests you most."""
# THRESHOLD ------------------------------------------------------------------------------------------------------------
"""The most important questions in systematic trading. Using just trade commission as your barrier threshold is 
typically too narrow and leads to several problems:
Problems with Commission-Only Thresholds:

Noise Trading: You'll trigger on every tiny price move, most of which are just market noise
High Turnover: Excessive trading costs from frequent entries/exits
Poor Risk-Adjusted Returns: Many small wins get wiped out by occasional larger losses
Ignores Market Volatility: Same threshold in low-vol and high-vol periods makes no sense

Better Threshold Optimization Methods:
1. Volatility-Adjusted Thresholds
threshold = k * σ_t * √(holding_period)
Where k is optimized through backtesting (typically 0.5-2.0)

Adapts to current market conditions
Larger thresholds in volatile periods, smaller in calm periods

2. Information-Theoretic Approach

Set thresholds based on signal-to-noise ratio of your features
Higher thresholds when your predictive features are weak
Lower when features show strong predictive power

3. Sharpe Ratio Optimization

Run backtests across different threshold multipliers
Choose the combination that maximizes risk-adjusted returns
Typically involves testing k values from 0.25 to 3.0 in increments

4. Transaction Cost + Market Impact Model
optimal_threshold = trading_costs + α * volatility + β * illiquidity_measure

Includes bid-ask spreads, slippage, market impact
More realistic than just commission

5. Meta-Learning Approach (Advanced)

Use machine learning to predict optimal thresholds based on market regime features
Features: volatility, volume, time of day, market stress indicators
Adapts thresholds dynamically

My Recommended Approach:
Two-Stage Optimization:
Stage 1: Volatility-Based Foundation
profit_threshold = commission + k_profit * daily_volatility
stop_threshold = commission + k_stop * daily_volatility
Stage 2: Sharpe Optimization

Test k_profit values: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
Test k_stop values: [0.5, 0.75, 1.0, 1.25, 1.5]
Test asymmetric ratios (profit ≠ stop)

Key Considerations:

Asymmetric Thresholds: Often optimal to have profit_target ≠ stop_loss
Time Decay: Consider adding time-based exits (your vertical barrier)
Market Regime Awareness: Different thresholds for trending vs ranging markets
Sample Size: Ensure enough events for statistical significance

Practical Implementation:

Start with k=1.0 (1x daily volatility) as baseline
Run walk-forward optimization over your historical data
Monitor out-of-sample performance carefully
Re-optimize quarterly or when Sharpe ratio degrades significantly

The optimal threshold will likely be 2-5x your commission costs in typical crypto market conditions, 
but this varies significantly with market volatility and your holding period."""
# INDICATORS -----------------------------------------------------------------------------------------------------------
"""creating relative indicators (distance from EMA, ratios, differences) is excellent practice for ML models. This is much better than using raw price levels because:
Why Relative Indicators Are Better:

Stationarity: Raw prices are non-stationary, but price-EMA differences are more stationary
Scale Independence: Models don't get confused by absolute price levels (ETH at $1000 vs $4000)
Feature Stability: Relationships remain consistent across different market periods
Reduced Multicollinearity: Price-derived ratios are less correlated than raw prices

Your approach of Close - EMA is good, but you can make it even better.
Recommended Indicators for Lopez de Prado + Elder Framework:
1. Trend Identification (Elder's Screen 1 - Market Tide)
Multi-Timeframe Momentum Ratios:
python# Better than raw MACD - normalized by volatility
macd_ratio = (MACD_line - MACD_signal) / ATR
trend_strength = (Close - EMA_long) / (ATR * sqrt(lookback_days))
momentum_divergence = (price_momentum_short / price_momentum_long) - 1
Trend Persistence Measures:
python# Fits Lopez de Prado's emphasis on autocorrelation
trend_consistency = rolling_correlation(returns, lagged_returns)
directional_strength = abs(Close - EMA) / (High - Low)  # Trend vs noise ratio
2. Market Structure (Lopez de Prado Microstructure)
Volume-Price Relationships:
pythonvolume_price_trend = correlation(volume_changes, price_changes, window=20)
volume_efficiency = abs(price_change) / volume_normalized  # Price impact per unit volume
accumulation_distribution = cumsum((Close - Low) - (High - Close)) / (High - Low) * Volume
Volatility Structure:
python# Lopez de Prado's volatility clustering
vol_regime = current_vol / rolling_mean(vol, long_window)
vol_persistence = correlation(vol_t, vol_t-1, window=20)
realized_vs_implied = realized_vol / implied_vol  # If you have options data
3. Entry Timing (Elder's Screen 2 - Wave Direction)
Mean Reversion vs Momentum:
python# Better than raw RSI - volatility adjusted
rsi_normalized = (RSI - 50) / ATR_percentile
mean_reversion_strength = (Close - VWAP) / daily_range
momentum_vs_reversion = short_momentum / mean_reversion_indicator
Oscillator Divergences:
pythonprice_oscillator = (EMA_fast - EMA_slow) / ATR
volume_oscillator = (Volume_MA_short - Volume_MA_long) / Volume_MA_long
oscillator_divergence = price_oscillator - volume_oscillator
4. Regime Detection (Lopez de Prado Structural Breaks)
Market Regime Indicators:
python# Volatility regime changes
vol_breakout = (current_vol - vol_MA) / vol_std
correlation_breakdown = rolling_corr_btc_eth - long_term_corr_btc_eth
liquidity_stress = bid_ask_spread / price  # If available
Information Flow:
python# Lopez de Prado's information-driven events
surprise_volume = (current_volume - expected_volume) / volume_std
price_acceleration = second_derivative(log_prices)
information_ratio = abs(returns) / volume_normalized
5. Risk Management (Both Frameworks)
Dynamic Position Sizing Inputs:
python# Lopez de Prado's bet sizing
signal_confidence = abs(prediction_probability - 0.5) * 2
volatility_forecast = GARCH_volatility or exponential_weighted_vol
correlation_adjustment = 1 / (1 + avg_correlation_with_other_positions)
Specific Recommendations for Your Script:
Replace These:

Raw EMAs → EMA distance ratios: (Close - EMA) / ATR
Raw MACD → MACD efficiency: MACD / volatility
Raw RSI → RSI momentum: (RSI - 50) * momentum_direction

Add These Cross-Timeframe Features:
python# Elder's triple screen across your 30m/4H/1D structure
trend_alignment = sign(30m_trend) * sign(4H_trend) * sign(1D_trend)
momentum_cascade = (30m_momentum / 4H_momentum) - 1
volatility_cascade = 30m_vol / 4H_vol
Meta-Features (Lopez de Prado Style):
pythonfeature_importance_decay = exponential_weight(feature_age)
prediction_confidence = abs(ensemble_prediction - 0.5)
regime_stability = rolling_std(market_regime_indicator)
Implementation Priority:

Start with volatility-normalized indicators - biggest bang for buck
Add cross-timeframe momentum ratios - fits Elder perfectly
Include volume-price efficiency measures - crucial for crypto
Build regime detection features - Lopez de Prado's key insight

This approach will give you features that are:

Stationary for ML models
Theoretically grounded in both frameworks
Robust across market regimes
Computationally efficient"""