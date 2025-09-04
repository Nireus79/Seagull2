import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
import warnings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from indicators import indicated

warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', None)


class TripleBarrierLabeling:
    """
    Simplified Triple Barrier Labeling System
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with enhanced dataset from indicators script

        Args:
            data: DataFrame from indicators.py with events and technical indicators
        """
        self.data = data.copy()

        # Validate required columns
        self._validate_data()

        # Detect available event types
        self.available_events = self._detect_available_events()
        print(f"Available event types: {self.available_events}")

    def _validate_data(self):
        """Validate required columns are present"""
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_cols if col not in self.data.columns]

        if missing:
            raise ValueError(f"Missing required price columns: {missing}")

        # Check for datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except:
                raise ValueError("Index cannot be converted to DatetimeIndex")

    def _detect_available_events(self) -> List[str]:
        """Detect available event columns"""
        event_patterns = ['_event', 'any_event']
        available = []

        for col in self.data.columns:
            if any(pattern in col for pattern in event_patterns):
                if col != 'event_type':  # event_type is categorical, not boolean
                    available.append(col)

        return available

    def get_events_to_label(self, event_columns: Union[str, List[str], None] = None) -> pd.Series:
        """
        Get events that should be labeled - SIMPLIFIED VERSION

        Args:
            event_columns:
                - None: Use 'any_event' if available, otherwise all event columns
                - str: Use specific event column (e.g., 'outlier_event')
                - List[str]: Combine multiple event columns
        """

        if event_columns is None:
            # Default behavior: use any_event if available
            if 'any_event' in self.data.columns:
                print("Using 'any_event' column")
                return self.data.index[self.data['any_event'] == True]
            else:
                # Fallback: combine all available event columns
                print("No 'any_event' found, combining all event columns")
                event_columns = [col for col in self.available_events if col != 'any_event']

        # Convert single string to list
        if isinstance(event_columns, str):
            event_columns = [event_columns]

        # Combine specified event types
        event_mask = pd.Series(False, index=self.data.index)

        for event_type in event_columns:
            if event_type in self.data.columns:
                count = (self.data[event_type] == True).sum()
                print(f"Found {count} events in '{event_type}'")
                event_mask |= (self.data[event_type] == True)
            else:
                print(f"Warning: Event column '{event_type}' not found")

        total_events = event_mask.sum()
        print(f"Total events to label: {total_events}")
        return self.data.index[event_mask]

    def calculate_barriers(self,
                           events: pd.Series,
                           profit_take: float = 0.02,
                           stop_loss: float = 0.015,
                           holding_days: float = 1.0,
                           use_dynamic: bool = True,
                           volatility_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculate barriers for events - SIMPLIFIED VERSION

        Args:
            events: Event timestamps
            profit_take: Static profit take level (e.g., 0.02 = 2%)
            stop_loss: Static stop loss level (e.g., 0.015 = 1.5%)
            holding_days: Vertical barrier in days
            use_dynamic: If True, use ATR-based dynamic sizing
            volatility_multiplier: Multiplier for ATR-based barriers
        """
        barriers = pd.DataFrame(index=events)

        if use_dynamic and 'ATR_pct' in self.data.columns:
            print("Using dynamic ATR-based barriers")
            # Get ATR for dynamic sizing
            atr_values = self.data.loc[events, 'ATR_pct'] / 100  # Convert percentage to decimal

            # Dynamic barriers
            barriers['profit_take'] = atr_values * volatility_multiplier
            barriers['stop_loss'] = atr_values * (volatility_multiplier * 0.75)  # Slightly tighter stop loss

            # Ensure minimum levels
            barriers['profit_take'] = barriers['profit_take'].clip(lower=profit_take / 2)
            barriers['stop_loss'] = barriers['stop_loss'].clip(lower=stop_loss / 2)

        else:
            print("Using static barriers")
            # Static barriers
            barriers['profit_take'] = profit_take
            barriers['stop_loss'] = stop_loss

        # Calculate vertical barrier timestamps
        vertical_timedelta = pd.Timedelta(days=holding_days)
        barriers['vertical_barrier'] = events + vertical_timedelta

        # Ensure vertical barriers don't exceed data range
        max_date = self.data.index.max()
        barriers['vertical_barrier'] = barriers['vertical_barrier'].clip(upper=max_date)

        return barriers

    def apply_triple_barrier_single(self, event_info: Tuple) -> Dict:
        """Apply triple barrier method to a single event"""
        event_idx, barriers_row = event_info

        try:
            # Get price path from event start to vertical barrier
            start_price = self.data.loc[event_idx, 'Close']
            vertical_barrier = barriers_row['vertical_barrier']

            # Get the price path
            path_data = self.data.loc[event_idx:vertical_barrier]

            if len(path_data) <= 1:
                return {
                    'event_idx': event_idx,
                    'label': 0,
                    'barrier_touched': 'vertical',
                    'touch_time': vertical_barrier,
                    'return_achieved': 0.0,
                    'holding_period_hours': 0.0
                }

            # Calculate returns from entry price
            path_high_returns = (path_data['High'] / start_price) - 1
            path_low_returns = (path_data['Low'] / start_price) - 1

            # Define barriers
            profit_take_level = barriers_row['profit_take']
            stop_loss_level = -barriers_row['stop_loss']  # Negative for stop loss

            # Find first barrier touch using High/Low for intrabar precision
            profit_touches = path_high_returns >= profit_take_level
            loss_touches = path_low_returns <= stop_loss_level

            profit_touch_times = path_data.index[profit_touches]
            loss_touch_times = path_data.index[loss_touches]

            # Determine which barrier was hit first
            first_profit_touch = profit_touch_times[0] if len(profit_touch_times) > 0 else pd.NaT
            first_loss_touch = loss_touch_times[0] if len(loss_touch_times) > 0 else pd.NaT

            # Handle NaT comparisons
            if pd.isna(first_profit_touch) and pd.isna(first_loss_touch):
                # No barrier touched, vertical barrier wins
                label = 0
                barrier_touched = 'vertical'
                touch_time = vertical_barrier
                return_achieved = (path_data['Close'].iloc[-1] / start_price) - 1
            elif pd.isna(first_loss_touch) or (
                    not pd.isna(first_profit_touch) and first_profit_touch <= first_loss_touch):
                # Profit take hit first
                label = 1
                barrier_touched = 'profit_take'
                touch_time = first_profit_touch
                return_achieved = profit_take_level
            else:
                # Stop loss hit first
                label = 0
                barrier_touched = 'stop_loss'
                touch_time = first_loss_touch
                return_achieved = stop_loss_level

            # Calculate holding period
            holding_period = (touch_time - event_idx).total_seconds() / 3600  # Hours

            return {
                'event_idx': event_idx,
                'label': label,
                'barrier_touched': barrier_touched,
                'touch_time': touch_time,
                'return_achieved': return_achieved,
                'holding_period_hours': holding_period
            }

        except Exception as e:
            print(f"Error processing event {event_idx}: {e}")
            return {
                'event_idx': event_idx,
                'label': 0,
                'barrier_touched': 'error',
                'touch_time': event_idx,
                'return_achieved': 0.0,
                'holding_period_hours': 0.0
            }

    def apply_triple_barriers(self, events: pd.Series, barriers: pd.DataFrame,
                              use_parallel: bool = False) -> pd.DataFrame:
        """Apply triple barrier method to all events"""
        print(f"Applying triple barriers to {len(events)} events...")

        # Prepare event data for processing
        event_data = [(event_idx, barriers.loc[event_idx]) for event_idx in events]

        if use_parallel and len(events) > 1000:
            # Parallel processing for large datasets
            num_threads = min(4, mp.cpu_count())
            print(f"Using parallel processing with {num_threads} threads")

            with mp.Pool(processes=num_threads) as pool:
                results = list(tqdm(
                    pool.imap(self.apply_triple_barrier_single, event_data),
                    total=len(event_data),
                    desc="Processing events"
                ))
        else:
            # Sequential processing
            results = []
            for event_info in tqdm(event_data, desc="Processing events"):
                results.append(self.apply_triple_barrier_single(event_info))

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('event_idx', inplace=True)

        return results_df

    def create_labeled_dataset(self,
                               event_columns: Union[str, List[str], None] = None,
                               profit_take: float = 0.02,
                               stop_loss: float = 0.015,
                               holding_days: float = 1.0,
                               use_dynamic: bool = True,
                               volatility_multiplier: float = 2.0,
                               use_parallel: bool = False) -> pd.DataFrame:
        """
        SIMPLIFIED main function to create labeled dataset with triple barriers

        Args:
            event_columns: Which events to label (None = any_event, str = specific column, list = multiple columns)
            profit_take: Profit take level (default 2%)
            stop_loss: Stop loss level (default 1.5%)
            holding_days: Vertical barrier in days (default 1 day)
            use_dynamic: Use ATR-based dynamic barriers (default True)
            volatility_multiplier: Multiplier for ATR (default 2.0)
            use_parallel: Use parallel processing for large datasets (default False)
        """

        # Get events to label
        events = self.get_events_to_label(event_columns)

        if len(events) == 0:
            print("No events found to label!")
            return self.data.copy()

        # Calculate barriers
        barriers = self.calculate_barriers(
            events, profit_take, stop_loss, holding_days,
            use_dynamic, volatility_multiplier
        )

        # Apply triple barriers
        barrier_results = self.apply_triple_barriers(events, barriers, use_parallel)

        # Add results to original dataset
        labeled_data = self.data.copy()

        # Initialize label columns
        labeled_data['triple_barrier_label'] = np.nan
        labeled_data['barrier_touched'] = None
        labeled_data['barrier_touch_time'] = pd.NaT
        labeled_data['barrier_return'] = np.nan
        labeled_data['holding_period_hours'] = np.nan

        # Fill in the results
        for event_idx in barrier_results.index:
            result = barrier_results.loc[event_idx]
            labeled_data.loc[event_idx, 'triple_barrier_label'] = result['label']
            labeled_data.loc[event_idx, 'barrier_touched'] = result['barrier_touched']
            labeled_data.loc[event_idx, 'barrier_touch_time'] = result['touch_time']
            labeled_data.loc[event_idx, 'barrier_return'] = result['return_achieved']
            labeled_data.loc[event_idx, 'holding_period_hours'] = result['holding_period_hours']

        return labeled_data

    def get_summary(self, labeled_data: pd.DataFrame) -> Dict:
        """Generate summary statistics of the labeling process"""

        # Get labeled events only
        labeled_events = labeled_data.dropna(subset=['triple_barrier_label'])

        if len(labeled_events) == 0:
            return {"error": "No labeled events found"}

        summary = {
            'total_events_labeled': len(labeled_events),
            'label_distribution': labeled_events['triple_barrier_label'].value_counts().to_dict(),
            'barrier_hit_distribution': labeled_events['barrier_touched'].value_counts().to_dict(),
            'average_holding_period_hours': labeled_events['holding_period_hours'].mean(),
            'average_return_achieved': labeled_events['barrier_return'].mean(),
            'success_rate': (labeled_events['triple_barrier_label'] == 1).mean(),
        }

        # Add return statistics by label
        for label in [0, 1]:
            label_data = labeled_events[labeled_events['triple_barrier_label'] == label]
            if len(label_data) > 0:
                summary[f'avg_return_label_{label}'] = label_data['barrier_return'].mean()
                summary[f'avg_holding_period_label_{label}'] = label_data['holding_period_hours'].mean()

        return summary


# SIMPLIFIED USAGE FUNCTIONS

def label_events(data: pd.DataFrame,
                 event_columns: Union[str, List[str], None] = None,
                 profit_take: float = 0.02,
                 stop_loss: float = 0.015,
                 holding_days: float = 1.0,
                 use_dynamic: bool = True,
                 use_parallel: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    SUPER SIMPLE function to label events with triple barriers

    Args:
        data: Your enhanced dataset from indicators.py
        event_columns:
            - None: Use any_event (default)
            - 'outlier_event': Use only outlier events
            - ['outlier_event', 'momentum_regime_event']: Use multiple event types
        profit_take: Profit target as decimal (0.02 = 2%)
        stop_loss: Stop loss as decimal (0.015 = 1.5%)
        holding_days: How long to hold before giving up (days)
        use_dynamic: Use ATR for dynamic barrier sizing
        use_parallel: Use parallel processing (for large datasets)

    Returns:
        (labeled_dataset, summary_stats)
    """

    labeler = TripleBarrierLabeling(data)

    labeled_data = labeler.create_labeled_dataset(
        event_columns=event_columns,
        profit_take=profit_take,
        stop_loss=stop_loss,
        holding_days=holding_days,
        use_dynamic=use_dynamic,
        use_parallel=use_parallel
    )

    summary = labeler.get_summary(labeled_data)

    return labeled_data, summary


# PRESET CONFIGURATIONS for common use cases

def quick_label_outliers(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Quick labeling for outlier events with tight barriers"""
    return label_events(
        data,
        event_columns='outlier_event',
        profit_take=0.015,  # 1.5%
        stop_loss=0.01,  # 1%
        holding_days=0.5,  # 12 hours
        use_dynamic=True
    )


def quick_label_momentum(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Quick labeling for momentum events with wider barriers"""
    return label_events(
        data,
        event_columns='momentum_regime_event',
        profit_take=0.03,  # 3%
        stop_loss=0.02,  # 2%
        holding_days=2.0,  # 2 days
        use_dynamic=True
    )


def quick_label_all_events(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Quick labeling for all events"""
    return label_events(
        data,
        event_columns=None,  # Uses any_event
        use_dynamic=True
    )


# EXAMPLE USAGE
if __name__ == "__main__":
    enhanced_data = indicated

    print("=== SIMPLIFIED USAGE EXAMPLES ===\n")

    # Example 1: Just outlier events (should give you 5,853 events)
    print("1. Labeling only outlier events:")
    labeled_data, summary = label_events(
        enhanced_data,
        event_columns='outlier_event'
    )
    print(labeled_data)
    print(f"Events labeled: {summary['total_events_labeled']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Label distribution: {summary['label_distribution']}\n")

    # Example 2: Multiple specific event types
    print("2. Labeling outlier + momentum events:")
    labeled_data2, summary2 = label_events(
        enhanced_data,
        event_columns=['outlier_event', 'momentum_regime_event']
    )
    print(labeled_data2)
    print(f"Events labeled: {summary2['total_events_labeled']}")
    print(f"Success rate: {summary2['success_rate']:.1%}\n")

    # Example 3: All events (any_event column)
    print("3. Labeling all events:")
    labeled_data3, summary3 = label_events(enhanced_data)
    print(labeled_data3)
    print(f"Events labeled: {summary3['total_events_labeled']}")
    print(f"Success rate: {summary3['success_rate']:.1%}\n")

    # Example 4: Custom parameters
    print("4. Custom tight parameters for scalping:")
    labeled_data4, summary4 = label_events(
        enhanced_data,
        event_columns='vpd_volatility_event',
        profit_take=0.01,  # 1%
        stop_loss=0.008,  # 0.8%
        holding_days=0.25,  # 6 hours
        use_dynamic=True
    )
    print(labeled_data4)
    print(f"Events labeled: {summary4['total_events_labeled']}")
    print(f"Success rate: {summary4['success_rate']:.1%}")
