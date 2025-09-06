import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
import warnings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from indicators import indicated

warnings.filterwarnings('ignore')


class MultiEventTripleBarrierLabeling:
    """
    Enhanced Triple Barrier Labeling System with Multi-Event Support
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with enhanced dataset from indicators script

        Args:
            data: DataFrame from indicators.py with events and technical indicators
        """
        self.data = data.copy()
        self._validate_data()
        self.available_events = self._detect_available_events()
        print(f"Available event types: {self.available_events}")

    def _validate_data(self):
        """Validate required columns are present"""
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_cols if col not in self.data.columns]

        if missing:
            raise ValueError(f"Missing required price columns: {missing}")

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
                if col != 'event_type':
                    available.append(col)

        return available

    def get_events_for_labeling(self,
                                event_selection: Union[str, List[str], Dict[str, Union[str, List[str]]]],
                                mode: Literal['individual', 'combined', 'simultaneous'] = 'individual') -> Dict[
        str, pd.Series]:
        """
        Get events for labeling with flexible selection modes

        Args:
            event_selection:
                - str: Single event type (e.g., 'outlier_event')
                - List[str]: Multiple events for combined or individual processing
                - Dict: Custom event combinations {label_name: event_columns}
            mode:
                - 'individual': Label each event type separately
                - 'combined': Merge all events into one label
                - 'simultaneous': Only label when multiple events occur together

        Returns:
            Dict mapping label names to event indices
        """
        result = {}

        if isinstance(event_selection, str):
            # Single event type
            if event_selection in self.data.columns:
                events = self.data.index[self.data[event_selection] == True]
                result[f"{event_selection}_label"] = events
                print(f"Found {len(events)} events in '{event_selection}'")
            else:
                print(f"Warning: Event column '{event_selection}' not found")

        elif isinstance(event_selection, list):
            if mode == 'individual':
                # Each event type gets its own label
                for event_type in event_selection:
                    if event_type in self.data.columns:
                        events = self.data.index[self.data[event_type] == True]
                        result[f"{event_type}_label"] = events
                        print(f"Found {len(events)} events in '{event_type}'")
                    else:
                        print(f"Warning: Event column '{event_type}' not found")

            elif mode == 'combined':
                # Combine all events into one label
                event_mask = pd.Series(False, index=self.data.index)
                valid_events = []

                for event_type in event_selection:
                    if event_type in self.data.columns:
                        count = (self.data[event_type] == True).sum()
                        print(f"Found {count} events in '{event_type}'")
                        event_mask |= (self.data[event_type] == True)
                        valid_events.append(event_type)
                    else:
                        print(f"Warning: Event column '{event_type}' not found")

                if valid_events:
                    label_name = "_".join([e.replace('_event', '') for e in valid_events]) + "_combined_label"
                    result[label_name] = self.data.index[event_mask]
                    print(f"Total combined events: {event_mask.sum()}")

            elif mode == 'simultaneous':
                # Only events that occur simultaneously
                if len(event_selection) < 2:
                    raise ValueError("Simultaneous mode requires at least 2 event types")

                # Find simultaneous events
                simultaneous_mask = pd.Series(True, index=self.data.index)
                valid_events = []

                for event_type in event_selection:
                    if event_type in self.data.columns:
                        simultaneous_mask &= (self.data[event_type] == True)
                        valid_events.append(event_type)
                    else:
                        print(f"Warning: Event column '{event_type}' not found")
                        simultaneous_mask = pd.Series(False, index=self.data.index)
                        break

                if valid_events and simultaneous_mask.sum() > 0:
                    label_name = "_".join([e.replace('_event', '') for e in valid_events]) + "_simultaneous_label"
                    result[label_name] = self.data.index[simultaneous_mask]
                    print(f"Found {simultaneous_mask.sum()} simultaneous events")
                else:
                    print("No simultaneous events found")

        elif isinstance(event_selection, dict):
            # Custom combinations
            for label_name, event_columns in event_selection.items():
                if isinstance(event_columns, str):
                    event_columns = [event_columns]

                event_mask = pd.Series(False, index=self.data.index)
                valid_events = []

                for event_type in event_columns:
                    if event_type in self.data.columns:
                        event_mask |= (self.data[event_type] == True)
                        valid_events.append(event_type)
                    else:
                        print(f"Warning: Event column '{event_type}' not found")

                if valid_events:
                    result[label_name] = self.data.index[event_mask]
                    print(f"Custom label '{label_name}': {event_mask.sum()} events")

        return result

    def calculate_barriers_for_events(self,
                                      events_dict: Dict[str, pd.Series],
                                      barrier_params: Dict[str, Dict] = None,
                                      default_params: Dict = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate barriers for multiple event types with different parameters

        Args:
            events_dict: Dictionary mapping label names to event indices
            barrier_params: Custom parameters per event type
            default_params: Default parameters for all event types
        """
        if default_params is None:
            default_params = {
                'profit_take': 0.02,
                'stop_loss': 0.015,
                'holding_days': 1.0,
                'use_dynamic': True,
                'volatility_multiplier': 2.0
            }

        if barrier_params is None:
            barrier_params = {}

        barriers_dict = {}

        for label_name, events in events_dict.items():
            if len(events) == 0:
                continue

            # Get parameters for this event type
            params = default_params.copy()
            if label_name in barrier_params:
                params.update(barrier_params[label_name])

            barriers = pd.DataFrame(index=events)

            # Dynamic or static barriers
            if params['use_dynamic'] and 'ATR_pct' in self.data.columns:
                print(f"Using dynamic barriers for {label_name}")
                atr_values = self.data.loc[events, 'ATR_pct'] / 100

                barriers['profit_take'] = atr_values * params['volatility_multiplier']
                barriers['stop_loss'] = atr_values * (params['volatility_multiplier'] * 0.75)

                # Minimum levels
                barriers['profit_take'] = barriers['profit_take'].clip(lower=params['profit_take'] / 2)
                barriers['stop_loss'] = barriers['stop_loss'].clip(lower=params['stop_loss'] / 2)
            else:
                print(f"Using static barriers for {label_name}")
                barriers['profit_take'] = params['profit_take']
                barriers['stop_loss'] = params['stop_loss']

            # Vertical barriers
            vertical_timedelta = pd.Timedelta(days=params['holding_days'])
            barriers['vertical_barrier'] = events + vertical_timedelta

            max_date = self.data.index.max()
            barriers['vertical_barrier'] = barriers['vertical_barrier'].clip(upper=max_date)

            barriers_dict[label_name] = barriers

        return barriers_dict

    def apply_triple_barrier_single(self, event_info: Tuple) -> Dict:
        """Apply triple barrier method to a single event"""
        event_idx, barriers_row, label_name = event_info

        try:
            start_price = self.data.loc[event_idx, 'Close']
            vertical_barrier = barriers_row['vertical_barrier']
            path_data = self.data.loc[event_idx:vertical_barrier]

            if len(path_data) <= 1:
                return {
                    'event_idx': event_idx,
                    'label_name': label_name,
                    'label': 0,
                    'barrier_touched': 'vertical',
                    'touch_time': vertical_barrier,
                    'return_achieved': 0.0,
                    'holding_period_hours': 0.0
                }

            # Calculate returns
            path_high_returns = (path_data['High'] / start_price) - 1
            path_low_returns = (path_data['Low'] / start_price) - 1

            # Barriers
            profit_take_level = barriers_row['profit_take']
            stop_loss_level = -barriers_row['stop_loss']

            # Find touches
            profit_touches = path_high_returns >= profit_take_level
            loss_touches = path_low_returns <= stop_loss_level

            profit_touch_times = path_data.index[profit_touches]
            loss_touch_times = path_data.index[loss_touches]

            first_profit_touch = profit_touch_times[0] if len(profit_touch_times) > 0 else pd.NaT
            first_loss_touch = loss_touch_times[0] if len(loss_touch_times) > 0 else pd.NaT

            # Determine outcome
            if pd.isna(first_profit_touch) and pd.isna(first_loss_touch):
                label = 0
                barrier_touched = 'vertical'
                touch_time = vertical_barrier
                return_achieved = (path_data['Close'].iloc[-1] / start_price) - 1
            elif pd.isna(first_loss_touch) or (
                    not pd.isna(first_profit_touch) and first_profit_touch <= first_loss_touch):
                label = 1
                barrier_touched = 'profit_take'
                touch_time = first_profit_touch
                return_achieved = profit_take_level
            else:
                label = 0
                barrier_touched = 'stop_loss'
                touch_time = first_loss_touch
                return_achieved = stop_loss_level

            holding_period = (touch_time - event_idx).total_seconds() / 3600

            return {
                'event_idx': event_idx,
                'label_name': label_name,
                'label': label,
                'barrier_touched': barrier_touched,
                'touch_time': touch_time,
                'return_achieved': return_achieved,
                'holding_period_hours': holding_period
            }

        except Exception as e:
            print(f"Error processing event {event_idx} for {label_name}: {e}")
            return {
                'event_idx': event_idx,
                'label_name': label_name,
                'label': 0,
                'barrier_touched': 'error',
                'touch_time': event_idx,
                'return_achieved': 0.0,
                'holding_period_hours': 0.0
            }

    def apply_multi_barriers(self, events_dict: Dict[str, pd.Series],
                             barriers_dict: Dict[str, pd.DataFrame],
                             use_parallel: bool = False) -> Dict[str, pd.DataFrame]:
        """Apply triple barriers to multiple event types"""
        results_dict = {}

        for label_name in events_dict.keys():
            if label_name not in barriers_dict:
                continue

            events = events_dict[label_name]
            barriers = barriers_dict[label_name]

            print(f"Processing {len(events)} events for {label_name}...")

            event_data = [(event_idx, barriers.loc[event_idx], label_name) for event_idx in events]

            if use_parallel and len(events) > 1000:
                num_threads = min(4, mp.cpu_count())
                with mp.Pool(processes=num_threads) as pool:
                    results = list(tqdm(
                        pool.imap(self.apply_triple_barrier_single, event_data),
                        total=len(event_data),
                        desc=f"Processing {label_name}"
                    ))
            else:
                results = []
                for event_info in tqdm(event_data, desc=f"Processing {label_name}"):
                    results.append(self.apply_triple_barrier_single(event_info))

            results_df = pd.DataFrame(results)
            results_df.set_index('event_idx', inplace=True)
            results_dict[label_name] = results_df

        return results_dict

    def create_multi_labeled_dataset(self,
                                     event_selection: Union[str, List[str], Dict[str, Union[str, List[str]]]],
                                     mode: Literal['individual', 'combined', 'simultaneous'] = 'individual',
                                     barrier_params: Dict[str, Dict] = None,
                                     default_params: Dict = None,
                                     use_parallel: bool = False) -> pd.DataFrame:
        """
        Main function to create multi-labeled dataset

        Args:
            event_selection: Events to label (str, list, or dict)
            mode: How to handle multiple events ('individual', 'combined', 'simultaneous')
            barrier_params: Custom parameters per event type
            default_params: Default parameters for all events
            use_parallel: Use parallel processing
        """

        # Get events for labeling
        events_dict = self.get_events_for_labeling(event_selection, mode)

        if not events_dict:
            print("No events found to label!")
            return self.data.copy()

        # Calculate barriers
        barriers_dict = self.calculate_barriers_for_events(events_dict, barrier_params, default_params)

        # Apply barriers
        results_dict = self.apply_multi_barriers(events_dict, barriers_dict, use_parallel)

        # Create labeled dataset
        labeled_data = self.data.copy()

        # Add columns for each label type
        for label_name in results_dict.keys():
            labeled_data[label_name] = np.nan
            labeled_data[f"{label_name.replace('_label', '')}_barrier_touched"] = None
            labeled_data[f"{label_name.replace('_label', '')}_touch_time"] = pd.NaT
            labeled_data[f"{label_name.replace('_label', '')}_return"] = np.nan
            labeled_data[f"{label_name.replace('_label', '')}_holding_hours"] = np.nan

            # Fill results
            results = results_dict[label_name]
            for event_idx in results.index:
                result = results.loc[event_idx]
                labeled_data.loc[event_idx, label_name] = result['label']
                labeled_data.loc[event_idx, f"{label_name.replace('_label', '')}_barrier_touched"] = result[
                    'barrier_touched']
                labeled_data.loc[event_idx, f"{label_name.replace('_label', '')}_touch_time"] = result['touch_time']
                labeled_data.loc[event_idx, f"{label_name.replace('_label', '')}_return"] = result['return_achieved']
                labeled_data.loc[event_idx, f"{label_name.replace('_label', '')}_holding_hours"] = result[
                    'holding_period_hours']

        return labeled_data

    def get_multi_summary(self, labeled_data: pd.DataFrame) -> Dict:
        """Generate summary for all label types"""
        summary = {}

        # Find all label columns
        label_columns = [col for col in labeled_data.columns if col.endswith('_label')]

        for label_col in label_columns:
            event_type = label_col.replace('_label', '')
            labeled_events = labeled_data.dropna(subset=[label_col])

            if len(labeled_events) == 0:
                continue

            summary[event_type] = {
                'total_events_labeled': len(labeled_events),
                'label_distribution': labeled_events[label_col].value_counts().to_dict(),
                'success_rate': (labeled_events[label_col] == 1).mean(),
                'average_holding_period_hours': labeled_events[f"{event_type}_holding_hours"].mean(),
                'average_return_achieved': labeled_events[f"{event_type}_return"].mean(),
            }

            # Barrier hit distribution
            barrier_col = f"{event_type}_barrier_touched"
            if barrier_col in labeled_data.columns:
                summary[event_type]['barrier_hit_distribution'] = labeled_events[barrier_col].value_counts().to_dict()

        return summary


# SIMPLIFIED USAGE FUNCTIONS

def label_multiple_events(data: pd.DataFrame,
                          event_selection: Union[str, List[str], Dict[str, Union[str, List[str]]]],
                          mode: Literal['individual', 'combined', 'simultaneous'] = 'individual',
                          barrier_params: Dict[str, Dict] = None,
                          default_params: Dict = None,
                          use_parallel: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced function to label multiple events with flexible options

    Examples:
    --------
    # Individual labeling
    labeled_data, summary = label_multiple_events(
        data,
        ['outlier_event', 'momentum_regime_event'],
        mode='individual'
    )

    # Combined labeling
    labeled_data, summary = label_multiple_events(
        data,
        ['outlier_event', 'momentum_regime_event'],
        mode='combined'
    )

    # Simultaneous events only
    labeled_data, summary = label_multiple_events(
        data,
        ['outlier_event', 'momentum_regime_event'],
        mode='simultaneous'
    )

    # Custom combinations
    labeled_data, summary = label_multiple_events(
        data,
        {
            'high_vol_events': ['outlier_event', 'vpd_volatility_event'],
            'momentum_events': 'momentum_regime_event'
        }
    )

    # Different parameters per event type
    barrier_params = {
        'outlier_event_label': {'profit_take': 0.015, 'holding_days': 0.5},
        'momentum_regime_event_label': {'profit_take': 0.03, 'holding_days': 2.0}
    }
    labeled_data, summary = label_multiple_events(
        data,
        ['outlier_event', 'momentum_regime_event'],
        mode='individual',
        barrier_params=barrier_params
    )
    """

    if default_params is None:
        default_params = {
            'profit_take': 0.02,
            'stop_loss': 0.015,
            'holding_days': 1.0,
            'use_dynamic': True,
            'volatility_multiplier': 2.0
        }

    labeler = MultiEventTripleBarrierLabeling(data)

    labeled_data = labeler.create_multi_labeled_dataset(
        event_selection=event_selection,
        mode=mode,
        barrier_params=barrier_params,
        default_params=default_params,
        use_parallel=use_parallel
    )

    summary = labeler.get_multi_summary(labeled_data)

    return labeled_data, summary


# PRESET CONFIGURATIONS

def label_all_events_individually(data: pd.DataFrame,
                                  custom_params: Dict[str, Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """Label all available events individually with optional custom parameters"""
    labeler = MultiEventTripleBarrierLabeling(data)
    available_events = labeler.available_events

    return label_multiple_events(
        data,
        available_events,
        mode='individual',
        barrier_params=custom_params
    )


def label_high_confidence_combinations(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Label events that occur simultaneously (high confidence signals)"""
    return label_multiple_events(
        data,
        ['outlier_event', 'momentum_regime_event', 'vpd_volatility_event'],
        mode='simultaneous'
    )


def label_custom_strategies(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Example of custom event strategies with different parameters"""

    custom_combinations = {
        'scalping_events': ['outlier_event'],
        'swing_events': ['momentum_regime_event'],
        'volatility_breakout': ['vpd_volatility_event'],
        'high_confidence_combo': ['outlier_event', 'momentum_regime_event']
    }

    custom_params = {
        'scalping_events': {
            'profit_take': 0.01,
            'stop_loss': 0.008,
            'holding_days': 0.25,
            'volatility_multiplier': 1.5
        },
        'swing_events': {
            'profit_take': 0.03,
            'stop_loss': 0.02,
            'holding_days': 3.0,
            'volatility_multiplier': 2.5
        },
        'volatility_breakout': {
            'profit_take': 0.025,
            'stop_loss': 0.015,
            'holding_days': 1.5,
            'volatility_multiplier': 3.0
        },
        'high_confidence_combo': {
            'profit_take': 0.04,
            'stop_loss': 0.025,
            'holding_days': 2.0,
            'volatility_multiplier': 2.0
        }
    }

    return label_multiple_events(
        data,
        custom_combinations,
        barrier_params=custom_params
    )


# EXAMPLE USAGE
if __name__ == "__main__":
    enhanced_data = indicated

    print("=== MULTI-EVENT LABELING EXAMPLES ===\n")

    # Example 1: Individual event labeling
    print("1. Individual event labeling:")
    labeled_data1, summary1 = label_multiple_events(
        enhanced_data,
        ['outlier_event', 'momentum_regime_event'],
        mode='individual'
    )
    print("Summary:", summary1)
    print()

    # Example 2: Combined events
    print("2. Combined event labeling:")
    labeled_data2, summary2 = label_multiple_events(
        enhanced_data,
        ['outlier_event', 'momentum_regime_event'],
        mode='combined'
    )
    print("Summary:", summary2)
    print()

    # Example 3: Simultaneous events only
    print("3. Simultaneous events labeling:")
    labeled_data3, summary3 = label_multiple_events(
        enhanced_data,
        ['outlier_event', 'momentum_regime_event'],
        mode='simultaneous'
    )
    print("Summary:", summary3)
    print()

    # Example 4: Custom combinations with different parameters
    print("4. Custom strategies:")
    labeled_data4, summary4 = label_custom_strategies(enhanced_data)
    print("Summary:", summary4)
    print()

    # Example 5: All events individually
    print("5. All events individually:")
    labeled_data5, summary5 = label_all_events_individually(enhanced_data)
    print(labeled_data5)
    print("Summary:", summary5)

"""1. Multiple Labeling Modes:

individual: Each event type gets its own separate label column
combined: Multiple events merged into one label using OR logic
simultaneous: Only label when multiple events occur at the same timestamp

2. Flexible Event Selection:

Single event: 'outlier_event'
Multiple events: ['outlier_event', 'momentum_regime_event']
Custom combinations: {'scalping_events': ['outlier_event'], 'swing_events': ['momentum_regime_event']}

3. Per-Event Custom Parameters:
You can specify different barrier parameters for each event type:
pythonbarrier_params = {
    'outlier_event_label': {'profit_take': 0.015, 'holding_days': 0.5},
    'momentum_regime_event_label': {'profit_take': 0.03, 'holding_days': 2.0}
}
4. Enhanced Output:
Instead of one triple_barrier_label column, you now get separate columns for each event type:

outlier_event_label
momentum_regime_event_label
outlier_barrier_touched
momentum_barrier_touched
etc.

Usage Examples
Individual Labeling (most common):
pythonlabeled_data, summary = label_multiple_events(
    data, 
    ['outlier_event', 'momentum_regime_event'], 
    mode='individual'
)
Simultaneous Events Only (high confidence):
pythonlabeled_data, summary = label_multiple_events(
    data, 
    ['outlier_event', 'momentum_regime_event'], 
    mode='simultaneous'  # Only when both occur together
)
Custom Strategies with Different Parameters:
pythoncustom_combinations = {
    'scalping_events': ['outlier_event'],
    'swing_events': ['momentum_regime_event']
}

custom_params = {
    'scalping_events': {'profit_take': 0.01, 'holding_days': 0.25},
    'swing_events': {'profit_take': 0.03, 'holding_days': 2.0}
}

labeled_data, summary = label_multiple_events(
    data, 
    custom_combinations,
    barrier_params=custom_params
)"""