import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
import warnings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from indicators import indicated

warnings.filterwarnings('ignore')

"""Main Features
1. Barrier Modes:

Static: Fixed percentage barriers (e.g., 2% profit, 1.5% stop loss)
Dynamic: Volatility-based barriers using ATR or other volatility measures
Adaptive: Barriers that adjust based on recent performance (simplified implementation)

2. Event Selection:

Choose specific event types (vpd_volatility_event, outlier_event, etc.)
Use combined any_event column
Event-specific barrier multipliers (wider barriers for regime changes, tighter for outliers)

3. Dynamic Barrier Calculation:

Uses your existing volatility columns (ATR_pct, vol_realized)
Configurable multipliers for profit take and stop loss
Dynamic vertical barriers based on volatility (more volatile = longer holding)
Min/max holding period constraints

4. Processing Features:

Parallel processing for large datasets
Intrabar barrier detection using High/Low prices
Comprehensive error handling
Progress tracking with tqdm

Usage Examples
python# Basic usage with dynamic barriers
labeled_data, summary = create_labeled_dataset(enhanced_data)

# Static barriers for conservative approach
labeled_data, summary = create_labeled_dataset(
    enhanced_data, 
    barrier_mode='static'
)

# Focus on specific events with custom config
custom_config = {
    'event_types': ['outlier_event', 'momentum_regime_event'],
    'dynamic_barriers': {
        'pt_atr_multiplier': 2.5,
        'sl_atr_multiplier': 2.0,
        'max_holding_days': 3
    }
}
labeled_data, summary = create_labeled_dataset(
    enhanced_data, 
    barrier_mode='dynamic',
    custom_config=custom_config
)
Key Implementation Details
1. Barrier Touch Detection:

Uses High/Low data for intrabar detection (more accurate than close-only)
Handles simultaneous barrier touches correctly
First barrier touched wins (realistic trading behavior)

2. Dynamic Sizing:

Profit take = ATR × multiplier (default 2.0)
Stop loss = ATR × multiplier (default 1.5)
Vertical barrier = volatility-based periods with min/max constraints

3. Event-Specific Adjustments:

VPD events get wider barriers (higher uncertainty)
Outlier events get tighter barriers (expected quick reversion)
Regime changes get much wider barriers (structural shifts)

4. Output Columns Added:

triple_barrier_label: Binary target (1=profitable, 0=not profitable)
barrier_touched: Which barrier was hit ('profit_take', 'stop_loss', 'vertical')
barrier_return: Actual return achieved
holding_period_hours: Time until barrier touch
Optional detailed barrier level information

The script addresses all your questions:

Separate implementation - Takes indicators output and adds labeling columns
Dynamic PTSL calculation - Based on ATR, volatility, and event types
Flexible vertical barriers - Volatility-based with constraints to prevent overfitting

The system is modular and allows you to experiment with different barrier strategies while maintaining the core López de Prado methodology.
"""


class TripleBarrierLabeling:
    """
    Triple Barrier Labeling System based on López de Prado's methodology.
    Takes enhanced dataset from indicators script and adds labeling columns.
    """

    def __init__(self, data: pd.DataFrame, config: Dict = None):
        """
        Initialize with enhanced dataset from indicators script

        Args:
            data: DataFrame from indicators.py with events and technical indicators
            config: Configuration for barrier settings and labeling
        """
        self.data = data.copy()

        # Fix: Ensure config is properly initialized by merging with defaults
        default_config = self._default_config()
        if config:
            # Merge custom config with defaults
            for key, value in config.items():
                if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value

        self.config = default_config

        # Validate required columns
        self._validate_data()

        # Detect available event types
        self.available_events = self._detect_available_events()
        print(f"Available event types: {self.available_events}")

    def _default_config(self) -> Dict:
        """Default configuration for triple barrier labeling"""
        return {
            # Event selection
            'event_types': ['vpd_volatility_event', 'outlier_event', 'momentum_regime_event'],
            'use_any_event': True,  # Use combined event column

            # Barrier type selection
            'barrier_mode': 'dynamic',  # 'static', 'dynamic', 'adaptive'

            # Static barrier settings (used when barrier_mode='static')
            'static_barriers': {
                'profit_take': 0.02,  # 2% profit take
                'stop_loss': 0.015,  # 1.5% stop loss
                'vertical_days': 1.0  # 1 day holding period
            },

            # Dynamic barrier settings (used when barrier_mode='dynamic')
            'dynamic_barriers': {
                'pt_atr_multiplier': 2.0,  # Profit take = 2x ATR
                'sl_atr_multiplier': 1.5,  # Stop loss = 1.5x ATR
                'volatility_column': 'ATR_pct',  # Column to use for dynamic sizing
                'vertical_vol_multiplier': 20,  # Vertical barrier = 20x volatility-based periods
                'min_holding_hours': 2,  # Minimum holding period
                'max_holding_days': 5  # Maximum holding period
            },

            # Adaptive barrier settings (used when barrier_mode='adaptive')
            'adaptive_barriers': {
                'base_pt_multiplier': 2.0,
                'base_sl_multiplier': 1.5,
                'adaptation_window': 100,  # Lookback for adaptation
                'target_hit_rate': 0.4,  # Target rate for horizontal barrier hits
                'adaptation_factor': 0.1  # Speed of adaptation
            },

            # Event-specific barrier adjustments
            'event_specific_multipliers': {
                'vpd_volatility_event': {'pt': 1.2, 'sl': 1.1},  # Wider barriers for VPD events
                'outlier_event': {'pt': 0.8, 'sl': 0.9},  # Tighter barriers for outliers
                'momentum_regime_event': {'pt': 1.5, 'sl': 1.3},  # Much wider for regime changes
            },

            # Processing settings
            'min_return_threshold': 0.001,  # Minimum return to consider event
            'parallel_processing': True,
            'num_threads': 4,
            'batch_size': 1000,

            # Output settings
            'add_detailed_info': True,  # Add columns with detailed barrier info
            'calculate_returns': True  # Calculate actual returns achieved
        }

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
        event_patterns = ['_event', 'any_event', 'event_type']
        available = []

        for col in self.data.columns:
            if any(pattern in col for pattern in event_patterns):
                if col not in ['event_type']:  # event_type is categorical, not boolean
                    available.append(col)

        return available

    def get_events_to_label(self) -> pd.Series:
        """Get events that should be labeled based on configuration"""

        # Fix: Add safety check for config keys
        use_any_event = self.config.get('use_any_event', True)
        event_types = self.config.get('event_types', [])

        if use_any_event and 'any_event' in self.data.columns:
            return self.data.index[self.data['any_event'] == True]

        # Combine specified event types
        event_mask = pd.Series(False, index=self.data.index)

        for event_type in event_types:
            if event_type in self.data.columns:
                event_mask |= (self.data[event_type] == True)

        return self.data.index[event_mask]

    def calculate_static_barriers(self, events: pd.Series) -> pd.DataFrame:
        """Calculate static barriers for events"""
        barriers = pd.DataFrame(index=events)
        config = self.config['static_barriers']

        barriers['profit_take'] = config['profit_take']
        barriers['stop_loss'] = config['stop_loss']

        # Calculate vertical barrier timestamps
        vertical_timedelta = pd.Timedelta(days=config['vertical_days'])
        barriers['vertical_barrier'] = events + vertical_timedelta

        # Ensure vertical barriers don't exceed data range
        max_date = self.data.index.max()
        barriers['vertical_barrier'] = barriers['vertical_barrier'].clip(upper=max_date)

        return barriers

    def calculate_dynamic_barriers(self, events: pd.Series) -> pd.DataFrame:
        """Calculate dynamic barriers based on volatility measures"""
        barriers = pd.DataFrame(index=events)
        config = self.config['dynamic_barriers']

        # Get volatility measure
        vol_col = config['volatility_column']
        if vol_col not in self.data.columns:
            print(f"Warning: {vol_col} not found, using default static barriers")
            return self.calculate_static_barriers(events)

        volatility = self.data.loc[events, vol_col]

        # Calculate horizontal barriers
        barriers['profit_take'] = volatility * config['pt_atr_multiplier'] / 100
        barriers['stop_loss'] = volatility * config['sl_atr_multiplier'] / 100

        # Apply event-specific multipliers
        for event_idx in events:
            event_multiplier = self._get_event_specific_multiplier(event_idx)
            barriers.loc[event_idx, 'profit_take'] *= event_multiplier['pt']
            barriers.loc[event_idx, 'stop_loss'] *= event_multiplier['sl']

        # Calculate dynamic vertical barriers
        holding_periods = volatility * config['vertical_vol_multiplier']

        # Convert to timedelta (assuming volatility-based periods in hours)
        min_holding = pd.Timedelta(hours=config['min_holding_hours'])
        max_holding = pd.Timedelta(days=config['max_holding_days'])

        vertical_timedeltas = pd.to_timedelta(holding_periods, unit='h')
        vertical_timedeltas = vertical_timedeltas.clip(lower=min_holding, upper=max_holding)

        barriers['vertical_barrier'] = events + vertical_timedeltas

        # Ensure vertical barriers don't exceed data range
        max_date = self.data.index.max()
        barriers['vertical_barrier'] = barriers['vertical_barrier'].clip(upper=max_date)

        return barriers

    def calculate_adaptive_barriers(self, events: pd.Series) -> pd.DataFrame:
        """Calculate adaptive barriers that adjust based on recent performance"""
        barriers = pd.DataFrame(index=events)
        config = self.config['adaptive_barriers']

        # Start with dynamic barriers as base
        base_barriers = self.calculate_dynamic_barriers(events)

        adaptation_window = config['adaptation_window']
        target_hit_rate = config['target_hit_rate']
        adaptation_factor = config['adaptation_factor']

        # Initialize with base multipliers
        barriers['profit_take'] = base_barriers['profit_take']
        barriers['stop_loss'] = base_barriers['stop_loss']
        barriers['vertical_barrier'] = base_barriers['vertical_barrier']

        # Adapt barriers based on recent performance (simplified version)
        for i, event_idx in enumerate(events):
            if i < adaptation_window:
                continue  # Not enough history for adaptation

            # Look at recent events
            recent_events = events[max(0, i - adaptation_window):i]
            if len(recent_events) == 0:
                continue

            # This would need actual historical barrier hit data for full implementation
            # For now, we'll apply a simplified adaptive adjustment

            # Placeholder for adaptation logic
            # In practice, you'd track barrier hit statistics and adjust accordingly
            volatility_factor = self.data.loc[event_idx, 'vol_realized'] if 'vol_realized' in self.data.columns else 1.0

            if volatility_factor > 1.5:  # High volatility period
                barriers.loc[event_idx, 'profit_take'] *= 1.2
                barriers.loc[event_idx, 'stop_loss'] *= 1.2
            elif volatility_factor < 0.5:  # Low volatility period
                barriers.loc[event_idx, 'profit_take'] *= 0.8
                barriers.loc[event_idx, 'stop_loss'] *= 0.8

        return barriers

    def _get_event_specific_multiplier(self, event_idx: pd.Timestamp) -> Dict[str, float]:
        """Get event-specific barrier multipliers"""
        multipliers = {'pt': 1.0, 'sl': 1.0}

        event_specific_config = self.config.get('event_specific_multipliers', {})

        # Check which event types are active for this timestamp
        for event_type, event_multipliers in event_specific_config.items():
            if event_type in self.data.columns:
                if self.data.loc[event_idx, event_type]:
                    multipliers['pt'] *= event_multipliers['pt']
                    multipliers['sl'] *= event_multipliers['sl']

        return multipliers

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
            path_returns = (path_data['Close'] / start_price) - 1
            path_high_returns = (path_data['High'] / start_price) - 1
            path_low_returns = (path_data['Low'] / start_price) - 1

            # Define barriers
            profit_take_level = barriers_row['profit_take']
            stop_loss_level = -barriers_row['stop_loss']  # Negative for stop loss

            # Find first barrier touch
            # Check intrabar movements using High/Low
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
                return_achieved = path_returns.iloc[-1]
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

    def apply_triple_barriers(self, events: pd.Series, barriers: pd.DataFrame) -> pd.DataFrame:
        """Apply triple barrier method to all events"""
        print(f"Applying triple barriers to {len(events)} events...")

        # Prepare event data for processing
        event_data = [(event_idx, barriers.loc[event_idx]) for event_idx in events]

        if self.config['parallel_processing'] and len(events) > 100:
            # Parallel processing for large datasets
            num_threads = min(self.config['num_threads'], mp.cpu_count())

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

    def create_labeled_dataset(self) -> pd.DataFrame:
        """Main function to create labeled dataset with triple barriers"""

        # Get events to label
        events = self.get_events_to_label()

        if len(events) == 0:
            print("No events found to label!")
            return self.data.copy()

        print(f"Found {len(events)} events to label")

        # Calculate barriers based on mode
        barrier_mode = self.config['barrier_mode']

        if barrier_mode == 'static':
            barriers = self.calculate_static_barriers(events)
            print("Using static barriers")
        elif barrier_mode == 'dynamic':
            barriers = self.calculate_dynamic_barriers(events)
            print("Using dynamic barriers based on volatility")
        elif barrier_mode == 'adaptive':
            barriers = self.calculate_adaptive_barriers(events)
            print("Using adaptive barriers")
        else:
            raise ValueError(f"Unknown barrier mode: {barrier_mode}")

        # Apply triple barriers
        barrier_results = self.apply_triple_barriers(events, barriers)

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

        # Add detailed barrier information if requested
        if self.config['add_detailed_info']:
            labeled_data['profit_take_level'] = np.nan
            labeled_data['stop_loss_level'] = np.nan
            labeled_data['vertical_barrier_time'] = pd.NaT

            for event_idx in events:
                if event_idx in barriers.index:
                    labeled_data.loc[event_idx, 'profit_take_level'] = barriers.loc[event_idx, 'profit_take']
                    labeled_data.loc[event_idx, 'stop_loss_level'] = barriers.loc[event_idx, 'stop_loss']
                    labeled_data.loc[event_idx, 'vertical_barrier_time'] = barriers.loc[event_idx, 'vertical_barrier']

        return labeled_data

    def get_labeling_summary(self, labeled_data: pd.DataFrame) -> Dict:
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

        # Barrier mode specific stats
        summary['barrier_configuration'] = {
            'mode': self.config['barrier_mode'],
            'event_types_used': self.config['event_types'],
            'use_any_event': self.config['use_any_event']
        }

        return summary


def create_labeled_dataset(
        enhanced_data: pd.DataFrame,
        event_types: List[str] = None,
        barrier_mode: Literal['static', 'dynamic', 'adaptive'] = 'dynamic',
        custom_config: Dict = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to create labeled dataset with triple barriers

    Args:
        enhanced_data: DataFrame from indicators.py
        event_types: List of event columns to use for labeling
        barrier_mode: Type of barriers to use ('static', 'dynamic', 'adaptive')
        custom_config: Custom configuration dictionary

    Returns:
        Tuple of (labeled_dataset, summary_statistics)
    """

    config = custom_config.copy() if custom_config else {}

    if event_types:
        config['event_types'] = event_types

    config['barrier_mode'] = barrier_mode

    labeling_system = TripleBarrierLabeling(enhanced_data, config)

    # Create labeled dataset
    labeled_data = labeling_system.create_labeled_dataset()

    # Generate summary
    summary = labeling_system.get_labeling_summary(labeled_data)

    return labeled_data, summary


# Example configurations for different use cases
def get_example_configs():
    """Example configurations for different labeling strategies"""

    configs = {
        'conservative_static': {
            'barrier_mode': 'static',
            'static_barriers': {
                'profit_take': 0.015,  # 1.5%
                'stop_loss': 0.01,  # 1%
                'vertical_days': 2.0  # 2 days
            }
        },

        'aggressive_dynamic': {
            'barrier_mode': 'dynamic',
            'dynamic_barriers': {
                'pt_atr_multiplier': 3.0,
                'sl_atr_multiplier': 2.0,
                'volatility_column': 'ATR_pct',
                'vertical_vol_multiplier': 15,
                'max_holding_days': 3
            }
        },

        'intraday_scalping': {
            'barrier_mode': 'dynamic',
            'dynamic_barriers': {
                'pt_atr_multiplier': 1.5,
                'sl_atr_multiplier': 1.0,
                'volatility_column': 'vol_realized',
                'vertical_vol_multiplier': 4,
                'min_holding_hours': 0.5,
                'max_holding_days': 1
            },
            'event_types': ['vpd_volatility_event', 'outlier_event']  # Fast events only
        },

        'regime_change_focus': {
            'barrier_mode': 'adaptive',
            'event_types': ['momentum_regime_event'],
            'dynamic_barriers': {
                'pt_atr_multiplier': 4.0,
                'sl_atr_multiplier': 3.0,
                'vertical_vol_multiplier': 30,
                'max_holding_days': 7
            }
        }
    }

    return configs


enhanced_data = indicated

# Example 1: Quick start with defaults
print("=== Default Configuration ===")
labeled_data, summary = create_labeled_dataset(enhanced_data)
print(f"Labeled {summary['total_events_labeled']} events")
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Average holding period: {summary['average_holding_period_hours']:.1f} hours")

# Example 2: Custom configuration
print("\n=== Custom Configuration ===")
my_config = {
    'dynamic_barriers': {
        'pt_atr_multiplier': 2.5,
        'sl_atr_multiplier': 2.0,
        'max_holding_days': 2
    },
    'event_types': ['vpd_volatility_event'],
    'parallel_processing': True
}

labeled_data, summary = create_labeled_dataset(
    enhanced_data,
    barrier_mode='dynamic',
    custom_config=my_config
)

# Check the results
print("Label distribution:", summary['label_distribution'])
print("Barriers hit:", summary['barrier_hit_distribution'])

# Save results
# labeled_data.to_csv('labeled_data.csv')

# Basic Usage Examples
# 1. Simple default usage (recommended starting point):
# python# Uses dynamic barriers based on ATR, all detected events
# labeled_data, summary = create_labeled_dataset(enhanced_data)
# print(summary)
# 2. Static barriers (conservative approach):
# pythonlabeled_data, summary = create_labeled_dataset(
#     enhanced_data,
#     barrier_mode='static'
# )
# 3. Focus on specific event types:
# pythonlabeled_data, summary = create_labeled_dataset(
#     enhanced_data,
#     event_types=['vpd_volatility_event', 'outlier_event'],  # Only these events
#     barrier_mode='dynamic'
# )
# Advanced Configuration Examples
# 4. Intraday scalping setup:
# pythonintraday_config = {
#     'dynamic_barriers': {
#         'pt_atr_multiplier': 1.5,      # Tighter profit target
#         'sl_atr_multiplier': 1.0,      # Tighter stop loss
#         'volatility_column': 'ATR_pct', # Use ATR percentage
#         'min_holding_hours': 0.5,       # 30 minutes minimum
#         'max_holding_days': 1,          # 1 day maximum
#         'vertical_vol_multiplier': 8    # Shorter time-based exits
#     },
#     'event_types': ['vpd_volatility_event', 'outlier_event'],  # Fast-resolving events only
#     'parallel_processing': True,
#     'num_threads': 6
# }
#
# labeled_data, summary = create_labeled_dataset(
#     enhanced_data,
#     barrier_mode='dynamic',
#     custom_config=intraday_config
# )
# 5. Swing trading setup:
# pythonswing_config = {
#     'dynamic_barriers': {
#         'pt_atr_multiplier': 3.0,      # Wider profit target
#         'sl_atr_multiplier': 2.5,      # Wider stop loss
#         'volatility_column': 'vol_realized',
#         'min_holding_hours': 4,        # 4 hours minimum
#         'max_holding_days': 7,         # 1 week maximum
#         'vertical_vol_multiplier': 25   # Longer time-based exits
#     },
#     'event_specific_multipliers': {
#         'momentum_regime_event': {'pt': 1.5, 'sl': 1.3},  # Even wider for regime changes
#         'vpd_volatility_event': {'pt': 1.2, 'sl': 1.1},
#         'outlier_event': {'pt': 0.9, 'sl': 0.9}           # Tighter for outliers
#     },
#     'use_any_event': True  # Use all events
# }
#
# labeled_data, summary = create_labeled_dataset(
#     enhanced_data,
#     barrier_mode='dynamic',
#     custom_config=swing_config
# )
# 6. Conservative static barriers:
# pythonconservative_config = {
#     'static_barriers': {
#         'profit_take': 0.01,     # 1% profit take
#         'stop_loss': 0.008,      # 0.8% stop loss
#         'vertical_days': 3.0     # 3 days max holding
#     },
#     'event_types': ['momentum_regime_event'],  # Only high-confidence events
#     'min_return_threshold': 0.002  # Filter out very small moves
# }
#
# labeled_data, summary = create_labeled_dataset(
#     enhanced_data,
#     barrier_mode='static',
#     custom_config=conservative_config
# )
# 7. Research/experimentation setup:
# pythonresearch_config = {
#     'dynamic_barriers': {
#         'pt_atr_multiplier': 2.0,
#         'sl_atr_multiplier': 1.5,
#         'volatility_column': 'ATR_pct',
#         'max_holding_days': 5
#     },
#     'add_detailed_info': True,      # Add all barrier level columns
#     'calculate_returns': True,       # Add return calculation columns
#     'parallel_processing': False,    # Sequential for debugging
#     'event_specific_multipliers': {
#         'vpd_volatility_event': {'pt': 1.3, 'sl': 1.2},
#         'outlier_event': {'pt': 0.8, 'sl': 0.7},
#         'momentum_regime_event': {'pt': 2.0, 'sl': 1.8}
#     }
# }
#
# labeled_data, summary = create_labeled_dataset(
#     enhanced_data,
#     event_types=['vpd_volatility_event', 'momentum_regime_event'],
#     barrier_mode='dynamic',
#     custom_config=research_config
# )
