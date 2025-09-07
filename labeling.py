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
    Enhanced Triple Barrier Labeling System with Multi-Event Support and Theory-Based Barriers
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

    def calculate_theory_based_barriers(self,
                                        events_dict: Dict[str, pd.Series],
                                        barrier_params: Dict[str, Dict] = None,
                                        default_params: Dict = None,
                                        commission_structure: Dict = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate theory-based barriers with commission awareness

        Args:
            events_dict: Dictionary mapping label names to event indices
            barrier_params: Custom parameters per event type
            default_params: Default parameters for all event types
            commission_structure: Commission and spread parameters
        """
        if default_params is None:
            default_params = {
                'volatility_multiplier': 2.0,
                'stop_loss_multiplier': 1.2,
                'holding_days': 1.0,
                'use_theory_based': True,
                'min_reward_risk_ratio': 1.5,
                'volume_adjustment': True,
                'regime_awareness': True
            }

        if commission_structure is None:
            commission_structure = self._get_kraken_commission_structure()

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

            if params['use_theory_based']:
                print(f"Using theory-based barriers for {label_name}")
                barriers = self._calculate_theory_barriers(events, params, commission_structure)
            else:
                print(f"Using simple barriers for {label_name}")
                barriers = self._calculate_simple_barriers(events, params, commission_structure)

            # Vertical barriers
            vertical_timedelta = pd.Timedelta(days=params['holding_days'])
            barriers['vertical_barrier'] = events + vertical_timedelta

            max_date = self.data.index.max()
            barriers['vertical_barrier'] = barriers['vertical_barrier'].clip(upper=max_date)

            barriers_dict[label_name] = barriers

        return barriers_dict

    def _get_kraken_commission_structure(self) -> Dict:
        """Kraken commission structure (as of 2025) - Taker fees only"""
        return {
            'taker_fee': 0.0026,  # 0.26% for taker orders (standard trading)
            'spread_estimate': 0.0005,  # ~0.05% typical spread for major pairs
            'slippage_estimate': 0.0003,  # ~0.03% slippage estimate
            'funding_cost_daily': 0.00005,  # ~0.005% daily funding if held overnight
            'min_profit_multiplier': 2.5,  # Minimum profit should be 2.5x total costs
        }

    def _calculate_theory_barriers(self, events: pd.Series, params: Dict, commission: Dict) -> pd.DataFrame:
        """Calculate theory-based adaptive barriers"""
        barriers = pd.DataFrame(index=events)

        for event_idx in events:
            # Get market state at event time
            current_vol = self._get_volatility_at_event(event_idx)
            current_price = self.data.loc[event_idx, 'Close']
            liquidity_factor = self._get_liquidity_factor(event_idx) if params['volume_adjustment'] else 1.0
            regime_multiplier = self._get_regime_multiplier(event_idx) if params['regime_awareness'] else 1.0

            # Calculate minimum profitable barriers (commission-aware)
            min_profit_barrier = self._calculate_min_profit_barrier(current_price, commission)

            # Theory-based volatility barriers
            base_profit_barrier = current_vol * params['volatility_multiplier'] * regime_multiplier * liquidity_factor
            base_stop_barrier = current_vol * params['stop_loss_multiplier'] * regime_multiplier * liquidity_factor

            # Ensure barriers meet minimum profitability
            profit_take = max(min_profit_barrier, base_profit_barrier)
            stop_loss = max(min_profit_barrier * 0.6, base_stop_barrier)

            # Ensure minimum reward/risk ratio
            if profit_take / stop_loss < params['min_reward_risk_ratio']:
                profit_take = stop_loss * params['min_reward_risk_ratio']

            barriers.loc[event_idx, 'profit_take'] = profit_take
            barriers.loc[event_idx, 'stop_loss'] = stop_loss

        return barriers

    def _calculate_simple_barriers(self, events: pd.Series, params: Dict, commission: Dict) -> pd.DataFrame:
        """Calculate simple static barriers with commission awareness"""
        barriers = pd.DataFrame(index=events)

        # Calculate minimum profitable barriers (taker fees only)
        min_profit_barrier = commission['taker_fee'] * 2 + commission['spread_estimate'] + commission[
            'slippage_estimate']
        min_profit_barrier *= commission['min_profit_multiplier']

        # Static barriers with minimum profit enforcement
        profit_take = max(min_profit_barrier, params.get('profit_take', 0.02))
        stop_loss = max(min_profit_barrier * 0.6, params.get('stop_loss', 0.015))

        # Ensure minimum reward/risk ratio
        if profit_take / stop_loss < params['min_reward_risk_ratio']:
            profit_take = stop_loss * params['min_reward_risk_ratio']

        barriers['profit_take'] = profit_take
        barriers['stop_loss'] = stop_loss

        return barriers

    def _get_volatility_at_event(self, event_idx) -> float:
        """Get volatility measure at event time"""
        if 'ATR_pct' in self.data.columns:
            return self.data.loc[event_idx, 'ATR_pct'] / 100
        else:
            # Fallback: use rolling standard deviation
            lookback_period = 20
            end_idx = self.data.index.get_loc(event_idx)
            start_idx = max(0, end_idx - lookback_period)

            price_data = self.data.iloc[start_idx:end_idx + 1]['Close']
            returns = price_data.pct_change().dropna()
            return returns.std() if len(returns) > 5 else 0.02  # 2% fallback

    def _get_liquidity_factor(self, event_idx) -> float:
        """Adjust barriers based on liquidity (volume)"""
        if 'Volume' not in self.data.columns:
            return 1.0

        current_volume = self.data.loc[event_idx, 'Volume']

        # Calculate average volume over recent period
        lookback_period = 50
        end_idx = self.data.index.get_loc(event_idx)
        start_idx = max(0, end_idx - lookback_period)

        avg_volume = self.data.iloc[start_idx:end_idx + 1]['Volume'].mean()

        if avg_volume == 0:
            return 1.0

        volume_ratio = current_volume / avg_volume

        # Lower liquidity = wider barriers (higher multiplier)
        if volume_ratio < 0.5:  # Low liquidity
            return 1.4
        elif volume_ratio > 2.0:  # High liquidity
            return 0.8
        else:  # Normal liquidity
            return 1.0

    def _get_regime_multiplier(self, event_idx) -> float:
        """Adjust barriers based on volatility regime"""
        current_vol = self._get_volatility_at_event(event_idx)

        # Calculate long-term average volatility
        lookback_period = 100
        end_idx = self.data.index.get_loc(event_idx)
        start_idx = max(0, end_idx - lookback_period)

        # Get historical volatility
        historical_vols = []
        for i in range(start_idx, end_idx):
            hist_idx = self.data.index[i]
            vol = self._get_volatility_at_event(hist_idx)
            historical_vols.append(vol)

        if len(historical_vols) < 10:
            return 1.0

        avg_vol = np.mean(historical_vols)
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # High volatility regime = wider barriers
        if vol_ratio > 1.5:  # High volatility
            return 1.3
        elif vol_ratio < 0.7:  # Low volatility
            return 0.8
        else:  # Normal volatility
            return 1.0

    def _calculate_min_profit_barrier(self, current_price: float, commission: Dict) -> float:
        """Calculate minimum profit needed to overcome all trading costs (taker fees only)"""
        # Round-trip costs using taker fees for both entry and exit (standard trading)
        entry_cost = commission['taker_fee']  # Market order entry
        exit_cost = commission['taker_fee']  # Market order exit (standard approach)
        spread_cost = commission['spread_estimate']
        slippage_cost = commission['slippage_estimate']

        # Total round-trip cost
        total_cost = entry_cost + exit_cost + spread_cost + slippage_cost

        # Add buffer for overnight funding if position held
        funding_buffer = commission['funding_cost_daily']

        # Minimum profit with safety multiplier
        min_profit = (total_cost + funding_buffer) * commission['min_profit_multiplier']

        return min_profit

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

        # Calculate theory-based barriers with commission awareness
        barriers_dict = self.calculate_theory_based_barriers(events_dict, barrier_params, default_params)

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

def label_multiple_events_theory_based(data: pd.DataFrame,
                                       event_selection: Union[str, List[str], Dict[str, Union[str, List[str]]]],
                                       mode: Literal['individual', 'combined', 'simultaneous'] = 'individual',
                                       barrier_params: Dict[str, Dict] = None,
                                       default_params: Dict = None,
                                       commission_structure: Dict = None,
                                       use_parallel: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Theory-based multi-event labeling with Kraken commission structure

    Args:
        data: Your enhanced dataset from indicators.py
        event_selection: Events to label (str, list, or dict)
        mode: How to handle multiple events ('individual', 'combined', 'simultaneous')
        barrier_params: Custom parameters per event type
        default_params: Default theory-based parameters
        commission_structure: Trading costs (defaults to Kraken structure)
        use_parallel: Use parallel processing

    Examples:
    --------
    # Theory-based with Kraken commissions
    labeled_data, summary = label_multiple_events_theory_based(
        data,
        ['outlier_event', 'momentum_regime_event'],
        mode='individual'
    )

    # Custom commission structure (e.g., for different exchange)
    custom_commission = {
        'taker_fee': 0.002,     # 0.2%
        'spread_estimate': 0.0003,
        'slippage_estimate': 0.0002,
        'min_profit_multiplier': 3.0
    }

    labeled_data, summary = label_multiple_events_theory_based(
        data,
        'outlier_event',
        commission_structure=custom_commission
    )

    # Event-specific theory parameters
    barrier_params = {
        'outlier_event_label': {
            'volatility_multiplier': 1.5,      # Tighter for scalping
            'min_reward_risk_ratio': 2.0,      # 2:1 minimum
            'volume_adjustment': True,          # Adjust for liquidity
            'regime_awareness': True            # Adjust for vol regime
        },
        'momentum_regime_event_label': {
            'volatility_multiplier': 2.5,      # Wider for swing trades
            'min_reward_risk_ratio': 1.8,
            'holding_days': 2.0                # Longer holding period
        }
    }

    labeled_data, summary = label_multiple_events_theory_based(
        data,
        ['outlier_event', 'momentum_regime_event'],
        mode='individual',
        barrier_params=barrier_params
    )
    """

    if default_params is None:
        default_params = {
            'volatility_multiplier': 2.0,  # ATR multiplier for profit take
            'stop_loss_multiplier': 1.2,  # ATR multiplier for stop loss
            'holding_days': 1.0,  # Maximum holding period
            'use_theory_based': True,  # Use theory-based barriers
            'min_reward_risk_ratio': 1.5,  # Minimum profit/loss ratio
            'volume_adjustment': True,  # Adjust for liquidity
            'regime_awareness': True  # Adjust for volatility regime
        }

    if commission_structure is None:
        # Default to Kraken structure (taker fees only)
        commission_structure = {
            'taker_fee': 0.0026,  # 0.26% (standard trading)
            'spread_estimate': 0.0005,  # ~0.05%
            'slippage_estimate': 0.0003,  # ~0.03%
            'funding_cost_daily': 0.00005,  # ~0.005% daily
            'min_profit_multiplier': 2.5  # 2.5x costs minimum
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

    # Add commission analysis to summary
    summary['commission_analysis'] = {
        'total_round_trip_cost': (commission_structure['taker_fee'] * 2 +
                                  commission_structure['spread_estimate'] +
                                  commission_structure['slippage_estimate']),
        'min_profit_required': (commission_structure['taker_fee'] * 2 +
                                commission_structure['spread_estimate'] +
                                commission_structure['slippage_estimate']) * commission_structure[
                                   'min_profit_multiplier'],
        'daily_funding_cost': commission_structure['funding_cost_daily'],
        'commission_structure_used': commission_structure
    }

    return labeled_data, summary


# ENHANCED PRESET CONFIGURATIONS with Theory-Based Barriers

def label_kraken_scalping_strategy(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Scalping strategy optimized for Kraken commission structure"""

    barrier_params = {
        'outlier_event_label': {
            'volatility_multiplier': 1.5,  # Tighter multiplier for scalping
            'stop_loss_multiplier': 1.0,  # Tight stops
            'holding_days': 0.5,  # 12 hours max
            'min_reward_risk_ratio': 2.0,  # 2:1 minimum for scalping
            'volume_adjustment': True,  # Critical for scalping
            'regime_awareness': True
        }
    }

    # Kraken-optimized commission structure (taker fees only)
    kraken_commission = {
        'taker_fee': 0.0026,  # 0.26% (standard trading)
        'spread_estimate': 0.0005,
        'slippage_estimate': 0.0002,  # Lower slippage assumption for scalping
        'funding_cost_daily': 0.00005,
        'min_profit_multiplier': 3.0  # Higher multiplier for scalping
    }

    return label_multiple_events_theory_based(
        data,
        'outlier_event',
        barrier_params=barrier_params,
        commission_structure=kraken_commission
    )


def label_kraken_swing_strategy(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Swing trading strategy for Kraken with longer holding periods"""

    barrier_params = {
        'momentum_regime_event_label': {
            'volatility_multiplier': 2.5,  # Wider for swing trades
            'stop_loss_multiplier': 1.5,  # Wider stops
            'holding_days': 3.0,  # 3 days max
            'min_reward_risk_ratio': 1.8,  # Slightly lower for longer timeframe
            'volume_adjustment': False,  # Less critical for swing trades
            'regime_awareness': True
        }
    }

    return label_multiple_events_theory_based(
        data,
        'momentum_regime_event',
        barrier_params=barrier_params
    )


def label_kraken_multi_timeframe(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Multi-timeframe strategy with different parameters per event type"""

    barrier_params = {
        'outlier_event_label': {
            'volatility_multiplier': 1.8,  # Scalping-style
            'stop_loss_multiplier': 1.1,
            'holding_days': 0.75,
            'min_reward_risk_ratio': 2.2,
            'volume_adjustment': True,
            'regime_awareness': True
        },
        'momentum_regime_event_label': {
            'volatility_multiplier': 2.3,  # Swing-style
            'stop_loss_multiplier': 1.4,
            'holding_days': 2.5,
            'min_reward_risk_ratio': 1.7,
            'volume_adjustment': False,
            'regime_awareness': True
        },
        'vpd_volatility_event_label': {
            'volatility_multiplier': 3.0,  # Breakout-style
            'stop_loss_multiplier': 1.8,
            'holding_days': 1.5,
            'min_reward_risk_ratio': 1.6,
            'volume_adjustment': True,
            'regime_awareness': True
        }
    }

    return label_multiple_events_theory_based(
        data,
        ['outlier_event', 'momentum_regime_event', 'vpd_volatility_event'],
        mode='individual',
        barrier_params=barrier_params
    )


def label_high_confidence_kraken(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """High-confidence simultaneous events with wider targets"""

    barrier_params = {
        'outlier_momentum_simultaneous_label': {
            'volatility_multiplier': 2.8,  # Wider for high confidence
            'stop_loss_multiplier': 1.6,
            'holding_days': 2.0,
            'min_reward_risk_ratio': 1.5,
            'volume_adjustment': True,
            'regime_awareness': True
        }
    }

    return label_multiple_events_theory_based(
        data,
        ['outlier_event', 'momentum_regime_event'],
        mode='simultaneous',
        barrier_params=barrier_params
    )


# LEGACY COMPATIBILITY FUNCTIONS

def label_events(data: pd.DataFrame,
                 event_columns: Union[str, List[str], None] = None,
                 profit_take: float = 0.02,
                 stop_loss: float = 0.015,
                 holding_days: float = 1.0,
                 use_dynamic: bool = True,
                 use_parallel: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Legacy compatibility function - converts old parameters to new theory-based approach
    """

    # Convert legacy parameters to theory-based
    default_params = {
        'volatility_multiplier': 2.0 if use_dynamic else None,
        'stop_loss_multiplier': 1.2 if use_dynamic else None,
        'holding_days': holding_days,
        'use_theory_based': use_dynamic,
        'profit_take': profit_take,  # Used if not theory-based
        'stop_loss': stop_loss,  # Used if not theory-based
        'min_reward_risk_ratio': 1.5,
        'volume_adjustment': use_dynamic,
        'regime_awareness': use_dynamic
    }

    # Handle legacy event selection
    if event_columns is None:
        event_selection = 'any_event'
    elif isinstance(event_columns, str):
        event_selection = event_columns
    else:
        event_selection = event_columns

    return label_multiple_events_theory_based(
        data,
        event_selection,
        mode='combined' if isinstance(event_columns, list) and len(event_columns) > 1 else 'individual',
        default_params=default_params,
        use_parallel=use_parallel
    )


def quick_label_outliers(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Quick labeling for outlier events with tight barriers"""
    return label_kraken_scalping_strategy(data)


def quick_label_momentum(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Quick labeling for momentum events with wider barriers"""
    return label_kraken_swing_strategy(data)


def quick_label_all_events(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Quick labeling for all events"""
    return label_multiple_events_theory_based(
        data,
        event_columns=None  # Uses any_event
    )


# Detect all event types dynamically
labeler = MultiEventTripleBarrierLabeling(indicated)
all_events = labeler.available_events
# print(all_events)

# Run labeling: theory-based is ON by default
labeled_data, summary = label_multiple_events_theory_based(
    indicated,
    event_selection=all_events,  # e.g. ["outlier_event", "momentum_regime_event", ...]
    mode="individual",  # one label column per event type -> "<event>_label"
    use_parallel=False  # optional
)
# print(labeled_data)
print(summary)

# 1. Theory-Based Barriers:
#
# ATR/volatility-based adaptive barriers
# Commission-aware minimum profit requirements
# Liquidity and volatility regime adjustments
# No overfitting - purely theory-driven
#
# 2. Multi-Event Support:
#
# Individual labeling per event type
# Combined events (OR logic)
# Simultaneous events (AND logic)
# Custom event combinations
#
# 3. Kraken Commission Integration:
#
# Taker fees only (0.26% standard trading)
# Spread and slippage estimates
# Minimum profit multipliers
# Real-world cost analysis
#
# 4. Preset Strategies:
#
# label_kraken_scalping_strategy() - Tight barriers, short holding
# label_kraken_swing_strategy() - Wider barriers, longer holding
# label_kraken_multi_timeframe() - Different params per event
# label_high_confidence_kraken() - Simultaneous events only
#
# 5. Usage Examples:
# python# Simple usage
# labeled_data, summary = label_multiple_events_theory_based(
#     data,
#     ['outlier_event', 'momentum_regime_event']
# )
#
# # Custom parameters
# barrier_params = {
#     'outlier_event_label': {
#         'volatility_multiplier': 1.5,
#         'min_reward_risk_ratio': 2.0
#     }
# }
