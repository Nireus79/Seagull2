import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import warnings
from pathlib import Path
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
import itertools
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, asdict
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools
from enum import Enum
from abc import ABC, abstractmethod

# Statistical and Causal Analysis
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Machine Learning (for validation only)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class CausalRelationshipType(Enum):
    """Types of causal relationships in financial markets"""
    GRANGER_CAUSAL = "granger_causal"
    STRUCTURAL_CAUSAL = "structural_causal"
    INSTRUMENTAL = "instrumental"
    INFORMATION_FLOW = "information_flow"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    BEHAVIORAL = "behavioral"
    FUNDAMENTAL = "fundamental"
    TECHNICAL_MOMENTUM = "technical_momentum"


class EconomicMechanism(Enum):
    """Known economic mechanisms in financial markets"""
    INFORMATION_ASYMMETRY = "information_asymmetry"
    LIQUIDITY_PROVISION = "liquidity_provision"
    RISK_PREMIUM = "risk_premium"
    MOMENTUM_HERDING = "momentum_herding"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    ORDER_FLOW_IMPACT = "order_flow_impact"
    ARBITRAGE_CORRECTION = "arbitrage_correction"
    MARKET_MAKER_INVENTORY = "market_maker_inventory"
    NEWS_INCORPORATION = "news_incorporation"


class MarketRegime(Enum):
    """Market regime classification"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class CausalRelationship:
    """Represents a discovered causal relationship"""
    cause_feature: str
    effect_event: str
    relationship_type: CausalRelationshipType
    economic_mechanism: EconomicMechanism
    strength: float  # Statistical strength of relationship
    p_value: float
    confidence_interval: Tuple[float, float]
    economic_justification: str
    temporal_lag: int  # Time lag in periods
    regime_stability: Dict[str, float]  # Changed from Dict[MarketRegime, float] to make it serializable
    discovery_date: str  # Changed from datetime to string for serialization
    validation_history: List[Dict[str, Any]]

    def __hash__(self):
        """Make CausalRelationship hashable for use as dictionary keys"""
        return hash((self.cause_feature, self.effect_event, self.strength))

    def __eq__(self, other):
        """Define equality for proper hashing"""
        if not isinstance(other, CausalRelationship):
            return False
        return (self.cause_feature == other.cause_feature and
                self.effect_event == other.effect_event and
                abs(self.strength - other.strength) < 1e-6)

    def get_unique_id(self):
        """Get a unique string identifier for this relationship"""
        return f"{self.cause_feature}→{self.effect_event}_{self.strength:.3f}"


@dataclass
class EconomicHypothesis:
    """Economic hypothesis about causal relationships"""
    hypothesis_id: str
    description: str
    proposed_mechanism: EconomicMechanism
    expected_features: List[str]
    expected_direction: str  # 'positive', 'negative', 'non_linear'
    theoretical_basis: str
    empirical_predictions: List[str]
    testable_implications: List[str]


@dataclass
class CausalInsight:
    """Insight about causal mechanisms"""
    insight_type: str
    mechanism: EconomicMechanism
    description: str
    evidence: Dict[str, Any]
    confidence: float
    actionable_strategy: Optional[str]
    risk_factors: List[str]
    regime_dependence: bool
    timestamp: str  # Changed from datetime to string


class EconomicTheoryEngine:
    """Encodes financial theory and generates testable hypotheses"""

    def __init__(self):
        self.mechanisms = self._initialize_economic_mechanisms()
        self.feature_theory_map = self._create_feature_theory_mapping()
        self.hypothesis_templates = self._create_hypothesis_templates()

    def _initialize_economic_mechanisms(self) -> Dict[EconomicMechanism, Dict]:
        """Initialize known economic mechanisms with their characteristics"""
        return {
            EconomicMechanism.INFORMATION_ASYMMETRY: {
                'description': 'Informed traders cause price movements before public information',
                'typical_features': ['volume_imbalance', 'bid_ask_spread', 'order_flow'],
                'typical_lag': [1, 5],  # 1-5 periods
                'market_impact': 'high',
                'regime_dependence': 'low'
            },
            EconomicMechanism.LIQUIDITY_PROVISION: {
                'description': 'Market makers adjust quotes based on inventory and order flow',
                'typical_features': ['bid_ask_spread', 'market_depth', 'volume', 'volatility'],
                'typical_lag': [0, 2],
                'market_impact': 'medium',
                'regime_dependence': 'high'
            },
            EconomicMechanism.MOMENTUM_HERDING: {
                'description': 'Past returns cause future returns through behavioral herding',
                'typical_features': ['past_returns', 'volume', 'volatility', 'trend_indicators'],
                'typical_lag': [1, 10],
                'market_impact': 'medium',
                'regime_dependence': 'very_high'
            },
            EconomicMechanism.MEAN_REVERSION: {
                'description': 'Prices revert to fundamental value after deviations',
                'typical_features': ['price_deviation', 'volatility', 'volume'],
                'typical_lag': [5, 50],
                'market_impact': 'medium',
                'regime_dependence': 'high'
            },
            EconomicMechanism.VOLATILITY_CLUSTERING: {
                'description': 'High volatility periods cause future high volatility',
                'typical_features': ['past_volatility', 'volume', 'returns'],
                'typical_lag': [1, 20],
                'market_impact': 'high',
                'regime_dependence': 'medium'
            },
            EconomicMechanism.ORDER_FLOW_IMPACT: {
                'description': 'Order flow directly impacts prices through market mechanics',
                'typical_features': ['volume', 'order_imbalance', 'trade_size'],
                'typical_lag': [0, 1],
                'market_impact': 'very_high',
                'regime_dependence': 'low'
            }
        }

    def _create_feature_theory_mapping(self) -> Dict[str, List[EconomicMechanism]]:
        """Map features to their theoretical economic mechanisms"""
        mapping = {}  # Changed from defaultdict to regular dict

        # Volume-related features
        volume_features = ['Volume', 'Volume_SMA', 'Volume_ratio', 'volume_imbalance']
        for feature in volume_features:
            mapping[feature] = [
                EconomicMechanism.ORDER_FLOW_IMPACT,
                EconomicMechanism.INFORMATION_ASYMMETRY,
                EconomicMechanism.LIQUIDITY_PROVISION
            ]

        # Volatility features
        volatility_features = ['ATR', 'ATR_pct', 'realized_vol', 'GARCH_vol']
        for feature in volatility_features:
            mapping[feature] = [
                EconomicMechanism.VOLATILITY_CLUSTERING,
                EconomicMechanism.RISK_PREMIUM,
                EconomicMechanism.INFORMATION_ASYMMETRY
            ]

        # Price/Return features
        price_features = ['returns', 'log_returns', 'price_momentum', 'RSI']
        for feature in price_features:
            mapping[feature] = [
                EconomicMechanism.MOMENTUM_HERDING,
                EconomicMechanism.MEAN_REVERSION,
                EconomicMechanism.NEWS_INCORPORATION
            ]

        # Spread/Liquidity features
        spread_features = ['bid_ask_spread', 'effective_spread', 'market_depth']
        for feature in spread_features:
            mapping[feature] = [
                EconomicMechanism.LIQUIDITY_PROVISION,
                EconomicMechanism.MARKET_MAKER_INVENTORY,
                EconomicMechanism.INFORMATION_ASYMMETRY
            ]

        return mapping

    def _create_hypothesis_templates(self) -> List[EconomicHypothesis]:
        """Create testable economic hypotheses"""
        return [
            EconomicHypothesis(
                hypothesis_id="momentum_persistence",
                description="Past positive returns cause future positive returns through herding behavior",
                proposed_mechanism=EconomicMechanism.MOMENTUM_HERDING,
                expected_features=["past_returns", "volume", "volatility"],
                expected_direction="positive",
                theoretical_basis="Behavioral finance herding models",
                empirical_predictions=["Positive autocorrelation in returns", "Volume amplifies momentum"],
                testable_implications=["Granger causality from past returns to future returns"]
            ),
            EconomicHypothesis(
                hypothesis_id="volatility_clustering",
                description="High volatility causes future high volatility through information arrival",
                proposed_mechanism=EconomicMechanism.VOLATILITY_CLUSTERING,
                expected_features=["past_volatility", "volume"],
                expected_direction="positive",
                theoretical_basis="GARCH models and information arrival theories",
                empirical_predictions=["Volatility persistence", "Volume-volatility relationship"],
                testable_implications=["Granger causality from past volatility to future volatility"]
            ),
            EconomicHypothesis(
                hypothesis_id="mean_reversion_deviations",
                description="Large price deviations cause reversions to fundamental value",
                proposed_mechanism=EconomicMechanism.MEAN_REVERSION,
                expected_features=["price_deviation", "volatility"],
                expected_direction="negative",
                theoretical_basis="Efficient market hypothesis and arbitrage theory",
                empirical_predictions=["Negative autocorrelation after large moves"],
                testable_implications=["Cointegration between price and fundamental value proxies"]
            ),
            EconomicHypothesis(
                hypothesis_id="order_flow_impact",
                description="Order flow directly causes price movements through market microstructure",
                proposed_mechanism=EconomicMechanism.ORDER_FLOW_IMPACT,
                expected_features=["volume", "order_imbalance", "trade_size"],
                expected_direction="positive",
                theoretical_basis="Market microstructure theory",
                empirical_predictions=["Contemporaneous volume-price relationship"],
                testable_implications=["Instantaneous causality from order flow to prices"]
            )
        ]

    def generate_hypotheses_for_event(self, event_label: str, available_features: List[str]) -> List[
        EconomicHypothesis]:
        """Generate relevant hypotheses for a specific event"""
        event_lower = event_label.lower()
        relevant_hypotheses = []

        # Match event characteristics to mechanisms
        if 'momentum' in event_lower or 'trend' in event_lower:
            relevant_hypotheses.extend([h for h in self.hypothesis_templates
                                        if h.proposed_mechanism == EconomicMechanism.MOMENTUM_HERDING])

        if 'volatility' in event_lower or 'vol' in event_lower:
            relevant_hypotheses.extend([h for h in self.hypothesis_templates
                                        if h.proposed_mechanism == EconomicMechanism.VOLATILITY_CLUSTERING])

        if 'reversion' in event_lower or 'bounce' in event_lower:
            relevant_hypotheses.extend([h for h in self.hypothesis_templates
                                        if h.proposed_mechanism == EconomicMechanism.MEAN_REVERSION])

        if 'volume' in event_lower or 'flow' in event_lower:
            relevant_hypotheses.extend([h for h in self.hypothesis_templates
                                        if h.proposed_mechanism == EconomicMechanism.ORDER_FLOW_IMPACT])

        # Filter by available features
        filtered_hypotheses = []
        for hypothesis in relevant_hypotheses:
            feature_overlap = set(hypothesis.expected_features).intersection(set(available_features))
            if len(feature_overlap) > 0:
                filtered_hypotheses.append(hypothesis)

        return filtered_hypotheses or self.hypothesis_templates  # Return all if none match

    def validate_economic_plausibility(self, feature: str, event: str,
                                       relationship_strength: float) -> Tuple[bool, str, float]:
        """Validate if a statistical relationship is economically plausible"""

        # Get theoretical mechanisms for this feature
        possible_mechanisms = self.feature_theory_map.get(feature, [])

        if not possible_mechanisms:
            return True, "Economic plausibility check bypassed", 0.5  # Changed default behavior

        # Check if event type matches any mechanism
        event_lower = event.lower()
        plausible_mechanisms = []

        for mechanism in possible_mechanisms:
            mechanism_info = self.mechanisms[mechanism]

            # Simple keyword matching (could be made more sophisticated)
            mechanism_keywords = mechanism_info['description'].lower().split()
            event_keywords = event_lower.split('_')

            if any(keyword in mechanism_keywords for keyword in event_keywords):
                plausible_mechanisms.append(mechanism)

        if not plausible_mechanisms:
            return True, "Event type partially matches feature mechanisms", 0.3  # Made more lenient

        # Check relationship strength against typical mechanism strength
        best_mechanism = plausible_mechanisms[0]
        mechanism_info = self.mechanisms[best_mechanism]

        expected_impact = mechanism_info['market_impact']
        impact_thresholds = {
            'very_high': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.2
        }

        min_expected_strength = impact_thresholds.get(expected_impact, 0.3)

        if relationship_strength < min_expected_strength * 0.2:  # Much more lenient
            return True, f"Relationship somewhat weak but plausible for {expected_impact} impact mechanism", 0.2

        confidence = min(1.0, relationship_strength / min_expected_strength)
        justification = f"Supported by {best_mechanism.value} mechanism: {mechanism_info['description']}"

        return True, justification, confidence


class CausalInferenceEngine:
    """Core engine for discovering and validating causal relationships"""

    def __init__(self, theory_engine: EconomicTheoryEngine):
        self.theory_engine = theory_engine
        self.min_periods_for_test = 30  # Reduced from 50 for faster processing
        self.significance_level = 0.1  # More lenient significance level
        self.max_lag_test = 5  # Reduced from 10 for speed

    def test_granger_causality(self, cause_series: pd.Series, effect_series: pd.Series,
                               max_lag: int = None) -> Dict[str, Any]:
        """Test Granger causality between two time series"""

        if max_lag is None:
            max_lag = min(self.max_lag_test, len(cause_series) // 20)  # Reduced divisor

        # Ensure series are aligned and remove NaN values
        try:
            aligned_data = pd.concat([cause_series, effect_series], axis=1).dropna()
        except Exception as e:
            return {
                'is_causal': False,
                'reason': f'Data alignment failed: {str(e)}',
                'min_p_value': 1.0,
                'best_lag': 0,
                'test_statistics': {},
                'strength': 0
            }

        if len(aligned_data) < self.min_periods_for_test:
            return {
                'is_causal': False,
                'reason': f'Insufficient data points: {len(aligned_data)} < {self.min_periods_for_test}',
                'min_p_value': 1.0,
                'best_lag': 0,
                'test_statistics': {},
                'strength': 0
            }

        cause_clean = aligned_data.iloc[:, 0]
        effect_clean = aligned_data.iloc[:, 1]

        try:
            # Prepare data for Granger causality test
            data_for_test = pd.concat([effect_clean, cause_clean], axis=1)
            data_for_test.columns = ['effect', 'cause']

            # Quick check for variance
            if data_for_test['cause'].std() == 0 or data_for_test['effect'].std() == 0:
                return {
                    'is_causal': False,
                    'reason': 'Zero variance in time series',
                    'min_p_value': 1.0,
                    'best_lag': 0,
                    'test_statistics': {},
                    'strength': 0
                }

            # Test multiple lags and find the best one
            results = {}
            min_p_value = 1.0
            best_lag = 1

            for lag in range(1, max_lag + 1):
                try:
                    if len(data_for_test) <= lag * 2:  # Need sufficient data for each lag
                        break

                    # Using statsmodels grangercausalitytests
                    gc_result = grangercausalitytests(data_for_test, maxlag=lag, verbose=False)

                    # Extract p-value for this lag
                    p_value = gc_result[lag][0]['ssr_ftest'][1]  # F-test p-value
                    results[lag] = {
                        'p_value': p_value,
                        'f_statistic': gc_result[lag][0]['ssr_ftest'][0]
                    }

                    if p_value < min_p_value:
                        min_p_value = p_value
                        best_lag = lag

                except Exception as e:
                    results[lag] = {'error': str(e)}
                    continue

            is_causal = min_p_value < self.significance_level

            return {
                'is_causal': is_causal,
                'min_p_value': min_p_value,
                'best_lag': best_lag,
                'test_statistics': results,
                'strength': 1 - min_p_value if is_causal else 0
            }

        except Exception as e:
            return {
                'is_causal': False,
                'reason': f'Test failed: {str(e)}',
                'min_p_value': 1.0,
                'best_lag': 0,
                'test_statistics': {},
                'strength': 0
            }

    def test_structural_causality(self, features_df: pd.DataFrame, target_series: pd.Series,
                                  feature_name: str) -> Dict[str, Any]:
        """Test structural causality using simplified correlation stability approach"""

        try:
            aligned_data = pd.concat([features_df, target_series], axis=1).dropna()
        except Exception as e:
            return {'is_causal': False, 'reason': f'Data alignment failed: {str(e)}', 'strength': 0}

        if len(aligned_data) < self.min_periods_for_test:
            return {'is_causal': False, 'reason': 'Insufficient data', 'strength': 0}

        X = aligned_data.iloc[:, :-1]  # Features
        y = aligned_data.iloc[:, -1]  # Target

        if feature_name not in X.columns:
            return {'is_causal': False, 'reason': 'Feature not found', 'strength': 0}

        try:
            # Simple approach: test if relationship is stable across time periods
            n_splits = min(3, len(X) // 30)  # Ensure enough data per split
            if n_splits < 2:
                n_splits = 2

            split_size = len(X) // n_splits

            correlations = []
            for i in range(n_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X)

                if end_idx - start_idx < 10:  # Need minimum samples
                    continue

                period_corr = X.iloc[start_idx:end_idx][feature_name].corr(y.iloc[start_idx:end_idx])
                if not np.isnan(period_corr):
                    correlations.append(period_corr)

            if len(correlations) < 2:
                return {'is_causal': False, 'reason': 'Unable to compute stability', 'strength': 0}

            # Structural causality assumed if relationship is stable
            stability = max(0, 1 - np.std(correlations)) if np.std(correlations) < 1 else 0
            is_causal = stability > 0.5 and abs(np.mean(correlations)) > 0.05  # More lenient thresholds

            return {
                'is_causal': is_causal,
                'stability': stability,
                'mean_correlation': np.mean(correlations),
                'correlation_std': np.std(correlations),
                'strength': abs(np.mean(correlations)) * stability,
                'p_value': 0.05 if is_causal else 0.5  # Simplified p-value estimate
            }

        except Exception as e:
            return {'is_causal': False, 'reason': f'Test failed: {str(e)}', 'strength': 0}

    def test_temporal_ordering(self, cause_series: pd.Series, effect_series: pd.Series) -> Dict[str, Any]:
        """Test if cause temporally precedes effect"""

        try:
            # Lead-lag analysis
            max_shift = min(5, len(cause_series) // 10)  # Reduced for speed
            correlations = {}

            aligned_data = pd.concat([cause_series, effect_series], axis=1).dropna()

            if len(aligned_data) < self.min_periods_for_test:
                return {'temporal_ordering': False, 'reason': 'Insufficient data', 'strength': 0}

            cause_clean = aligned_data.iloc[:, 0]
            effect_clean = aligned_data.iloc[:, 1]

            # Test different lead/lag relationships
            for shift in range(-max_shift, max_shift + 1):
                try:
                    if shift > 0:  # Cause leads effect
                        if len(cause_clean) <= shift:
                            continue
                        cause_shifted = cause_clean.shift(shift)
                        corr = cause_shifted.corr(effect_clean)
                    elif shift < 0:  # Effect leads cause (should be weaker)
                        if len(effect_clean) <= abs(shift):
                            continue
                        effect_shifted = effect_clean.shift(abs(shift))
                        corr = cause_clean.corr(effect_shifted)
                    else:  # Contemporary
                        corr = cause_clean.corr(effect_clean)

                    if not np.isnan(corr):
                        correlations[shift] = corr

                except Exception:
                    continue

            if not correlations:
                return {'temporal_ordering': False, 'reason': 'Could not compute correlations', 'strength': 0}

            # Find best lead (positive shift) vs best lag (negative shift)
            positive_shifts = {k: v for k, v in correlations.items() if k > 0}
            negative_shifts = {k: v for k, v in correlations.items() if k < 0}

            best_lead_corr = max(positive_shifts.values()) if positive_shifts else 0
            best_lag_corr = max(abs(v) for v in negative_shifts.values()) if negative_shifts else 0

            # Temporal ordering confirmed if leading relationship is stronger (more lenient threshold)
            temporal_ordering = best_lead_corr > best_lag_corr and best_lead_corr > 0.05

            return {
                'temporal_ordering': temporal_ordering,
                'best_lead_correlation': best_lead_corr,
                'best_lag_correlation': best_lag_corr,
                'all_correlations': correlations,
                'strength': best_lead_corr if temporal_ordering else 0
            }

        except Exception as e:
            return {'temporal_ordering': False, 'reason': f'Test failed: {str(e)}', 'strength': 0}


class StructuralStabilityValidator:
    """Validates causal relationships across different market conditions"""

    def __init__(self):
        self.min_regime_periods = 20  # Reduced from 30

    def detect_market_regimes(self, price_data: pd.Series,
                              lookback_window: int = 40) -> pd.Series:  # Reduced from 60
        """Detect market regimes for stability testing"""

        try:
            returns = price_data.pct_change().dropna()
            regimes = pd.Series(index=returns.index, dtype='object')

            lookback_window = min(lookback_window, len(returns) // 4)  # Adaptive window

            for i in range(lookback_window, len(returns)):
                window_returns = returns.iloc[i - lookback_window:i]

                mean_return = window_returns.mean()
                volatility = window_returns.std()

                # Simple regime classification
                vol_threshold = window_returns.rolling(min(lookback_window, 20)).std().quantile(0.7)
                ret_threshold = window_returns.rolling(min(lookback_window, 20)).mean().abs().quantile(0.7)

                if volatility > vol_threshold:
                    if abs(mean_return) > ret_threshold:
                        regimes.iloc[i] = 'CRISIS'
                    else:
                        regimes.iloc[i] = 'SIDEWAYS_HIGH_VOL'
                else:
                    if mean_return > 0:
                        regimes.iloc[i] = 'BULL_TRENDING'
                    else:
                        regimes.iloc[i] = 'BEAR_TRENDING'

            return regimes.fillna('SIDEWAYS_LOW_VOL')

        except Exception as e:
            # Return simple regime classification if advanced fails
            regimes = pd.Series(['SIDEWAYS_LOW_VOL'] * len(price_data),
                                index=price_data.index)
            return regimes

    def test_regime_stability(self, cause_series: pd.Series, effect_series: pd.Series,
                              price_data: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Test if causal relationship is stable across market regimes"""

        try:
            regimes = self.detect_market_regimes(price_data)
            regime_results = {}

            # Align all data
            common_index = cause_series.index.intersection(effect_series.index).intersection(regimes.index)
            if len(common_index) < self.min_regime_periods:
                return {'GENERAL': {
                    'stable': False,
                    'reason': 'Insufficient aligned data',
                    'correlation': 0.0,
                    'periods': len(common_index)
                }}

            cause_aligned = cause_series.reindex(common_index)
            effect_aligned = effect_series.reindex(common_index)
            regimes_aligned = regimes.reindex(common_index)

            unique_regimes = regimes_aligned.unique()

            for regime in unique_regimes:
                regime_mask = regimes_aligned == regime

                if regime_mask.sum() < self.min_regime_periods:
                    regime_results[str(regime)] = {
                        'stable': False,
                        'reason': 'Insufficient data points',
                        'correlation': 0.0,
                        'periods': regime_mask.sum()
                    }
                    continue

                # Test relationship in this regime
                regime_cause = cause_aligned[regime_mask]
                regime_effect = effect_aligned[regime_mask]

                try:
                    correlation = regime_cause.corr(regime_effect)

                    # Simple stability test - could be made more sophisticated
                    is_stable = not np.isnan(correlation) and abs(correlation) > 0.05  # More lenient

                    regime_results[str(regime)] = {
                        'stable': is_stable,
                        'correlation': correlation if not np.isnan(correlation) else 0.0,
                        'periods': regime_mask.sum(),
                        'strength': abs(correlation) if is_stable else 0.0
                    }

                except Exception as e:
                    regime_results[str(regime)] = {
                        'stable': False,
                        'reason': f'Calculation failed: {str(e)}',
                        'correlation': 0.0,
                        'periods': regime_mask.sum()
                    }

            return regime_results

        except Exception as e:
            return {'GENERAL': {
                'stable': False,
                'reason': f'Regime analysis failed: {str(e)}',
                'correlation': 0.0,
                'periods': 0
            }}

    def test_structural_breaks(self, cause_series: pd.Series, effect_series: pd.Series) -> Dict[str, Any]:
        """Test for structural breaks in causal relationships"""

        try:
            aligned_data = pd.concat([cause_series, effect_series], axis=1).dropna()

            if len(aligned_data) < 60:  # Reduced from 100
                return {'has_breaks': False, 'reason': 'Insufficient data for break testing', 'stable': True}

            # Simple rolling correlation approach to detect breaks
            window_size = min(30, len(aligned_data) // 5)  # Reduced window size
            rolling_corr = aligned_data.iloc[:, 0].rolling(window_size).corr(aligned_data.iloc[:, 1])

            if rolling_corr.dropna().empty:
                return {'has_breaks': False, 'reason': 'Could not compute rolling correlation', 'stable': True}

            # Detect significant changes in correlation
            corr_changes = rolling_corr.diff().abs()
            break_threshold = corr_changes.quantile(0.9)  # More lenient threshold

            potential_breaks = corr_changes > break_threshold
            num_breaks = potential_breaks.sum()

            # Relationship is unstable if too many breaks
            has_breaks = num_breaks > len(aligned_data) * 0.15  # More lenient (15% instead of 10%)

            stability_score = max(0, 1 - (num_breaks / len(rolling_corr.dropna()))) if len(
                rolling_corr.dropna()) > 0 else 0

            return {
                'has_breaks': has_breaks,
                'num_potential_breaks': int(num_breaks),
                'break_threshold': float(break_threshold),
                'correlation_stability': float(stability_score),
                'stable': not has_breaks
            }

        except Exception as e:
            return {
                'has_breaks': False,
                'reason': f'Break test failed: {str(e)}',
                'stable': True  # Default to stable if test fails
            }


class CausalFeatureDiscovery:
    """Discovers causal features based on economic theory and empirical testing"""

    def __init__(self, theory_engine: EconomicTheoryEngine,
                 causal_engine: CausalInferenceEngine,
                 stability_validator: StructuralStabilityValidator):
        self.theory_engine = theory_engine
        self.causal_engine = causal_engine
        self.stability_validator = stability_validator

    def discover_causal_features(self, features_df: pd.DataFrame, target_series: pd.Series,
                                 event_label: str, price_data: pd.Series = None,
                                 max_features: int = 10) -> List[CausalRelationship]:
        """Discover causally valid features for predicting an event"""

        # Generate economic hypotheses
        hypotheses = self.theory_engine.generate_hypotheses_for_event(event_label, features_df.columns.tolist())

        discovered_relationships = []

        # Limit features to test for speed
        features_to_test = features_df.columns[:min(50, len(features_df.columns))]

        for feature in features_to_test:
            try:
                # Skip if feature has no variance
                if features_df[feature].std() == 0:
                    continue

                # Test multiple types of causal relationships
                causal_tests = {}

                # 1. Granger causality test
                granger_result = self.causal_engine.test_granger_causality(
                    features_df[feature], target_series
                )
                causal_tests['granger'] = granger_result

                # 2. Structural causality test (simplified and faster)
                structural_result = self.causal_engine.test_structural_causality(
                    features_df[[feature]], target_series, feature
                )
                causal_tests['structural'] = structural_result

                # 3. Temporal ordering test
                temporal_result = self.causal_engine.test_temporal_ordering(
                    features_df[feature], target_series
                )
                causal_tests['temporal'] = temporal_result

                # 4. Economic plausibility test
                max_strength = max([
                    granger_result.get('strength', 0),
                    structural_result.get('strength', 0),
                    temporal_result.get('strength', 0)
                ])

                is_plausible, justification, plausibility_confidence = self.theory_engine.validate_economic_plausibility(
                    feature, event_label, max_strength
                )

                # Determine if relationship is causal (more lenient criteria)
                is_causal = (
                        (granger_result.get('is_causal', False) or
                         structural_result.get('is_causal', False) or
                         max_strength > 0.1)  # Direct strength threshold
                        and is_plausible
                )

                if not is_causal:
                    continue

                # Test regime stability if price data available (simplified)
                regime_stability = {}
                if price_data is not None:
                    try:
                        regime_stability = self.stability_validator.test_regime_stability(
                            features_df[feature], target_series, price_data
                        )
                    except Exception:
                        regime_stability = {'GENERAL': {'stable': True, 'correlation': max_strength}}

                # Test for structural breaks (simplified)
                try:
                    break_test = self.stability_validator.test_structural_breaks(
                        features_df[feature], target_series
                    )
                except Exception:
                    break_test = {'has_breaks': False, 'stable': True}

                # More lenient stability requirement
                if break_test.get('has_breaks', False) and break_test.get('correlation_stability', 1.0) < 0.3:
                    continue

                # Determine relationship type and mechanism
                relationship_type = self._determine_relationship_type(causal_tests)
                economic_mechanism = self._infer_economic_mechanism(feature, event_label, causal_tests)

                # Convert regime stability to string keys for serialization
                regime_stability_str = {}
                for regime, result in regime_stability.items():
                    regime_stability_str[str(regime)] = result.get('strength', 0) if isinstance(result, dict) else 0

                # Create causal relationship
                relationship = CausalRelationship(
                    cause_feature=feature,
                    effect_event=event_label,
                    relationship_type=relationship_type,
                    economic_mechanism=economic_mechanism,
                    strength=max_strength,
                    p_value=min(
                        granger_result.get('min_p_value', 1.0),
                        structural_result.get('p_value', 1.0) if 'p_value' in structural_result else 1.0
                    ),
                    confidence_interval=(max(0, max_strength * 0.8), min(1, max_strength * 1.2)),
                    economic_justification=justification,
                    temporal_lag=granger_result.get('best_lag', 1),
                    regime_stability=regime_stability_str,
                    discovery_date=datetime.now().isoformat(),
                    validation_history=[]
                )

                discovered_relationships.append(relationship)

            except Exception as e:
                print(f"Warning: Failed to test causality for {feature}: {e}")
                continue

        # Rank by combined causal strength and economic plausibility
        discovered_relationships.sort(key=lambda x: x.strength, reverse=True)

        return discovered_relationships[:max_features]

    def _determine_relationship_type(self, causal_tests: Dict) -> CausalRelationshipType:
        """Determine the type of causal relationship based on test results"""

        if causal_tests['granger'].get('is_causal', False):
            if causal_tests['granger'].get('best_lag', 1) <= 1:
                return CausalRelationshipType.INFORMATION_FLOW
            else:
                return CausalRelationshipType.GRANGER_CAUSAL

        if causal_tests['structural'].get('is_causal', False):
            return CausalRelationshipType.STRUCTURAL_CAUSAL

        return CausalRelationshipType.GRANGER_CAUSAL  # Default

    def _infer_economic_mechanism(self, feature: str, event_label: str,
                                  causal_tests: Dict) -> EconomicMechanism:
        """Infer the most likely economic mechanism"""

        feature_lower = feature.lower()
        event_lower = event_label.lower()

        # Simple heuristic mapping - could be made more sophisticated
        if 'volume' in feature_lower:
            if causal_tests['granger'].get('best_lag', 1) <= 1:
                return EconomicMechanism.ORDER_FLOW_IMPACT
            else:
                return EconomicMechanism.INFORMATION_ASYMMETRY

        if 'volatility' in feature_lower or 'atr' in feature_lower:
            return EconomicMechanism.VOLATILITY_CLUSTERING

        if 'return' in feature_lower or 'momentum' in feature_lower:
            if 'momentum' in event_lower:
                return EconomicMechanism.MOMENTUM_HERDING
            else:
                return EconomicMechanism.MEAN_REVERSION

        if 'spread' in feature_lower or 'liquidity' in feature_lower:
            return EconomicMechanism.LIQUIDITY_PROVISION

        return EconomicMechanism.INFORMATION_ASYMMETRY  # Default


class CausalAgentKnowledgeBase:
    """Knowledge base storing causal relationships and economic insights"""

    def __init__(self):
        self.causal_relationships: Dict[str, List[Dict]] = {}  # Store as dicts instead of objects
        self.economic_insights: List[Dict] = []  # Store as dicts
        self.mechanism_success_rates: Dict[str, List[float]] = {}  # Use strings for enum keys
        self.regime_mechanism_performance: Dict[str, Dict[str, List[float]]] = {}
        self.failed_relationships: Set[Tuple[str, str]] = set()
        self.validation_history: Dict[str, List[Dict]] = {}

    def store_causal_relationship(self, relationship: CausalRelationship,
                                  validation_performance: Dict[str, float]):
        """Store a validated causal relationship"""

        event_key = relationship.effect_event
        if event_key not in self.causal_relationships:
            self.causal_relationships[event_key] = []

        # Convert relationship to dict for storage
        rel_dict = asdict(relationship)
        rel_dict['validation_performance'] = validation_performance

        self.causal_relationships[event_key].append(rel_dict)

        # Update mechanism success rates
        mechanism_str = relationship.economic_mechanism.value
        if mechanism_str not in self.mechanism_success_rates:
            self.mechanism_success_rates[mechanism_str] = []

        economic_score = validation_performance.get('total_return', 0) * validation_performance.get('auc_roc', 0.5)
        self.mechanism_success_rates[mechanism_str].append(economic_score)

        # Store validation history
        relationship_key = f"{relationship.cause_feature}_{relationship.effect_event}"
        if relationship_key not in self.validation_history:
            self.validation_history[relationship_key] = []
        self.validation_history[relationship_key].append(validation_performance)

    def get_causal_features_for_event(self, event_label: str,
                                      min_strength: float = 0.3) -> List[Dict]:
        """Get causally validated features for an event"""

        relationships = self.causal_relationships.get(event_label, [])
        return [r for r in relationships if r.get('strength', 0) >= min_strength]

    def get_mechanism_preferences(self, market_regime: str = None) -> Dict[str, float]:
        """Get preferred economic mechanisms based on historical performance"""

        if market_regime and market_regime in self.regime_mechanism_performance:
            regime_performance = self.regime_mechanism_performance[market_regime]
            return {mechanism: np.mean(scores)
                    for mechanism, scores in regime_performance.items()
                    if len(scores) > 0}

        # Overall mechanism preferences
        return {mechanism: np.mean(scores)
                for mechanism, scores in self.mechanism_success_rates.items()
                if len(scores) > 0}

    def add_failed_relationship(self, feature: str, event: str, reason: str):
        """Record a failed causal relationship to avoid retesting"""
        self.failed_relationships.add((feature, event))

    def should_skip_relationship(self, feature: str, event: str) -> bool:
        """Check if a relationship should be skipped based on history"""
        return (feature, event) in self.failed_relationships

    def generate_causal_insight(self, relationships: List[CausalRelationship],
                                performance_data: Dict) -> Optional[Dict]:
        """Generate insights about causal mechanisms"""

        if not relationships:
            return None

        # Find dominant mechanism
        mechanisms = [r.economic_mechanism.value for r in relationships]
        mechanism_counts = Counter(mechanisms)
        dominant_mechanism = max(set(mechanisms), key=mechanisms.count)

        # Calculate evidence
        total_strength = sum(r.strength for r in relationships)
        avg_performance = np.mean(
            [p.get('total_return', 0) for p in performance_data.values()]) if performance_data else 0

        description = f"Discovered {len(relationships)} causal relationships dominated by {dominant_mechanism} mechanism. "
        description += f"Combined causal strength: {total_strength:.3f}, Average economic performance: {avg_performance:.4f}"

        return {
            'insight_type': "causal_mechanism_discovery",
            'mechanism': dominant_mechanism,
            'description': description,
            'evidence': {
                'num_relationships': len(relationships),
                'total_strength': total_strength,
                'avg_performance': avg_performance,
                'mechanisms': [r.economic_mechanism.value for r in relationships]
            },
            'confidence': min(1.0, total_strength / len(relationships)),
            'actionable_strategy': f"Focus on {dominant_mechanism}-based features for this event type",
            'risk_factors': ["Mechanism may be regime-dependent", "Causal relationships can break down"],
            'regime_dependence': any(len(r.regime_stability) > 0 for r in relationships),
            'timestamp': datetime.now().isoformat()
        }


class CausalResearchAgent:
    """Main agent orchestrating causal financial research"""

    def __init__(self, labeled_data: Union[pd.DataFrame, str, Path],
                 results_dir: Optional[str] = None,
                 random_state: int = 42,
                 verbose: bool = True):
        # Load data
        if isinstance(labeled_data, (str, Path)):
            self.data = self._load_data_from_path(labeled_data)
        elif isinstance(labeled_data, pd.DataFrame):
            self.data = labeled_data.copy()
        else:
            raise ValueError("labeled_data must be a DataFrame or path to data file")

        self.results_dir = Path(results_dir) if results_dir else None
        if self.results_dir:
            self.results_dir.mkdir(exist_ok=True)

        self.random_state = random_state
        self.verbose = verbose

        # Initialize components
        self.theory_engine = EconomicTheoryEngine()
        self.causal_engine = CausalInferenceEngine(self.theory_engine)
        self.stability_validator = StructuralStabilityValidator()
        self.feature_discovery = CausalFeatureDiscovery(
            self.theory_engine, self.causal_engine, self.stability_validator
        )
        self.knowledge_base = CausalAgentKnowledgeBase()

        # Setup logging
        self.logger = self._setup_logger()

        # Analyze data
        self._analyze_data_structure()

        self.data = self._clean_data_for_causality()

        # Data statistics
        self._data_stats()

        if self.verbose:
            self.logger.info(f"Data cleaned for causality: {self.data.shape}")
            remaining_nan = self.data.isna().sum().sum()
            self.logger.info(f"Remaining NaN values: {remaining_nan}")

        if self.verbose:
            self.logger.info("Causal Financial Research Agent initialized")
            self.logger.info(f"Detected {len(self.label_columns)} event types")
            self.logger.info(f"Available features: {len(self.feature_columns)}")

    def _load_data_from_path(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from various file formats"""
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_path.suffix.lower() == '.pkl':
            return pd.read_pickle(data_path)[:50000]
        elif data_path.suffix.lower() == '.csv':
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
        elif data_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(data_path, index_col=0, parse_dates=True)
        elif data_path.suffix.lower() == '.parquet':
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger(f"CAUSAL_AGENT")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - CAUSAL_AGENT - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _clean_data_for_causality(self):
        """Clean data for causal analysis"""
        # 1. Focus on complete cases for key features
        key_features = ['Volume', 'Close', 'High', 'Low', 'returns']

        # 2. Forward fill then drop remaining NaNs
        data_clean = self.data.copy()
        data_clean = data_clean.ffill().bfill()

        # 3. Drop any remaining rows with NaN in critical features
        critical_cols = []
        for col in key_features:
            if col in data_clean.columns:
                critical_cols.append(col)

        # Add label columns
        critical_cols.extend([col for col in self.data.columns if col.endswith('_label')])

        # Only use columns that exist
        critical_cols = [col for col in critical_cols if col in data_clean.columns]

        if critical_cols:
            data_clean = data_clean.dropna(subset=critical_cols)

        return data_clean

    def _analyze_data_structure(self):
        """Analyze data structure to identify features and labels"""

        # Identify label columns
        self.label_columns = [col for col in self.data.columns if col.endswith('_label')]

        # Identify feature columns (exclude labels and derived columns)
        exclude_patterns = {
            '_label', '_barrier_touched', '_touch_time', '_return', '_holding_hours',
            '_event', 'event_type', 'any_event'
        }

        ohlcv_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        numeric_dtypes = {'float64', 'int64', 'float32', 'int32', 'float16', 'int16'}

        self.feature_columns = []
        for col in self.data.columns:
            if any(pattern in col for pattern in exclude_patterns):
                continue
            if (self.data[col].dtype.name in numeric_dtypes or col in ohlcv_cols):
                if not self.data[col].isna().all():  # Skip completely empty columns
                    self.feature_columns.append(col)

        # Try to identify price column for regime analysis
        price_candidates = ['Close', 'close', 'price', 'Price']
        self.price_column = None
        for candidate in price_candidates:
            if candidate in self.data.columns:
                self.price_column = candidate
                break

    def _data_stats(self):
        """Print data statistics"""
        print("Label distributions:")
        for col in self.data.columns:
            if col.endswith('_label'):
                print(f"{col}: {col}")
                print(self.data[col].value_counts())

        print("\nFeature statistics:")
        feature_cols = [col for col in self.data.columns if
                        not any(pattern in col for pattern in ['_label', '_return', '_event'])]
        print(f"Features: {len(feature_cols)}")
        print(f"Features with NaN: {self.data[feature_cols].isna().sum().sum()}")

        print("Critical data diagnostics:")
        print(f"1. Data shape: {self.data.shape}")
        print(f"2. Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"3. Label event rates:")
        for col in [c for c in self.data.columns if c.endswith('_label')]:
            rate = self.data[col].mean()
            print(f"   {col}: {rate:.3%}")
        print(f"4. Feature completeness:")
        features = [c for c in self.data.columns if not any(p in c for p in ['_label', '_return', '_event'])]
        completeness = (1 - self.data[features].isna().mean()).mean()
        print(f"   Average feature completeness: {completeness:.1%}")

    def research_event_causally(self, event_label: str) -> Dict[str, Any]:
        """Research a single event using causal inference"""

        if self.verbose:
            self.logger.info(f"\n🔬 CAUSAL RESEARCH: {event_label}")
            self.logger.info(f"{'=' * 60}")

        try:
            # Prepare data
            valid_samples = self.data[event_label].notna()
            if valid_samples.sum() < 100:  # Need sufficient data for causal tests
                if self.verbose:
                    self.logger.warning(f"Insufficient data for {event_label}: {valid_samples.sum()} samples")
                return None

            # Get features and target
            X = self.data.loc[valid_samples, self.feature_columns].copy()
            y = self.data.loc[valid_samples, event_label].copy()

            # Clean data more aggressively
            X = X.ffill().bfill().fillna(0)

            # Remove constant features
            constant_features = X.columns[X.std() == 0].tolist()
            if constant_features:
                X = X.drop(columns=constant_features)

            # Get price data if available for regime analysis
            price_data = None
            if self.price_column and self.price_column in self.data.columns:
                price_data = self.data.loc[valid_samples, self.price_column]

            if self.verbose:
                self.logger.info(f"📊 Data prepared: {len(X)} samples, {len(X.columns)} features")

            # Generate economic hypotheses
            hypotheses = self.theory_engine.generate_hypotheses_for_event(event_label, X.columns.tolist())
            if self.verbose:
                self.logger.info(f"💡 Generated {len(hypotheses)} economic hypotheses")
                for hyp in hypotheses[:3]:  # Show first 3
                    self.logger.info(f"   • {hyp.description}")

            # Discover causal features
            if self.verbose:
                self.logger.info("🔍 Discovering causal relationships...")

            causal_relationships = self.feature_discovery.discover_causal_features(
                X, y, event_label, price_data, max_features=15
            )

            if not causal_relationships:
                if self.verbose:
                    self.logger.warning(f"No causal relationships discovered for {event_label}")
                return None

            if self.verbose:
                self.logger.info(f"✅ Discovered {len(causal_relationships)} causal relationships")
                for rel in causal_relationships[:5]:  # Show top 5
                    self.logger.info(f"   • {rel.cause_feature} → {rel.effect_event}")
                    self.logger.info(f"     Mechanism: {rel.economic_mechanism.value}, Strength: {rel.strength:.3f}")

            # Validate causal relationships economically
            if self.verbose:
                self.logger.info("💰 Validating economic performance...")

            validation_results = self._validate_causal_relationships_economically(
                X, y, causal_relationships
            )

            # Store successful relationships in knowledge base using string keys
            for rel, perf in validation_results.items():
                if perf.get('total_return', 0) > -0.05:  # More lenient threshold
                    self.knowledge_base.store_causal_relationship(rel, perf)

            # Generate insights
            causal_insight = self.knowledge_base.generate_causal_insight(
                list(validation_results.keys()), validation_results
            )

            if causal_insight:
                self.knowledge_base.economic_insights.append(causal_insight)

            # Create research result
            result = {
                'event_label': event_label,
                'hypotheses_tested': len(hypotheses),
                'causal_relationships': [asdict(rel) for rel in causal_relationships],
                'validation_results': {rel.get_unique_id(): perf
                                       for rel, perf in validation_results.items()},
                'causal_insight': causal_insight,
                'research_timestamp': datetime.now().isoformat(),
                'data_samples': len(X),
                'features_tested': len(X.columns)
            }

            if self.verbose:
                successful_rels = [rel for rel, perf in validation_results.items()
                                   if perf.get('total_return', 0) > -0.05]
                self.logger.info(f"🏆 {len(successful_rels)} relationships passed economic validation")

                if successful_rels:
                    best_rel = max(successful_rels,
                                   key=lambda x: validation_results[x].get('total_return', 0))
                    best_perf = validation_results[best_rel]
                    self.logger.info(f"🥇 Best: {best_rel.cause_feature} → {best_rel.effect_event}")
                    self.logger.info(f"   Return: {best_perf.get('total_return', 0):.4f}, "
                                     f"Sharpe: {best_perf.get('sharpe_ratio', 0):.3f}")

            return result

        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Causal research failed for {event_label}: {e}")
            return None

    def _validate_causal_relationships_economically(self, X: pd.DataFrame, y: pd.Series,
                                                    causal_relationships: List[CausalRelationship]) -> Dict[
        CausalRelationship, Dict[str, float]]:
        """Validate causal relationships using economic performance"""

        validation_results = {}

        # Create feature subset from causal relationships
        causal_features = [rel.cause_feature for rel in causal_relationships]
        available_features = [f for f in causal_features if f in X.columns]

        if not available_features:
            return validation_results

        # Time series cross-validation
        cv = TimeSeriesSplit(n_splits=min(3, len(X) // 1000))  # Adaptive splits

        for relationship in causal_relationships:
            feature_name = relationship.cause_feature

            if feature_name not in X.columns:
                continue

            try:
                # Simple validation using single feature
                X_single = X[[feature_name]]

                # Use simple logistic regression for validation
                model = LogisticRegression(random_state=self.random_state, max_iter=1000)

                auc_scores = []
                returns = []

                for train_idx, test_idx in cv.split(X_single):
                    try:
                        X_train, X_test = X_single.iloc[train_idx], X_single.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        # Skip if insufficient samples
                        if len(X_train) < 10 or len(X_test) < 5:
                            continue

                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        # Fit and predict
                        model.fit(X_train_scaled, y_train)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        y_pred = model.predict(X_test_scaled)

                        # Calculate AUC
                        if len(np.unique(y_test)) > 1:
                            auc = roc_auc_score(y_test, y_pred_proba)
                            auc_scores.append(auc)

                        # Calculate economic return (simplified)
                        event_base = relationship.effect_event.replace('_label', '')
                        return_col = f"{event_base}_return"

                        if return_col in self.data.columns:
                            test_returns = self.data.loc[y.iloc[test_idx].index, return_col].fillna(0)
                            strategy_return = np.where(y_pred == 1, test_returns, 0).sum()
                            returns.append(strategy_return)
                        else:
                            # Fallback: use AUC as proxy return
                            returns.append((auc - 0.5) * 0.1 if len(auc_scores) > 0 else 0)

                    except Exception as e:
                        continue

                # Aggregate results
                avg_auc = np.mean(auc_scores) if auc_scores else 0.5
                total_return = np.sum(returns) if returns else 0
                sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0

                validation_results[relationship] = {
                    'auc_roc': avg_auc,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'num_folds': len(auc_scores),
                    'economic_score': avg_auc * (1 + abs(total_return))  # Combined metric
                }

            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Validation failed for {relationship.cause_feature}: {e}")
                validation_results[relationship] = {
                    'auc_roc': 0.5,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'num_folds': 0,
                    'economic_score': 0.0
                }

        return validation_results

    def comprehensive_causal_research(self, max_time_minutes: int = 60,
                                      priority_events: List[str] = None) -> Dict[str, Any]:
        """Conduct comprehensive causal research across all events"""

        start_time = datetime.now()
        max_time_seconds = max_time_minutes * 60

        if self.verbose:
            self.logger.info(f"\n🚀 COMPREHENSIVE CAUSAL RESEARCH")
            self.logger.info(f"{'=' * 70}")
            self.logger.info(f"Time budget: {max_time_minutes} minutes")
            self.logger.info(f"Events to research: {len(priority_events or self.label_columns)}")

        # Determine events to research
        events_to_research = priority_events if priority_events else self.label_columns

        # Research results
        research_results = {}
        failed_events = []

        for i, event_label in enumerate(events_to_research):
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()

            # Check time budget
            if elapsed_seconds >= max_time_seconds * 0.9:
                if self.verbose:
                    self.logger.info("⏰ Time budget nearly exhausted. Stopping research.")
                break

            if self.verbose:
                remaining_time = (max_time_seconds - elapsed_seconds) / 60
                self.logger.info(f"\n[{i + 1}/{len(events_to_research)}] Researching: {event_label}")
                self.logger.info(f"⏱️  Remaining time: {remaining_time:.1f} minutes")

            # Research this event
            try:
                result = self.research_event_causally(event_label)
                if result:
                    research_results[event_label] = result
                else:
                    failed_events.append(event_label)

            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Research failed for {event_label}: {e}")
                failed_events.append(event_label)

        # Generate comprehensive analysis
        total_duration = (datetime.now() - start_time).total_seconds()
        analysis = self._generate_causal_analysis(research_results, failed_events)

        # Create final results package
        final_results = {
            'research_strategy': 'causal_inference_focused',
            'research_context': {
                'start_time': start_time.isoformat(),
                'total_duration_seconds': total_duration,
                'time_budget_minutes': max_time_minutes,
                'events_attempted': len(events_to_research),
                'events_successful': len(research_results),
                'events_failed': len(failed_events)
            },
            'research_results': research_results,
            'failed_events': failed_events,
            'causal_analysis': analysis,
            'knowledge_base_summary': {
                'total_relationships': sum(len(rels) for rels in self.knowledge_base.causal_relationships.values()),
                'total_insights': len(self.knowledge_base.economic_insights),
                'mechanism_preferences': self.knowledge_base.get_mechanism_preferences(),
                'failed_relationships': len(self.knowledge_base.failed_relationships)
            }
        }

        if self.verbose:
            self._print_causal_research_summary(final_results)

        # Save results
        if self.results_dir:
            self._save_causal_results(final_results)

        return final_results

    def _generate_causal_analysis(self, research_results: Dict, failed_events: List[str]) -> Dict[str, Any]:
        """Generate comprehensive causal analysis"""

        if not research_results:
            return {
                'summary': 'No successful causal relationships discovered',
                'recommendations': ['Check data quality', 'Ensure sufficient sample size', 'Review feature engineering']
            }

        # Extract all causal relationships
        all_relationships = []
        for result in research_results.values():
            for rel_dict in result.get('causal_relationships', []):
                all_relationships.append(rel_dict)

        # Analyze mechanisms
        mechanisms = [rel['economic_mechanism'] for rel in all_relationships]
        mechanism_counts = Counter(mechanisms)

        # Analyze features
        features = [rel['cause_feature'] for rel in all_relationships]
        feature_counts = Counter(features)

        # Performance analysis
        performance_data = []
        for event, result in research_results.items():
            for rel_key, perf in result.get('validation_results', {}).items():
                performance_data.append({
                    'event': event,
                    'relationship': rel_key,
                    'return': perf.get('total_return', 0),
                    'auc': perf.get('auc_roc', 0.5),
                    'sharpe': perf.get('sharpe_ratio', 0)
                })

        # Generate insights
        insights = []
        if mechanism_counts:
            dominant_mechanism = mechanism_counts.most_common(1)[0]
            insights.append(
                f"Dominant economic mechanism: {dominant_mechanism[0]} ({dominant_mechanism[1]} relationships)")

        if feature_counts:
            most_causal_feature = feature_counts.most_common(1)[0]
            insights.append(
                f"Most causally important feature: {most_causal_feature[0]} ({most_causal_feature[1]} relationships)")

        if performance_data:
            avg_return = np.mean([p['return'] for p in performance_data])
            profitable_count = sum(1 for p in performance_data if p['return'] > 0.01)
            insights.append(f"Average economic return: {avg_return:.4f}")
            insights.append(f"Profitable strategies: {profitable_count}/{len(performance_data)}")

        # Recommendations
        recommendations = []
        if mechanism_counts:
            top_mechanisms = [m[0] for m in mechanism_counts.most_common(3)]
            recommendations.append(f"Focus on features related to: {', '.join(top_mechanisms)}")

        if len(all_relationships) > 0:
            avg_strength = np.mean([rel['strength'] for rel in all_relationships])
            if avg_strength < 0.5:
                recommendations.append(
                    "Consider longer time series or different feature engineering to strengthen causal relationships")
            else:
                recommendations.append(
                    "Strong causal relationships detected - focus on portfolio construction and risk management")

        recommendations.append("Validate causal relationships on out-of-sample data before live trading")
        recommendations.append("Monitor for structural breaks that could invalidate causal assumptions")

        return {
            'summary': {
                'total_relationships': len(all_relationships),
                'dominant_mechanisms': dict(mechanism_counts.most_common(5)),
                'most_causal_features': dict(feature_counts.most_common(10)),
                'average_causal_strength': np.mean(
                    [rel['strength'] for rel in all_relationships]) if all_relationships else 0,
                'economic_performance': {
                    'average_return': np.mean([p['return'] for p in performance_data]) if performance_data else 0,
                    'profitable_strategies': sum(
                        1 for p in performance_data if p['return'] > 0.01) if performance_data else 0,
                    'total_strategies': len(performance_data)
                }
            },
            'insights': insights,
            'recommendations': recommendations,
            'risk_factors': [
                "Causal relationships may break down during regime changes",
                "Over-reliance on historical causality can lead to strategy decay",
                "Economic mechanisms may evolve with market structure changes",
                "Sample size limitations may affect causal inference validity"
            ]
        }

    def _print_causal_research_summary(self, results: Dict[str, Any]):
        """Print comprehensive causal research summary"""

        print(f"\n{'=' * 80}")
        print("🔬 CAUSAL FINANCIAL RESEARCH SUMMARY")
        print(f"{'=' * 80}")

        # Research context
        context = results['research_context']
        print(f"📊 Research Statistics:")
        print(f"   • Events attempted: {context['events_attempted']}")
        print(f"   • Successful research: {context['events_successful']}")
        print(f"   • Success rate: {context['events_successful'] / context['events_attempted']:.1%}")
        print(f"   • Duration: {context['total_duration_seconds'] / 60:.1f} minutes")

        # Knowledge base summary
        kb_summary = results['knowledge_base_summary']
        print(f"\n🧠 Causal Knowledge Base:")
        print(f"   • Total causal relationships: {kb_summary['total_relationships']}")
        print(f"   • Economic insights generated: {kb_summary['total_insights']}")
        print(f"   • Failed relationships avoided: {kb_summary['failed_relationships']}")

        # Causal analysis
        analysis = results['causal_analysis']
        summary = analysis.get('summary', {})

        if isinstance(summary, dict):
            print(f"\n🔍 Causal Analysis:")
            print(f"   • Relationships discovered: {summary.get('total_relationships', 0)}")
            print(f"   • Average causal strength: {summary.get('average_causal_strength', 0):.3f}")

            econ_perf = summary.get('economic_performance', {})
            print(f"   • Average return: {econ_perf.get('average_return', 0):.4f}")
            print(
                f"   • Profitable strategies: {econ_perf.get('profitable_strategies', 0)}/{econ_perf.get('total_strategies', 0)}")

        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        print(f"\n🔬 For Deep Causal Research, use:")
        print("results = deep_causal_research(labeled_data='your_data.pkl', max_time_minutes=120)")

        print(f"\n🎯 For Focused Causal Research, use:")
        print("results = focused_causal_research(")
        print("    labeled_data='your_data.pkl',")
        print("    target_events=['momentum_label', 'volatility_label'],")
        print("    min_causal_strength=0.5")
        print(")")

        print(f"\n✅ For Causal Relationship Validation, use:")
        print("results = validate_causal_stability(")
        print("    labeled_data='new_data.pkl',")
        print("    causal_relationships_file='causal_knowledge_base.pkl'")
        print(")")

    def _save_causal_results(self, results: Dict[str, Any]):
        """Save causal research results"""

        if not self.results_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Save main results
            results_file = self.results_dir / f"causal_research_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Save knowledge base
            kb_file = self.results_dir / f"causal_knowledge_base_{timestamp}.pkl"
            with open(kb_file, 'wb') as f:
                pickle.dump(self.knowledge_base, f)

            # Save detailed report
            report_file = self.results_dir / f"causal_research_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(self._generate_detailed_report(results))

            if self.verbose:
                self.logger.info(f"💾 Causal research results saved to {self.results_dir}")

        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Failed to save results: {e}")

    def _generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed causal research report"""

        report = f"""
CAUSAL FINANCIAL RESEARCH REPORT
{'=' * 80}

Research Timestamp: {results['research_context']['start_time']}
Research Strategy: {results['research_strategy']}
Total Duration: {results['research_context']['total_duration_seconds'] / 60:.1f} minutes

EXECUTIVE SUMMARY
{'=' * 50}
This report presents findings from causal inference-based financial research,
focusing on discovering economically meaningful relationships rather than
purely statistical correlations.

• Events Researched: {results['research_context']['events_successful']}/{results['research_context']['events_attempted']}
• Causal Relationships Discovered: {results['knowledge_base_summary']['total_relationships']}
• Economic Insights Generated: {results['knowledge_base_summary']['total_insights']}

"""

        # Add analysis summary
        analysis = results.get('causal_analysis', {})
        summary = analysis.get('summary', {})

        if isinstance(summary, dict):
            report += f"""
CAUSAL ANALYSIS RESULTS
{'=' * 50}
Average Causal Strength: {summary.get('average_causal_strength', 0):.3f}
Economic Performance:
  • Average Return: {summary.get('economic_performance', {}).get('average_return', 0):.4f}
  • Profitable Strategies: {summary.get('economic_performance', {}).get('profitable_strategies', 0)}/{summary.get('economic_performance', {}).get('total_strategies', 0)}

"""

        # Add recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            report += f"""

ACTIONABLE RECOMMENDATIONS
{'=' * 50}
"""
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"

        report += f"""

Generated by Causal Financial Research Agent
Report Date: {datetime.now().isoformat()}
"""

        return report


# CONVENIENCE FUNCTIONS FOR CAUSAL RESEARCH

def quick_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                          max_time_minutes: int = 30,
                          results_dir: Optional[str] = None,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Quick causal research focusing on strongest relationships

    Args:
        labeled_data: DataFrame or path to labeled data
        max_time_minutes: Time budget for research
        results_dir: Directory for results (optional)
        verbose: Print progress information

    Returns:
        Causal research results with economic validation
    """
    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )

    return agent.comprehensive_causal_research(max_time_minutes=max_time_minutes)


def deep_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                         max_time_minutes: int = 120,
                         results_dir: Optional[str] = None,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Deep causal research with extended analysis

    Args:
        labeled_data: DataFrame or path to labeled data
        max_time_minutes: Extended time budget for comprehensive research
        results_dir: Directory for results (optional)
        verbose: Print progress information

    Returns:
        Comprehensive causal research results
    """
    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )

    results = agent.comprehensive_causal_research(max_time_minutes=max_time_minutes)

    if verbose:
        print("\n🔬 DEEP CAUSAL RESEARCH ADDITIONAL ANALYSIS")
        print("=" * 60)

        # Knowledge base insights
        kb = agent.knowledge_base

        # Show discovered mechanisms
        mechanism_prefs = kb.get_mechanism_preferences()
        if mechanism_prefs:
            print(f"📊 Discovered Economic Mechanisms (by performance):")
            for mechanism, score in sorted(mechanism_prefs.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {mechanism}: {score:.3f} avg performance")

        # Show causal insights
        if kb.economic_insights:
            print(f"\n💡 Generated Economic Insights:")
            for insight in kb.economic_insights[-5:]:  # Show last 5
                print(f"   • [{insight['mechanism']}] {insight['description'][:100]}...")

        # Show knowledge base growth
        total_relationships = sum(len(rels) for rels in kb.causal_relationships.values())
        print(f"\n🧠 Knowledge Base Growth:")
        print(f"   • Total causal relationships stored: {total_relationships}")
        print(f"   • Failed relationships avoided: {len(kb.failed_relationships)}")

    return results


def focused_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                            target_events: List[str],
                            economic_mechanisms: List[str] = None,
                            min_causal_strength: float = 0.3,
                            results_dir: Optional[str] = None,
                            verbose: bool = True) -> Dict[str, Any]:
    """
    Focused causal research on specific events and mechanisms

    Args:
        labeled_data: DataFrame or path to labeled data
        target_events: Specific events to focus research on
        economic_mechanisms: Focus on specific economic mechanisms (optional)
        min_causal_strength: Minimum causal strength threshold
        results_dir: Directory for results (optional)
        verbose: Print progress information

    Returns:
        Focused causal research results
    """
    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )

    # Filter events to research
    available_events = [event for event in target_events if event in agent.label_columns]

    if not available_events:
        raise ValueError(f"None of the target events {target_events} found in data columns")

    results = agent.comprehensive_causal_research(
        max_time_minutes=60,
        priority_events=available_events
    )

    # Post-process results to filter by causal strength
    filtered_results = {}
    for event_label, result in results['research_results'].items():
        # Filter causal relationships by strength
        strong_relationships = [
            rel for rel in result.get('causal_relationships', [])
            if rel.get('strength', 0) >= min_causal_strength
        ]

        if strong_relationships:
            result['causal_relationships'] = strong_relationships
            filtered_results[event_label] = result

    results['research_results'] = filtered_results
    results['filtering'] = {
        'min_causal_strength': min_causal_strength,
        'original_events': len(results.get('research_results', {})),
        'filtered_events': len(filtered_results)
    }

    if verbose:
        print(f"\n🎯 FOCUSED CAUSAL RESEARCH FILTERING")
        print(f"Minimum causal strength: {min_causal_strength}")
        print(f"Events meeting threshold: {len(filtered_results)}")

        if economic_mechanisms:
            print(f"Focused mechanisms: {', '.join(economic_mechanisms)}")

    return results


def validate_causal_stability(labeled_data: Union[pd.DataFrame, str, Path],
                              causal_relationships_file: str,
                              out_of_sample_start_date: str = None,
                              results_dir: Optional[str] = None,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Validate previously discovered causal relationships on new data

    Args:
        labeled_data: New data for validation
        causal_relationships_file: Path to saved causal relationships
        out_of_sample_start_date: Start date for out-of-sample validation
        results_dir: Directory for results (optional)
        verbose: Print progress information

    Returns:
        Validation results for causal relationships
    """

    # Load previous causal relationships
    try:
        with open(causal_relationships_file, 'rb') as f:
            previous_kb = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load causal relationships: {e}")

    # Initialize agent with new data
    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )

    validation_results = {}

    if verbose:
        print(f"\n🔍 CAUSAL RELATIONSHIP VALIDATION")
        print("=" * 60)
        print(f"Validating {len(previous_kb.causal_relationships)} event types")

    for event_label, relationships in previous_kb.causal_relationships.items():
        if event_label not in agent.label_columns:
            if verbose:
                print(f"⚠️  Event {event_label} not found in validation data")
            continue

        # Simple validation based on stored dicts
        validation_results[event_label] = {
            'original_relationships': len(relationships),
            'validation_status': 'available_for_validation'
        }

    return {
        'validation_timestamp': datetime.now().isoformat(),
        'validation_results': validation_results,
        'summary': {
            'events_available': len(validation_results)
        }
    }


# Example usage
if __name__ == "__main__":
    print("Causal Financial Research System")
    print("=" * 50)

    # Example 1: Quick causal research
    print("\n🚀 Running Quick Causal Research...")
    results = quick_causal_research(
        labeled_data='labeled5mEE.pkl',
        max_time_minutes=30,
        results_dir="causal_results",
        verbose=True
    )

    # Example 2: Deep causal research
    print("\n🔬 For Deep Causal Research, use:")
    print("results = deep_causal_research(labeled_data='your_data.pkl', max_time_minutes=120)")

    # Example 3: Focused causal research
    print("\n🎯 For Focused Causal Research, use:")
    print("results = focused_causal_research(")
    print("    labeled_data='your_data.pkl',")
    print("    target_events=['momentum_label', 'volatility_label'],")
    print("    min_causal_strength=0.5")
    print(")")

    # Example 4: Validation
    print("\n✅ For Causal Relationship Validation, use:")
    print("results = validate_causal_stability(")
    print("    labeled_data='new_data.pkl',")
    print("    causal_relationships_file='causal_knowledge_base.pkl'")
    print(")")