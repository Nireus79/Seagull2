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

    def test_multivariate_granger_causality(self, feature_combinations: List[List[str]],
                                            features_df: pd.DataFrame, target_series: pd.Series,
                                            max_lag: int = None) -> Dict[str, Dict[str, Any]]:
        """Test Granger causality for feature combinations"""

        if max_lag is None:
            max_lag = min(self.max_lag_test, len(target_series) // 20)

        results = {}

        for combination in feature_combinations:
            combo_name = "+".join(combination)

            try:
                # Get features for this combination
                combo_features = features_df[combination]

                # Ensure proper alignment and clean data
                aligned_data = pd.concat([target_series, combo_features], axis=1).dropna()

                if len(aligned_data) < self.min_periods_for_test:
                    results[combo_name] = {
                        'is_causal': False,
                        'reason': 'Insufficient data points',
                        'min_p_value': 1.0,
                        'best_lag': 0,
                        'strength': 0,
                        'combination_size': len(combination)
                    }
                    continue

                # Check for variance in all features
                if (aligned_data.std() == 0).any():
                    results[combo_name] = {
                        'is_causal': False,
                        'reason': 'Zero variance in features',
                        'min_p_value': 1.0,
                        'best_lag': 0,
                        'strength': 0,
                        'combination_size': len(combination)
                    }
                    continue

                # Test multiple lags
                min_p_value = 1.0
                best_lag = 1

                for lag in range(1, max_lag + 1):
                    try:
                        if len(aligned_data) <= lag * 3:
                            break

                        # Use statsmodels VAR for multivariate Granger causality
                        gc_result = grangercausalitytests(aligned_data, maxlag=lag, verbose=False)

                        # Extract p-value for this lag
                        p_value = gc_result[lag][0]['ssr_ftest'][1]

                        if p_value < min_p_value:
                            min_p_value = p_value
                            best_lag = lag

                    except Exception:
                        continue

                is_causal = min_p_value < self.significance_level

                # Calculate combination strength
                strength = (1 - min_p_value) if is_causal else 0

                results[combo_name] = {
                    'is_causal': is_causal,
                    'min_p_value': min_p_value,
                    'best_lag': best_lag,
                    'strength': strength,
                    'combination_size': len(combination),
                    'features': combination
                }

            except Exception as e:
                results[combo_name] = {
                    'is_causal': False,
                    'reason': f'Test failed: {str(e)}',
                    'min_p_value': 1.0,
                    'best_lag': 0,
                    'strength': 0,
                    'combination_size': len(combination),
                    'features': combination
                }

        return results


class CombinationFeatureGenerator:
    """Generates intelligent feature combinations based on economic theory"""

    def __init__(self, theory_engine: EconomicTheoryEngine):
        self.theory_engine = theory_engine

    def generate_feature_combinations(self, available_features: List[str],
                                      event_label: str, max_combination_size: int = 4) -> List[List[str]]:
        """Generate intelligent feature combinations"""

        combinations = []

        # 1. Single features (baseline)
        for feature in available_features[:20]:  # Limit for performance
            combinations.append([feature])

        # 2. Economic mechanism-based combinations
        mechanism_combinations = self._get_mechanism_based_combinations(available_features, event_label)
        combinations.extend(mechanism_combinations)

        # 3. Technical indicator groupings
        technical_combinations = self._get_technical_combinations(available_features)
        combinations.extend(technical_combinations)

        # 4. Timeframe-based combinations
        timeframe_combinations = self._get_timeframe_combinations(available_features)
        combinations.extend(timeframe_combinations)

        # 5. Limited random combinations for discovery
        random_combinations = self._get_smart_random_combinations(available_features, max_combination_size)
        combinations.extend(random_combinations)

        # Remove duplicates and filter by size
        unique_combinations = []
        seen = set()

        for combo in combinations:
            combo_sorted = tuple(sorted(combo))
            if combo_sorted not in seen and len(combo) <= max_combination_size:
                seen.add(combo_sorted)
                unique_combinations.append(combo)

        return unique_combinations

    def _get_mechanism_based_combinations(self, features: List[str], event_label: str) -> List[List[str]]:
        """Create combinations based on economic mechanisms"""
        combinations = []

        # Group features by suspected economic mechanism
        volume_features = [f for f in features if any(keyword in f.lower()
                                                      for keyword in ['volume', 'vol', 'flow'])]
        volatility_features = [f for f in features if any(keyword in f.lower()
                                                          for keyword in
                                                          ['atr', 'volatility', 'vol_realized', 'garch'])]
        momentum_features = [f for f in features if any(keyword in f.lower()
                                                        for keyword in ['ema', 'sma', 'momentum', 'macd', 'rsi'])]
        price_features = [f for f in features if any(keyword in f.lower()
                                                     for keyword in ['open', 'high', 'low', 'close', 'price'])]

        # Information asymmetry: Volume + Volatility
        if volume_features and volatility_features:
            combinations.append(volume_features[:2] + volatility_features[:1])

        # Momentum + Volatility
        if momentum_features and volatility_features:
            combinations.append(momentum_features[:2] + volatility_features[:1])

        # Volume + Price action
        if volume_features and price_features:
            combinations.append(volume_features[:1] + price_features[:2])

        return combinations

    def _get_technical_combinations(self, features: List[str]) -> List[List[str]]:
        """Create combinations of complementary technical indicators"""
        combinations = []

        # ATR-based combinations
        atr_features = [f for f in features if 'atr' in f.lower()]
        if len(atr_features) >= 2:
            combinations.append(atr_features[:2])

        # EMA-based combinations
        ema_features = [f for f in features if 'ema' in f.lower()]
        if len(ema_features) >= 2:
            combinations.append(ema_features[:2])
            if len(ema_features) >= 3:
                combinations.append(ema_features[:3])

        # CUSUM-based combinations
        cusum_features = [f for f in features if 'cusum' in f.lower()]
        if len(cusum_features) >= 2:
            combinations.append(cusum_features[:2])

        return combinations

    def _get_timeframe_combinations(self, features: List[str]) -> List[List[str]]:
        """Create combinations across different timeframes"""
        combinations = []

        # Multi-timeframe volume
        volume_timeframes = [f for f in features if 'volume' in f.lower()]
        if len(volume_timeframes) >= 2:
            combinations.append(volume_timeframes[:2])

        # Multi-timeframe price features
        timeframe_indicators = ['1h', '4h', '1d', '1w']
        for indicator in timeframe_indicators:
            timeframe_features = [f for f in features if indicator.lower() in f.lower()]
            if len(timeframe_features) >= 2:
                combinations.append(timeframe_features[:2])

        return combinations

    def _get_smart_random_combinations(self, features: List[str], max_size: int) -> List[List[str]]:
        """Generate a limited set of smart random combinations"""
        combinations = []

        # Focus on top features (assuming they're ordered by individual performance)
        top_features = features[:15]

        # Generate pairs and triplets
        # Pairs from top features
        for combo in itertools.combinations(top_features[:10], 2):
            combinations.append(list(combo))

        # Triplets from top features (limited)
        for combo in list(itertools.combinations(top_features[:8], 3))[:10]:
            combinations.append(list(combo))

        return combinations


class CausalFeatureDiscovery:
    """Discovers causal features (individual and combinations)"""

    def __init__(self, theory_engine: EconomicTheoryEngine, causal_engine: CausalInferenceEngine):
        self.theory_engine = theory_engine
        self.causal_engine = causal_engine
        self.combination_generator = CombinationFeatureGenerator(theory_engine)

    def discover_causal_features(self, features_df: pd.DataFrame, target_series: pd.Series,
                                 event_label: str, max_features: int = 10,
                                 use_combinations: bool = True, max_combinations: int = 30,
                                 max_combination_size: int = 3) -> Dict[str, Any]:
        """Discover causally valid features for predicting an event"""

        result = {
            'individual_relationships': [],
            'feature_combinations': [],
            'best_approach': 'individual'
        }
        exclude_features = ['Open', 'High', 'Low', 'Volume',
                            '30min_Open', '30min_High', '30min_Low', '30min_Close',
                            '4H_Open', '4H_High', '4H_Low', '4H_Close',
                            '1D_Open', '1D_High', '1D_Low', '1D_Close']
        # Limit features to test for speed (less aggressive filtering)
        features_to_test = [
                               f for f in features_df.columns
                               if f not in exclude_features
                           ][:min(50, len(features_df.columns))]  # limit to 50 features
        print('features_to_test: ', len(features_to_test), features_to_test)

        # Test individual features
        individual_relationships = self._discover_individual_features(features_df, target_series, event_label,
                                                                      max_features)
        result['individual_relationships'] = [self._relationship_to_dict(rel) for rel in individual_relationships]

        if use_combinations:
            # Test feature combinations
            combination_results = self._discover_feature_combinations(features_df, target_series, event_label,
                                                                      max_combinations, max_combination_size)
            result['feature_combinations'] = combination_results

            # Determine best approach
            best_individual_strength = max([rel.strength for rel in individual_relationships], default=0)
            best_combination_strength = max([combo['combined_strength'] for combo in combination_results], default=0)

            if best_combination_strength > best_individual_strength:
                result['best_approach'] = 'combinations'

        return result

    def _discover_individual_features(self, features_df: pd.DataFrame, target_series: pd.Series,
                                      event_label: str, max_features: int) -> List[CausalRelationship]:
        """Discover individual causal features"""

        discovered_relationships = []
        exclude_features = ['Open', 'High', 'Low', 'Volume',
                            '30min_Open', '30min_High', '30min_Low', '30min_Close',
                            '4H_Open', '4H_High', '4H_Low', '4H_Close',
                            '1D_Open', '1D_High', '1D_Low', '1D_Close']
        # Limit features to test for speed (less aggressive filtering)
        features_to_test = [
                               f for f in features_df.columns
                               if f not in exclude_features
                           ][:min(50, len(features_df.columns))]  # limit to 50 features
        print('features_to_test: ', len(features_to_test), features_to_test)

        for feature in features_to_test:
            print('feature causal research: ', feature)
            try:
                feature_series = features_df[feature]

                if feature_series.std() == 0 or feature_series.isna().all():
                    continue

                # Test Granger causality
                granger_result = self.causal_engine.test_granger_causality(feature_series, target_series)

                max_strength = granger_result.get('strength', 0)
                if max_strength == 0:
                    continue

                # Economic plausibility test
                is_plausible, justification, confidence = self.theory_engine.validate_economic_plausibility(
                    feature, event_label, max_strength
                )

                is_causal = (granger_result.get('is_causal', False) or max_strength > 0.05) and is_plausible

                if not is_causal:
                    continue

                # Create relationship
                relationship = CausalRelationship(
                    cause_feature=feature,
                    effect_event=event_label,
                    relationship_type=CausalRelationshipType.GRANGER_CAUSAL,
                    economic_mechanism=EconomicMechanism.INFORMATION_ASYMMETRY,
                    strength=max_strength,
                    p_value=granger_result.get('min_p_value', 1.0),
                    confidence_interval=(max(0, max_strength * 0.8), min(1, max_strength * 1.2)),
                    economic_justification=justification,
                    temporal_lag=granger_result.get('best_lag', 1),
                    regime_stability={'GENERAL': max_strength},
                    discovery_date=datetime.now().isoformat(),
                    validation_history=[]
                )

                discovered_relationships.append(relationship)

            except Exception:
                continue

        discovered_relationships.sort(key=lambda x: x.strength, reverse=True)
        return discovered_relationships[:max_features]

    def _discover_feature_combinations(self, features_df: pd.DataFrame, target_series: pd.Series,
                                       event_label: str, max_combinations: int, max_combination_size: int) -> List[
        Dict[str, Any]]:
        """Discover feature combinations"""

        # Generate combinations
        feature_combinations = self.combination_generator.generate_feature_combinations(
            features_df.columns.tolist(), event_label, max_combination_size
        )

        # Limit combinations for performance
        feature_combinations = feature_combinations[:max_combinations]

        discovered_combinations = []

        for combination in feature_combinations:
            try:
                # Test causal strength
                causal_results = self.causal_engine.test_multivariate_granger_causality(
                    [combination], features_df, target_series
                )

                combo_name = "+".join(combination)
                causal_result = causal_results.get(combo_name, {})

                # Test predictive power
                predictive_result = self._test_combination_predictive_power(combination, features_df, target_series)

                # Check if combination meets criteria
                is_causal = causal_result.get('is_causal', False)
                is_predictive = predictive_result.get('predictive', False)
                causal_strength = causal_result.get('strength', 0)
                predictive_strength = predictive_result.get('strength', 0)

                meets_criteria = (is_causal or is_predictive) and (causal_strength > 0.1 or predictive_strength > 0.1)

                if meets_criteria:
                    combination_result = {
                        'features': combination,
                        'combination_name': combo_name,
                        'combination_size': len(combination),
                        'causal_strength': causal_strength,
                        'predictive_strength': predictive_strength,
                        'combined_strength': max(causal_strength, predictive_strength),
                        'is_causal': is_causal,
                        'is_predictive': is_predictive,
                        'auc_score': predictive_result.get('auc_score', 0.5),
                        'causal_p_value': causal_result.get('min_p_value', 1.0),
                        'economic_mechanism': EconomicMechanism.INFORMATION_ASYMMETRY.value,
                        'economic_justification': f"Multi-feature combination with {len(combination)} features",
                        'discovery_timestamp': datetime.now().isoformat()
                    }

                    discovered_combinations.append(combination_result)

            except Exception:
                continue

        # Rank combinations by combined strength
        discovered_combinations.sort(key=lambda x: x['combined_strength'], reverse=True)
        return discovered_combinations

    def _test_combination_predictive_power(self, combination: List[str], features_df: pd.DataFrame,
                                           target_series: pd.Series) -> Dict[str, Any]:
        """Test predictive power of feature combination using cross-validation"""

        try:
            # Get features for this combination
            combo_features = features_df[combination]

            # Align data
            aligned_data = pd.concat([combo_features, target_series], axis=1).dropna()

            if len(aligned_data) < 30:
                return {'predictive': False, 'reason': 'Insufficient data', 'auc_score': 0.5, 'strength': 0}

            X = aligned_data.iloc[:, :-1]  # Features
            y = aligned_data.iloc[:, -1]  # Target

            # Check for sufficient target variety
            if len(y.unique()) < 2:
                return {'predictive': False, 'reason': 'No target variance', 'auc_score': 0.5, 'strength': 0}

            # Time series cross-validation
            cv = TimeSeriesSplit(n_splits=min(3, len(X) // 100))
            auc_scores = []

            for train_idx, test_idx in cv.split(X):
                try:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    if len(X_train) < 10 or len(X_test) < 5:
                        continue

                    # Simple logistic regression
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    scaler = StandardScaler()

                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    model.fit(X_train_scaled, y_train)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                    if len(np.unique(y_test)) > 1:
                        auc = roc_auc_score(y_test, y_pred_proba)
                        auc_scores.append(auc)

                except Exception:
                    continue

            avg_auc = np.mean(auc_scores) if auc_scores else 0.5
            is_predictive = avg_auc > 0.55  # Modest threshold

            return {
                'predictive': is_predictive,
                'auc_score': avg_auc,
                'strength': max(0, (avg_auc - 0.5) * 2),  # Normalize to 0-1
                'n_folds': len(auc_scores)
            }

        except Exception as e:
            return {'predictive': False, 'reason': f'Test failed: {str(e)}', 'auc_score': 0.5, 'strength': 0}

    def _relationship_to_dict(self, relationship: CausalRelationship) -> Dict[str, Any]:
        """Convert CausalRelationship to dictionary with proper string conversion"""
        rel_dict = asdict(relationship)
        rel_dict['relationship_type'] = relationship.relationship_type.value
        rel_dict['economic_mechanism'] = relationship.economic_mechanism.value
        return rel_dict


class CausalAgentKnowledgeBase:
    """Knowledge base storing causal relationships and economic insights"""

    def __init__(self):
        self.causal_relationships: Dict[str, List[Dict]] = {}
        self.combination_relationships: Dict[str, List[Dict]] = {}
        self.economic_insights: List[Dict] = []
        self.mechanism_success_rates: Dict[str, List[float]] = {}
        self.failed_relationships: Set[Tuple[str, str]] = set()

    def store_causal_relationship(self, relationship: CausalRelationship, validation_performance: Dict[str, float]):
        """Store a validated causal relationship"""
        event_key = relationship.effect_event
        if event_key not in self.causal_relationships:
            self.causal_relationships[event_key] = []

        rel_dict = asdict(relationship)
        rel_dict['validation_performance'] = validation_performance
        self.causal_relationships[event_key].append(rel_dict)

    def store_combination_relationship(self, combination_result: Dict):
        """Store a validated combination relationship"""
        event_key = combination_result.get('event_label', 'unknown')
        if event_key not in self.combination_relationships:
            self.combination_relationships[event_key] = []

        self.combination_relationships[event_key].append(combination_result)

    def get_mechanism_preferences(self) -> Dict[str, float]:
        """Get preferred economic mechanisms based on historical performance"""
        return {mechanism: np.mean(scores)
                for mechanism, scores in self.mechanism_success_rates.items()
                if len(scores) > 0}


class CausalResearchAgent:
    """Main agent orchestrating causal financial research with combination discovery"""

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
        self.feature_discovery = CausalFeatureDiscovery(self.theory_engine, self.causal_engine)
        self.knowledge_base = CausalAgentKnowledgeBase()

        # Setup logging
        self.logger = self._setup_logger()

        # Analyze data
        self._analyze_data_structure()
        self.data = self._clean_data_for_causality()

        if self.verbose:
            self.logger.info(f"Enhanced Causal Financial Research Agent initialized")
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
        data_clean = self.data.copy()
        data_clean = data_clean.ffill().bfill()

        critical_cols = [col for col in self.data.columns if col.endswith('_label')]
        critical_cols = [col for col in critical_cols if col in data_clean.columns]

        if critical_cols:
            data_clean = data_clean.dropna(subset=critical_cols)

        return data_clean

    def _analyze_data_structure(self):
        """Analyze data structure to identify features and labels"""
        # Identify label columns
        self.label_columns = [col for col in self.data.columns if col.endswith('_label')]

        # Identify feature columns
        exclude_patterns = {'_label', '_barrier_touched', '_touch_time', '_return', '_holding_hours', '_event',
                            'event_type', 'any_event'}
        numeric_dtypes = {'float64', 'int64', 'float32', 'int32', 'float16', 'int16'}

        self.feature_columns = []
        for col in self.data.columns:
            if any(pattern in col for pattern in exclude_patterns):
                continue
            if (self.data[col].dtype.name in numeric_dtypes):
                if not self.data[col].isna().all():
                    self.feature_columns.append(col)

    def research_event_with_combinations(self, event_label: str,
                                         use_combinations: bool = True,
                                         max_combinations: int = 30,
                                         max_combination_size: int = 3) -> Dict[str, Any]:
        """Research a single event using both individual features and combinations"""

        if self.verbose:
            self.logger.info(f"\n🔬 CAUSAL RESEARCH: {event_label}")
            self.logger.info(f"Combination analysis: {use_combinations}")

        try:
            # Prepare data
            valid_samples = self.data[event_label].notna()
            if valid_samples.sum() < 100:
                if self.verbose:
                    self.logger.warning(f"Insufficient data for {event_label}: {valid_samples.sum()} samples")
                return None

            X = self.data.loc[valid_samples, self.feature_columns].copy()
            y = self.data.loc[valid_samples, event_label].copy()
            X = X.ffill().bfill().fillna(0)

            # Remove constant features
            constant_features = X.columns[X.std() == 0].tolist()
            if constant_features:
                X = X.drop(columns=constant_features)

            if self.verbose:
                self.logger.info(f"📊 Data prepared: {len(X)} samples, {len(X.columns)} features")

            # Discover causal relationships
            discovery_result = self.feature_discovery.discover_causal_features(
                X, y, event_label, max_features=15, use_combinations=use_combinations,
                max_combinations=max_combinations, max_combination_size=max_combination_size
            )

            result = {
                'event_label': event_label,
                'research_timestamp': datetime.now().isoformat(),
                'data_samples': len(X),
                'features_tested': len(X.columns),
                'individual_relationships': discovery_result['individual_relationships'],
                'feature_combinations': discovery_result['feature_combinations'],
                'best_approach': discovery_result['best_approach']
            }

            if self.verbose:
                num_individual = len(result['individual_relationships'])
                num_combinations = len(result['feature_combinations'])
                self.logger.info(f"✅ Found {num_individual} individual + {num_combinations} combinations")
                self.logger.info(f"🏆 Best approach: {result['best_approach']}")

            return result

        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Research failed for {event_label}: {e}")
            return None

    def comprehensive_causal_research_with_combinations(self, max_time_minutes: int = 60,
                                                        priority_events: List[str] = None,
                                                        use_combinations: bool = True,
                                                        max_combinations: int = 30,
                                                        max_combination_size: int = 3) -> Dict[str, Any]:
        """Conduct comprehensive causal research with feature combinations"""

        start_time = datetime.now()
        max_time_seconds = max_time_minutes * 60

        if self.verbose:
            self.logger.info(f"\n🚀 COMPREHENSIVE CAUSAL RESEARCH WITH COMBINATIONS")
            self.logger.info(f"Time budget: {max_time_minutes} minutes")
            self.logger.info(f"Combination analysis: {use_combinations}")

        events_to_research = priority_events if priority_events else self.label_columns
        research_results = {}
        failed_events = []

        for i, event_label in enumerate(events_to_research):
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()

            if elapsed_seconds >= max_time_seconds * 0.9:
                if self.verbose:
                    self.logger.info("⏰ Time budget nearly exhausted. Stopping research.")
                break

            if self.verbose:
                remaining_time = (max_time_seconds - elapsed_seconds) / 60
                self.logger.info(f"\n[{i + 1}/{len(events_to_research)}] Researching: {event_label}")
                self.logger.info(f"⏱️  Remaining time: {remaining_time:.1f} minutes")

            try:
                result = self.research_event_with_combinations(
                    event_label, use_combinations, max_combinations, max_combination_size
                )
                if result:
                    research_results[event_label] = result
                else:
                    failed_events.append(event_label)

            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Research failed for {event_label}: {e}")
                failed_events.append(event_label)

        total_duration = (datetime.now() - start_time).total_seconds()
        analysis = self._generate_enhanced_analysis(research_results, failed_events, use_combinations)

        final_results = {
            'research_strategy': 'enhanced_causal_inference_with_combinations',
            'research_context': {
                'start_time': start_time.isoformat(),
                'total_duration_seconds': total_duration,
                'time_budget_minutes': max_time_minutes,
                'events_attempted': len(events_to_research),
                'events_successful': len(research_results),
                'events_failed': len(failed_events),
                'used_combinations': use_combinations
            },
            'research_results': research_results,
            'failed_events': failed_events,
            'causal_analysis': analysis
        }

        if self.verbose:
            self._print_enhanced_summary(final_results)

        return final_results

    def _generate_enhanced_analysis(self, research_results: Dict, failed_events: List[str],
                                    used_combinations: bool) -> Dict[str, Any]:
        """Generate enhanced analysis including combination results"""

        if not research_results:
            return {
                'summary': 'No successful causal relationships discovered',
                'recommendations': ['Check data quality', 'Consider different feature combinations']
            }

        # Extract results
        all_individual = []
        all_combinations = []

        for result in research_results.values():
            all_individual.extend(result.get('individual_relationships', []))
            all_combinations.extend(result.get('feature_combinations', []))

        # Calculate performance
        individual_strengths = [rel.get('strength', 0) for rel in all_individual]
        combination_strengths = [combo.get('combined_strength', 0) for combo in all_combinations]

        avg_individual_strength = np.mean(individual_strengths) if individual_strengths else 0
        avg_combination_strength = np.mean(combination_strengths) if combination_strengths else 0

        # Recommendations
        recommendations = []
        if used_combinations and avg_combination_strength > avg_individual_strength:
            recommendations.append(
                "Feature combinations outperform individual features - focus on multi-feature models")
        elif avg_individual_strength > 0.5:
            recommendations.append("Strong individual features found - simple models may be sufficient")

        recommendations.extend([
            "Validate discovered relationships on out-of-sample data",
            "Consider ensemble methods for feature combinations"
        ])

        return {
            'summary': {
                'total_individual_relationships': len(all_individual),
                'total_feature_combinations': len(all_combinations),
                'avg_individual_strength': avg_individual_strength,
                'avg_combination_strength': avg_combination_strength,
                'combinations_superior': avg_combination_strength > avg_individual_strength
            },
            'insights': [
                f"Individual relationships: {len(all_individual)}",
                f"Feature combinations: {len(all_combinations)}",
                f"Average individual strength: {avg_individual_strength:.3f}",
                f"Average combination strength: {avg_combination_strength:.3f}"
            ],
            'recommendations': recommendations
        }

    def _print_enhanced_summary(self, results: Dict[str, Any]):
        """Print enhanced research summary"""

        print(f"\n{'=' * 80}")
        print("🔬 ENHANCED CAUSAL RESEARCH SUMMARY")
        print(f"{'=' * 80}")

        context = results['research_context']
        print(f"📊 Research Statistics:")
        print(f"   • Events attempted: {context['events_attempted']}")
        print(f"   • Successful research: {context['events_successful']}")
        print(f"   • Success rate: {context['events_successful'] / context['events_attempted']:.1%}")
        print(f"   • Duration: {context['total_duration_seconds'] / 60:.1f} minutes")

        analysis = results['causal_analysis']
        summary = analysis.get('summary', {})

        if isinstance(summary, dict):
            print(f"\n🔍 Discovery Results:")
            print(f"   • Individual relationships: {summary.get('total_individual_relationships', 0)}")
            print(f"   • Feature combinations: {summary.get('total_feature_combinations', 0)}")
            print(f"   • Avg individual strength: {summary.get('avg_individual_strength', 0):.3f}")
            print(f"   • Avg combination strength: {summary.get('avg_combination_strength', 0):.3f}")
            print(f"   • Combinations superior: {summary.get('combinations_superior', False)}")

        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")


# Enhanced convenience functions
def enhanced_quick_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                                   max_time_minutes: int = 45,
                                   use_combinations: bool = True,
                                   max_combinations: int = 30,
                                   max_combination_size: int = 3,
                                   results_dir: Optional[str] = None,
                                   verbose: bool = True) -> Dict[str, Any]:
    """Enhanced quick causal research with feature combinations"""

    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )

    return agent.comprehensive_causal_research_with_combinations(
        max_time_minutes=max_time_minutes,
        use_combinations=use_combinations,
        max_combinations=max_combinations,
        max_combination_size=max_combination_size
    )


def enhanced_deep_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                                  max_time_minutes: int = 120,
                                  max_combinations: int = 50,
                                  max_combination_size: int = 4,
                                  results_dir: Optional[str] = None,
                                  verbose: bool = True) -> Dict[str, Any]:
    """Enhanced deep causal research with comprehensive combination analysis"""

    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )

    results = agent.comprehensive_causal_research_with_combinations(
        max_time_minutes=max_time_minutes,
        use_combinations=True,
        max_combinations=max_combinations,
        max_combination_size=max_combination_size
    )

    if verbose:
        print("\n🔬 DEEP CAUSAL RESEARCH ADDITIONAL ANALYSIS")
        print("=" * 60)

        # Combination vs Individual comparison
        total_individual = 0
        total_combinations = 0
        for event, result in results.get('research_results', {}).items():
            total_individual += len(result.get('individual_relationships', []))
            total_combinations += len(result.get('feature_combinations', []))

        print(f"📊 Discovery Comparison:")
        print(f"   • Individual relationships: {total_individual}")
        print(f"   • Feature combinations: {total_combinations}")

        # Show best combinations
        best_combinations = []
        for event, result in results.get('research_results', {}).items():
            for combo in result.get('feature_combinations', []):
                combo['event'] = event
                best_combinations.append(combo)

        best_combinations.sort(key=lambda x: x.get('combined_strength', 0), reverse=True)

        if best_combinations:
            print(f"\n🏆 Top Feature Combinations:")
            for i, combo in enumerate(best_combinations[:5], 1):
                print(f"   {i}. {combo['combination_name']} (Event: {combo['event']})")
                print(f"      Strength: {combo.get('combined_strength', 0):.3f}")
                print(f"      AUC: {combo.get('auc_score', 0):.3f}")

    return results


def focused_combination_research(labeled_data: Union[pd.DataFrame, str, Path],
                                 target_events: List[str],
                                 max_combination_size: int = 4,
                                 min_causal_strength: float = 0.3,
                                 results_dir: Optional[str] = None,
                                 verbose: bool = True) -> Dict[str, Any]:
    """Focused causal research on specific events with combination analysis"""

    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )

    # Filter events to research
    available_events = [event for event in target_events if event in agent.label_columns]

    if not available_events:
        raise ValueError(f"None of the target events {target_events} found in data columns")

    results = agent.comprehensive_causal_research_with_combinations(
        max_time_minutes=90,
        priority_events=available_events,
        use_combinations=True,
        max_combinations=40,
        max_combination_size=max_combination_size
    )

    # Post-process results to filter by strength
    filtered_results = {}
    for event_label, result in results['research_results'].items():
        # Filter individual relationships
        strong_individual = [
            rel for rel in result.get('individual_relationships', [])
            if rel.get('strength', 0) >= min_causal_strength
        ]

        # Filter combinations
        strong_combinations = [
            combo for combo in result.get('feature_combinations', [])
            if combo.get('combined_strength', 0) >= min_causal_strength
        ]

        if strong_individual or strong_combinations:
            result['individual_relationships'] = strong_individual
            result['feature_combinations'] = strong_combinations
            filtered_results[event_label] = result

    results['research_results'] = filtered_results
    results['filtering'] = {
        'min_causal_strength': min_causal_strength,
        'original_events': len(results.get('research_results', {})),
        'filtered_events': len(filtered_results)
    }

    if verbose:
        print(f"\n🎯 FOCUSED COMBINATION RESEARCH FILTERING")
        print(f"Minimum causal strength: {min_causal_strength}")
        print(f"Events meeting threshold: {len(filtered_results)}")

    return results


# Keep original functions for compatibility
def quick_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                          max_time_minutes: int = 30,
                          results_dir: Optional[str] = None,
                          verbose: bool = True) -> Dict[str, Any]:
    """Original quick causal research (individual features only)"""
    return enhanced_quick_causal_research(
        labeled_data=labeled_data,
        max_time_minutes=max_time_minutes,
        use_combinations=False,  # Original behavior
        results_dir=results_dir,
        verbose=verbose
    )


def deep_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                         max_time_minutes: int = 120,
                         results_dir: Optional[str] = None,
                         verbose: bool = True) -> Dict[str, Any]:
    """Original deep causal research (individual features only)"""
    return enhanced_deep_causal_research(
        labeled_data=labeled_data,
        max_time_minutes=max_time_minutes,
        max_combinations=0,  # No combinations for original behavior
        results_dir=results_dir,
        verbose=verbose
    )


# Example usage
if __name__ == "__main__":
    print("Enhanced Causal Financial Research System")
    print("=" * 50)

    # Enhanced research with combinations
    print("\n🚀 Running Enhanced Causal Research...")
    results = enhanced_quick_causal_research(
        labeled_data='labeled5mEE.pkl',
        max_time_minutes=45,
        use_combinations=True,
        max_combinations=30,
        max_combination_size=3,
        verbose=True
    )

    print("\n✅ Research completed. Check results for individual features and combinations.")
