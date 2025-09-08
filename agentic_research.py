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
from sklearn.feature_selection import mutual_info_regression
import itertools
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, asdict
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools
from enum import Enum
from abc import ABC, abstractmethod

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, validation_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, log_loss
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import (
    mutual_info_classif, SelectKBest, RFE, RFECV,
    chi2, f_classif
)

# Models for binary classification
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Advanced models
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Market regime classification"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class EventType(Enum):
    """Statistical event types"""
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"
    UNKNOWN = "unknown"


@dataclass
class AgentMemory:
    """Agent's memory of successful strategies"""
    successful_combinations: Dict[str, Dict[str, float]]
    failed_combinations: Set[Tuple[str, str]]  # (event_type, feature_combo)
    regime_preferences: Dict[MarketRegime, Dict[str, float]]
    model_performance_history: Dict[str, deque]
    feature_importance_trends: Dict[str, deque]
    economic_performance_memory: Dict[str, Dict[str, float]]
    last_updated: datetime


@dataclass
class AgentInsight:
    """Represents an insight discovered by the agent"""
    insight_type: str
    description: str
    confidence: float
    supporting_evidence: Dict[str, Any]
    timestamp: datetime
    actionable: bool


@dataclass
class ResearchStrategy:
    """Strategy for conducting research"""
    priority_events: List[str]
    feature_selection_method: str
    model_preferences: List[str]
    time_budget_allocation: Dict[str, float]
    exploration_vs_exploitation: float  # 0 = full exploitation, 1 = full exploration
    economic_focus: bool


class MarketContextAnalyzer:
    """Analyzes market context to inform agent decisions"""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.ohlc_cols = ['Open', 'High', 'Low', 'Close']
        self.volume_col = 'Volume' if 'Volume' in data.columns else None

    def detect_market_regime(self, lookback_days: int = 30) -> MarketRegime:
        """Detect current market regime"""
        if not all(col in self.data.columns for col in self.ohlc_cols):
            return MarketRegime.UNKNOWN

        recent_data = self.data.tail(lookback_days)

        # Calculate regime indicators
        returns = recent_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        trend = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * 100

        # Volume analysis if available
        volume_trend = 0
        if self.volume_col and self.volume_col in recent_data.columns:
            volume_ma = recent_data[self.volume_col].rolling(10).mean()
            volume_trend = (volume_ma.iloc[-1] / volume_ma.iloc[0] - 1) * 100

        # Regime classification logic
        high_vol_threshold = 25  # 25% annualized volatility
        crisis_vol_threshold = 40  # 40% annualized volatility
        trend_threshold = 5  # 5% trend over lookback period

        if volatility > crisis_vol_threshold:
            return MarketRegime.CRISIS
        elif trend > trend_threshold:
            if volatility > high_vol_threshold:
                return MarketRegime.BULL_TRENDING
            else:
                return MarketRegime.RECOVERY if trend > 10 else MarketRegime.BULL_TRENDING
        elif trend < -trend_threshold:
            return MarketRegime.BEAR_TRENDING
        elif volatility > high_vol_threshold:
            return MarketRegime.SIDEWAYS_HIGH_VOL
        else:
            return MarketRegime.SIDEWAYS_LOW_VOL

    def analyze_event_frequency(self, event_labels: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze frequency and characteristics of different events"""
        event_analysis = {}

        for label in event_labels:
            if label not in self.data.columns:
                continue

            events = self.data[label].dropna()
            if len(events) == 0:
                continue

            positive_events = events[events == 1]
            event_frequency = len(positive_events) / len(events) if len(events) > 0 else 0

            # Calculate event clustering (how often events occur in succession)
            event_series = self.data[label].fillna(0)
            clustering_score = self._calculate_event_clustering(event_series)

            # Market regime association
            regime_association = self._analyze_regime_association(label)

            event_analysis[label] = {
                'frequency': event_frequency,
                'total_events': len(positive_events),
                'clustering_score': clustering_score,
                'regime_association': regime_association,
                'data_quality': 1 - (self.data[label].isna().sum() / len(self.data))
            }

        return event_analysis

    def _calculate_event_clustering(self, event_series: pd.Series) -> float:
        """Calculate how clustered events are (0 = random, 1 = highly clustered)"""
        events = event_series.values
        if len(events) < 10:
            return 0.5

        # Calculate runs test statistic
        runs, n1, n2 = 0, 0, 0
        for i in range(len(events)):
            if events[i] == 1:
                n1 += 1
            else:
                n2 += 1

            if i > 0 and events[i] != events[i - 1]:
                runs += 1

        if n1 == 0 or n2 == 0:
            return 0.5

        # Expected number of runs under random distribution
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1

        # Clustering score (inverse of runs ratio)
        clustering_score = 1 - min(runs / expected_runs, 1) if expected_runs > 0 else 0.5
        return clustering_score

    def _analyze_regime_association(self, label: str) -> Dict[str, float]:
        """Analyze which market regimes are associated with this event"""
        # This would require historical regime classification
        # For now, return uniform distribution
        regimes = list(MarketRegime)
        return {regime.value: 1.0 / len(regimes) for regime in regimes}


class EventTypeClassifier:
    """Classifies events based on their characteristics"""

    def __init__(self):
        self.event_patterns = {
            'breakout': ['break', 'burst', 'penetration', 'resistance', 'support'],
            'mean_reversion': ['revert', 'return', 'bounce', 'correction'],
            'momentum': ['momentum', 'trend', 'acceleration', 'velocity'],
            'volatility': ['vol', 'volatility', 'variance', 'deviation'],
            'volume': ['volume', 'liquidity', 'participation'],
            'pattern': ['pattern', 'formation', 'signal', 'flag', 'triangle']
        }

    def classify_event(self, label_name: str, event_data: pd.Series = None) -> EventType:
        """Classify event type based on name and optionally data characteristics"""
        label_lower = label_name.lower()

        # First try pattern matching on the name
        for event_type, keywords in self.event_patterns.items():
            if any(keyword in label_lower for keyword in keywords):
                return EventType(event_type)

        # If we have event data, try to infer from statistical properties
        if event_data is not None:
            return self._classify_from_data_properties(event_data)

        return EventType.UNKNOWN

    def _classify_from_data_properties(self, event_data: pd.Series) -> EventType:
        """Classify based on statistical properties of the event data"""
        # This is a simplified classification - could be made much more sophisticated
        event_frequency = event_data.sum() / len(event_data)

        if event_frequency < 0.05:  # Very rare events
            return EventType.BREAKOUT
        elif event_frequency > 0.3:  # Common events
            return EventType.MOMENTUM
        else:
            return EventType.MEAN_REVERSION


class AgentKnowledgeBase:
    """Agent's knowledge base for storing and retrieving insights"""

    def __init__(self):
        self.memory = AgentMemory(
            successful_combinations={},
            failed_combinations=set(),
            regime_preferences={},
            model_performance_history=defaultdict(lambda: deque(maxlen=50)),
            feature_importance_trends=defaultdict(lambda: deque(maxlen=20)),
            economic_performance_memory={},
            last_updated=datetime.now()
        )
        self.insights = deque(maxlen=100)  # Store recent insights
        self.feature_synergies = {}  # Features that work well together
        self.anti_synergies = {}  # Features that work poorly together

    def learn_from_result(self, label_type: str, features: List[str],
                          model_name: str, performance: Dict[str, float],
                          market_regime: MarketRegime):
        """Learn from a research result"""
        feature_combo = tuple(sorted(features))

        # Economic performance threshold for success
        success_threshold = {
            'auc_roc': 0.55,
            'total_return': 0.02,
            'sharpe_ratio': 0.5,
            'win_rate': 0.52
        }

        # Determine if this was a successful combination
        is_successful = all(
            performance.get(metric, 0) >= threshold
            for metric, threshold in success_threshold.items()
        )

        if is_successful:
            # Store successful combination
            combo_key = f"{label_type}_{model_name}"
            if combo_key not in self.memory.successful_combinations:
                self.memory.successful_combinations[combo_key] = {}

            self.memory.successful_combinations[combo_key][str(feature_combo)] = \
                performance.get('total_return', 0) * performance.get('auc_roc', 0.5)

            # Update regime preferences
            if market_regime not in self.memory.regime_preferences:
                self.memory.regime_preferences[market_regime] = {}

            for feature in features:
                if feature not in self.memory.regime_preferences[market_regime]:
                    self.memory.regime_preferences[market_regime][feature] = 0.0
                self.memory.regime_preferences[market_regime][feature] += 0.1
        else:
            # Store failed combination
            event_type = self._infer_event_type(label_type)
            self.memory.failed_combinations.add((event_type, str(feature_combo)))

        # Update model performance history
        model_score = performance.get('auc_roc', 0) * performance.get('total_return', 0)
        self.memory.model_performance_history[model_name].append(model_score)

        # Update feature importance trends
        for feature in features:
            importance_score = performance.get('auc_roc', 0.5)
            self.memory.feature_importance_trends[feature].append(importance_score)

        # Learn feature synergies
        self._update_feature_synergies(features, performance.get('auc_roc', 0.5))

        self.memory.last_updated = datetime.now()

    def _infer_event_type(self, label_type: str) -> str:
        """Infer event type from label name"""
        classifier = EventTypeClassifier()
        return classifier.classify_event(label_type).value

    def _update_feature_synergies(self, features: List[str], performance: float):
        """Update knowledge about feature synergies"""
        if performance > 0.6:  # Good performance threshold
            for i, feat1 in enumerate(features):
                for feat2 in features[i + 1:]:
                    pair = tuple(sorted([feat1, feat2]))
                    if pair not in self.feature_synergies:
                        self.feature_synergies[pair] = []
                    self.feature_synergies[pair].append(performance)

    def get_preferred_features_for_regime(self, regime: MarketRegime, top_k: int = 10) -> List[str]:
        """Get preferred features for a specific market regime"""
        if regime not in self.memory.regime_preferences:
            return []

        regime_prefs = self.memory.regime_preferences[regime]
        return sorted(regime_prefs.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_model_recommendations(self, top_k: int = 5) -> List[str]:
        """Get model recommendations based on historical performance"""
        model_scores = {}

        for model, scores in self.memory.model_performance_history.items():
            if len(scores) > 0:
                # Weight recent performance more heavily
                weights = np.exp(np.linspace(0, 1, len(scores)))
                weighted_score = np.average(scores, weights=weights)
                model_scores[model] = weighted_score

        return sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def should_skip_combination(self, event_type: str, features: List[str]) -> bool:
        """Check if a feature combination should be skipped based on memory"""
        feature_combo = tuple(sorted(features))
        return (event_type, str(feature_combo)) in self.memory.failed_combinations

    def add_insight(self, insight: AgentInsight):
        """Add a new insight to the knowledge base"""
        self.insights.append(insight)


class IntelligentFeatureSelector:
    """Intelligent feature selection that adapts based on context"""

    def __init__(self, knowledge_base: AgentKnowledgeBase):
        self.kb = knowledge_base
        self.selection_methods = {
            'mutual_info': self._mutual_info_selection,
            'random_forest': self._rf_importance_selection,
            'correlation_analysis': self._correlation_based_selection,
            'statistical': self._statistical_selection,
            'synergy_based': self._synergy_based_selection
        }

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        event_type: EventType, market_regime: MarketRegime,
                        max_features: int = 15) -> Tuple[List[str], str, float]:
        """Intelligently select features based on context"""

        # Get regime-preferred features
        regime_features = [f[0] for f in self.kb.get_preferred_features_for_regime(market_regime)]

        # Choose selection method based on event type and context
        method = self._choose_selection_method(event_type, len(X.columns), market_regime)

        # Apply the chosen method
        selected_features = self.selection_methods[method](X, y, max_features, regime_features)

        # Calculate confidence in selection
        confidence = self._calculate_selection_confidence(selected_features, X, y)

        return selected_features, method, confidence

    def _choose_selection_method(self, event_type: EventType, n_features: int,
                                 regime: MarketRegime) -> str:
        """Choose the best selection method for the context"""

        # High-dimensional data prefers statistical methods
        if n_features > 100:
            return 'statistical'

        # Event-specific preferences
        if event_type == EventType.BREAKOUT:
            return 'correlation_analysis'  # Breakouts often involve price relationships
        elif event_type == EventType.MOMENTUM:
            return 'mutual_info'  # Non-linear relationships important
        elif event_type == EventType.VOLATILITY:
            return 'random_forest'  # Tree-based methods handle volatility well
        elif event_type in [EventType.MEAN_REVERSION, EventType.PATTERN]:
            return 'synergy_based'  # These often involve feature combinations

        # Crisis regimes prefer robust methods
        if regime == MarketRegime.CRISIS:
            return 'statistical'

        # Default to mutual information
        return 'mutual_info'

    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series,
                               max_features: int, regime_features: List[str]) -> List[str]:
        """Mutual information based selection with regime bias"""
        scores = mutual_info_classif(X, y, random_state=42)
        feature_scores = dict(zip(X.columns, scores))

        # Boost scores for regime-preferred features
        for feature in regime_features:
            if feature in feature_scores:
                feature_scores[feature] *= 1.3  # 30% boost

        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:max_features]]

    def _rf_importance_selection(self, X: pd.DataFrame, y: pd.Series,
                                 max_features: int, regime_features: List[str]) -> List[str]:
        """Random forest importance with regime bias"""
        rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1)
        rf.fit(X, y)

        feature_scores = dict(zip(X.columns, rf.feature_importances_))

        # Boost regime-preferred features
        for feature in regime_features:
            if feature in feature_scores:
                feature_scores[feature] *= 1.2

        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:max_features]]

    def _correlation_based_selection(self, X: pd.DataFrame, y: pd.Series,
                                     max_features: int, regime_features: List[str]) -> List[str]:
        """Correlation-based selection focusing on price relationships"""
        correlations = X.corrwith(y).abs()

        # Prefer OHLC features for breakout events
        ohlc_features = [f for f in X.columns if any(ohlc in f for ohlc in ['Open', 'High', 'Low', 'Close'])]
        for feature in ohlc_features:
            if feature in correlations:
                correlations[feature] *= 1.4

        # Boost regime features
        for feature in regime_features:
            if feature in correlations:
                correlations[feature] *= 1.2

        sorted_features = correlations.sort_values(ascending=False)
        return sorted_features.head(max_features).index.tolist()

    def _statistical_selection(self, X: pd.DataFrame, y: pd.Series,
                               max_features: int, regime_features: List[str]) -> List[str]:
        """F-statistic based selection"""
        f_scores, _ = f_classif(X, y)
        feature_scores = dict(zip(X.columns, f_scores))

        # Boost regime features
        for feature in regime_features:
            if feature in feature_scores:
                feature_scores[feature] *= 1.2

        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:max_features]]

    def _synergy_based_selection(self, X: pd.DataFrame, y: pd.Series,
                                 max_features: int, regime_features: List[str]) -> List[str]:
        """Selection based on known feature synergies"""
        # Start with top individual features
        correlations = X.corrwith(y).abs()
        individual_top = correlations.sort_values(ascending=False).head(max_features * 2).index.tolist()

        # Add features with known synergies
        selected = []
        for feature in individual_top:
            if len(selected) >= max_features:
                break

            selected.append(feature)

            # Look for synergistic partners
            for pair, performances in self.kb.feature_synergies.items():
                if feature in pair and len(performances) > 2:
                    partner = pair[1] if pair[0] == feature else pair[0]
                    if partner in X.columns and partner not in selected:
                        if np.mean(performances) > 0.6:  # Good synergy threshold
                            selected.append(partner)
                            if len(selected) >= max_features:
                                break

        return selected[:max_features]

    def _calculate_selection_confidence(self, features: List[str],
                                        X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate confidence in feature selection"""
        if len(features) == 0:
            return 0.0

        # Base confidence on mutual information scores
        selected_X = X[features]
        mi_scores = mutual_info_classif(selected_X, y, random_state=42)
        avg_mi = np.mean(mi_scores)

        # Boost confidence if using known good features
        regime_boost = 0.0
        for regime_prefs in self.kb.memory.regime_preferences.values():
            for feature in features:
                if feature in regime_prefs:
                    regime_boost += regime_prefs[feature] / len(features)

        # Synergy boost
        synergy_boost = 0.0
        for i, feat1 in enumerate(features):
            for feat2 in features[i + 1:]:
                pair = tuple(sorted([feat1, feat2]))
                if pair in self.kb.feature_synergies:
                    synergy_boost += np.mean(self.kb.feature_synergies[pair])

        synergy_boost = synergy_boost / (len(features) * (len(features) - 1) / 2) if len(features) > 1 else 0

        confidence = min(1.0, avg_mi + regime_boost * 0.1 + synergy_boost * 0.2)
        return confidence


class AdaptiveModelSelector:
    """Intelligently selects and configures models based on context"""

    def __init__(self, knowledge_base: AgentKnowledgeBase):
        self.kb = knowledge_base
        self.model_configs = self._initialize_model_configs()

    def _initialize_model_configs(self) -> Dict[str, Dict]:
        """Initialize model configurations for different contexts"""
        return {
            # Fast models for exploration
            'exploration': {
                'random_forest': {'n_estimators': 30, 'max_depth': 6},
                'logistic_regression': {'max_iter': 300},
                'naive_bayes': {},
                'knn': {'n_neighbors': 5}
            },
            # Robust models for exploitation
            'exploitation': {
                'random_forest': {'n_estimators': 100, 'max_depth': 10},
                'gradient_boosting': {'n_estimators': 100, 'max_depth': 6},
                'extra_trees': {'n_estimators': 80, 'max_depth': 8}
            },
            # Crisis-resistant models
            'crisis': {
                'ridge_classifier': {'alpha': 1.0},
                'logistic_regression': {'C': 0.1, 'penalty': 'l2'},
                'lda': {}
            }
        }

    def select_models(self, event_type: EventType, market_regime: MarketRegime,
                      n_features: int, strategy: ResearchStrategy) -> List[str]:
        """Select appropriate models for the context"""

        # Get recommendations from knowledge base
        kb_recommendations = [model[0] for model in self.kb.get_model_recommendations()]

        # Context-based selection
        if strategy.exploration_vs_exploitation > 0.7:
            context_models = list(self.model_configs['exploration'].keys())
        elif market_regime == MarketRegime.CRISIS:
            context_models = list(self.model_configs['crisis'].keys())
        else:
            context_models = list(self.model_configs['exploitation'].keys())

        # Event-type specific preferences
        event_preferences = self._get_event_model_preferences(event_type)

        # Combine all preferences with weights
        model_scores = defaultdict(float)

        # Knowledge base recommendations (highest weight)
        for i, model in enumerate(kb_recommendations[:5]):
            model_scores[model] += (5 - i) * 3

        # Context models (medium weight)
        for model in context_models:
            model_scores[model] += 2

        # Event preferences (medium weight)
        for model in event_preferences:
            model_scores[model] += 2

        # High-dimensional data preferences
        if n_features > 50:
            model_scores['random_forest'] += 1
            model_scores['extra_trees'] += 1
            model_scores['lda'] += 1  # Dimensionality reduction

        # Sort by score and return top models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [model[0] for model in sorted_models[:5]]

    def _get_event_model_preferences(self, event_type: EventType) -> List[str]:
        """Get model preferences for specific event types"""
        preferences = {
            EventType.BREAKOUT: ['random_forest', 'gradient_boosting', 'svm'],
            EventType.MEAN_REVERSION: ['logistic_regression', 'ridge_classifier', 'lda'],
            EventType.MOMENTUM: ['gradient_boosting', 'xgboost', 'lightgbm'],
            EventType.VOLATILITY: ['extra_trees', 'random_forest', 'gradient_boosting'],
            EventType.VOLUME: ['knn', 'naive_bayes', 'random_forest'],
            EventType.PATTERN: ['svm', 'neural_network', 'gradient_boosting']
        }
        return preferences.get(event_type, ['random_forest', 'logistic_regression'])


class AgenticFeatureResearchAgent:
    """Main agentic research agent that orchestrates the entire process"""

    def __init__(self, labeled_data: Union[pd.DataFrame, str, Path],
                 results_dir: Optional[str] = None,
                 random_state: int = 42,
                 verbose: bool = True):

        # Load and store data
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
        self.knowledge_base = AgentKnowledgeBase()
        self.market_analyzer = MarketContextAnalyzer(self.data)
        self.event_classifier = EventTypeClassifier()
        self.feature_selector = IntelligentFeatureSelector(self.knowledge_base)
        self.model_selector = AdaptiveModelSelector(self.knowledge_base)

        # Set up logging
        self.logger = self._setup_logger()

        # Analyze initial context
        self._analyze_initial_context()

        if self.verbose:
            self.logger.info("Agentic Feature Research Agent initialized")
            self.logger.info(f"Current market regime: {self.current_regime.value}")
            self.logger.info(f"Detected {len(self.label_columns)} event types")

    def _load_data_from_path(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from various file formats"""
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_path.suffix.lower() == '.pkl':
            return pd.read_pickle(data_path)
        elif data_path.suffix.lower() == '.csv':
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
        elif data_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(data_path, index_col=0, parse_dates=True)
        elif data_path.suffix.lower() == '.parquet':
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _setup_logger(self):
        """Set up logging"""
        logger = logging.getLogger(f"agent_{id(self)}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - AGENT - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _analyze_initial_context(self):
        """Analyze initial market context and data characteristics"""
        # Detect features and labels
        self.label_columns = [col for col in self.data.columns if col.endswith('_label')]

        # Exclude non-feature columns
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
                self.feature_columns.append(col)

        # Analyze current market regime
        self.current_regime = self.market_analyzer.detect_market_regime()

        # Analyze event characteristics
        self.event_analysis = self.market_analyzer.analyze_event_frequency(self.label_columns)

        # Classify event types
        self.event_types = {}
        for label in self.label_columns:
            event_data = self.data[label].dropna() if label in self.data.columns else None
            self.event_types[label] = self.event_classifier.classify_event(label, event_data)

        if self.verbose:
            self.logger.info(f"Detected {len(self.feature_columns)} features")
            self.logger.info(f"Event frequency analysis completed for {len(self.event_analysis)} events")

    def _create_research_strategy(self, priority_labels: List[str] = None) -> ResearchStrategy:
        """Create adaptive research strategy based on current context and knowledge"""

        if priority_labels is None:
            # Prioritize based on event frequency and data quality
            event_scores = {}
            for label, analysis in self.event_analysis.items():
                # Score based on frequency (not too rare, not too common) and data quality
                frequency_score = 1 - abs(analysis['frequency'] - 0.15)  # Optimal around 15%
                quality_score = analysis['data_quality']
                total_events_score = min(analysis['total_events'] / 100, 1.0)  # Prefer more events

                event_scores[label] = (frequency_score * 0.4 +
                                       quality_score * 0.3 +
                                       total_events_score * 0.3)

            priority_labels = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)
            priority_labels = [label[0] for label in priority_labels]

        # Determine exploration vs exploitation balance
        # More exploration in crisis, more exploitation in stable markets
        if self.current_regime == MarketRegime.CRISIS:
            exploration_vs_exploitation = 0.8  # High exploration
        elif self.current_regime in [MarketRegime.SIDEWAYS_LOW_VOL, MarketRegime.RECOVERY]:
            exploration_vs_exploitation = 0.3  # High exploitation
        else:
            exploration_vs_exploitation = 0.6  # Balanced

        # Feature selection method preference
        regime_method_prefs = {
            MarketRegime.CRISIS: 'statistical',
            MarketRegime.BULL_TRENDING: 'mutual_info',
            MarketRegime.BEAR_TRENDING: 'correlation_analysis',
            MarketRegime.SIDEWAYS_HIGH_VOL: 'random_forest',
            MarketRegime.SIDEWAYS_LOW_VOL: 'synergy_based',
            MarketRegime.RECOVERY: 'mutual_info'
        }

        preferred_method = regime_method_prefs.get(self.current_regime, 'mutual_info')

        # Model preferences based on successful history
        model_recommendations = self.knowledge_base.get_model_recommendations(5)
        model_prefs = [model[0] for model in model_recommendations] if model_recommendations else [
            'random_forest', 'gradient_boosting', 'logistic_regression', 'extra_trees'
        ]

        # Time allocation based on event priorities
        total_labels = len(priority_labels)
        time_budget = {}
        for i, label in enumerate(priority_labels):
            # More time for higher priority events
            priority_weight = (total_labels - i) / total_labels
            base_time = 1.0 / total_labels
            time_budget[label] = base_time * (1 + priority_weight)

        # Normalize time budget
        total_time = sum(time_budget.values())
        time_budget = {k: v / total_time for k, v in time_budget.items()}

        return ResearchStrategy(
            priority_events=priority_labels,
            feature_selection_method=preferred_method,
            model_preferences=model_prefs,
            time_budget_allocation=time_budget,
            exploration_vs_exploitation=exploration_vs_exploitation,
            economic_focus=True  # Always focus on economic performance
        )

    def _reason_about_event(self, label_type: str) -> Dict[str, Any]:
        """Agent reasoning about a specific event type"""
        reasoning = {
            'event_type': self.event_types.get(label_type, EventType.UNKNOWN),
            'market_context': self.current_regime,
            'event_characteristics': self.event_analysis.get(label_type, {}),
            'recommended_approach': {},
            'confidence': 0.5,
            'insights': []
        }

        event_type = reasoning['event_type']
        event_chars = reasoning['event_characteristics']

        # Reasoning based on event type
        if event_type == EventType.BREAKOUT:
            reasoning['recommended_approach'] = {
                'features': 'price_action_and_volume',
                'models': ['random_forest', 'gradient_boosting'],
                'time_horizon': 'short_term',
                'risk_consideration': 'high_false_positive_risk'
            }
            reasoning['insights'].append("Breakout events require price and volume confirmation")

        elif event_type == EventType.MEAN_REVERSION:
            reasoning['recommended_approach'] = {
                'features': 'statistical_indicators',
                'models': ['logistic_regression', 'lda'],
                'time_horizon': 'medium_term',
                'risk_consideration': 'timing_critical'
            }
            reasoning['insights'].append("Mean reversion works better in sideways markets")

        elif event_type == EventType.MOMENTUM:
            reasoning['recommended_approach'] = {
                'features': 'trend_and_momentum_indicators',
                'models': ['gradient_boosting', 'xgboost'],
                'time_horizon': 'variable',
                'risk_consideration': 'regime_dependent'
            }
            reasoning['insights'].append("Momentum strategies highly regime-dependent")

        # Market regime considerations
        if self.current_regime == MarketRegime.CRISIS:
            reasoning['insights'].append("Crisis regime: Focus on defensive features and robust models")
            reasoning['confidence'] *= 0.8  # Lower confidence in crisis

        elif self.current_regime == MarketRegime.SIDEWAYS_LOW_VOL:
            reasoning['insights'].append("Low volatility: Mean reversion strategies may work well")
            reasoning['confidence'] *= 1.2  # Higher confidence in stable markets

        # Data quality considerations
        if 'data_quality' in event_chars and event_chars['data_quality'] < 0.8:
            reasoning['insights'].append("Low data quality detected - using robust preprocessing")
            reasoning['confidence'] *= 0.9

        # Event frequency considerations
        if 'frequency' in event_chars:
            freq = event_chars['frequency']
            if freq < 0.05:
                reasoning['insights'].append("Rare event - risk of overfitting")
                reasoning['confidence'] *= 0.8
            elif freq > 0.4:
                reasoning['insights'].append("Common event - may lack predictive power")
                reasoning['confidence'] *= 0.9

        return reasoning

    def prepare_data_for_label(self, label_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Intelligently prepare data with context awareness"""
        # Get valid samples
        valid_samples = self.data[label_type].notna()

        if valid_samples.sum() == 0:
            raise ValueError(f"No valid samples found for label {label_type}")

        X = self.data.loc[valid_samples, self.feature_columns].copy()
        y = self.data.loc[valid_samples, label_type].copy()

        # Intelligent missing value handling based on market regime
        if X.isnull().any().any():
            if self.current_regime == MarketRegime.CRISIS:
                # Conservative approach in crisis
                X = X.fillna(method='bfill').fillna(method='ffill')
                X = X.fillna(0)  # Zero fill as last resort
            else:
                # Normal forward/backward fill with median
                X = X.fillna(method='ffill').fillna(method='bfill')
                X = X.fillna(X.median())

        # Remove problematic features
        # 1. Constant features
        feature_vars = X.var()
        constant_features = feature_vars[feature_vars == 0].index.tolist()
        if constant_features:
            X = X.drop(columns=constant_features)
            if self.verbose:
                self.logger.info(f"Removed {len(constant_features)} constant features")

        # 2. Highly correlated features (adaptive threshold based on regime)
        if len(X.columns) > 1:
            corr_threshold = 0.98 if self.current_regime == MarketRegime.CRISIS else 0.95
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            high_corr_features = [column for column in upper_tri.columns
                                  if any(upper_tri[column] > corr_threshold)]
            if high_corr_features:
                # Keep the feature with higher correlation to target
                features_to_drop = []
                target_corrs = X.corrwith(y).abs()

                for feature in high_corr_features:
                    correlated_with = upper_tri[upper_tri[feature] > corr_threshold].index.tolist()
                    if correlated_with:
                        # Among correlated features, keep the one with highest target correlation
                        all_features = [feature] + correlated_with
                        feature_target_corrs = [(f, target_corrs.get(f, 0)) for f in all_features]
                        best_feature = max(feature_target_corrs, key=lambda x: x[1])[0]

                        for f in all_features:
                            if f != best_feature and f not in features_to_drop:
                                features_to_drop.append(f)

                if features_to_drop:
                    X = X.drop(columns=features_to_drop)
                    if self.verbose:
                        self.logger.info(f"Removed {len(features_to_drop)} highly correlated features")

        # 3. Features with too many outliers (market-regime aware)
        if self.current_regime != MarketRegime.CRISIS:  # Don't remove in crisis (outliers may be informative)
            outlier_features = []
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    z_scores = np.abs(stats.zscore(X[col]))
                    outlier_ratio = (z_scores > 3).mean()
                    if outlier_ratio > 0.1:  # More than 10% outliers
                        outlier_features.append(col)

            if outlier_features and len(outlier_features) < len(X.columns) * 0.3:  # Don't remove too many
                X = X.drop(columns=outlier_features)
                if self.verbose:
                    self.logger.info(f"Removed {len(outlier_features)} features with excessive outliers")

        if self.verbose:
            self.logger.info(f"Prepared data for {label_type}: {X.shape[0]} samples, {X.shape[1]} features")
            self.logger.info(f"Label distribution: {y.value_counts().to_dict()}")

        return X, y

    def intelligent_model_evaluation(self, X: pd.DataFrame, y: pd.Series,
                                     models: List[str], label_type: str) -> Dict[str, Dict]:
        """Intelligent model evaluation with economic focus"""

        results = {}
        event_reasoning = self._reason_about_event(label_type)

        # Create time-aware splits (crucial for financial data)
        cv = TimeSeriesSplit(n_splits=3)

        for model_name in models:
            if self.verbose:
                self.logger.info(f"Evaluating {model_name} for {label_type}")

            try:
                # Get model instance
                model = self._get_model_instance(model_name, event_reasoning)

                # Economic-focused evaluation
                economic_results = self._evaluate_model_economically(model, X, y, cv)

                if economic_results:
                    results[model_name] = economic_results

                    # Learn from this result
                    self.knowledge_base.learn_from_result(
                        label_type, X.columns.tolist(), model_name,
                        economic_results, self.current_regime
                    )

            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Model {model_name} evaluation failed: {e}")
                continue

        return results

    def _get_model_instance(self, model_name: str, reasoning: Dict[str, Any]):
        """Get optimally configured model instance"""
        base_params = {'random_state': self.random_state}

        # Model zoo with intelligent configuration
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=100 if reasoning['confidence'] > 0.7 else 50,
                max_depth=10 if self.current_regime != MarketRegime.CRISIS else 6,
                min_samples_split=5,
                n_jobs=1,
                **base_params
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100 if reasoning['confidence'] > 0.7 else 50,
                max_depth=6 if self.current_regime != MarketRegime.CRISIS else 4,
                learning_rate=0.1,
                **base_params
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=80,
                max_depth=8,
                min_samples_split=5,
                n_jobs=1,
                **base_params
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                C=0.1 if self.current_regime == MarketRegime.CRISIS else 1.0,
                **base_params
            ),
            'ridge_classifier': RidgeClassifier(**base_params),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=7),
            'decision_tree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                **base_params
            ),
            'lda': LinearDiscriminantAnalysis(),
            'svm': SVC(probability=True, **base_params)
        }

        # Add advanced models if available
        if HAS_XGBOOST:
            model_configs['xgboost'] = xgb.XGBClassifier(
                n_estimators=80,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss',
                verbosity=0,
                **base_params
            )

        if HAS_LIGHTGBM:
            model_configs['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=80,
                max_depth=6,
                learning_rate=0.1,
                verbosity=-1,
                **base_params
            )

        return model_configs.get(model_name, RandomForestClassifier(**base_params))

    def _evaluate_model_economically(self, model, X: pd.DataFrame, y: pd.Series, cv) -> Dict[str, float]:
        """Evaluate model with focus on economic metrics"""
        try:
            # Statistical metrics
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
            precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=1)
            recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=1)
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=1)

            # Economic evaluation using walk-forward analysis
            economic_metrics = self._calculate_economic_metrics(model, X, y)

            results = {
                'auc_roc': auc_scores.mean(),
                'auc_roc_std': auc_scores.std(),
                'precision': precision_scores.mean(),
                'precision_std': precision_scores.std(),
                'recall': recall_scores.mean(),
                'recall_std': recall_scores.std(),
                'f1_score': f1_scores.mean(),
                'f1_score_std': f1_scores.std(),
                **economic_metrics
            }

            return results

        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Economic evaluation failed: {e}")
            return {}

    def _calculate_economic_metrics(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate economic performance metrics"""

        # Use walk-forward analysis for realistic economic evaluation
        n_samples = len(X)
        train_size = int(n_samples * 0.7)

        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        if len(X_test) < 10:  # Need minimum samples for evaluation
            return {'total_return': 0, 'sharpe_ratio': 0, 'win_rate': 0.5, 'max_drawdown': 0}

        # Train and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        try:
            probabilities = model.predict_proba(X_test)[:, 1]
        except:
            probabilities = predictions.astype(float)

        # Get return data if available
        label_base = y.name.replace('_label', '')
        return_col = f"{label_base}_return"

        economic_metrics = {
            'total_return': 0,
            'sharpe_ratio': 0,
            'win_rate': 0.5,
            'max_drawdown': 0,
            'num_trades': 0,
            'profit_factor': 1.0
        }

        if return_col in self.data.columns:
            # Get returns for test period
            test_returns = self.data.loc[X_test.index, return_col].fillna(0)

            # Strategy returns: only take positions when model predicts positive
            strategy_returns = np.where(predictions == 1, test_returns, 0)
            strategy_returns = pd.Series(strategy_returns, index=test_returns.index)

            # Calculate metrics
            total_return = strategy_returns.sum()

            # Only calculate other metrics if we have trades
            trades = strategy_returns[strategy_returns != 0]
            if len(trades) > 0:
                win_rate = (trades > 0).mean()
                num_trades = len(trades)

                # Sharpe ratio (assuming daily returns)
                if strategy_returns.std() > 0:
                    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0

                # Maximum drawdown
                cumulative_returns = (1 + strategy_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()

                # Profit factor
                winning_trades = trades[trades > 0]
                losing_trades = trades[trades < 0]

                if len(losing_trades) > 0:
                    profit_factor = winning_trades.sum() / abs(losing_trades.sum())
                else:
                    profit_factor = float('inf') if len(winning_trades) > 0 else 1.0

                economic_metrics.update({
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'num_trades': num_trades,
                    'profit_factor': profit_factor
                })

        return economic_metrics

    def research_event_intelligently(self, label_type: str, strategy: ResearchStrategy) -> Dict[str, Any]:
        """Intelligently research a single event type"""

        if self.verbose:
            self.logger.info(f"\n🔍 INTELLIGENT RESEARCH: {label_type}")
            self.logger.info(f"{'=' * 60}")

        # Reason about the event
        reasoning = self._reason_about_event(label_type)

        if self.verbose:
            self.logger.info(f"Event type: {reasoning['event_type'].value}")
            self.logger.info(f"Confidence: {reasoning['confidence']:.3f}")
            for insight in reasoning['insights']:
                self.logger.info(f"💡 Insight: {insight}")

        # Check if we should skip based on memory
        event_type_str = reasoning['event_type'].value
        if len(self.knowledge_base.memory.failed_combinations) > 0:
            # This is a simplified check - in practice would be more sophisticated
            skip_probability = len([fc for fc in self.knowledge_base.memory.failed_combinations
                                    if fc[0] == event_type_str]) / 10.0  # Arbitrary threshold

            if skip_probability > 0.8 and reasoning['confidence'] < 0.6:
                if self.verbose:
                    self.logger.info(f"⚠️  Skipping {label_type} due to high historical failure rate")
                return None

        try:
            # Prepare data
            X, y = self.prepare_data_for_label(label_type)

            # Intelligent feature selection
            selected_features, selection_method, selection_confidence = self.feature_selector.select_features(
                X, y, reasoning['event_type'], self.current_regime, max_features=15
            )

            if len(selected_features) == 0:
                if self.verbose:
                    self.logger.warning(f"No features selected for {label_type}")
                return None

            if self.verbose:
                self.logger.info(f"🎯 Selected {len(selected_features)} features using {selection_method}")
                self.logger.info(f"Selection confidence: {selection_confidence:.3f}")
                self.logger.info(
                    f"Features: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")

            # Intelligent model selection
            recommended_models = self.model_selector.select_models(
                reasoning['event_type'], self.current_regime, len(selected_features), strategy
            )

            if self.verbose:
                self.logger.info(f"🤖 Recommended models: {', '.join(recommended_models)}")

            # Evaluate models
            X_selected = X[selected_features]
            model_results = self.intelligent_model_evaluation(X_selected, y, recommended_models, label_type)

            if not model_results:
                if self.verbose:
                    self.logger.warning(f"No successful model evaluations for {label_type}")
                return None

            # Find best model based on economic score
            best_model = self._select_best_model(model_results)

            # Generate insights
            insights = self._generate_insights(label_type, selected_features, model_results, reasoning)

            # Store insights
            for insight in insights:
                self.knowledge_base.add_insight(insight)

            research_result = {
                'label_type': label_type,
                'event_type': reasoning['event_type'].value,
                'market_regime': self.current_regime.value,
                'reasoning': reasoning,
                'selected_features': selected_features,
                'selection_method': selection_method,
                'selection_confidence': selection_confidence,
                'model_results': model_results,
                'best_model': best_model,
                'best_performance': model_results.get(best_model, {}),
                'insights': [asdict(insight) for insight in insights],
                'research_timestamp': datetime.now().isoformat()
            }

            if self.verbose:
                best_perf = model_results.get(best_model, {})
                self.logger.info(f"🏆 Best model: {best_model}")
                self.logger.info(f"📊 Performance - AUC: {best_perf.get('auc_roc', 0):.3f}, "
                                 f"Return: {best_perf.get('total_return', 0):.4f}, "
                                 f"Sharpe: {best_perf.get('sharpe_ratio', 0):.3f}")

            return research_result

        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Research failed for {label_type}: {e}")
            return None

    def _select_best_model(self, model_results: Dict[str, Dict]) -> str:
        """Select best model based on economic performance"""
        model_scores = {}

        for model_name, results in model_results.items():
            # Economic score combining multiple metrics
            auc = results.get('auc_roc', 0.5)
            total_return = results.get('total_return', 0)
            sharpe = results.get('sharpe_ratio', 0)
            win_rate = results.get('win_rate', 0.5)
            num_trades = results.get('num_trades', 0)

            # Penalize models with too few trades
            trade_penalty = 1.0 if num_trades >= 10 else num_trades / 10.0

            # Combined economic score
            economic_score = (
                                     auc * 0.3 +  # Statistical performance
                                     (total_return + 1) * 0.4 +  # Total return (shifted to be positive)
                                     (sharpe + 1) * 0.2 +  # Sharpe ratio (shifted)
                                     win_rate * 0.1  # Win rate
                             ) * trade_penalty

            model_scores[model_name] = economic_score

        return max(model_scores.items(), key=lambda x: x[1])[0] if model_scores else 'random_forest'

    def _generate_insights(self, label_type: str, features: List[str],
                           model_results: Dict[str, Dict], reasoning: Dict) -> List[AgentInsight]:
        """Generate actionable insights from research results"""
        insights = []

        # Feature insight
        if reasoning['confidence'] > 0.7:
            insights.append(AgentInsight(
                insight_type="feature_selection",
                description=f"High confidence feature selection for {label_type}. "
                            f"Key features: {', '.join(features[:3])}",
                confidence=reasoning['confidence'],
                supporting_evidence={'selected_features': features, 'method': 'intelligent_selection'},
                timestamp=datetime.now(),
                actionable=True
            ))

        # Model performance insight
        best_model = self._select_best_model(model_results)
        best_performance = model_results.get(best_model, {})

        if best_performance.get('total_return', 0) > 0.05:  # 5% return threshold
            insights.append(AgentInsight(
                insight_type="profitable_strategy",
                description=f"Profitable strategy identified for {label_type} using {best_model}. "
                            f"Expected return: {best_performance.get('total_return', 0):.3f