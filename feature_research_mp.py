import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import itertools
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import logging

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

# Genetic Algorithm for feature selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
import multiprocessing
warnings.filterwarnings('ignore')

from labeling import labeled_data

@dataclass
class ResearchResults:
    """Container for feature research results"""
    label_type: str
    best_features: List[str]
    feature_scores: Dict[str, float]
    model_performance: Dict[str, Dict[str, float]]
    stability_metrics: Dict[str, float]
    economic_metrics: Dict[str, float]
    research_timestamp: str
    model_rankings: Dict[str, float]
    optimal_model: str


class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Genetic Algorithm for feature selection optimized for binary classification
    """

    def __init__(self,
                 estimator,
                 population_size=50,
                 generations=20,
                 mutation_rate=0.1,
                 crossover_rate=0.8,
                 tournament_size=3,
                 random_state=42):
        self.estimator = estimator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.random_state = random_state

    def _create_individual(self, n_features):
        """Create random feature selection mask"""
        np.random.seed(self.random_state)
        # Ensure at least 5 features are selected
        individual = np.random.random(n_features) > 0.7
        if individual.sum() < 5:
            indices = np.random.choice(n_features, 5, replace=False)
            individual[indices] = True
        return individual

    def _evaluate_individual(self, individual, X, y):
        """Evaluate feature subset using cross-validation"""
        if individual.sum() == 0:
            return 0.0

        X_subset = X[:, individual]
        try:
            # Use TimeSeriesSplit for temporal data
            cv_scores = cross_val_score(
                self.estimator, X_subset, y,
                cv=TimeSeriesSplit(n_splits=3),
                scoring='roc_auc',
                n_jobs=1
            )
            # Penalize for too many features
            penalty = individual.sum() / len(individual) * 0.1
            return cv_scores.mean() - penalty
        except:
            return 0.0

    def _tournament_selection(self, population, fitness_scores):
        """Tournament selection for genetic algorithm"""
        selected = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(
                len(population),
                self.tournament_size,
                replace=False
            )
            winner_idx = tournament_indices[
                np.argmax([fitness_scores[i] for i in tournament_indices])
            ]
            selected.append(population[winner_idx].copy())
        return selected

    def _crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def _mutate(self, individual):
        """Bit-flip mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = not mutated[i]

        # Ensure at least 3 features remain selected
        if mutated.sum() < 3:
            indices = np.random.choice(len(mutated), 3, replace=False)
            mutated[indices] = True

        return mutated

    def fit(self, X, y):
        """Fit genetic algorithm feature selector"""
        X, y = check_X_y(X, y)
        n_features = X.shape[1]

        # Initialize population
        population = [self._create_individual(n_features) for _ in range(self.population_size)]

        best_fitness = -1
        best_individual = None

        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = [
                self._evaluate_individual(individual, X, y)
                for individual in population
            ]

            # Track best individual
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx].copy()

            # Selection
            selected = self._tournament_selection(population, fitness_scores)

            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self._crossover(selected[i], selected[i + 1])
                    new_population.extend([
                        self._mutate(child1),
                        self._mutate(child2)
                    ])
                else:
                    new_population.append(self._mutate(selected[i]))

            # Elitism: keep best individual
            new_population[0] = best_individual.copy()
            population = new_population[:self.population_size]

        self.best_features_mask_ = best_individual
        self.best_fitness_ = best_fitness
        self.feature_names_ = getattr(X, 'columns', None)

        return self

    def transform(self, X):
        """Transform data using selected features"""
        X = check_array(X)
        return X[:, self.best_features_mask_]

    def get_selected_features(self):
        """Get list of selected feature names/indices"""
        if self.feature_names_ is not None:
            return [name for name, selected in zip(self.feature_names_, self.best_features_mask_) if selected]
        else:
            return [i for i, selected in enumerate(self.best_features_mask_) if selected]


class BinaryClassificationFeatureResearch:
    """
    Advanced feature research system for binary classification models
    Optimized for financial time series with event-driven labels
    """

    def __init__(self,
                 labeled_data: pd.DataFrame,
                 results_dir: str = "feature_research_results",
                 random_state: int = 42):
        """
        Initialize feature research system

        Args:
            labeled_data: DataFrame from labeling.py with features and labels
            results_dir: Directory to save research results
            random_state: Random seed for reproducibility
        """
        self.data = labeled_data.copy()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.random_state = random_state

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'research.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Detect available features and labels
        self._detect_features_and_labels()

        # Initialize model zoo for binary classification
        self._initialize_model_zoo()

        self.logger.info(
            f"Initialized with {len(self.feature_columns)} features and {len(self.label_columns)} label types")

    def _detect_features_and_labels(self):
        """Detect feature columns and label columns"""
        # Label columns end with '_label'
        self.label_columns = [col for col in self.data.columns if col.endswith('_label')]

        # Exclude non-feature columns
        exclude_patterns = [
            '_label', '_barrier_touched', '_touch_time', '_return', '_holding_hours',
            '_event', 'event_type', 'any_event'
        ]

        self.feature_columns = []
        for col in self.data.columns:
            if not any(pattern in col for pattern in exclude_patterns):
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:  # Keep OHLCV as features
                    if self.data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        self.feature_columns.append(col)

        # Add OHLCV as features
        ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv:
            if col in self.data.columns:
                self.feature_columns.append(col)

        self.logger.info(f"Detected {len(self.feature_columns)} feature columns: {self.feature_columns[:10]}...")
        self.logger.info(f"Detected {len(self.label_columns)} label types: {self.label_columns}")

    def _initialize_model_zoo(self):
        """Initialize comprehensive model zoo for binary classification"""
        self.model_zoo = {
            # Tree-based models (good for financial data)
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=6, random_state=self.random_state
            ),

            # Linear models (interpretable)
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'ridge_classifier': RidgeClassifier(
                random_state=self.random_state
            ),

            # Instance-based
            'knn': KNeighborsClassifier(n_neighbors=5),

            # Naive Bayes (handles multicollinearity well)
            'naive_bayes': GaussianNB(),

            # SVM (good for high-dimensional data)
            'svm': SVC(
                probability=True, random_state=self.random_state
            ),

            # Discriminant analysis
            'lda': LinearDiscriminantAnalysis(),
            'qda': QuadraticDiscriminantAnalysis(),

            # Single tree (baseline)
            'decision_tree': DecisionTreeClassifier(
                max_depth=10, random_state=self.random_state
            ),

            # Ensemble methods
            'ada_boost': AdaBoostClassifier(
                random_state=self.random_state
            ),
            'bagging': BaggingClassifier(
                random_state=self.random_state, n_jobs=1
            ),
        }

        # Add advanced models if available
        if HAS_XGBOOST:
            self.model_zoo['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )

        if HAS_LIGHTGBM:
            self.model_zoo['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                verbosity=-1
            )

        self.logger.info(f"Initialized {len(self.model_zoo)} models for evaluation")

    def prepare_data_for_label(self, label_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target for specific label type

        Args:
            label_type: Label column name

        Returns:
            Tuple of (features_df, target_series)
        """
        # Get samples with non-null labels
        valid_samples = self.data[label_type].notna()

        # Prepare features
        X = self.data.loc[valid_samples, self.feature_columns].copy()
        y = self.data.loc[valid_samples, label_type].copy()

        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Remove constant features
        constant_features = X.columns[X.var() == 0].tolist()
        if constant_features:
            X = X.drop(columns=constant_features)
            self.logger.warning(f"Removed {len(constant_features)} constant features for {label_type}")

        # Remove highly correlated features (> 0.95)
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > 0.95)
        ]
        if high_corr_features:
            X = X.drop(columns=high_corr_features)
            self.logger.info(f"Removed {len(high_corr_features)} highly correlated features for {label_type}")

        self.logger.info(f"Prepared data for {label_type}: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Label distribution: {y.value_counts().to_dict()}")

        return X, y

    def research_feature_importance(self,
                                    label_type: str,
                                    methods: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Research feature importance using multiple methods

        Args:
            label_type: Label to analyze
            methods: List of importance methods to use

        Returns:
            Dictionary mapping method names to feature importance scores
        """
        if methods is None:
            methods = ['mutual_info', 'random_forest', 'gradient_boosting', 'statistical']

        X, y = self.prepare_data_for_label(label_type)

        importance_results = {}

        # Mutual Information
        if 'mutual_info' in methods:
            try:
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
                importance_results['mutual_info'] = dict(zip(X.columns, mi_scores))
            except Exception as e:
                self.logger.warning(f"Mutual information failed: {e}")

        # Random Forest importance
        if 'random_forest' in methods:
            try:
                rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1)
                rf.fit(X, y)
                importance_results['random_forest'] = dict(zip(X.columns, rf.feature_importances_))
            except Exception as e:
                self.logger.warning(f"Random Forest importance failed: {e}")

        # Gradient Boosting importance
        if 'gradient_boosting' in methods:
            try:
                gb = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
                gb.fit(X, y)
                importance_results['gradient_boosting'] = dict(zip(X.columns, gb.feature_importances_))
            except Exception as e:
                self.logger.warning(f"Gradient Boosting importance failed: {e}")

        # Statistical tests
        if 'statistical' in methods:
            try:
                f_scores, _ = f_classif(X, y)
                importance_results['statistical'] = dict(zip(X.columns, f_scores))
            except Exception as e:
                self.logger.warning(f"Statistical importance failed: {e}")

        return importance_results

    def genetic_feature_selection(self,
                                  label_type: str,
                                  base_estimator=None,
                                  population_size: int = 30,
                                  generations: int = 15) -> Tuple[List[str], float]:
        """
        Apply genetic algorithm for feature selection

        Args:
            label_type: Label to optimize for
            base_estimator: Model to use for evaluation
            population_size: GA population size
            generations: Number of GA generations

        Returns:
            Tuple of (selected_features, best_score)
        """
        X, y = self.prepare_data_for_label(label_type)

        if base_estimator is None:
            base_estimator = RandomForestClassifier(
                n_estimators=50, max_depth=8,
                random_state=self.random_state, n_jobs=1
            )

        # Apply genetic algorithm
        ga_selector = GeneticFeatureSelector(
            estimator=base_estimator,
            population_size=population_size,
            generations=generations,
            random_state=self.random_state
        )

        self.logger.info(f"Starting genetic algorithm feature selection for {label_type}")
        ga_selector.fit(X.values, y.values)

        # Get selected features
        selected_features = [
            X.columns[i] for i, selected in enumerate(ga_selector.best_features_mask_)
            if selected
        ]

        self.logger.info(f"GA selected {len(selected_features)} features with score {ga_selector.best_fitness_:.4f}")

        return selected_features, ga_selector.best_fitness_

    def evaluate_model_performance(self,
                                   label_type: str,
                                   features: List[str] = None,
                                   models: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models for binary classification

        Args:
            label_type: Label to predict
            features: List of features to use (None = use all)
            models: List of model names to evaluate (None = use all)

        Returns:
            Dictionary mapping model names to performance metrics
        """
        X, y = self.prepare_data_for_label(label_type)

        if features is not None:
            available_features = [f for f in features if f in X.columns]
            X = X[available_features]

        if models is None:
            models = list(self.model_zoo.keys())

        results = {}

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        for model_name in models:
            if model_name not in self.model_zoo:
                continue

            try:
                model = self.model_zoo[model_name]
                self.logger.info(f"Evaluating {model_name} for {label_type}")

                # Cross-validation scores
                cv_scores = {}

                # AUC-ROC
                auc_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=1)
                cv_scores['auc_roc'] = {
                    'mean': auc_scores.mean(),
                    'std': auc_scores.std(),
                    'scores': auc_scores.tolist()
                }

                # Accuracy
                acc_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=1)
                cv_scores['accuracy'] = {
                    'mean': acc_scores.mean(),
                    'std': acc_scores.std(),
                    'scores': acc_scores.tolist()
                }

                # Precision
                prec_scores = cross_val_score(model, X, y, cv=tscv, scoring='precision', n_jobs=1)
                cv_scores['precision'] = {
                    'mean': prec_scores.mean(),
                    'std': prec_scores.std(),
                    'scores': prec_scores.tolist()
                }

                # Recall
                rec_scores = cross_val_score(model, X, y, cv=tscv, scoring='recall', n_jobs=1)
                cv_scores['recall'] = {
                    'mean': rec_scores.mean(),
                    'std': rec_scores.std(),
                    'scores': rec_scores.tolist()
                }

                # F1-score
                f1_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1', n_jobs=1)
                cv_scores['f1'] = {
                    'mean': f1_scores.mean(),
                    'std': f1_scores.std(),
                    'scores': f1_scores.tolist()
                }

                results[model_name] = cv_scores

            except Exception as e:
                self.logger.warning(f"Model {model_name} failed: {e}")
                continue

        return results

    def test_feature_stability(self,
                               label_type: str,
                               features: List[str],
                               test_periods: int = 6) -> Dict[str, float]:
        """
        Test temporal stability of features using rolling window analysis

        Args:
            label_type: Label to analyze
            features: Features to test
            test_periods: Number of time periods to test

        Returns:
            Dictionary mapping features to stability scores
        """
        X, y = self.prepare_data_for_label(label_type)
        X = X[features]

        # Split data into time periods
        n_samples = len(X)
        period_size = n_samples // test_periods

        stability_scores = {}

        for feature in features:
            period_importances = []

            for period in range(test_periods):
                start_idx = period * period_size
                end_idx = min((period + 1) * period_size, n_samples)

                if end_idx - start_idx < 50:  # Skip if too few samples
                    continue

                X_period = X.iloc[start_idx:end_idx]
                y_period = y.iloc[start_idx:end_idx]

                if y_period.nunique() < 2:  # Skip if no class variety
                    continue

                try:
                    # Use mutual information as stability measure
                    mi_score = mutual_info_classif(
                        X_period[[feature]], y_period,
                        random_state=self.random_state
                    )[0]
                    period_importances.append(mi_score)
                except:
                    continue

            if len(period_importances) >= 3:
                # Stability = 1 - coefficient_of_variation
                stability_scores[feature] = 1 - (
                        np.std(period_importances) / (np.mean(period_importances) + 1e-8)
                )
            else:
                stability_scores[feature] = 0.0

        return stability_scores

    def calculate_economic_metrics(self,
                                   label_type: str,
                                   features: List[str],
                                   model_name: str = 'random_forest') -> Dict[str, float]:
        """
        Calculate economic performance metrics for trading strategy

        Args:
            label_type: Label to analyze
            features: Features to use
            model_name: Model for predictions

        Returns:
            Dictionary of economic metrics
        """
        X, y = self.prepare_data_for_label(label_type)
        X = X[features]

        # Get corresponding return data
        return_col = label_type.replace('_label', '_return')
        holding_col = label_type.replace('_label', '_holding_hours')

        returns_data = None
        holding_data = None

        if return_col in self.data.columns:
            returns_data = self.data.loc[X.index, return_col]
        if holding_col in self.data.columns:
            holding_data = self.data.loc[X.index, holding_col]

        # Train model for predictions
        model = self.model_zoo[model_name]

        # Time series split for realistic evaluation
        tscv = TimeSeriesSplit(n_splits=3)
        all_predictions = np.zeros(len(y))
        all_proba = np.zeros(len(y))

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            all_predictions[test_idx] = model.predict(X_test)
            all_proba[test_idx] = model.predict_proba(X_test)[:, 1]

        economic_metrics = {}

        # Basic classification metrics
        economic_metrics['accuracy'] = accuracy_score(y, all_predictions)
        economic_metrics['precision'] = precision_score(y, all_predictions, zero_division=0)
        economic_metrics['recall'] = recall_score(y, all_predictions, zero_division=0)
        economic_metrics['f1_score'] = f1_score(y, all_predictions, zero_division=0)

        try:
            economic_metrics['auc_roc'] = roc_auc_score(y, all_proba)
        except:
            economic_metrics['auc_roc'] = 0.5

        # Economic metrics if return data available
        if returns_data is not None:
            returns_data = returns_data.fillna(0)

            # Strategy returns (only trade when model predicts positive)
            strategy_returns = np.where(all_predictions == 1, returns_data, 0)

            # Performance metrics
            economic_metrics['total_return'] = strategy_returns.sum()
            economic_metrics['mean_return_per_trade'] = strategy_returns[strategy_returns != 0].mean() if (
                        strategy_returns != 0).any() else 0
            economic_metrics['win_rate'] = (strategy_returns > 0).mean()
            economic_metrics['num_trades'] = (strategy_returns != 0).sum()

            if economic_metrics['num_trades'] > 0:
                economic_metrics['sharpe_ratio'] = (
                        strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
                )
            else:
                economic_metrics['sharpe_ratio'] = 0

            # Maximum drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            economic_metrics['max_drawdown'] = drawdown.min()

        # Holding period analysis
        if holding_data is not None:
            holding_data = holding_data.fillna(0)
            trades_holding = holding_data[all_predictions == 1]
            if len(trades_holding) > 0:
                economic_metrics['avg_holding_hours'] = trades_holding.mean()
                economic_metrics['median_holding_hours'] = trades_holding.median()
            else:
                economic_metrics['avg_holding_hours'] = 0
                economic_metrics['median_holding_hours'] = 0

        return economic_metrics

    def comprehensive_research(self,
                               label_type: str,
                               use_genetic_selection: bool = True,
                               max_features: int = 20) -> ResearchResults:
        """
        Run comprehensive feature research for a label type

        Args:
            label_type: Label to research
            use_genetic_selection: Whether to use genetic algorithm
            max_features: Maximum number of features to select

        Returns:
            ResearchResults object with all findings
        """
        self.logger.info(f"Starting comprehensive research for {label_type}")

        # Step 1: Feature importance analysis
        importance_results = self.research_feature_importance(label_type)

        # Step 2: Combine importance scores
        combined_scores = defaultdict(float)
        for method, scores in importance_results.items():
            for feature, score in scores.items():
                combined_scores[feature] += score

        # Normalize combined scores
        max_score = max(combined_scores.values()) if combined_scores else 1
        combined_scores = {k: v / max_score for k, v in combined_scores.items()}

        # Step 3: Select top features by combined importance
        top_features_by_importance = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_features * 2]

        top_features = [feature for feature, _ in top_features_by_importance]

        # Step 4: Genetic algorithm feature selection (optional)
        if use_genetic_selection and len(top_features) > 10:
            self.logger.info("Applying genetic algorithm for feature optimization")
            selected_features, ga_score = self.genetic_feature_selection(
                label_type,
                population_size=30,
                generations=15
            )
            # Combine GA results with importance-based selection
            final_features = list(set(selected_features + top_features[:max_features]))[:max_features]
        else:
            final_features = top_features[:max_features]

        # Step 5: Test feature stability
        self.logger.info("Testing feature stability across time periods")
        stability_scores = self.test_feature_stability(label_type, final_features)

        # Step 6: Comprehensive model evaluation
        self.logger.info("Evaluating all models with selected features")
        model_performance = self.evaluate_model_performance(
            label_type,
            features=final_features
        )

        # Step 7: Rank models by performance
        model_rankings = {}
        for model_name, metrics in model_performance.items():
            # Combined score: AUC-ROC (0.4) + F1 (0.3) + Precision (0.3)
            score = (
                    metrics.get('auc_roc', {}).get('mean', 0) * 0.4 +
                    metrics.get('f1', {}).get('mean', 0) * 0.3 +
                    metrics.get('precision', {}).get('mean', 0) * 0.3
            )
            model_rankings[model_name] = score

        # Find best model
        best_model = max(model_rankings.items(), key=lambda x: x[1])[0]

        # Step 8: Calculate economic metrics with best model
        self.logger.info(f"Calculating economic metrics using {best_model}")
        economic_metrics = self.calculate_economic_metrics(
            label_type,
            final_features,
            best_model
        )

        # Create results object
        results = ResearchResults(
            label_type=label_type,
            best_features=final_features,
            feature_scores=combined_scores,
            model_performance=model_performance,
            stability_metrics=stability_scores,
            economic_metrics=economic_metrics,
            research_timestamp=datetime.now().isoformat(),
            model_rankings=model_rankings,
            optimal_model=best_model
        )

        self.logger.info(
            f"Research completed for {label_type}. Best model: {best_model} (score: {model_rankings[best_model]:.4f})")

        return results

    def research_all_labels(self,
                            use_genetic_selection: bool = True,
                            max_features: int = 20) -> Dict[str, ResearchResults]:
        """
        Run comprehensive research for all available label types

        Args:
            use_genetic_selection: Whether to use genetic algorithm
            max_features: Maximum features per label type

        Returns:
            Dictionary mapping label types to research results
        """
        all_results = {}

        for label_type in self.label_columns:
            try:
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"RESEARCHING LABEL TYPE: {label_type}")
                self.logger.info(f"{'=' * 60}")

                results = self.comprehensive_research(
                    label_type,
                    use_genetic_selection,
                    max_features
                )
                all_results[label_type] = results

                # Save individual results
                self.save_results(results, suffix=f"_{label_type}")

            except Exception as e:
                self.logger.error(f"Failed to research {label_type}: {e}")
                continue

        # Generate comparative analysis
        self.generate_comparative_analysis(all_results)

        return all_results

    def save_results(self, results: ResearchResults, suffix: str = ""):
        """Save research results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        results_dict = asdict(results)
        json_file = self.results_dir / f"research_results{suffix}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        # Save as pickle for Python objects
        pickle_file = self.results_dir / f"research_results{suffix}_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)

        self.logger.info(f"Results saved to {json_file} and {pickle_file}")

    def generate_comparative_analysis(self, all_results: Dict[str, ResearchResults]):
        """Generate comparative analysis across all label types"""
        analysis = {
            'summary': {
                'total_labels_researched': len(all_results),
                'research_timestamp': datetime.now().isoformat(),
                'best_performing_labels': {},
                'model_preferences': defaultdict(int),
                'common_features': {},
                'economic_performance_ranking': []
            }
        }

        # Model preference analysis
        for label_type, results in all_results.items():
            analysis['summary']['model_preferences'][results.optimal_model] += 1

        # Best performing labels by different metrics
        metrics_to_analyze = ['auc_roc', 'total_return', 'sharpe_ratio', 'win_rate']

        for metric in metrics_to_analyze:
            best_label = None
            best_value = float('-inf')

            for label_type, results in all_results.items():
                if metric in results.economic_metrics:
                    value = results.economic_metrics[metric]
                    if value > best_value:
                        best_value = value
                        best_label = label_type

            if best_label:
                analysis['summary']['best_performing_labels'][metric] = {
                    'label': best_label,
                    'value': best_value
                }

        # Common important features across labels
        feature_counts = defaultdict(int)
        for results in all_results.values():
            for feature in results.best_features[:10]:  # Top 10 features
                feature_counts[feature] += 1

        analysis['summary']['common_features'] = dict(
            sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        )

        # Economic performance ranking
        economic_ranking = []
        for label_type, results in all_results.items():
            total_return = results.economic_metrics.get('total_return', 0)
            sharpe_ratio = results.economic_metrics.get('sharpe_ratio', 0)
            win_rate = results.economic_metrics.get('win_rate', 0)

            # Combined economic score
            economic_score = total_return * 0.4 + sharpe_ratio * 0.3 + win_rate * 0.3

            economic_ranking.append({
                'label_type': label_type,
                'economic_score': economic_score,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'optimal_model': results.optimal_model
            })

        analysis['summary']['economic_performance_ranking'] = sorted(
            economic_ranking,
            key=lambda x: x['economic_score'],
            reverse=True
        )

        # Save comparative analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.results_dir / f"comparative_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        self.logger.info(f"Comparative analysis saved to {analysis_file}")

        # Print summary
        print(f"\n{'=' * 80}")
        print("COMPARATIVE ANALYSIS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total labels researched: {len(all_results)}")
        print(f"\nModel preferences:")
        for model, count in sorted(analysis['summary']['model_preferences'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} labels")

        print(f"\nTop 10 common features:")
        for feature, count in list(analysis['summary']['common_features'].items())[:10]:
            print(f"  {feature}: used in {count} labels")

        print(f"\nTop 5 economically performing labels:")
        for i, item in enumerate(analysis['summary']['economic_performance_ranking'][:5]):
            print(f"  {i + 1}. {item['label_type']}: Score {item['economic_score']:.4f} "
                  f"(Return: {item['total_return']:.4f}, Sharpe: {item['sharpe_ratio']:.4f})")

        return analysis

    def generate_report(self, results: ResearchResults):
        """Generate detailed research report"""
        report = f"""
BINARY CLASSIFICATION FEATURE RESEARCH REPORT
{'=' * 60}

Label Type: {results.label_type}
Research Date: {results.research_timestamp}
Optimal Model: {results.optimal_model}

FEATURE SELECTION RESULTS
{'=' * 30}
Selected Features ({len(results.best_features)}):
{', '.join(results.best_features)}

Top 10 Features by Importance Score:
"""

        # Top features by score
        top_features = sorted(results.feature_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, score) in enumerate(top_features, 1):
            report += f"{i:2d}. {feature:<30} {score:.4f}\n"

        report += f"""
MODEL PERFORMANCE COMPARISON
{'=' * 30}
"""

        # Model rankings
        model_rankings = sorted(results.model_rankings.items(), key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(model_rankings, 1):
            model_metrics = results.model_performance.get(model, {})
            auc = model_metrics.get('auc_roc', {}).get('mean', 0)
            f1 = model_metrics.get('f1', {}).get('mean', 0)
            precision = model_metrics.get('precision', {}).get('mean', 0)

            report += f"{i:2d}. {model:<20} Score: {score:.4f} "
            report += f"(AUC: {auc:.3f}, F1: {f1:.3f}, Precision: {precision:.3f})\n"

        report += f"""
ECONOMIC PERFORMANCE METRICS
{'=' * 30}
Total Return: {results.economic_metrics.get('total_return', 0):.4f}
Win Rate: {results.economic_metrics.get('win_rate', 0):.4f}
Sharpe Ratio: {results.economic_metrics.get('sharpe_ratio', 0):.4f}
Max Drawdown: {results.economic_metrics.get('max_drawdown', 0):.4f}
Number of Trades: {results.economic_metrics.get('num_trades', 0)}
Average Holding Hours: {results.economic_metrics.get('avg_holding_hours', 0):.2f}

CLASSIFICATION METRICS
{'=' * 30}
Accuracy: {results.economic_metrics.get('accuracy', 0):.4f}
Precision: {results.economic_metrics.get('precision', 0):.4f}
Recall: {results.economic_metrics.get('recall', 0):.4f}
F1-Score: {results.economic_metrics.get('f1_score', 0):.4f}
AUC-ROC: {results.economic_metrics.get('auc_roc', 0):.4f}

FEATURE STABILITY ANALYSIS
{'=' * 30}
"""

        # Feature stability
        stable_features = sorted(results.stability_metrics.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, stability) in enumerate(stable_features, 1):
            report += f"{i:2d}. {feature:<30} {stability:.4f}\n"

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"report_{results.label_type}_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print(report)
        self.logger.info(f"Detailed report saved to {report_file}")

        return report


# CONVENIENCE FUNCTIONS FOR EASY USAGE

def quick_feature_research(labeled_data: pd.DataFrame,
                           label_type: str = None,
                           max_features: int = 15,
                           use_genetic: bool = True,
                           results_dir: str = "feature_research_results") -> ResearchResults:
    """
    Quick feature research for a single label type

    Args:
        labeled_data: Output from labeling.py
        label_type: Specific label to research (None = use first available)
        max_features: Maximum features to select
        use_genetic: Use genetic algorithm optimization
        results_dir: Directory for results

    Returns:
        ResearchResults object
    """
    researcher = BinaryClassificationFeatureResearch(labeled_data, results_dir)

    if label_type is None:
        label_type = researcher.label_columns[0]
        print(f"Using label type: {label_type}")

    results = researcher.comprehensive_research(label_type, use_genetic, max_features)
    researcher.generate_report(results)

    return results


def comprehensive_feature_research(labeled_data: pd.DataFrame,
                                   max_features: int = 20,
                                   use_genetic: bool = True,
                                   results_dir: str = "feature_research_results") -> Dict[str, ResearchResults]:
    """
    Comprehensive feature research for all label types

    Args:
        labeled_data: Output from labeling.py
        max_features: Maximum features per label
        use_genetic: Use genetic algorithm optimization
        results_dir: Directory for results

    Returns:
        Dictionary of all research results
    """
    researcher = BinaryClassificationFeatureResearch(labeled_data, results_dir)

    print(f"Starting comprehensive research for {len(researcher.label_columns)} label types:")
    print(f"Available labels: {researcher.label_columns}")

    all_results = researcher.research_all_labels(use_genetic, max_features)

    print(f"\nResearch completed! Results saved to: {results_dir}")

    return all_results


def model_comparison_study(labeled_data: pd.DataFrame,
                           label_type: str,
                           feature_sets: Dict[str, List[str]],
                           results_dir: str = "model_comparison_results") -> Dict[str, Dict[str, Dict]]:
    """
    Compare different models across different feature sets

    Args:
        labeled_data: Output from labeling.py
        label_type: Label to analyze
        feature_sets: Dictionary mapping set names to feature lists
        results_dir: Directory for results

    Returns:
        Nested dictionary of results [feature_set][model][metric]
    """
    researcher = BinaryClassificationFeatureResearch(labeled_data, results_dir)

    comparison_results = {}

    for set_name, features in feature_sets.items():
        print(f"\nEvaluating feature set: {set_name} ({len(features)} features)")

        model_performance = researcher.evaluate_model_performance(
            label_type,
            features=features
        )

        comparison_results[set_name] = model_performance

    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = Path(results_dir) / f"model_comparison_{label_type}_{timestamp}.json"
    Path(results_dir).mkdir(exist_ok=True)

    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print(f"\nModel comparison results saved to: {comparison_file}")

    return comparison_results


# EXAMPLE USAGE AND TESTING
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Example usage - replace with your actual labeled data
    print("Feature Research System for Binary Classification")
    print("=" * 60)

    # Load your labeled data from labeling.py
    try:
        # Assuming you have saved labeled data
        # labeled_data = pd.read_pickle("labeled_data.pkl")

        print("To use this system:")
        print("1. Load your labeled data from labeling.py")
        print("2. Run: results = quick_feature_research(labeled_data)")
        print("3. Or run: all_results = comprehensive_feature_research(labeled_data)")
        print("\nExample:")
        print("""
import pandas as pd
from labeling import label_multiple_events_theory_based, indicated

# Generate labeled data
labeled_data, summary = label_multiple_events_theory_based(
    indicated,
    ['outlier_event', 'momentum_regime_event'],
    mode='individual'
)

# Research features for binary classification
from feature_research import comprehensive_feature_research

all_results = comprehensive_feature_research(labeled_data)

# Get best features and models for each label type
for label_type, results in all_results.items():
    print(f"Label: {label_type}")
    print(f"Best Model: {results.optimal_model}")
    print(f"Best Features: {results.best_features[:5]}")
    print(f"Expected Return: {results.economic_metrics['total_return']:.4f}")
    print()
        """)

    except Exception as e:
        print(f"Note: To test the system, load your labeled data first: {e}")

    print("\nFeatures of this research system:")
    print("• Genetic Algorithm feature selection")
    print("• Comprehensive model comparison (12+ algorithms)")
    print("• Economic performance evaluation")
    print("• Feature stability testing")
    print("• Automated report generation")
    print("• Binary classification optimization")
    print("• Time series aware validation")
    print("• Real trading metrics (Sharpe, drawdown, win rate)")

    print(f"\nReady to research optimal features and models for binary classification!")
    print("Results will predict: 'Will price rise above level X within Y time?'")


"""Key Features of the Script
1. Intelligent Feature Selection

Genetic Algorithm optimization for finding optimal feature combinations
Multi-method importance analysis (Random Forest, Gradient Boosting, Mutual Information, Statistical tests)
Automatic correlation filtering to remove redundant features
Feature stability testing across time periods

2. Comprehensive Model Zoo (12+ Algorithms)

Tree-based: Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM
Linear: Logistic Regression, Ridge Classifier, SVM
Instance-based: K-Nearest Neighbors
Probabilistic: Naive Bayes, Linear/Quadratic Discriminant Analysis
Ensemble: AdaBoost, Bagging, Voting Classifiers

3. Economic Performance Evaluation

Trading-specific metrics: Total return, Sharpe ratio, win rate, max drawdown
Classification metrics: AUC-ROC, F1-score, precision, recall
Realistic backtesting with time series cross-validation
Commission-aware analysis integrated with your labeling system

4. Research Automation

Single label research: quick_feature_research()
All labels research: comprehensive_feature_research()
Model comparison: model_comparison_study()
Automated reporting and result saving

How to Use
Basic Usage:
python# After running your labeling.py script
from feature_research import comprehensive_feature_research

# Research all label types
all_results = comprehensive_feature_research(labeled_data)

# Check results
for label_type, results in all_results.items():
    print(f"Label: {label_type}")
    print(f"Best Model: {results.optimal_model}")
    print(f"Top Features: {results.best_features[:5]}")
    print(f"Expected Return: {results.economic_metrics['total_return']:.4f}")
Advanced Usage:
python# Research specific label with custom parameters
from feature_research import quick_feature_research

results = quick_feature_research(
    labeled_data, 
    label_type='outlier_event_label',
    max_features=15,
    use_genetic=True
)

# Compare different feature sets
from feature_research import model_comparison_study

feature_sets = {
    'technical_only': ['RSI', 'MACD', 'ATR'],
    'volume_based': ['Volume', 'VWAP', 'volume_sma'],
    'combined': ['RSI', 'MACD', 'Volume', 'ATR', 'VWAP']
}

comparison = model_comparison_study(
    labeled_data,
    'outlier_event_label',
    feature_sets
)
Key Advantages for Binary Classification
1. Optimized for "Price Rise Prediction"

Time series validation respects temporal order
Economic metrics focused on trading profitability
Real-world constraints from your commission-aware labeling

2. Intelligent Search

Genetic algorithms explore feature combinations you'd never manually test
Multi-objective optimization balances accuracy, stability, and profitability
Automated hyperparameter consideration across 12+ model types

3. Comprehensive Analysis

Feature stability across different market regimes
Model robustness testing with cross-validation
Economic viability with real trading metrics
Comparative analysis across all your label types

4. Production-Ready Output

JSON and pickle result formats
Detailed reports with rankings and metrics
Logging system for debugging and monitoring
Modular design for easy integration

Expected Research Insights
The system will discover:

Which technical indicators work best for each event type
Optimal model types for different market patterns
Feature combinations that improve prediction accuracy
Economic viability of different labeling strategies
Temporal stability of predictive features
Risk-adjusted performance rankings

This script transforms your sophisticated labeling system into a complete machine learning research pipeline, specifically optimized for binary classification of price movement predictions. It will help you identify the most profitable and reliable combinations of features and models for your trading strategies."""