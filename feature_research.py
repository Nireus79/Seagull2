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
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools

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

warnings.filterwarnings('ignore')


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


class OptimizedGeneticFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Optimized Genetic Algorithm for feature selection with sequential processing support
    """

    def __init__(self,
                 estimator,
                 population_size=30,
                 generations=15,
                 mutation_rate=0.15,
                 crossover_rate=0.8,
                 tournament_size=3,
                 random_state=42,
                 early_stopping_rounds=5,
                 min_features=3,
                 max_features_ratio=0.5):
        self.estimator = estimator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.min_features = min_features
        self.max_features_ratio = max_features_ratio

    def _create_individual(self, n_features):
        """Create random feature selection mask with improved initialization"""
        np.random.seed(self.random_state + np.random.randint(0, 1000))

        # Calculate target number of features
        max_features = max(self.min_features, int(n_features * self.max_features_ratio))
        target_features = np.random.randint(self.min_features, max_features + 1)

        # Create individual with target number of features
        individual = np.zeros(n_features, dtype=bool)
        selected_indices = np.random.choice(n_features, target_features, replace=False)
        individual[selected_indices] = True

        return individual

    def _evaluate_individual(self, individual, X, y):
        """Evaluate feature subset using cross-validation with caching"""
        if individual.sum() < self.min_features:
            return 0.0

        X_subset = X[:, individual]
        try:
            # Use simplified cross-validation for speed
            cv_scores = cross_val_score(
                self.estimator, X_subset, y,
                cv=3,  # Reduced from TimeSeriesSplit for speed
                scoring='roc_auc',
                n_jobs=1
            )
            # Penalty for too many features (encourage parsimony)
            feature_penalty = (individual.sum() / len(individual)) * 0.05
            stability_bonus = 0.01 if individual.sum() <= len(individual) * 0.3 else 0

            return cv_scores.mean() - feature_penalty + stability_bonus
        except Exception:
            return 0.0

    def _tournament_selection(self, population, fitness_scores):
        """Optimized tournament selection"""
        selected = []
        population_size = len(population)

        for _ in range(self.population_size):
            tournament_indices = np.random.choice(
                population_size, self.tournament_size, replace=False
            )
            winner_idx = tournament_indices[
                np.argmax([fitness_scores[i] for i in tournament_indices])
            ]
            selected.append(population[winner_idx].copy())
        return selected

    def _crossover(self, parent1, parent2):
        """Improved crossover with feature preservation"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Two-point crossover
        n_features = len(parent1)
        point1, point2 = sorted(np.random.choice(n_features, 2, replace=False))

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1[point1:point2] = parent2[point1:point2]
        child2[point1:point2] = parent1[point1:point2]

        return child1, child2

    def _mutate(self, individual):
        """Enhanced mutation with feature count preservation"""
        mutated = individual.copy()
        n_features = len(individual)

        # Adaptive mutation rate based on current feature count
        current_features = individual.sum()
        if current_features < self.min_features:
            # Force addition of features
            available_indices = np.where(~individual)[0]
            if len(available_indices) > 0:
                add_count = min(self.min_features - current_features, len(available_indices))
                add_indices = np.random.choice(available_indices, add_count, replace=False)
                mutated[add_indices] = True
        else:
            # Standard mutation
            for i in range(n_features):
                if np.random.random() < self.mutation_rate:
                    mutated[i] = not mutated[i]

        # Ensure minimum features
        if mutated.sum() < self.min_features:
            available_indices = np.where(~mutated)[0]
            if len(available_indices) > 0:
                add_count = min(self.min_features - mutated.sum(), len(available_indices))
                add_indices = np.random.choice(available_indices, add_count, replace=False)
                mutated[add_indices] = True

        return mutated

    def fit(self, X, y):
        """Fit genetic algorithm with early stopping and progress tracking"""
        X, y = check_X_y(X, y)
        n_features = X.shape[1]

        # Initialize population
        population = [self._create_individual(n_features) for _ in range(self.population_size)]

        best_fitness = -np.inf
        best_individual = None
        no_improvement_count = 0
        fitness_history = []

        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = [
                self._evaluate_individual(individual, X, y)
                for individual in population
            ]

            # Track best individual
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            fitness_history.append(best_fitness)

            # Early stopping
            if no_improvement_count >= self.early_stopping_rounds:
                break

            # Selection, crossover, and mutation
            selected = self._tournament_selection(population, fitness_scores)
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

            # Elitism: preserve best individual
            new_population = new_population[:self.population_size - 1]
            new_population.insert(0, best_individual.copy())
            population = new_population

        self.best_features_mask_ = best_individual
        self.best_fitness_ = best_fitness
        self.fitness_history_ = fitness_history
        self.generations_used_ = generation + 1

        return self

    def transform(self, X):
        """Transform data using selected features"""
        X = check_array(X)
        return X[:, self.best_features_mask_]

    def get_selected_features(self, feature_names=None):
        """Get list of selected feature names/indices"""
        if feature_names is not None:
            return [name for name, selected in zip(feature_names, self.best_features_mask_) if selected]
        else:
            return [i for i, selected in enumerate(self.best_features_mask_) if selected]


class OptimizedBinaryClassificationFeatureResearch:
    """
    Optimized feature research system with DataFrame support and sequential processing
    """

    def __init__(self,
                 labeled_data: Union[pd.DataFrame, str, Path],
                 results_dir: Optional[str] = None,
                 random_state: int = 42,
                 n_jobs: int = 1,
                 use_caching: bool = True,
                 verbose: bool = True):
        """
        Initialize feature research system

        Args:
            labeled_data: DataFrame with features and labels, or path to data file
            results_dir: Directory to save results (optional)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
            use_caching: Whether to cache intermediate results
            verbose: Whether to print progress information
        """
        # Handle different input types
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
        self.n_jobs = max(1, n_jobs)  # Ensure at least 1 job
        self.use_caching = use_caching
        self.verbose = verbose

        # Initialize cache
        self._cache = {} if use_caching else None

        # Set up logging
        if self.results_dir and verbose:
            self._setup_logging()
        else:
            self.logger = self._setup_simple_logger()

        # Detect features and labels
        self._detect_features_and_labels()

        # Initialize optimized model zoo
        self._initialize_optimized_model_zoo()

        if self.verbose:
            self.logger.info(
                f"Initialized with {len(self.feature_columns)} features and {len(self.label_columns)} label types"
            )
            self.logger.info(f"Processing mode: {'Sequential' if n_jobs == 1 else f'Parallel ({n_jobs} jobs)'}")

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

    def _setup_logging(self):
        """Set up file and console logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'research.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_simple_logger(self):
        """Set up simple console logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _detect_features_and_labels(self):
        """Optimized feature and label detection"""
        # Label columns end with '_label'
        self.label_columns = [col for col in self.data.columns if col.endswith('_label')]

        # Exclude non-feature columns with optimized pattern matching
        exclude_patterns = {
            '_label', '_barrier_touched', '_touch_time', '_return', '_holding_hours',
            '_event', 'event_type', 'any_event'
        }

        # Include OHLCV as features
        ohlcv_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}

        self.feature_columns = []
        numeric_dtypes = {'float64', 'int64', 'float32', 'int32', 'float16', 'int16'}

        for col in self.data.columns:
            # Skip if matches exclude patterns
            if any(pattern in col for pattern in exclude_patterns):
                continue

            # Include if numeric and not explicitly excluded
            if (self.data[col].dtype.name in numeric_dtypes or
                    col in ohlcv_cols):
                self.feature_columns.append(col)

        if self.verbose:
            self.logger.info(f"Detected {len(self.feature_columns)} features")
            self.logger.info(f"Detected {len(self.label_columns)} label types: {self.label_columns}")

    def _initialize_optimized_model_zoo(self):
        """Initialize optimized model zoo for sequential and parallel processing"""
        # Base parameters optimized for speed and performance
        base_params = {'random_state': self.random_state}

        self.model_zoo = {
            # Fast tree-based models
            'random_forest': RandomForestClassifier(
                n_estimators=50, max_depth=8, min_samples_split=10,
                n_jobs=1, **base_params  # Always use 1 job per model for better control
            ),

            'extra_trees': ExtraTreesClassifier(
                n_estimators=50, max_depth=8, min_samples_split=10,
                n_jobs=1, **base_params
            ),

            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                **base_params
            ),

            # Fast linear models
            'logistic_regression': LogisticRegression(
                max_iter=500, **base_params
            ),

            'ridge_classifier': RidgeClassifier(**base_params),

            # Efficient models
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=7),

            'decision_tree': DecisionTreeClassifier(
                max_depth=8, min_samples_split=10, **base_params
            ),

            # Discriminant analysis
            'lda': LinearDiscriminantAnalysis(),
        }

        # Add advanced models if available
        if HAS_XGBOOST:
            self.model_zoo['xgboost'] = xgb.XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                eval_metric='logloss', verbosity=0, **base_params
            )

        if HAS_LIGHTGBM:
            self.model_zoo['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                verbosity=-1, **base_params
            )

        if self.verbose:
            self.logger.info(f"Initialized {len(self.model_zoo)} optimized models")

    def prepare_data_for_label(self, label_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Optimized data preparation with caching
        """
        cache_key = f"prepared_data_{label_type}"

        if self.use_caching and cache_key in self._cache:
            return self._cache[cache_key]

        # Get samples with non-null labels
        valid_samples = self.data[label_type].notna()

        if valid_samples.sum() == 0:
            raise ValueError(f"No valid samples found for label {label_type}")

        # Prepare features and target
        X = self.data.loc[valid_samples, self.feature_columns].copy()
        y = self.data.loc[valid_samples, label_type].copy()

        # Optimized missing value handling
        if X.isnull().any().any():
            # Fill with forward/backward fill, then median
            X = X.fillna(method='ffill').fillna(method='bfill')
            X = X.fillna(X.median())

        # Remove constant features efficiently
        feature_vars = X.var()
        constant_features = feature_vars[feature_vars == 0].index.tolist()
        if constant_features:
            X = X.drop(columns=constant_features)
            if self.verbose:
                self.logger.warning(f"Removed {len(constant_features)} constant features")

        # Remove highly correlated features (optimized)
        if len(X.columns) > 1:
            corr_matrix = X.corr().abs()
            # Get upper triangle
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            # Find highly correlated pairs
            high_corr_features = [
                column for column in upper_tri.columns
                if any(upper_tri[column] > 0.95)
            ]
            if high_corr_features:
                X = X.drop(columns=high_corr_features[:len(high_corr_features) // 2])
                if self.verbose:
                    self.logger.info(f"Removed {len(high_corr_features) // 2} highly correlated features")

        # Ensure binary classification
        if y.nunique() != 2:
            if self.verbose:
                self.logger.warning(f"Label {label_type} is not binary: {y.unique()}")

        result = (X, y)

        if self.use_caching:
            self._cache[cache_key] = result

        if self.verbose:
            self.logger.info(f"Prepared data for {label_type}: {X.shape[0]} samples, {X.shape[1]} features")
            self.logger.info(f"Label distribution: {y.value_counts().to_dict()}")

        return result

    def research_feature_importance(self,
                                    label_type: str,
                                    methods: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Optimized feature importance research with sequential processing support
        """
        if methods is None:
            methods = ['mutual_info', 'random_forest', 'statistical']

        X, y = self.prepare_data_for_label(label_type)
        importance_results = {}

        # Sequential processing of importance methods
        for method in methods:
            try:
                if method == 'mutual_info':
                    scores = mutual_info_classif(
                        X, y, random_state=self.random_state, n_neighbors=3
                    )
                    importance_results[method] = dict(zip(X.columns, scores))

                elif method == 'random_forest':
                    rf = RandomForestClassifier(
                        n_estimators=30, max_depth=6,
                        random_state=self.random_state, n_jobs=1
                    )
                    rf.fit(X, y)
                    importance_results[method] = dict(zip(X.columns, rf.feature_importances_))

                elif method == 'statistical':
                    f_scores, _ = f_classif(X, y)
                    importance_results[method] = dict(zip(X.columns, f_scores))

                elif method == 'gradient_boosting':
                    gb = GradientBoostingClassifier(
                        n_estimators=30, max_depth=4,
                        random_state=self.random_state
                    )
                    gb.fit(X, y)
                    importance_results[method] = dict(zip(X.columns, gb.feature_importances_))

            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Method {method} failed: {e}")
                continue

        return importance_results

    def genetic_feature_selection(self,
                                  label_type: str,
                                  base_estimator=None,
                                  population_size: int = 20,
                                  generations: int = 10) -> Tuple[List[str], float]:
        """
        Optimized genetic algorithm feature selection
        """
        X, y = self.prepare_data_for_label(label_type)

        if base_estimator is None:
            base_estimator = RandomForestClassifier(
                n_estimators=30, max_depth=6,
                random_state=self.random_state, n_jobs=1
            )

        # Use optimized genetic selector
        ga_selector = OptimizedGeneticFeatureSelector(
            estimator=base_estimator,
            population_size=population_size,
            generations=generations,
            random_state=self.random_state
        )

        if self.verbose:
            self.logger.info(f"Starting optimized genetic algorithm for {label_type}")

        ga_selector.fit(X.values, y.values)

        selected_features = ga_selector.get_selected_features(X.columns.tolist())

        if self.verbose:
            self.logger.info(
                f"GA selected {len(selected_features)} features "
                f"(score: {ga_selector.best_fitness_:.4f}, "
                f"generations: {ga_selector.generations_used_})"
            )

        return selected_features, ga_selector.best_fitness_

    def _evaluate_single_model(self, model_name: str, model, X, y) -> Dict[str, Dict[str, float]]:
        """Helper function for single model evaluation"""
        try:
            # Fast cross-validation with fewer splits
            cv = TimeSeriesSplit(n_splits=3)

            # Calculate metrics
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
            acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
            prec_scores = cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=1)
            rec_scores = cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=1)
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=1)

            return {
                'auc_roc': {'mean': auc_scores.mean(), 'std': auc_scores.std()},
                'accuracy': {'mean': acc_scores.mean(), 'std': acc_scores.std()},
                'precision': {'mean': prec_scores.mean(), 'std': prec_scores.std()},
                'recall': {'mean': rec_scores.mean(), 'std': rec_scores.std()},
                'f1': {'mean': f1_scores.mean(), 'std': f1_scores.std()}
            }
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Model {model_name} evaluation failed: {e}")
            return {}

    def evaluate_model_performance(self,
                                   label_type: str,
                                   features: List[str] = None,
                                   models: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Optimized model evaluation with sequential/parallel processing support
        """
        X, y = self.prepare_data_for_label(label_type)

        if features is not None:
            available_features = [f for f in features if f in X.columns]
            X = X[available_features]

        if models is None:
            models = list(self.model_zoo.keys())

        results = {}

        if self.n_jobs == 1:
            # Sequential processing
            for model_name in models:
                if model_name not in self.model_zoo:
                    continue

                if self.verbose:
                    self.logger.info(f"Evaluating {model_name} for {label_type}")

                model = self.model_zoo[model_name]
                model_results = self._evaluate_single_model(model_name, model, X, y)

                if model_results:
                    results[model_name] = model_results
        else:
            # Parallel processing using ThreadPoolExecutor for better compatibility
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {}

                for model_name in models:
                    if model_name in self.model_zoo:
                        future = executor.submit(
                            self._evaluate_single_model,
                            model_name, self.model_zoo[model_name], X, y
                        )
                        futures[future] = model_name

                for future in futures:
                    model_name = futures[future]
                    try:
                        model_results = future.result()
                        if model_results:
                            results[model_name] = model_results
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(f"Parallel evaluation of {model_name} failed: {e}")

        return results

    def calculate_economic_metrics(self,
                                   label_type: str,
                                   features: List[str],
                                   model_name: str = 'random_forest') -> Dict[str, float]:
        """
        Optimized economic metrics calculation
        """
        X, y = self.prepare_data_for_label(label_type)
        X = X[features]

        # Get return data if available
        return_col = label_type.replace('_label', '_return')
        returns_data = None

        if return_col in self.data.columns:
            returns_data = self.data.loc[X.index, return_col].fillna(0)

        # Fast model training and prediction
        model = self.model_zoo[model_name]

        # Simplified time series validation for speed
        split_point = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        try:
            probabilities = model.predict_proba(X_test)[:, 1]
        except:
            probabilities = predictions.astype(float)

        # Calculate metrics
        economic_metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1_score': f1_score(y_test, predictions, zero_division=0),
        }

        try:
            economic_metrics['auc_roc'] = roc_auc_score(y_test, probabilities)
        except:
            economic_metrics['auc_roc'] = 0.5

        # Economic metrics if return data available
        if returns_data is not None:
            test_returns = returns_data.iloc[split_point:]
            strategy_returns = np.where(predictions == 1, test_returns, 0)

            economic_metrics.update({
                'total_return': strategy_returns.sum(),
                'mean_return_per_trade': strategy_returns[strategy_returns != 0].mean() if (
                            strategy_returns != 0).any() else 0,
                'win_rate': (strategy_returns > 0).mean(),
                'num_trades': (strategy_returns != 0).sum(),
                'sharpe_ratio': strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(
                    252) if strategy_returns.std() > 0 else 0
            })

            # Maximum drawdown calculation
            if len(strategy_returns) > 0:
                cumulative_returns = (1 + strategy_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                economic_metrics['max_drawdown'] = drawdown.min()
            else:
                economic_metrics['max_drawdown'] = 0

        return economic_metrics

    def test_feature_stability(self,
                               label_type: str,
                               features: List[str],
                               test_periods: int = 4) -> Dict[str, float]:
        """
        Optimized feature stability testing with reduced periods for speed
        """
        X, y = self.prepare_data_for_label(label_type)
        X = X[features]

        stability_scores = {}
        n_samples = len(X)
        period_size = max(50, n_samples // test_periods)

        for feature in features:
            period_importances = []

            for period in range(test_periods):
                start_idx = period * period_size
                end_idx = min(start_idx + period_size, n_samples)

                if end_idx - start_idx < 30:  # Minimum samples required
                    continue

                X_period = X.iloc[start_idx:end_idx]
                y_period = y.iloc[start_idx:end_idx]

                if y_period.nunique() < 2:
                    continue

                try:
                    mi_score = mutual_info_classif(
                        X_period[[feature]], y_period,
                        random_state=self.random_state
                    )[0]
                    period_importances.append(mi_score)
                except:
                    continue

            if len(period_importances) >= 2:
                mean_importance = np.mean(period_importances)
                std_importance = np.std(period_importances)
                stability_scores[feature] = 1 - (std_importance / (mean_importance + 1e-8))
            else:
                stability_scores[feature] = 0.0

        return stability_scores

    def comprehensive_research(self,
                               label_type: str,
                               use_genetic_selection: bool = True,
                               max_features: int = 15) -> ResearchResults:
        """
        Optimized comprehensive feature research
        """
        if self.verbose:
            self.logger.info(f"Starting comprehensive research for {label_type}")

        # Step 1: Feature importance analysis
        importance_results = self.research_feature_importance(label_type)

        # Step 2: Combine importance scores
        combined_scores = defaultdict(float)
        for method, scores in importance_results.items():
            # Normalize scores within each method
            if scores:
                max_score = max(scores.values())
                for feature, score in scores.items():
                    combined_scores[feature] += score / max_score if max_score > 0 else 0

        # Normalize combined scores
        if combined_scores:
            max_combined = max(combined_scores.values())
            combined_scores = {k: v / max_combined for k, v in combined_scores.items()}

        # Step 3: Select top features
        top_features_by_importance = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_features * 2]

        top_features = [feature for feature, _ in top_features_by_importance]

        # Step 4: Genetic algorithm optimization (optional)
        if use_genetic_selection and len(top_features) > 8:
            if self.verbose:
                self.logger.info("Applying genetic algorithm for feature optimization")
            try:
                selected_features, ga_score = self.genetic_feature_selection(
                    label_type,
                    population_size=20,
                    generations=10
                )
                # Combine GA results with importance-based selection
                final_features = list(set(selected_features + top_features[:max_features]))[:max_features]
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Genetic algorithm failed, using importance-based selection: {e}")
                final_features = top_features[:max_features]
        else:
            final_features = top_features[:max_features]

        # Step 5: Feature stability testing
        if self.verbose:
            self.logger.info("Testing feature stability")
        try:
            stability_scores = self.test_feature_stability(label_type, final_features)
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Feature stability testing failed: {e}")
            stability_scores = {f: 0.5 for f in final_features}

        # Step 6: Model evaluation
        if self.verbose:
            self.logger.info("Evaluating models with selected features")
        model_performance = self.evaluate_model_performance(
            label_type,
            features=final_features
        )

        # Step 7: Rank models
        model_rankings = {}
        for model_name, metrics in model_performance.items():
            # Weighted score: AUC-ROC (50%) + F1 (30%) + Precision (20%)
            auc = metrics.get('auc_roc', {}).get('mean', 0)
            f1 = metrics.get('f1', {}).get('mean', 0)
            precision = metrics.get('precision', {}).get('mean', 0)

            score = auc * 0.5 + f1 * 0.3 + precision * 0.2
            model_rankings[model_name] = score

        # Find best model
        if model_rankings:
            best_model = max(model_rankings.items(), key=lambda x: x[1])[0]
        else:
            best_model = 'random_forest'  # Default fallback

        # Step 8: Economic metrics
        if self.verbose:
            self.logger.info(f"Calculating economic metrics using {best_model}")
        try:
            economic_metrics = self.calculate_economic_metrics(
                label_type, final_features, best_model
            )
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Economic metrics calculation failed: {e}")
            economic_metrics = {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1_score': 0.5}

        # Create results
        results = ResearchResults(
            label_type=label_type,
            best_features=final_features,
            feature_scores=dict(combined_scores),
            model_performance=model_performance,
            stability_metrics=stability_scores,
            economic_metrics=economic_metrics,
            research_timestamp=datetime.now().isoformat(),
            model_rankings=model_rankings,
            optimal_model=best_model
        )

        if self.verbose:
            best_score = model_rankings.get(best_model, 0)
            self.logger.info(
                f"Research completed for {label_type}. "
                f"Best model: {best_model} (score: {best_score:.4f}), "
                f"Features: {len(final_features)}"
            )

        return results

    def research_all_labels(self,
                            use_genetic_selection: bool = True,
                            max_features: int = 15) -> Dict[str, ResearchResults]:
        """
        Research all available label types with progress tracking
        """
        all_results = {}
        total_labels = len(self.label_columns)

        for i, label_type in enumerate(self.label_columns, 1):
            try:
                if self.verbose:
                    self.logger.info(f"\n{'=' * 60}")
                    self.logger.info(f"RESEARCHING LABEL {i}/{total_labels}: {label_type}")
                    self.logger.info(f"{'=' * 60}")

                results = self.comprehensive_research(
                    label_type, use_genetic_selection, max_features
                )
                all_results[label_type] = results

                # Save individual results if results directory is available
                if self.results_dir:
                    self.save_results(results, suffix=f"_{label_type}")

            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Failed to research {label_type}: {e}")
                continue

        # Generate comparative analysis
        if all_results:
            if self.verbose:
                self.logger.info("Generating comparative analysis")
            self.generate_comparative_analysis(all_results)

        return all_results

    def save_results(self, results: ResearchResults, suffix: str = ""):
        """Save research results to files"""
        if not self.results_dir:
            if self.verbose:
                self.logger.warning("No results directory specified, skipping save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        try:
            results_dict = asdict(results)
            json_file = self.results_dir / f"research_results{suffix}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)

            # Save as pickle
            pickle_file = self.results_dir / f"research_results{suffix}_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(results, f)

            if self.verbose:
                self.logger.info(f"Results saved to {json_file}")

        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Failed to save results: {e}")

    def generate_comparative_analysis(self, all_results: Dict[str, ResearchResults]):
        """Generate and save comparative analysis"""
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
        for results in all_results.values():
            analysis['summary']['model_preferences'][results.optimal_model] += 1

        # Best performing labels by metrics
        metrics_to_analyze = ['auc_roc', 'total_return', 'sharpe_ratio', 'win_rate']

        for metric in metrics_to_analyze:
            best_label = None
            best_value = float('-inf')

            for label_type, results in all_results.items():
                value = results.economic_metrics.get(metric, float('-inf'))
                if value > best_value:
                    best_value = value
                    best_label = label_type

            if best_label:
                analysis['summary']['best_performing_labels'][metric] = {
                    'label': best_label,
                    'value': best_value
                }

        # Common features analysis
        feature_counts = defaultdict(int)
        for results in all_results.values():
            for feature in results.best_features:
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
            auc_roc = results.economic_metrics.get('auc_roc', 0.5)

            # Combined economic score
            economic_score = (
                    total_return * 0.3 +
                    sharpe_ratio * 0.25 +
                    win_rate * 0.25 +
                    auc_roc * 0.2
            )

            economic_ranking.append({
                'label_type': label_type,
                'economic_score': economic_score,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'auc_roc': auc_roc,
                'optimal_model': results.optimal_model
            })

        analysis['summary']['economic_performance_ranking'] = sorted(
            economic_ranking, key=lambda x: x['economic_score'], reverse=True
        )

        # Save analysis
        if self.results_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = self.results_dir / f"comparative_analysis_{timestamp}.json"
            try:
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                if self.verbose:
                    self.logger.info(f"Comparative analysis saved to {analysis_file}")
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to save comparative analysis: {e}")

        # Print summary
        self._print_analysis_summary(analysis)

        return analysis

    def _print_analysis_summary(self, analysis):
        """Print comparative analysis summary"""
        print(f"\n{'=' * 80}")
        print("COMPARATIVE ANALYSIS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total labels researched: {analysis['summary']['total_labels_researched']}")

        print(f"\nModel preferences:")
        model_prefs = dict(analysis['summary']['model_preferences'])
        for model, count in sorted(model_prefs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} labels")

        print(f"\nTop 10 common features:")
        common_features = analysis['summary']['common_features']
        for i, (feature, count) in enumerate(list(common_features.items())[:10], 1):
            print(f"  {i:2d}. {feature}: used in {count} labels")

        print(f"\nTop 5 economically performing labels:")
        rankings = analysis['summary']['economic_performance_ranking']
        for i, item in enumerate(rankings[:5], 1):
            print(f"  {i}. {item['label_type']}: Score {item['economic_score']:.4f}")
            print(f"     Return: {item['total_return']:.4f}, Sharpe: {item['sharpe_ratio']:.4f}, "
                  f"Win Rate: {item['win_rate']:.3f}")

    def generate_report(self, results: ResearchResults) -> str:
        """Generate detailed research report"""
        report = f"""
OPTIMIZED BINARY CLASSIFICATION FEATURE RESEARCH REPORT
{'=' * 70}

Label Type: {results.label_type}
Research Date: {results.research_timestamp}
Optimal Model: {results.optimal_model}

FEATURE SELECTION RESULTS
{'=' * 35}
Selected Features ({len(results.best_features)}):
{', '.join(results.best_features)}

Top 10 Features by Importance Score:
"""

        # Top features by score
        top_features = sorted(
            results.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for i, (feature, score) in enumerate(top_features, 1):
            report += f"{i:2d}. {feature:<30} {score:.4f}\n"

        report += f"""
MODEL PERFORMANCE COMPARISON
{'=' * 35}
"""

        # Model rankings
        model_rankings = sorted(
            results.model_rankings.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (model, score) in enumerate(model_rankings, 1):
            model_metrics = results.model_performance.get(model, {})
            auc = model_metrics.get('auc_roc', {}).get('mean', 0)
            f1 = model_metrics.get('f1', {}).get('mean', 0)
            precision = model_metrics.get('precision', {}).get('mean', 0)

            report += f"{i:2d}. {model:<20} Score: {score:.4f} "
            report += f"(AUC: {auc:.3f}, F1: {f1:.3f}, Prec: {precision:.3f})\n"

        report += f"""
ECONOMIC PERFORMANCE METRICS
{'=' * 35}
Accuracy: {results.economic_metrics.get('accuracy', 0):.4f}
Precision: {results.economic_metrics.get('precision', 0):.4f}
Recall: {results.economic_metrics.get('recall', 0):.4f}
F1-Score: {results.economic_metrics.get('f1_score', 0):.4f}
AUC-ROC: {results.economic_metrics.get('auc_roc', 0):.4f}

Trading Performance:
Total Return: {results.economic_metrics.get('total_return', 0):.4f}
Win Rate: {results.economic_metrics.get('win_rate', 0):.4f}
Sharpe Ratio: {results.economic_metrics.get('sharpe_ratio', 0):.4f}
Max Drawdown: {results.economic_metrics.get('max_drawdown', 0):.4f}
Number of Trades: {results.economic_metrics.get('num_trades', 0)}

FEATURE STABILITY ANALYSIS
{'=' * 35}
"""

        # Feature stability (top 10)
        stable_features = sorted(
            results.stability_metrics.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for i, (feature, stability) in enumerate(stable_features, 1):
            report += f"{i:2d}. {feature:<30} {stability:.4f}\n"

        # Save report if results directory available
        if self.results_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.results_dir / f"report_{results.label_type}_{timestamp}.txt"
            try:
                with open(report_file, 'w') as f:
                    f.write(report)
                if self.verbose:
                    self.logger.info(f"Report saved to {report_file}")
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to save report: {e}")

        return report

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of available features and labels"""
        return {
            'total_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'total_labels': len(self.label_columns),
            'label_columns': self.label_columns,
            'data_shape': self.data.shape,
            'data_index': {
                'start': self.data.index[0] if len(self.data) > 0 else None,
                'end': self.data.index[-1] if len(self.data) > 0 else None,
                'frequency': pd.infer_freq(self.data.index) if len(self.data) > 1 else None
            }
        }


# OPTIMIZED CONVENIENCE FUNCTIONS

def quick_feature_research(labeled_data: Union[pd.DataFrame, str, Path],
                           label_type: str = None,
                           max_features: int = 15,
                           use_genetic: bool = True,
                           results_dir: Optional[str] = None,
                           n_jobs: int = 1,
                           verbose: bool = True) -> ResearchResults:
    """
    Quick optimized feature research for a single label type

    Args:
        labeled_data: DataFrame or path to labeled data
        label_type: Specific label to research (None = use first available)
        max_features: Maximum features to select
        use_genetic: Use genetic algorithm optimization
        results_dir: Directory for results (optional)
        n_jobs: Number of parallel jobs (1 for sequential)
        verbose: Print progress information

    Returns:
        ResearchResults object
    """
    researcher = OptimizedBinaryClassificationFeatureResearch(
        labeled_data, results_dir, n_jobs=n_jobs, verbose=verbose
    )

    if label_type is None:
        if not researcher.label_columns:
            raise ValueError("No label columns found in data")
        label_type = researcher.label_columns[0]
        if verbose:
            print(f"Using label type: {label_type}")

    results = researcher.comprehensive_research(label_type, use_genetic, max_features)

    # Generate and print report
    report = researcher.generate_report(results)
    if verbose:
        print(report)

    return results


def comprehensive_feature_research(labeled_data: Union[pd.DataFrame, str, Path],
                                   max_features: int = 15,
                                   use_genetic: bool = True,
                                   results_dir: Optional[str] = None,
                                   n_jobs: int = 1,
                                   verbose: bool = True) -> Dict[str, ResearchResults]:
    """
    Optimized comprehensive feature research for all label types

    Args:
        labeled_data: DataFrame or path to labeled data
        max_features: Maximum features per label
        use_genetic: Use genetic algorithm optimization
        results_dir: Directory for results (optional)
        n_jobs: Number of parallel jobs (1 for sequential)
        verbose: Print progress information

    Returns:
        Dictionary of all research results
    """
    researcher = OptimizedBinaryClassificationFeatureResearch(
        labeled_data, results_dir, n_jobs=n_jobs, verbose=verbose
    )

    if not researcher.label_columns:
        raise ValueError("No label columns found in data")

    if verbose:
        print(f"Starting comprehensive research for {len(researcher.label_columns)} label types:")
        print(f"Available labels: {researcher.label_columns}")
        print(f"Processing mode: {'Sequential' if n_jobs == 1 else f'Parallel ({n_jobs} jobs)'}")

    all_results = researcher.research_all_labels(use_genetic, max_features)

    if verbose and results_dir:
        print(f"\nResearch completed! Results saved to: {results_dir}")

    return all_results


def model_comparison_study(labeled_data: Union[pd.DataFrame, str, Path],
                           label_type: str,
                           feature_sets: Dict[str, List[str]],
                           results_dir: Optional[str] = None,
                           n_jobs: int = 1,
                           verbose: bool = True) -> Dict[str, Dict[str, Dict]]:
    """
    Compare different models across different feature sets

    Args:
        labeled_data: DataFrame or path to labeled data
        label_type: Label to analyze
        feature_sets: Dictionary mapping set names to feature lists
        results_dir: Directory for results (optional)
        n_jobs: Number of parallel jobs (1 for sequential)
        verbose: Print progress information

    Returns:
        Nested dictionary of results [feature_set][model][metric]
    """
    researcher = OptimizedBinaryClassificationFeatureResearch(
        labeled_data, results_dir, n_jobs=n_jobs, verbose=verbose
    )

    comparison_results = {}

    for set_name, features in feature_sets.items():
        if verbose:
            print(f"\nEvaluating feature set: {set_name} ({len(features)} features)")

        model_performance = researcher.evaluate_model_performance(
            label_type, features=features
        )
        comparison_results[set_name] = model_performance

    # Save comparison results
    if results_dir:
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = results_path / f"model_comparison_{label_type}_{timestamp}.json"

        try:
            with open(comparison_file, 'w') as f:
                json.dump(comparison_results, f, indent=2, default=str)
            if verbose:
                print(f"\nModel comparison results saved to: {comparison_file}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save comparison results: {e}")

    return comparison_results


def create_feature_research_pipeline(labeled_data: Union[pd.DataFrame, str, Path],
                                     config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a complete feature research pipeline with customizable configuration

    Args:
        labeled_data: DataFrame or path to labeled data
        config: Configuration dictionary with research parameters

    Returns:
        Dictionary containing all research results and pipeline information
    """
    # Default configuration
    default_config = {
        'max_features': 15,
        'use_genetic': True,
        'n_jobs': 1,
        'verbose': True,
        'results_dir': 'feature_research_pipeline_results',
        'save_intermediate_results': True,
        'generate_plots': False,  # Could be extended for visualization
        'quick_mode': False  # Faster research with reduced parameters
    }

    if config:
        default_config.update(config)

    config = default_config

    if config['verbose']:
        print("=" * 70)
        print("OPTIMIZED FEATURE RESEARCH PIPELINE")
        print("=" * 70)
        print(f"Configuration: {json.dumps(config, indent=2)}")

    # Initialize researcher
    researcher = OptimizedBinaryClassificationFeatureResearch(
        labeled_data,
        results_dir=config['results_dir'],
        n_jobs=config['n_jobs'],
        verbose=config['verbose']
    )

    # Get data summary
    data_summary = researcher.get_feature_summary()

    if config['verbose']:
        print(f"\nData Summary:")
        print(f"  Features: {data_summary['total_features']}")
        print(f"  Labels: {data_summary['total_labels']}")
        print(f"  Data shape: {data_summary['data_shape']}")
        print(f"  Time range: {data_summary['data_index']['start']} to {data_summary['data_index']['end']}")

    # Quick mode adjustments
    if config['quick_mode']:
        config['max_features'] = min(10, config['max_features'])
        config['use_genetic'] = False
        if config['verbose']:
            print("  Quick mode: Reduced parameters for faster execution")

    # Run comprehensive research
    start_time = datetime.now()

    all_results = researcher.research_all_labels(
        use_genetic_selection=config['use_genetic'],
        max_features=config['max_features']
    )

    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    # Create pipeline results
    pipeline_results = {
        'config': config,
        'data_summary': data_summary,
        'research_results': all_results,
        'execution_info': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'execution_time_seconds': execution_time,
            'labels_processed': len(all_results),
            'successful_results': len([r for r in all_results.values() if r.best_features])
        }
    }

    if config['verbose']:
        print(f"\n{'=' * 70}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Labels processed: {len(all_results)}")
        print(f"Successful results: {pipeline_results['execution_info']['successful_results']}")

        if all_results:
            best_label = max(all_results.items(),
                             key=lambda x: x[1].model_rankings.get(x[1].optimal_model, 0))
            print(f"Best performing label: {best_label[0]}")
            print(f"Best model: {best_label[1].optimal_model}")

    return pipeline_results


