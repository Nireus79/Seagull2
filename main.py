from feature_research import (
    quick_feature_research,
    comprehensive_feature_research,
    OptimizedBinaryClassificationFeatureResearch
)
from labeling import labeled_data


# ============================================================================
# USING YOUR EXISTING DATAFRAME - NO SAMPLE DATA
# ============================================================================

# Assuming you already have your DataFrame loaded as 'df'
# df = your_existing_dataframe

# ============================================================================
# METHOD 1: Quick Research (Single Label)
# ============================================================================

def research_single_label(df, label_name=None):
    """Research a single label from your DataFrame"""

    results = quick_feature_research(
        labeled_data=df,  # YOUR DataFrame
        label_type=label_name,  # Specify label or None for first one
        max_features=15,
        use_genetic=True,
        results_dir='my_results',
        n_jobs=1,
        verbose=True
    )

    return results


# ============================================================================
# METHOD 2: Research All Labels
# ============================================================================

def research_all_labels(df):
    """Research all labels in your DataFrame"""

    all_results = comprehensive_feature_research(
        labeled_data=df,  # YOUR DataFrame
        max_features=15,
        use_genetic=True,
        results_dir='all_labels_results',
        n_jobs=1,
        verbose=True
    )

    return all_results


# ============================================================================
# METHOD 3: Using the Class Directly
# ============================================================================

def research_with_class(df):
    """Use the main class with your DataFrame"""

    # Initialize with YOUR DataFrame
    researcher = OptimizedBinaryClassificationFeatureResearch(
        labeled_data=df,  # YOUR DataFrame
        results_dir='class_results',
        n_jobs=1,
        verbose=True
    )

    # See what labels are available
    summary = researcher.get_feature_summary()
    print(f"Available labels: {summary['label_columns']}")

    # Research first label (or specify which one you want)
    if summary['label_columns']:
        label_to_research = summary['label_columns'][0]  # or choose specific one
        results = researcher.comprehensive_research(
            label_type=label_to_research,
            use_genetic_selection=True,
            max_features=20
        )

        # Generate report
        report = researcher.generate_report(results)
        print(report)

        return researcher, results
    else:
        print("No label columns found! Make sure your columns end with '_label'")
        return None, None


# ============================================================================
# SIMPLE ONE-LINER USAGE
# ============================================================================

# If your DataFrame is called 'df', just run this:
# results = quick_feature_research(df)

# ============================================================================
# CHECK YOUR DATAFRAME FORMAT
# ============================================================================

def check_dataframe_format(df):
    """Check if your DataFrame has the right format"""

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    # Find label columns
    label_cols = [col for col in df.columns if col.endswith('_label')]
    print(f"\nLabel columns found ({len(label_cols)}): {label_cols}")

    # Find potential feature columns
    exclude_patterns = ['_label', '_return', '_barrier', '_touch', '_event']
    feature_cols = [col for col in df.columns
                    if not any(pattern in col for pattern in exclude_patterns)]
    print(f"Feature columns found ({len(feature_cols)}): {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    # Check data types
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    print(f"Numeric columns ({len(numeric_cols)}): {len(numeric_cols)} found")

    if len(label_cols) == 0:
        print("\n❌ WARNING: No label columns found!")
        print("   Label columns must end with '_label' and contain binary values (0 or 1)")
    else:
        print(f"\n✅ Found {len(label_cols)} label column(s)")

        # Check label values
        for label in label_cols[:3]:  # Check first 3 labels
            unique_vals = df[label].dropna().unique()
            print(f"   {label}: unique values = {unique_vals}")

    if len(feature_cols) == 0:
        print("\n❌ WARNING: No feature columns found!")
    else:
        print(f"\n✅ Found {len(feature_cols)} feature column(s)")

    return label_cols, feature_cols


# ============================================================================
# MAIN USAGE EXAMPLES
# ============================================================================

# Replace 'your_df' with your actual DataFrame variable name

def main_example(your_df):
    """Main example using your DataFrame"""

    print("Checking DataFrame format...")
    label_cols, feature_cols = check_dataframe_format(your_df)

    if not label_cols:
        print("Cannot proceed - no label columns found!")
        return None

    print(f"\nStarting research on {len(label_cols)} labels...")

    # Option 1: Quick research on first label
    results = quick_feature_research(
        labeled_data=your_df,
        label_type=label_cols[0],  # Use first available label
        verbose=True
    )

    print(f"\nResults for {label_cols[0]}:")
    print(f"Best features: {results.best_features}")
    print(f"Optimal model: {results.optimal_model}")
    print(f"AUC-ROC: {results.economic_metrics.get('auc_roc', 0):.3f}")

    return results


# main_example(labeled_data)
# OPTIMIZED BINARY CLASSIFICATION FEATURE RESEARCH REPORT
# ======================================================================
#
# Label Type: CUSUM_event_label
# Research Date: 2025-09-07T20:05:53.827563
# Optimal Model: random_forest
#
# FEATURE SELECTION RESULTS =================================== Selected Features (15): EMA_medium_distance_pct,
# EMA_slow_distance_atr, momentum_short, momentum_medium, vol_breakout_score, EMA_fast_distance_pct,
# EMA_slow_distance_pct, 1D_EMA_short, 4H_volatility, volume_zscore, 4H_EMA_short, MACD_signal, momentum_ratio_4H,
# BB_position, momentum_regime_score
#
# Top 10 Features by Importance Score:
#  1. momentum_long                  1.0000
#  2. price_vs_1D_pct                0.9440
#  3. momentum_medium                0.7682
#  4. BB_position                    0.7248
#  5. 4H_EMA_short                   0.6943
#  6. price_vs_4H_pct                0.6774
#  7. EMA_slow_distance_pct          0.6369
#  8. vol_breakout_score             0.6334
#  9. 1D_momentum_medium             0.6153
# 10. EMA_medium_distance_pct        0.6086
#
# MODEL PERFORMANCE COMPARISON
# ===================================
#  1. random_forest        Score: 0.4242 (AUC: 0.502, F1: 0.293, Prec: 0.427)
#  2. decision_tree        Score: 0.4187 (AUC: 0.499, F1: 0.326, Prec: 0.357)
#  3. knn                  Score: 0.4157 (AUC: 0.491, F1: 0.329, Prec: 0.356)
#  4. extra_trees          Score: 0.4129 (AUC: 0.503, F1: 0.246, Prec: 0.439)
#  5. gradient_boosting    Score: 0.4016 (AUC: 0.494, F1: 0.288, Prec: 0.342)
#  6. naive_bayes          Score: 0.3959 (AUC: 0.516, F1: 0.196, Prec: 0.396)
#  7. lda                  Score: 0.3510 (AUC: 0.526, F1: 0.071, Prec: 0.333)
#  8. logistic_regression  Score: 0.3430 (AUC: 0.524, F1: 0.056, Prec: 0.323)
#  9. ridge_classifier     Score: 0.3096 (AUC: 0.524, F1: 0.060, Prec: 0.147)
#
# ECONOMIC PERFORMANCE METRICS
# ===================================
# Accuracy: 0.5000
# Precision: 0.5000
# Recall: 0.5000
# F1-Score: 0.5000
# AUC-ROC: 0.0000
#
# Trading Performance:
# Total Return: 0.0000
# Win Rate: 0.0000
# Sharpe Ratio: 0.0000
# Max Drawdown: 0.0000
# Number of Trades: 0
#
# FEATURE STABILITY ANALYSIS
# ===================================
#  1. momentum_short                 1.0000
#  2. volume_zscore                  1.0000
#  3. BB_position                    1.0000
#  4. EMA_fast_distance_pct          0.5693
#  5. 4H_volatility                  0.4189
#  6. 4H_EMA_short                   0.3369
#  7. momentum_ratio_4H              -0.0000
#  8. momentum_medium                -0.0159
#  9. 1D_EMA_short                   -0.1492
# 10. EMA_slow_distance_atr          -0.2287
#
#
# Results for CUSUM_event_label: Best features: ['EMA_medium_distance_pct', 'EMA_slow_distance_atr',
# 'momentum_short', 'momentum_medium', 'vol_breakout_score', 'EMA_fast_distance_pct', 'EMA_slow_distance_pct',
# '1D_EMA_short', '4H_volatility', 'volume_zscore', '4H_EMA_short', 'MACD_signal', 'momentum_ratio_4H',
# 'BB_position', 'momentum_regime_score'] Optimal model: random_forest AUC-ROC: 0.000


research_with_class(labeled_data)
# OPTIMIZED BINARY CLASSIFICATION FEATURE RESEARCH REPORT
# ======================================================================
#
# Label Type: CUSUM_event_label
# Research Date: 2025-09-08T00:09:55.603309
# Optimal Model: decision_tree
#
# FEATURE SELECTION RESULTS =================================== Selected Features (20): ATR, returns, MACD_signal,
# 4H_EMA_short, 4H_momentum_short, momentum_long, Volume, momentum_ratio_4H, momentum_medium, 1D_momentum_medium,
# price_vs_4H_pct, momentum_short, SMA_medium_50, MACD_histogram, EMA_medium_26, BB_width_pct, RSI,
# EMA_medium_distance_pct, 1D_RSI, EMA_slow_distance_pct
#
# Top 10 Features by Importance Score:
#  1. momentum_long                  1.0000
#  2. price_vs_1D_pct                0.9440
#  3. momentum_medium                0.7682
#  4. BB_position                    0.7248
#  5. 4H_EMA_short                   0.6943
#  6. price_vs_4H_pct                0.6774
#  7. EMA_slow_distance_pct          0.6369
#  8. vol_breakout_score             0.6334
#  9. 1D_momentum_medium             0.6153
# 10. EMA_medium_distance_pct        0.6086
#
# MODEL PERFORMANCE COMPARISON
# ===================================
#  1. decision_tree        Score: 0.4423 (AUC: 0.516, F1: 0.350, Prec: 0.397)
#  2. gradient_boosting    Score: 0.4301 (AUC: 0.540, F1: 0.284, Prec: 0.375)
#  3. random_forest        Score: 0.4149 (AUC: 0.529, F1: 0.257, Prec: 0.367)
#  4. knn                  Score: 0.4100 (AUC: 0.475, F1: 0.339, Prec: 0.355)
#  5. naive_bayes          Score: 0.4037 (AUC: 0.526, F1: 0.199, Prec: 0.404)
#  6. lda                  Score: 0.3996 (AUC: 0.515, F1: 0.129, Prec: 0.516)
#  7. extra_trees          Score: 0.3939 (AUC: 0.511, F1: 0.193, Prec: 0.402)
#  8. ridge_classifier     Score: 0.3814 (AUC: 0.511, F1: 0.129, Prec: 0.436)
#  9. logistic_regression  Score: 0.3673 (AUC: 0.508, F1: 0.126, Prec: 0.379)
#
# ECONOMIC PERFORMANCE METRICS
# ===================================
# Accuracy: 0.5000
# Precision: 0.5000
# Recall: 0.5000
# F1-Score: 0.5000
# AUC-ROC: 0.0000
#
# Trading Performance:
# Total Return: 0.0000
# Win Rate: 0.0000
# Sharpe Ratio: 0.0000
# Max Drawdown: 0.0000
# Number of Trades: 0
#
# FEATURE STABILITY ANALYSIS
# ===================================
#  1. returns                        1.0000
#  2. Volume                         1.0000
#  3. price_vs_4H_pct                1.0000
#  4. momentum_short                 1.0000
#  5. SMA_medium_50                  1.0000
#  6. 4H_EMA_short                   0.3369
#  7. EMA_medium_26                  0.1845
#  8. 1D_momentum_medium             0.0684
#  9. momentum_ratio_4H              -0.0000
# 10. BB_width_pct                   -0.0009
#
# 2025-09-08 00:09:55,603 - WARNING - Economic metrics calculation failed: 'numpy.ndarray' object has no attribute 'expanding'
# 2025-09-08 00:09:55,603 - INFO - Research completed for CUSUM_event_label. Best model: decision_tree (score: 0.4423), Features: 20
# 2025-09-08 00:09:55,605 - INFO - Report saved to class_results\report_CUSUM_event_label_20250908_000955.txt

