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


main_example(labeled_data)


