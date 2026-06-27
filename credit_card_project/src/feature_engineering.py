# Feature engineering is intentionally minimal for this dataset.
# The raw technical indicators (RSI, MACD, BB, etc.) are already
# well-formed features. Adding derived features did not improve
# predictive power and caused train/predict inconsistency.

def add_features(df):
    """
    Optional: add features only if they improve model performance.
    For the synthetic trading dataset, raw features are used directly.
    """
    print(f"✅ Feature engineering skipped (raw features used). Shape: {df.shape}")
    return df