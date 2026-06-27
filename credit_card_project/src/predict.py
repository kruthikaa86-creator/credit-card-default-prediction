import joblib
import numpy as np
import pandas as pd
import yaml

def load_artifacts(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = joblib.load(config["model"]["save_path"])
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

def predict_single(features: dict, config_path="config/config.yaml"):
    """
    Predict default risk for a single record.
    features: dict with keys matching training feature names.
    """
    model, scaler, feature_names = load_artifacts(config_path)
    
    # Build dataframe in exact training column order
    row = {f: features.get(f, 0.0) for f in feature_names}
    df = pd.DataFrame([row])[feature_names]
    
    df_scaled = scaler.transform(df)
    prediction = int(model.predict(df_scaled)[0])
    probability = float(model.predict_proba(df_scaled)[0][1])
    
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_label": "HIGH RISK" if prediction == 1 else "LOW RISK",
        "features_used": feature_names
    }

if __name__ == "__main__":
    # Test with sample values
    sample = {
        "open": 98.5, "high": 100.2, "low": 97.1, "close": 99.3,
        "volume": 7500000, "SMA_10": 98.0, "SMA_20": 97.5,
        "EMA_10": 98.2, "EMA_20": 97.8, "RSI": 55.0,
        "MACD": 0.15, "MACD_signal": 0.10, "MACD_hist": 0.05,
        "BB_middle": 97.5, "BB_std": 2.1,
        "BB_upper": 101.7, "BB_lower": 93.3
    }
    result = predict_single(sample)
    print(result)