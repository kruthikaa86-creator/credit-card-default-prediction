import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_credit_dataset(n=5000, save_path="data/credit_data.csv"):
    """
    Generate a realistic credit card default dataset with genuine signal.
    Based on the structure of the UCI Credit Card Default dataset.
    AUC achievable: ~0.77-0.82
    """
    np.random.seed(42)

    LIMIT_BAL = np.random.lognormal(10.5, 0.8, n).clip(10000, 500000)
    AGE       = np.random.normal(35, 10, n).clip(21, 75)
    SEX       = np.random.choice([1, 2], n, p=[0.40, 0.60])
    EDUCATION = np.random.choice([1, 2, 3, 4], n, p=[0.35, 0.46, 0.16, 0.03])
    MARRIAGE  = np.random.choice([1, 2, 3], n, p=[0.45, 0.53, 0.02])

    PAY_0 = np.random.choice([-2,-1,0,1,2,3,4,5,6,7,8], n,
                              p=[0.15,0.25,0.30,0.10,0.07,0.05,0.03,0.02,0.01,0.01,0.01])
    PAY_2 = np.random.choice([-2,-1,0,1,2,3,4,5,6,7,8], n,
                              p=[0.14,0.24,0.31,0.11,0.07,0.05,0.03,0.02,0.01,0.01,0.01])
    PAY_3 = np.random.choice([-2,-1,0,1,2,3,4,5,6,7,8], n,
                              p=[0.13,0.24,0.32,0.11,0.07,0.05,0.03,0.02,0.01,0.01,0.01])

    BILL_AMT1 = np.random.lognormal(9.5, 1.2, n).clip(0, 400000)
    BILL_AMT2 = BILL_AMT1 * np.random.uniform(0.8, 1.2, n)
    PAY_AMT1  = np.maximum(0, BILL_AMT1 * np.random.uniform(0, 0.5, n))
    PAY_AMT2  = np.maximum(0, BILL_AMT2 * np.random.uniform(0, 0.5, n))

    # Target with REAL causal signal
    logit = (
        -3.5
        - 0.0000015 * LIMIT_BAL
        - 0.01      * AGE
        + 0.55      * PAY_0
        + 0.35      * PAY_2
        + 0.20      * PAY_3
        + 0.0000008 * BILL_AMT1
        - 0.000001  * PAY_AMT1
        - 0.000001  * PAY_AMT2
        + 0.15      * (EDUCATION == 3).astype(float)
        - 0.10      * (MARRIAGE  == 1).astype(float)
        + np.random.normal(0, 0.3, n)
    )
    prob   = 1 / (1 + np.exp(-logit))
    target = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'LIMIT_BAL': LIMIT_BAL, 'SEX': SEX, 'EDUCATION': EDUCATION,
        'MARRIAGE': MARRIAGE,   'AGE': AGE,
        'PAY_0': PAY_0,         'PAY_2': PAY_2, 'PAY_3': PAY_3,
        'BILL_AMT1': BILL_AMT1, 'BILL_AMT2': BILL_AMT2,
        'PAY_AMT1': PAY_AMT1,   'PAY_AMT2': PAY_AMT2,
        'target': target
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Credit dataset generated: {df.shape[0]} rows, "
          f"{target.sum()} defaults ({target.mean()*100:.1f}%)")
    return df

def load_data(config):
    path = config["data"]["raw_path"]
    if not os.path.exists(path):
        print("Dataset not found — generating credit default dataset...")
        df = generate_credit_dataset(save_path=path)
    else:
        df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df, config):
    drop_cols = config["features"].get("drop_columns", [])
    existing  = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing).drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    print(f"✅ Data cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def split_data(df, config):
    target   = config["data"]["target_column"]
    X, y     = df.drop(columns=[target]), df[target]
    os.makedirs("models", exist_ok=True)
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = config["data"]["test_size"],
        random_state= config["data"]["random_state"],
        stratify    = y
    )
    print(f"✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"   Train defaults: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler        = StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Features scaled and scaler saved.")
    return X_train_scaled, X_test_scaled