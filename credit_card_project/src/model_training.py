import numpy as np
import joblib
import yaml
import os
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model(X_train, y_train, config):
    p  = config["training"]
    rs = config["data"]["random_state"]

    # Apply SMOTE — needed now because dataset is imbalanced (~6% default rate)
    sm = SMOTE(random_state=rs)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    unique, counts = np.unique(y_res, return_counts=True)
    print(f"✅ SMOTE applied: {dict(zip(unique.tolist(), counts.tolist()))}")

    model = XGBClassifier(
        n_estimators     = p["n_estimators"],
        max_depth        = p["max_depth"],
        learning_rate    = p["learning_rate"],
        subsample        = p["subsample"],
        colsample_bytree = p["colsample_bytree"],
        min_child_weight = p["min_child_weight"],
        gamma            = p["gamma"],
        reg_alpha        = p["reg_alpha"],
        reg_lambda       = p["reg_lambda"],
        eval_metric      = "logloss",
        random_state     = rs
    )
    model.fit(X_res, y_res)

    preds = model.predict(X_train)
    print(f"✅ Model trained: XGBoost")
    print(f"   Train prediction dist: {dict(zip(*np.unique(preds, return_counts=True)))}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, config["model"]["save_path"])
    print(f"✅ Model saved to {config['model']['save_path']}")
    return model