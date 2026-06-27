from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import yaml
import os

app = Flask(__name__)

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_feature_names():
    """Load the EXACT feature names saved during training."""
    return joblib.load("models/feature_names.pkl")

def get_sample_values():
    """Return sample/median values from the dataset for the form."""
    config = load_config()
    df = pd.read_csv(config["data"]["raw_path"])
    feature_names = get_feature_names()
    medians = df[feature_names].median().round(4).to_dict()
    return medians

@app.route("/")
def index():
    features = get_feature_names()
    samples = get_sample_values()
    return render_template("index.html", features=features, samples=samples)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model = joblib.load(load_config()["model"]["save_path"])
        scaler = joblib.load("models/scaler.pkl")
        feature_names = get_feature_names()

        data = request.get_json()
        
        # Build row in EXACT training order
        values = []
        for f in feature_names:
            v = data.get(f)
            if v is None or v == "":
                return jsonify({"error": f"Missing feature: {f}"}), 400
            values.append(float(v))

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
        prob_high = float(probabilities[1])
        prob_low = float(probabilities[0])

        return jsonify({
            "prediction": prediction,
            "probability": round(prob_high, 4),
            "prob_low": round(prob_low, 4),
            "prob_high": round(prob_high, 4),
            "risk_label": "HIGH RISK" if prediction == 1 else "LOW RISK",
            "feature_count": len(feature_names)
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    config = load_config()
    app.run(
        host=config["app"]["host"],
        port=config["app"]["port"],
        debug=config["app"]["debug"]
    )