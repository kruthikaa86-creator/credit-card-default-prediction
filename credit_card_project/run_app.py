"""
run_app.py — Full pipeline: load → clean → split → scale → train → evaluate
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import load_config, load_data, clean_data, split_data, scale_features
from src.feature_engineering import add_features
from src.model_training import train_model
from src.evaluation import evaluate_model

def run_pipeline():
    print("\n🚀 Credit Default Risk Pipeline")
    print("=" * 50)

    config = load_config()

    # Step 1: Load
    df = load_data(config)

    # Step 2: Clean (drops 'date' column per config)
    df = clean_data(df, config)

    # Step 3: Feature engineering (minimal for this dataset)
    df = add_features(df)

    # Step 4: Split — also saves feature_names.pkl
    X_train, X_test, y_train, y_test = split_data(df, config)

    # Step 5: Scale — saves scaler.pkl
    X_train_s, X_test_s = scale_features(X_train, X_test)

    # Step 6: Train — saves model
    model = train_model(X_train_s, y_train, config)

    # Step 7: Evaluate — saves plots
    results = evaluate_model(model, X_test_s, y_test, save_plots=True)

    print(f"\n✅ Pipeline complete!")
    print(f"   ROC-AUC:  {results['roc_auc']:.4f}")
    print(f"   Saved:    models/credit_model.pkl")
    print(f"             models/scaler.pkl")
    print(f"             models/feature_names.pkl")
    print(f"   Plots:    app/static/confusion_matrix.png")
    print(f"             app/static/roc_curve.png")
    print(f"             app/static/prob_dist.png")
    print(f"\n🌐 Now run: python app/app.py")

if __name__ == "__main__":
    run_pipeline()