import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import os

def evaluate_model(model, X_test, y_test, save_plots=True):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"🎯 ROC-AUC Score: {roc_auc:.4f}")
    
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"📈 Prediction distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"📊 Probability range: min={y_prob.min():.3f}, max={y_prob.max():.3f}, mean={y_prob.mean():.3f}")

    if save_plots:
        os.makedirs("app/static", exist_ok=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk (0)", "High Risk (1)"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("app/static/confusion_matrix.png", dpi=120)
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="darkorange", lw=2)
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
        plt.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("app/static/roc_curve.png", dpi=120)
        plt.close()

        # Probability distribution plot
        plt.figure(figsize=(7, 4))
        import pandas as pd
        
        pred_df_0 = y_prob[np.array(y_test) == 0]
        pred_df_1 = y_prob[np.array(y_test) == 1]
        plt.hist(pred_df_0, bins=30, alpha=0.6, label="Actual: Low Risk (0)", color="steelblue")
        plt.hist(pred_df_1, bins=30, alpha=0.6, label="Actual: High Risk (1)", color="tomato")
        plt.axvline(0.5, color="black", linestyle="--", label="Decision boundary (0.5)")
        plt.xlabel("Predicted Probability of High Risk")
        plt.ylabel("Count")
        plt.title("Prediction Probability Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig("app/static/prob_dist.png", dpi=120)
        plt.close()

        print("✅ All plots saved to app/static/")
        

    return {"roc_auc": roc_auc, "y_pred": y_pred, "y_prob": y_prob}