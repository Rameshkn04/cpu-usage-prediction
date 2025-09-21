import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import numpy as np

def plot_feature_importance(model, feature_names, out_path):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(8,5))
        sns.barplot(x=importances[order], y=np.array(feature_names)[order], color="orange")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    else:
        print("Model has no feature_importances_ attribute")

def plot_pred_vs_actual(preds_csv, out_path):
    df = pd.read_csv(preds_csv)
    plt.figure(figsize=(6,6))
    plt.scatter(df["y_test"], df["y_pred"], alpha=0.6)
    mn = min(df["y_test"].min(), df["y_pred"].min())
    mx = max(df["y_test"].max(), df["y_pred"].max())
    plt.plot([mn,mx], [mn,mx], 'r--')
    plt.xlabel("Actual CPU Usage")
    plt.ylabel("Predicted CPU Usage")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_residuals(preds_csv, out_path):
    df = pd.read_csv(preds_csv)
    residuals = df["y_test"] - df["y_pred"]
    plt.figure(figsize=(7,5))
    plt.scatter(df["y_pred"], residuals, alpha=0.6)
    plt.axhline(0, linestyle='--', color='red')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_error_distribution(preds_csv, out_path):
    df = pd.read_csv(preds_csv)
    residuals = df["y_test"] - df["y_pred"]
    plt.figure(figsize=(7,5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_learning_curve(metric_csv=None, out_path=None):
    # Placeholder: we used metrics/preds.csv only — for true learning curve produce via sklearn.learning_curve
    pass

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    preds_csv = "metrics/preds.csv"
    model_path = "models/rf_model.joblib"   # change per model
    # load model to get feature names (assumes you saved training feature names separately if needed)
    model = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    # infer feature names by reading preds.csv? Best practice is to save X.columns during preprocessing.
    if os.path.exists(preds_csv):
        df_preds = pd.read_csv(preds_csv)
    # plot functions
    # if you saved feature names at models/feature_names.json, read and pass
    feat_names_path = "models/feature_names.json"
    if os.path.exists(feat_names_path):
        import json
        feat_names = json.load(open(feat_names_path))
    else:
        feat_names = None

    if model is not None and feat_names is not None:
        plot_feature_importance(model, feat_names, "plots/feature_importance.png")
    if os.path.exists(preds_csv):
        plot_pred_vs_actual(preds_csv, "plots/pred_vs_actual.png")
        plot_residuals(preds_csv, "plots/residuals.png")
        plot_error_distribution(preds_csv, "plots/error_distribution.png")
    print("Plots saved under plots/")
