#!/usr/bin/env python3
"""
train.py
Usage:
  python src/train.py --data data/data.csv --out models/rf_model.joblib --metrics metrics/metrics.json
"""

import argparse, json, joblib, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(path, use_cols=None):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        # fallback to xlsx
        return pd.read_excel(path, engine='openpyxl', usecols=use_cols)

def preprocess(df):
    # Keep expected columns, drop NA rows
    expected = ['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes','controller_kind','cpu_usage']
    df = df.copy()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[expected].dropna()

    # One-hot encode controller_kind
    df = pd.get_dummies(df, columns=['controller_kind'], prefix='controller_kind')

    X = df.drop(columns=['cpu_usage'])
    y = df['cpu_usage'].astype(float)
    return X, y

def train_and_eval(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    return model, preds, {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}

def main(args):
    df = load_data(args.data)
    X, y = preprocess(df)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # scale for SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # pick model from args
    if args.model == "rf":
        model = RandomForestRegressor(n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth, random_state=args.random_state)
        trained, preds, metrics = train_and_eval(X_train, X_test, y_train, y_test, model)
    elif args.model == "svr":
        model = SVR(C=args.svr_C, epsilon=args.svr_epsilon)
        trained, preds, metrics = train_and_eval(X_train_scaled, X_test_scaled, y_train, y_test, model)
    elif args.model == "linear":
        model = LinearRegression()
        trained, preds, metrics = train_and_eval(X_train, X_test, y_train, y_test, model)
    else:
        raise ValueError("Unsupported model. Choose from rf, svr, linear.")

    # save model (and scaler if needed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(trained, args.out)

    # save scaler for SVR
    if args.model == "svr":
        joblib.dump(scaler, args.out + ".scaler.joblib")

    # save predictions and metrics
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # optionally save predictions.csv for plotting
    preds_df = pd.DataFrame({"y_test": list(map(float, y_test)), "y_pred": list(map(float, preds))})
    preds_df.to_csv(os.path.join("metrics", "preds.csv"), index=False)

    print("Done. Metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--model", default="rf", choices=["rf", "svr", "linear"])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    # RF hyperparams
    parser.add_argument("--rf_n_estimators", type=int, default=200)
    parser.add_argument("--rf_max_depth", type=int, default=10)
    # SVR hyperparams
    parser.add_argument("--svr_C", type=float, default=1.0)
    parser.add_argument("--svr_epsilon", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
