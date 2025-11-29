# inference.py (put in project root)
import json, os, joblib, pandas as pd

def init():
    global model, scaler, feature_columns
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "rf_model.joblib")
    model = joblib.load(model_path)
    scaler_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "rf_scaler.joblib")
    feature_names_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "feature_names.json")
    scaler = None
    feature_columns = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(feature_names_path):
        feature_columns = json.load(open(feature_names_path))

def run(raw_data):
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data
        df = pd.DataFrame(data if isinstance(data, list) else [data])
        if feature_columns:
            for c in feature_columns:
                if c not in df.columns:
                    df[c] = 0
            df = df[feature_columns]
        X = df.values
        if scaler is not None:
            X = scaler.transform(X)
        preds = model.predict(X)
        return json.dumps({"predictions": [float(p) for p in preds]})
    except Exception as e:
        return json.dumps({"error": str(e)})
