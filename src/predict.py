"""Score network connections with the trained hybrid IDS.

Usage:  python predict.py path/to/connections.txt   (NSL-KDD format, with or without label columns)
"""

import sys

import joblib
import pandas as pd

from data_prep import COLUMNS, ROOT, TEST_PATH, features

MODEL_PATH = ROOT / "models" / "ids_model.joblib"


def predict(df, bundle=None):
    bundle = bundle or joblib.load(MODEL_PATH)
    X = features(df)
    predicted_class = bundle["classifier"].predict(X)
    anomaly_score = -bundle["anomaly_detector"].score_samples(X)
    is_anomaly = anomaly_score > bundle["anomaly_threshold"]

    out = pd.DataFrame(index=df.index)
    out["predicted_class"] = predicted_class
    out["anomaly_score"] = anomaly_score.round(3)
    out["alert"] = (predicted_class != "Normal") | is_anomaly
    # An alert from the anomaly detector alone means "unusual, type unknown", which may be a new attack
    out.loc[(predicted_class == "Normal") & is_anomaly, "predicted_class"] = "Unknown anomaly"
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else TEST_PATH
    raw = pd.read_csv(path, header=None)
    raw.columns = COLUMNS[: raw.shape[1]]
    result = predict(raw)
    print(result.head(15).to_string())
    print(f"\n{result['alert'].sum():,} of {len(result):,} connections raised an alert")
    print(result["predicted_class"].value_counts().to_string())
