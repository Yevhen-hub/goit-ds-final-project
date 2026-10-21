"""Прогнозування відтоку для одного клієнта (використовується і в app/, і з командного рядка)."""
import json
import sys
from functools import lru_cache

import joblib
import pandas as pd

from src.preprocessing import RAW_FEATURES, ROOT

MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "metadata.json"


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def predict_customer(customer: dict) -> dict:
    """customer: словник із RAW_FEATURES. reamining_contract / download_avg / upload_avg можуть бути None."""
    threshold = json.loads(META_PATH.read_text())["threshold"]
    row = {f: customer.get(f) for f in RAW_FEATURES}
    X = pd.DataFrame([row], columns=RAW_FEATURES).astype(float)
    proba = float(load_model().predict_proba(X)[0, 1])
    high = proba >= threshold
    return {
        "probability": round(proba, 4),
        "risk": "high" if high else "low",
        "message": "Клієнт має високу ймовірність відтоку" if high else "Клієнт має низьку ймовірність відтоку",
    }


if __name__ == "__main__":
    # python -m src.predict '{"is_tv_subscriber":0,"is_movie_package_subscriber":0,"subscription_age":2.5,
    #   "bill_avg":20,"reamining_contract":null,"service_failure_count":0,"download_avg":5,"upload_avg":1,"download_over_limit":0}'
    print(json.dumps(predict_customer(json.loads(sys.argv[1])), ensure_ascii=False, indent=2))
