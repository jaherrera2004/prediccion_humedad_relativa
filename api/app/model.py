from datetime import datetime, timezone
from pathlib import Path
import json
import joblib
import numpy as np

ROOT          = Path(__file__).resolve().parents[2]
MODEL_PATH    = ROOT / "models" / "model.pkl"
FEATURES_PATH = ROOT / "models" / "features.json"
METRICS_PATH  = ROOT / "models" / "metrics.json"
DATA_PATH     = ROOT / "data"   / "weatherHistory_clean.csv"

CORR_COLS = [
    "temperature", "apparent_temperature", "humidity",
    "wind_speed", "wind_bearing", "visibility", "pressure",
]

PAGE_SIZE_MAX = 50

model       = None
features    = None
metrics     = None
correlation = None
_dataset_df = None


def load():
    global model, features, metrics, correlation, _dataset_df
    import pandas as pd
    model   = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    _dataset_df = pd.read_csv(DATA_PATH)
    correlation = _compute_correlation()


def _compute_correlation() -> dict:
    df   = _dataset_df[CORR_COLS].copy()
    corr = df.corr().round(3)
    return {"labels": CORR_COLS, "values": corr.values.tolist()}


def get_dataset_page(page: int, page_size: int) -> dict:
    page_size = min(page_size, PAGE_SIZE_MAX)
    total     = len(_dataset_df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page      = max(1, min(page, total_pages))
    start     = (page - 1) * page_size
    chunk     = _dataset_df.iloc[start : start + page_size]
    records   = chunk.where(chunk.notna(), other=None).to_dict(orient="records")
    return {"data": records, "page": page, "page_size": page_size,
            "total": total, "total_pages": total_pages}


def _extract_rule(X_input: np.ndarray, max_steps: int = 5) -> str:
    tree         = model.tree_
    node_ids     = model.decision_path(X_input).indices

    rules = []
    for node_id in node_ids[:max_steps]:
        if tree.feature[node_id] == -2:  # nodo hoja
            break
        feature   = features[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        value     = X_input[0][tree.feature[node_id]]
        direction = "<=" if value <= threshold else ">"
        rules.append(f"{feature} {direction} {threshold:.2f}")

    return " → ".join(rules)


def predict(data: dict) -> dict:
    now = datetime.now(timezone.utc)

    # Agregar features temporales automáticamente
    data["mes"]        = now.month
    data["hora"]       = now.hour
    data["dia_semana"] = now.weekday()

    X = np.array([[data[f] for f in features]])

    clase        = model.predict(X)[0]
    probs        = model.predict_proba(X)[0]
    probabilidades = {cls: round(float(p), 4) for cls, p in zip(model.classes_, probs)}
    regla        = _extract_rule(X)

    return {"clase": clase, "probabilidades": probabilidades, "regla": regla}
