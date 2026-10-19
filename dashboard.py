"""Dashboard.   python dashboard.py   then open http://127.0.0.1:5000"""
import glob
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "results.json")
REPLAY = os.path.join(HERE, "results", "replay.csv")
_models = {}


def load_models():
    if not _models:
        for f in sorted(glob.glob(os.path.join(HERE, "models", "*.joblib"))):
            m = joblib.load(f)
            _models[m["name"]] = m
    return _models


def describe(row):
    """Short human-readable summary of one record."""
    if "protocol_type" in row:
        return (f"{row['protocol_type']}/{row['service']} flag={row['flag']} count={row['count']} "
                f"src_bytes={row['src_bytes']}")
    cols = [c for c in row.index if c not in ("attack_name", "category")][:3]
    return ", ".join(f"{c}={row[c]}" for c in cols)


@app.get("/")
def home():
    return render_template("dashboard.html")


@app.get("/api/results")
def api_results():
    if not os.path.exists(RESULTS):
        return jsonify({"error": "No results yet. Run: python train.py"}), 404
    with open(RESULTS) as f:
        return jsonify(json.load(f))


@app.get("/api/replay")
def api_replay():
    if not (os.path.exists(REPLAY) and os.path.isdir(os.path.join(HERE, "models"))):
        return jsonify({"error": "No models yet. Run: python train.py"}), 404
    n = max(1, min(request.args.get("n", 15, type=int), 50))
    pool = pd.read_csv(REPLAY)
    atk, norm = pool[pool["category"] != "Normal"], pool[pool["category"] == "Normal"]
    k = min(len(atk), -(-n * 6 // 10))  # about 60% attacks, so the replay is interesting
    df = pd.concat([atk.sample(k), norm.sample(min(len(norm), n - k))]).sample(frac=1)
    models = load_models()
    preds = {}
    for name, m in models.items():
        idx = m["pipe"].predict(df[m["features"]]).astype(int)
        preds[name] = [m["classes"][i] for i in idx]
    rows = [{"attack": r["attack_name"], "truth": r["category"], "info": describe(r),
             "preds": {name: preds[name][i] for name in preds}} for i, (_, r) in enumerate(df.iterrows())]
    return jsonify({"rows": rows, "models": list(models)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)  # localhost only
