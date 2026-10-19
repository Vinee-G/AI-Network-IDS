"""Classify records from a CSV file.
  python predict.py --csv data/synthetic_nsl/KDDTest+.txt            (NSL-KDD format, no header)
  python predict.py --csv my_flows.csv --model "Random Forest"         (CSV with a header row of feature names)
Writes results/predictions.csv
"""
import argparse
import glob
import sys

import joblib
import pandas as pd

from nidslib.schema import NSL_FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--model", default=None, help="model name from results.json (default: best model)")
a = ap.parse_args()

import json
best = json.load(open("results/results.json"))["best"] if not a.model else a.model
bundle = next((m for m in (joblib.load(f) for f in glob.glob("models/*.joblib")) if m["name"] == best), None)
if bundle is None:
    sys.exit(f"Model '{best}' not found in models/. Run train.py first.")

df = pd.read_csv(a.csv, low_memory=False)
df.columns = [str(c).strip() for c in df.columns]
if "protocol_type" not in df.columns and bundle["kind"] == "nsl":   # NSL-KDD files have no header row
    df = pd.read_csv(a.csv, header=None)
    df = df.iloc[:, :41]
    df.columns = NSL_FEATURES
missing = [c for c in bundle["features"] if c not in df.columns]
if missing:
    sys.exit(f"Missing columns: {missing[:5]}...")
pred = bundle["pipe"].predict(df[bundle["features"]]).astype(int)
df["predicted_class"] = [bundle["classes"][i] for i in pred]
df.to_csv("results/predictions.csv", index=False)
print(f"Model: {best}\n{df['predicted_class'].value_counts().to_string()}\nSaved results/predictions.csv")
