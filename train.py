"""Train and compare the three models, then save results for the dashboard.

  python train.py                                              # synthetic NSL-KDD-format practice data
  python train.py --dataset nsl --path data/nsl_kdd            # REAL NSL-KDD files
  python train.py --dataset cicids --path data/cicids2017 --sample 150000
  python train.py --binary                                     # Normal vs Attack only
"""
import argparse
import datetime
import json
import os
import time

import joblib
import numpy as np

from nidslib.data import load_dataset
from nidslib.evaluate import by_attack, evaluate
from nidslib.models import feature_importances, make_models, oversample

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["nsl", "cicids"], default="nsl")
ap.add_argument("--path", default="data/synthetic_nsl")
ap.add_argument("--sample", type=int, default=200_000, help="CICIDS2017: max rows to use")
ap.add_argument("--max-train", type=int, default=None, help="NSL-KDD: use only N training rows (faster)")
ap.add_argument("--binary", action="store_true", help="Normal vs Attack instead of attack families")
ap.add_argument("--no-balance", action="store_true", help="skip oversampling of rare classes")
ap.add_argument("--balance-target", type=int, default=3000)
a = ap.parse_args()

t0 = time.time()
ds = load_dataset(a.dataset, a.path, a.sample, a.binary, a.max_train)
classes = ds.classes
cid = {c: i for i, c in enumerate(classes)}
ytr, yte = ds.y_train.map(cid).to_numpy(), ds.y_test.map(cid).to_numpy()
synthetic = "synthetic" in a.path.lower()
print(f"\nDataset: {ds.name}{' (SYNTHETIC practice data)' if synthetic else ''} | "
      f"train {len(ds.X_train)} | test {len(ds.X_test)} | classes {classes}")
for n in ds.notes:
    print("  note:", n)
print("Train class counts:", ds.y_train.value_counts().to_dict())
print("Test  class counts:", ds.y_test.value_counts().to_dict())

Xtr, ytr_b = (ds.X_train, ytr) if a.no_balance else oversample(ds.X_train, ytr, a.balance_target)
if not a.no_balance:
    print(f"Oversampled rare classes in TRAINING data only: {len(ds.X_train)} -> {len(Xtr)} rows")

seen = set(ds.names_train.unique())
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
results = {}
print(f"\n{'model':44s} {'macroF1':>8s} {'acc':>6s} {'DR':>6s} {'FAR':>6s} {'train s':>8s} {'ms/1k':>7s}")
for name, pipe in make_models(Xtr, ds.cat_cols, ds.num_cols).items():
    t = time.time()
    pipe.fit(Xtr, ytr_b)
    train_s = time.time() - t
    t = time.time()
    pred = np.asarray(pipe.predict(ds.X_test)).astype(int)
    pred_ms = (time.time() - t) * 1000 / max(len(ds.X_test), 1) * 1000
    m = evaluate(yte, pred, classes)
    m.update(train_seconds=round(train_s, 2), predict_ms_per_1k=round(pred_ms, 1),
             importances=feature_importances(pipe, ds.cat_cols), by_attack=by_attack(ds.names_test, yte, pred, seen))
    # by_attack compares ints to ints; keep names readable in results
    results[name] = m
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
    joblib.dump({"pipe": pipe, "classes": classes, "features": list(ds.X_train.columns), "kind": ds.kind,
                 "name": name}, f"models/{slug}.joblib")
    print(f"{name:44s} {m['macro_f1']:8.3f} {m['accuracy']:6.3f} {m['detection_rate']:6.3f} "
          f"{m['false_alarm_rate']:6.3f} {train_s:8.1f} {pred_ms:7.1f}")

best = max(results, key=lambda k: results[k]["macro_f1"])
print(f"\nBest model by macro-F1: {best}")
print("DR = detection rate (attacks caught) | FAR = false alarm rate (normal traffic wrongly flagged)")

payload = {
    "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset": {"name": ds.name, "kind": ds.kind, "path": a.path, "synthetic": synthetic, "binary": a.binary,
                "classes": classes, "n_train": int(len(ds.X_train)), "n_test": int(len(ds.X_test)),
                "train_counts": ds.y_train.value_counts().to_dict(), "test_counts": ds.y_test.value_counts().to_dict(),
                "balanced_train": not a.no_balance, "notes": ds.notes},
    "best": best, "models": results,
}
with open("results/results.json", "w") as f:
    json.dump(payload, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

sample = ds.X_test.copy()
sample["attack_name"], sample["category"] = ds.names_test.values, ds.y_test.values
sample.sample(min(300, len(sample)), random_state=1).to_csv("results/replay.csv", index=False)
print(f"Saved models/*.joblib, results/results.json, results/replay.csv  (total {time.time() - t0:.0f}s)")
