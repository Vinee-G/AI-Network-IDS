"""Run:  python tests/test_sanity.py      (needs no downloads; builds tiny data in a temp folder)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import make_synthetic
from nidslib.data import load_dataset
from nidslib.evaluate import evaluate
from nidslib.models import make_models, make_preprocessor, oversample


def _tiny():
    d = tempfile.mkdtemp()
    make_synthetic.main(d, n_train=3000, n_test=1200, seed=1)
    return load_dataset("nsl", d)


def test_nsl_loader():
    ds = _tiny()
    assert ds.classes == ["Normal", "DoS", "Probe", "R2L", "U2R"]
    assert ds.X_train.shape[1] == 41 and len(ds.y_train) == len(ds.X_train)
    unseen = set(ds.names_test) - set(ds.names_train)
    assert unseen, "test set should contain attacks never seen in training"


def test_oversample_only_grows_rare_classes():
    ds = _tiny()
    y = ds.y_train.map({c: i for i, c in enumerate(ds.classes)}).to_numpy()
    X2, y2 = oversample(ds.X_train, y, 800)
    counts = pd.Series(y2).value_counts()
    assert counts.min() >= 800 and len(X2) == len(y2)
    assert (pd.Series(y).value_counts().max() == counts.max())  # biggest class untouched


def test_models_train_and_score():
    ds = _tiny()
    y = ds.y_train.map({c: i for i, c in enumerate(ds.classes)}).to_numpy()
    yt = ds.y_test.map({c: i for i, c in enumerate(ds.classes)}).to_numpy()
    pipe = Pipeline([("pre", make_preprocessor(ds.X_train, ds.cat_cols, ds.num_cols)),
                     ("clf", RandomForestClassifier(n_estimators=40, random_state=0))]).fit(ds.X_train, y)
    m = evaluate(yt, pipe.predict(ds.X_test), ds.classes)
    assert m["macro_f1"] > 0.6 and 0 <= m["false_alarm_rate"] <= 0.2
    assert len(make_models(ds.X_train, ds.cat_cols, ds.num_cols)) == 3


def test_metrics_math():
    classes = ["Normal", "DoS"]
    m = evaluate([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0], classes)
    assert abs(m["detection_rate"] - 0.5) < 1e-9 and abs(m["false_alarm_rate"] - 0.25) < 1e-9


def test_cicids_loader_handles_messy_csv():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    labels = ["BENIGN"] * 300 + ["DoS Hulk"] * 120 + ["PortScan"] * 80 + ["Web Attack \x96 XSS"] * 40
    df = pd.DataFrame({" Flow Duration": rng.integers(1, 10**6, len(labels)).astype(float),
                       " Total Fwd Packets": rng.integers(1, 100, len(labels)),
                       "Flow Bytes/s": rng.random(len(labels)) * 1000, " Label": labels})
    df.loc[3, "Flow Bytes/s"] = np.inf          # real files contain inf / NaN
    df.loc[5, " Total Fwd Packets"] = np.nan
    df["Flow ID"] = "x"                          # identifier column must be dropped
    df.to_csv(os.path.join(d, "Monday.csv"), index=False, encoding="latin-1")
    ds = load_dataset("cicids", d, sample=2000)
    assert "Flow ID" not in ds.X_train.columns and ds.cat_cols == []
    assert set(ds.classes) >= {"Normal", "DoS", "PortScan", "WebAttack"}
    assert np.isfinite(ds.X_train.to_numpy()).all()


def test_dashboard_api_if_trained():
    if not os.path.exists("results/results.json"):
        return print("   (skipped: run train.py first)")
    from dashboard import app
    c = app.test_client()
    assert c.get("/").status_code == 200
    r = c.get("/api/results").get_json()
    assert r["best"] in r["models"]
    rep = c.get("/api/replay?n=5").get_json()
    assert len(rep["rows"]) == 5 and set(rep["models"]) == set(r["models"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
