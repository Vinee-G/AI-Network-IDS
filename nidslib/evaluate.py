"""Metrics that matter for an IDS: per-class recall, detection rate, false-alarm rate."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate(y_true, y_pred, classes, normal="Normal"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    ok = s > 0
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ni = classes.index(normal)
    at, ap = y_true != ni, y_pred != ni
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(p[ok].mean()), "macro_recall": float(r[ok].mean()), "macro_f1": float(f[ok].mean()),
        "weighted_f1": float((f * s).sum() / s.sum()),
        "detection_rate": float((at & ap).sum() / max(at.sum(), 1)),      # attacks caught (any attack class)
        "false_alarm_rate": float((~at & ap).sum() / max((~at).sum(), 1)),  # normal traffic wrongly flagged
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion": cm.astype(int).tolist(),
    }


def by_attack(names_test, y_true, y_pred, seen_names, min_n=5):
    """Recall per specific attack name; flags attacks that never appeared in training."""
    df = pd.DataFrame({"name": np.asarray(names_test), "ok": np.asarray(y_true) == np.asarray(y_pred)})
    out = []
    for name, g in df.groupby("name"):
        if len(g) >= min_n and name != "normal" and name != "benign":
            out.append({"attack": name, "n": int(len(g)), "recall": float(g["ok"].mean()),
                        "seen_in_train": name in seen_names})
    return sorted(out, key=lambda d: (d["seen_in_train"], d["recall"]))
