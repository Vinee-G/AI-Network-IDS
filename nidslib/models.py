"""Preprocessing + the three models (Random Forest, XGBoost, small neural network)."""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # xgboost missing: keep the project runnable
    HAVE_XGB = False


def _log1p_clip(x):
    return np.log1p(np.clip(x, 0, None))


def make_preprocessor(X, cat_cols, num_cols):
    """Heavy-tailed non-negative columns get log1p; everything numeric is scaled; categories one-hot."""
    skew = [c for c in num_cols if X[c].min() >= 0 and X[c].max() > 1000]
    plain = [c for c in num_cols if c not in skew]
    parts = []
    if plain:
        parts.append(("num", StandardScaler(), plain))
    if skew:
        parts.append(("log", Pipeline([("log", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
                                       ("sc", StandardScaler())]), skew))
    if cat_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(parts)


def make_models(X, cat_cols, num_cols, seed=42):
    def pipe(clf):
        return Pipeline([("pre", make_preprocessor(X, cat_cols, num_cols)), ("clf", clf)])

    models = {"Random Forest": pipe(RandomForestClassifier(
        n_estimators=150, min_samples_leaf=1, n_jobs=-1, random_state=seed))}
    if HAVE_XGB:
        models["XGBoost"] = pipe(XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, eval_metric="mlogloss", random_state=seed))
    else:
        models["HistGradientBoosting (XGBoost not installed)"] = pipe(
            HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed))
    models["Neural Network (MLP)"] = pipe(MLPClassifier(
        hidden_layer_sizes=(64, 32), early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=8, max_iter=80, random_state=seed))
    return models


def oversample(X, y, target, seed=42):
    """Repeat rows of rare classes until each has `target` rows (training data only!)."""
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        rows = np.where(y == cls)[0]
        idx.extend(rows)
        if len(rows) < target:
            idx.extend(rng.choice(rows, target - len(rows), replace=True))
    idx = np.array(idx)
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def feature_importances(pipe, cat_cols, top=12):
    """Importances of tree models, with one-hot columns folded back to the original feature."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    names = pipe.named_steps["pre"].get_feature_names_out()
    agg = {}
    for n, v in zip(names, clf.feature_importances_):
        base = n.split("__", 1)[1]
        for c in cat_cols:
            if base.startswith(c + "_"):
                base = c
                break
        agg[base] = agg.get(base, 0.0) + float(v)
    return sorted(agg.items(), key=lambda t: -t[1])[:top]
