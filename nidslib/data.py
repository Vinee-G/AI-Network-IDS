"""Loaders for NSL-KDD and CICIDS2017. Both return the same Dataset object."""
import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import ATTACK_MAP, NSL_CATEGORICAL, NSL_FEATURES, NSL_ORDER

TRAIN_FILES = ["KDDTrain+.txt", "KDDTrain+.csv", "KDDTrain.txt", "train.csv"]
TEST_FILES = ["KDDTest+.txt", "KDDTest+.csv", "KDDTest.txt", "test.csv"]
CICIDS_DROP = {"flow id", "source ip", "destination ip", "src ip", "dst ip", "timestamp", "source port",
               "src port", "simillarhttp", "unnamed: 0"}


@dataclass
class Dataset:
    name: str
    kind: str
    X_train: pd.DataFrame
    y_train: pd.Series
    names_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    names_test: pd.Series
    cat_cols: list = field(default_factory=list)
    num_cols: list = field(default_factory=list)
    synthetic: bool = False
    notes: list = field(default_factory=list)

    @property
    def classes(self):
        present = set(self.y_train.unique())
        first = [c for c in NSL_ORDER if c in present]
        return first + sorted(present - set(first))


# ---------------------------------------------------------------- NSL-KDD
def _find(folder, candidates):
    for c in candidates:
        p = os.path.join(folder, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"None of {candidates} found in '{folder}'. Put the NSL-KDD files there "
        f"(or run make_synthetic.py for practice data).")


def _read_nsl(path):
    first = pd.read_csv(path, header=None, nrows=1)
    n = first.shape[1]
    cols = (NSL_FEATURES + ["attack_name", "difficulty"])[:n]
    token = str(first.iloc[0, 0]).strip().lstrip("-").replace(".", "", 1)
    header = 0 if not token.isdigit() else None  # some mirrors include a header row
    df = pd.read_csv(path, header=header, names=cols)
    df["attack_name"] = df["attack_name"].astype(str).str.strip().str.rstrip(".").str.lower()
    return df


def load_nsl(folder, max_train=None, seed=42):
    tr, te = _read_nsl(_find(folder, TRAIN_FILES)), _read_nsl(_find(folder, TEST_FILES))
    if max_train and len(tr) > max_train:
        tr = tr.sample(max_train, random_state=seed).reset_index(drop=True)
    notes = []
    out = []
    for df in (tr, te):
        cat = df["attack_name"].map(lambda a: ATTACK_MAP.get(a, "Other"))
        unknown = sorted(set(df.loc[cat == "Other", "attack_name"]))
        if unknown:
            notes.append(f"unmapped attack names treated as 'Other': {unknown[:8]}")
        out.append((df[NSL_FEATURES].copy(), cat, df["attack_name"]))
    (Xtr, ytr, ntr), (Xte, yte, nte) = out
    num = [c for c in NSL_FEATURES if c not in NSL_CATEGORICAL]
    return Dataset("NSL-KDD", "nsl", Xtr, ytr, ntr, Xte, yte, nte, list(NSL_CATEGORICAL), num, notes=notes)


# ---------------------------------------------------------------- CICIDS2017
def _cicids_group(label):
    l = str(label).lower()
    if "benign" in l:
        return "Normal"
    if "ddos" in l:
        return "DDoS"
    if "dos" in l:
        return "DoS"
    if "portscan" in l:
        return "PortScan"
    if "patator" in l:
        return "BruteForce"
    if "web attack" in l or "xss" in l or "sql" in l:
        return "WebAttack"
    if l.strip() == "bot":
        return "Bot"
    return "Other"


def _cap_per_class(df, col, cap, seed):
    parts = [g.sample(min(len(g), cap), random_state=seed) for _, g in df.groupby(col)]
    return pd.concat(parts, ignore_index=True)


def load_cicids(path, sample=200_000, seed=42):
    """CICIDS2017 'MachineLearningCSV' files. Class-capped sampling keeps training cheap."""
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"No CSV files in '{path}'.")
    cap_file = max(500, int(sample / len(files) / 2))
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.str.endswith(".1")]
        df = df.drop(columns=[c for c in df.columns if c.lower() in CICIDS_DROP])
        feats = [c for c in df.columns if c != "Label"]
        df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
        df["name"] = df["Label"].astype(str).str.replace("\ufffd", "-", regex=False).str.strip().str.lower()
        df["category"] = df["Label"].map(_cicids_group)
        frames.append(_cap_per_class(df, "name", cap_file, seed))
    df = pd.concat(frames, ignore_index=True)
    notes = [f"read {len(files)} file(s); class-capped sample of {len(df)} rows"]
    counts = df["category"].value_counts()
    keep = counts[counts >= 10].index
    if len(keep) < len(counts):
        notes.append(f"dropped classes with <10 rows: {sorted(set(counts.index) - set(keep))}")
    df = df[df["category"].isin(keep)]
    if len(df) > sample:
        df = _cap_per_class(df, "category", max(2000, sample // df["category"].nunique()), seed)
    feats = [c for c in df.columns if c not in ("Label", "name", "category")]
    Xtr, Xte, ytr, yte, ntr, nte = train_test_split(
        df[feats], df["category"], df["name"], test_size=0.3, random_state=seed, stratify=df["category"])
    notes.append("random stratified 70/30 split (near-duplicate flows can inflate scores: see README)")
    return Dataset("CICIDS2017", "cicids", Xtr.reset_index(drop=True), ytr.reset_index(drop=True),
                   ntr.reset_index(drop=True), Xte.reset_index(drop=True), yte.reset_index(drop=True),
                   nte.reset_index(drop=True), [], list(feats), notes=notes)


def load_dataset(kind, path, sample=200_000, binary=False, max_train=None, seed=42):
    ds = load_nsl(path, max_train, seed) if kind == "nsl" else load_cicids(path, sample, seed)
    if binary:
        ds.y_train = ds.y_train.where(ds.y_train == "Normal", "Attack")
        ds.y_test = ds.y_test.where(ds.y_test == "Normal", "Attack")
        ds.notes.append("binary mode: Normal vs Attack")
    known = set(ds.y_train.unique())
    mask = ds.y_test.isin(known)
    if not mask.all():  # test classes never seen in training cannot be scored by a classifier
        ds.notes.append(f"removed {int((~mask).sum())} test rows whose class is absent from training")
        ds.X_test, ds.y_test, ds.names_test = ds.X_test[mask], ds.y_test[mask], ds.names_test[mask]
    return ds
