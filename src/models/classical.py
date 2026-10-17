"""
Arm 1, classical baseline: one binary classifier per output on the ten
hand-crafted features from src/features/classical.py.

Outputs are the six modelled flaws (>=2 votes) plus UNREC (unrecognizable
>=2, the dataset authors' "poor image" definition). Two model kinds:

- "lr": log1p on the heavy-tailed features, standardise, logistic
  regression. The sharpness, noise and edge features span orders of
  magnitude; without the log a handful of extreme images dominate the fit.
- "gb": histogram gradient boosting on the raw features (trees are
  invariant to monotone transforms, so no log or scaling). Library
  defaults, no tuning (docs/decisions.md).

Both use class_weight="balanced" for the label imbalance. Every fitted
component, scaler included, sees the training partition only.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data.splits import MODELLED_FLAWS, SEED, flaw_vector, recognisability_target
from src.features.classical import FEATURE_NAMES

UNREC = "UNREC"
OUTPUTS = MODELLED_FLAWS + [UNREC]
KINDS = ("lr", "gb")
LOG_FEATURES = [
    "laplacian_var",
    "tenengrad",
    "hf_residual_energy",
    "border_edge_energy",
    "center_edge_energy",
]


def load_features(data_dir, resolution):
    """classical_features_<resolution>.csv indexed by image name: the ten
    features plus the native_width/native_height/upsampled metadata."""
    path = Path(data_dir) / f"classical_features_{resolution}.csv"
    return pd.read_csv(path, index_col="image")


def build_targets(records):
    """0/1 targets, one row per image, one column per output. Thresholds come
    from src/data/splits.py so labels are defined in one place."""
    rows = [flaw_vector(r) + [recognisability_target(r)] for r in records]
    return pd.DataFrame(rows, index=[r["image"] for r in records], columns=OUTPUTS)


def make_model(kind):
    if kind == "lr":
        log = ColumnTransformer(
            [("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), LOG_FEATURES)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        return Pipeline([
            ("log", log),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ])
    if kind == "gb":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    raise ValueError(f"unknown model kind {kind!r}, expected one of {KINDS}")


def train_models(X, Y, kind):
    """Fit one model per output column of Y on the FEATURE_NAMES columns of X.
    Returns {output: fitted model}."""
    X = X[FEATURE_NAMES]
    models = {}
    for output in Y.columns:
        start = time.time()
        model = make_model(kind)
        model.fit(X, Y[output])
        models[output] = model
        print(f"  [{kind}] {output:<5} {int(Y[output].sum()):>6} positives  "
              f"fitted in {time.time() - start:5.1f}s")
    return models


def predict_scores(models, X):
    """Positive-class probability per image (rows) and output (columns)."""
    X = X[FEATURE_NAMES]
    return pd.DataFrame({out: m.predict_proba(X)[:, 1] for out, m in models.items()}, index=X.index)


def lr_coefficients(models):
    """Logistic regression weights on standardised (log1p where applied)
    features, features x outputs. Comparable within an output: sign is the
    direction, magnitude the strength of the feature's pull."""
    coefs = {}
    for out, model in models.items():
        names = model.named_steps["log"].get_feature_names_out()
        coefs[out] = pd.Series(model.named_steps["clf"].coef_[0], index=names)
    return pd.DataFrame(coefs).loc[FEATURE_NAMES]
