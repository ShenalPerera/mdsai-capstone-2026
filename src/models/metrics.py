"""
Evaluation shared by all three arms (classical, CNN, hybrid), so every cell
of the comparison table is computed by the same code.

- AP (average precision) is the primary, threshold-free metric. A random
  scorer's AP equals the base rate, so AP is always read against it.
- Precision, recall and F1 are taken at a fixed threshold of 0.5 for every
  arm and output: models are class-weighted, so 0.5 is a meaningful cut,
  and tuning a threshold on val then reporting val F1 would overstate it
  (docs/decisions.md).
- Macro rows average over the six flaws only; UNREC is a separate head.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from src.data.splits import MODELLED_FLAWS

THRESHOLD = 0.5
MACRO = "macro (6 flaws)"


def average_precision(y_true, score):
    """AP, or NaN when the subset has no positives (AP is undefined there)."""
    y_true = np.asarray(y_true)
    return average_precision_score(y_true, score) if y_true.sum() else np.nan


def normalised_gain(ap, base_rate):
    """(AP - base rate) / (1 - base rate): the share of the distance from a
    random scorer (0) to a perfect one (1). Makes AP comparable across
    outputs with different base rates."""
    return (ap - base_rate) / (1 - base_rate)


def evaluate(targets, scores, threshold=THRESHOLD, flaws=MODELLED_FLAWS):
    """Per-output metrics plus a macro row over `flaws`.
    targets: 0/1 DataFrame; scores: DataFrame of probabilities, same index
    and columns."""
    scores = scores.loc[targets.index]
    rows = {}
    for out in targets.columns:
        y = targets[out].to_numpy()
        s = scores[out].to_numpy()
        n_pos = int(y.sum())
        if n_pos == 0:
            rows[out] = {"n": len(y), "positives": 0, "base rate": 0.0, "AP": np.nan,
                         "precision": np.nan, "recall": np.nan, "F1": np.nan}
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            y, s >= threshold, average="binary", zero_division=0)
        rows[out] = {"n": len(y), "positives": n_pos, "base rate": y.mean(),
                     "AP": average_precision_score(y, s),
                     "precision": p, "recall": r, "F1": f1}
    table = pd.DataFrame(rows).T
    present = [f for f in flaws if f in table.index]
    table.loc[MACRO] = table.loc[present, ["base rate", "AP", "precision", "recall", "F1"]].mean()
    table[["n", "positives"]] = table[["n", "positives"]].astype("Int64")
    return table


def single_flaw_ap(targets, scores, flaws=MODELLED_FLAWS):
    """AP for each flaw on the images carrying no OTHER modelled flaw: the
    positives have that flaw alone, the negatives no flaw at all. Separates
    a flaw's own signal from co-occurring flaws (e.g. blur riding along with
    DRK/BRT)."""
    rows = {}
    for flaw in flaws:
        others = [f for f in flaws if f != flaw]
        mask = targets[others].sum(axis=1) == 0
        y = targets.loc[mask, flaw]
        rows[flaw] = {
            "n": int(mask.sum()),
            "positives (flaw alone)": int(y.sum()),
            "base rate": y.mean(),
            "AP": average_precision(y, scores.loc[y.index, flaw]),
        }
    return pd.DataFrame(rows).T
