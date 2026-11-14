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

from src.data.splits import MODELLED_FLAWS, SEED

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


def bootstrap_ap(targets, scores, n_resamples=1000, seed=SEED, flaws=MODELLED_FLAWS):
    """AP on bootstrap resamples of the images, for several score sets at once.

    targets: 0/1 DataFrame; scores: {name: DataFrame of probabilities, same
    index and columns}. Every score set is evaluated on the SAME resampled
    images in each draw, so differences between two sets are paired.
    Returns {name: DataFrame, one row per resample, one column per output plus
    MACRO (mean AP over `flaws`)}. The spread reflects which val images happen
    to be in the partition, not training randomness (that is the seed spread)."""
    rng = np.random.default_rng(seed)
    y = targets.to_numpy()
    arrays = {name: s.loc[targets.index, targets.columns].to_numpy() for name, s in scores.items()}
    flaw_cols = [targets.columns.get_loc(f) for f in flaws]
    draws = {name: [] for name in scores}
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        for name, s in arrays.items():
            aps = [average_precision(yb[:, j], s[idx, j]) for j in range(y.shape[1])]
            draws[name].append(aps + [np.nanmean([aps[j] for j in flaw_cols])])
    columns = list(targets.columns) + [MACRO]
    return {name: pd.DataFrame(d, columns=columns) for name, d in draws.items()}


def percentile_interval(draws, level=0.95):
    """Percentile interval per column of a bootstrap draws DataFrame:
    DataFrame with columns low, high (one row per output)."""
    tail = (1 - level) / 2 * 100
    return pd.DataFrame({"low": np.nanpercentile(draws, tail, axis=0),
                         "high": np.nanpercentile(draws, 100 - tail, axis=0)}, index=draws.columns)


def ap_with_interval(targets, scores, n_resamples=1000, seed=SEED, level=0.95):
    """AP and its bootstrap percentile interval per output (plus MACRO), for
    each score set: {name: DataFrame with columns AP, low, high}."""
    draws = bootstrap_ap(targets, scores, n_resamples, seed)
    tables = {}
    for name, s in scores.items():
        table = percentile_interval(draws[name], level)
        table.insert(0, "AP", evaluate(targets, s).loc[table.index, "AP"].astype(float))
        tables[name] = table
    return tables


def paired_difference(targets, scores_a, scores_b, n_resamples=1000, seed=SEED, level=0.95):
    """AP(a) - AP(b) per output (plus MACRO) with a paired bootstrap interval:
    both score sets are evaluated on the same resampled images. An interval
    that excludes 0 means the difference is larger than the variation due to
    which images are in the val partition (training randomness is separate)."""
    draws = bootstrap_ap(targets, {"a": scores_a, "b": scores_b}, n_resamples, seed)
    table = percentile_interval(draws["a"] - draws["b"], level)
    point = evaluate(targets, scores_a)["AP"].astype(float) - evaluate(targets, scores_b)["AP"].astype(float)
    table.insert(0, "difference", point.loc[table.index])
    table["excludes 0"] = (table["low"] > 0) | (table["high"] < 0)
    return table
