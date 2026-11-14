"""
Human performance ceiling (Stage G): how well one annotator agrees with the
others, so model scores can be read against human consistency instead of a
perfect 1.0.

The annotations hold only the number of "yes" votes out of five per image,
not which annotator gave them, so a held-out annotator is simulated. In each
repeat, for every image and output, one of the five annotators is drawn at
random: with k yes-votes they said yes with probability k / 5. Their yes/no is
the human prediction. The reference label comes from the other four: positive
when at least REFERENCE_VOTES of them said yes, mirroring the >=2-vote rule.
Models are scored against the same references in the same repeat, so humans
and models are compared on identical targets.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

from src.data.splits import SEED
from src.eda.labels import N_RATERS
from src.models.metrics import THRESHOLD, average_precision

REFERENCE_VOTES = 2  # yes-votes needed among the four remaining annotators


def simulate_split(counts, rng):
    """One held-out-annotator draw. counts: DataFrame of yes-votes (0-5).
    Returns (human, reference), 0/1 DataFrames of the same shape."""
    k = counts.to_numpy()
    human = rng.random(k.shape) < k / N_RATERS
    reference = (k - human) >= REFERENCE_VOTES
    return (pd.DataFrame(human.astype(int), index=counts.index, columns=counts.columns),
            pd.DataFrame(reference.astype(int), index=counts.index, columns=counts.columns))


def best_f1(y_true, score):
    """Highest F1 over all thresholds. Optimistic: the threshold is chosen on
    the labels it is scored against."""
    precision, recall, _ = precision_recall_curve(y_true, score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(f1.max())


def _prf(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return p, r, f1


def human_ceiling(counts, scores, n_repeats=200, seed=SEED, threshold=THRESHOLD):
    """Run the held-out-annotator simulation.

    counts: DataFrame of 0-5 yes-votes, one column per output.
    scores: {model name: DataFrame of probabilities, same index and columns}.
    Returns a long DataFrame, one row per (repeat, output, who): reference base
    rate, precision, recall and F1 (models at `threshold`), plus best-threshold
    F1 and AP for models."""
    rng = np.random.default_rng(seed)
    arrays = {name: s.loc[counts.index, counts.columns].to_numpy() for name, s in scores.items()}
    rows = []
    for repeat in range(n_repeats):
        human, reference = simulate_split(counts, rng)
        for j, output in enumerate(counts.columns):
            y = reference[output].to_numpy()
            base = {"repeat": repeat, "output": output, "reference base rate": y.mean()}
            p, r, f1 = _prf(y, human[output].to_numpy())
            rows.append({**base, "who": "human", "precision": p, "recall": r, "F1": f1})
            for name, s in arrays.items():
                p, r, f1 = _prf(y, s[:, j] >= threshold)
                rows.append({**base, "who": name, "precision": p, "recall": r, "F1": f1,
                             "best F1": best_f1(y, s[:, j]), "AP": average_precision(y, s[:, j])})
    return pd.DataFrame(rows)


def summarise(runs, statistic="F1"):
    """Mean over repeats of one statistic, outputs x who, with the 2.5-97.5
    percentile range across repeats as a second frame."""
    grouped = runs.groupby(["output", "who"], sort=False)[statistic]
    mean = grouped.mean().unstack("who")
    low, high = grouped.quantile(0.025).unstack("who"), grouped.quantile(0.975).unstack("who")
    return mean, low, high
