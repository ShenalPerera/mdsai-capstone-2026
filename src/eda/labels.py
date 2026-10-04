"""
Label-only exploratory analyses (Stage B). Every function takes annotation
records as produced by src.data.splits.get_splits() and needs no images.

Votes are counts out of 5 crowdworkers per image, so per-flaw agreement can
be computed from the counts alone, without per-worker labels.
"""
import numpy as np
import pandas as pd

from src.data.splits import FLAW_CODES, MODELLED_FLAWS

N_RATERS = 5
POSITIVE_THRESHOLD = 2
UNREC = "UNREC"  # recognisability treated as a ninth label in the tables


def vote_matrix(records, codes=FLAW_CODES, include_unrec=True):
    """DataFrame of raw 0-5 vote counts, one row per image, one column per
    code (plus UNREC for the unrecognizable votes)."""
    rows = []
    for record in records:
        row = {code: int(record["flaws"].get(code, 0)) for code in codes}
        if include_unrec:
            row[UNREC] = int(record.get("unrecognizable", 0))
        rows.append(row)
    return pd.DataFrame(rows, index=[r["image"] for r in records])


def prevalence_table(splits, partitions=("train", "val")):
    """% of images positive at >=1 and >=2 votes, per code and partition."""
    columns = {}
    for name in partitions:
        votes = vote_matrix(splits[name])
        for threshold in (1, 2):
            columns[f"{name} >={threshold}"] = (votes >= threshold).mean() * 100
    return pd.DataFrame(columns).round(1)


def cooccurrence(records, codes=MODELLED_FLAWS, threshold=POSITIVE_THRESHOLD):
    """Conditional co-occurrence: cell [A, B] is P(B positive | A positive),
    in %. The diagonal is 100 by construction. Also returns the raw
    joint-count matrix so small denominators can be checked."""
    positive = (vote_matrix(records, codes, include_unrec=False) >= threshold).astype(int)
    joint = positive.T @ positive
    conditional = joint.div(np.diag(joint), axis=0) * 100
    return conditional.round(1), joint


def lift_matrix(records, codes=MODELLED_FLAWS, threshold=POSITIVE_THRESHOLD):
    """P(B | A) / P(B): above 1 means A and B occur together more often than
    their prevalences alone would predict."""
    positive = vote_matrix(records, codes, include_unrec=False) >= threshold
    conditional, _ = cooccurrence(records, codes, threshold)
    return (conditional / 100).div(positive.mean(), axis=1).round(2)


def vote_distribution(records, codes=MODELLED_FLAWS):
    """Rows: code (6 modelled flaws + UNREC). Columns: % of images with
    0..5 votes."""
    votes = vote_matrix(records, codes)
    table = pd.DataFrame(
        {v: (votes == v).mean() * 100 for v in range(N_RATERS + 1)}
    )
    return table.round(1)


def fleiss_kappa_binary(positive_votes, n_raters=N_RATERS):
    """Fleiss' kappa for a yes/no judgement made by n_raters per item,
    computed from the per-item count of "yes" votes."""
    k = np.asarray(positive_votes, dtype=float)
    n = n_raters
    per_item = (k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))
    p_observed = per_item.mean()
    p_yes = k.sum() / (len(k) * n)
    p_expected = p_yes ** 2 + (1 - p_yes) ** 2
    if p_expected == 1:
        return float("nan")
    return (p_observed - p_expected) / (1 - p_expected)


def agreement_table(records, codes=MODELLED_FLAWS):
    """Per code: Fleiss' kappa, % of images where all 5 raters agree (0 or 5
    votes), and % that sit at the 2/3 boundary - the images whose >=2 label
    flips on a single rater."""
    votes = vote_matrix(records, codes)
    rows = {}
    for code in votes.columns:
        v = votes[code]
        rows[code] = {
            "fleiss_kappa": fleiss_kappa_binary(v),
            "unanimous %": v.isin([0, N_RATERS]).mean() * 100,
            "borderline 2-3 %": v.isin([2, 3]).mean() * 100,
            "positive >=2 %": (v >= POSITIVE_THRESHOLD).mean() * 100,
        }
    return pd.DataFrame(rows).T.round(3)


def unrecognisable_by_flaw_count(records, codes=MODELLED_FLAWS,
                                 threshold=POSITIVE_THRESHOLD):
    """Images grouped by how many modelled flaws they carry at >=threshold:
    group size and % unrecognisable (>=2 votes)."""
    votes = vote_matrix(records, codes)
    n_flaws = (votes[codes] >= threshold).sum(axis=1)
    unrec = votes[UNREC] >= POSITIVE_THRESHOLD
    table = pd.DataFrame({"n_flaws": n_flaws, "unrec": unrec})
    out = table.groupby("n_flaws")["unrec"].agg(n="size", unrecognisable_pct="mean")
    out["unrecognisable_pct"] = (out["unrecognisable_pct"] * 100).round(1)
    return out


def unrecognisable_by_flaw(records, codes=MODELLED_FLAWS,
                           threshold=POSITIVE_THRESHOLD):
    """Per flaw: % unrecognisable among images with vs without that flaw,
    and the ratio between the two."""
    votes = vote_matrix(records, codes)
    unrec = votes[UNREC] >= POSITIVE_THRESHOLD
    rows = {}
    for code in codes:
        has = votes[code] >= threshold
        with_pct = unrec[has].mean() * 100
        without_pct = unrec[~has].mean() * 100
        rows[code] = {
            "n with flaw": int(has.sum()),
            "unrec % | flaw": round(with_pct, 1),
            "unrec % | no flaw": round(without_pct, 1),
            "ratio": round(with_pct / without_pct, 2) if without_pct else float("nan"),
        }
    return pd.DataFrame(rows).T
