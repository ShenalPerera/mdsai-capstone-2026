"""
Single source of truth for dataset splits.

Loads train.json and val.json and constructs three partitions from
annotation RECORDS, never from directory listings:

- train: all 23,431 train.json records.
- val / test: val.json's 7,750 records split 50/50 under a fixed seed.
  The official test.json split is withheld by the dataset authors and is
  not used anywhere (see docs/decisions.md, 2026-09-25).

Every other module that needs a split must call get_splits() rather than
re-deriving one, so there is exactly one definition of what "val" and
"test" mean in this project.
"""
import json
import random
from pathlib import Path

SEED = 42
FLAW_CODES = ["BLR", "BRT", "DRK", "FRM", "NON", "OBS", "OTH", "ROT"]
MODELLED_FLAWS = ["BLR", "BRT", "DRK", "FRM", "OBS", "ROT"]


def _load_records(data_dir, name):
    path = Path(data_dir) / "annotations" / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_splits(data_dir):
    """Return {"train": [...], "val": [...], "test": [...]}, each a list of
    annotation records (dicts with "image", "flaws", "unrecognizable").

    val/test are deterministic: val.json's records are sorted by image name
    (so result does not depend on on-disk JSON key order), shuffled with a
    seeded RNG, and split in half. Calling this twice on the same data
    always returns the same three lists in the same order.
    """
    train_records = _load_records(data_dir, "train")
    val_records = _load_records(data_dir, "val")

    val_sorted = sorted(val_records, key=lambda r: r["image"])
    rng = random.Random(SEED)
    rng.shuffle(val_sorted)
    half = len(val_sorted) // 2

    return {
        "train": train_records,
        "val": val_sorted[:half],
        "test": val_sorted[half:],
    }


def image_subdir(image_name):
    """Physical images/<subdir>/ an image lives under, derived from its own
    filename rather than from which logical partition it was assigned to.
    This matters because our "test" partition is a held-out half of
    val.json - its images physically live under images/val/, not
    images/test/ (the official, unused test archive)."""
    if "_train_" in image_name:
        return "train"
    if "_val_" in image_name:
        return "val"
    if "_test_" in image_name:
        return "test"
    raise ValueError(f"Cannot determine image subdirectory for {image_name!r}")


def flaw_vector(record, codes=MODELLED_FLAWS, threshold=2):
    """Binary multi-label target vector for the modelled flaw codes."""
    flaws = record.get("flaws", {})
    return [1 if flaws.get(code, 0) >= threshold else 0 for code in codes]


def raw_vote_vector(record, codes=MODELLED_FLAWS):
    """Raw 0-5 vote counts for the modelled flaw codes, for ordinal and
    threshold-sensitivity work."""
    flaws = record.get("flaws", {})
    return [flaws.get(code, 0) for code in codes]


def recognisability_target(record, threshold=2):
    return 1 if record.get("unrecognizable", 0) >= threshold else 0
