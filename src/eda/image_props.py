"""
Image-property analyses (Stage B): native resolution, aspect ratio, file
size and EXIF orientation of the ORIGINAL images, read straight from the
train/val zip archives without extracting them.

Only headers are read (PIL opens lazily, so pixels are never decoded), and
width/height are the raw stored pixel dimensions: no EXIF orientation is
applied, matching how annotators saw the images and how every other part
of the pipeline reads them (see docs/decisions.md, no auto-rotation).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.splits import image_subdir

ORIENTATION_TAG = 0x0112
NO_ORIENTATION = 0  # neither EXIF nor an Orientation tag present
PROPERTY_COLUMNS = ["image", "partition", "width", "height", "orientation", "file_bytes"]

# Pre-registered rule (docs/decisions.md, 2026-10-04): keep classical@512
# unless more than this share of originals would be upsampled to reach it.
UPSAMPLE_SHARE_LIMIT = 0.10
CLASSICAL_RESOLUTIONS = (224, 512)


def zip_path(data_dir, subdir):
    return Path(data_dir) / "images" / f"{subdir}.zip"


def zip_member_index(archive):
    """Map bare filename -> member path inside the archive, so lookups work
    whether the zip stores files at its root or under a train/ or val/
    folder."""
    return {
        Path(name).name: name
        for name in archive.namelist()
        if name.lower().endswith(".jpg")
    }


def read_header(archive, member):
    """(width, height, orientation, file_bytes) for one zip member."""
    file_bytes = archive.getinfo(member).file_size
    with archive.open(member) as fh, Image.open(fh) as im:
        width, height = im.size
        orientation = im.getexif().get(ORIENTATION_TAG, NO_ORIENTATION)
    return width, height, int(orientation), file_bytes


def load_properties(path):
    df = pd.read_csv(path)
    df["long_edge"] = df[["width", "height"]].max(axis=1)
    df["short_edge"] = df[["width", "height"]].min(axis=1)
    df["aspect"] = df["width"] / df["height"]  # >1 means landscape-shaped raw pixels
    df["landscape"] = df["width"] > df["height"]
    return df


def resolution_summary(props):
    """Long-edge percentiles and the share of originals each classical
    resolution would upsample."""
    long_edge = props["long_edge"]
    percentiles = long_edge.quantile([0.0, 0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 1.0])
    percentiles.index = [f"p{int(q * 100)}" for q in percentiles.index]
    below = {
        f"long edge < {r} %": round((long_edge < r).mean() * 100, 2)
        for r in CLASSICAL_RESOLUTIONS
    }
    return percentiles.astype(int), below


def resolution_verdict(props, resolution=512, limit=UPSAMPLE_SHARE_LIMIT):
    """Apply the pre-registered upsampling rule to one classical resolution.
    If the rule fails, suggest the largest multiple of 32 that at least
    (1 - limit) of originals reach natively."""
    share = (props["long_edge"] < resolution).mean()
    if share <= limit:
        return share, f"KEEP {resolution}: {share:.1%} of originals below it (limit {limit:.0%})"
    reachable = int(props["long_edge"].quantile(limit)) // 32 * 32
    return share, (
        f"LOWER {resolution}: {share:.1%} of originals below it (limit {limit:.0%}); "
        f"largest multiple of 32 reached by {1 - limit:.0%} of originals is {reachable}"
    )


def average_precision(y_true, score):
    """Average precision with tied scores handled as one threshold step
    (same definition as sklearn.metrics.average_precision_score)."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0:
        return float("nan")
    ap, prev_recall = 0.0, 0.0
    for threshold in np.unique(s)[::-1]:
        predicted = s >= threshold
        tp = (predicted & y).sum()
        precision = tp / predicted.sum()
        recall = tp / y.sum()
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def rot_shortcut_table(props, votes, threshold=2):
    """ROT>=threshold rate by EXIF orientation x raw pixel shape.

    props: load_properties() output. votes: DataFrame indexed by image with
    a ROT column of raw vote counts (src.eda.labels.vote_matrix)."""
    df = props.set_index("image").join(votes[["ROT"]], how="inner")
    df["rot_pos"] = df["ROT"] >= threshold
    df["shape"] = np.where(df["landscape"], "landscape", "portrait/square")
    table = df.groupby(["orientation", "shape"])["rot_pos"].agg(n="size", rot_rate="mean")
    table["rot_rate"] = (table["rot_rate"] * 100).round(1)
    return table


def rot_shortcut_scores(props, votes, threshold=2):
    """How much ROT can be predicted from geometry alone, no image content.

    Each row scores one 0/1 cue as a ROT predictor: average precision (to
    compare with the ROT prevalence, which is the AP of a random guess) and
    the ROT rate when the cue is on vs off. Computed over all images and
    within EXIF orientation 1 only, where the shape cue cannot be standing
    in for the orientation tag."""
    df = props.set_index("image").join(votes[["ROT"]], how="inner")
    rot = df["ROT"] >= threshold
    cues = {
        "landscape-shaped pixels (all images)": (df["landscape"], rot),
        "EXIF orientation 6 (all images)": (df["orientation"] == 6, rot),
    }
    tag1 = df["orientation"] == 1
    cues["landscape-shaped pixels (orientation 1 only)"] = (df.loc[tag1, "landscape"], rot[tag1])

    rows = {}
    for name, (cue, target) in cues.items():
        cue = cue.astype(bool)
        rows[name] = {
            "n": len(cue),
            "cue on %": round(cue.mean() * 100, 1),
            "ROT prevalence %": round(target.mean() * 100, 1),
            "AP": round(average_precision(target, cue), 3),
            "ROT % | cue on": round(target[cue].mean() * 100, 1) if cue.any() else float("nan"),
            "ROT % | cue off": round(target[~cue].mean() * 100, 1) if (~cue).any() else float("nan"),
        }
    return pd.DataFrame(rows).T


def shape_by_orientation(props):
    """Cross-tab of EXIF orientation vs raw pixel shape (counts)."""
    return pd.crosstab(props["orientation"], props["landscape"].map(
        {True: "landscape", False: "portrait/square"}))


def records_by_archive(records):
    """Group records by the zip archive their image physically lives in."""
    groups = {}
    for record in records:
        groups.setdefault(image_subdir(record["image"]), []).append(record)
    return groups


def open_archive(data_dir, subdir):
    return zipfile.ZipFile(zip_path(data_dir, subdir))
