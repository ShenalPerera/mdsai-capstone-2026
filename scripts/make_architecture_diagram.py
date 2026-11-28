#!/usr/bin/env python3
"""
Draw the system architecture diagram (data -> splits -> preprocessing ->
three arms -> evaluation -> demo app) used in the README.

Usage: python scripts/make_architecture_diagram.py [--out reports/figures/architecture.png]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
COLOURS = {
    "data": "#dbe9f6", "prep": "#e8e8e8", "classical": "#fde2c8", "cnn": "#d5ecd4",
    "hybrid": "#e6dcf2", "eval": "#fff4c2", "app": "#f8d7da",
}
EDGE = "#444444"


def box(ax, x, y, w, h, title, body, colour):
    """Rounded box with a bold title line and smaller body text; returns its anchors."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=COLOURS[colour], edgecolor=EDGE, linewidth=1.2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.58, body, ha="center", va="top", fontsize=8.6, linespacing=1.35)
    return {"top": (x + w / 2, y + h), "bottom": (x + w / 2, y), "left": (x, y + h / 2),
            "right": (x + w, y + h / 2)}


def arrow(ax, start, end, label=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 color=EDGE, linewidth=1.2, shrinkA=2, shrinkB=2))
    if label:
        ax.text((start[0] + end[0]) / 2 + 0.08, (start[1] + end[1]) / 2, label, fontsize=8,
                style="italic", va="center")


def draw(out):
    fig, ax = plt.subplots(figsize=(14, 10.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    data = box(ax, 0.3, 9.0, 4.6, 1.3, "Data: VizWiz-QualityIssues",
               "photos taken by blind and low-vision users\n"
               "train.json 23,431  |  val.json 7,750 (test labels withheld)", "data")
    splits = box(ax, 5.8, 9.0, 7.9, 1.3, "Splits (src/data/splits.py, seed 42)",
                 "train 23,431  |  val 3,875 (model selection)  |  test 3,875 (final run only)\n"
                 "labels: flaw present at >= 2 of 5 votes; BLR BRT DRK FRM OBS ROT + UNREC",
                 "data")
    arrow(ax, data["right"], splits["left"])

    feats = box(ax, 0.3, 6.6, 5.6, 1.6, "Classical features (src/features/classical.py)",
                "original image, decoded without EXIF rotation\n"
                "resized to 224 px long edge (aspect kept)\n"
                "10 features: sharpness, exposure, noise, border edges", "prep")
    cache = box(ax, 8.1, 6.6, 5.6, 1.6, "Image cache (scripts/prepare_cache.py)",
                "original image, decoded without EXIF rotation\n"
                "letterboxed to 224 x 224 with mean-grey padding\n"
                "never stretched or cropped (framing is a label)", "prep")
    arrow(ax, (7.2, 9.0), feats["top"])
    arrow(ax, (11.0, 9.0), cache["top"])

    arm1 = box(ax, 0.3, 3.9, 4.0, 1.9, "Arm 1: Classical",
               "gradient boosting\none model per output\n(no deep learning)\n\n"
               "test macro AP 0.482", "classical")
    arm3 = box(ax, 5.0, 3.9, 4.0, 1.9, "Arm 3: Hybrid",
               "CNN + the 10 classical\nfeatures at the heads\ntrained end to end\n\n"
               "test macro AP 0.688", "hybrid")
    arm2 = box(ax, 9.7, 3.9, 4.0, 1.9, "Arm 2: CNN",
               "EfficientNet-B0 (ImageNet)\nfine-tuned, shared encoder\n"
               "heads: 6 flaws + UNREC\n\ntest macro AP 0.686", "cnn")
    arrow(ax, (2.3, 6.6), (2.3, 5.8))
    arrow(ax, (4.6, 6.6), (6.2, 5.8))
    arrow(ax, (9.4, 6.6), (7.8, 5.8))
    arrow(ax, (11.7, 6.6), arm2["top"])

    evaluation = box(ax, 0.3, 1.5, 9.4, 1.7, "Evaluation (src/models/metrics.py)",
                     "development on val: AP per flaw vs base rate, F1 at 0.5, bootstrap intervals,\n"
                     "ablations, comparison with a single human annotator\n"
                     "then the test partition, scored once (reports/results_test.md)", "eval")
    for anchor in (arm1, arm3):
        arrow(ax, anchor["bottom"], (anchor["bottom"][0], 3.2))
    arrow(ax, (10.8, 3.9), (8.9, 3.2))

    app = box(ax, 10.2, 1.5, 3.5, 1.7, "Demo app (Gradio)",
              "upload a photo -> fine-tuned CNN\n-> verdict + one fixed retake\n"
              "instruction (src/app/demo.py)", "app")
    arrow(ax, (12.4, 3.9), app["top"], "seed-42 weights")

    ax.text(0.3, 0.95, "Labels refer to the photo as stored: no step rotates images by their EXIF "
            "orientation tag.", fontsize=8.6, style="italic", color="#333333")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[diagram] wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO_DIR / "reports" / "figures" / "architecture.png")
    draw(parser.parse_args().out)


if __name__ == "__main__":
    main()
