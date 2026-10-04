"""
Figures and the reports/eda.md writer for Stage B. The notebook calls these;
it holds no analysis logic of its own.

reports/eda.md is regenerated from the data on every run, except for the
hand-written findings block between FINDINGS_START and FINDINGS_END, which
is carried over unchanged so that rerunning the notebook never erases the
interpretation.
"""
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FINDINGS_START = "<!-- findings:start -->"
FINDINGS_END = "<!-- findings:end -->"
FINDINGS_PLACEHOLDER = "_Findings not yet written._"


def df_to_markdown(df, index=True):
    """Minimal DataFrame -> GitHub markdown table (avoids a tabulate
    dependency)."""
    frame = df.reset_index() if index else df
    header = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return "nan" if np.isnan(value) else f"{value:g}"
    return str(value)


def environment_lines():
    """Runtime description recorded in every report (Conventions)."""
    lines = [f"- Python {platform.python_version()} on {platform.platform()}"]
    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
        lines.append(f"- RAM: {kb / 1024 ** 2:.1f} GB")
    except (OSError, ValueError, IndexError):
        lines.append("- RAM: unknown")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        gpu = "none (torch not installed)"
    lines.append(f"- GPU: {gpu} (not needed: Stage B reads labels and image headers only)")
    lines.append("- Seed: 42 (only randomness is the val/test split in src/data/splits.py)")
    return lines


def existing_findings(path):
    path = Path(path)
    if not path.exists():
        return FINDINGS_PLACEHOLDER
    text = path.read_text(encoding="utf-8")
    if FINDINGS_START in text and FINDINGS_END in text:
        return text.split(FINDINGS_START, 1)[1].split(FINDINGS_END, 1)[0].strip()
    return FINDINGS_PLACEHOLDER


def write_report(path, sections):
    """sections: list of (heading, body) pairs, body already markdown."""
    path = Path(path)
    findings = existing_findings(path)
    parts = ["# Stage B: Exploratory Data Analysis", "",
             "Partitions: train and val only. The test partition is not read "
             "during development.", "",
             "## Environment", "", *environment_lines(), "",
             "## Findings and impact on the modelling plan", "",
             FINDINGS_START, findings, FINDINGS_END, ""]
    for heading, body in sections:
        parts += [f"## {heading}", "", body.strip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_prevalence(prevalence):
    fig, ax = plt.subplots(figsize=(8, 4))
    prevalence.plot.bar(ax=ax)
    ax.set_ylabel("% of images")
    ax.set_title("Label prevalence at >=1 and >=2 votes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_matrix(matrix, title, fmt="{:.0f}", cmap="Blues"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix.values, cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # white text on the dark end of the colour scale so it stays legible
            shade = im.norm(matrix.iat[i, j])
            color = "white" if (shade > 0.6 or (cmap == "RdBu_r" and shade < 0.15)) else "black"
            ax.text(j, i, fmt.format(matrix.iat[i, j]), ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xlabel("B")
    ax.set_ylabel("A")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_vote_distribution(distribution):
    fig, ax = plt.subplots(figsize=(8, 4))
    distribution.plot.bar(ax=ax, stacked=True, colormap="viridis")
    ax.set_ylabel("% of images")
    ax.set_title("Vote-count distribution (0-5 of 5 raters)")
    ax.legend(title="votes", fontsize=8)
    fig.tight_layout()
    return fig


def plot_long_edge(props, resolutions=(224, 512)):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(props["long_edge"], bins=60)
    for r in resolutions:
        ax.axvline(r, color="red", linestyle="--")
        ax.text(r, ax.get_ylim()[1] * 0.9, f" {r}", color="red")
    ax.set_xlabel("native long edge (px)")
    ax.set_ylabel("images")
    ax.set_title("Native resolution of originals")
    fig.tight_layout()
    return fig


def plot_aspect(props):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log2(props["aspect"]), bins=60)
    ax.set_xlabel("log2(width / height)  (<0 portrait, >0 landscape)")
    ax.set_ylabel("images")
    ax.set_title("Raw-pixel aspect ratio")
    fig.tight_layout()
    return fig


def save(fig, figures_dir, name):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{name}.png"
    fig.savefig(out, dpi=110)
    return out


def file_size_summary(props):
    kb = props["file_bytes"] / 1024
    return pd.DataFrame({"file size (KB)": kb.describe(percentiles=[0.01, 0.05, 0.5, 0.95])}).round(1)
