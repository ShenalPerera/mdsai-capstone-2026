#!/usr/bin/env python3
"""
Verify Stage A's exit condition for src/data/splits.py:
- splits reproduce identically across two independent calls
- partition sizes and per-flaw positive counts
- zero image overlap between any two partitions

Usage: python scripts/report_splits.py DATA_DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import MODELLED_FLAWS, flaw_vector, get_splits  # noqa: E402

PARTITIONS = ("train", "val", "test")
report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/report_splits.py DATA_DIR", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"DATA_DIR does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    splits_a = get_splits(data_dir)
    splits_b = get_splits(data_dir)

    log("# Split Report")
    log(f"\nDATA_DIR: `{data_dir}`")

    log("\n## Reproducibility check")
    reproducible = all(
        [r["image"] for r in splits_a[name]] == [r["image"] for r in splits_b[name]]
        for name in PARTITIONS
    )
    log(f"- two independent calls to get_splits() produce identical ordering: {reproducible}")

    log("\n## Partition sizes")
    log("\n| partition | images |")
    log("|-----------|--------|")
    for name in PARTITIONS:
        log(f"| {name} | {len(splits_a[name])} |")

    log("\n## Cross-partition overlap")
    name_sets = {name: {r["image"] for r in splits_a[name]} for name in PARTITIONS}
    clean = True
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            overlap = name_sets[a] & name_sets[b]
            if overlap:
                clean = False
                log(f"- {a} and {b} overlap: {len(overlap)} images (e.g. {sorted(overlap)[:5]})")
    if clean:
        log("- zero image overlap between any two partitions")

    log("\n## Per-flaw positive counts (>=2 votes, 6 modelled flaws)")
    log("\n| partition | " + " | ".join(MODELLED_FLAWS) + " |")
    log("|-----------|" + "|".join(["---"] * len(MODELLED_FLAWS)) + "|")
    for name in PARTITIONS:
        counts = [0] * len(MODELLED_FLAWS)
        for record in splits_a[name]:
            vec = flaw_vector(record)
            counts = [c + v for c, v in zip(counts, vec)]
        log(f"| {name} | " + " | ".join(str(c) for c in counts) + " |")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "splits_report.md"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
