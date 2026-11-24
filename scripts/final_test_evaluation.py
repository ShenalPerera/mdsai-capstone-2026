#!/usr/bin/env python3
"""
Final evaluation on the held-out test partition: run once, select nothing.

Scores the models of the comparison table on the test half of val.json
(3,875 images never used for any decision during development) and writes
reports/results_test.md and reports/results_test.csv:

  Classical GB @ 224 (entry) and @ 512 (deployment configuration): refit on
      the training partition with the Stage C settings. Before any test
      input is read, the refit models must reproduce the saved Stage C val
      predictions; otherwise the script stops.
  CNN fine-tuned and hybrid, seeds 42 (entries), 43, 44: the saved best.pt
      checkpoints, no training.

Metrics are those of the val analysis (src/models/metrics.py): per-output AP,
gain, precision/recall/F1 at 0.5, bootstrap intervals and paired differences,
seed mean and range, the ROT shape checks, and a val-vs-test table.

Run-once guard: the script refuses to run if the report already exists.

Inputs (built at this final run by scripts/prepare_test_inputs.py):
  --cache-dir     extracted 224px test cache (flat folder of JPEGs)
  --test-inputs   folder with classical_features_{224,512}.csv for the test
                  partition and image_properties.csv covering it

Usage:
  python -u scripts/final_test_evaluation.py DATA_DIR --cache-dir DIR --test-inputs DIR
      [--report reports/results_test.md] [--workers N]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
from src.data.dataset import QualityIssuesDataset  # noqa: E402
from src.data.splits import MODELLED_FLAWS, get_splits  # noqa: E402
from src.eda import image_props as ip  # noqa: E402
from src.eda import labels as lb  # noqa: E402
from src.eda import report as rp  # noqa: E402
from src.features.classical import FEATURE_NAMES  # noqa: E402
from src.models import classical as cm  # noqa: E402
from src.models import metrics as mt  # noqa: E402
from src.models.cnn import QualityNet, image_transform  # noqa: E402
from train_cnn import make_loader, predict  # noqa: E402

SEEDS = (42, 43, 44)
N_RESAMPLES = 1000
REPRODUCTION_TOLERANCE = 1e-6  # max |refit - saved| val probability
CLASSICAL, CLASSICAL_512, CNN, HYBRID = "Classical GB @ 224", "Classical GB @ 512", "CNN fine-tuned", "Hybrid"
FLAW_TYPE = {"BLR": "photometric", "DRK": "photometric", "BRT": "photometric",
             "FRM": "semantic", "OBS": "semantic", "ROT": "semantic", cm.UNREC: "recognisability"}
ROWS = cm.OUTPUTS + [mt.MACRO]


def suffix(seed):
    return "" if seed == 42 else f"_seed{seed}"


def read_predictions(path, index):
    s = pd.read_csv(path, index_col=0)  # first column: image name (header may be empty)
    return s.loc[index, cm.OUTPUTS]


def evaluate_with_gain(targets, s):
    table = mt.evaluate(targets, s)
    table.insert(table.columns.get_loc("AP") + 1, "gain", mt.normalised_gain(table["AP"], table["base rate"]))
    table.loc[mt.MACRO, "gain"] = np.nan
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--test-inputs", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=REPO_DIR / "reports" / "results_test.md")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    if args.report.exists():
        sys.exit(f"[test] {args.report} already exists: the test partition is evaluated once. Not running.")
    predictions_dir = args.data_dir / "predictions"
    splits = get_splits(args.data_dir)
    train_records, val_records, test_records = splits["train"], splits["val"], splits["test"]
    train_names = [r["image"] for r in train_records]
    val_index = pd.Index([r["image"] for r in val_records])
    test_index = pd.Index([r["image"] for r in test_records])
    sections = []

    # 1. Classical: refit on train, prove it is the Stage C model, before any test input is read
    train_targets = cm.build_targets(train_records)
    features_trainval = {res: cm.load_features(args.data_dir, res) for res in (224, 512)}
    classical_models, reproduction = {}, {}
    for res in (224, 512):
        print(f"[test] refitting classical GB @ {res} on train", flush=True)
        models = cm.train_models(features_trainval[res].loc[train_names], train_targets, "gb")
        refit_val = cm.predict_scores(models, features_trainval[res].loc[val_index])
        saved_val = read_predictions(predictions_dir / f"classical_gb_{res}_val.csv", val_index)
        diff = float(np.abs(refit_val.to_numpy() - saved_val.to_numpy()).max())
        reproduction[f"GB @ {res}"] = diff
        print(f"[test] GB @ {res}: max |refit - saved Stage C val score| = {diff:.2e}", flush=True)
        if diff > REPRODUCTION_TOLERANCE:
            sys.exit(f"[test] GB @ {res} does not reproduce the Stage C val predictions "
                     f"(max difference {diff:.2e} > {REPRODUCTION_TOLERANCE}); test not read. Stopping.")
        classical_models[res] = models

    # 2. Test inputs (first use of the test partition)
    targets = cm.build_targets(test_records)
    print(f"[test] test partition: {len(targets):,} images", flush=True)
    features_test = {res: cm.load_features(args.test_inputs, res) for res in (224, 512)}
    for res, f in features_test.items():
        missing = test_index.difference(f.index)
        if len(missing):
            sys.exit(f"[test] {len(missing)} test images missing from the {res}px test features")
    missing = [n for n in test_index if not (args.cache_dir / n).exists()]
    if missing:
        sys.exit(f"[test] {len(missing)} test images missing from {args.cache_dir}, e.g. {missing[:3]}")

    scores = {CLASSICAL: cm.predict_scores(classical_models[224], features_test[224].loc[test_index]),
              CLASSICAL_512: cm.predict_scores(classical_models[512], features_test[512].loc[test_index])}

    # 3. CNN and hybrid checkpoints
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    preprocessor = cm.feature_preprocessor().fit(features_trainval[224].loc[train_names, FEATURE_NAMES])
    extra = pd.DataFrame(preprocessor.transform(features_test[224].loc[test_index, FEATURE_NAMES]),
                         index=test_index, columns=FEATURE_NAMES)
    for arm, prefix in ((CNN, ""), (HYBRID, "hybrid_")):
        for seed in SEEDS:
            run_dir = args.data_dir / "cnn_runs" / f"{prefix}finetune{suffix(seed)}"
            hybrid = arm == HYBRID
            model = QualityNet(pretrained=False, n_extra=len(FEATURE_NAMES) if hybrid else 0)
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
            model.to(device)
            dataset = QualityIssuesDataset(test_records, args.cache_dir, image_transform(False), flat=True,
                                           extra=extra if hybrid else None)
            print(f"[test] scoring {arm}, seed {seed} ({run_dir.name})", flush=True)
            scores[arm, seed] = predict(model, make_loader(dataset, 64, False, args.workers), device, use_amp,
                                        label=f"{arm} seed {seed}").loc[test_index]
    scores[CNN], scores[HYBRID] = scores[CNN, 42], scores[HYBRID, 42]

    names = {CLASSICAL: "classical_gb_224", CLASSICAL_512: "classical_gb_512",
             **{(a, s): f"{'cnn' if a == CNN else 'hybrid'}_finetune{suffix(s)}" for a in (CNN, HYBRID) for s in SEEDS}}
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for key, stem in names.items():
        scores[key].to_csv(predictions_dir / f"{stem}_test.csv")
    print(f"[test] test scores saved to {predictions_dir}", flush=True)

    # 4. Metrics, as in the val analysis
    results = {key: evaluate_with_gain(targets, s) for key, s in scores.items()}
    positives = pd.DataFrame({"test positives": targets.sum(), "test base rate": targets.mean()})
    sections.append(("Inputs and checks", (
        f"Test partition: {len(targets):,} images (the held-out half of val.json, seed 42), never used for "
        "any decision during development. Classical GB refit on train with the Stage C settings; it "
        "reproduces the saved Stage C val predictions (max absolute difference: "
        + ", ".join(f"{k} {v:.1e}" for k, v in reproduction.items())
        + "). CNN and hybrid: saved best.pt checkpoints, no training.\n\n" + rp.df_to_markdown(positives.round(3)))))

    comparison = pd.DataFrame({"type": [FLAW_TYPE.get(o, "") for o in ROWS],
                               "base rate": results[CNN].loc[ROWS, "base rate"],
                               CLASSICAL: results[CLASSICAL].loc[ROWS, "AP"]}, index=ROWS)
    for arm in (CNN, HYBRID):
        aps = pd.concat([results[arm, s].loc[ROWS, "AP"] for s in SEEDS], axis=1).astype(float)
        comparison[f"{arm} (seed 42)"] = results[arm].loc[ROWS, "AP"]
        comparison[f"{arm} mean of 3 seeds"] = aps.mean(axis=1)
        comparison[f"{arm} seed range"] = [f"{a:.3f}-{b:.3f}" for a, b in zip(aps.min(axis=1), aps.max(axis=1))]
    comparison[f"{CLASSICAL_512} (deployment)"] = results[CLASSICAL_512].loc[ROWS, "AP"]
    gains = pd.DataFrame({n: results[n].loc[cm.OUTPUTS, "gain"] for n in (CLASSICAL, CNN, HYBRID, CLASSICAL_512)})
    by_type = gains.loc[MODELLED_FLAWS].groupby(pd.Series(FLAW_TYPE)).mean()
    f1 = pd.DataFrame({n: results[n].loc[ROWS, "F1"] for n in (CLASSICAL, CNN, HYBRID, CLASSICAL_512)})
    sections.append(("Comparison table (test AP)", (
        rp.df_to_markdown(comparison.round(3)) + "\n\nGain = (AP - base rate) / (1 - base rate):\n\n"
        + rp.df_to_markdown(gains.round(3)) + "\n\nMean gain by flaw type:\n\n" + rp.df_to_markdown(by_type.round(3))
        + "\n\nF1 at threshold 0.5:\n\n" + rp.df_to_markdown(f1.round(3)))))
    for name in (CLASSICAL, CNN, HYBRID):
        sections.append((f"Full table: {name} (test)", rp.df_to_markdown(results[name].round(3))))

    print("[test] bootstrap intervals", flush=True)
    intervals = mt.ap_with_interval(targets, {n: scores[n] for n in (CLASSICAL, CNN, HYBRID)}, N_RESAMPLES)
    ci = pd.DataFrame({n: [f"{r.AP:.3f} [{r.low:.3f}, {r.high:.3f}]" for r in t.itertuples()]
                       for n, t in intervals.items()}, index=intervals[CNN].index)
    differences = {"CNN - classical": mt.paired_difference(targets, scores[CNN], scores[CLASSICAL], N_RESAMPLES),
                   "hybrid - CNN": mt.paired_difference(targets, scores[HYBRID], scores[CNN], N_RESAMPLES)}
    sections.append(("Bootstrap intervals (test)", (
        f"95% percentile intervals, {N_RESAMPLES} resamples of the test images, seed 42, paired across models."
        "\n\n" + rp.df_to_markdown(ci) + "".join(f"\n\nPaired difference, {label}:\n\n" + rp.df_to_markdown(t.round(3))
                                                for label, t in differences.items()))))

    props = ip.load_properties(args.test_inputs / "image_properties.csv")
    test_props = props[props["image"].isin(test_index)]
    if len(test_props) != len(test_index):
        sys.exit(f"[test] image_properties.csv covers {len(test_props)} of {len(test_index)} test images")
    shape_ref = ip.rot_shortcut_scores(test_props, lb.vote_matrix(test_records))
    orientation = test_props.set_index("image").loc[test_index, "orientation"]
    rot_rows = {}
    for name in (CLASSICAL, CNN, HYBRID):
        row = {f"AP, all (base {targets['ROT'].mean():.3f})": mt.average_precision(targets["ROT"], scores[name]["ROT"])}
        for tag in (1, 6):
            idx = orientation.index[orientation == tag]
            y = targets.loc[idx, "ROT"]
            row[f"AP, orientation {tag} (n={len(idx)}, base {y.mean():.3f})"] = mt.average_precision(y, scores[name].loc[idx, "ROT"])
        rot_rows[name] = row
    sections.append(("ROT: content or shape? (test)", (
        "Geometry-only cues on test:\n\n" + rp.df_to_markdown(shape_ref) + "\n\nROT AP by EXIF orientation:\n\n"
        + rp.df_to_markdown(pd.DataFrame(rot_rows).T.round(3)))))

    val_targets = cm.build_targets(val_records)
    val_files = {CLASSICAL: "classical_gb_224_val.csv", CNN: "cnn_finetune_val.csv", HYBRID: "hybrid_finetune_val.csv",
                 CLASSICAL_512: "classical_gb_512_val.csv"}
    val_vs_test = {}
    for name, filename in val_files.items():
        val_ap = mt.evaluate(val_targets, read_predictions(predictions_dir / filename, val_index))["AP"]
        val_vs_test[f"{name} val"] = val_ap.loc[ROWS].astype(float)
        val_vs_test[f"{name} test"] = results[name].loc[ROWS, "AP"].astype(float)
        val_vs_test[f"{name} test - val"] = val_vs_test[f"{name} test"] - val_vs_test[f"{name} val"]
    val_vs_test = pd.DataFrame(val_vs_test)
    sections.append(("Val vs test (AP)", (
        "Val AP (the partition that selected epochs and entries) against test AP (never used).\n\n"
        + rp.df_to_markdown(val_vs_test.round(3)))))

    long = pd.concat({(k if isinstance(k, str) else f"{k[0]}, seed {k[1]}"): t for k, t in results.items()
                      if not (isinstance(k, str) and k in (CNN, HYBRID))}, names=["model", "output"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    long.reset_index().to_csv(args.report.with_suffix(".csv"), index=False, float_format="%.4f")
    environment = rp.environment_lines(
        gpu_note="inference only: saved checkpoints scored on the test partition",
        seed_note=f"bootstrap ({N_RESAMPLES} resamples) and gradient boosting refit use seed 42")
    rp.write_report(args.report, sections, title="Final evaluation on the test partition",
                    environment=environment, findings_heading="Findings",
                    partitions_note="Partitions: models trained on train and selected on val during development; "
                                    "this report is the single evaluation on the held-out test partition.")
    print(f"[test] report written to {args.report} and {args.report.with_suffix('.csv')}", flush=True)
    print("[test] DONE", flush=True)


if __name__ == "__main__":
    main()
