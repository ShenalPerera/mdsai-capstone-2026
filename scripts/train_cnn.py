#!/usr/bin/env python3
"""
Train the Stage D CNN (src/models/cnn.py) on the extracted 224px image cache.

Modes, run in this order (Build plan, Stage D):
  overfit   200 fixed train images, whole model trainable, no flip, up to 100
            epochs; PASS when the loss on those 200 images, measured after the
            epoch in eval mode (no dropout), falls below 0.05. Checks the
            pipeline (labels, normalisation, loss, optimiser), not the model.
  frozen    full train set, backbone frozen, heads only, 5 epochs.
  finetune  full train set, whole model, up to 15 epochs.

frozen/finetune score the val partition after every epoch with
src/models/metrics.py; the best epoch and early stopping (3 epochs without
improvement) use val macro AP over the six flaws. The test partition is never
loaded.

Outputs (OUT_DIR defaults to DATA_DIR/cnn_runs/<mode>):
  last.pt       model + optimiser + epoch, written every epoch -> resume
  best.pt       model weights of the best epoch
  history.json  config, environment, per-epoch train loss and val AP
  DATA_DIR/predictions/cnn_<mode>_val.csv   val scores of the best epoch,
                same format as the classical prediction files
  embeddings.npz  (finetune) 1280-d embeddings of train and val from the
                best epoch, for the hybrid arm (Stage E)

Resumable: rerunning the same command continues from last.pt; a finished run
only rewrites its outputs. Progress line every 30 s.

Usage:
  python -u scripts/train_cnn.py DATA_DIR --cache-dir DIR --mode {overfit,frozen,finetune}
      [--out-dir DIR] [--epochs N] [--batch-size 64] [--workers N]
"""
import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
from src.data.dataset import QualityIssuesDataset  # noqa: E402
from src.data.splits import SEED, get_splits  # noqa: E402
from src.eda.report import library_versions  # noqa: E402
from src.models import metrics  # noqa: E402
from src.models.classical import OUTPUTS, UNREC, build_targets  # noqa: E402
from src.models.cnn import QualityNet, image_transform, pos_weights  # noqa: E402

PROGRESS_INTERVAL_SEC = 30
MODE_DEFAULTS = {"overfit": 100, "frozen": 5, "finetune": 15}  # max epochs
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
PATIENCE = 3
OVERFIT_IMAGES = 200
OVERFIT_TARGET_LOSS = 0.05


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomic(obj, path):
    """Write via a temporary file, so a disconnect mid-write never leaves a
    corrupt checkpoint in place of the previous one."""
    tmp = path.with_name(path.name + ".part")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_loader(dataset, batch_size, shuffle, workers, epoch=0):
    """The shuffle order and the workers' flip randomness are seeded from
    SEED + epoch (dropout likewise, in the training loop), so a resumed run
    replays exactly the epochs it missed."""
    generator = torch.Generator().manual_seed(SEED + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=generator, pin_memory=torch.cuda.is_available())


class Progress:
    """Prints a progress line at most every PROGRESS_INTERVAL_SEC."""

    def __init__(self, label, total, unit):
        self.label, self.total, self.unit = label, total, unit
        self.start = self.last = time.time()

    def update(self, done, extra=""):
        now = time.time()
        if now - self.last < PROGRESS_INTERVAL_SEC:
            return
        self.last = now
        rate = done / (now - self.start)
        filled = int(20 * done / self.total)
        print(f"  {self.label} [{'#' * filled}{'-' * (20 - filled)}] {done / self.total:4.0%} "
              f"{done}/{self.total} {self.unit} | {rate:.2f} {self.unit}/s | "
              f"elapsed {fmt_time(now - self.start)} | "
              f"ETA {fmt_time((self.total - done) / rate)}{extra}")


def train_one_epoch(model, loader, criterion, optimiser, scaler, device, label):
    model.train()
    progress = Progress(label, len(loader), "batches")
    total_loss, total_n = 0.0, 0
    for i, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        targets = torch.cat([batch["flaw_targets"], batch["recognisability_target"][:, None]],
                            dim=1).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model(images)
        # loss in float32: mean BCE over the 7 outputs, each pos_weight-balanced
        loss = criterion(logits.float(), targets)
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * len(images)
        total_n += len(images)
        progress.update(i, f" | loss {total_loss / total_n:.4f}")
    return total_loss / total_n


@torch.no_grad()
def fit_loss(model, loader, criterion, device, use_amp):
    """Mean loss over a dataset in eval mode (no dropout, no augmentation):
    how well the model has fitted these images."""
    model.eval()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = torch.cat([batch["flaw_targets"], batch["recognisability_target"][:, None]],
                            dim=1).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        total_loss += criterion(logits.float(), targets).item() * len(images)
        total_n += len(images)
    return total_loss / total_n


@torch.no_grad()
def predict(model, loader, device, use_amp, with_embeddings=False, label="scoring"):
    """Sigmoid scores (DataFrame, image x OUTPUTS) and optionally the 1280-d
    embeddings, keyed by the image names the loader returns."""
    model.eval()
    progress = Progress(label, len(loader), "batches")
    names, scores, embeddings = [], [], []
    for i, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, embedding = model(images, return_embedding=True)
        names += list(batch["image_name"])
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        if with_embeddings:
            embeddings.append(embedding.float().cpu().numpy().astype(np.float16))
        progress.update(i)
    frame = pd.DataFrame(np.concatenate(scores), index=pd.Index(names, name="image"), columns=OUTPUTS)
    return (frame, np.concatenate(embeddings)) if with_embeddings else frame


def ap_line(table):
    return " ".join(f"{out} {table.loc[out, 'AP']:.3f}" for out in OUTPUTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="folder holding the extracted cache images (flat)")
    parser.add_argument("--mode", choices=list(MODE_DEFAULTS), required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    mode = args.mode
    out_dir = args.out_dir or args.data_dir / "cnn_runs" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = args.epochs or MODE_DEFAULTS[mode]
    config = {"mode": mode, "max_epochs": max_epochs, "batch_size": args.batch_size,
              "lr_head": LR_HEAD, "lr_backbone": None if mode == "frozen" else LR_BACKBONE,
              "patience": None if mode == "overfit" else PATIENCE, "seed": SEED,
              "backbone": "efficientnet_b0 IMAGENET1K_V1"}

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    environment = {"gpu": torch.cuda.get_device_name(0) if use_amp else "none (CPU)",
                   "python": platform.python_version(), "libraries": library_versions()}
    print(f"[train] mode {mode} | max {max_epochs} epochs | batch {args.batch_size} | "
          f"device {environment['gpu']} | AMP {use_amp} | seed {SEED}")
    print(f"[train] {environment['libraries']}")

    # Data: train and val only. Labels come from the annotation records.
    splits = get_splits(args.data_dir)
    train_records, val_records = splits["train"], splits["val"]
    missing = [r["image"] for r in train_records + val_records
               if not (args.cache_dir / r["image"]).exists()]
    if missing:
        sys.exit(f"[train] {len(missing)} images missing from {args.cache_dir}, e.g. {missing[:3]}")
    train_targets = build_targets(train_records)
    weights = pos_weights(train_targets)
    print(f"[train] {len(train_records):,} train / {len(val_records):,} val images "
          f"(test partition not loaded)")
    print("[train] pos_weight (neg/pos on train): "
          + " ".join(f"{o} {w:.2f}" for o, w in zip(OUTPUTS, weights.tolist())))

    if mode == "overfit":
        subset = sorted(random.Random(SEED).sample(range(len(train_records)), OVERFIT_IMAGES))
        train_set = Subset(QualityIssuesDataset(train_records, args.cache_dir, image_transform(False),
                                                flat=True), subset)
        print(f"[train] overfit subset: {OVERFIT_IMAGES} fixed train images, no flip")
    else:
        train_set = QualityIssuesDataset(train_records, args.cache_dir, image_transform(True), flat=True)
    val_set = QualityIssuesDataset(val_records, args.cache_dir, image_transform(False), flat=True)
    val_targets = build_targets(val_records)

    # Model. Comparing the first convolution with a randomly initialised
    # copy shows the ImageNet weights were actually loaded.
    model = QualityNet(pretrained=True, freeze_backbone=(mode == "frozen")).to(device)
    random_conv = QualityNet(pretrained=False).backbone[0][0].weight
    loaded_diff = (model.backbone[0][0].weight.detach().cpu() - random_conv.detach()).abs().mean().item()
    print(f"[train] ImageNet weights loaded: first conv differs from random init by "
          f"{loaded_diff:.4f} (0 would mean not loaded)")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable parameters {n_trainable:,} of {n_total:,}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    optimiser = torch.optim.AdamW(model.param_groups(LR_HEAD, LR_BACKBONE))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume
    state = {"epoch": 0, "best_ap": -1.0, "best_epoch": None, "since_best": 0,
             "finished": False, "history": []}
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["config"] != config:
            sys.exit(f"[train] {last_path} was written with a different configuration: "
                     f"{checkpoint['config']}. Use another --out-dir.")
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        print(f"[train] resuming after epoch {state['epoch']} "
              f"(best so far: epoch {state['best_epoch']}, val macro AP {state['best_ap']:.4f})"
              + (" - run already finished, rewriting outputs" if state["finished"] else ""))

    run_start = time.time()
    while not state["finished"] and state["epoch"] < max_epochs:
        epoch = state["epoch"] + 1
        epoch_start = time.time()
        seed_everything(SEED + epoch)
        loader = make_loader(train_set, args.batch_size, True, args.workers, epoch)
        train_loss = train_one_epoch(model, loader, criterion, optimiser, scaler, device,
                                     f"epoch {epoch}/{max_epochs}")
        record = {"epoch": epoch, "train_loss": train_loss}
        state["epoch"] = epoch

        if mode == "overfit":
            loss = fit_loss(model, make_loader(train_set, args.batch_size, False, args.workers),
                            criterion, device, use_amp)
            record["fit_loss"] = loss
            print(f"epoch {epoch} done | train loss {train_loss:.4f} | fit loss (eval mode) {loss:.4f} "
                  f"(target < {OVERFIT_TARGET_LOSS}) | {time.time() - epoch_start:.0f}s")
            state["finished"] = loss < OVERFIT_TARGET_LOSS
        else:
            scores = predict(model, make_loader(val_set, args.batch_size, False, args.workers),
                             device, use_amp, label="val")
            table = metrics.evaluate(val_targets, scores)
            macro_ap = table.loc[metrics.MACRO, "AP"]
            record.update({"val_ap": {o: table.loc[o, "AP"] for o in OUTPUTS},
                           "val_macro_ap": macro_ap, "val_unrec_f1": table.loc[UNREC, "F1"]})
            improved = macro_ap > state["best_ap"]
            if improved:
                state.update(best_ap=macro_ap, best_epoch=epoch, since_best=0)
                save_atomic(model.state_dict(), best_path)
            else:
                state["since_best"] += 1
            print(f"epoch {epoch} done | train loss {train_loss:.4f} | val macro AP {macro_ap:.4f} | "
                  f"{ap_line(table)} | {time.time() - epoch_start:.0f}s | "
                  + ("best so far -> saved best.pt" if improved
                     else f"no improvement for {state['since_best']} (best epoch {state['best_epoch']})"))
            if state["since_best"] >= PATIENCE:
                print(f"[train] early stop: {PATIENCE} epochs without improvement")
                state["finished"] = True

        record["seconds"] = round(time.time() - epoch_start, 1)
        state["history"].append(record)
        if state["epoch"] >= max_epochs:
            state["finished"] = True
        save_atomic({"config": config, "model": model.state_dict(), "optimiser": optimiser.state_dict(),
                     "scaler": scaler.state_dict(), "state": state}, last_path)

    history = {"config": config, "environment": environment, "best_epoch": state["best_epoch"],
               "best_val_macro_ap": None if mode == "overfit" else state["best_ap"],
               "epochs": state["history"]}

    if mode == "overfit":
        final = state["history"][-1]["fit_loss"]
        verdict = "PASS" if final < OVERFIT_TARGET_LOSS else "FAIL"
        history["overfit"] = {"final_fit_loss": final, "target": OVERFIT_TARGET_LOSS, "verdict": verdict}
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[train] overfit {verdict}: fit loss {final:.4f} after {state['epoch']} epochs "
              f"(target < {OVERFIT_TARGET_LOSS})")
        print(f"[train] DONE in {fmt_time(time.time() - run_start)}: {out_dir}")
        return

    # Outputs from the best epoch
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    print(f"[train] best epoch {state['best_epoch']} (val macro AP {state['best_ap']:.4f}): writing outputs")
    eval_train = QualityIssuesDataset(train_records, args.cache_dir, image_transform(False), flat=True)
    with_embeddings = mode == "finetune"
    val_out = predict(model, make_loader(val_set, args.batch_size, False, args.workers),
                      device, use_amp, with_embeddings, label="val")
    val_scores = val_out[0] if with_embeddings else val_out
    predictions_dir = args.data_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / f"cnn_{mode}_val.csv"
    val_scores.to_csv(predictions_path)
    print(f"[train] val scores -> {predictions_path}")
    if with_embeddings:
        train_scores, train_emb = predict(model, make_loader(eval_train, args.batch_size, False, args.workers),
                                          device, use_amp, True, label="train embeddings")
        np.savez(out_dir / "embeddings.npz", train=train_emb, train_images=train_scores.index.to_numpy(str),
                 val=val_out[1], val_images=val_scores.index.to_numpy(str))
        print(f"[train] embeddings train {train_emb.shape}, val {val_out[1].shape} -> {out_dir / 'embeddings.npz'}")

    table = metrics.evaluate(val_targets, val_scores)
    history["best_val"] = {o: {k: table.loc[o, k] for k in ("AP", "F1")} for o in OUTPUTS + [metrics.MACRO]}
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=float), encoding="utf-8")
    print(f"[train] best val: macro AP {table.loc[metrics.MACRO, 'AP']:.4f} | {ap_line(table)}")
    print(f"[train] DONE in {fmt_time(time.time() - run_start)}: {out_dir}")


if __name__ == "__main__":
    main()
