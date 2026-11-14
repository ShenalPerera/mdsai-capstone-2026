# Stage E: Hybrid (Arm 3)

Partitions: train and val only. The test partition is not read during development.

## Environment

- Python 3.13.18 on Linux-6.6.159+-x86_64-with-glibc2.39
- RAM: 12.7 GB
- GPU: none (not needed here: this notebook scores saved predictions; the training GPU is under Training runs)
- Libraries: numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, torch 2.14.1+cpu, torchvision 0.29.1+cpu
- Seed: 42 (arm entries use seed 42; seeds 43 and 44 measure run-to-run spread only)

## Findings (does the hybrid add anything?)

<!-- findings:start -->
_Findings not yet written._
<!-- findings:end -->

## Data

Val 3,875 images, scored by every run: the classical entry (GB @ 224) and the CNN and hybrid at seeds 42, 43, 44.

## Training runs

`scripts/train_cnn.py --mode finetune [--hybrid] [--seed N]`: efficientnet_b0 IMAGENET1K_V1, batch 64, AdamW, lr 0.001 heads / 0.0001 backbone, BCE with pos_weight, mixed precision, early stop after 3 epochs without improvement in val macro AP. Hybrid extra inputs: classical_features_224: laplacian_var, tenengrad, lum_mean, lum_std, clip_dark_frac, clip_bright_frac, hf_residual_energy, border_edge_energy, center_edge_energy, border_edge_share (log1p + standardised on train).

| index | run | epochs run | best epoch | best val macro AP | s per epoch | GPU | torch |
|---|---|---|---|---|---|---|---|
| CNN fine-tuned, seed 42 | finetune | 6 | 3 | 0.6922 | 74 | Tesla T4 | 2.11.0+cu130 |
| CNN fine-tuned, seed 43 | finetune_seed43 | 5 | 2 | 0.6866 | 82 | Tesla T4 | 2.14.1+cu130 |
| CNN fine-tuned, seed 44 | finetune_seed44 | 7 | 4 | 0.6896 | 84 | Tesla T4 | 2.14.1+cu130 |
| Hybrid, seed 42 | hybrid_finetune | 6 | 3 | 0.6954 | 83 | Tesla T4 | 2.14.1+cu130 |
| Hybrid, seed 43 | hybrid_finetune_seed43 | 5 | 2 | 0.6912 | 84 | Tesla T4 | 2.14.1+cu130 |
| Hybrid, seed 44 | hybrid_finetune_seed44 | 7 | 4 | 0.6924 | 83 | Tesla T4 | 2.14.1+cu130 |

## Full table: Hybrid (seed 42)

| index | n | positives | base rate | AP | gain | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| BLR | 3875 | 1636 | 0.422 | 0.885 | 0.8 | 0.817 | 0.767 | 0.791 |
| BRT | 3875 | 224 | 0.058 | 0.588 | 0.563 | 0.272 | 0.808 | 0.407 |
| DRK | 3875 | 244 | 0.063 | 0.63 | 0.605 | 0.267 | 0.832 | 0.405 |
| FRM | 3875 | 2148 | 0.554 | 0.83 | 0.618 | 0.785 | 0.74 | 0.762 |
| OBS | 3875 | 137 | 0.035 | 0.55 | 0.533 | 0.185 | 0.832 | 0.302 |
| ROT | 3875 | 664 | 0.171 | 0.691 | 0.627 | 0.492 | 0.794 | 0.607 |
| UNREC | 3875 | 606 | 0.156 | 0.797 | 0.759 | 0.533 | 0.843 | 0.653 |
| macro (6 flaws) | <NA> | <NA> | 0.217 | 0.695 | nan | 0.47 | 0.795 | 0.546 |

## Three arms on val

AP and gain per output (entries: classical GB @ 224, CNN and hybrid at seed 42).

| index | type | base rate | AP Classical GB @ 224 | AP CNN fine-tuned | AP Hybrid | gain Classical GB @ 224 | gain CNN fine-tuned | gain Hybrid |
|---|---|---|---|---|---|---|---|---|
| BLR | photometric | 0.422 | 0.809 | 0.885 | 0.885 | 0.669 | 0.8 | 0.8 |
| BRT | photometric | 0.058 | 0.472 | 0.564 | 0.588 | 0.44 | 0.537 | 0.563 |
| DRK | photometric | 0.063 | 0.551 | 0.623 | 0.63 | 0.521 | 0.597 | 0.605 |
| FRM | semantic | 0.554 | 0.698 | 0.823 | 0.83 | 0.322 | 0.603 | 0.618 |
| OBS | semantic | 0.035 | 0.126 | 0.563 | 0.55 | 0.094 | 0.547 | 0.533 |
| ROT | semantic | 0.171 | 0.238 | 0.696 | 0.691 | 0.08 | 0.633 | 0.627 |
| UNREC | recognisability | 0.156 | 0.73 | 0.802 | 0.797 | 0.68 | 0.765 | 0.759 |
| macro (6 flaws) |  | 0.217 | 0.482 | 0.692 | 0.695 | nan | nan | nan |

Mean gain by flaw type:

| type | gain Classical GB @ 224 | gain CNN fine-tuned | gain Hybrid |
|---|---|---|---|
| photometric | 0.543 | 0.645 | 0.656 |
| semantic | 0.165 | 0.594 | 0.593 |

F1 at threshold 0.5:

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid |
|---|---|---|---|
| BLR | 0.728 | 0.792 | 0.791 |
| BRT | 0.359 | 0.41 | 0.407 |
| DRK | 0.362 | 0.406 | 0.405 |
| FRM | 0.658 | 0.759 | 0.762 |
| OBS | 0.146 | 0.29 | 0.302 |
| ROT | 0.329 | 0.632 | 0.607 |
| UNREC | 0.61 | 0.642 | 0.653 |
| macro (6 flaws) | 0.43 | 0.548 | 0.546 |

## Seed spread and paired differences (val)

Macro AP over the six flaws by seed:

| index | CNN fine-tuned | Hybrid | hybrid - CNN |
|---|---|---|---|
| seed 42 | 0.6922 | 0.6954 | 0.0032 |
| seed 43 | 0.6866 | 0.6912 | 0.0046 |
| seed 44 | 0.6896 | 0.6924 | 0.0029 |
| mean | 0.6894 | 0.693 | 0.0036 |
| min | 0.6866 | 0.6912 | 0.0029 |
| max | 0.6922 | 0.6954 | 0.0046 |

Paired AP difference, hybrid - CNN, same seed:

| index | seed 42 | seed 43 | seed 44 | mean | seeds hybrid ahead | seeds hybrid behind |
|---|---|---|---|---|---|---|
| BLR | -0.0001 | 0.0037 | 0.0025 | 0.002 | 2 | 1 |
| BRT | 0.0239 | 0.0233 | 0.0123 | 0.0198 | 3 | 0 |
| DRK | 0.0072 | -0.0106 | -0.0157 | -0.0064 | 1 | 2 |
| FRM | 0.0065 | 0.0029 | -0.0005 | 0.003 | 2 | 1 |
| OBS | -0.0131 | 0.0081 | 0.0156 | 0.0035 | 2 | 1 |
| ROT | -0.0052 | 0.0006 | 0.003 | -0.0005 | 2 | 1 |
| UNREC | -0.0052 | -0.0038 | -0 | -0.003 | 0 | 3 |
| macro (6 flaws) | 0.0032 | 0.0046 | 0.0029 | 0.0036 | 3 | 0 |

AP across seeds:

| index | CNN fine-tuned mean | CNN fine-tuned min | CNN fine-tuned max | Hybrid mean | Hybrid min | Hybrid max |
|---|---|---|---|---|---|---|
| BLR | 0.882 | 0.878 | 0.885 | 0.884 | 0.882 | 0.886 |
| BRT | 0.573 | 0.564 | 0.585 | 0.593 | 0.583 | 0.608 |
| DRK | 0.629 | 0.623 | 0.634 | 0.622 | 0.618 | 0.63 |
| FRM | 0.823 | 0.815 | 0.831 | 0.826 | 0.818 | 0.831 |
| OBS | 0.534 | 0.518 | 0.563 | 0.538 | 0.531 | 0.55 |
| ROT | 0.695 | 0.689 | 0.701 | 0.695 | 0.689 | 0.704 |
| UNREC | 0.798 | 0.788 | 0.805 | 0.795 | 0.788 | 0.801 |
| macro (6 flaws) | 0.689 | 0.687 | 0.692 | 0.693 | 0.691 | 0.695 |

## Hybrid head weights on the classical features (seed 42)

| index | BLR | BRT | DRK | FRM | OBS | ROT | UNREC |
|---|---|---|---|---|---|---|---|
| laplacian_var | -0.24 | -0.04 | -0.04 | -0.05 | -0.01 | 0.04 | -0.1 |
| tenengrad | -0 | 0.01 | -0.07 | 0.03 | 0 | 0.06 | -0.12 |
| lum_mean | -0.25 | -0 | -0.04 | -0.11 | -0 | 0.03 | -0.1 |
| lum_std | -0 | -0.03 | -0.08 | 0.19 | -0.01 | 0.03 | -0.05 |
| clip_dark_frac | -0 | 0.01 | -0.04 | 0.02 | -0 | 0.04 | -0.12 |
| clip_bright_frac | 0.07 | 0.1 | -0.24 | -0.04 | 0 | -0.02 | 0.05 |
| hf_residual_energy | 0.07 | 0.11 | -0.04 | 0.08 | 0.06 | 0.01 | -0.05 |
| border_edge_energy | -0.11 | -0.15 | 0.15 | -0.04 | -0.06 | -0.1 | 0.07 |
| center_edge_energy | -0.05 | 0.19 | -0.06 | -0.09 | 0.02 | -0.1 | 0.08 |
| border_edge_share | 0.01 | -0 | -0.08 | 0.19 | 0 | -0.04 | 0.12 |
| classical share of logit (std ratio) | 0.19 | 0.11 | 0.15 | 0.18 | 0.02 | 0.11 | 0.2 |

## ROT: content or shape? (val)

Geometry-only cues scored as ROT predictors on val:

| index | n | cue on % | ROT prevalence % | AP | ROT % \| cue on | ROT % \| cue off |
|---|---|---|---|---|---|---|
| landscape-shaped pixels (all images) | 3875 | 6.8 | 17.1 | 0.254 | 54.9 | 14.4 |
| EXIF orientation 6 (all images) | 3875 | 6.1 | 17.1 | 0.267 | 60.9 | 14.3 |
| landscape-shaped pixels (orientation 1 only) | 3637 | 0.7 | 14.3 | 0.143 | 0 | 14.4 |

ROT AP by EXIF orientation:

| index | AP, all (n=3875, base 0.171) | AP, orientation 1 (n=3637, base 0.143) | gain, orientation 1 | AP, orientation 6 (n=238, base 0.609) | gain, orientation 6 |
|---|---|---|---|---|---|
| Classical GB @ 224 | 0.238 | 0.197 | 0.064 | 0.747 | 0.353 |
| CNN fine-tuned | 0.696 | 0.62 | 0.557 | 0.891 | 0.722 |
| Hybrid | 0.691 | 0.619 | 0.555 | 0.895 | 0.731 |

## BLR by native resolution (val)

| index | native < 512 (n=361, base 0.427) | native >= 512 (n=3514, base 0.422) |
|---|---|---|
| Classical GB @ 224 | 0.759 | 0.813 |
| CNN fine-tuned | 0.872 | 0.887 |
| Hybrid | 0.86 | 0.888 |

## Single-flaw check (val)

| index | n | positives (flaw alone) | base rate | AP Classical GB @ 224 | AP CNN fine-tuned | AP Hybrid | gain Classical GB @ 224 | gain CNN fine-tuned | gain Hybrid |
|---|---|---|---|---|---|---|---|---|---|
| BLR | 1254 | 402 | 0.321 | 0.729 | 0.845 | 0.845 | 0.6 | 0.772 | 0.772 |
| BRT | 876 | 24 | 0.027 | 0.447 | 0.464 | 0.462 | 0.431 | 0.449 | 0.447 |
| DRK | 910 | 58 | 0.064 | 0.81 | 0.858 | 0.866 | 0.797 | 0.849 | 0.856 |
| FRM | 1644 | 792 | 0.482 | 0.626 | 0.793 | 0.795 | 0.278 | 0.6 | 0.604 |
| OBS | 869 | 17 | 0.02 | 0.115 | 0.697 | 0.683 | 0.097 | 0.691 | 0.677 |
| ROT | 1011 | 159 | 0.157 | 0.194 | 0.701 | 0.682 | 0.044 | 0.645 | 0.622 |
