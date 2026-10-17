# Stage C: Classical baseline (Arm 1)

Partitions: train and val only. The test partition is not read during development.

## Environment

- Python 3.13.16 on Linux-6.6.122+-x86_64-with-glibc2.39
- RAM: 12.7 GB
- GPU: none (not needed: scikit-learn models on precomputed features)
- Seed: 42 (gradient boosting random_state; logistic regression is deterministic)

## Findings (hypothesis check)

<!-- findings:start -->
_Findings not yet written._
<!-- findings:end -->

## Data

Train 23,431 images (fitting), val 3,875 images (scoring). Positives per output at >=2 votes:

| index | train | val | val % |
|---|---|---|---|
| BLR | 9552 | 1636 | 42.2 |
| BRT | 1248 | 224 | 5.8 |
| DRK | 1367 | 244 | 6.3 |
| FRM | 12934 | 2148 | 55.4 |
| OBS | 840 | 137 | 3.5 |
| ROT | 3986 | 664 | 17.1 |
| UNREC | 3558 | 606 | 15.6 |

## Feature extraction

Features from `scripts/extract_classical_features.py` on the original images (PIL, no EXIF rotation), resized to a fixed long edge with aspect preserved and no padding. One code path, both resolutions.

| index | 224px | 512px |
|---|---|---|
| rows | 27306 | 27306 |
| upsampled (native long edge < target) | 92 | 2546 |
| upsampled % | 0.3 | 9.3 |

- 19 images are fully uniform (luminance std 0).
- Run: 2026-10-10 on a Colab CPU runtime, single process, about 50 minutes for 27,306 images at both resolutions (~560 images/min).
- Verification (runner check, 26/26 passed): exact train+val image set with no duplicates and no test images; native sizes equal `image_properties.csv` (no EXIF rotation); upsampled flag consistent; all values finite and in range; five random images recomputed from the original zips match their rows exactly.

## Fully uniform images

19 images with luminance std 0. Share positive per output:

| BLR | BRT | DRK | FRM | OBS | ROT | UNREC |
|---|---|---|---|---|---|---|
| 0 | 37 | 11 | 0 | 0 | 0 | 100 |

## Results on val: AP

Average precision per output. The base rate is the AP of a random scorer.

| index | base rate | LR @ 224 | GB @ 224 | LR @ 512 | GB @ 512 |
|---|---|---|---|---|---|
| BLR | 0.422 | 0.798 | 0.809 | 0.738 | 0.774 |
| BRT | 0.058 | 0.428 | 0.472 | 0.425 | 0.451 |
| DRK | 0.063 | 0.556 | 0.551 | 0.549 | 0.541 |
| FRM | 0.554 | 0.679 | 0.698 | 0.663 | 0.698 |
| OBS | 0.035 | 0.102 | 0.126 | 0.103 | 0.107 |
| ROT | 0.171 | 0.219 | 0.238 | 0.213 | 0.23 |
| UNREC | 0.156 | 0.731 | 0.73 | 0.722 | 0.729 |
| macro (6 flaws) | 0.217 | 0.464 | 0.482 | 0.449 | 0.467 |

## Results on val: F1 at threshold 0.5

| index | LR @ 224 | GB @ 224 | LR @ 512 | GB @ 512 |
|---|---|---|---|---|
| BLR | 0.708 | 0.728 | 0.698 | 0.715 |
| BRT | 0.301 | 0.359 | 0.308 | 0.334 |
| DRK | 0.342 | 0.362 | 0.337 | 0.371 |
| FRM | 0.628 | 0.658 | 0.639 | 0.666 |
| OBS | 0.132 | 0.146 | 0.129 | 0.132 |
| ROT | 0.32 | 0.329 | 0.311 | 0.324 |
| UNREC | 0.583 | 0.61 | 0.582 | 0.611 |
| macro (6 flaws) | 0.405 | 0.43 | 0.404 | 0.424 |

## Full table: LR @ 224

| index | n | positives | base rate | AP | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| BLR | 3875 | 1636 | 0.422 | 0.798 | 0.74 | 0.679 | 0.708 |
| BRT | 3875 | 224 | 0.058 | 0.428 | 0.189 | 0.732 | 0.301 |
| DRK | 3875 | 244 | 0.063 | 0.556 | 0.22 | 0.77 | 0.342 |
| FRM | 3875 | 2148 | 0.554 | 0.679 | 0.644 | 0.613 | 0.628 |
| OBS | 3875 | 137 | 0.035 | 0.102 | 0.073 | 0.657 | 0.132 |
| ROT | 3875 | 664 | 0.171 | 0.219 | 0.213 | 0.645 | 0.32 |
| UNREC | 3875 | 606 | 0.156 | 0.731 | 0.467 | 0.776 | 0.583 |
| macro (6 flaws) | <NA> | <NA> | 0.217 | 0.464 | 0.346 | 0.683 | 0.405 |

## Full table: GB @ 224

| index | n | positives | base rate | AP | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| BLR | 3875 | 1636 | 0.422 | 0.809 | 0.743 | 0.714 | 0.728 |
| BRT | 3875 | 224 | 0.058 | 0.472 | 0.235 | 0.754 | 0.359 |
| DRK | 3875 | 244 | 0.063 | 0.551 | 0.242 | 0.717 | 0.362 |
| FRM | 3875 | 2148 | 0.554 | 0.698 | 0.672 | 0.645 | 0.658 |
| OBS | 3875 | 137 | 0.035 | 0.126 | 0.084 | 0.577 | 0.146 |
| ROT | 3875 | 664 | 0.171 | 0.238 | 0.222 | 0.634 | 0.329 |
| UNREC | 3875 | 606 | 0.156 | 0.73 | 0.509 | 0.762 | 0.61 |
| macro (6 flaws) | <NA> | <NA> | 0.217 | 0.482 | 0.366 | 0.674 | 0.43 |

## Full table: LR @ 512

| index | n | positives | base rate | AP | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| BLR | 3875 | 1636 | 0.422 | 0.738 | 0.68 | 0.718 | 0.698 |
| BRT | 3875 | 224 | 0.058 | 0.425 | 0.195 | 0.737 | 0.308 |
| DRK | 3875 | 244 | 0.063 | 0.549 | 0.217 | 0.754 | 0.337 |
| FRM | 3875 | 2148 | 0.554 | 0.663 | 0.645 | 0.632 | 0.639 |
| OBS | 3875 | 137 | 0.035 | 0.103 | 0.072 | 0.628 | 0.129 |
| ROT | 3875 | 664 | 0.171 | 0.213 | 0.204 | 0.649 | 0.311 |
| UNREC | 3875 | 606 | 0.156 | 0.722 | 0.466 | 0.774 | 0.582 |
| macro (6 flaws) | <NA> | <NA> | 0.217 | 0.449 | 0.336 | 0.686 | 0.404 |

## Full table: GB @ 512

| index | n | positives | base rate | AP | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| BLR | 3875 | 1636 | 0.422 | 0.774 | 0.694 | 0.737 | 0.715 |
| BRT | 3875 | 224 | 0.058 | 0.451 | 0.22 | 0.696 | 0.334 |
| DRK | 3875 | 244 | 0.063 | 0.541 | 0.25 | 0.717 | 0.371 |
| FRM | 3875 | 2148 | 0.554 | 0.698 | 0.672 | 0.66 | 0.666 |
| OBS | 3875 | 137 | 0.035 | 0.107 | 0.076 | 0.518 | 0.132 |
| ROT | 3875 | 664 | 0.171 | 0.23 | 0.22 | 0.611 | 0.324 |
| UNREC | 3875 | 606 | 0.156 | 0.729 | 0.516 | 0.749 | 0.611 |
| macro (6 flaws) | <NA> | <NA> | 0.217 | 0.467 | 0.355 | 0.657 | 0.424 |

## ROT against the shape-only reference (val)

Geometry-only cues scored as ROT predictors on val:

| index | n | cue on % | ROT prevalence % | AP | ROT % \| cue on | ROT % \| cue off |
|---|---|---|---|---|---|---|
| landscape-shaped pixels (all images) | 3875 | 6.8 | 17.1 | 0.254 | 54.9 | 14.4 |
| EXIF orientation 6 (all images) | 3875 | 6.1 | 17.1 | 0.267 | 60.9 | 14.3 |
| landscape-shaped pixels (orientation 1 only) | 3637 | 0.7 | 14.3 | 0.143 | 0 | 14.4 |

Classical ROT AP by EXIF orientation:

| index | AP, orientation 1 (n=3637, base 0.143) | AP, orientation 6 (n=238, base 0.609) |
|---|---|---|
| LR @ 224 | 0.183 | 0.786 |
| GB @ 224 | 0.197 | 0.747 |
| LR @ 512 | 0.182 | 0.742 |
| GB @ 512 | 0.196 | 0.73 |

## BLR by native resolution (val)

| index | native < 512 (n=361, base 0.427) | native >= 512 (n=3514, base 0.422) |
|---|---|---|
| LR @ 224 | 0.716 | 0.807 |
| GB @ 224 | 0.759 | 0.813 |
| LR @ 512 | 0.624 | 0.752 |
| GB @ 512 | 0.71 | 0.78 |

## Single-flaw check (val)

| index | n | positives (flaw alone) | base rate | AP LR @ 224 | AP GB @ 224 | AP LR @ 512 | AP GB @ 512 |
|---|---|---|---|---|---|---|---|
| BLR | 1254 | 402 | 0.321 | 0.73 | 0.729 | 0.648 | 0.682 |
| BRT | 876 | 24 | 0.027 | 0.361 | 0.447 | 0.35 | 0.43 |
| DRK | 910 | 58 | 0.064 | 0.853 | 0.81 | 0.851 | 0.812 |
| FRM | 1644 | 792 | 0.482 | 0.582 | 0.626 | 0.581 | 0.627 |
| OBS | 869 | 17 | 0.02 | 0.079 | 0.115 | 0.07 | 0.115 |
| ROT | 1011 | 159 | 0.157 | 0.181 | 0.194 | 0.177 | 0.194 |

## Logistic regression weights (224px, standardised features)

| index | BLR | BRT | DRK | FRM | OBS | ROT | UNREC |
|---|---|---|---|---|---|---|---|
| laplacian_var | -4.84 | -0.98 | 0.21 | -0.78 | 0.29 | -1 | -1.02 |
| tenengrad | 0.41 | 1.76 | -1.18 | -0.75 | -0.25 | 0.25 | -1.6 |
| lum_mean | -0.14 | -0.67 | -1.41 | -0.08 | -0.12 | -0.12 | -0.15 |
| lum_std | -0.01 | 0.32 | 0.39 | 0.04 | 0.59 | 0.05 | -0.03 |
| clip_dark_frac | -0.22 | -0.53 | 0.15 | -0.1 | -0.2 | -0.08 | 0.1 |
| clip_bright_frac | -0.03 | 1.24 | 0.23 | -0.1 | 0.02 | -0.17 | 0.36 |
| hf_residual_energy | 1.42 | 0.77 | -0.24 | -0.06 | -0.46 | 0.77 | -0.06 |
| border_edge_energy | 0.09 | -0.04 | -0.25 | 1.1 | -0.12 | 0.21 | 0.64 |
| center_edge_energy | 1.93 | -2.41 | 0.16 | 0.37 | -0.43 | 0.21 | 0.44 |
| border_edge_share | 0.29 | -0.54 | -0.04 | 0.03 | 0.16 | -0.14 | 0.14 |
