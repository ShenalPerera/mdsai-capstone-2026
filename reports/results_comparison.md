# Stage F: Comparison and ablations

Partitions: train and val only. The test partition is not read during development.

## Environment

- Python 3.13.18 on Linux-6.6.159+-x86_64-with-glibc2.39
- RAM: 12.7 GB
- GPU: none (not needed here: this notebook scores saved predictions and refits one gradient-boosting baseline; training GPUs are recorded in the Stage D/E reports and below)
- Libraries: numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, torch 2.14.1+cpu, torchvision 0.29.1+cpu
- Seed: 42 (bootstrap (1000 resamples) and gradient boosting use seed 42; CNN entries seed 42)

## Findings

<!-- findings:start -->
_Findings not yet written._
<!-- findings:end -->

## Comparison table (val AP)

Same val partition (3,875 images, >=2-vote labels) and the same metrics code in every cell. CNN and hybrid: seed-42 entry, mean and range over seeds 42-44.

| index | type | base rate | Classical GB @ 224 | CNN fine-tuned (seed 42) | CNN fine-tuned mean of 3 seeds | CNN fine-tuned seed range | Hybrid (seed 42) | Hybrid mean of 3 seeds | Hybrid seed range | Classical GB @ 512 (deployment) |
|---|---|---|---|---|---|---|---|---|---|---|
| BLR | photometric | 0.422 | 0.809 | 0.885 | 0.882 | 0.878-0.885 | 0.885 | 0.884 | 0.882-0.886 | 0.774 |
| BRT | photometric | 0.058 | 0.472 | 0.564 | 0.573 | 0.564-0.585 | 0.588 | 0.593 | 0.583-0.608 | 0.451 |
| DRK | photometric | 0.063 | 0.551 | 0.623 | 0.629 | 0.623-0.634 | 0.63 | 0.622 | 0.618-0.630 | 0.541 |
| FRM | semantic | 0.554 | 0.698 | 0.823 | 0.823 | 0.815-0.831 | 0.83 | 0.826 | 0.818-0.831 | 0.698 |
| OBS | semantic | 0.035 | 0.126 | 0.563 | 0.534 | 0.518-0.563 | 0.55 | 0.538 | 0.531-0.550 | 0.107 |
| ROT | semantic | 0.171 | 0.238 | 0.696 | 0.695 | 0.689-0.701 | 0.691 | 0.695 | 0.689-0.704 | 0.23 |
| UNREC | recognisability | 0.156 | 0.73 | 0.802 | 0.798 | 0.788-0.805 | 0.797 | 0.795 | 0.788-0.801 | 0.729 |
| macro (6 flaws) |  | 0.217 | 0.482 | 0.692 | 0.689 | 0.687-0.692 | 0.695 | 0.693 | 0.691-0.695 | 0.467 |

Gain = (AP - base rate) / (1 - base rate):

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid | Classical GB @ 512 |
|---|---|---|---|---|
| BLR | 0.669 | 0.8 | 0.8 | 0.609 |
| BRT | 0.44 | 0.537 | 0.563 | 0.417 |
| DRK | 0.521 | 0.597 | 0.605 | 0.51 |
| FRM | 0.322 | 0.603 | 0.618 | 0.322 |
| OBS | 0.094 | 0.547 | 0.533 | 0.074 |
| ROT | 0.08 | 0.633 | 0.627 | 0.07 |
| UNREC | 0.68 | 0.765 | 0.759 | 0.679 |

Mean gain by flaw type:

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid | Classical GB @ 512 |
|---|---|---|---|---|
| photometric | 0.543 | 0.645 | 0.656 | 0.512 |
| semantic | 0.165 | 0.594 | 0.593 | 0.155 |

F1 at threshold 0.5:

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid | Classical GB @ 512 |
|---|---|---|---|---|
| BLR | 0.728 | 0.792 | 0.791 | 0.715 |
| BRT | 0.359 | 0.41 | 0.407 | 0.334 |
| DRK | 0.362 | 0.406 | 0.405 | 0.371 |
| FRM | 0.658 | 0.759 | 0.762 | 0.666 |
| OBS | 0.146 | 0.29 | 0.302 | 0.132 |
| ROT | 0.329 | 0.632 | 0.607 | 0.324 |
| UNREC | 0.61 | 0.642 | 0.653 | 0.611 |
| macro (6 flaws) | 0.43 | 0.548 | 0.546 | 0.424 |

## Pre-registered 224 vs 512 rule

Classical GB, BLR and macro AP, with the paired bootstrap interval of the 224 - 512 difference:

| index | Classical GB @ 224 | Classical GB @ 512 | 224 - 512 |
|---|---|---|---|
| BLR | 0.809 | 0.774 | 0.035 |
| macro (6 flaws) | 0.482 | 0.467 | 0.015 |

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | 0.035 | 0.024 | 0.045 | True |
| macro (6 flaws) | 0.015 | 0.009 | 0.022 | True |

## Bootstrap intervals (val)

AP with 95% bootstrap interval (1,000 resamples of the val images, seed 42):

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid |
|---|---|---|---|
| BLR | 0.809 [0.790, 0.828] | 0.885 [0.872, 0.898] | 0.885 [0.871, 0.899] |
| BRT | 0.472 [0.404, 0.542] | 0.564 [0.496, 0.626] | 0.588 [0.517, 0.651] |
| DRK | 0.551 [0.484, 0.607] | 0.623 [0.562, 0.677] | 0.630 [0.564, 0.685] |
| FRM | 0.698 [0.676, 0.723] | 0.823 [0.806, 0.840] | 0.830 [0.813, 0.847] |
| OBS | 0.126 [0.092, 0.179] | 0.563 [0.491, 0.639] | 0.550 [0.471, 0.632] |
| ROT | 0.238 [0.217, 0.267] | 0.696 [0.660, 0.732] | 0.691 [0.652, 0.730] |
| UNREC | 0.730 [0.700, 0.763] | 0.802 [0.776, 0.828] | 0.797 [0.770, 0.823] |
| macro (6 flaws) | 0.482 [0.464, 0.502] | 0.692 [0.672, 0.714] | 0.695 [0.674, 0.717] |

Paired difference, CNN - classical:

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | 0.076 | 0.061 | 0.091 | True |
| BRT | 0.092 | 0.033 | 0.145 | True |
| DRK | 0.072 | 0.036 | 0.105 | True |
| FRM | 0.125 | 0.105 | 0.144 | True |
| OBS | 0.437 | 0.353 | 0.506 | True |
| ROT | 0.458 | 0.422 | 0.493 | True |
| UNREC | 0.072 | 0.053 | 0.092 | True |
| macro (6 flaws) | 0.21 | 0.189 | 0.227 | True |

Paired difference, hybrid - CNN:

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | -0 | -0.004 | 0.004 | False |
| BRT | 0.024 | 0.002 | 0.046 | True |
| DRK | 0.007 | -0.014 | 0.026 | False |
| FRM | 0.006 | -0.001 | 0.013 | False |
| OBS | -0.013 | -0.04 | 0.009 | False |
| ROT | -0.005 | -0.018 | 0.008 | False |
| UNREC | -0.005 | -0.012 | 0.002 | False |
| macro (6 flaws) | 0.003 | -0.004 | 0.01 | False |

## Ablation 1: training labels at >=1 vote (val, scored at >=2)

Training positives:

| index | >= 2 votes | >= 1 vote | ratio |
|---|---|---|---|
| BLR | 9552 | 14183 | 1.48 |
| BRT | 1248 | 4151 | 3.33 |
| DRK | 1367 | 4331 | 3.17 |
| FRM | 12934 | 17713 | 1.37 |
| OBS | 840 | 3268 | 3.89 |
| ROT | 3986 | 6699 | 1.68 |
| UNREC | 3558 | 3558 | 1 |

Val AP:

| index | base rate | Classical GB @ 224 (>= 2) | Classical GB @ 224, flaws >= 1 vote | CNN fine-tuned (>= 2) | CNN, flaws >= 1 vote |
|---|---|---|---|---|---|
| BLR | 0.422 | 0.809 | 0.808 | 0.885 | 0.878 |
| BRT | 0.058 | 0.472 | 0.468 | 0.564 | 0.596 |
| DRK | 0.063 | 0.551 | 0.556 | 0.623 | 0.6 |
| FRM | 0.554 | 0.698 | 0.696 | 0.823 | 0.8 |
| OBS | 0.035 | 0.126 | 0.11 | 0.563 | 0.523 |
| ROT | 0.171 | 0.238 | 0.238 | 0.696 | 0.68 |
| UNREC | 0.156 | 0.73 | 0.73 | 0.802 | 0.801 |
| macro (6 flaws) | 0.217 | 0.482 | 0.479 | 0.692 | 0.679 |

Paired difference, classical: >= 1 - >= 2:

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | -0.001 | -0.008 | 0.007 | False |
| BRT | -0.005 | -0.025 | 0.016 | False |
| DRK | 0.005 | -0.01 | 0.023 | False |
| FRM | -0.002 | -0.011 | 0.008 | False |
| OBS | -0.016 | -0.043 | 0.008 | False |
| ROT | 0.001 | -0.02 | 0.02 | False |
| UNREC | 0 | 0 | 0 | False |
| macro (6 flaws) | -0.003 | -0.01 | 0.005 | False |

Paired difference, CNN: >= 1 - >= 2:

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | -0.007 | -0.013 | -0.001 | True |
| BRT | 0.032 | -0.008 | 0.073 | False |
| DRK | -0.023 | -0.05 | 0.004 | False |
| FRM | -0.023 | -0.032 | -0.013 | True |
| OBS | -0.04 | -0.082 | 0.002 | False |
| ROT | -0.016 | -0.033 | 0.001 | False |
| UNREC | -0.001 | -0.011 | 0.008 | False |
| macro (6 flaws) | -0.013 | -0.024 | -0.002 | True |

## Ablation 2: ImageNet weights vs random initialisation (val)

| index | epochs run | best epoch | best val macro AP | train loss at best epoch | torch |
|---|---|---|---|---|---|
| CNN fine-tuned | 6 | 3 | 0.6922 | 0.488 | 2.11.0+cu130 |
| CNN, random init | 15 | 13 | 0.5171 | 0.665 | 2.14.1+cu130 |

| index | type | base rate | AP CNN fine-tuned | AP CNN, random init | gain CNN fine-tuned | gain CNN, random init |
|---|---|---|---|---|---|---|
| BLR | photometric | 0.422 | 0.885 | 0.729 | 0.8 | 0.531 |
| BRT | photometric | 0.058 | 0.564 | 0.469 | 0.537 | 0.436 |
| DRK | photometric | 0.063 | 0.623 | 0.517 | 0.597 | 0.485 |
| FRM | semantic | 0.554 | 0.823 | 0.687 | 0.603 | 0.297 |
| OBS | semantic | 0.035 | 0.563 | 0.251 | 0.547 | 0.223 |
| ROT | semantic | 0.171 | 0.696 | 0.45 | 0.633 | 0.336 |
| UNREC | recognisability | 0.156 | 0.802 | 0.725 | 0.765 | 0.674 |
| macro (6 flaws) |  | 0.217 | 0.692 | 0.517 | nan | nan |

Mean gain by flaw type:

| type | gain CNN fine-tuned | gain CNN, random init |
|---|---|---|
| photometric | 0.645 | 0.484 |
| semantic | 0.594 | 0.285 |

Paired difference, pretrained - random init:

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | 0.155 | 0.137 | 0.173 | True |
| BRT | 0.095 | 0.041 | 0.147 | True |
| DRK | 0.105 | 0.069 | 0.142 | True |
| FRM | 0.137 | 0.116 | 0.156 | True |
| OBS | 0.312 | 0.226 | 0.389 | True |
| ROT | 0.246 | 0.207 | 0.282 | True |
| UNREC | 0.077 | 0.056 | 0.099 | True |
| macro (6 flaws) | 0.175 | 0.154 | 0.193 | True |

## Ablation 3: frozen backbone vs full fine-tuning (val)

| index | epochs run | best epoch | best val macro AP | train loss at best epoch | torch |
|---|---|---|---|---|---|
| CNN fine-tuned | 6 | 3 | 0.6922 | 0.488 | 2.11.0+cu130 |
| CNN frozen | 5 | 5 | 0.592 | 0.68 | 2.11.0+cu130 |

| index | type | base rate | AP CNN fine-tuned | AP CNN frozen | gain CNN fine-tuned | gain CNN frozen |
|---|---|---|---|---|---|---|
| BLR | photometric | 0.422 | 0.885 | 0.809 | 0.8 | 0.669 |
| BRT | photometric | 0.058 | 0.564 | 0.412 | 0.537 | 0.376 |
| DRK | photometric | 0.063 | 0.623 | 0.545 | 0.597 | 0.514 |
| FRM | semantic | 0.554 | 0.823 | 0.795 | 0.603 | 0.54 |
| OBS | semantic | 0.035 | 0.563 | 0.354 | 0.547 | 0.33 |
| ROT | semantic | 0.171 | 0.696 | 0.637 | 0.633 | 0.562 |
| UNREC | recognisability | 0.156 | 0.802 | 0.775 | 0.765 | 0.733 |
| macro (6 flaws) |  | 0.217 | 0.692 | 0.592 | nan | nan |

Mean gain by flaw type:

| type | gain CNN fine-tuned | gain CNN frozen |
|---|---|---|
| photometric | 0.645 | 0.52 |
| semantic | 0.594 | 0.478 |

Paired difference, fine-tuned - frozen:

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | 0.076 | 0.064 | 0.088 | True |
| BRT | 0.152 | 0.096 | 0.201 | True |
| DRK | 0.078 | 0.038 | 0.118 | True |
| FRM | 0.028 | 0.013 | 0.043 | True |
| OBS | 0.209 | 0.14 | 0.276 | True |
| ROT | 0.059 | 0.033 | 0.083 | True |
| UNREC | 0.027 | 0.011 | 0.043 | True |
| macro (6 flaws) | 0.1 | 0.083 | 0.118 | True |

## Ablation 4: backbone comparison

Not run. EfficientNet-B0 vs ResNet-50 is the lowest-priority ablation in the plan and the first to drop under the compute and time contingency.
