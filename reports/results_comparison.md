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
**The hypothesis holds with statistical support, and the ablations locate where the CNN's
advantage comes from.** Val AP, three arms (CNN and hybrid: seed-42 entry, mean and range over
seeds 42-44); gain = (AP - base rate) / (1 - base rate):

| Output | Type | Classical GB @ 224 | CNN | CNN 3 seeds | Hybrid | Hybrid 3 seeds |
|---|---|---|---|---|---|---|
| BLR | photometric | 0.809 | 0.885 | 0.882 (0.878-0.885) | 0.885 | 0.884 (0.882-0.886) |
| DRK | photometric | 0.551 | 0.623 | 0.629 (0.623-0.634) | 0.630 | 0.622 (0.618-0.630) |
| BRT | photometric | 0.472 | 0.564 | 0.573 (0.564-0.585) | 0.588 | 0.593 (0.583-0.608) |
| FRM | semantic | 0.698 | 0.823 | 0.823 (0.815-0.831) | 0.830 | 0.826 (0.818-0.831) |
| OBS | semantic | 0.126 | 0.563 | 0.534 (0.518-0.563) | 0.550 | 0.538 (0.531-0.550) |
| ROT | semantic | 0.238 | 0.696 | 0.695 (0.689-0.701) | 0.691 | 0.695 (0.689-0.704) |
| UNREC | overall | 0.730 | 0.802 | 0.798 (0.788-0.805) | 0.797 | 0.795 (0.788-0.801) |
| macro (6 flaws) | | 0.482 | 0.692 | 0.689 (0.687-0.692) | 0.695 | 0.693 (0.691-0.695) |

Mean gain, classical / CNN / hybrid: photometric 0.543 / 0.645 / 0.656, semantic 0.165 / 0.594 /
0.593. Classical GB @ 512 (deployment configuration): macro AP 0.467, photometric 0.512, semantic
0.155.

1. **Classical to CNN: a large gain on every output, concentrated in the semantic flaws.** Paired
   bootstrap intervals (1,000 val resamples) exclude 0 for all outputs: ROT +0.458 (0.422-0.493),
   OBS +0.437 (0.353-0.506), FRM +0.125 (0.105-0.144), BRT +0.092 (0.033-0.145), BLR +0.076
   (0.061-0.091), DRK +0.072 (0.036-0.105), UNREC +0.072 (0.053-0.092); macro +0.210
   (0.189-0.227). The worst CNN seed (0.687) is 0.205 above the classical arm, so the gap does
   not depend on training randomness either.
2. **CNN to hybrid: not distinguishable overall.** Macro +0.003 (-0.004 to +0.010). BRT is the
   only output whose interval excludes 0, +0.024 (0.002-0.046), and the only one ahead in all
   three seeds. With eight outputs tested, a lower bound of 0.002 is marginal; it is reported as a
   small over-exposure gain supported by both checks, not as a general hybrid advantage.
3. **224 px beats 512 px for the classical features beyond sampling noise.** BLR +0.035
   (0.024-0.045), macro +0.015 (0.009-0.022). The pre-registered rule (2026-10-04) anticipated
   224 = 512 or 512 better; the outcome lies outside it, was recorded as a finding in Stage C, and
   the rule was not amended. Classical GB @ 512 is lower than @ 224 on every output.
4. **Pretraining supplies most of the CNN's advantage, twice as much on semantic flaws.** The same
   network from random initialisation reaches macro AP 0.517 (pretrained 0.692; difference +0.175,
   0.154-0.193, every output excludes 0). Mean gain falls from 0.645 to 0.484 on photometric flaws
   (about a quarter) and from 0.594 to 0.285 on semantic flaws (about half). From scratch, the CNN
   is below classical on the photometric flaws (BLR 0.729 vs 0.809) and still above it on the
   semantic ones (OBS 0.251 vs 0.126, ROT 0.450 vs 0.238). The random-init run used all 15 epochs
   with its best at epoch 13 and a train loss of 0.665 at that epoch (pretrained: epoch 3, 0.488),
   so it was cut off while still improving: the gap is an upper bound on the contribution of
   pretraining under this recipe.
5. **Fine-tuning is what beats classical on the photometric flaws.** Fine-tuned over frozen: +0.100
   macro (0.083-0.118), every output excludes 0; mean gain +0.125 photometric, +0.116 semantic.
   The frozen backbone already reaches semantic gain 0.478 (classical 0.165) but photometric gain
   0.520, below classical 0.543. ImageNet features carry object identity and pose but are largely
   invariant to blur and exposure; fine-tuning adapts them. Largest gains: OBS +0.209, BRT +0.152.
   The frozen run peaked at its fifth and last epoch (+0.007 in that epoch), so a small part of
   the gap reflects its length; one fine-tuned epoch (0.666) already exceeds it (0.592).
6. **Training on >=1-vote labels does not help the rare flaws and hurts the CNN.** Training
   positives grow up to 3.9x (OBS 840 -> 3,268), but these are mostly images that four of five
   annotators judged acceptable. Scored at >=2 votes: classical unchanged (macro -0.003,
   -0.010 to +0.005; no output excludes 0); CNN macro -0.013 (-0.024 to -0.002), below all three
   CNN seeds, with supported drops on FRM -0.023 (-0.032 to -0.013) and BLR -0.007, and OBS -0.040
   (-0.082 to +0.002). At >=1 vote, 76% of training images are FRM-positive (17,713 of 23,431),
   so the label barely separates images. The BRT difference (+0.032, -0.008 to +0.073) is not
   supported. The flexible model absorbs the label noise; the feature-based model is insensitive
   to it. The >=2-vote threshold is retained, and the OBS contingency (near-zero AP after class
   weighting and >=1 labels) is not triggered: OBS AP is 0.563 at >=2.
7. **How precisely each number is known.** Bootstrap 95% interval widths for the CNN range from
   about 0.03 (BLR, FRM) to 0.13-0.15 (BRT, OBS); seed ranges add up to 0.045 on OBS and 0.021 on
   BRT. Single-run differences below about 0.02 on the common flaws, or 0.05 on the rare ones,
   are not interpretable without both checks.

Caveats: all numbers are on the val partition, which also selected each CNN's best epoch, so CNN
values are slightly optimistic; the test half, read once in a separate final run, gives the
unbiased figures. The ablation runs and seeds 43-44 ran on torch 2.14, the seed-42 CNN and frozen
runs on torch 2.11; the >=1-vote CNN (0.679) is also below the two torch-2.14 CNN seeds (0.687,
0.690). Bootstrap intervals cover val sampling only; seed ranges cover training randomness only;
ablations are single-seed.
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
