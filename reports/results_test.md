# Final evaluation on the test partition

Partitions: models trained on train and selected on val during development; this report is the single evaluation on the held-out test partition.

## Environment

- Python 3.13.16 on Linux-6.6.159+-x86_64-with-glibc2.39
- RAM: 12.7 GB
- GPU: Tesla T4 (inference only: saved checkpoints scored on the test partition)
- Libraries: numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, torch 2.14.1+cu130, torchvision 0.29.1+cu130
- Seed: 42 (bootstrap (1000 resamples) and gradient boosting refit use seed 42)

## Findings

<!-- findings:start -->
**The development conclusions hold on the held-out test partition.** Test AP (3,875 images never
used for any decision), against val; CNN and hybrid: seed-42 entry, mean and range over seeds 42-44:

| Output | Type | Classical GB @ 224 | CNN | CNN 3 seeds | Hybrid | Hybrid 3 seeds |
|---|---|---|---|---|---|---|
| BLR | photometric | 0.791 | 0.879 | 0.877 (0.874-0.879) | 0.882 | 0.881 (0.877-0.883) |
| DRK | photometric | 0.587 | 0.645 | 0.633 (0.620-0.645) | 0.641 | 0.639 (0.635-0.641) |
| BRT | photometric | 0.495 | 0.585 | 0.590 (0.585-0.600) | 0.593 | 0.602 (0.593-0.618) |
| FRM | semantic | 0.693 | 0.822 | 0.819 (0.811-0.823) | 0.828 | 0.822 (0.815-0.828) |
| OBS | semantic | 0.100 | 0.483 | 0.476 (0.472-0.483) | 0.489 | 0.474 (0.457-0.489) |
| ROT | semantic | 0.223 | 0.702 | 0.699 (0.694-0.702) | 0.696 | 0.696 (0.690-0.703) |
| UNREC | overall | 0.746 | 0.808 | 0.805 (0.798-0.808) | 0.808 | 0.803 (0.797-0.808) |
| macro (6 flaws) | | 0.482 | 0.686 | 0.682 (0.680-0.686) | 0.688 | 0.686 (0.682-0.688) |

Mean gain = (AP - base rate) / (1 - base rate), classical / CNN / hybrid: photometric 0.552 / 0.656 /
0.659, semantic 0.141 / 0.564 / 0.569 (val: 0.543 / 0.645 / 0.656 and 0.165 / 0.594 / 0.593).

1. **Selection on val inflated the CNN and hybrid only slightly.** Macro AP val to test: CNN 0.692
   to 0.686 (-0.006), hybrid 0.695 to 0.688 (-0.007), classical GB @ 224 0.482 to 0.482 (not
   tuned on val). The common flaws move by at most 0.006 for the CNN (BLR -0.006, FRM -0.002,
   ROT +0.006).
2. **Classical to CNN is confirmed on every output.** Paired bootstrap intervals (1,000 test
   resamples) exclude 0 for all outputs: ROT +0.479 (0.442-0.512), OBS +0.382 (0.298-0.469), FRM
   +0.128 (0.110-0.146), BRT +0.091 (0.033-0.146), BLR +0.088 (0.073-0.104), UNREC +0.061
   (0.042-0.081), DRK +0.058 (0.021-0.096); macro +0.205 (0.184-0.223), against +0.210 on val.
   The gain remains concentrated in the semantic flaws, the hypothesis of the project.
3. **The hybrid is not distinguishable from the CNN.** No hybrid - CNN interval excludes 0 on test;
   macro +0.002 (-0.004 to +0.008). The marginal BRT gain seen on val (+0.024, 0.002-0.046) does not
   replicate as a single-seed difference (+0.007, -0.014 to +0.028); the three-seed means still
   differ in the same direction (0.602 vs 0.590) with overlapping ranges. Adding the classical
   features to a fine-tuned CNN gives no measurable improvement.
4. **The ROT result is content, not shape, on unseen images.** CNN ROT AP 0.702 against 0.239 for
   landscape shape alone; within EXIF orientation 1 (n = 3,620), where shape carries no signal,
   0.644 against a base rate of 0.144 (classical 0.192).
5. **OBS is the least stable output.** CNN OBS AP falls from 0.563 (val) to 0.483 (test). Seed 42
   was the highest of the three CNN seeds on val OBS (0.518-0.563); on test the seeds span
   0.472-0.483 and the 95% interval is 0.397-0.575 (130 positives). The CNN-classical gap on OBS
   (0.483 vs 0.100) is unaffected. BRT, DRK and UNREC rise on test by 0.006-0.023 for the CNN:
   rare-class variation runs in both directions.
6. **224 vs 512 px for the classical features: 224 is better for blur only.** Test BLR AP 0.791 at
   224 vs 0.758 at 512 (val 0.809 vs 0.774); macro AP 0.482 vs 0.481 (val 0.482 vs 0.467), as BRT
   is higher at 512 on test (0.508 vs 0.495) and DRK level (0.585 vs 0.587). The
   Stage C statement that 224 is better on every output does not hold on test; the blur result does.
7. **F1 at 0.5 and the precision pattern repeat.** Macro F1: classical 0.433, CNN 0.543, hybrid
   0.541 (val 0.430, 0.548, 0.546). The CNN keeps high recall on every output (0.72-0.84) with low
   precision on the rare flaws (OBS 0.17, DRK 0.25, BRT 0.31), as on val.

Protocol: classical GB @ 224 and @ 512 were refit on train and reproduced the saved Stage C val
predictions to 3.3e-16 before any test input was read; CNN and hybrid used the saved best.pt
checkpoints. Test inputs were built at this run (reports/cache_verification_test.md, 10/10
checks). The script that produced this report refuses to run again while the report exists.
<!-- findings:end -->

## Inputs and checks

Test partition: 3,875 images (the held-out half of val.json, seed 42), never used for any decision during development. Classical GB refit on train with the Stage C settings; it reproduces the saved Stage C val predictions (max absolute difference: GB @ 224 3.3e-16, GB @ 512 3.3e-16). CNN and hybrid: saved best.pt checkpoints, no training.

| index | test positives | test base rate |
|---|---|---|
| BLR | 1643 | 0.424 |
| BRT | 252 | 0.065 |
| DRK | 232 | 0.06 |
| FRM | 2200 | 0.568 |
| OBS | 130 | 0.034 |
| ROT | 661 | 0.171 |
| UNREC | 625 | 0.161 |

## Comparison table (test AP)

| index | type | base rate | Classical GB @ 224 | CNN fine-tuned (seed 42) | CNN fine-tuned mean of 3 seeds | CNN fine-tuned seed range | Hybrid (seed 42) | Hybrid mean of 3 seeds | Hybrid seed range | Classical GB @ 512 (deployment) |
|---|---|---|---|---|---|---|---|---|---|---|
| BLR | photometric | 0.424 | 0.791 | 0.879 | 0.877 | 0.874-0.879 | 0.882 | 0.881 | 0.877-0.883 | 0.758 |
| BRT | photometric | 0.065 | 0.495 | 0.585 | 0.59 | 0.585-0.600 | 0.593 | 0.602 | 0.593-0.618 | 0.508 |
| DRK | photometric | 0.06 | 0.587 | 0.645 | 0.633 | 0.620-0.645 | 0.641 | 0.639 | 0.635-0.641 | 0.585 |
| FRM | semantic | 0.568 | 0.693 | 0.822 | 0.819 | 0.811-0.823 | 0.828 | 0.822 | 0.815-0.828 | 0.696 |
| OBS | semantic | 0.034 | 0.1 | 0.483 | 0.476 | 0.472-0.483 | 0.489 | 0.474 | 0.457-0.489 | 0.108 |
| ROT | semantic | 0.171 | 0.223 | 0.702 | 0.699 | 0.694-0.702 | 0.696 | 0.696 | 0.690-0.703 | 0.228 |
| UNREC | recognisability | 0.161 | 0.746 | 0.808 | 0.805 | 0.798-0.808 | 0.808 | 0.803 | 0.797-0.808 | 0.742 |
| macro (6 flaws) |  | 0.22 | 0.482 | 0.686 | 0.682 | 0.680-0.686 | 0.688 | 0.686 | 0.682-0.688 | 0.481 |

Gain = (AP - base rate) / (1 - base rate):

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid | Classical GB @ 512 |
|---|---|---|---|---|
| BLR | 0.637 | 0.79 | 0.794 | 0.58 |
| BRT | 0.46 | 0.557 | 0.564 | 0.474 |
| DRK | 0.561 | 0.623 | 0.618 | 0.559 |
| FRM | 0.291 | 0.587 | 0.602 | 0.296 |
| OBS | 0.069 | 0.465 | 0.471 | 0.077 |
| ROT | 0.063 | 0.641 | 0.634 | 0.07 |
| UNREC | 0.698 | 0.771 | 0.771 | 0.693 |

Mean gain by flaw type:

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid | Classical GB @ 512 |
|---|---|---|---|---|
| photometric | 0.552 | 0.656 | 0.659 | 0.538 |
| semantic | 0.141 | 0.564 | 0.569 | 0.148 |

F1 at threshold 0.5:

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid | Classical GB @ 512 |
|---|---|---|---|---|
| BLR | 0.728 | 0.783 | 0.788 | 0.711 |
| BRT | 0.384 | 0.445 | 0.434 | 0.4 |
| DRK | 0.37 | 0.386 | 0.391 | 0.38 |
| FRM | 0.662 | 0.743 | 0.746 | 0.665 |
| OBS | 0.147 | 0.287 | 0.286 | 0.145 |
| ROT | 0.308 | 0.611 | 0.599 | 0.314 |
| UNREC | 0.627 | 0.665 | 0.665 | 0.626 |
| macro (6 flaws) | 0.433 | 0.543 | 0.541 | 0.436 |

## Full table: Classical GB @ 224 (test)

| index | n | positives | base rate | AP | gain | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| BLR | 3875 | 1643 | 0.424 | 0.791 | 0.637 | 0.722 | 0.735 | 0.728 |
| BRT | 3875 | 252 | 0.065 | 0.495 | 0.46 | 0.256 | 0.774 | 0.384 |
| DRK | 3875 | 232 | 0.06 | 0.587 | 0.561 | 0.244 | 0.767 | 0.37 |
| FRM | 3875 | 2200 | 0.568 | 0.693 | 0.291 | 0.673 | 0.651 | 0.662 |
| OBS | 3875 | 130 | 0.034 | 0.1 | 0.069 | 0.083 | 0.623 | 0.147 |
| ROT | 3875 | 661 | 0.171 | 0.223 | 0.063 | 0.21 | 0.576 | 0.308 |
| UNREC | 3875 | 625 | 0.161 | 0.746 | 0.698 | 0.524 | 0.781 | 0.627 |
| macro (6 flaws) | <NA> | <NA> | 0.22 | 0.482 | nan | 0.365 | 0.688 | 0.433 |

## Full table: CNN fine-tuned (test)

| index | n | positives | base rate | AP | gain | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| BLR | 3875 | 1643 | 0.424 | 0.879 | 0.79 | 0.81 | 0.758 | 0.783 |
| BRT | 3875 | 252 | 0.065 | 0.585 | 0.557 | 0.31 | 0.794 | 0.445 |
| DRK | 3875 | 232 | 0.06 | 0.645 | 0.623 | 0.251 | 0.836 | 0.386 |
| FRM | 3875 | 2200 | 0.568 | 0.822 | 0.587 | 0.766 | 0.721 | 0.743 |
| OBS | 3875 | 130 | 0.034 | 0.483 | 0.465 | 0.174 | 0.8 | 0.287 |
| ROT | 3875 | 661 | 0.171 | 0.702 | 0.641 | 0.502 | 0.781 | 0.611 |
| UNREC | 3875 | 625 | 0.161 | 0.808 | 0.771 | 0.549 | 0.842 | 0.665 |
| macro (6 flaws) | <NA> | <NA> | 0.22 | 0.686 | nan | 0.469 | 0.782 | 0.543 |

## Full table: Hybrid (test)

| index | n | positives | base rate | AP | gain | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| BLR | 3875 | 1643 | 0.424 | 0.882 | 0.794 | 0.809 | 0.767 | 0.788 |
| BRT | 3875 | 252 | 0.065 | 0.593 | 0.564 | 0.299 | 0.794 | 0.434 |
| DRK | 3875 | 232 | 0.06 | 0.641 | 0.618 | 0.254 | 0.858 | 0.391 |
| FRM | 3875 | 2200 | 0.568 | 0.828 | 0.602 | 0.765 | 0.728 | 0.746 |
| OBS | 3875 | 130 | 0.034 | 0.489 | 0.471 | 0.173 | 0.815 | 0.286 |
| ROT | 3875 | 661 | 0.171 | 0.696 | 0.634 | 0.485 | 0.782 | 0.599 |
| UNREC | 3875 | 625 | 0.161 | 0.808 | 0.771 | 0.544 | 0.854 | 0.665 |
| macro (6 flaws) | <NA> | <NA> | 0.22 | 0.688 | nan | 0.464 | 0.791 | 0.541 |

## Bootstrap intervals (test)

95% percentile intervals, 1000 resamples of the test images, seed 42, paired across models.

| index | Classical GB @ 224 | CNN fine-tuned | Hybrid |
|---|---|---|---|
| BLR | 0.791 [0.769, 0.812] | 0.879 [0.865, 0.893] | 0.882 [0.867, 0.895] |
| BRT | 0.495 [0.432, 0.565] | 0.585 [0.519, 0.648] | 0.593 [0.529, 0.650] |
| DRK | 0.587 [0.523, 0.649] | 0.645 [0.587, 0.700] | 0.641 [0.581, 0.696] |
| FRM | 0.693 [0.671, 0.716] | 0.822 [0.802, 0.838] | 0.828 [0.809, 0.844] |
| OBS | 0.100 [0.076, 0.141] | 0.483 [0.397, 0.575] | 0.489 [0.402, 0.583] |
| ROT | 0.223 [0.202, 0.250] | 0.702 [0.668, 0.738] | 0.696 [0.662, 0.733] |
| UNREC | 0.746 [0.715, 0.776] | 0.808 [0.779, 0.833] | 0.808 [0.781, 0.834] |
| macro (6 flaws) | 0.482 [0.465, 0.500] | 0.686 [0.664, 0.708] | 0.688 [0.667, 0.710] |

Paired difference, CNN - classical:

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | 0.088 | 0.073 | 0.104 | True |
| BRT | 0.091 | 0.033 | 0.146 | True |
| DRK | 0.058 | 0.021 | 0.096 | True |
| FRM | 0.128 | 0.11 | 0.146 | True |
| OBS | 0.382 | 0.298 | 0.469 | True |
| ROT | 0.479 | 0.442 | 0.512 | True |
| UNREC | 0.061 | 0.042 | 0.081 | True |
| macro (6 flaws) | 0.205 | 0.184 | 0.223 | True |

Paired difference, hybrid - CNN:

| index | difference | low | high | excludes 0 |
|---|---|---|---|---|
| BLR | 0.002 | -0.001 | 0.006 | False |
| BRT | 0.007 | -0.014 | 0.028 | False |
| DRK | -0.005 | -0.018 | 0.01 | False |
| FRM | 0.006 | -0.001 | 0.013 | False |
| OBS | 0.006 | -0.018 | 0.03 | False |
| ROT | -0.006 | -0.015 | 0.004 | False |
| UNREC | 0 | -0.006 | 0.008 | False |
| macro (6 flaws) | 0.002 | -0.004 | 0.008 | False |

## ROT: content or shape? (test)

Geometry-only cues on test:

| index | n | cue on % | ROT prevalence % | AP | ROT % \| cue on | ROT % \| cue off |
|---|---|---|---|---|---|---|
| landscape-shaped pixels (all images) | 3875 | 7.2 | 17.1 | 0.239 | 49.6 | 14.5 |
| EXIF orientation 6 (all images) | 3875 | 6.6 | 17.1 | 0.249 | 54.5 | 14.4 |
| landscape-shaped pixels (orientation 1 only) | 3620 | 0.7 | 14.4 | 0.144 | 0 | 14.5 |

ROT AP by EXIF orientation:

| index | AP, all (base 0.171) | AP, orientation 1 (n=3620, base 0.144) | AP, orientation 6 (n=255, base 0.545) |
|---|---|---|---|
| Classical GB @ 224 | 0.223 | 0.192 | 0.679 |
| CNN fine-tuned | 0.702 | 0.644 | 0.868 |
| Hybrid | 0.696 | 0.642 | 0.855 |

## Val vs test (AP)

Val AP (the partition that selected epochs and entries) against test AP (never used).

| index | Classical GB @ 224 val | Classical GB @ 224 test | Classical GB @ 224 test - val | CNN fine-tuned val | CNN fine-tuned test | CNN fine-tuned test - val | Hybrid val | Hybrid test | Hybrid test - val | Classical GB @ 512 val | Classical GB @ 512 test | Classical GB @ 512 test - val |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BLR | 0.809 | 0.791 | -0.018 | 0.885 | 0.879 | -0.006 | 0.885 | 0.882 | -0.003 | 0.774 | 0.758 | -0.016 |
| BRT | 0.472 | 0.495 | 0.023 | 0.564 | 0.585 | 0.022 | 0.588 | 0.593 | 0.005 | 0.451 | 0.508 | 0.057 |
| DRK | 0.551 | 0.587 | 0.036 | 0.623 | 0.645 | 0.023 | 0.63 | 0.641 | 0.011 | 0.541 | 0.585 | 0.044 |
| FRM | 0.698 | 0.693 | -0.005 | 0.823 | 0.822 | -0.002 | 0.83 | 0.828 | -0.002 | 0.698 | 0.696 | -0.002 |
| OBS | 0.126 | 0.1 | -0.026 | 0.563 | 0.483 | -0.08 | 0.55 | 0.489 | -0.061 | 0.107 | 0.108 | 0.001 |
| ROT | 0.238 | 0.223 | -0.015 | 0.696 | 0.702 | 0.006 | 0.691 | 0.696 | 0.005 | 0.23 | 0.228 | -0.001 |
| UNREC | 0.73 | 0.746 | 0.016 | 0.802 | 0.808 | 0.006 | 0.797 | 0.808 | 0.011 | 0.729 | 0.742 | 0.013 |
| macro (6 flaws) | 0.482 | 0.482 | -0.001 | 0.692 | 0.686 | -0.006 | 0.695 | 0.688 | -0.007 | 0.467 | 0.481 | 0.014 |
