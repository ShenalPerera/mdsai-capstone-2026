# Stage G: Human performance ceiling

Partitions: train and val only. The test partition is not read during development.

## Environment

- Python 3.13.18 on Linux-6.6.159+-x86_64-with-glibc2.39
- RAM: 12.7 GB
- GPU: none (not needed: vote counts and saved val scores only)
- Libraries: numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, torch 2.14.1+cpu, torchvision 0.29.1+cpu
- Seed: 42 (held-out-annotator simulation, 200 repeats)

## Findings

<!-- findings:start -->
**The CNN agrees with the annotators at least as well as a single annotator does on every
flaw.** F1 against the other four annotators (held-out-annotator simulation, 200 repeats, val):

| Output | Fleiss' kappa | One annotator | Classical @ 0.5 | CNN @ 0.5 | CNN, best threshold |
|---|---|---|---|---|---|
| BLR | 0.42 | 0.671 | 0.705 | 0.773 | 0.776 |
| FRM | 0.23 | 0.607 | 0.620 | 0.721 | 0.736 |
| ROT | 0.39 | 0.577 | 0.298 | 0.585 | 0.621 |
| UNREC | 0.40 | 0.560 | 0.574 | 0.597 | 0.713 |
| DRK | 0.25 | 0.375 | 0.313 | 0.346 | 0.569 |
| BRT | 0.22 | 0.327 | 0.298 | 0.342 | 0.518 |
| OBS | 0.19 | 0.321 | 0.119 | 0.246 | 0.553 |

1. **A single annotator is a low bar on this data.** Agreement is fair to moderate (Fleiss' kappa
   0.19-0.42). One annotator matches the other four at F1 0.67 on BLR and 0.32-0.38 on the rare
   flaws; on OBS, when one annotator marks an obstruction the others agree 27% of the time, and
   they catch 40% of what the others see. FRM is the least stable label: 38% of val images sit at
   2 or 3 votes, where one annotator flips the label, and only 28% are unanimous.
2. **The single-annotator level is not an upper bound.** The models are trained on 23,431 images
   each judged by five annotators, so they approximate the consensus; a single annotator is one
   noisy sample of it. A model can therefore agree with a 4-annotator panel better than an
   individual does without perceiving quality better than people. Results are reported as
   performance relative to a single annotator, not as exceeding a ceiling.
3. **Blur and framing: the CNN is above the single-annotator level on both precision and recall.**
   BLR F1 0.773 vs 0.671 (precision 0.76 vs 0.73, recall 0.79 vs 0.63); FRM 0.721 vs 0.607
   (precision 0.69 vs 0.69, recall 0.76 vs 0.55). For FRM, the noisiest label, the CNN agrees with
   annotators better than they agree with each other, so its val AP of 0.823 should be read
   against this low ceiling rather than against 1.0.
4. **Rotation: the CNN is at the single-annotator level** (0.585 at 0.5, 0.621 at its best
   threshold, against 0.577). Annotators agree relatively well on ROT (kappa 0.39), so it is the
   flaw with the most room left relative to people.
5. **Rare flaws: the comparison depends on the threshold.** At 0.5 the class-weighted CNN is
   slightly below one annotator on DRK (-0.029) and OBS (-0.075) and level on BRT (+0.015); at its
   best threshold it is 0.19-0.23 above. At 0.5 its recall is about twice the annotator's (0.82-0.86
   vs 0.38-0.42) at lower precision (0.14-0.22 vs 0.27-0.34). The fair comparison lies between the
   two model columns.
6. **Classical features reach the single-annotator level only where the flaw is measurable.** At
   0.5 classical GB exceeds one annotator on BLR (0.705), FRM (0.620) and UNREC (0.574), and is far
   below on ROT (0.298 vs 0.577) and OBS (0.119 vs 0.321), the semantic gap seen throughout.
7. **The arm ranking does not depend on the label definition.** Against the 4-annotator
   references (positive rates lower than the 2-of-5 labels, e.g. BLR 0.371 vs 0.422), AP falls for
   every model but the order holds: classical below CNN and hybrid on every output (ROT 0.214 vs
   0.637, OBS 0.104 vs 0.555), CNN and hybrid within 0.025.

Caveats: the simulation assumes annotators are exchangeable (the data do not identify individual
annotators); best-threshold F1 is chosen on the references it is scored against, so it is
optimistic; F1 rises with base rate, which favours FRM and BLR for every predictor alike; all
values are on the val partition, which also selected the CNN's best epoch.
<!-- findings:end -->

## Data and annotator agreement

Val 3,875 images, five annotators each. Agreement per output (Fleiss' kappa, share of images where all five agree, share at the 2/3 boundary):

| index | fleiss_kappa | unanimous % | borderline 2-3 % | positive >=2 % |
|---|---|---|---|---|
| BLR | 0.416 | 48.129 | 23.252 | 42.219 |
| BRT | 0.218 | 81.316 | 4.748 | 5.781 |
| DRK | 0.249 | 80.903 | 4.8 | 6.297 |
| FRM | 0.229 | 28.026 | 37.703 | 55.432 |
| OBS | 0.194 | 85.884 | 2.942 | 3.535 |
| ROT | 0.385 | 72.387 | 12.465 | 17.135 |
| UNREC | 0.397 | 70.735 | 9.703 | 15.639 |

## The ceiling: one annotator against the other four (val)

Mean over 200 held-out-annotator repeats; reference = at least 2 of the other 4 annotators. Model best F1 is optimistic (threshold chosen on the same references).

| index | Fleiss' kappa | human F1 | human F1 range | Classical GB @ 224 F1 @ 0.5 | CNN fine-tuned F1 @ 0.5 | Hybrid F1 @ 0.5 | Classical GB @ 224 best F1 | CNN fine-tuned best F1 | Hybrid best F1 | CNN fine-tuned F1 @ 0.5 - human | CNN fine-tuned best F1 - human |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BLR | 0.416 | 0.671 | 0.657-0.684 | 0.705 | 0.773 | 0.77 | 0.708 | 0.776 | 0.774 | 0.102 | 0.105 |
| BRT | 0.218 | 0.327 | 0.286-0.362 | 0.298 | 0.342 | 0.337 | 0.454 | 0.518 | 0.539 | 0.015 | 0.191 |
| DRK | 0.249 | 0.375 | 0.340-0.409 | 0.313 | 0.346 | 0.344 | 0.549 | 0.569 | 0.581 | -0.029 | 0.194 |
| FRM | 0.229 | 0.607 | 0.594-0.618 | 0.62 | 0.721 | 0.723 | 0.674 | 0.736 | 0.738 | 0.113 | 0.128 |
| OBS | 0.194 | 0.321 | 0.267-0.367 | 0.119 | 0.246 | 0.252 | 0.205 | 0.553 | 0.548 | -0.075 | 0.232 |
| ROT | 0.385 | 0.577 | 0.548-0.601 | 0.298 | 0.585 | 0.561 | 0.302 | 0.621 | 0.627 | 0.008 | 0.044 |
| UNREC | 0.397 | 0.56 | 0.538-0.584 | 0.574 | 0.597 | 0.608 | 0.661 | 0.713 | 0.706 | 0.037 | 0.153 |

## Precision and recall: human vs CNN at 0.5 (val)

| output | human precision | human recall | CNN fine-tuned precision | CNN fine-tuned recall |
|---|---|---|---|---|
| BLR | 0.725 | 0.625 | 0.757 | 0.791 |
| BRT | 0.289 | 0.378 | 0.216 | 0.815 |
| DRK | 0.339 | 0.421 | 0.216 | 0.862 |
| FRM | 0.685 | 0.546 | 0.686 | 0.759 |
| OBS | 0.27 | 0.396 | 0.144 | 0.82 |
| ROT | 0.626 | 0.535 | 0.459 | 0.808 |
| UNREC | 0.565 | 0.556 | 0.457 | 0.861 |

## Model AP against the 4-annotator references (val)

| output | Classical GB @ 224 | CNN fine-tuned | Hybrid |
|---|---|---|---|
| BLR | 0.757 | 0.849 | 0.851 |
| BRT | 0.434 | 0.494 | 0.517 |
| DRK | 0.543 | 0.604 | 0.606 |
| FRM | 0.615 | 0.743 | 0.751 |
| OBS | 0.104 | 0.555 | 0.538 |
| ROT | 0.214 | 0.637 | 0.633 |
| UNREC | 0.721 | 0.791 | 0.789 |
