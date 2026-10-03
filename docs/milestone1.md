# Actionable Image Quality Assessment for Assistive Photography

**Name:** Perera K.S.D.
**Index No:** 268478R
**Module:** CS5998 Capstone Project, MSc Data Science and Artificial Intelligence, University of Moratuwa
**Supervisor:** Dr. Chathuranga Hettiarachchi
**Repository:** https://github.com/ShenalPerera/mdsai-capstone-2026

## Project scope declaration

| | |
|---|---|
| **Problem type** | Classification: multi-label (quality flaws) and binary (recognisability) |
| **Data type** | Image |
| **Technique category** | Computer Vision: classical feature engineering and deep learning |
| **System context** | End-to-end pipeline and application, with embedded comparative analysis |

## 1. Problem statement

People who are blind or have low vision (BLV) routinely photograph objects in order to identify them, read labels, or check expiry dates. Because they cannot review the photograph they have just taken, they receive no feedback on whether it is usable. A sighted photographer immediately sees that a shot is blurred or that the object is out of frame; a BLV photographer does not.

This produces two failure modes. First, a recognition system returns a confident but incorrect answer on an unusable image, and the user has no signal that anything went wrong, which is consequential when the subject is medication, a cleaning product, or an expiry date. Second, when the user suspects the answer is wrong and retakes the photograph, they do not know what to correct, and repeat the same error.

Published analysis of 39,181 photographs taken by BLV users shows that a substantial proportion are affected by quality issues, with 41% blurred and 56% improperly framed, and that conventional no-reference image quality assessment methods cannot separate recognisable from unrecognisable images in this setting.

This project builds and evaluates the quality-assessment component required to close that feedback loop. Given a single photograph, the system predicts whether the content is recognisable at all, and which specific quality flaws are present, so that a downstream recognition system can request a targeted corrective retake instead of returning an unreliable answer.

The task is distinct from conventional image quality assessment, which predicts perceptual or aesthetic quality on a continuous scale. Here, quality is defined by task usability, and the required output is diagnostic rather than a score, because the flaw identity is what makes the feedback actionable.

## 2. Data source

VizWiz-QualityIssues (Chiu, Zhao and Gurari, CVPR 2020; arXiv:2003.12511), licensed CC BY 4.0 and publicly downloadable from vizwiz.org. The dataset comprises 39,181 photographs taken by BLV users in authentic assistive-technology use, each annotated by five crowdworkers for recognisability and for six quality flaws. Labels are vote counts in the range 0 to 5.

The dataset was downloaded and verified independently; all figures below are measured from the downloaded files rather than quoted from publication.

| Split | Records | Labels |
|-------|---------|--------|
| train | 23,431 | Present |
| val | 7,750 | Present |
| test | 8,000 | Withheld by the dataset authors |

**Label distribution (training split, at least 2 of 5 votes)**

| Flaw | n | % |
|------|---|---|
| Improper framing (FRM) | 12,934 | 55.2 |
| Blur (BLR) | 9,552 | 40.8 |
| Rotation (ROT) | 3,986 | 17.0 |
| Too dark (DRK) | 1,367 | 5.8 |
| Too bright (BRT) | 1,248 | 5.3 |
| Obscured (OBS) | 840 | 3.6 |
| Other (OTH) | 186 | 0.8 |

Unrecognisable at 2 or more votes accounts for 3,558 of 23,431 images (15.2%). The validation split follows the same distribution to within one percentage point on every flaw.

Verification performed. Record counts match the published figures exactly for all three splits adn every annotated image is present on disk; all vote values are integers in 0 to 5; and no filename appears in more than one split. The training image archive contains 23,954 files against 23,431 annotated records. The 523 additional files are privacy-redacted images documented by the dataset authors, lying entirely outside the annotated ID range, and are excluded by constructing all splits from the annotation records rather than from directory listings.

## 3. Intended methods

Three approaches are implemented and compared per flaw.

**Arm 1: classical vision features.** Hand-crafted descriptors are extracted per image: variance of the Laplacian and Tenengrad gradient energy for sharpness; luminance mean, standard deviation and clipped-pixel fraction for exposure; high-frequency residual energy for noise; and edge energy at the image border for framing. These features are classified using logistic regression and gradient boosting. Classical descriptors are the established approach to measuring image sharpness and exposure, and they measure these properties directly rather than learning them. Using them as the baseline isolates the variable under test, which is whether learned representation adds information beyond direct photometric measurement, rather than whether a larger network outperforms a smaller one.
*Modules: CS5803 Advanced Data Mining (feature construction, knowledge-discovery process); CS5617 Data Science.*

**Arm 2: convolutional neural network with transfer learning.** An ImageNet-pretrained backbone (ResNet-50 and EfficientNet-B0 compared) is fine-tuned with a shared encoder and two output heads, one for the six flaw labels and one for recognisability. Training uses class-weighted binary cross-entropy to address label imbalance, with frozen-backbone and full fine-tuning strategies compared. A pretrained versus randomly initialised ablation isolates the contribution of the transferred representation.
*Modules: CS5804 Advanced Deep Learning (CNN architectures; optimizers and hyperparameter tuning); CS5802 Advanced Machine Learning (transfer learning).*

**Arm 3: hybrid.** The hand-crafted descriptors are concatenated with the network's penultimate feature vector before classification, testing whether explicitly supplying photometric measurements improves on requiring the network to learn them from data.
*Module: CS5804; identified in the course guide as an established computer-vision technique.*

**Central hypothesis.** Hand-crafted descriptors directly measure the photometric flaws (blur, dark, bright) and should perform competitively on them, but cannot represent the semantic flaws (framing, obscured, rotation), which require knowledge of what the image contains and where the subject lies. The contribution of deep learning is therefore expected to be concentrated in the semantic flaws. The project measures where this holds rather than assuming it.

**Evaluation protocol.** Since the dataset authors withhold the test labels, held-out evaluation uses a fixed-seed partition of the validation split, with model selection performed on one partition and final evaluation on the other. Metrics are per-flaw average precision and macro-F1, chosen over accuracy because flaw prevalence ranges from 3.6% to 55.2% and accuracy would be dominated by the majority class. The recognisability head is additionally reported as precision and recall. Inter-annotator agreement is computed and reported as a human performance ceiling alongside model scores.
*Modules: CS5803 (evaluation of mining techniques); CS5617 Data Science (ILO 5).*

**Labelling conventions.** The primary threshold is 2 of 5 votes, following the dataset authors' definition of a poor-quality image; measured prevalence at this threshold reproduces published figures to within 0.5 percentage points across all eight codes. OTH is excluded at 0.8% prevalence with unspecified cause, and NON is excluded as the negative class rather than a flaw. Six flaws are therefore modelled: BLR, BRT, DRK, FRM, OBS, ROT.

**Ethical considerations.** The photographs are authentic images captured inside the homes of BLV users and may contain personal information; the dataset authors redacted a documented range of 528 image on these grounds. Data handling, the exclusion of redacted images, and the accessibility implications of false-negative predictions are discussed explicitly.
*Module: CS5617 Data Science (ILO 4, ethics of data collection and privacy).*

## 4. Expected outputs

1. A reproducible preprocessing and feature-extraction pipeline, with all splits constructed from annotation records under fixed random seeds.
2. Three trained model families (classical, CNN, hybrid), each predicting six flaws and recognisability.
3. A per-flaw comparative evaluation across all three approaches, reported against a human performance ceiling derived from annotator agreement, together with ablations on threshold sensitivity, backbone choice, freezing strategy, and pretraining.
4. A demonstration application accepting a photograph and returning a usability verdict with a plain-language corrective instruction, for example "too dark, increase lighting" or "subject is cut off, step back".
5. A public repository containing a decision log, generated verification reports, and a reproducibility checklist.

## 5. Key risks and assumptions

**Sparse flaw classes.** Obscured has 840 positive training examples at the primary threshold, and bright and dark approximately 1,250 and 1,370 respectively. Per-class average precision for these classes will be less stable than for the majority classes. Mitigation: class-weighted loss, targeted augmentation, and a threshold-sensitivity ablation, since a threshold of 1 vote yields 3,268 positives for obscured.

**Annotator disagreement bounds achievable performance.** Labels derive from five crowdworkers who do not always agree, so measured scores are bounded above by human consistency. Mitigation: inter-annotator agreement is computed before modelling and reported as a ceiling, so that model performance is interpreted relative to it rather than in absolute terms.

**Test labels are unavailable.** The dataset authors withhold them, and an accompanying per-worker CSV was investigated as a reconstruction source and found to use an incompatible legacy image numbering. Mitigation: the fixed-seed partition of the validation split described above, which was the intended protocol independently of this finding.

**Single-source data.** All data originates from one collection effort, so generalisation beyond it is not directly evidenced. Mitigation: acknowledged as a limitation, with a small independently captured test set identified as an optional extension.

**Assumptions.** That crowd-sourced consensus at 2 of 5 votes is an acceptable proxy for whether an image is usable; that flaw types are correlated but predictable independently under a multi-label formulation; and that a single photograph provides sufficient evidence for a flaw judgement without reference to a corrected counterpart.
