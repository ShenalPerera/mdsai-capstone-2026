# mdsai-capstone-2026

## Actionable Image Quality Assessment for Assistive Photography

*A comparative study of classical, deep, and hybrid approaches*

Capstone project for Master in Data Science and AI, UoM. Predicts whether a
photo taken by a blind and low-vision (BLV) user is usable, and if not,
which quality flaw (blur, framing, dark/bright exposure, obstruction,
rotation) is responsible, so a downstream recognition system can request a
corrective retake instead of returning a wrong answer. Three approaches are
trained and compared per flaw: a classical (non-deep-learning) feature
baseline, a fine-tuned CNN, and a hybrid of the two.

## Dataset

**VizWiz-QualityIssues** (CC BY 4.0)
Chiu, Zhao, Gurari, *"Assessing Image Quality Issues for Real-World
Problems"*, CVPR 2020. [arXiv:2003.12511](https://arxiv.org/abs/2003.12511)

- Source / task page: https://vizwiz.org/tasks-and-datasets/image-quality-issues/
- Official baseline code: https://github.com/chiutaiyin/VizWiz-QualityIssues
- 39,181 images total (23,431 train / 7,750 val / 8,000 test)
- Licence: CC BY 4.0

## Demonstration app

Upload a photo, get a usability verdict and one plain-language retake instruction from the
fine-tuned CNN (`src/app/demo.py`, `scripts/run_demo.py`). Instructions come from a fixed phrase
set (never generated text); a flaw counts as detected at confidence 0.5; photos are used exactly
as stored, without EXIF auto-rotation, matching training.

**Run in Google Colab (nothing to install):**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShenalPerera/mdsai-capstone-2026/blob/main/notebooks/07_demo.ipynb)
— Runtime > Run all, then open the printed `gradio.live` link. The notebook downloads the trained
weights (`best.pt`, about 16 MB; weights are not committed to the repository).

**Run locally:**

```bash
pip install -r requirements.txt
# download best.pt from the link in notebooks/07_demo.ipynb (cell 3)
python scripts/run_demo.py --checkpoint best.pt     # opens http://127.0.0.1:7860
```

## Repository structure

```
.
├── data/               # downloaded dataset (gitignored, not committed)
├── docs/               # design docs, decision log
│   └── decisions.md
├── notebooks/          # exploratory / training notebooks
├── reports/            # generated reports (e.g. data_summary.md)
├── scripts/            # data download + verification scripts
│   ├── download_data.sh
│   └── verify_data.py
├── src/                # project source code (features, models, pipeline)
├── requirements.txt
└── README.md
```

`notes/` and `drafts/` are local-only scratch directories (gitignored).

## Reproduction steps

1. Create and activate a Python environment, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset (defaults to `./data`, pass a different path as
   the first argument if desired):

   ```bash
   bash scripts/download_data.sh ./data
   ```

3. Verify the download and inspect the label schema:

   ```bash
   python scripts/verify_data.py ./data
   ```

   This writes a summary to `reports/data_summary.md`.

4. See `docs/decisions.md` for key modelling decisions and their rationale.
