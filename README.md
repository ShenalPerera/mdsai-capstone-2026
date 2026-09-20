# mdsai-capstone-2026

## VizWiz Image Quality Assessment

Capstone project for Master in Data Science and AI, UoM. Predicts whether a
photo taken by a blind user is usable, and if not, which quality flaw
(blur, framing, dark/bright exposure, obstruction, rotation) is responsible,
so a downstream recognition system can request a corrective retake instead
of returning a wrong answer. Three approaches are trained and compared per
flaw: a classical (non-deep-learning) feature baseline, a fine-tuned CNN,
and a hybrid of the two.

## Dataset

**VizWiz-QualityIssues** (CC BY 4.0)
Chiu, Zhao, Gurari, *"Assessing Image Quality Issues for Real-World
Problems"*, CVPR 2020. [arXiv:2003.12511](https://arxiv.org/abs/2003.12511)

- Source / task page: https://vizwiz.org/tasks-and-datasets/image-quality-issues/
- Official baseline code: https://github.com/chiutaiyin/VizWiz-QualityIssues
- 39,181 images total (23,431 train / 7,750 val / 8,000 test)
- Licence: CC BY 4.0

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
