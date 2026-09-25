# Data Verification Report

DATA_DIR: `/content/drive/MyDrive/Masters/Sem 3/capstone/data`

## Image file counts

Note: files on disk can exceed the annotated count - the raw VizWiz image pool is shared across tasks and is not limited to images annotated for quality issues. See the per-split cross-check below for the number that actually matters (annotated records with no file, and vice versa).

| split | files on disk | expected (paper) |
|-------|----------------|-------------------|
| train | 23954 | 23431 |
| val | 7750 | 7750 |
| test | 8000 | 8000 |

## Discovered annotation files
- /content/drive/MyDrive/Masters/Sem 3/capstone/data/annotations/test.json: 304.7KB
- /content/drive/MyDrive/Masters/Sem 3/capstone/data/annotations/train.json: 3.4MB
- /content/drive/MyDrive/Masters/Sem 3/capstone/data/annotations/val.json: 1.1MB
- /content/drive/MyDrive/Masters/Sem 3/capstone/data/annotations/VizWiz_quality_issues_train_val_test.csv: 13.8MB

## File contents (schema inspection)
  test.json: JSON list, 8000 records
    sample record keys: ['image']
  train.json: JSON list, 23431 records
    sample record keys: ['flaws', 'image', 'unrecognizable']
  val.json: JSON list, 7750 records
    sample record keys: ['flaws', 'image', 'unrecognizable']
  VizWiz_quality_issues_train_val_test.csv: CSV, 196065 rows
    columns: ['IMG', 'WORKERID', 'NON', 'BLR', 'BRT', 'DRK', 'OBS', 'FRM', 'ROT', 'OTH', 'OTH_TEXT', 'REJECT', 'SPLIT']

## Per-split analysis

### Split: train
- records: 23431
- image files on disk: 23954
- image files with no matching annotation record: 523 (e.g. ['VizWiz_train_00023431.jpg', 'VizWiz_train_00023432.jpg', 'VizWiz_train_00023433.jpg', 'VizWiz_train_00023434.jpg', 'VizWiz_train_00023435.jpg'])
- annotated records with no matching image file: 0
- flaw labels present: True

| Flaw | >=1 vote (n) | >=1 vote (%) | >=2 votes (n) | >=2 votes (%) |
|------|--------------|---------------|----------------|----------------|
| BLR | 14183 | 60.5% | 9552 | 40.8% |
| BRT | 4151 | 17.7% | 1248 | 5.3% |
| DRK | 4331 | 18.5% | 1367 | 5.8% |
| FRM | 17713 | 75.6% | 12934 | 55.2% |
| NON | 17046 | 72.7% | 11341 | 48.4% |
| OBS | 3268 | 13.9% | 840 | 3.6% |
| OTH | 1873 | 8.0% | 186 | 0.8% |
| ROT | 6699 | 28.6% | 3986 | 17.0% |

- unrecognizable >= 2: 3558 / 23431 (15.2%)

| unrecognizable votes | count |
|-----------------------|-------|
| 0 | 15751 |
| 1 | 4122 |
| 2 | 1424 |
| 3 | 878 |
| 4 | 725 |
| 5 | 531 |

| flaws per image (>=1 vote) | count |
|------------------------------|-------|
| 1 | 2249 |
| 2 | 6337 |
| 3 | 7712 |
| 4 | 4939 |
| 5 | 1756 |
| 6 | 397 |
| 7 | 41 |

### Split: val
- records: 7750
- image files on disk: 7750
- image files with no matching annotation record: 0
- annotated records with no matching image file: 0
- flaw labels present: True

| Flaw | >=1 vote (n) | >=1 vote (%) | >=2 votes (n) | >=2 votes (%) |
|------|--------------|---------------|----------------|----------------|
| BLR | 4702 | 60.7% | 3279 | 42.3% |
| BRT | 1492 | 19.3% | 476 | 6.1% |
| DRK | 1544 | 19.9% | 476 | 6.1% |
| FRM | 5953 | 76.8% | 4348 | 56.1% |
| NON | 5552 | 71.6% | 3620 | 46.7% |
| OBS | 1097 | 14.2% | 267 | 3.4% |
| OTH | 626 | 8.1% | 65 | 0.8% |
| ROT | 2207 | 28.5% | 1325 | 17.1% |

- unrecognizable >= 2: 1231 / 7750 (15.9%)

| unrecognizable votes | count |
|-----------------------|-------|
| 0 | 5222 |
| 1 | 1297 |
| 2 | 480 |
| 3 | 286 |
| 4 | 273 |
| 5 | 192 |

| flaws per image (>=1 vote) | count |
|------------------------------|-------|
| 1 | 688 |
| 2 | 2084 |
| 3 | 2515 |
| 4 | 1700 |
| 5 | 618 |
| 6 | 133 |
| 7 | 12 |

### Split: test
- records: 8000
- image files on disk: 8000
- image files with no matching annotation record: 0
- annotated records with no matching image file: 0
- flaw labels present: False
- (skipping label statistics: no flaw votes found for this split)

