# Test-Label Reconstruction Validation

**Verdict: REJECTED.** Reconstruction does not work, and it is not a fixable
script bug — see Conclusion at the end. Do not use this CSV to reconstruct
test-split labels.

DATA_DIR: `/content/drive/MyDrive/Masters/Sem 3/capstone/data`

CSV source: `/content/drive/MyDrive/Masters/Sem 3/capstone/data/annotations/VizWiz_quality_issues_train_val_test.csv`
CSV rows: 196065

## Split: train
- published records: 23431

### Variant: excluding REJECT=1 rows
- exact matches: 331 / 23431 (1.41%)
- mismatches: 23100

| image | mismatched flaws | published | reconstructed |
|-------|-------------------|-----------|----------------|
| VizWiz_train_00022585.jpg | ALL | {'BLR': 5, 'BRT': 3, 'DRK': 0, 'FRM': 1, 'NON': 0, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |
| VizWiz_train_00004143.jpg | BLR,BRT,DRK,FRM,NON,ROT | {'BLR': 3, 'BRT': 0, 'DRK': 0, 'FRM': 4, 'NON': 1, 'OBS': 0, 'OTH': 0, 'ROT': 0} | {'BLR': 5, 'BRT': 1, 'DRK': 1, 'FRM': 1, 'NON': 0, 'OBS': 0, 'OTH': 0, 'ROT': 1} |
| VizWiz_train_00002092.jpg | BLR,DRK,FRM,OBS,OTH,ROT | {'BLR': 3, 'BRT': 0, 'DRK': 2, 'FRM': 0, 'NON': 0, 'OBS': 1, 'OTH': 2, 'ROT': 0} | {'BLR': 4, 'BRT': 0, 'DRK': 0, 'FRM': 3, 'NON': 0, 'OBS': 0, 'OTH': 0, 'ROT': 5} |
| VizWiz_train_00008936.jpg | BLR,NON | {'BLR': 4, 'BRT': 0, 'DRK': 0, 'FRM': 4, 'NON': 0, 'OBS': 1, 'OTH': 0, 'ROT': 0} | {'BLR': 0, 'BRT': 0, 'DRK': 0, 'FRM': 4, 'NON': 1, 'OBS': 1, 'OTH': 0, 'ROT': 0} |
| VizWiz_train_00020927.jpg | ALL | {'BLR': 2, 'BRT': 0, 'DRK': 0, 'FRM': 0, 'NON': 3, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |

### Variant: including all rows
- exact matches: 365 / 23431 (1.56%)
- mismatches: 23066

| image | mismatched flaws | published | reconstructed |
|-------|-------------------|-----------|----------------|
| VizWiz_train_00022585.jpg | ALL | {'BLR': 5, 'BRT': 3, 'DRK': 0, 'FRM': 1, 'NON': 0, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |
| VizWiz_train_00004143.jpg | BLR,BRT,DRK,FRM,NON,ROT | {'BLR': 3, 'BRT': 0, 'DRK': 0, 'FRM': 4, 'NON': 1, 'OBS': 0, 'OTH': 0, 'ROT': 0} | {'BLR': 5, 'BRT': 1, 'DRK': 1, 'FRM': 1, 'NON': 0, 'OBS': 0, 'OTH': 0, 'ROT': 1} |
| VizWiz_train_00002092.jpg | BLR,DRK,FRM,OBS,OTH,ROT | {'BLR': 3, 'BRT': 0, 'DRK': 2, 'FRM': 0, 'NON': 0, 'OBS': 1, 'OTH': 2, 'ROT': 0} | {'BLR': 4, 'BRT': 0, 'DRK': 0, 'FRM': 3, 'NON': 0, 'OBS': 0, 'OTH': 0, 'ROT': 5} |
| VizWiz_train_00008936.jpg | BLR,NON | {'BLR': 4, 'BRT': 0, 'DRK': 0, 'FRM': 4, 'NON': 0, 'OBS': 1, 'OTH': 0, 'ROT': 0} | {'BLR': 0, 'BRT': 0, 'DRK': 0, 'FRM': 4, 'NON': 1, 'OBS': 1, 'OTH': 0, 'ROT': 0} |
| VizWiz_train_00020927.jpg | ALL | {'BLR': 2, 'BRT': 0, 'DRK': 0, 'FRM': 0, 'NON': 3, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |

## Split: val
- published records: 7750

### Variant: excluding REJECT=1 rows
- exact matches: 0 / 7750 (0.00%)
- mismatches: 7750

| image | mismatched flaws | published | reconstructed |
|-------|-------------------|-----------|----------------|
| VizWiz_val_00002854.jpg | ALL | {'BLR': 0, 'BRT': 0, 'DRK': 0, 'FRM': 2, 'NON': 0, 'OBS': 1, 'OTH': 0, 'ROT': 5} | no CSV rows found |
| VizWiz_val_00001607.jpg | ALL | {'BLR': 0, 'BRT': 0, 'DRK': 0, 'FRM': 0, 'NON': 5, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |
| VizWiz_val_00005141.jpg | ALL | {'BLR': 3, 'BRT': 0, 'DRK': 0, 'FRM': 0, 'NON': 2, 'OBS': 0, 'OTH': 0, 'ROT': 1} | no CSV rows found |
| VizWiz_val_00000516.jpg | ALL | {'BLR': 5, 'BRT': 0, 'DRK': 0, 'FRM': 4, 'NON': 0, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |
| VizWiz_val_00007258.jpg | ALL | {'BLR': 0, 'BRT': 0, 'DRK': 0, 'FRM': 3, 'NON': 2, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |

### Variant: including all rows
- exact matches: 0 / 7750 (0.00%)
- mismatches: 7750

| image | mismatched flaws | published | reconstructed |
|-------|-------------------|-----------|----------------|
| VizWiz_val_00002854.jpg | ALL | {'BLR': 0, 'BRT': 0, 'DRK': 0, 'FRM': 2, 'NON': 0, 'OBS': 1, 'OTH': 0, 'ROT': 5} | no CSV rows found |
| VizWiz_val_00001607.jpg | ALL | {'BLR': 0, 'BRT': 0, 'DRK': 0, 'FRM': 0, 'NON': 5, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |
| VizWiz_val_00005141.jpg | ALL | {'BLR': 3, 'BRT': 0, 'DRK': 0, 'FRM': 0, 'NON': 2, 'OBS': 0, 'OTH': 0, 'ROT': 1} | no CSV rows found |
| VizWiz_val_00000516.jpg | ALL | {'BLR': 5, 'BRT': 0, 'DRK': 0, 'FRM': 4, 'NON': 0, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |
| VizWiz_val_00007258.jpg | ALL | {'BLR': 0, 'BRT': 0, 'DRK': 0, 'FRM': 3, 'NON': 2, 'OBS': 0, 'OTH': 0, 'ROT': 0} | no CSV rows found |


## Conclusion

Three real bugs were found and fixed along the way before reaching this
result:
1. UTF-8 BOM on the CSV's first header, turning `IMG` into `﻿IMG` and
   silently breaking every row lookup.
2. In verify_data.py, the combined filename
   `VizWiz_quality_issues_train_val_test.csv` contains the substring
   "train", which fooled filename-based split inference into dumping all
   196,065 raw rows into the "train" bucket regardless of each row's own
   SPLIT value (train's record count showed 219,496 instead of 23,431).
3. The CSV zero-pads image IDs to 12 digits (`000000020000`) while
   train.json/val.json/test.json and the on-disk filenames use 8 digits
   (`00022585`) for the same photo - fixed by joining on the extracted
   numeric ID instead of the raw filename string.

After all three fixes, exact-match rate is still only 1.41-1.56% on train
(331-365 / 23,431) and exactly 0% on val (0 / 7,750). Follow-up diagnostics
ruled out a remaining padding/format issue:

- SPLIT value counts in the CSV: TRAIN=134,745, TEST=39,965, VAL=21,355
  (sums to the full 196,065 rows).
- Where a CSV filename actually contains train/val/test as a substring, it
  always agrees with that row's SPLIT column - zero contradictions found
  (e.g. no row with filename containing "val" but SPLIT=TRAIN).
- But the numeric ID ranges do not correspond between the two sources:
  - `train.json` ids span 0-23,430 (23,431 unique ids, matches record count)
  - CSV `TRAIN` ids span 0-44,734 (26,949 unique ids) - overlap with
    train.json is only 19,972 (85%), not the ~100% a shared numbering
    scheme would produce
  - `val.json` ids span 0-7,749 (7,750 unique ids)
  - CSV `VAL` ids span 28,000-44,551 (4,271 unique ids) - **zero overlap**
    with val.json's range

**Root cause: the CSV's IMG numbering is a different ID space from
train.json/val.json/test.json's numbering**, not merely a padding-format
difference. The most likely explanation is that the CSV uses the original/
global VizWiz corpus image ID (shared across VizWiz's other tasks), while
the quality-issues JSON files use a task-local renumbering restarted at 0
for each split. No crosswalk between the two ID spaces is provided in the
released files.

**Consequence:** test-split labels - both the six flaws and
recognisability - cannot be reliably reconstructed from this CSV. See
docs/decisions.md (2026-09-25, "SUPERSEDES the row above") for the decision
this drives: the val-split-in-half fallback (fixed seed 42) is needed for
ALL heads on the primary evaluation, not recognisability alone.
