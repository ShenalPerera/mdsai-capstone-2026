# Split Report

DATA_DIR: `/content/drive/MyDrive/Masters/Sem 3/capstone/data`

## Reproducibility check
- two independent calls to get_splits() produce identical ordering: True

## Partition sizes

| partition | images |
|-----------|--------|
| train | 23431 |
| val | 3875 |
| test | 3875 |

## Cross-partition overlap
- zero image overlap between any two partitions

## Per-flaw positive counts (>=2 votes, 6 modelled flaws)

| partition | BLR | BRT | DRK | FRM | OBS | ROT |
|-----------|---|---|---|---|---|---|
| train | 9552 | 1248 | 1367 | 12934 | 840 | 3986 |
| val | 1636 | 224 | 244 | 2148 | 137 | 664 |
| test | 1643 | 252 | 232 | 2200 | 130 | 661 |
