# EXIF Orientation Check (annotated train images)

DATA_DIR: `/content/drive/MyDrive/Masters/Sem 3/capstone/data`

## Summary
- images checked: 23431
- no EXIF at all: 0 (0.0%)
- EXIF present, no Orientation tag: 0 (0.0%)
- Orientation = 1 (identity): 21483 (91.7%)
- non-identity Orientation (2-8): 1948 (8.3%)

Some images carry a non-identity Orientation tag, so EXIF auto-rotation WOULD change their pixels. Compare ROT rates below: a much higher ROT>=2 rate for tags 6/8 than for identity/untagged images indicates annotators saw raw (uncorrected) orientation.

## Orientation vs ROT vote count

| orientation | n | ROT=0 | ROT=1 | ROT=2 | ROT=3 | ROT=4 | ROT=5 | ROT>=2 rate |
|-------------|---|-------|-------|-------|-------|-------|-------|-------------|
| 1 (normal) | 21483 | 16150 | 2387 | 1175 | 990 | 583 | 198 | 13.7% |
| 6 (rotated 90 CW) | 1948 | 582 | 326 | 372 | 363 | 235 | 70 | 53.4% |
