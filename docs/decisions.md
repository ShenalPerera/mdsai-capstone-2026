# Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-09-20 | Recognisability threshold = 2 of 5 votes | The dataset authors' own definition of a poor image (Chiu et al., CVPR 2020) |
| 2026-09-20 | Drop the OTH flaw class | ~0.8% prevalence, reason unspecified, not learnable |
| 2026-09-20 | Classical (non-DL) baseline by design | Makes the CNN comparison meaningful rather than big-CNN vs small-CNN |
| 2026-09-25 | Train image loader must filter to the 23,431 IDs in train.json, not glob the images/train folder | Verified the raw train.zip pool contains 523 extra unannotated images (VizWiz_train_00023431.jpg-00023953.jpg, a contiguous block right after the last annotated ID, confirmed via a full disk-vs-annotation diff on the downloaded data) with zero missing annotated files - harmless, but must be excluded since they have no labels |
| 2026-09-25 | Test-split flaw labels are recoverable; recognisability is not | VizWiz_quality_issues_train_val_test.csv holds raw per-worker votes (IMG, WORKERID, per-flaw 0/1, REJECT, SPLIT) for all three splits, so per-flaw vote counts for test can be reconstructed by grouping on IMG and summing non-rejected rows - but the CSV has no unrecognizable column, so the recognisability head still has no test labels and needs the val-split-in-half fallback |
