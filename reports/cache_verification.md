# CNN image cache verification

Cache: `image_cache_224/`, built by `scripts/prepare_cache.py`, checked by `scripts/verify_cache.py`.

- 27,306 images (train 23,431, val 3,875) in 14 shards, 454 MB
- 224x224 letterbox: long edge exactly 224px, aspect preserved, padding RGB (124, 116, 104) (ImageNet mean), JPEG quality 95, no EXIF rotation
- 92 images natively smaller than 224px were upscaled

| Check | Result | Detail |
|---|---|---|
| manifest holds exactly the expected images, no duplicates | PASS | 27,306 rows, 27,306 expected |
| no images from partitions outside ['train', 'val'] | PASS |  |
| shard assignment matches the fixed plan | PASS |  |
| native size = image_properties.csv (no EXIF rotation) | PASS | 0 mismatches |
| content size = aspect-preserving resize of the native size | PASS | 0 mismatches |
| long edge exactly 224px (small images upscaled) | PASS | 0 mismatches; 92 upscaled |
| shards hold exactly the manifest images, each once | PASS | 27,306 files in 14 shards |
| every image decodes to 224x224 RGB | PASS | 0 bad |
| padding bars are the ImageNet-mean colour (bar median; pixels >= 16 px from content) | PASS | 0 images off, 0 not in manifest |
| 5 random images rebuilt from the zips match the cache | PASS | mean abs diff 1.98, 1.64, 0.51, 1.01, 1.08 |

**10/10 checks passed.**
