# CNN image cache verification

Cache: `image_cache_224/`, built by `scripts/prepare_cache.py`, checked by `scripts/verify_cache.py`.

- 3,875 images (test 3,875) in 2 shards, 63 MB
- 224x224 letterbox: long edge exactly 224px, aspect preserved, padding RGB (124, 116, 104) (ImageNet mean), JPEG quality 95, no EXIF rotation
- 15 images natively smaller than 224px were upscaled

| Check | Result | Detail |
|---|---|---|
| manifest holds exactly the expected images, no duplicates | PASS | 3,875 rows, 3,875 expected |
| no images from partitions outside ['test'] | PASS |  |
| shard assignment matches the fixed plan | PASS |  |
| native size = image_properties.csv (no EXIF rotation) | PASS | 0 mismatches |
| content size = aspect-preserving resize of the native size | PASS | 0 mismatches |
| long edge exactly 224px (small images upscaled) | PASS | 0 mismatches; 15 upscaled |
| shards hold exactly the manifest images, each once | PASS | 3,875 files in 2 shards |
| every image decodes to 224x224 RGB | PASS | 0 bad |
| padding bars are the ImageNet-mean colour (bar median; pixels >= 16 px from content) | PASS | 0 images off, 0 not in manifest |
| 5 random images rebuilt from the zips match the cache | PASS | mean abs diff 2.21, 1.17, 1.16, 0.53, 0.64 |

**10/10 checks passed.**
