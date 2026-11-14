"""
PyTorch Dataset for VizWiz-QualityIssues images and labels.

Images are read either from the original layout (images_dir/<train|val>/name)
or from the extracted CNN image cache (flat=True: images_dir/name, the
224px letterboxed JPEGs built by scripts/prepare_cache.py). Images are opened
with PIL, which never applies EXIF orientation.

extra (optional): a DataFrame of per-image numeric features indexed by image
name, e.g. the classical features for the hybrid arm. Rows are looked up by
name, never by position, so features cannot drift out of step with images.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.data.splits import (
    flaw_vector,
    image_subdir,
    raw_vote_vector,
    recognisability_target,
)

DEFAULT_TRANSFORM = T.ToTensor()


class QualityIssuesDataset(Dataset):
    """Returns image, flaw targets, and recognisability target for a list
    of annotation records (as produced by src.data.splits.get_splits).

    flaw_targets are the 6 modelled codes (BLR BRT DRK FRM OBS ROT)
    thresholded at >=2 votes by default. raw_votes exposes the same 6
    codes' underlying 0-5 vote counts, needed for ordinal and
    threshold-sensitivity work later (see Build plan, Stage F).
    """

    def __init__(
        self,
        records,
        images_dir,
        transform=DEFAULT_TRANSFORM,
        flaw_threshold=2,
        recognisability_threshold=2,
        flat=False,
        extra=None,
    ):
        self.records = records
        self.images_dir = Path(images_dir)
        self.flat = flat
        self.extra = None
        if extra is not None:
            names = [r["image"] for r in records]
            missing = [n for n in names if n not in extra.index]
            if missing:
                raise ValueError(f"{len(missing)} images have no extra features, e.g. {missing[:3]}")
            self.extra = extra.astype("float32")
        self.transform = transform
        self.flaw_threshold = flaw_threshold
        self.recognisability_threshold = recognisability_threshold

    def __len__(self):
        return len(self.records)

    def _image_path(self, image_name):
        if self.flat:
            return self.images_dir / image_name
        return self.images_dir / image_subdir(image_name) / image_name

    def __getitem__(self, idx):
        record = self.records[idx]
        image_name = record["image"]

        with Image.open(self._image_path(image_name)) as im:
            image = im.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        flaw_targets = torch.tensor(
            flaw_vector(record, threshold=self.flaw_threshold), dtype=torch.float32
        )
        raw_votes = torch.tensor(raw_vote_vector(record), dtype=torch.float32)
        recognisability = torch.tensor(
            recognisability_target(record, threshold=self.recognisability_threshold),
            dtype=torch.float32,
        )

        item = {
            "image": image,
            "flaw_targets": flaw_targets,
            "raw_votes": raw_votes,
            "recognisability_target": recognisability,
            "image_name": image_name,
        }
        if self.extra is not None:
            item["extra"] = torch.from_numpy(self.extra.loc[image_name].to_numpy())
        return item
