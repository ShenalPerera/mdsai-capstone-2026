"""
PyTorch Dataset for VizWiz-QualityIssues images and labels.
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
    ):
        self.records = records
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.flaw_threshold = flaw_threshold
        self.recognisability_threshold = recognisability_threshold

    def __len__(self):
        return len(self.records)

    def _image_path(self, image_name):
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

        return {
            "image": image,
            "flaw_targets": flaw_targets,
            "raw_votes": raw_votes,
            "recognisability_target": recognisability,
            "image_name": image_name,
        }
