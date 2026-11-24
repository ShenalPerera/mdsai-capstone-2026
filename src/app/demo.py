"""
Demonstration application logic (Stage H): one photo in, a usability verdict
and a corrective instruction out.

The model is the fine-tuned EfficientNet-B0 entry (seed 42). The input is
prepared exactly as in training: decoded with PIL without applying EXIF
orientation (ROT labels refer to raw stored orientation), letterboxed to
224 x 224 with ImageNet-mean padding, ImageNet-normalised. Every output uses
the project's fixed 0.5 threshold, the operating point of the reported
precision and recall. Instructions come from a fixed phrase set, never from
generated text; when several flaws are detected, the main instruction follows
a fixed retake priority (docs/decisions.md).
"""
import torch
from PIL import Image

from src.data.cache import letterbox
from src.features.classical import read_image
from src.models.classical import OUTPUTS, UNREC
from src.models.cnn import QualityNet, image_transform
from src.models.metrics import THRESHOLD

FLAW_NAMES = {
    "DRK": "too dark", "OBS": "obstructed", "BRT": "too bright",
    "BLR": "blurry", "FRM": "badly framed", "ROT": "rotated",
}
PHRASES = {
    "DRK": "The photo is too dark. Turn on a light or move somewhere brighter.",
    "OBS": "Something is covering the lens or the object. Check your fingers are clear of the camera.",
    "BRT": "The photo is too bright. Avoid pointing at a lamp or window.",
    "BLR": "The photo is blurry. Hold the phone steady and wait a moment before taking it.",
    "FRM": "The object may be cut off. Move the phone further back and centre the object.",
    "ROT": "The photo looks sideways. Hold the phone upright.",
}
# Retake priority: flaws most often making a photo unrecognisable first (Stage B)
PRIORITY = ["DRK", "OBS", "BRT", "BLR", "FRM", "ROT"]
VERDICT_UNUSABLE = "Probably unusable - please retake."
VERDICT_USABLE = "Looks usable."
NO_FLAW = "No quality problem detected."
EXIF_ORIENTATION_TAG = 274


def load_model(checkpoint, device="cpu"):
    """The fine-tuned CNN from its best.pt state dict (no ImageNet download)."""
    model = QualityNet(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def exif_orientation(path):
    """EXIF orientation tag of the file (1 = upright, 6 = stored rotated 90
    degrees clockwise), read but never applied."""
    with Image.open(path) as im:
        return int(im.getexif().get(EXIF_ORIENTATION_TAG, 1))


def model_input(image):
    """The 224 x 224 letterboxed image the model sees (training preprocessing)."""
    square, _ = letterbox(image)
    return square


@torch.no_grad()
def predict(model, image, device="cpu"):
    """Probability per output (OUTPUTS order) for one PIL image."""
    x = image_transform(train=False)(model_input(image)).unsqueeze(0).to(device)
    return dict(zip(OUTPUTS, torch.sigmoid(model(x))[0].tolist()))


def advise(probabilities, threshold=THRESHOLD):
    """Verdict, main instruction and the detected flaws in priority order."""
    detected = [code for code in PRIORITY if probabilities[code] >= threshold]
    return {
        "usable": probabilities[UNREC] < threshold,
        "verdict": VERDICT_USABLE if probabilities[UNREC] < threshold else VERDICT_UNUSABLE,
        "main_instruction": PHRASES[detected[0]] if detected else NO_FLAW,
        "detected": [(code, FLAW_NAMES[code], probabilities[code], PHRASES[code]) for code in detected],
    }


def assess(model, path, device="cpu"):
    """Full pipeline for one uploaded file: raw decode, predict, advise."""
    image = read_image(path)
    probabilities = predict(model, image, device)
    return {**advise(probabilities), "probabilities": probabilities,
            "orientation": exif_orientation(path), "model_input": model_input(image),
            "native_size": image.size}
