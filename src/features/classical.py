"""
Hand-crafted image quality descriptors for the classical baseline (Arm 1).

Images are read with PIL, never cv2.imread: OpenCV applies EXIF orientation
by default, PIL does not, and the project convention is raw orientation
(ROT is a target label - see docs/decisions.md). OpenCV is used only on
already-decoded arrays.

Each image is resized so its long edge equals a fixed target (aspect ratio
preserved, no padding) before any feature is computed, because sharpness
measures are scale-dependent. All features are computed on 8-bit
luminance (PIL "L" mode, ITU-R 601 weights).
"""
import cv2
import numpy as np
from PIL import Image

BORDER_FRACTION = 0.10
CLIP_DARK = 5
CLIP_BRIGHT = 250
EPS = 1e-8

FEATURE_NAMES = [
    "laplacian_var",
    "tenengrad",
    "lum_mean",
    "lum_std",
    "clip_dark_frac",
    "clip_bright_frac",
    "hf_residual_energy",
    "border_edge_energy",
    "center_edge_energy",
    "border_edge_share",
]


def read_image(path):
    """Decode to RGB without applying EXIF orientation."""
    with Image.open(path) as im:
        return im.convert("RGB")


def resize_long_edge(image, long_edge):
    scale = long_edge / max(image.size)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def compute_features(image):
    gray_u8 = np.asarray(image.convert("L"), dtype=np.uint8)
    gray = gray_u8.astype(np.float64)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_energy = gx ** 2 + gy ** 2

    # Median-filter residual rather than Gaussian: the median preserves
    # edges, so the residual is dominated by noise instead of edge content.
    residual = gray - cv2.medianBlur(gray_u8, 3).astype(np.float64)

    h, w = gray.shape
    band = max(1, round(min(h, w) * BORDER_FRACTION))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:band, :] = True
    border_mask[-band:, :] = True
    border_mask[:, :band] = True
    border_mask[:, -band:] = True
    border_energy = grad_energy[border_mask].mean()
    center_energy = grad_energy[~border_mask].mean() if (~border_mask).any() else 0.0

    return {
        "laplacian_var": laplacian.var(),
        "tenengrad": grad_energy.mean(),
        "lum_mean": gray.mean(),
        "lum_std": gray.std(),
        "clip_dark_frac": (gray_u8 <= CLIP_DARK).mean(),
        "clip_bright_frac": (gray_u8 >= CLIP_BRIGHT).mean(),
        "hf_residual_energy": (residual ** 2).mean(),
        "border_edge_energy": border_energy,
        "center_edge_energy": center_energy,
        # A share, not a border/centre ratio: the ratio explodes when the
        # centre is flat (dark or blank images), producing extreme outliers.
        "border_edge_share": border_energy / (border_energy + center_energy + EPS),
    }
