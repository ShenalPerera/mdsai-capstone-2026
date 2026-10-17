"""
Letterboxed square images for the CNN (Stage D), and the shard layout of
the image cache built by scripts/prepare_cache.py.

Each image is resized so its long edge equals the target exactly (aspect
preserved, small images upscaled), with the same resize_long_edge used for
the classical features, then padded to a square. Never stretched, never
cropped: framing (FRM) is a target label, so geometry and frame content must
survive unchanged. No EXIF rotation: ROT labels refer to raw orientation.

Padding is the ImageNet mean colour, which becomes 0 after the network's
input normalisation, so the bars carry no brightness signal (black bars
would read as dark content, the DRK label). The bars still encode the raw
shape; that cue is measured by the shape-only ROT reference
(docs/decisions.md).
"""
from PIL import Image

from src.features.classical import resize_long_edge

CACHE_SIZE = 224
PAD_RGB = (124, 116, 104)  # ImageNet mean (0.485, 0.456, 0.406) x 255
SHARD_SIZE = 2000
JPEG_QUALITY = 95
MANIFEST_COLUMNS = [
    "image", "partition", "shard", "native_width", "native_height",
    "content_width", "content_height", "offset_x", "offset_y",
]


def letterbox(image, size=CACHE_SIZE, fill=PAD_RGB):
    """Return (size x size RGB image, placement dict) for a PIL image."""
    content = resize_long_edge(image, size)
    canvas = Image.new("RGB", (size, size), fill)
    offset_x = (size - content.width) // 2
    offset_y = (size - content.height) // 2
    canvas.paste(content, (offset_x, offset_y))
    return canvas, {
        "content_width": content.width,
        "content_height": content.height,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def expected_content_size(native_width, native_height, size=CACHE_SIZE):
    """Content size letterbox() produces for an image of this native size
    (same arithmetic as resize_long_edge)."""
    scale = size / max(native_width, native_height)
    return max(1, round(native_width * scale)), max(1, round(native_height * scale))


def shard_name(index):
    return f"shard_{index:03d}"


def plan_shards(image_names, shard_size=SHARD_SIZE):
    """Fixed assignment of images to shards: sorted names, consecutive
    chunks. Deterministic, so a resumed build reproduces the same shards."""
    names = sorted(image_names)
    return [names[i:i + shard_size] for i in range(0, len(names), shard_size)]
