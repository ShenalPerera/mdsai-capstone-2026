#!/usr/bin/env python3
"""
Demonstration app (Stage H): upload a photo, get a usability verdict and a
corrective instruction from the fine-tuned CNN (src/app/demo.py).

The upload is a plain file input, so no browser or widget step rotates the
image before the model sees it; the preview shows the 224 x 224 letterboxed
image exactly as the model received it, in raw stored orientation.

Runs locally or in Colab; --share prints a temporary public gradio.live link.
Nothing is hosted permanently.

Usage:
  python scripts/run_demo.py --checkpoint PATH/best.pt [--share] [--port 7860]
  python scripts/run_demo.py --data-dir DATA_DIR [--share]     (uses DATA_DIR/cnn_runs/finetune/best.pt)
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.app import demo  # noqa: E402
from src.models.classical import OUTPUTS, UNREC  # noqa: E402

LABELS = {**{code: f"{code} ({name})" for code, name in demo.FLAW_NAMES.items()}, UNREC: "UNREC (unrecognisable)"}
ORIENTATION_NOTE = {1: "upright", 3: "stored upside down", 6: "stored rotated 90° clockwise",
                    8: "stored rotated 90° anticlockwise"}
INTRO = """# Is this photo usable?
Upload a photo. The fine-tuned CNN from this project checks it for six quality flaws and
whether the object is likely recognisable, and gives one plain-language instruction for a retake.
A flaw counts as detected at confidence 0.5 or higher. The photo is used exactly as stored
(no automatic rotation), as during training."""


def build_app(model, device):
    def check(upload):
        if upload is None:
            return None, "Upload a photo first.", "", {}, ""
        path = getattr(upload, "name", upload)
        try:
            result = demo.assess(model, path, device)
        except Exception as error:  # unreadable or non-image file
            return None, f"Could not read this file as an image ({error}).", "", {}, ""
        verdict = ("### ✅ " if result["usable"] else "### ⚠️ ") + result["verdict"]
        instruction = f"**Main instruction:** {result['main_instruction']}"
        others = result["detected"][1:]
        if others:
            instruction += "\n\nAlso detected:\n" + "\n".join(f"- *{name}*: {phrase}" for _, name, _, phrase in others)
        confidences = {LABELS[code]: result["probabilities"][code] for code in OUTPUTS}
        width, height = result["native_size"]
        orientation = result["orientation"]
        details = (f"Original size {width} x {height}; EXIF orientation {orientation} "
                   f"({ORIENTATION_NOTE.get(orientation, 'other')}). Preview: the image as the model saw it.")
        return result["model_input"], verdict, instruction, confidences, details

    with gr.Blocks(title="Actionable image quality assessment") as app:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                upload = gr.File(label="Photo", file_types=["image"])
                button = gr.Button("Check photo", variant="primary")
                preview = gr.Image(label="Model input (224 x 224, raw orientation)", interactive=False)
            with gr.Column():
                verdict = gr.Markdown()
                instruction = gr.Markdown()
                confidences = gr.Label(label="Confidence per output", num_top_classes=len(OUTPUTS))
                details = gr.Markdown()
        outputs = [preview, verdict, instruction, confidences, details]
        button.click(check, inputs=upload, outputs=outputs)
        upload.change(check, inputs=upload, outputs=outputs)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--share", action="store_true", help="temporary public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    checkpoint = args.checkpoint or args.data_dir / "cnn_runs" / "finetune" / "best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = demo.load_model(checkpoint, device)
    print(f"[demo] model {checkpoint} loaded on {device}", flush=True)
    build_app(model, device).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
