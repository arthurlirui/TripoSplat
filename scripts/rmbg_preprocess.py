"""Preprocess a single image with BiRefNet and write an RGBA PNG for the C++ library.

The C++ Stage-A library does not run BiRefNet (its torchvision deform_conv2d has no
libtorch binding). Instead, the caller cuts out the background ahead of time with
this script and feeds the resulting RGBA image to the C++ pipeline with
remove_background handled externally.

Usage:
    python scripts/rmbg_preprocess.py input.jpg output_rgba.png [--ckpt ckpts/background_removal/birefnet.safetensors]

This mirrors TripoSplatPipeline.preprocess_image's remove_background path minus the
final black-background composite, so the alpha matte (eroded) is preserved for the
C++ side to do the composite itself.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from triposplat import BiRefNet, load_rmbg  # noqa: E402

_CANVAS_SIZE = 1024


def cutout(image_path: str, ckpt_path: str, erode_radius: int = 1,
           device: str = "cuda") -> Image.Image:
    """Run BiRefNet, return an RGBA PIL image eroded+cropped+resized to 1024x1024.

    Mirrors the alpha branch of triposplat.preprocess_image but keeps the alpha
    channel (does not composite on black), so the C++ library can composite.
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    s = _CANVAS_SIZE / min(w, h)
    image = image.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                         Image.LANCZOS)

    rmbg = load_rmbg(ckpt_path, device=torch.device(device), dtype=torch.float16)
    rgba = rmbg.remove_background(image)  # RGBA from BiRefNet

    if erode_radius > 0:
        rgba.putalpha(rgba.getchannel(3).filter(ImageFilter.MinFilter(2 * erode_radius + 1)))

    alpha = np.array(rgba.getchannel(3))
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        # no foreground — fall back to full frame
        bbox = [0, 0, rgba.width, rgba.height]
    else:
        bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 * 1.2
    rgba = rgba.crop([int(cx - half), int(cy - half), int(cx + half), int(cy + half)])
    rgba = rgba.resize((_CANVAS_SIZE, _CANVAS_SIZE), Image.LANCZOS)
    return rgba


def main():
    ap = argparse.ArgumentParser(description="BiRefNet cutout -> RGBA PNG for the C++ library.")
    ap.add_argument("input", help="Input RGB image path.")
    ap.add_argument("output", help="Output RGBA PNG path.")
    ap.add_argument("--ckpt", default=str(_REPO_ROOT / "ckpts" / "background_removal" / "birefnet.safetensors"),
                    help="BiRefNet safetensors path.")
    ap.add_argument("--erode-radius", type=int, default=1, help="Alpha erosion radius (px).")
    ap.add_argument("--device", default="cuda", help="torch device.")
    args = ap.parse_args()

    out = cutout(args.input, args.ckpt, erode_radius=args.erode_radius, device=args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"[rmbg] wrote {args.output} ({out.size} {out.mode})")


if __name__ == "__main__":
    main()
