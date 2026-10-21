"""Coordinate-system display & conversion for TripoSplat Gaussian splats.

Loads a TripoSplat-generated splat (PLY) and a Nerfstudio-style transforms
JSON, then re-expresses both in a unified target convention (OpenCV / OpenGL /
SplatShop), optionally exporting the converted splat + cameras and/or opening
the browser viewer in that convention.

The TripoSplat splat is generated in a model/scene frame; the transforms JSON
stores camera-to-world matrices in the OpenGL / Nerfstudio convention
(+X right, +Y up, -Z forward).  :mod:`multiview_refine.cameras` already
converts those to OpenCV on load.  This script layers a second, explicit
conversion to a chosen unified frame — SplatShop by default — and makes the
applied transform visible.

Usage
    python run_coord_viewer.py
    python run_coord_viewer.py --target splatshop --world-up auto --out export
    python run_coord_viewer.py --target opencv --serve --port 8848
    python run_coord_viewer.py --model-to-world 1 0 0 0 0 0 -1 0 0 1 0 0
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, quote

import numpy as np
import torch

from multiview_refine.cameras import load_cameras
from multiview_refine.coord_transform import (
    convert,
    detect_scene_up,
    export_cameras_json,
    export_ply,
    Target,
    WorldUp,
)
from multiview_refine.ply_io import load_gaussians_ply


def _parse_model_to_world(values):
    """Parse 12 floats (row-major 3x4 [R|t]) into a (3,4) tensor."""
    if values is None:
        return None
    if len(values) != 12:
        raise argparse.ArgumentTypeError(
            "--model-to-world requires exactly 12 values (row-major 3x4 [R|t])"
        )
    return torch.tensor([float(v) for v in values], dtype=torch.float32).reshape(3, 4)


def find_free_port(preferred=None):
    import socket
    if preferred is not None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", preferred))
                return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(
        description="Display & convert a TripoSplat splat + cameras to a unified coordinate frame."
    )
    parser.add_argument("--ply", default="static/00.ply",
                        help="Path to the TripoSplat-generated splat PLY.")
    parser.add_argument("--transforms", default="static/000000/mvs/transforms_train.json",
                        help="Path to a Nerfstudio-style transforms_*.json.")
    parser.add_argument("--target", choices=["opencv", "opengl", "splatshop"],
                        default="splatshop",
                        help="Unified target convention. Default: splatshop.")
    parser.add_argument("--world-up", choices=["x", "y", "z", "auto"], default="auto",
                        help="Source scene world-up axis. 'auto' detects it from "
                             "camera positions. Default: auto.")
    parser.add_argument("--src-frame", choices=["auto", "opencv", "opengl"],
                        default="opencv",
                        help="Source world frame of the splat. 'opencv' = Y-down "
                             "(canonical, post-refinement). 'auto' = infer from "
                             "camera positions. Default: opencv.")
    parser.add_argument("--model-to-world", type=float, nargs=12, default=None,
                        metavar="R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz",
                        help="Optional 3x4 [R|t] (row-major) bringing the splat from "
                             "raw model frame into the scene world frame. Default identity.")
    parser.add_argument("--out", default=None,
                        help="Export directory; if set, writes splat_<target>.ply and "
                             "cameras_<target>.json there.")
    parser.add_argument("--serve", action="store_true",
                        help="Start the browser viewer on the converted scene.")
    parser.add_argument("--port", type=int, default=None,
                        help="Port for --serve. Default: auto-select a free port.")
    parser.add_argument("--no-browser", action="store_true",
                        help="With --serve, don't auto-open the browser.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for conversion math. Default: cpu.")
    args = parser.parse_args()

    ply_path = args.ply
    transforms_path = args.transforms
    if not os.path.isfile(ply_path):
        raise SystemExit(f"PLY not found: {ply_path}")
    if not os.path.isfile(transforms_path):
        raise SystemExit(f"transforms JSON not found: {transforms_path}")

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"Loading splat   : {ply_path}")
    means, scales_raw, quats, opacities_raw, sh0, shN = load_gaussians_ply(ply_path, device="cpu")
    print(f"  Gaussians: {means.shape[0]}")

    print(f"Loading cameras : {transforms_path}")
    cameras = load_cameras(transforms_path, load_image=False, device="cpu")
    print(f"  Cameras  : {len(cameras)}")

    # ── Source world-up ──────────────────────────────────────────────────────
    if args.src_frame == "auto" and args.world_up == "auto":
        try:
            src_up: WorldUp = detect_scene_up(cameras)
            print(f"  Detected scene up axis: {src_up!r} (override with --world-up)")
        except ValueError as exc:
            # Degenerate camera set (e.g. 2 nearly-aligned views): guessing the
            # up axis would rotate the whole scene wrongly, so stop and let the
            # user state it rather than silently laying the subject on its side.
            print(f"  ⚠️  Cannot infer the scene up axis: {exc}")
            print("     Re-run with --world-up y (this project's scenes are Y-up).")
            return
    elif args.src_frame == "opencv":
        src_up = "y"  # OpenCV world: Y is the vertical axis (down)
        print(f"  Source frame: opencv (Y-down canonical)")
    else:
        src_up = args.world_up
        print(f"  Source frame: {args.src_frame}, world up: {src_up}")

    model_to_world = _parse_model_to_world(args.model_to_world)

    # ── Convert ──────────────────────────────────────────────────────────────
    result = convert(
        means, scales_raw, quats, cameras,
        target=args.target,
        src_world_up=src_up,
        src_frame=args.src_frame,
        model_to_world=model_to_world,
        device=args.device,
    )
    print("\n── Conversion ──")
    print(f"  Target   : {args.target}")
    print(f"  Src up   : {src_up}")
    print(f"  Dst up   : {'z' if args.target == 'splatshop' else src_up}")
    print(f"  R_w      :\n{result['R_w'].numpy()}")
    print(f"  Axis map : {result['axis_map']}")
    print(f"  Splats   : {result['means'].shape[0]}   Cameras: {len(result['cameras'])}")

    # ── Export ───────────────────────────────────────────────────────────────
    export_paths = {}
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ply_out = out_dir / f"splat_{args.target}.ply"
        json_out = out_dir / f"cameras_{args.target}.json"
        export_ply(str(ply_out), result["means"], result["scales"], result["quats"],
                   opacities_raw, sh0, shN)
        export_cameras_json(str(json_out), result["cameras"], result["intrinsics"],
                            convention=args.target)
        export_paths["ply"] = str(ply_out)
        export_paths["json"] = str(json_out)
        print(f"\n── Export ──\n  PLY : {ply_out}\n  JSON: {json_out}")

    # ── Serve (optional) ─────────────────────────────────────────────────────
    if args.serve:
        _serve(args, ply_path, transforms_path, export_paths, src_up)


def _serve(args, ply_path, transforms_path, export_paths, src_up):
    project_root = Path(__file__).resolve().parent
    html_rel = "static/cam_viewer/cam_viewer.html"
    html_path = project_root / html_rel
    if not html_path.is_file():
        raise SystemExit(f"cam_viewer.html not found at {html_path}")

    def to_url_path(p):
        pp = Path(p)
        if not pp.is_absolute():
            pp = (project_root / pp)
        pp = pp.resolve()
        try:
            rel = pp.relative_to(project_root)
        except ValueError:
            raise SystemExit(f"path must be inside project root ({project_root}), got {p}")
        return "/" + rel.as_posix()

    base_url = to_url_path(os.path.dirname(os.path.abspath(transforms_path)))
    if not base_url.endswith("/"):
        base_url += "/"

    # Prefer the exported (converted) splat if produced, else the input splat.
    splat_arg = export_paths.get("ply", ply_path)

    qs = {
        "base": base_url,
        "splat": to_url_path(splat_arg),
        "coord": args.target,
        "worldup": src_up,
    }
    port = find_free_port(args.port)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(project_root), **kw)
        def log_message(self, fmt, *a):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/{html_rel}?{urlencode(qs, quote_via=quote)}"
    print(f"\n── Viewer ──\nServing {project_root} on http://127.0.0.1:{port}")
    print(f"Viewer URL: {url}")
    for k, v in qs.items():
        print(f"  {k}: {v}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
