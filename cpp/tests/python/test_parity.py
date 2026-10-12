"""Numerical parity test: C++ pipeline vs Python pipeline.

Loads the same TorchScript ``.pt`` weights and runs both the Python
``TripoSplatPipeline`` and the C++ ``TripoSplatPipeline`` on an identical
input image with the same seed, then compares the output Gaussians
(means, scales, quats, opacities, sh0) to fp16 precision.

Requirements
------------
* TripoSplat conda env (torch, safetensors, torchvision).
* TorchScript weights exported via ``scripts/export_torchscript.py``.
* The C++ ``run_example`` executable built under ``cpp/build/Release/``.

Usage
-----
::

    conda activate rtg132
    python cpp/tests/python/test_parity.py \
        --weights cpp/weights \
        --cpp-exe cpp/build/Release/run_example.exe \
        --image input_rgba.png \
        --seed 42 --steps 20 --num-gaussians 262144

The C++ executable writes ``gaussians.ply``; this script loads that PLY and
compares against the Python pipeline output on the same input.
"""

import argparse
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from multiview_refine.ply_io import load_gaussians_ply


def run_python_pipeline(image_path, weights_dir, seed, steps, num_gaussians, device):
    """Run the Python TripoSplatPipeline and return Gaussian attributes."""
    from triposplat import TripoSplatPipeline

    pipe = TripoSplatPipeline(
        ckpt_path=os.path.join(weights_dir, "triposplat_fp16.safetensors"),
        # The Python pipeline loads from the original safetensors, not the
        # traced .pt files. If only traced weights are available, this will
        # fail — in that case, use the traced-module Python path instead.
        device=device,
    )
    gaussian, _ = pipe.run(
        image_path,
        num_gaussians=num_gaussians,
        remove_background=True,
        show_progress=False,
        seed=seed,
        steps=steps,
    )
    return {
        "means": gaussian.get_xyz.detach().float().cpu(),
        "scales": gaussian.get_scaling.detach().float().cpu(),
        "quats": (gaussian._rotation + gaussian.rots_bias[None, :]).detach().float().cpu(),
        "opacities": gaussian.get_opacity.detach().squeeze(-1).float().cpu(),
        "sh0": gaussian._features_dc.detach().float().cpu(),
    }


def run_cpp_pipeline(cpp_exe, image_path, weights_dir, output_dir,
                     seed, steps, num_gaussians, device):
    """Run the C++ run_example executable and load the output PLY."""
    import subprocess

    cmd = [
        cpp_exe, image_path, output_dir,
        "--weights", weights_dir,
        "--device", device,
        "--seed", str(seed),
        "--steps", str(steps),
        "--num-gaussians", str(num_gaussians),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ executable failed (exit {result.returncode}):\n{result.stderr}"
        )
    ply_path = os.path.join(output_dir, "gaussians.ply")
    if not os.path.isfile(ply_path):
        raise RuntimeError(f"C++ pipeline did not produce {ply_path}")
    means, scales_raw, quats, opac_raw, sh0, shN = load_gaussians_ply(ply_path, "cpu")
    return {
        "means": means.float(),
        "scales": scales_raw.float().exp(),  # activate for comparison
        "quats": quats.float(),
        "opacities": opac_raw.float().sigmoid(),
        "sh0": sh0.float(),
    }


def compare(py_attrs, cpp_attrs, tol=1e-2):
    """Compare Python and C++ Gaussian attributes. Returns (passed, report)."""
    report = []
    all_pass = True
    for name in ("means", "scales", "quats", "opacities", "sh0"):
        py_t = py_attrs[name]
        cpp_t = cpp_attrs[name]
        if py_t.shape != cpp_t.shape:
            report.append(f"  {name}: SHAPE MISMATCH py={tuple(py_t.shape)} cpp={tuple(cpp_t.shape)}")
            all_pass = False
            continue
        max_diff = (py_t - cpp_t).abs().max().item()
        mean_diff = (py_t - cpp_t).abs().mean().item()
        passed = max_diff < tol
        status = "PASS" if passed else "FAIL"
        report.append(f"  {name}: {status}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  (tol={tol})")
        if not passed:
            all_pass = False
    return all_pass, "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="C++ vs Python TripoSplat parity test")
    parser.add_argument("--weights", default="cpp/weights",
                        help="Directory containing TorchScript .pt / safetensors weights")
    parser.add_argument("--cpp-exe", default="cpp/build/Release/run_example.exe",
                        help="Path to the C++ run_example executable")
    parser.add_argument("--image", required=True,
                        help="Input RGBA image (background pre-removed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-gaussians", type=int, default=262144)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tol", type=float, default=1e-2,
                        help="Max absolute difference tolerance (fp16 precision)")
    args = parser.parse_args()

    print("=" * 60)
    print("TripoSplat C++ vs Python Parity Test")
    print("=" * 60)
    print(f"  Image:          {args.image}")
    print(f"  Weights:        {args.weights}")
    print(f"  C++ executable: {args.cpp_exe}")
    print(f"  Seed:           {args.seed}, Steps: {args.steps}, N: {args.num_gaussians}")
    print(f"  Tolerance:      {args.tol}")
    print()

    # Run Python pipeline.
    print("[1/2] Running Python pipeline...")
    py_attrs = run_python_pipeline(
        args.image, args.weights, args.seed, args.steps,
        args.num_gaussians, args.device)
    print(f"  Python: {py_attrs['means'].shape[0]} Gaussians")

    # Run C++ pipeline.
    print("[2/2] Running C++ pipeline...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cpp_attrs = run_cpp_pipeline(
            args.cpp_exe, args.image, args.weights, tmpdir,
            args.seed, args.steps, args.num_gaussians, args.device)
        print(f"  C++:    {cpp_attrs['means'].shape[0]} Gaussians")

    # Compare.
    print("\n" + "=" * 60)
    print("Comparison results:")
    print("=" * 60)
    if py_attrs["means"].shape[0] != cpp_attrs["means"].shape[0]:
        print(f"  WARNING: Gaussian count differs (py={py_attrs['means'].shape[0]}, "
              f"cpp={cpp_attrs['means'].shape[0]}). Comparison may be unreliable.")

    passed, report = compare(py_attrs, cpp_attrs, tol=args.tol)
    print(report)
    print()
    if passed:
        print("RESULT: PASS — C++ output matches Python within tolerance.")
        return 0
    else:
        print("RESULT: FAIL — differences exceed tolerance.")
        print("  Note: PIL LANCZOS vs libtorch bilinear image resampling introduces")
        print("  sub-pixel input variance, so exact bit-equality is not expected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
