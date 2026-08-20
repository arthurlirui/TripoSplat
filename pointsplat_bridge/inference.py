"""Run PointSplat inference in a dedicated conda env and locate its output PLY.

PointSplat is driven by Hydra config overrides. The two shipped experiments are
``pointsplat_renbody`` (DNA-Rendering) and ``pointsplat_thuman`` (THuman2.0).
With ``test.save_gaussian=true`` the model wrapper writes one PLY per scene to
``<output_dir>/gaussians/<scene>_<timestep>.ply`` (standard 3DGS layout), which
:func:`multiview_refine.ply_io.load_gaussians_ply` can read directly.

This module never imports torch or PointSplat itself — it only shells out to a
conda env that has them, so it stays importable from TripoSplat's own (different)
runtime environment.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Repository root = parent of this package's directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
POINTSPLAT_DIR = REPO_ROOT / "third_party" / "pointsplat"
DEFAULT_CKPT = REPO_ROOT / "pretrained" / "pointsplat" / "pointsplat_mixed.pt"


@dataclass
class PointSplatResult:
    """Outcome of a PointSplat reconstruction run."""

    ply_path: Path
    """Path to the single output PLY (one scene expected)."""

    output_dir: Path
    """PointSplat ``output_dir`` (contains gaussians/, images/, ...)."""

    experiment: str
    """Hydra experiment used (``pointsplat_renbody`` / ``pointsplat_thuman``)."""


def _conda_run_python(env: str) -> list[str]:
    """Build a ``conda run -n <env> python`` prefix that works under Git Bash."""
    # ``conda run`` is used so PATH/activation matches a manual ``conda activate``.
    return ["conda", "run", "--no-capture-output", "-n", env, "python"]


def _find_output_ply(output_dir: Path) -> Path:
    """Locate the (single) Gaussian PLY PointSplat wrote under *output_dir*.

    PointSplat saves to ``<output_dir>/gaussians/<scene>_<timestep>.ply``. For a
    single-scene example dataset exactly one file is expected; if several are
    present (multi-scene dataset) the lexicographically first is returned and a
    note is printed.
    """
    gaussians_dir = output_dir / "gaussians"
    if not gaussians_dir.is_dir():
        raise FileNotFoundError(
            f"PointSplat did not write a gaussians/ directory under {output_dir}. "
            "Did the run finish, and was test.save_gaussian=true set?"
        )
    plys = sorted(glob_(gaussians_dir, "*.ply"))
    if not plys:
        raise FileNotFoundError(
            f"No .ply found in {gaussians_dir}. Check the PointSplat log for errors."
        )
    if len(plys) > 1:
        print(f"  [bridge] {len(plys)} PLYs found; using {plys[0].name} (first scene).")
    return plys[0]


def glob_(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern))


def reconstruct_gaussians(
    dataset_root: str | os.PathLike,
    *,
    experiment: str = "pointsplat_thuman",
    conda_env: str = "pointsplat",
    checkpoint: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
    extra_overrides: list[str] | None = None,
    cwd: str | os.PathLike | None = None,
    env: dict | None = None,
) -> PointSplatResult:
    """Run PointSplat feed-forward reconstruction and return the output PLY path.

    Parameters
    ----------
    dataset_root
        Path passed to PointSplat as ``dataset.roots``. Must already be laid out
        per the chosen experiment (THuman2.0: ``val/<scene>_<view>/{img,mask,parm}``;
        DNA-Rendering: ``scenes/<scene>/transforms.json`` + ``masks/<scene>/fmasks``).
    experiment
        Hydra experiment: ``pointsplat_thuman`` or ``pointsplat_renbody``. Must
        match the layout of *dataset_root*.
    conda_env
        Conda environment that has PointSplat's dependencies installed
        (torch 2.4, gsplat 1.5.x, xformers 0.0.27, pytorch_lightning 2.4, hydra).
        Created per ``pointsplat_bridge/README.md``.
    checkpoint
        Path to ``pointsplat_mixed.pt``. Defaults to
        ``pretrained/pointsplat/pointsplat_mixed.pt`` under the repo root.
    output_dir
        Where PointSplat writes its outputs (gaussians/, images/). Defaults to a
        ``pointsplat_bridge/runs/<timestamp>`` directory.
    extra_overrides
        Additional Hydra override strings, e.g.
        ``["dataset.view_sampler.num_context_views=4"]``.
    cwd
        Working directory for the subprocess. Defaults to the vendored
        ``third_party/pointsplat`` so ``python -m src.main`` resolves.
    env
        Optional extra environment variables for the subprocess.

    Returns
    -------
    PointSplatResult
        With ``ply_path`` pointing at the generated Gaussian PLY.

    Notes
    -----
    The call forces ``test.save_gaussian=true`` (the renbody experiment sets this
    already, the thuman one does not) so a PLY is always produced. It also turns
    off score computation and image saving to keep the run lightweight.
    """
    ckpt = Path(checkpoint) if checkpoint is not None else DEFAULT_CKPT
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"PointSplat checkpoint not found at {ckpt}. Download with:\n"
            f"  hf download Yujie0012/PointSplat_pretrained_weights "
            f"pointsplat_mixed.pt --local-dir {ckpt.parent}"
        )

    work_dir = Path(cwd) if cwd is not None else POINTSPLAT_DIR
    if not (work_dir / "src" / "main.py").is_file():
        raise FileNotFoundError(
            f"{work_dir}/src/main.py not found. Is third_party/pointsplat present?"
        )

    if output_dir is None:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = REPO_ROOT / "pointsplat_bridge" / "runs" / ts
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hydra overrides. `output_dir` is read by main.py (not the hydra run dir).
    overrides = [
        f"+experiment={experiment}",
        f"output_dir={output_dir.as_posix()}",
        f"dataset.roots=['{Path(dataset_root).as_posix()}']",
        f"checkpointing.pretrained_encoder={ckpt.as_posix()}",
        # Always emit a Gaussian PLY; skip heavy extras we don't need.
        "test.save_gaussian=true",
        "test.compute_scores=false",
        "test.save_image=false",
        "test.save_gt_image=false",
        "test.save_input_images=false",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    cmd = _conda_run_python(conda_env) + ["-m", "src.main", *overrides]

    # `conda run` needs to find the `conda` executable; inherit the parent env
    # but let the caller add/override variables.
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    print(f"[bridge] Running PointSplat ({conda_env}) …")
    print(f"[bridge]   cwd   : {work_dir}")
    print(f"[bridge]   out   : {output_dir}")
    print(f"[bridge]   exp   : {experiment}")
    print(f"[bridge]   data  : {dataset_root}")
    print(f"[bridge]   ckpt  : {ckpt}")
    print(f"[bridge]   cmd   : {' '.join(cmd)}")

    completed = subprocess.run(
        cmd,
        cwd=str(work_dir),
        env=run_env,
        # Stream PointSplat output through so progress/errors are visible.
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PointSplat exited with code {completed.returncode}. "
            "See the log above; common causes: wrong conda env name, missing "
            "checkpoint, or a dataset layout that doesn't match the experiment."
        )

    ply_path = _find_output_ply(output_dir)
    print(f"[bridge] Output PLY: {ply_path} ({ply_path.stat().st_size // 1024} KB)")
    return PointSplatResult(
        ply_path=ply_path,
        output_dir=output_dir,
        experiment=experiment,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run PointSplat and print the PLY path.")
    ap.add_argument("dataset_root", help="Dataset root passed as dataset.roots")
    ap.add_argument(
        "--experiment",
        default="pointsplat_thuman",
        choices=["pointsplat_thuman", "pointsplat_renbody"],
    )
    ap.add_argument("--conda-env", default="pointsplat")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    res = reconstruct_gaussians(
        args.dataset_root,
        experiment=args.experiment,
        conda_env=args.conda_env,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    )
    print(res.ply_path)
