"""Bridge: run PointSplat (zju3dv, ECCV 2026) as a feed-forward multi-view
human-Gaussian reconstruction front-end, then hand its output to TripoSplat's
existing SMPL-X skinning pipeline.

Why a subprocess bridge
-----------------------
PointSplat pins Python 3.10 + torch 2.4 + xformers 0.0.27 + numpy 1.24.4 +
pytorch_lightning 2.4 + gsplat 1.5.x, which conflicts with TripoSplat's runtime
(generally torch 2.6 / gsplat 0.1.0 on Python 3.11). Loading both in one process
is impractical, so this bridge drives PointSplat as a *child process* in a
dedicated conda env and exchanges data through the PLY file PointSplat writes
to disk (``test.save_gaussian=true``). The PLY is standard 3DGS format and is
read back by ``multiview_refine.ply_io.load_gaussians_ply``.

Public entry point: :func:`reconstruct_gaussians`.
"""

from .inference import reconstruct_gaussians, PointSplatResult

__all__ = ["reconstruct_gaussians", "PointSplatResult"]
