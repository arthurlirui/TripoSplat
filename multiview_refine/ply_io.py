"""PLY I/O for 3D Gaussian Splatting.

Provides a single, shared implementation of PLY reading and writing used
across ``run_pipeline.py``, ``render_turntable.py``, ``rigging.py``, and
``prune_floaters.py``.

.. note::

    The core library modules ``triposplat.py`` (:class:`Gaussian.save_ply`)
    and ``gaussian_model.py`` (:class:`RefineableGaussians.save_ply`) retain
    their own manual binary PLY writers to avoid pulling in ``plyfile`` as a
    hard dependency of the core library.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def save_gaussians_ply(
    path: str,
    means: torch.Tensor,
    scales_raw: torch.Tensor,
    quats: torch.Tensor,
    opacities_raw: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor | None = None,
):
    """Save Gaussians to a 3DGS-compatible PLY file (raw/pre-activation values).

    Args:
        path: Output ``.ply`` file path.
        means: (N, 3) world-space centres.
        scales_raw: (N, 3) log-space scales.
        quats: (N, 4) quaternions (will be normalised before writing).
        opacities_raw: (N,) or (N, 1) logit opacities.
        sh0: (N, 1, 3) SH degree-0 DC coefficients.
        shN: (N, K-1, 3) higher-order SH, or ``None`` for degree 0.
    """
    N = means.shape[0]
    means_np = means.detach().cpu().numpy()
    sh0_np = sh0.detach().cpu().numpy()  # (N, 1, 3)
    f_dc = sh0_np.transpose(0, 2, 1).reshape(N, -1)  # (N, 3)

    if shN is not None and shN.numel() > 0:
        shN_np = shN.detach().cpu().numpy()
        f_rest = shN_np.transpose(0, 2, 1).reshape(N, -1)
    else:
        f_rest = np.zeros((N, 0), dtype=np.float32)

    normals = np.zeros_like(means_np)
    opacities_np = opacities_raw.detach().cpu().numpy().reshape(-1, 1)
    scales_np = scales_raw.detach().cpu().numpy()
    rotations_np = F.normalize(quats, dim=-1).detach().cpu().numpy()

    data = np.concatenate(
        [means_np, normals, f_dc, f_rest, opacities_np, scales_np, rotations_np],
        axis=1,
    )

    names = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(f_dc.shape[1]):
        names.append(f"f_dc_{i}")
    for i in range(f_rest.shape[1]):
        names.append(f"f_rest_{i}")
    names.append("opacity")
    for i in range(3):
        names.append(f"scale_{i}")
    for i in range(4):
        names.append(f"rot_{i}")

    dtype = [(n, "f4") for n in names]
    elements = np.empty(N, dtype=dtype)
    elements[:] = list(map(tuple, data))

    el = PlyElement.describe(elements, "vertex")
    PlyData([el], byte_order="<", text=False).write(path)


def load_gaussians_ply(
    path: str, device: str | torch.device = "cuda"
) -> tuple[torch.Tensor, ...]:
    """Load Gaussians from a 3DGS-compatible PLY file.

    Returns a 6-tuple: ``(means, scales_raw, quats, opacities_raw, sh0, shN)``.
    All tensors are placed on *device*.

    Attributes read: x, y, z, f_dc_0..2, [f_rest_*], opacity, scale_0..2,
    rot_0..3.  Normal vectors (nx, ny, nz) are ignored.
    """
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    data = np.stack([v[name] for name in v.data.dtype.names], axis=1)
    names = list(v.data.dtype.names)

    def col(name):
        return data[:, names.index(name)]

    means = np.stack([col("x"), col("y"), col("z")], axis=-1)
    f_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=-1)
    sh0 = f_dc[:, None, :]  # (N, 1, 3)

    f_rest_names = sorted([n for n in names if n.startswith("f_rest_")])
    if f_rest_names:
        f_rest = np.stack([col(n) for n in f_rest_names], axis=-1)
        shN = f_rest.reshape(-1, len(f_rest_names) // 3, 3)
    else:
        shN = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    opacities = col("opacity")
    scales = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=-1)
    quats = np.stack(
        [col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=-1
    )

    dev = torch.device(device)
    return (
        torch.from_numpy(means).float().to(dev),
        torch.from_numpy(scales).float().to(dev),
        torch.from_numpy(quats).float().to(dev),
        torch.from_numpy(opacities).float().to(dev),
        torch.from_numpy(sh0).float().to(dev),
        torch.from_numpy(shN).float().to(dev),
    )
