"""Preload the cached gsplat CUDA extension to bypass broken JIT on torch 2.12+.

gsplat 1.5.3's ``cuda/_backend.py`` calls ``_jit_compile`` with a signature
that torch>=2.7 no longer accepts, so even when a compiled ``gsplat_cuda.pyd``
sits in the torch extension cache, ``load_extension`` raises ``TypeError``
before reaching the cache-load fallback (which only catches ``OSError``).

This module short-circuits that by directly loading the cached ``.pyd`` and
registering it as ``gsplat.csrc`` before gsplat's ``_backend`` runs, so the
``from gsplat import csrc as _C`` line succeeds.

Import it (or set ``PYTHONSTARTUP``) *before* importing anything from
``gsplat`` or ``multiview_refine``.
"""

import importlib.util
import os
import sys


def preload_gsplat_cuda(pyd_path=None):
    """Load the cached gsplat_cuda extension and register it as gsplat.csrc."""
    import gsplat  # noqa: F401  (ensure package exists, but don't trigger _C)

    if pyd_path is None:
        # Default: the torch 2.12 / CUDA 13.2 cache built on 2025-08-12.
        pyd_path = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "torch_extensions", "torch_extensions", "Cache",
            "py312_cu132", "gsplat_cuda", "gsplat_cuda.pyd",
        )

    if not os.path.exists(pyd_path):
        raise FileNotFoundError(
            f"Cached gsplat_cuda.pyd not found at {pyd_path}. The render env "
            "needs a precompiled gsplat CUDA extension; see README troubleshooting."
        )

    # The .pyd exports PyInit_gsplat_cuda, so the spec name must match that.
    spec = importlib.util.spec_from_file_location("gsplat_cuda", pyd_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build spec for {pyd_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gsplat.csrc = mod
    sys.modules["gsplat.csrc"] = mod
    return mod


# Auto-preload on import so a plain ``import _gsplat_preload`` is enough.
if os.environ.get("GSPLAT_PRELOAD", "1") != "0":
    try:
        preload_gsplat_cuda()
    except Exception as _e:  # pragma: no cover - environment-dependent
        # Stay silent on failure; the normal (broken) path will surface a
        # clearer error if the caller actually needs rasterization.
        pass
