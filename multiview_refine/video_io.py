"""Video I/O utilities with automatic codec fallback.

Provides a single :func:`create_video_writer` used by ``render_turntable.py``,
``run_pipeline.py``, and ``animation.py`` to ensure robust video export across
systems where ``mp4v`` may not be available.
"""

from __future__ import annotations

import cv2


def create_video_writer(
    output_path: str,
    fourcc_str: str,
    fps: float,
    size: tuple[int, int],
) -> tuple[cv2.VideoWriter, str]:
    """Create a :class:`cv2.VideoWriter` with automatic codec fallback.

    Tries the requested *fourcc_str* (e.g. ``"mp4v"``).  If the writer
    fails to open, falls back to ``XVID`` and ``.avi`` extension.

    Args:
        output_path: Desired output path (e.g. ``"result.mp4"``).
        fourcc_str: FourCC code string (e.g. ``"mp4v"``, ``"XVID"``).
        fps: Frames per second.
        size: ``(width, height)`` tuple.

    Returns:
        ``(writer, actual_path)`` where *actual_path* may differ from
        *output_path* if a fallback was applied.
    """
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    if not writer.isOpened():
        print(f"WARNING: VideoWriter failed with codec '{fourcc_str}', "
              f"falling back to XVID + .avi ...")
        fallback_path = output_path.rsplit(".", 1)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(fallback_path, fourcc, fps, size)
        return writer, fallback_path

    return writer, output_path
