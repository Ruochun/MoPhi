"""Internal visualization utilities shared by all MoPhi visualizer backends.

This module is not part of the public API.  It is imported by
:mod:`mophi.opengl_visualizer` and :mod:`mophi.omniverse_visualizer`.
"""

from __future__ import annotations

import numpy as np


def _rotate_batch(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a set of vectors by a batch of quaternions (vectorised Rodrigues).

    Uses the double-cross-product form of Rodrigues' rotation formula::

        v' = v + 2w (q × v) + 2 (q × (q × v))

    where ``q`` is the vector part ``[qx, qy, qz]`` and ``w = qw`` is the
    scalar part of the xyzw quaternion.

    Args:
        q_xyzw: ``(N, 4)`` float array of unit xyzw quaternions.
        v:      ``(K, 3)`` float array of vectors to rotate.

    Returns:
        ``(N, K, 3)`` float32 array of rotated vectors — one ``(K, 3)``
        block per quaternion.
    """
    q_vec = q_xyzw[:, :3].astype(np.float64)  # (N, 3)
    w = q_xyzw[:, 3].astype(np.float64)  # (N,)

    q_b = q_vec[:, np.newaxis, :]  # (N, 1, 3)
    w_b = w[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    v_b = v[np.newaxis, :, :].astype(np.float64)  # (1, K, 3)

    qxv = np.cross(q_b, v_b)  # (N, K, 3)
    qxqxv = np.cross(q_b, qxv)  # (N, K, 3)
    return (v_b + 2.0 * w_b * qxv + 2.0 * qxqxv).astype(np.float32)
