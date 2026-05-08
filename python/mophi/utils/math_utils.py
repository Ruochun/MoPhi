"""mophi.utils.math_utils — General-purpose math helpers for MoPhi demos and couplers.

These helpers are solver-agnostic: they operate on standard NumPy arrays and
can be called from any MoPhi coupler or directly from user scripts and demos.
"""

from __future__ import annotations

import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions given as [x, y, z, w] NumPy arrays.

    Returns the product q1 * q2 as a [x, y, z, w] float32 array.
    Both Warp and DEME use the [x, y, z, w] quaternion convention.

    Args:
        q1: First quaternion as a (4,) array [x, y, z, w].
        q2: Second quaternion as a (4,) array [x, y, z, w].

    Returns:
        Product q1 * q2 as a float32 (4,) array [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )
