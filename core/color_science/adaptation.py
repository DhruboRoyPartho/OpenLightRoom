"""Chromatic adaptation (Bradford transform).

Needed whenever two color spaces are defined relative to different
reference whites - e.g. ProPhoto RGB (D50) vs sRGB (D65). Converting
between them by naively swapping RGB<->XYZ matrices without adapting the
white point produces a visible color cast; the Bradford transform is the
standard, widely-used correction (used by ICC profiles and virtually every
production color pipeline).
"""

import numpy as np

_BRADFORD = np.array([
    [0.8951000, 0.2664000, -0.1614000],
    [-0.7502000, 1.7135000, 0.0367000],
    [0.0389000, -0.0685000, 1.0296000],
], dtype=np.float64)

_BRADFORD_INV = np.linalg.inv(_BRADFORD)


def chromatic_adaptation_matrix(src_white_xyz: np.ndarray, dst_white_xyz: np.ndarray) -> np.ndarray:
    """3x3 matrix that maps XYZ values adapted to src_white_xyz onto the
    equivalent XYZ values adapted to dst_white_xyz."""
    if np.allclose(src_white_xyz, dst_white_xyz):
        return np.eye(3)

    src_cone = _BRADFORD @ src_white_xyz
    dst_cone = _BRADFORD @ dst_white_xyz
    scale = np.diag(dst_cone / src_cone)
    return _BRADFORD_INV @ scale @ _BRADFORD
