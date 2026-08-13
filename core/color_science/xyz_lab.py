"""CIE XYZ <-> CIE L*a*b* (1976), the classical perceptually-motivated
color space. Reference white defaults to D65 (matching sRGB/Display P3/
Rec.709's native white point) but is parameterizable for D50-native spaces
like ProPhoto RGB.
"""

import numpy as np

# D65 reference white in CIE XYZ (Y normalized to 1.0).
D65_WHITE_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
D50_WHITE_XYZ = np.array([0.96422, 1.00000, 0.82521], dtype=np.float64)

_DELTA = 6.0 / 29.0


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4.0 / 29.0)


def _f_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = D65_WHITE_XYZ) -> np.ndarray:
    """xyz: (..., 3) array. Returns Lab with L in [0, 100], a/b roughly
    [-128, 127] for in-gamut colors (unbounded for out-of-gamut input)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    fx = _f(xyz[..., 0] / white[0])
    fy = _f(xyz[..., 1] / white[1])
    fz = _f(xyz[..., 2] / white[2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray, white: np.ndarray = D65_WHITE_XYZ) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    x = white[0] * _f_inv(fx)
    y = white[1] * _f_inv(fy)
    z = white[2] * _f_inv(fz)
    return np.stack([x, y, z], axis=-1)
