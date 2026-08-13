"""Linear sRGB <-> OKLab <-> OKLCH.

OKLab (Bjorn Ottosson, 2020) is a perceptually uniform color space
designed so that a fixed-size step anywhere in the space looks like a
similar-sized perceptual change - unlike HSL/HSV, where the same numeric
saturation/lightness delta can look wildly different depending on hue.
That property is exactly what makes it a better basis for hue-preserving
saturation/lightness grading (color wheels, HSL-by-hue-range) than naive
RGB or HSV math, which is why the color-grading tools in this app are
built on it rather than reimplementing ad-hoc HSV tweaks per tool.

OKLCH is simply OKLab in polar (cylindrical) form: C is chroma (distance
from the neutral axis), H is hue angle in degrees.

Matrices are the values published in Ottosson's reference implementation.
Defined relative to *linear sRGB primaries* - color_science/spaces.py
routes other RGB spaces through CIE XYZ to get here, since OKLab isn't
natively defined for arbitrary primaries.
"""

import numpy as np

_LINEAR_SRGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float64)

_LMS_TO_OKLAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float64)

_OKLAB_TO_LMS = np.linalg.inv(_LMS_TO_OKLAB)
_LMS_TO_LINEAR_SRGB = np.linalg.inv(_LINEAR_SRGB_TO_LMS)


def linear_srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) linear-light sRGB, any range (not clamped to [0,1] -
    OKLab is well-defined for out-of-gamut/HDR values, which matters since
    this pipeline carries unclamped scene-linear data)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = rgb @ _LINEAR_SRGB_TO_LMS.T
    # Cube root of a negative LMS value (possible for saturated/out-of-
    # gamut colors) must stay real and sign-preserving - np.cbrt does this
    # correctly, unlike lms ** (1/3) which produces NaN for negative input.
    lms_ = np.cbrt(lms)
    return lms_ @ _LMS_TO_OKLAB.T


def oklab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = lab @ _OKLAB_TO_LMS.T
    lms = lms_ ** 3
    return lms @ _LMS_TO_LINEAR_SRGB.T


def oklab_to_oklch(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    H = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, H], axis=-1)


def oklch_to_oklab(lch: np.ndarray) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float64)
    L, C, H = lch[..., 0], lch[..., 1], lch[..., 2]
    h_rad = np.radians(H)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)
    return np.stack([L, a, b], axis=-1)


def linear_srgb_to_oklch(rgb: np.ndarray) -> np.ndarray:
    return oklab_to_oklch(linear_srgb_to_oklab(rgb))


def oklch_to_linear_srgb(lch: np.ndarray) -> np.ndarray:
    return oklab_to_linear_srgb(oklch_to_oklab(lch))
