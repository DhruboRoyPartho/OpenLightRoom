"""RGB primary chromaticities and white points, and the standard
colorimetric procedure for deriving RGB<->CIE XYZ matrices from them.

Adding a new RGB color space is a matter of adding one Primaries entry
here - build_rgb_to_xyz() derives a correct matrix from first principles
rather than requiring a hand-copied matrix per space, which is both less
error-prone and matches the "additional color spaces can be added later"
requirement.
"""

from dataclasses import dataclass
import numpy as np

# Standard illuminant white points (CIE 1931 2-degree observer), as (x, y)
# chromaticity coordinates.
ILLUMINANT_D65 = (0.31270, 0.32900)
ILLUMINANT_D50 = (0.34570, 0.35850)


@dataclass(frozen=True)
class Primaries:
    """Chromaticity coordinates (x, y) of the red/green/blue primaries and
    the reference white point of an RGB color space."""
    red: tuple
    green: tuple
    blue: tuple
    white: tuple


SRGB_PRIMARIES = Primaries(red=(0.6400, 0.3300), green=(0.3000, 0.6000), blue=(0.1500, 0.0600), white=ILLUMINANT_D65)
REC709_PRIMARIES = SRGB_PRIMARIES  # identical primaries and white point to sRGB
DISPLAY_P3_PRIMARIES = Primaries(red=(0.6800, 0.3200), green=(0.2650, 0.6900), blue=(0.1500, 0.0600), white=ILLUMINANT_D65)
ADOBE_RGB_PRIMARIES = Primaries(red=(0.6400, 0.3300), green=(0.2100, 0.7100), blue=(0.1500, 0.0600), white=ILLUMINANT_D65)
PROPHOTO_RGB_PRIMARIES = Primaries(red=(0.7347, 0.2653), green=(0.1596, 0.8404), blue=(0.0366, 0.0001), white=ILLUMINANT_D50)


def _xy_to_xyz(xy) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y], dtype=np.float64)


def build_rgb_to_xyz(primaries: Primaries) -> np.ndarray:
    """Standard derivation of the 3x3 linear-RGB -> CIE XYZ matrix from a
    set of primary chromaticities and a reference white point:
    1. Each primary's XYZ at unit luminance forms a column of M.
    2. Solve for the per-column scale factors that make M @ scale equal
       the white point's XYZ (i.e. R=G=B=1 must map to the white point).
    3. The final matrix is M with those columns scaled.
    """
    m = np.column_stack([
        _xy_to_xyz(primaries.red),
        _xy_to_xyz(primaries.green),
        _xy_to_xyz(primaries.blue),
    ])
    white_xyz = _xy_to_xyz(primaries.white)
    scale = np.linalg.solve(m, white_xyz)
    return m * scale[np.newaxis, :]


def build_xyz_to_rgb(primaries: Primaries) -> np.ndarray:
    return np.linalg.inv(build_rgb_to_xyz(primaries))
