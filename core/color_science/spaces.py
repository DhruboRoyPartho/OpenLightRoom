"""Named RGB color spaces (primaries + transfer function bundled together)
and conversion between them, routed through CIE XYZ - the standard
interchange space - with Bradford chromatic adaptation when the two
spaces' reference whites differ (e.g. ProPhoto RGB's D50 vs sRGB's D65).

This is the single place color-space identity and conversion logic lives;
core/processing/color_space.py (the render pipeline's linear<->display
transform) delegates to this module instead of redefining the sRGB curve
a second time.
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np

from core.color_science import primaries as prim
from core.color_science import transfer_functions as tf
from core.color_science.adaptation import chromatic_adaptation_matrix


@dataclass(frozen=True)
class ColorSpace:
    name: str
    primaries: prim.Primaries
    eotf: Callable[[np.ndarray], np.ndarray]   # encoded -> linear
    oetf: Callable[[np.ndarray], np.ndarray]   # linear -> encoded

    @property
    def white_xyz(self) -> np.ndarray:
        x, y = self.primaries.white
        return np.array([x / y, 1.0, (1 - x - y) / y], dtype=np.float64)


SRGB = ColorSpace("sRGB", prim.SRGB_PRIMARIES, tf.srgb_eotf, tf.srgb_oetf)
REC709 = ColorSpace("Rec.709", prim.REC709_PRIMARIES, tf.rec709_eotf, tf.rec709_oetf)
DISPLAY_P3 = ColorSpace("Display P3", prim.DISPLAY_P3_PRIMARIES, tf.srgb_eotf, tf.srgb_oetf)
ADOBE_RGB = ColorSpace("Adobe RGB", prim.ADOBE_RGB_PRIMARIES, tf.adobergb_eotf, tf.adobergb_oetf)
PROPHOTO_RGB = ColorSpace("ProPhoto RGB", prim.PROPHOTO_RGB_PRIMARIES, tf.prophoto_eotf, tf.prophoto_oetf)

REGISTRY = {cs.name: cs for cs in (SRGB, REC709, DISPLAY_P3, ADOBE_RGB, PROPHOTO_RGB)}


def get(name: str) -> ColorSpace:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown color space '{name}'. Known: {sorted(REGISTRY)}")


def to_xyz(encoded_rgb: np.ndarray, space: ColorSpace) -> np.ndarray:
    """Display-referred RGB in `space` -> CIE XYZ (adapted to that space's
    own native white point)."""
    linear = space.eotf(encoded_rgb)
    matrix = prim.build_rgb_to_xyz(space.primaries)
    return linear @ matrix.T


def from_xyz(xyz: np.ndarray, space: ColorSpace) -> np.ndarray:
    """CIE XYZ (adapted to `space`'s native white point) -> display-referred
    RGB in `space`."""
    matrix = prim.build_xyz_to_rgb(space.primaries)
    linear = xyz @ matrix.T
    return space.oetf(linear)


def convert(encoded_rgb: np.ndarray, from_space: ColorSpace, to_space: ColorSpace) -> np.ndarray:
    """Display-referred RGB in from_space -> display-referred RGB in
    to_space, via XYZ with chromatic adaptation if the two spaces' white
    points differ."""
    xyz = to_xyz(encoded_rgb, from_space)
    if not np.allclose(from_space.white_xyz, to_space.white_xyz):
        adapt = chromatic_adaptation_matrix(from_space.white_xyz, to_space.white_xyz)
        xyz = xyz @ adapt.T
    return from_xyz(xyz, to_space)
