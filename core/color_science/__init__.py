"""Reusable, UI-independent color science: transfer functions, RGB primary
matrices, named color spaces, and RGB <-> XYZ <-> Lab <-> OKLab <-> OKLCH
conversions.

Nothing in this package knows about Qt, the layer/pipeline model, or any
particular tool - it's pure math, so it can be unit tested in isolation
and reused by every color-grading tool (HSL, color wheels, selective
color, white balance) without duplicating conversion logic.
"""

from core.color_science.spaces import (
    ColorSpace, SRGB, REC709, DISPLAY_P3, ADOBE_RGB, PROPHOTO_RGB,
    get as get_color_space, to_xyz, from_xyz, convert,
)
from core.color_science.xyz_lab import xyz_to_lab, lab_to_xyz, D65_WHITE_XYZ, D50_WHITE_XYZ
from core.color_science.oklab import (
    linear_srgb_to_oklab, oklab_to_linear_srgb,
    oklab_to_oklch, oklch_to_oklab,
    linear_srgb_to_oklch, oklch_to_linear_srgb,
)

__all__ = [
    "ColorSpace", "SRGB", "REC709", "DISPLAY_P3", "ADOBE_RGB", "PROPHOTO_RGB",
    "get_color_space", "to_xyz", "from_xyz", "convert",
    "xyz_to_lab", "lab_to_xyz", "D65_WHITE_XYZ", "D50_WHITE_XYZ",
    "linear_srgb_to_oklab", "oklab_to_linear_srgb",
    "oklab_to_oklch", "oklch_to_oklab",
    "linear_srgb_to_oklch", "oklch_to_linear_srgb",
]
