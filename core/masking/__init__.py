"""Reusable, UI-independent selection masks.

Two layers:
- Range masks (smooth hue/saturation/luminance ranges, feathered via
  smoothstep instead of hard thresholds) combinable into a
  SelectiveColorMask - shared by every region-based color tool (HSL,
  Color Wheels, Selective Color, tonal separation).
- The Masking panel's mask types (Brush, Radial, Ellipse, Linear
  Gradient, Rectangle, Polygon, Color Range, Luminance Range, Subject,
  Sky, Skin) and operations (Add/Subtract/Intersect, Invert, Feather,
  Blur, Density), combined by Mask/MaskComponent into the single alpha a
  MaskedAdjustmentLayer blends its local edit through.
"""

from core.masking.range_mask import range_mask
from core.masking.hue_range import hue_range_mask, circular_hue_distance
from core.masking.saturation_range import saturation_range_mask
from core.masking.luminance_range import luminance_zone_mask, three_zone_luminance_masks
from core.masking.combine import combine_masks
from core.masking.selective_color_mask import SelectiveColorMask

from core.masking.shapes import (
    radial_mask, ellipse_mask, rectangle_mask, linear_gradient_mask, polygon_mask,
    brush_mask, rasterize_stroke,
)
from core.masking.color_range import color_range_mask
from core.masking.skin_mask import skin_mask
from core.masking.subject_sky_mask import subject_mask, sky_mask
from core.masking.mask import Mask, MaskComponent, COMBINE_OPS, COMPONENT_KINDS

__all__ = [
    "range_mask",
    "hue_range_mask", "circular_hue_distance",
    "saturation_range_mask",
    "luminance_zone_mask", "three_zone_luminance_masks",
    "combine_masks",
    "SelectiveColorMask",
    "radial_mask", "ellipse_mask", "rectangle_mask", "linear_gradient_mask", "polygon_mask",
    "brush_mask", "rasterize_stroke",
    "color_range_mask",
    "skin_mask",
    "subject_mask", "sky_mask",
    "Mask", "MaskComponent", "COMBINE_OPS", "COMPONENT_KINDS",
]
