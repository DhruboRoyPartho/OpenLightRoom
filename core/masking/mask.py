from dataclasses import dataclass, field
import cv2
import numpy as np

from core.processing.color_space import luminance, smoothstep
from core.masking.shapes import (
    brush_mask, radial_mask, ellipse_mask, rectangle_mask, linear_gradient_mask, polygon_mask,
)
from core.masking.color_range import color_range_mask
from core.masking.luminance_range import luminance_zone_mask
from core.masking.skin_mask import skin_mask
from core.masking.subject_sky_mask import subject_mask, sky_mask

COMBINE_OPS = ("add", "subtract", "intersect")
COMPONENT_KINDS = (
    "brush", "radial", "ellipse", "linear_gradient", "rectangle", "polygon",
    "color_range", "luminance_range", "subject", "sky", "skin",
)


@dataclass
class MaskComponent:
    """One shape/selection contributing to a Mask. `params` is a plain,
    JSON-serializable dict whose keys depend on `kind` (see
    core/masking/shapes.py and color_range.py/skin_mask.py/
    subject_sky_mask.py for what each kind expects - all defaulted, so an
    empty dict is always valid).

    `op` says how this component combines with everything combined so far
    (ignored for the first component in a Mask, which always becomes the
    starting selection); `invert` flips this one component before it's
    combined, independent of the Mask-level `invert`.
    """

    kind: str
    params: dict = field(default_factory=dict)
    op: str = "add"
    invert: bool = False

    def __post_init__(self):
        if self.kind not in COMPONENT_KINDS:
            raise ValueError(f"Unknown mask component kind {self.kind!r}, expected one of {COMPONENT_KINDS}")
        if self.op not in COMBINE_OPS:
            raise ValueError(f"Unknown combine op {self.op!r}, expected one of {COMBINE_OPS}")


def _evaluate_component(component: MaskComponent, image: np.ndarray, ai_registry) -> np.ndarray:
    h, w = image.shape[:2]
    p = component.params
    kind = component.kind

    if kind == "brush":
        return brush_mask(h, w, p.get("strokes", []))
    if kind == "radial":
        return radial_mask(h, w, p.get("center_x", 0.5), p.get("center_y", 0.5),
                            p.get("radius_x", 0.25), p.get("radius_y", 0.25),
                            p.get("angle_deg", 0.0), p.get("feather", 50.0))
    if kind == "ellipse":
        return ellipse_mask(h, w, p.get("center_x", 0.5), p.get("center_y", 0.5),
                             p.get("radius_x", 0.25), p.get("radius_y", 0.25),
                             p.get("angle_deg", 0.0), p.get("feather", 0.0))
    if kind == "linear_gradient":
        return linear_gradient_mask(h, w, p.get("x0", 0.3), p.get("y0", 0.5),
                                     p.get("x1", 0.7), p.get("y1", 0.5))
    if kind == "rectangle":
        return rectangle_mask(h, w, p.get("center_x", 0.5), p.get("center_y", 0.5),
                               p.get("half_width", 0.25), p.get("half_height", 0.25),
                               p.get("angle_deg", 0.0), p.get("feather", 0.0))
    if kind == "polygon":
        return polygon_mask(h, w, p.get("points", []), p.get("feather", 0.0))
    if kind == "color_range":
        return color_range_mask(image, p.get("sample_rgb", (0.5, 0.5, 0.5)), p.get("refine", 50.0))
    if kind == "luminance_range":
        luma = luminance(np.clip(image, 0.0, 1.0))
        feather_frac = max(p.get("feather", 20.0), 1.0) / 100.0
        return luminance_zone_mask(luma, p.get("low", 0.0), p.get("high", 1.0), feather_frac)
    if kind == "subject":
        return subject_mask(image, ai_registry)
    if kind == "sky":
        return sky_mask(image, ai_registry)
    if kind == "skin":
        return skin_mask(image, p.get("feather", 20.0))

    raise ValueError(f"Unknown mask component kind {kind!r}")  # pragma: no cover - guarded by __post_init__


def _apply_feather(mask: np.ndarray, feather_pct: float) -> np.ndarray:
    """Distance-transform-based edge softening (0..100): traces the
    combined mask's actual boundary (wherever it crosses 0.5) and blends a
    symmetric soft transition around it - a controlled, general-purpose
    softening that works for any mask shape or combination, including ones
    with no closed-form edge (Brush, Subject, Color Range)."""
    h, w = mask.shape
    feather_px = max(1.0, (feather_pct / 100.0) * min(h, w) * 0.5)
    binary = (mask >= 0.5).astype(np.uint8)
    dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5).astype(np.float32)
    signed_distance = dist_inside - dist_outside
    t = np.clip((signed_distance / feather_px + 1.0) / 2.0, 0.0, 1.0)
    return smoothstep(t).astype(np.float32)


def _apply_blur(mask: np.ndarray, blur_pct: float) -> np.ndarray:
    """Gaussian blur (0..100, as a percent of the shorter image
    dimension) applied directly to the mask - general smoothing distinct
    from Feather's boundary-precise softening: good for taming a noisy/
    high-frequency mask (Brush, Subject) or adding extra softness on top
    of Feather."""
    h, w = mask.shape
    sigma = max(0.5, (blur_pct / 100.0) * min(h, w) * 0.05)
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(mask, (ksize, ksize), sigma).astype(np.float32)


class Mask:
    """A combinable stack of mask components (see MaskComponent) plus
    whole-mask operations: Invert, Feather, Blur, and Density - the
    "Mask operations" the Masking panel offers on top of individual shape
    tools. evaluate() is the single place all of that comes together into
    one final HxW float32 alpha in [0, 1].
    """

    def __init__(self, components=None, feather: float = 0.0, blur: float = 0.0,
                 density: float = 100.0, invert: bool = False):
        self.components = list(components) if components else []
        self.feather = feather
        self.blur = blur
        self.density = density
        self.invert = invert

    def is_empty(self) -> bool:
        return len(self.components) == 0

    def evaluate(self, image: np.ndarray, ai_registry=None) -> np.ndarray:
        h, w = image.shape[:2]

        if not self.components:
            result = np.zeros((h, w), dtype=np.float32)
        else:
            result = None
            for component in self.components:
                comp_mask = _evaluate_component(component, image, ai_registry)
                if component.invert:
                    comp_mask = 1.0 - comp_mask

                if result is None:
                    # The first component always defines the starting
                    # selection - its own `op` is meaningless with nothing
                    # yet to combine against.
                    result = comp_mask.copy()
                elif component.op == "subtract":
                    result = np.clip(result - comp_mask, 0.0, 1.0)
                elif component.op == "intersect":
                    result = result * comp_mask
                else:  # "add"
                    result = np.clip(result + comp_mask, 0.0, 1.0)

        if self.invert:
            result = 1.0 - result

        if self.feather > 0:
            result = _apply_feather(result, self.feather)

        if self.blur > 0:
            result = _apply_blur(result, self.blur)

        result = result * (np.clip(self.density, 0.0, 100.0) / 100.0)
        return np.clip(result, 0.0, 1.0).astype(np.float32)
