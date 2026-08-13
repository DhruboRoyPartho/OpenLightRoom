import numpy as np

from core.processing.exposure import adjust_exposure
from core.processing.contrast import adjust_contrast
from core.processing.highlights import adjust_highlights
from core.processing.shadows import adjust_shadows
from core.processing.whites import adjust_whites
from core.processing.blacks import adjust_blacks
from core.processing.temperature import adjust_temperature
from core.processing.tint import adjust_tint
from core.processing.saturation import adjust_saturation
from core.processing.hue import adjust_hue
from core.processing.color_space import srgb_to_linear, linear_to_srgb
from core.masking.mask import Mask

ADJUSTMENT_FIELDS = (
    "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
    "temperature", "tint", "saturation", "hue",
)
ADJUSTMENT_DEFAULTS = {
    "exposure": 0.0, "contrast": 1.0, "highlights": 0.0, "shadows": 0.0,
    "whites": 0.0, "blacks": 0.0, "temperature": 0.0, "tint": 0.0,
    "saturation": 0.0, "hue": 0.0,
}


class MaskedAdjustmentLayer:
    """A local (masked) adjustment: a Mask (see core/masking/mask.py)
    paired with a small set of Basic/Color adjustments, applied only
    within the mask's selection and blended against the unaffected image
    by the mask's own alpha.

    Reuses the exact same per-pixel processing functions the equivalent
    global tools use (adjust_exposure, adjust_contrast, ...). Exposure,
    Temperature and Tint are physically-meaningful-in-linear-light
    operations (see their own docstrings) - globally they run in the
    pipeline's "Linear" stage, before the linear->display transform, for
    exactly that reason. A masked adjustment only ever sees the already
    display-referred (gamma-encoded) image, so those three round-trip
    through srgb_to_linear/linear_to_srgb here to get the same physically
    correct result the global sliders produce; applying "a stop doubles
    the value" directly to gamma-encoded bytes clips highlights far too
    early and makes even a modest push look harsh/blown-out. Every other
    local adjustment (Contrast, Highlights, Shadows, Whites, Blacks,
    Saturation, Hue) is a display-space *parametric* tool both globally
    and locally, so it needs no such conversion.

    A document can hold any number of these at once, each with its own
    unique pipeline name ("Mask 1", "Mask 2", ... - see
    ImageDocument.next_mask_name()) - the one layer type in this app not
    capped at a single instance per name; see
    core/pipeline/stage.py:Stage.dynamic_prefix.
    """

    def __init__(self, pipeline_name: str, mask: Mask = None, label: str = None,
                 visible: bool = True, ai_registry=None, **adjustments):
        self._pipeline_name = pipeline_name
        self.mask = mask if mask is not None else Mask()
        self.label = label or pipeline_name
        self.visible = visible
        # Only ever needed to override which AI engines a Subject/Sky mask
        # component queries (e.g. in a test); None (the default) falls
        # back to core.ai.default_registry, same as calling
        # core.masking.subject_mask()/sky_mask() with no registry.
        self.ai_registry = ai_registry

        for field in ADJUSTMENT_FIELDS:
            setattr(self, field, adjustments.get(field, ADJUSTMENT_DEFAULTS[field]))

    def __str__(self):
        return self._pipeline_name

    def _copy_adjustments(self) -> dict:
        return {field: getattr(self, field) for field in ADJUSTMENT_FIELDS}

    def with_adjustment(self, field: str, value) -> "MaskedAdjustmentLayer":
        if field not in ADJUSTMENT_FIELDS:
            raise ValueError(f"Unknown local adjustment field {field!r}, expected one of {ADJUSTMENT_FIELDS}")
        adjustments = self._copy_adjustments()
        adjustments[field] = value
        return MaskedAdjustmentLayer(
            self._pipeline_name, mask=self.mask, label=self.label,
            visible=self.visible, ai_registry=self.ai_registry, **adjustments,
        )

    def with_mask(self, mask: Mask) -> "MaskedAdjustmentLayer":
        return MaskedAdjustmentLayer(
            self._pipeline_name, mask=mask, label=self.label,
            visible=self.visible, ai_registry=self.ai_registry, **self._copy_adjustments(),
        )

    def with_label(self, label: str) -> "MaskedAdjustmentLayer":
        return MaskedAdjustmentLayer(
            self._pipeline_name, mask=self.mask, label=label,
            visible=self.visible, ai_registry=self.ai_registry, **self._copy_adjustments(),
        )

    def with_visible(self, visible: bool) -> "MaskedAdjustmentLayer":
        return MaskedAdjustmentLayer(
            self._pipeline_name, mask=self.mask, label=self.label,
            visible=visible, ai_registry=self.ai_registry, **self._copy_adjustments(),
        )

    def has_adjustments(self) -> bool:
        return any(getattr(self, field) != ADJUSTMENT_DEFAULTS[field] for field in ADJUSTMENT_FIELDS)

    def is_identity(self) -> bool:
        return (not self.visible) or self.mask.is_empty() or (not self.has_adjustments())

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.visible or not self.has_adjustments():
            return image

        mask = self.mask.evaluate(image, self.ai_registry)
        if not np.any(mask > 1e-6):
            return image

        adjusted = image
        if self.exposure or self.temperature or self.tint:
            # Round-trip through linear light for exactly these three -
            # see the class docstring for why they can't just operate on
            # the display-referred bytes like the rest of the tools here.
            linear = srgb_to_linear(np.clip(adjusted, 0.0, 1.0))
            if self.exposure:
                linear = adjust_exposure(linear, self.exposure)
            if self.temperature:
                linear = adjust_temperature(linear, self.temperature)
            if self.tint:
                linear = adjust_tint(linear, self.tint)
            # Generous headroom before re-encoding, matching
            # core.processing.color_space.linear_to_display's own clip -
            # keeps recoverable highlight detail from producing runaway
            # values once gamma-encoded, without hard-clipping early.
            adjusted = linear_to_srgb(np.clip(linear, 0.0, 16.0))
        if self.contrast != 1.0:
            adjusted = adjust_contrast(adjusted, self.contrast)
        if self.highlights:
            adjusted = adjust_highlights(adjusted, self.highlights)
        if self.shadows:
            adjusted = adjust_shadows(adjusted, self.shadows)
        if self.whites:
            adjusted = adjust_whites(adjusted, self.whites)
        if self.blacks:
            adjusted = adjust_blacks(adjusted, self.blacks)
        if self.saturation:
            adjusted = adjust_saturation(adjusted, self.saturation)
        if self.hue:
            adjusted = adjust_hue(adjusted, self.hue)

        mask3 = mask[..., None]
        blended = image * (1.0 - mask3) + adjusted * mask3
        return np.clip(blended, 0.0, 1.0).astype(np.float32)
