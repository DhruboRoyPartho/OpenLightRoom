import numpy as np
from core.processing.hsl_grading import apply_hsl_grading, HSL_CHANNELS

_AXES = ("hue", "saturation", "luminance")


class HSLLayer:
    """8-channel HSL color grading: independent Hue/Saturation/Luminance
    offsets for each of Red/Orange/Yellow/Green/Aqua/Blue/Purple/Magenta,
    bundled into one layer (like Lightroom's HSL panel is one tool with 24
    fields, not 24 separate layers) so it dedupes and undoes as a single
    unit the same way CurveLayer and ParametricCurveLayer do.
    """

    def __init__(self, hue: dict = None, saturation: dict = None, luminance: dict = None):
        self.hue = {k: v for k, v in (hue or {}).items() if v}
        self.saturation = {k: v for k, v in (saturation or {}).items() if v}
        self.luminance = {k: v for k, v in (luminance or {}).items() if v}

    def __str__(self):
        return "HSL"

    def is_identity(self) -> bool:
        return not self.hue and not self.saturation and not self.luminance

    def with_value(self, axis: str, channel: str, value: float) -> "HSLLayer":
        """Returns a new HSLLayer with `channel`'s value on `axis`
        ('hue' | 'saturation' | 'luminance') replaced by `value` (0 removes
        the channel's entry on that axis)."""
        if axis not in _AXES:
            raise ValueError(f"Unknown HSL axis {axis!r}, expected one of {_AXES}")
        if channel not in HSL_CHANNELS:
            raise ValueError(f"Unknown HSL channel {channel!r}")

        fields = {"hue": dict(self.hue), "saturation": dict(self.saturation), "luminance": dict(self.luminance)}
        target = fields[axis]
        if value:
            target[channel] = value
        else:
            target.pop(channel, None)
        return HSLLayer(**fields)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return apply_hsl_grading(image, self.hue, self.saturation, self.luminance)
