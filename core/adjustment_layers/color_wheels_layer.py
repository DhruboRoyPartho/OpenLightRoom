import numpy as np
from core.processing.color_wheels import apply_color_wheels, DEFAULT_WHEEL, ZONES

_ZONE_ATTR = {"shadows": "shadows", "midtones": "midtones", "highlights": "highlights", "global": "global_"}


class ColorWheelsLayer:
    """Shadows/Midtones/Highlights/Global color-grading wheels, bundled
    into one layer (like HSLLayer bundles its 8 channels) so the 4 wheels
    dedupe and undo as a single unit.
    """

    def __init__(self, shadows: dict = None, midtones: dict = None,
                 highlights: dict = None, global_: dict = None):
        self.shadows = {**DEFAULT_WHEEL, **(shadows or {})}
        self.midtones = {**DEFAULT_WHEEL, **(midtones or {})}
        self.highlights = {**DEFAULT_WHEEL, **(highlights or {})}
        self.global_ = {**DEFAULT_WHEEL, **(global_ or {})}

    def __str__(self):
        return "Color Wheels"

    def is_identity(self) -> bool:
        return all(
            w["chroma"] == 0 and w["luminance"] == 0
            for w in (self.shadows, self.midtones, self.highlights, self.global_)
        )

    def with_wheel(self, zone: str, hue_deg: float = None, chroma: float = None, luminance: float = None) -> "ColorWheelsLayer":
        """Returns a new ColorWheelsLayer with the given zone's wheel
        updated. Only the passed (non-None) fields change; hue_deg is
        wrapped into [0, 360)."""
        if zone not in ZONES:
            raise ValueError(f"Unknown color wheel zone {zone!r}, expected one of {ZONES}")

        fields = {z: dict(getattr(self, _ZONE_ATTR[z])) for z in ZONES}
        w = fields[zone]
        if hue_deg is not None:
            w["hue_deg"] = hue_deg % 360.0
        if chroma is not None:
            w["chroma"] = chroma
        if luminance is not None:
            w["luminance"] = luminance
        return ColorWheelsLayer(
            shadows=fields["shadows"], midtones=fields["midtones"],
            highlights=fields["highlights"], global_=fields["global"],
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        return apply_color_wheels(image, self.shadows, self.midtones, self.highlights, self.global_)
