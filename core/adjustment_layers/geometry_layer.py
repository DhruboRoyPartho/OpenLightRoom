import numpy as np
from core.processing.geometry import (
    rotate90, flip, crop_normalized, rotate_rect_90, flip_rect, rotate_arbitrary,
)

FULL_FRAME = (0.0, 0.0, 1.0, 1.0)


class GeometryLayer:
    """Non-destructive crop/rotate/flip/straighten, bundled into one layer
    (the same way Lightroom's Crop tool handles rotate-90, flip, straighten
    and crop together). Applied in a fixed order:
    rotate90 -> flip -> straighten (angle, auto-cropped) -> crop.

    crop_rect is normalized (fractions of the frame *after* rotate/flip/
    straighten), so it stays meaningful even though those steps change the
    frame's pixel dimensions. angle and crop_rect are always edited and
    committed together in the same crop-mode session (see CanvasToolbar),
    so crop_rect never needs to be re-derived for a changing angle.
    """

    def __init__(self, rotation90: int = 0, flip_h: bool = False, flip_v: bool = False,
                 crop_rect=None, angle: float = 0.0):
        self.rotation90 = rotation90 % 4
        self.flip_h = bool(flip_h)
        self.flip_v = bool(flip_v)
        self.crop_rect = tuple(crop_rect) if crop_rect else FULL_FRAME
        self.angle = float(angle)

    def __str__(self):
        return "Crop"

    def is_identity(self) -> bool:
        return (
            self.rotation90 == 0 and not self.flip_h and not self.flip_v
            and self.crop_rect == FULL_FRAME and self.angle == 0.0
        )

    def with_rotation(self, delta_quarter_turns: int) -> "GeometryLayer":
        # Rotate the existing crop rectangle along with the frame, so a crop
        # made before rotating keeps selecting the same visual content
        # instead of jumping to a different region of the (now-rotated) image.
        rect = self.crop_rect
        for _ in range(delta_quarter_turns % 4):
            rect = rotate_rect_90(rect, clockwise=True)
        return GeometryLayer(self.rotation90 + delta_quarter_turns, self.flip_h, self.flip_v, rect, self.angle)

    def with_flip_h(self) -> "GeometryLayer":
        rect = flip_rect(self.crop_rect, horizontal=True, vertical=False)
        return GeometryLayer(self.rotation90, not self.flip_h, self.flip_v, rect, self.angle)

    def with_flip_v(self) -> "GeometryLayer":
        rect = flip_rect(self.crop_rect, horizontal=False, vertical=True)
        return GeometryLayer(self.rotation90, self.flip_h, not self.flip_v, rect, self.angle)

    def with_crop(self, crop_rect) -> "GeometryLayer":
        return GeometryLayer(self.rotation90, self.flip_h, self.flip_v, crop_rect, self.angle)

    def with_angle_and_crop(self, angle: float, crop_rect) -> "GeometryLayer":
        """angle (straighten) and crop_rect are always set together, since
        crop_rect is only ever meaningful relative to whatever frame the
        current angle produces."""
        return GeometryLayer(self.rotation90, self.flip_h, self.flip_v, crop_rect, angle)

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = rotate90(image, self.rotation90)
        result = flip(result, self.flip_h, self.flip_v)
        result = rotate_arbitrary(result, self.angle)
        result = crop_normalized(result, self.crop_rect)
        return result
