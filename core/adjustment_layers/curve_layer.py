import numpy as np
from core.processing.curve import apply_curve, IDENTITY_POINTS

CHANNELS = ("RGB", "Red", "Green", "Blue")


class CurveLayer:
    def __init__(self, points_by_channel: dict = None):
        points_by_channel = points_by_channel or {}
        # Normalize every point to a plain (x, y) tuple of ints, regardless
        # of whether it arrived as tuples (from the curve widget) or as
        # JSON-loaded lists (from a saved project) - keeps equality checks
        # (used for the undo no-op guard and reset-enabled state) reliable.
        self.points_by_channel = {
            channel: [(int(x), int(y)) for x, y in points_by_channel.get(channel, IDENTITY_POINTS)]
            for channel in CHANNELS
        }

    def __str__(self):
        return "Curve"

    def with_channel(self, channel: str, points) -> "CurveLayer":
        updated = {ch: list(pts) for ch, pts in self.points_by_channel.items()}
        updated[channel] = list(points)
        return CurveLayer(updated)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return apply_curve(image, self.points_by_channel)
