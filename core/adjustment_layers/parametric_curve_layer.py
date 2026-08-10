import numpy as np
from core.processing.parametric_curve import apply_parametric_curve


class ParametricCurveLayer:
    def __init__(self, highlights: float = 0, lights: float = 0, darks: float = 0, shadows: float = 0):
        self.highlights = highlights
        self.lights = lights
        self.darks = darks
        self.shadows = shadows

    def __str__(self):
        return "Parametric Curve"

    def is_identity(self) -> bool:
        return self.highlights == 0 and self.lights == 0 and self.darks == 0 and self.shadows == 0

    def with_highlights(self, value: float) -> "ParametricCurveLayer":
        return ParametricCurveLayer(value, self.lights, self.darks, self.shadows)

    def with_lights(self, value: float) -> "ParametricCurveLayer":
        return ParametricCurveLayer(self.highlights, value, self.darks, self.shadows)

    def with_darks(self, value: float) -> "ParametricCurveLayer":
        return ParametricCurveLayer(self.highlights, self.lights, value, self.shadows)

    def with_shadows(self, value: float) -> "ParametricCurveLayer":
        return ParametricCurveLayer(self.highlights, self.lights, self.darks, value)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return apply_parametric_curve(image, self.highlights, self.lights, self.darks, self.shadows)
