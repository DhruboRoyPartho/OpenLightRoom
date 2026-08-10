import numpy as np
from core.processing.hue import adjust_hue

class HueLayer:
    def __init__(self, hue_value: float):
        self.hue_value = hue_value

    def __str__(self):
        return "Hue"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_hue(image, self.hue_value)
