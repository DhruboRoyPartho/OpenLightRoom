import numpy as np
from core.processing.saturation import adjust_saturation

class SaturationLayer:
    def __init__(self, saturation_value: float):
        self.saturation_value = saturation_value

    def __str__(self):
        return "Saturation"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_saturation(image, self.saturation_value)
