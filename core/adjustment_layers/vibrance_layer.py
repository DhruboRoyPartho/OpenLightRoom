import numpy as np
from core.processing.vibrance import adjust_vibrance

class VibranceLayer:
    def __init__(self, vibrance_value: float):
        self.vibrance_value = vibrance_value

    def __str__(self):
        return "Vibrance"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_vibrance(image, self.vibrance_value)
