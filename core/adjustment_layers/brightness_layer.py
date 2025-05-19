import numpy as np
from core.processing.brightness import adjust_brightness

class BrightnessLayer:
    def __init__(self, brightness_value: float):
        self.brightness_value = brightness_value

    def __str__(self):
        return f"Brightness ({self.brightness_value:+.2f})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_brightness(image, self.brightness_value)