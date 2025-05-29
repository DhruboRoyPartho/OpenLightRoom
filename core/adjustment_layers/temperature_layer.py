import numpy as np
from core.processing.temperature import adjust_temperature

class TemperatureLayer:
    def __init__(self, temperature_factor: float):
        self.temperature_factor = temperature_factor
    
    # def __str__(self):
    #     return f"Contrast (x{self.contrast_factor:+.2f})"
    def __str__(self):
        return "Temperature"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_temperature(image, self.temperature_factor)