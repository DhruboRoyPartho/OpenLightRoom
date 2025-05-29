import numpy as np
from core.processing.tint import adjust_tint

class TintLayer:
    def __init__(self, tint_factor: float):
        self.tint_factor = tint_factor
    
    # def __str__(self):
    #     return f"Contrast (x{self.contrast_factor:+.2f})"
    def __str__(self):
        return "Tint"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_tint(image, self.tint_factor)