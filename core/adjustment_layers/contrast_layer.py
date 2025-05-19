import numpy as np
from core.processing.contrast import adjust_contrast

class ContrastLayer:
    def __init__(self, contrast_factor: float):
        self.contrast_factor = contrast_factor
    
    def __str__(self):
        return f"Contrast (x{self.contrast_factor:+.2f})"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_contrast(image, self.contrast_factor)