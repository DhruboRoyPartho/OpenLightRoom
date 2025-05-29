import numpy as np
from core.processing.shadows import adjust_shadows

class ShadowsLayer:
    def __init__(self, shadows_factor: float):
        self.shadows_factor = shadows_factor
    
    def __str__(self):
        return "Shadows"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_shadows(image, self.shadows_factor)