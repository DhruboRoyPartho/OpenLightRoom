import numpy as np
from core.processing.whites import adjust_whites

class WhitesLayer:
    def __init__(self, whites_factor: float):
        self.whites_factor = whites_factor
    
    def __str__(self):
        return "Whites"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_whites(image, self.whites_factor)