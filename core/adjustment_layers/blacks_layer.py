import numpy as np
from core.processing.blacks import adjust_blacks

class BlacksLayer:
    def __init__(self, blacks_factor: float):
        self.blacks_factor = blacks_factor
    
    def __str__(self):
        return "Blacks"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_blacks(image, self.blacks_factor)