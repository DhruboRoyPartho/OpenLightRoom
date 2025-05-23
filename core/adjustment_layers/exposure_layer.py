import numpy as np
from core.processing.exposure import adjust_exposure

class ExposureLayer:
    def __init__(self, exposure_factor: float):
        self.exposure_factor = exposure_factor
    
    def __str__(self):
        return "Exposure"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_exposure(image, self.exposure_factor)