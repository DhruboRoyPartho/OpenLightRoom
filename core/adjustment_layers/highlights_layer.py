import numpy as np
from core.processing.highlights import adjust_highlights

class HighlightsLayer:
    def __init__(self, hightlights_factor: float):
        self.hightlights_factor = hightlights_factor
    
    def __str__(self):
        return "Highlights"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return adjust_highlights(image, self.hightlights_factor)