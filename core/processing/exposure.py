import numpy as np

def adjust_exposure(image: np.ndarray, value: float) -> np.ndarray:
    # value range: -100..100, mapped to -5..+5 stops. Operates on
    # scene-linear data, where a stop is literally a doubling of light -
    # this is the whole reason the pipeline carries a linear working stage.
    if value == 0:
        return image
    stops = value / 20.0
    return image * (2.0 ** stops)
