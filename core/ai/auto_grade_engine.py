from abc import ABC, abstractmethod
import numpy as np


class AutoGradeEngine(ABC):
    """Given an image, suggests a list of adjustment layer objects (tone +
    color) that would improve/normalize it - the "Auto" button most photo
    editors offer. Returns the same adjustment-layer objects the rest of
    the app already works with, so a suggestion can be applied via
    CompositeCommand exactly like a preset.

    Interface-only: a real implementation would need a trained model or a
    much more elaborate heuristic (exposure/contrast/tone-curve
    normalization) than belongs in this stub layer.
    """

    @abstractmethod
    def suggest(self, image: np.ndarray) -> list:
        raise NotImplementedError


class NullAutoGradeEngine(AutoGradeEngine):
    """Default implementation: builds a suggestion purely from the
    existing, non-AI gray-world White Balance estimator - a real, useful
    (if modest) "Auto" result that works everywhere, rather than an empty
    no-op. A future ML-based AutoGradeEngine can add tone-curve/contrast/
    exposure suggestions on top without changing this interface or
    anything that calls it."""

    def suggest(self, image: np.ndarray) -> list:
        from core.processing.white_balance import estimate_gray_world_white_balance
        from core.adjustment_layers.temperature_layer import TemperatureLayer
        from core.adjustment_layers.tint_layer import TintLayer

        temp, tint = estimate_gray_world_white_balance(np.clip(image, 0.0, 1.0))
        layers = []
        if abs(temp) > 1e-6:
            layers.append(TemperatureLayer(temp))
        if abs(tint) > 1e-6:
            layers.append(TintLayer(tint))
        return layers
