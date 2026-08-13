from abc import ABC, abstractmethod
import numpy as np


class ColorMatchEngine(ABC):
    """Given a reference image and the current image, returns a list of
    adjustment layer objects (from core.adjustment_layers) that would push
    the current image's look toward the reference's - "match this photo's
    grade to that one." Returning plain adjustment layers (the same
    objects ImageDocument.layers and the preset system already use) means
    a real implementation's output can be applied through the exact same
    CompositeCommand-based path Auto White Balance and Presets already
    use, with no new plumbing.

    Interface-only for now: no implementation here performs real
    reference-image color matching, which would need a trained/learned
    model - explicitly out of scope for this stub layer per the project's
    "do not fake AI" requirement.
    """

    @abstractmethod
    def match(self, reference_image: np.ndarray, current_image: np.ndarray) -> list:
        raise NotImplementedError


class NullColorMatchEngine(ColorMatchEngine):
    """Default implementation: suggests no changes. Always available so
    the UI can wire up a "Match Color" action today - it will start doing
    something real the moment a ColorMatchEngine implementation is
    plugged into the registry, without any calling code changing."""

    def match(self, reference_image: np.ndarray, current_image: np.ndarray) -> list:
        return []
