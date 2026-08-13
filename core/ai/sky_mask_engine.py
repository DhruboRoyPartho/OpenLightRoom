from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class SkyMaskEngine(ABC):
    """Given an image, returns a float32 HxW mask in [0, 1] identifying the
    sky, for sky-aware local adjustments - graduated sky replacement-style
    grading, haze/dehaze targeting, etc.

    Interface-only: real sky segmentation needs a trained model,
    explicitly out of scope for this stub layer.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Returns an HxW float32 mask in [0, 1], or None if no sky was
        found (or this engine doesn't support detection at all). Callers
        must treat None as "fall back to a whole-image edit," not as an
        error - the same contract as SubjectMaskEngine.detect()."""
        raise NotImplementedError


class NullSkyMaskEngine(SkyMaskEngine):
    """Default implementation: always reports "no sky detected" rather
    than guessing, so callers can depend on SkyMaskEngine unconditionally
    and correctly fall back to whole-image behavior until a real
    segmentation-based implementation is plugged in."""

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        return None
