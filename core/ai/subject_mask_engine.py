from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class SubjectMaskEngine(ABC):
    """Given an image, returns a float32 HxW mask in [0, 1] identifying the
    main subject (a person, an object) for subject-aware local adjustments
    - portrait retouching, background-only grading, etc.

    Interface-only: real subject detection needs a trained segmentation
    model, explicitly out of scope for this stub layer.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Returns an HxW float32 mask in [0, 1], or None if no subject
        was found (or this engine doesn't support detection at all).
        Callers must treat None as "fall back to a whole-image edit," not
        as an error."""
        raise NotImplementedError


class NullSubjectMaskEngine(SubjectMaskEngine):
    """Default implementation: always reports "no subject detected"
    rather than guessing, so callers can depend on SubjectMaskEngine
    unconditionally and correctly fall back to whole-image behavior until
    a real segmentation-based implementation is plugged in."""

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        return None
