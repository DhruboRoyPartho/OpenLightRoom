import numpy as np

from core.ai import default_registry


def subject_mask(image: np.ndarray, ai_registry=None) -> np.ndarray:
    """image: float32 RGB, HxWx3, in [0, 1]. Delegates to
    ai_registry.subject_mask (core.ai.default_registry if none is given).
    Per SubjectMaskEngine.detect()'s documented contract, None (no real
    engine registered, or no subject found) falls back to a whole-image
    (all-ones) mask - not an empty one - so a Subject mask component is
    still a usable, visible local adjustment (equivalent to a global one)
    rather than a mysteriously inert mask, until a real segmentation
    engine is registered.
    """
    registry = ai_registry or default_registry
    h, w = image.shape[:2]
    result = registry.subject_mask.detect(image)
    if result is None:
        return np.ones((h, w), dtype=np.float32)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def sky_mask(image: np.ndarray, ai_registry=None) -> np.ndarray:
    """Sky-detection counterpart to subject_mask() - same contract, same
    whole-image fallback when no real SkyMaskEngine is registered."""
    registry = ai_registry or default_registry
    h, w = image.shape[:2]
    result = registry.sky_mask.detect(image)
    if result is None:
        return np.ones((h, w), dtype=np.float32)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
