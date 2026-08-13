from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ColorAnalysis:
    """Descriptive analysis of an image's color/tone characteristics -
    what a ColorAnalysisEngine implementation computes, so other AI
    engines (AutoGradeEngine, ColorMatchEngine) or the UI can consume a
    consistent result shape without depending on any particular
    implementation.
    """
    dominant_hues_deg: list = field(default_factory=list)
    mean_luminance: float = 0.0
    estimated_temperature: float = 0.0   # app's -100..100 Temperature convention
    estimated_tint: float = 0.0          # app's -100..100 Tint convention
    labels: list = field(default_factory=list)  # free-form scene/content tags, e.g. ["outdoor", "portrait"]


class ColorAnalysisEngine(ABC):
    """Analyzes a rendered image and returns a ColorAnalysis. A real
    implementation might run a small CV/ML model; this interface makes no
    assumption about how - callers only ever depend on this contract, not
    on any concrete implementation."""

    @abstractmethod
    def analyze(self, image: np.ndarray) -> ColorAnalysis:
        raise NotImplementedError


class NullColorAnalysisEngine(ColorAnalysisEngine):
    """Default, always-available implementation. Computes only what's
    cheaply and deterministically derivable with existing, non-AI code
    (gray-world white balance, mean luminance) and leaves everything that
    would require real image understanding (dominant-hue clustering, scene
    labels) at its neutral default - this is NOT a real AI implementation,
    just enough that the rest of the app can depend on
    ColorAnalysisEngine unconditionally, with zero ML dependency, and get
    a genuinely useful (if modest) result rather than a hardcoded stub.
    """

    def analyze(self, image: np.ndarray) -> ColorAnalysis:
        from core.processing.white_balance import estimate_gray_world_white_balance
        from core.processing.color_space import luminance

        clipped = np.clip(image, 0.0, 1.0)
        temp, tint = estimate_gray_world_white_balance(clipped)
        mean_luma = float(np.mean(luminance(clipped)))

        return ColorAnalysis(
            dominant_hues_deg=[],
            mean_luminance=mean_luma,
            estimated_temperature=temp,
            estimated_tint=tint,
            labels=[],
        )
