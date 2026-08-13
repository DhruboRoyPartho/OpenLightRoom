"""AI architecture interfaces - ColorAnalysisEngine, ColorMatchEngine,
AutoGradeEngine, SubjectMaskEngine, SkyMaskEngine - and their default
Null* (no real AI) implementations, plus a small registry so the rest of
the app depends only on these interfaces and never on a concrete
implementation.

Nothing in this package uses machine learning. The Null* implementations
either return an honest "nothing to suggest"/"no subject found" result, or
(where a non-AI heuristic already exists elsewhere in the app, like
gray-world White Balance) reuse that - never a fabricated "AI" result. A
real implementation can be dropped in later by subclassing the relevant
engine and calling AIEngineRegistry.register_*(); no other code changes.
"""

from core.ai.color_analysis_engine import ColorAnalysisEngine, NullColorAnalysisEngine, ColorAnalysis
from core.ai.color_match_engine import ColorMatchEngine, NullColorMatchEngine
from core.ai.auto_grade_engine import AutoGradeEngine, NullAutoGradeEngine
from core.ai.subject_mask_engine import SubjectMaskEngine, NullSubjectMaskEngine
from core.ai.sky_mask_engine import SkyMaskEngine, NullSkyMaskEngine
from core.ai.registry import AIEngineRegistry, default_registry

__all__ = [
    "ColorAnalysisEngine", "NullColorAnalysisEngine", "ColorAnalysis",
    "ColorMatchEngine", "NullColorMatchEngine",
    "AutoGradeEngine", "NullAutoGradeEngine",
    "SubjectMaskEngine", "NullSubjectMaskEngine",
    "SkyMaskEngine", "NullSkyMaskEngine",
    "AIEngineRegistry", "default_registry",
]
