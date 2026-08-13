from core.ai.color_analysis_engine import ColorAnalysisEngine, NullColorAnalysisEngine
from core.ai.color_match_engine import ColorMatchEngine, NullColorMatchEngine
from core.ai.auto_grade_engine import AutoGradeEngine, NullAutoGradeEngine
from core.ai.subject_mask_engine import SubjectMaskEngine, NullSubjectMaskEngine
from core.ai.sky_mask_engine import SkyMaskEngine, NullSkyMaskEngine


class AIEngineRegistry:
    """Holds one implementation of each AI engine interface. Defaults to
    the Null* implementations, so the app is fully functional with zero AI
    dependencies; call register_* to plug in a real implementation later
    without touching any calling code - callers only ever go through a
    registry instance (or receive one injected), never import a concrete
    engine class directly.
    """

    def __init__(self):
        self.color_analysis: ColorAnalysisEngine = NullColorAnalysisEngine()
        self.color_match: ColorMatchEngine = NullColorMatchEngine()
        self.auto_grade: AutoGradeEngine = NullAutoGradeEngine()
        self.subject_mask: SubjectMaskEngine = NullSubjectMaskEngine()
        self.sky_mask: SkyMaskEngine = NullSkyMaskEngine()

    def register_color_analysis(self, engine: ColorAnalysisEngine):
        self.color_analysis = engine

    def register_color_match(self, engine: ColorMatchEngine):
        self.color_match = engine

    def register_auto_grade(self, engine: AutoGradeEngine):
        self.auto_grade = engine

    def register_subject_mask(self, engine: SubjectMaskEngine):
        self.subject_mask = engine

    def register_sky_mask(self, engine: SkyMaskEngine):
        self.sky_mask = engine


# A single shared instance is enough for this app (one process, one set of
# engines) - callers that want isolation (e.g. tests) can construct their
# own AIEngineRegistry() instead of using this one.
default_registry = AIEngineRegistry()
