"""Tests for core/ai: the interface contracts, the Null* default
implementations, and the registry. No real AI/ML is exercised here -
these confirm the stub layer behaves exactly as documented (honest
"nothing to suggest" results, or a reuse of an existing non-AI heuristic),
and that the abstract interfaces genuinely can't be instantiated without
implementing their contract."""

import numpy as np
import pytest

from core.ai import (
    ColorAnalysisEngine, NullColorAnalysisEngine, ColorAnalysis,
    ColorMatchEngine, NullColorMatchEngine,
    AutoGradeEngine, NullAutoGradeEngine,
    SubjectMaskEngine, NullSubjectMaskEngine,
    AIEngineRegistry, default_registry,
)


# --- interfaces are genuinely abstract -------------------------------------

def test_color_analysis_engine_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ColorAnalysisEngine()


def test_color_match_engine_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ColorMatchEngine()


def test_auto_grade_engine_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AutoGradeEngine()


def test_subject_mask_engine_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        SubjectMaskEngine()


# --- Null* implementations satisfy their interface -------------------------

def test_null_implementations_are_instances_of_their_interface():
    assert isinstance(NullColorAnalysisEngine(), ColorAnalysisEngine)
    assert isinstance(NullColorMatchEngine(), ColorMatchEngine)
    assert isinstance(NullAutoGradeEngine(), AutoGradeEngine)
    assert isinstance(NullSubjectMaskEngine(), SubjectMaskEngine)


# --- NullColorAnalysisEngine -------------------------------------------------

def test_null_color_analysis_returns_color_analysis_with_gray_world_estimate():
    rng = np.random.default_rng(0)
    neutral = rng.random((16, 16, 3)).astype(np.float32) * 0.5 + 0.2
    cast = neutral.copy()
    cast[..., 0] *= 1.5  # warm cast

    result = NullColorAnalysisEngine().analyze(cast)
    assert isinstance(result, ColorAnalysis)
    assert result.estimated_temperature < 0  # gray-world should suggest cooling it back down
    assert 0.0 <= result.mean_luminance <= 1.0
    assert result.dominant_hues_deg == []
    assert result.labels == []


def test_null_color_analysis_does_not_raise_on_out_of_gamut_or_nan():
    weird = np.array([[[np.nan, 2.0, -1.0]]], dtype=np.float32)
    result = NullColorAnalysisEngine().analyze(weird)
    assert isinstance(result, ColorAnalysis)


# --- NullColorMatchEngine ---------------------------------------------------

def test_null_color_match_always_suggests_no_changes():
    ref = np.random.default_rng(1).random((8, 8, 3)).astype(np.float32)
    current = np.random.default_rng(2).random((8, 8, 3)).astype(np.float32)
    result = NullColorMatchEngine().match(ref, current)
    assert result == []


# --- NullAutoGradeEngine ----------------------------------------------------

def test_null_auto_grade_suggests_nothing_for_an_already_neutral_image():
    neutral = np.full((10, 10, 3), 0.4, dtype=np.float32)
    layers = NullAutoGradeEngine().suggest(neutral)
    assert layers == []


def test_null_auto_grade_suggests_temperature_and_tint_layers_for_a_cast_image():
    rng = np.random.default_rng(3)
    neutral = rng.random((16, 16, 3)).astype(np.float32) * 0.5 + 0.1
    cast = neutral.copy()
    cast[..., 0] *= 1.6
    cast[..., 2] *= 0.6

    layers = NullAutoGradeEngine().suggest(cast)
    names = {str(l) for l in layers}
    assert "Temperature" in names
    # These are real adjustment-layer objects - applying them must not raise.
    image = cast.copy()
    for layer in layers:
        image = layer.apply(image)
    assert np.isfinite(image).all()


# --- NullSubjectMaskEngine ---------------------------------------------------

def test_null_subject_mask_always_returns_none():
    image = np.zeros((10, 10, 3), dtype=np.float32)
    assert NullSubjectMaskEngine().detect(image) is None


# --- registry ----------------------------------------------------------------

def test_default_registry_holds_null_implementations():
    assert isinstance(default_registry.color_analysis, NullColorAnalysisEngine)
    assert isinstance(default_registry.color_match, NullColorMatchEngine)
    assert isinstance(default_registry.auto_grade, NullAutoGradeEngine)
    assert isinstance(default_registry.subject_mask, NullSubjectMaskEngine)


def test_registry_register_methods_swap_in_a_new_implementation():
    class _FakeAutoGrade(AutoGradeEngine):
        def suggest(self, image):
            return ["sentinel"]

    registry = AIEngineRegistry()
    registry.register_auto_grade(_FakeAutoGrade())
    assert registry.auto_grade.suggest(None) == ["sentinel"]
    # Registering a replacement doesn't affect other engines or other instances.
    assert isinstance(registry.color_match, NullColorMatchEngine)
    assert isinstance(default_registry.auto_grade, NullAutoGradeEngine)


def test_a_fresh_registry_instance_is_independent_of_the_shared_default():
    registry = AIEngineRegistry()
    assert registry is not default_registry
    assert isinstance(registry.auto_grade, NullAutoGradeEngine)
