"""Tests for core/ai/sky_mask_engine.py and its registry wiring - mirrors
tests/ai/test_ai_engines.py's coverage of SubjectMaskEngine."""

import numpy as np
import pytest

from core.ai import SkyMaskEngine, NullSkyMaskEngine, AIEngineRegistry, default_registry


def test_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        SkyMaskEngine()


def test_null_implementation_is_an_instance_of_the_interface():
    assert isinstance(NullSkyMaskEngine(), SkyMaskEngine)


def test_null_sky_mask_always_returns_none():
    image = np.zeros((10, 10, 3), dtype=np.float32)
    assert NullSkyMaskEngine().detect(image) is None


def test_default_registry_holds_the_null_implementation():
    assert isinstance(default_registry.sky_mask, NullSkyMaskEngine)


def test_registry_register_sky_mask_swaps_in_a_new_implementation():
    class _FakeSkyMask(SkyMaskEngine):
        def detect(self, image):
            return np.ones(image.shape[:2], dtype=np.float32)

    registry = AIEngineRegistry()
    registry.register_sky_mask(_FakeSkyMask())
    result = registry.sky_mask.detect(np.zeros((4, 4, 3), dtype=np.float32))
    assert result.shape == (4, 4)
    assert isinstance(default_registry.sky_mask, NullSkyMaskEngine)  # unaffected
