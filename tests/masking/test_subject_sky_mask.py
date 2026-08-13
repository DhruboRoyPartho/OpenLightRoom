"""Tests for core/masking/subject_sky_mask.py - the AI-registry-backed
mask types, including the documented "None -> whole image" fallback
contract."""

import numpy as np

from core.ai import AIEngineRegistry, SubjectMaskEngine, SkyMaskEngine
from core.masking.subject_sky_mask import subject_mask, sky_mask


def test_subject_mask_falls_back_to_whole_image_with_the_null_engine():
    registry = AIEngineRegistry()  # holds NullSubjectMaskEngine by default
    image = np.zeros((10, 12, 3), dtype=np.float32)
    mask = subject_mask(image, registry)
    assert mask.shape == (10, 12)
    assert np.allclose(mask, 1.0)


def test_sky_mask_falls_back_to_whole_image_with_the_null_engine():
    registry = AIEngineRegistry()
    image = np.zeros((10, 12, 3), dtype=np.float32)
    mask = sky_mask(image, registry)
    assert mask.shape == (10, 12)
    assert np.allclose(mask, 1.0)


def test_subject_mask_uses_a_registered_real_engine_result():
    class _FakeSubject(SubjectMaskEngine):
        def detect(self, image):
            m = np.zeros(image.shape[:2], dtype=np.float32)
            m[2:5, 2:5] = 1.0
            return m

    registry = AIEngineRegistry()
    registry.register_subject_mask(_FakeSubject())
    image = np.zeros((10, 10, 3), dtype=np.float32)
    mask = subject_mask(image, registry)
    assert mask[3, 3] == 1.0
    assert mask[0, 0] == 0.0


def test_sky_mask_uses_a_registered_real_engine_result():
    class _FakeSky(SkyMaskEngine):
        def detect(self, image):
            m = np.zeros(image.shape[:2], dtype=np.float32)
            m[:3, :] = 1.0  # top strip = sky
            return m

    registry = AIEngineRegistry()
    registry.register_sky_mask(_FakeSky())
    image = np.zeros((10, 10, 3), dtype=np.float32)
    mask = sky_mask(image, registry)
    assert mask[0, 5] == 1.0
    assert mask[9, 5] == 0.0


def test_registered_engine_output_is_clipped_to_zero_one():
    class _OutOfRangeEngine(SubjectMaskEngine):
        def detect(self, image):
            return np.full(image.shape[:2], 5.0, dtype=np.float32)

    registry = AIEngineRegistry()
    registry.register_subject_mask(_OutOfRangeEngine())
    image = np.zeros((4, 4, 3), dtype=np.float32)
    mask = subject_mask(image, registry)
    assert mask.max() <= 1.0


def test_uses_default_registry_when_none_given():
    image = np.zeros((5, 5, 3), dtype=np.float32)
    mask = subject_mask(image)  # no registry passed - falls back to core.ai.default_registry
    assert mask.shape == (5, 5)
