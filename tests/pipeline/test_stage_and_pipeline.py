"""Tests for core/pipeline: Stage/Pipeline in isolation, using small dummy
layers rather than real adjustment layers, so these test the plumbing
(ordering, dedup-by-name lookup, transform stages, registration) independent
of any particular tool's math."""

import numpy as np
import pytest

from core.pipeline.stage import Stage
from core.pipeline.pipeline import Pipeline


class _AddLayer:
    """A trivial layer: adds a constant to every pixel, and records its name
    for order-of-application assertions."""

    def __init__(self, name, amount, log):
        self._name = name
        self.amount = amount
        self.log = log

    def __str__(self):
        return self._name

    def apply(self, image):
        self.log.append(self._name)
        return image + self.amount


def test_stage_requires_exactly_one_of_layer_order_or_transform():
    with pytest.raises(ValueError):
        Stage("Bad")
    with pytest.raises(ValueError):
        Stage("Bad", layer_order=["A"], transform=lambda x: x)


def test_stage_applies_only_layers_present_in_by_name_in_declared_order():
    log = []
    a = _AddLayer("A", 1.0, log)
    c = _AddLayer("C", 100.0, log)
    stage = Stage("S", layer_order=["A", "B", "C"])
    by_name = {"A": a, "C": c}  # "B" deliberately absent
    image = np.zeros((2, 2, 3), dtype=np.float32)
    out = stage.apply(image, by_name)
    assert log == ["A", "C"]
    assert np.allclose(out, 101.0)


def test_stage_transform_ignores_by_name():
    stage = Stage("T", transform=lambda img: img * 2.0)
    image = np.ones((1, 1, 3), dtype=np.float32)
    out = stage.apply(image, by_name={"whatever": object()})
    assert np.allclose(out, 2.0)


def test_pipeline_runs_stages_in_order():
    log = []
    a = _AddLayer("A", 1.0, log)
    b = _AddLayer("B", 1.0, log)
    pipeline = Pipeline([
        Stage("First", layer_order=["A"]),
        Stage("Double", transform=lambda img: img * 2.0),
        Stage("Second", layer_order=["B"]),
    ])
    image = np.zeros((1, 1, 3), dtype=np.float32)
    out = pipeline.render(image, {"A": a, "B": b})
    # (0 + 1) * 2 + 1 = 3
    assert np.allclose(out, 3.0)
    assert log == ["A", "B"]


def test_pipeline_stage_lookup_by_name():
    pipeline = Pipeline([Stage("Only", layer_order=["A"])])
    assert pipeline.stage("Only").layer_order == ["A"]
    with pytest.raises(KeyError):
        pipeline.stage("Missing")


def test_register_layer_extends_an_existing_stage_once():
    pipeline = Pipeline([Stage("S", layer_order=["A"])])
    pipeline.register_layer("S", "B")
    assert pipeline.stage("S").layer_order == ["A", "B"]
    pipeline.register_layer("S", "B")  # idempotent, no duplicate
    assert pipeline.stage("S").layer_order == ["A", "B"]


def test_register_layer_at_position():
    pipeline = Pipeline([Stage("S", layer_order=["A", "C"])])
    pipeline.register_layer("S", "B", position=1)
    assert pipeline.stage("S").layer_order == ["A", "B", "C"]


def test_register_layer_rejects_transform_stage():
    pipeline = Pipeline([Stage("T", transform=lambda img: img)])
    with pytest.raises(ValueError):
        pipeline.register_layer("T", "X")


def test_add_stage_at_index():
    pipeline = Pipeline([Stage("First", layer_order=[])])
    pipeline.add_stage(Stage("Inserted", layer_order=[]), index=0)
    names = [s.name for s in pipeline._stages]
    assert names == ["Inserted", "First"]
