"""A registrable render pipeline: an ordered list of Stages, each either a
set of named adjustment layers or a pure colorspace/geometry transform.

This exists so the render order is data (built once in default_pipeline.py)
rather than logic baked into ImageDocument.render(). New tools extend the
pipeline by registering their layer's name into an existing stage (or a new
stage) instead of editing ImageDocument itself - keeping the pipeline
decoupled from any particular tool's UI, per the "deterministic, internally
configurable pipeline" requirement.
"""

from core.pipeline.stage import Stage
from core.pipeline.pipeline import Pipeline
from core.pipeline.default_pipeline import build_default_pipeline

__all__ = ["Stage", "Pipeline", "build_default_pipeline"]
