from core.pipeline.stage import Stage
from core.pipeline.pipeline import Pipeline
from core.processing.color_space import linear_to_display


def build_default_pipeline() -> Pipeline:
    """The application's standard render order: geometry first (purely
    spatial, so it doesn't matter which color stage it runs in), then White
    Balance/Exposure on scene-linear light, then the linear -> display
    transform, then every other tone/color tool on the perceptually-encoded
    image. Each ImageDocument gets its own Pipeline instance (from this
    factory) so per-document layer registrations (e.g. a future tool wired
    in only for one document) can't leak across documents.
    """
    return Pipeline([
        Stage("Geometry", layer_order=["Crop"]),
        Stage("Linear", layer_order=["Temperature", "Tint", "Exposure"]),
        Stage("Linear-to-Display", transform=linear_to_display),
        Stage("Display", layer_order=[
            "Brightness", "Contrast", "Highlights", "Shadows", "Whites", "Blacks",
            "Parametric Curve", "Curve",
            "Vibrance", "Saturation", "Hue", "HSL", "Color Wheels",
        ]),
        # Local (masked) adjustments run last, as a final region-targeted
        # touch-up on top of the finished global grade - standard
        # workflow order (get the global look right first, then dodge/
        # burn/refine specific areas). Any number of "Mask N" layers can
        # coexist (unlike every other tool, capped at one layer per fixed
        # name) - see Stage's dynamic_prefix and
        # ImageDocument.next_mask_name().
        Stage("Masks", dynamic_prefix="Mask "),
    ])
