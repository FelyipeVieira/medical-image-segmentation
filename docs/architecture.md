# Architecture

Datasets return model-neutral tensors. Preprocessing owns spatial/intensity
semantics. A factory creates networks, Lightning owns training lifecycle, and an
inference service performs full-volume or sliding-window prediction. Metrics and
exports consume plain arrays, allowing new foundation-model adapters without
changing evaluation.

