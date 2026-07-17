# Usage

The manifest requires `image` and `mask`. Split cases by patient before fitting.
Instantiate `ModelFactory.create`, wrap it in `SegmentationTask`, and use
`create_trainer(...).fit(task, train_loader, validation_loader,
ckpt_path="last")`. For inference, reconstruct the identical model config, load
weights, then use `InferenceEngine` with a training-compatible ROI size.

