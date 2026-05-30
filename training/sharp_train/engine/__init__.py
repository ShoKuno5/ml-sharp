"""Training engine: model assembly, the render-supervised step, and the fit loop."""

from .trainer import build_model, build_renderer, fit, train_step

__all__ = ["build_model", "build_renderer", "train_step", "fit"]
