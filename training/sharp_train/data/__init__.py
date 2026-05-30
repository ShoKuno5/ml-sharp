"""Datasets and the batch contract for dual-branch render supervision."""

from .batch import BatchSample
from .synthetic_dummy import DummyDataset, make_dummy_sample

__all__ = ["BatchSample", "DummyDataset", "make_dummy_sample"]
