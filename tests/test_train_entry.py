"""CPU tests for the training entrypoint wiring (collate, dataset build, epoch cycling).

The full training step needs a CUDA GPU (1536 trunk + gsplat render) and runs on iruka2; here we
check the DataLoader plumbing that is easy to get wrong (passthrough collate, infinite cycling).
"""

from __future__ import annotations

from sharp_train.config import TrainConfig
from sharp_train.data import BatchSample, DummyDataset
from sharp_train.scripts.train import _passthrough_collate, build_dataset, cycle_batches
from torch.utils.data import DataLoader


def test_build_dataset_dummy() -> None:
    """dummy_data=True yields the DummyDataset."""
    dataset = build_dataset(TrainConfig(dummy_data=True, render_size=16))
    assert isinstance(dataset, DummyDataset)


def test_passthrough_collate_unwraps_single() -> None:
    """The collate returns the single BatchSample unchanged (B=1 is already baked in)."""
    sample = build_dataset(TrainConfig(dummy_data=True, render_size=16))[0]
    assert _passthrough_collate([sample]) is sample


def test_cycle_batches_yields_batchsamples() -> None:
    """DataLoader + cycle_batches yields BatchSamples indefinitely across epochs."""
    dataset = build_dataset(TrainConfig(dummy_data=True, render_size=16))
    loader = DataLoader(dataset, batch_size=1, collate_fn=_passthrough_collate, num_workers=0)
    iterator = cycle_batches(loader, sampler=None)
    for _ in range(3):
        batch = next(iterator)
        assert isinstance(batch, BatchSample)
        assert batch.image.shape[0] == 1 and batch.image.shape[1] == 3
        assert batch.gt_pinhole.shape[-1] == 16  # render_size
