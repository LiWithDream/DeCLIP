import types

import torch

from training.dcac_loss import compute_dcac_loss
from training.data import GridDistillDataset


def _make_overlap_meta(batch_size, grid_h, grid_w, start, size, device):
    v1_start = torch.tensor([start], dtype=torch.long, device=device).repeat(batch_size, 1)
    v2_start = torch.tensor([start], dtype=torch.long, device=device).repeat(batch_size, 1)
    overlap_size = torch.tensor([size], dtype=torch.long, device=device).repeat(batch_size, 1)
    grid_size = torch.tensor([[grid_h, grid_w]], dtype=torch.long, device=device).repeat(batch_size, 1)
    valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    return {
        "v1_start": v1_start,
        "v2_start": v2_start,
        "overlap_size": overlap_size,
        "grid_size": grid_size,
        "valid": valid,
    }


def test_compute_dcac_loss_directional():
    device = torch.device("cpu")
    batch = 1
    grid_h = 2
    grid_w = 2
    channels = 4
    feat1 = torch.zeros(batch, grid_h * grid_w, channels, device=device)
    feat2 = torch.zeros(batch, grid_h * grid_w, channels, device=device)
    feat1[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    feat2[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    feat1[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    feat2[0, 1] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    overlap_meta = _make_overlap_meta(batch, grid_h, grid_w, start=(0, 0), size=(1, 1), device=device)
    loss = compute_dcac_loss(feat1, feat2, overlap_meta, temp=0.1)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_compute_dcac_loss_invalid_overlap():
    device = torch.device("cpu")
    feat1 = torch.randn(1, 4, 8, device=device)
    feat2 = torch.randn(1, 4, 8, device=device)
    overlap_meta = {
        "v1_start": torch.zeros(1, 2, dtype=torch.long, device=device),
        "v2_start": torch.zeros(1, 2, dtype=torch.long, device=device),
        "overlap_size": torch.zeros(1, 2, dtype=torch.long, device=device),
        "grid_size": torch.tensor([[2, 2]], dtype=torch.long, device=device),
        "valid": torch.zeros(1, dtype=torch.bool, device=device),
    }
    loss = compute_dcac_loss(feat1, feat2, overlap_meta, temp=0.1)
    assert loss.item() == 0.0


def test_overlap_meta_mapping_valid():
    dummy = GridDistillDataset.__new__(GridDistillDataset)
    box1 = (0.0, 0.0, 100.0, 100.0)
    box2 = (50.0, 50.0, 150.0, 150.0)
    output_size = 224
    patch_size = 16
    meta = GridDistillDataset._compute_overlap_meta(dummy, box1, box2, output_size, patch_size)
    assert bool(meta["valid"].item()) is True
    assert meta["overlap_size"].min().item() > 0
