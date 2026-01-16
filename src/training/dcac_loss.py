import torch
import torch.nn.functional as F


def _build_overlap_indices(start, size, grid_w, device):
    y0 = int(start[0])
    x0 = int(start[1])
    h = int(size[0])
    w = int(size[1])
    if h <= 0 or w <= 0:
        return None
    ys = torch.arange(h, device=device)
    xs = torch.arange(w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return (grid_y + y0) * grid_w + (grid_x + x0)


def compute_dcac_loss(feat1, feat2, overlap_meta, temp=0.1, extra_negatives=None):
    if feat1 is None or feat2 is None or overlap_meta is None:
        return torch.tensor(0.0, device=feat1.device if feat1 is not None else "cpu")

    device = feat1.device
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)

    valid = overlap_meta["valid"]
    if valid.numel() == 0 or valid.sum().item() == 0:
        return torch.tensor(0.0, device=device)

    features1 = []
    features2 = []
    offsets = []
    offset = 0
    for i in range(feat1.shape[0]):
        if not bool(valid[i].item()):
            continue
        grid_w = int(overlap_meta["grid_size"][i, 1].item())
        idx1 = _build_overlap_indices(overlap_meta["v1_start"][i], overlap_meta["overlap_size"][i], grid_w, device)
        idx2 = _build_overlap_indices(overlap_meta["v2_start"][i], overlap_meta["overlap_size"][i], grid_w, device)
        if idx1 is None or idx2 is None:
            continue
        idx1 = idx1.flatten()
        idx2 = idx2.flatten()
        if idx1.numel() == 0 or idx2.numel() == 0:
            continue
        if idx1.max().item() >= feat1.size(1) or idx2.max().item() >= feat2.size(1):
            continue
        f1 = feat1[i].index_select(0, idx1)
        f2 = feat2[i].index_select(0, idx2)
        features1.append(f1)
        features2.append(f2)
        offsets.append(offset)
        offset += f2.shape[0]

    if not features1 or offset == 0:
        return torch.tensor(0.0, device=device)

    keys2 = torch.cat(features2, dim=0)
    keys1 = torch.cat(features1, dim=0)
    if extra_negatives is not None and extra_negatives.numel() > 0:
        extra_negatives = F.normalize(extra_negatives, dim=-1)

    loss_12 = torch.tensor(0.0, device=device)
    total_q = 0
    for f1, off in zip(features1, offsets):
        keys = keys2 if extra_negatives is None else torch.cat([keys2, extra_negatives], dim=0)
        logits = (f1 @ keys.T) / temp
        labels = torch.arange(f1.shape[0], device=device) + off
        loss_12 = loss_12 + F.cross_entropy(logits, labels, reduction="sum")
        total_q += f1.shape[0]
    loss_12 = loss_12 / max(total_q, 1)

    loss_21 = torch.tensor(0.0, device=device)
    total_q = 0
    for f2, off in zip(features2, offsets):
        keys = keys1 if extra_negatives is None else torch.cat([keys1, extra_negatives], dim=0)
        logits = (f2 @ keys.T) / temp
        labels = torch.arange(f2.shape[0], device=device) + off
        loss_21 = loss_21 + F.cross_entropy(logits, labels, reduction="sum")
        total_q += f2.shape[0]
    loss_21 = loss_21 / max(total_q, 1)

    return 0.5 * (loss_12 + loss_21)
