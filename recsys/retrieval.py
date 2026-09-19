"""Negative sampling (used during training) and top-k retrieval evaluation
(recall@k / ndcg@k, used for validation early-stopping and final test metrics).

Both consume the seen_mask / mask_items built in data.py so training negatives
and evaluation candidates are always masked consistently.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def sample_negatives(users: torch.Tensor, seen_mask: torch.Tensor, num_items: int) -> torch.Tensor:
    """Vectorized rejection sampling against seen_mask -- stays on-device,
    no per-example Python membership checks."""
    sampled = torch.randint(0, num_items, (users.size(0),), device=users.device)
    invalid = seen_mask[users, sampled]
    while invalid.any():
        sampled[invalid] = torch.randint(0, num_items, (int(invalid.sum()),), device=users.device)
        invalid = seen_mask[users, sampled]
    return sampled


@torch.no_grad()
def evaluate(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    targets: dict[int, int],
    mask_items: dict[int, np.ndarray],
    ks: tuple[int, ...],
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    """score_fn(user_indices) -> (batch, num_items) scores. Masks out each
    user's already-seen items, then reports recall@k / ndcg@k for the single
    held-out target per user."""
    max_k = max(ks)
    recall = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    users = np.array(sorted(targets.keys()))

    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = score_fn(torch.as_tensor(batch, device=device))

        for row, u in enumerate(batch):
            scores[row, torch.as_tensor(mask_items[u], device=device)] = float("-inf")

        tgt = torch.as_tensor([targets[u] for u in batch], device=device)
        topk = scores.topk(max_k, dim=1).indices
        hits = topk == tgt.unsqueeze(1)

        for k in ks:
            hit_k = hits[:, :k].any(dim=1).float()
            pos = hits[:, :k].float().argmax(dim=1).float()
            recall[k] += hit_k.sum().item()
            ndcg[k] += (hit_k / torch.log2(pos + 2)).sum().item()

    n = len(users)
    return {
        **{f"recall@{k}": recall[k] / n for k in ks},
        **{f"ndcg@{k}": ndcg[k] / n for k in ks},
    }
