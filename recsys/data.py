"""MovieLens loading/splitting, ID mapping, the bipartite edge index, and the
observed-item masks shared by negative sampling (retrieval.sample_negatives)
and evaluation (retrieval.evaluate).

`index_to_user` / `index_to_item` are the reverse of `user_to_index` /
`item_to_index` and exist specifically so a future serve.py can translate a
model's internal (contiguous) item indices back to original MovieLens IDs.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class SequenceData:
    train: dict[int, list[int]]
    validation: dict[int, int]
    test: dict[int, int]
    user_to_index: dict[int, int]
    item_to_index: dict[int, int]

    @property
    def num_users(self) -> int:
        return len(self.user_to_index)

    @property
    def num_items(self) -> int:
        return len(self.item_to_index)

    @cached_property
    def index_to_user(self) -> dict[int, int]:
        return {v: k for k, v in self.user_to_index.items()}

    @cached_property
    def index_to_item(self) -> dict[int, int]:
        return {v: k for k, v in self.item_to_index.items()}


def download_movielens(data_dir: Path, url: str) -> Path:
    ratings = data_dir / "ml-1m" / "ratings.dat"
    if ratings.exists():
        return ratings
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ml-1m.zip"
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(data_dir)
    return ratings


def load_movielens(
    path: Path,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    min_rating: float | None = None,
    index_offset: int = 0,  # 0 for LightGCN; 1 to reserve a padding index
) -> SequenceData:
    """Chronological leave-two-out splits with contiguous IDs."""
    frame = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    if min_rating is not None:
        frame = frame[frame.rating >= min_rating]

    # iterative k-core: filtering items can drop users below threshold and vice versa
    min_user = max(min_user_interactions, 3)  # need >=3 for leave-two-out
    while True:
        u_counts = frame.groupby("user_id").size()
        i_counts = frame.groupby("item_id").size()
        pruned = frame[
            frame.user_id.isin(u_counts[u_counts >= min_user].index)
            & frame.item_id.isin(i_counts[i_counts >= min_item_interactions].index)
        ]
        if len(pruned) == len(frame):
            frame = pruned
            break
        frame = pruned

    frame = frame.sort_values(["user_id", "timestamp"], kind="stable")

    users = sorted(frame.user_id.unique().tolist())
    items = sorted(frame.item_id.unique().tolist())
    user_map = {raw: idx + index_offset for idx, raw in enumerate(users)}
    item_map = {raw: idx + index_offset for idx, raw in enumerate(items)}
    frame["user"] = frame.user_id.map(user_map)
    frame["item"] = frame.item_id.map(item_map)

    train, validation, test = {}, {}, {}
    for user, group in frame.groupby("user", sort=False):
        sequence = group.item.astype(int).tolist()
        train[int(user)] = sequence[:-2]
        validation[int(user)] = sequence[-2]
        test[int(user)] = sequence[-1]
    return SequenceData(train, validation, test, user_map, item_map)


def save_mappings(data: SequenceData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_to_index": {str(k): v for k, v in data.user_to_index.items()},
        "item_to_index": {str(k): v for k, v in data.item_to_index.items()},
        "train": {str(k): v for k, v in data.train.items()},
        "validation": {str(k): v for k, v in data.validation.items()},
        "test": {str(k): v for k, v in data.test.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_mappings(path: Path) -> SequenceData:
    """Reconstruct a SequenceData from a mappings.json produced by save_mappings.

    Mainly for serve.py / offline inference: gives you index_to_item /
    item_to_index without re-running the MovieLens split.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SequenceData(
        train={int(k): v for k, v in payload["train"].items()},
        validation={int(k): v for k, v in payload["validation"].items()},
        test={int(k): v for k, v in payload["test"].items()},
        user_to_index={int(k): v for k, v in payload["user_to_index"].items()},
        item_to_index={int(k): v for k, v in payload["item_to_index"].items()},
    )


def build_edge_index(data: SequenceData, index_offset: int = 0) -> torch.Tensor:
    """Undirected user<->item edges in a shared node space (users first, then items)."""
    src, dst = [], []
    for user, seq in data.train.items():
        u = user - index_offset
        for item in seq:
            src.append(u)
            dst.append(data.num_users + (item - index_offset))
    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    return torch.stack(
        [torch.cat([src_t, dst_t]), torch.cat([dst_t, src_t])], dim=0
    )


def build_observed_items(data: SequenceData) -> dict[int, set[int]]:
    """Every item a user has touched across train+val+test.

    Negatives sampled during training, and candidates masked out at eval time,
    must both be drawn from/against this set so eval targets never leak in as
    training negatives and vice versa.
    """
    observed: dict[int, set[int]] = {u: set(v) for u, v in data.train.items()}
    for u, i in data.validation.items():
        observed.setdefault(u, set()).add(i)
    for u, i in data.test.items():
        observed.setdefault(u, set()).add(i)
    return observed


def build_seen_mask(
    data: SequenceData, observed: dict[int, set[int]], device: torch.device
) -> torch.Tensor:
    """Boolean (num_users, num_items) mask used to reject invalid negatives."""
    seen_mask = torch.zeros((data.num_users, data.num_items), dtype=torch.bool, device=device)
    for u, items in observed.items():
        if items:
            seen_mask[u, torch.tensor(list(items), device=device)] = True
    return seen_mask


def build_eval_masks(data: SequenceData) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Items to mask out of the score matrix at eval time.

    val_mask_items: just the train history (val target hasn't been "seen" yet).
    test_mask_items: train history + the val item (so val doesn't count as a hit
    when scoring against the test target).
    """
    val_mask_items = {u: np.array(v, dtype=np.int64) for u, v in data.train.items()}
    test_mask_items = {
        u: np.append(np.array(data.train[u], dtype=np.int64), data.validation[u])
        for u in data.train
    }
    return val_mask_items, test_mask_items
