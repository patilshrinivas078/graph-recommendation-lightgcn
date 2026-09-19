"""Central place for hyperparameters and paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- data ---
    ml1m_url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    min_rating: float | None = None

    # --- model ---
    embedding_dim: int = 64
    num_layers: int = 3  # LightGCN propagation layers

    # --- training ---
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 8192
    patience: int = 20
    lambda_reg: float = 1e-4
    eval_every: int = 5
    eval_batch_size: int = 1024
    target_metric: str = "ndcg@10"

    # --- evaluation ---
    ks: tuple[int, ...] = field(default_factory=lambda: (5, 10, 20))

    # --- other ---
    seed: int = 42

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)

    @property
    def ratings_path(self) -> Path:
        return self.data_dir / "ml-1m" / "ratings.dat"

    @property
    def mappings_path(self) -> Path:
        return self.artifacts_dir / "mappings.json"

    @property
    def edge_index_path(self) -> Path:
        return self.artifacts_dir / "edge_index.pt"

    def checkpoint_path(self, model_name: str) -> Path:
        return self.artifacts_dir / f"{model_name}.pt"

    def history_path(self, model_name: str) -> Path:
        return self.artifacts_dir / f"{model_name}_history.json"

    def test_metrics_path(self, model_name: str) -> Path:
        return self.artifacts_dir / f"{model_name}_test_metrics.json"
