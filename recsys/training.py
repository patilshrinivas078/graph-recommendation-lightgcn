from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as data_module
from .config import Config
from .models import LightGCNRecommender, MatrixFactorization
from .retrieval import evaluate, sample_negatives
from .utils import get_device, get_logger, seed_everything

logger = get_logger(__name__)


def train_loop(
    model: torch.nn.Module,
    model_name: str,
    train_users: np.ndarray,
    train_items: np.ndarray,
    seen_mask: torch.Tensor,
    num_items: int,
    val_targets: dict[int, int],
    val_mask_items: dict[int, np.ndarray],
    edge_index: torch.Tensor | None,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[list[dict], dict, int, float]:
    """Returns (history, best_state_dict, best_epoch, best_target_metric)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_metric, best_epoch, best_state = -1.0, -1, None
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = rng.permutation(len(train_users))
        total_loss = 0.0

        for start in range(0, len(perm), config.batch_size):
            idx = perm[start : start + config.batch_size]
            u_t = torch.as_tensor(train_users[idx], device=device)
            pos_t = torch.as_tensor(train_items[idx], device=device)
            neg_t = sample_negatives(u_t, seen_mask, num_items)

            optimizer.zero_grad()
            loss = model.compute_loss(u_t, pos_t, neg_t, edge_index=edge_index, lambda_reg=config.lambda_reg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if epoch % config.eval_every == 0 or epoch == 1:
            model.eval()
            metrics = evaluate(
                lambda b: model.score_all(b, edge_index=edge_index),
                val_targets,
                val_mask_items,
                ks=config.ks,
                device=device,
                batch_size=config.eval_batch_size,
            )
            avg_loss = total_loss / len(perm)
            history.append({"epoch": epoch, "model": model_name, "loss": avg_loss, **metrics})
            logger.info(
                "[%s] epoch %3d | loss %.4f | val %s %.4f",
                model_name, epoch, avg_loss, config.target_metric, metrics[config.target_metric],
            )

            if metrics[config.target_metric] > best_metric:
                best_metric, best_epoch = metrics[config.target_metric], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= config.patience:
                logger.info("[%s] early stop at epoch %d (best was %d)", model_name, epoch, best_epoch)
                break

    assert best_state is not None, "no evaluation ever ran -- check eval_every/epochs"
    model.load_state_dict(best_state)
    logger.info("[%s] restored best checkpoint from epoch %d", model_name, best_epoch)
    return history, best_state, best_epoch, best_metric


def _train_test_pairs(data: data_module.SequenceData) -> tuple[np.ndarray, np.ndarray]:
    train_users = np.concatenate([np.full(len(v), u) for u, v in data.train.items()])
    train_items = np.concatenate([np.array(v) for v in data.train.values()])
    return train_users, train_items


def run(model_name: str, config: Config) -> dict:
    """Loads data once, trains the requested model, evaluates on test, and
    writes checkpoint/history/metrics under config.artifacts_dir."""
    assert model_name in ("mf", "lightgcn")
    seed_everything(config.seed)
    device = get_device()

    data = data_module.load_movielens(
        data_module.download_movielens(config.data_dir, config.ml1m_url),
        min_user_interactions=config.min_user_interactions,
        min_item_interactions=config.min_item_interactions,
        min_rating=config.min_rating,
    )
    data_module.save_mappings(data, config.mappings_path)

    edge_index = data_module.build_edge_index(data)
    torch.save(edge_index, config.edge_index_path)
    edge_index = edge_index.to(device) if model_name == "lightgcn" else None

    observed = data_module.build_observed_items(data)
    seen_mask = data_module.build_seen_mask(data, observed, device)
    val_mask_items, test_mask_items = data_module.build_eval_masks(data)

    train_users, train_items = _train_test_pairs(data)
    rng = np.random.default_rng(config.seed)

    if model_name == "lightgcn":
        model = LightGCNRecommender(data.num_users, data.num_items, embedding_dim=config.embedding_dim, num_layers=config.num_layers).to(device)
    else:
        model = MatrixFactorization(data.num_users, data.num_items, dim=config.embedding_dim).to(device)

    history, _, best_epoch, best_metric = train_loop(
        model, model_name, train_users, train_items, seen_mask, data.num_items,
        data.validation, val_mask_items, edge_index, config, device, rng,
    )

    test_metrics = evaluate(
        lambda b: model.score_all(b, edge_index=edge_index),
        data.test, test_mask_items, ks=config.ks, device=device, batch_size=config.eval_batch_size,
    )
    for k in config.ks:
        logger.info("[%s] test recall@%-2d %.4f  ndcg@%-2d %.4f", model_name, k, test_metrics[f"recall@{k}"], k, test_metrics[f"ndcg@{k}"])

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_config = (
        {"num_users": data.num_users, "num_items": data.num_items, "embedding_dim": config.embedding_dim, "num_layers": config.num_layers}
        if model_name == "lightgcn"
        else {"num_users": data.num_users, "num_items": data.num_items, "dim": config.embedding_dim}
    )
    torch.save({"config": model_config, "state_dict": model.state_dict()}, config.checkpoint_path(model_name))
    config.history_path(model_name).write_text(json.dumps(history), encoding="utf-8")
    config.test_metrics_path(model_name).write_text(json.dumps(test_metrics), encoding="utf-8")

    return {"best_epoch": best_epoch, "best_val_metric": best_metric, "test_metrics": test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGCN and/or MF on MovieLens-1M")
    parser.add_argument("--model", choices=["mf", "lightgcn", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.artifacts_dir is not None:
        config.artifacts_dir = args.artifacts_dir

    models = ["mf", "lightgcn"] if args.model == "both" else [args.model]
    for model_name in models:
        run(model_name, config)


if __name__ == "__main__":
    main()
