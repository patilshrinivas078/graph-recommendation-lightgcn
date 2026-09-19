"""MatrixFactorization and LightGCN:

    model.compute_loss(users, pos_items, neg_items, edge_index=..., lambda_reg=...)
    model.score_all(users, edge_index=...)  -> (len(users), num_items) score matrix

`edge_index` is ignored by MatrixFactorization and required by LightGCNRecommender
(message passing needs the graph on every forward pass).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, reg_embeddings: torch.Tensor, lambda_reg: float = 1e-4) -> torch.Tensor:
    """-logsigmoid(pos - neg).mean() + L2 reg -- identical formula to PyG's
    LightGCN.recommendation_loss, so both models are trained with the same objective."""
    log_prob = F.logsigmoid(pos_scores - neg_scores).mean()
    reg = lambda_reg * reg_embeddings.norm(p=2).pow(2) / pos_scores.size(0)
    return -log_prob + reg


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, dim)
        self.item_embedding = nn.Embedding(num_items, dim)
        self.item_bias = nn.Embedding(num_items, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(users) * self.item_embedding(items)).sum(-1) \
            + self.item_bias(items).squeeze(-1)

    def score_all(self, users: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index  # unused, kept for interface parity with LightGCNRecommender
        return self.user_embedding(users) @ self.item_embedding.weight.T + self.item_bias.weight.T

    def query_vector(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(users)

    def compute_loss(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor, edge_index: torch.Tensor | None = None, lambda_reg: float = 1e-4) -> torch.Tensor:
        del edge_index
        pos_scores = self(users, pos_items)
        neg_scores = self(users, neg_items)
        reg_embeddings = torch.cat([self.user_embedding(users), self.item_embedding(pos_items), self.item_embedding(neg_items)])
        return bpr_loss(pos_scores, neg_scores, reg_embeddings, lambda_reg=lambda_reg)



