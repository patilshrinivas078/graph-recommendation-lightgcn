# LightGCN vs. Matrix Factorization on MovieLens-1M

Comparison of two models:

- Matrix Factorization — BPR-trained user/item embeddings plus an item bias;
- LightGCN — a graph convolutional recommender that propagates embeddings over the bipartite user-item interaction graph before scoring.

Both models are trained on the same chronological leave-two-out split, optimized with the same BPR loss, and evaluated with recall@k / ndcg@k, so the comparison isn't confounded by different objectives or data.

## Measured results

Full-catalog recall and NDCG on the held-out test interaction, one seed.

| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| LightGCN | 0.0469 | 0.0768 | 0.1376 | 0.0302 | 0.0397 | 0.0549 |
| Matrix factorization | **0.0606** | **0.1033** | **0.1634** | **0.0387** | **0.0524** | **0.0676** |

Matrix factorization won on every cutoff. This isn't a claim that graph propagation never helps. MovieLens-1M is small and fairly dense, so a 2-3 layer GCN has little sparse-neighborhood signal to exploit that a direct pairwise embedding can't already capture, while adding more parameters and a slower per-epoch forward pass (every batch requires message passing over the whole graph, not just the batch's user/item embeddings). Neither model's hyperparameters were tuned beyond the defaults in `recsys/config.py`, so this is a starting point rather than a final verdict on either architecture.

## Evaluation setup

For every user, the last interaction is the test target and the second-to-last is the validation target. Everything before those two is training data (leave-two-out).

During training, negatives are sampled uniformly while excluding every item a user has ever interacted with: train, validation, or test. So a training negative can never accidentally be an evaluation target. During evaluation, the target is ranked against the *entire* item catalog after already-seen items are masked out, rather than against a small sampled negative set, since sampled-negative evaluation can make ranking quality look better than it is.

I report NDCG and Recall at 5/10/20 rather than a single cutoff or classification accuracy.

## Structure

```
recsys/
  config.py      # all hyperparameters + paths (Config dataclass)
  utils.py       # seeding, device selection, logging
  data.py        # MovieLens download/split, ID mappings, edge index, masks
  models.py      # MatrixFactorization, LightGCNRecommender, bpr_loss
  retrieval.py   # negative sampling, evaluate() (recall@k / ndcg@k)
  training.py    # generic train_loop (works for both models) + CLI
scripts/
  compare_models.py  # comparison table + training-curve plots
artifacts/       # checkpoints, mappings.json, history/metrics JSON (git-ignored)
```

`models.py` gives both models the same interface --
`compute_loss(users, pos_items, neg_items, edge_index=..., lambda_reg=...)` and
`score_all(users, edge_index=...)` -- so `training.py` has one training loop
instead of two near-duplicates. `edge_index` is `None`/ignored for MF and
required for LightGCN (message passing needs the graph on every forward pass).

`data.py` also keeps `index_to_user` / `index_to_item` (the reverse of the
saved `user_to_index` / `item_to_index`), so a future `serve.py` can accept
and return original MovieLens IDs instead of the model's internal contiguous
indices.

## Usage

Install dependencies using uv
```bash
uv sync
```

### Training on MovieLens-1M

`recsys.training` downloads MovieLens-1M from GroupLens the first time it runs, applies a 5-core filter (users and items need at least 5 interactions), and trains either or both models:

```bash
python -m recsys.training --model both
```

Both models share one training loop (`recsys/training.py`) since `models.py` gives `MatrixFactorization` and `LightGCNRecommender` the same interface `compute_loss(...)` and `score_all(...)`, so the only branch between them is whether the graph edge index gets passed through.

Checkpoints, ID mappings, per-epoch history, and test metrics are written to `artifacts/`. Hyperparameters (embedding dim, learning rate, epochs, patience, ...) live in `recsys/config.py` and can be overridden by constructing a `Config` yourself or via the `--epochs`/`--artifacts-dir` CLI flags.

## Comparing results

```bash
python scripts/compare_models.py --artifacts-dir artifacts
```

Reads the saved history/metrics JSON and writes a comparison table plus training-curve and bar-chart plots to `artifacts/`.



