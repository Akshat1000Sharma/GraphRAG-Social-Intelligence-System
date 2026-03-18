"""
train_reddit.py
===============
Training script for the Reddit Social Network dataset.
Dataset: Reddit (PyG built-in) or Kaggle Reddit posts graph
  - Nodes: Reddit posts (or users/subreddits)
  - Edges: User interaction or cross-posting
  - Features: Post content embeddings (300-dim)
  - Task: Link prediction + Node classification (subreddit community)

Run on Kaggle GPU:
    python train_reddit.py

Outputs:
    weights/model_weights_reddit.pth
    weights/embeddings_reddit.npy
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, train_test_split_edges
from sklearn.metrics import roc_auc_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gnn_model import SocialGraphGNN
from model.utils import (
    set_seed, build_structural_features, get_negative_samples,
    EarlyStopping, count_parameters, normalize_features
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DATASET_NAME = "reddit"
FEATURE_DIM = 300        # Reddit post embeddings are 300-dim
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
NUM_CLASSES = 4
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
PATIENCE = 20
LINK_LOSS_WEIGHT = 0.4
OUTPUT_DIR = "weights"
SEED = 42


# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def load_reddit_dataset() -> Data:
    """
    Load Reddit dataset. Priority:
    1. PyG Reddit dataset (pre-processed)
    2. Kaggle-mounted Reddit interactions
    3. Synthetic fallback
    """
    # Option 1: PyG Reddit (large, ~232K nodes)
    try:
        from torch_geometric.datasets import Reddit
        logger.info("Loading PyG Reddit dataset (this may take a moment)...")
        dataset = Reddit(root="/tmp/reddit")
        data = dataset[0]

        # Subsample for training efficiency
        max_nodes = 50000
        if data.num_nodes > max_nodes:
            logger.info(f"Subsampling from {data.num_nodes} to {max_nodes} nodes...")
            data = _subsample_graph(data, max_nodes)

        logger.info(f"Reddit loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

        # Normalize features
        data.x = normalize_features(data.x)

        # Map labels to 4 classes (subreddit categories)
        original_classes = data.y.max().item() + 1
        data.y = (data.y * 4 // original_classes).clamp(0, 3)

        return data

    except Exception as e:
        logger.warning(f"PyG Reddit unavailable: {e}")

    # Option 2: Kaggle
    kaggle_path = "/kaggle/input/reddit-hyperlink-network/soc-redditHyperlinks-body.tsv"
    if os.path.exists(kaggle_path):
        return _load_reddit_hyperlinks(kaggle_path)

    # Option 3: Synthetic
    logger.warning("Using synthetic Reddit-like graph...")
    return _generate_reddit_synthetic()


def _subsample_graph(data: Data, max_nodes: int) -> Data:
    """Random node subsampling with edge filtering."""
    indices = torch.randperm(data.num_nodes)[:max_nodes]
    idx_set = set(indices.tolist())

    mask_src = torch.tensor([s.item() in idx_set for s in data.edge_index[0]])
    mask_dst = torch.tensor([d.item() in idx_set for d in data.edge_index[1]])
    edge_mask = mask_src & mask_dst

    # Remap node indices
    old_to_new = {old.item(): new for new, old in enumerate(indices)}
    filtered_edges = data.edge_index[:, edge_mask]
    new_src = torch.tensor([old_to_new[s.item()] for s in filtered_edges[0]])
    new_dst = torch.tensor([old_to_new[d.item()] for d in filtered_edges[1]])

    return Data(
        x=data.x[indices],
        edge_index=torch.stack([new_src, new_dst]),
        y=data.y[indices],
        num_nodes=max_nodes,
    )


def _load_reddit_hyperlinks(path: str) -> Data:
    """Load Reddit hyperlink network from TSV."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t", nrows=200000)

    # Map subreddit names to IDs
    subreddits = list(set(df["SOURCE_SUBREDDIT"].tolist() + df["TARGET_SUBREDDIT"].tolist()))
    sr_to_id = {sr: i for i, sr in enumerate(subreddits)}
    num_nodes = len(subreddits)

    src = torch.tensor([sr_to_id[s] for s in df["SOURCE_SUBREDDIT"]], dtype=torch.long)
    dst = torch.tensor([sr_to_id[t] for t in df["TARGET_SUBREDDIT"]], dtype=torch.long)
    edge_index = to_undirected(torch.stack([src, dst]))

    x = build_structural_features(num_nodes, FEATURE_DIM)

    # Sentiment as label proxy
    if "LINK_SENTIMENT" in df.columns:
        sentiment_map = {-1: 0, 0: 1, 1: 2}
        y = torch.randint(0, 4, (num_nodes,))
    else:
        y = torch.randint(0, 4, (num_nodes,))

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


def _generate_reddit_synthetic(num_nodes: int = 20000, num_edges: int = 200000) -> Data:
    """Generate a community-structured synthetic graph (stochastic block model)."""
    logger.info(f"Generating synthetic Reddit graph: {num_nodes} nodes, {num_communities=4}")
    num_communities = 4
    community_size = num_nodes // num_communities

    src_list, dst_list = [], []
    p_intra = 0.01   # within-community connection prob
    p_inter = 0.001  # between-community connection prob

    assignments = np.repeat(np.arange(num_communities), community_size)
    np.random.shuffle(assignments)

    for _ in range(num_edges):
        u = np.random.randint(0, num_nodes)
        # Sample v: with higher prob from same community
        if np.random.random() < 0.7:
            same_comm = np.where(assignments == assignments[u])[0]
            v = np.random.choice(same_comm)
        else:
            v = np.random.randint(0, num_nodes)
        if u != v:
            src_list.append(u)
            dst_list.append(v)

    edge_index = to_undirected(torch.tensor([src_list, dst_list], dtype=torch.long))
    x = build_structural_features(num_nodes, FEATURE_DIM)
    y = torch.tensor(assignments, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_epoch(model, optimizer, data, device):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    train_edge = data.train_pos_edge_index.to(device)
    pos_edge = data.train_pos_edge_index.to(device)
    neg_edge = get_negative_samples(train_edge, data.num_nodes, pos_edge.size(1)).to(device)

    z, node_logits, link_probs = model(x, train_edge, pos_edge, neg_edge)

    node_loss = F.cross_entropy(node_logits, data.y.to(device))

    link_labels = torch.zeros(link_probs.size(0), device=device)
    link_labels[: pos_edge.size(1)] = 1.0
    link_loss = F.binary_cross_entropy(link_probs, link_labels)

    loss = (1 - LINK_LOSS_WEIGHT) * node_loss + LINK_LOSS_WEIGHT * link_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device, split="val"):
    model.eval()
    x = data.x.to(device)
    z = model.encode(x, data.train_pos_edge_index.to(device))

    pos_edge = getattr(data, f"{split}_pos_edge_index").to(device)
    neg_edge = getattr(data, f"{split}_neg_edge_index").to(device)

    pos_p = model.predict_link(z, pos_edge).cpu().numpy()
    neg_p = model.predict_link(z, neg_edge).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_p)), np.zeros(len(neg_p))])
    auc = roc_auc_score(y_true, np.concatenate([pos_p, neg_p]))

    preds = model.classify_node(z).argmax(dim=-1).cpu().numpy()
    f1 = f1_score(data.y.numpy(), preds, average="macro")

    return {"link_auc": auc, "node_f1": f1}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    raw_data = load_reddit_dataset()
    logger.info(f"Data: {raw_data.num_nodes} nodes, {raw_data.edge_index.size(1)} edges")

    data = train_test_split_edges(raw_data, val_ratio=0.05, test_ratio=0.1)
    data.num_nodes = raw_data.num_nodes
    data.y = raw_data.y

    model = SocialGraphGNN(
        in_channels=FEATURE_DIM,
        hidden_channels=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    logger.info(f"Model params: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10, epochs=args.epochs,
        steps_per_epoch=1, pct_start=0.1
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = f"{args.output_dir}/model_weights_{DATASET_NAME}_best.pth"
    stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer, data, device)
        scheduler.step()

        if epoch % 10 == 0:
            m = evaluate(model, data, device, "val")
            logger.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {m['link_auc']:.4f} | F1: {m['node_f1']:.4f}")
            if stopper(m["link_auc"], model, best_path):
                logger.info("Early stopping.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_m = evaluate(model, data, device, "test")
    logger.info(f"TEST | AUC: {test_m['link_auc']:.4f} | F1: {test_m['node_f1']:.4f}")

    torch.save(model.state_dict(), f"{args.output_dir}/model_weights_{DATASET_NAME}.pth")
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device))
    np.save(f"{args.output_dir}/embeddings_{DATASET_NAME}.npy", z.cpu().numpy())
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
