"""
train_twitter.py
================
Training script for the Twitter Social Circles dataset (SNAP).
Dataset: ego-Twitter
  - Nodes: Twitter users
  - Edges: Follow relationships
  - Features: User profile features (anonymized)
  - Task: Link prediction + Node classification (community detection)

Run on Kaggle GPU:
    python train_twitter.py

Outputs:
    weights/model_weights_twitter.pth
    weights/embeddings_twitter.npy
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
from torch_geometric.datasets import SNAPDataset
from sklearn.metrics import roc_auc_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gnn_model import GATSocialGNN
from model.utils import (
    set_seed, build_structural_features, get_negative_samples,
    save_model_and_embeddings, EarlyStopping, count_parameters, normalize_features
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DATASET_NAME = "twitter"
FEATURE_DIM = 128
HIDDEN_DIM = 128
EMBEDDING_DIM = 64
NUM_CLASSES = 4
GAT_HEADS = 4
DROPOUT = 0.4
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 300
PATIENCE = 25
LINK_LOSS_WEIGHT = 0.6  # Twitter is more link-heavy
OUTPUT_DIR = "weights"
SEED = 42


# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def load_twitter_dataset() -> Data:
    """
    Load Twitter ego-network dataset.

    Option 1: PyG SNAPDataset (Twitter)
    Option 2: Kaggle-mounted Twitter dataset
    Option 3: Synthetic fallback
    """
    # Try PyG built-in SNAP
    try:
        logger.info("Attempting to load Twitter dataset via PyG SNAPDataset...")
        dataset = SNAPDataset(root="/tmp/snap", name="ego-Twitter")
        # ego-Twitter has multiple graphs; merge them
        all_edges_src, all_edges_dst = [], []
        offset = 0

        for d in dataset:
            if d.edge_index.size(1) == 0:
                continue
            all_edges_src.append(d.edge_index[0] + offset)
            all_edges_dst.append(d.edge_index[1] + offset)
            offset += d.num_nodes

        edge_src = torch.cat(all_edges_src)
        edge_dst = torch.cat(all_edges_dst)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_index = to_undirected(edge_index)

        num_nodes = offset
        logger.info(f"Loaded Twitter: {num_nodes} nodes, {edge_index.size(1)} edges")

        x = build_structural_features(num_nodes, FEATURE_DIM)
        y = _assign_community_labels(edge_index, num_nodes)

        return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    except Exception as e:
        logger.warning(f"Could not load SNAP Twitter dataset: {e}")

    # Try Kaggle path
    kaggle_path = "/kaggle/input/twitter-social-circles/twitter_edges.csv"
    if os.path.exists(kaggle_path):
        import pandas as pd
        df = pd.read_csv(kaggle_path)
        src = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        dst = torch.tensor(df.iloc[:, 1].values, dtype=torch.long)
        edge_index = to_undirected(torch.stack([src, dst]))
        num_nodes = edge_index.max().item() + 1
        x = build_structural_features(num_nodes, FEATURE_DIM)
        y = _assign_community_labels(edge_index, num_nodes)
        return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    logger.warning("Falling back to synthetic Twitter-like graph...")
    return _generate_twitter_synthetic()


def _assign_community_labels(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Assign pseudo-community labels based on degree quartiles.
    0: lurker, 1: regular, 2: active, 3: influencer
    """
    degrees = torch.zeros(num_nodes)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1)))

    q33, q66, q90 = (
        degrees.quantile(0.33).item(),
        degrees.quantile(0.66).item(),
        degrees.quantile(0.90).item(),
    )

    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[degrees > q33] = 1
    labels[degrees > q66] = 2
    labels[degrees > q90] = 3
    return labels


def _generate_twitter_synthetic(num_nodes: int = 8000, num_edges: int = 80000) -> Data:
    """Generate a Twitter-like directed graph (power law follow structure)."""
    logger.info(f"Generating synthetic Twitter graph: {num_nodes} nodes")
    alpha = 2.5
    degrees = np.random.zipf(alpha, num_nodes).astype(float)
    degrees = np.clip(degrees, 1, 500)

    src_list, dst_list = [], []
    for _ in range(num_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.choice(num_nodes, p=degrees / degrees.sum())
        if src != dst:
            src_list.append(src)
            dst_list.append(dst)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    x = build_structural_features(num_nodes, FEATURE_DIM)
    y = _assign_community_labels(edge_index, num_nodes)

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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()

    return {"total": loss.item(), "node": node_loss.item(), "link": link_loss.item()}


@torch.no_grad()
def evaluate(model, data, device, split="val"):
    model.eval()
    x = data.x.to(device)
    train_edge = data.train_pos_edge_index.to(device)
    z = model.encode(x, train_edge)

    pos_edge = getattr(data, f"{split}_pos_edge_index").to(device)
    neg_edge = getattr(data, f"{split}_neg_edge_index").to(device)

    pos_pred = model.predict_link(z, pos_edge).cpu().numpy()
    neg_pred = model.predict_link(z, neg_edge).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    y_pred = np.concatenate([pos_pred, neg_pred])
    auc = roc_auc_score(y_true, y_pred)

    logits = model.classify_node(z)
    preds = logits.argmax(dim=-1).cpu().numpy()
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

    raw_data = load_twitter_dataset()
    data = train_test_split_edges(raw_data, val_ratio=0.05, test_ratio=0.1)
    data.num_nodes = raw_data.num_nodes
    data.y = raw_data.y

    model = GATSocialGNN(
        in_channels=FEATURE_DIM,
        hidden_channels=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        heads=GAT_HEADS,
        dropout=DROPOUT,
    ).to(device)

    logger.info(f"GAT Model params: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = f"{args.output_dir}/model_weights_{DATASET_NAME}_best.pth"
    stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, optimizer, data, device)

        if epoch % 10 == 0:
            metrics = evaluate(model, data, device, "val")
            scheduler.step(1 - metrics["link_auc"])
            logger.info(
                f"Epoch {epoch:03d} | Loss: {losses['total']:.4f} | "
                f"AUC: {metrics['link_auc']:.4f} | F1: {metrics['node_f1']:.4f}"
            )
            if stopper(metrics["link_auc"], model, best_path):
                logger.info("Early stopping.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_m = evaluate(model, data, device, "test")
    logger.info(f"TEST | AUC: {test_m['link_auc']:.4f} | F1: {test_m['node_f1']:.4f}")

    torch.save(model.state_dict(), f"{args.output_dir}/model_weights_{DATASET_NAME}.pth")
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device))
    np.save(f"{args.output_dir}/embeddings_{DATASET_NAME}.npy", z.cpu().numpy())
    logger.info("Done.")


if __name__ == "__main__":
    main()
