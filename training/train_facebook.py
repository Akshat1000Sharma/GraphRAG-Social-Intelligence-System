"""
train_facebook.py
=================
Training script for the Facebook Large Page-Page Network dataset.
Dataset: facebook_large (Kaggle / SNAP)
  - Nodes: Facebook pages
  - Edges: Mutual likes
  - Node features: Not provided → use structural features
  - Task: Link prediction + Node classification

Run on Kaggle GPU:
    python train_facebook.py

Outputs:
    weights/model_weights_facebook.pth
    weights/embeddings_facebook.npy
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

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gnn_model import SocialGraphGNN
from model.utils import (
    set_seed, build_structural_features, get_negative_samples,
    save_model_and_embeddings, EarlyStopping, count_parameters
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DATASET_NAME = "facebook"
FEATURE_DIM = 128
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
NUM_CLASSES = 4
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 2048
LINK_LOSS_WEIGHT = 0.5
OUTPUT_DIR = "weights"
SEED = 42


# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def load_facebook_dataset() -> Data:
    """
    Load Facebook Large Page-Page Network.

    If running on Kaggle with dataset mounted, load from CSV.
    Otherwise, falls back to synthetic graph for testing.

    Kaggle dataset path: /kaggle/input/facebook-large-page-page-network/
    """
    kaggle_path = "/kaggle/input/facebook-large-page-page-network/musae_facebook_edges.csv"

    if os.path.exists(kaggle_path):
        logger.info("Loading Facebook dataset from Kaggle...")
        import pandas as pd

        edges_df = pd.read_csv(kaggle_path)
        edge_src = torch.tensor(edges_df["id_1"].values, dtype=torch.long)
        edge_dst = torch.tensor(edges_df["id_2"].values, dtype=torch.long)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_index = to_undirected(edge_index)

        num_nodes = edge_index.max().item() + 1
        logger.info(f"Nodes: {num_nodes}, Edges: {edge_index.size(1)}")

        # Try loading features
        features_path = "/kaggle/input/facebook-large-page-page-network/musae_facebook_features.json"
        if os.path.exists(features_path):
            import json
            with open(features_path) as f:
                feat_dict = json.load(f)
            # Multi-hot encoding of page categories
            max_feat = max(max(v) for v in feat_dict.values() if v) + 1
            x = torch.zeros(num_nodes, min(max_feat, FEATURE_DIM))
            for node_id_str, feats in feat_dict.items():
                nid = int(node_id_str)
                for f_idx in feats:
                    if f_idx < FEATURE_DIM:
                        x[nid, f_idx] = 1.0
        else:
            x = build_structural_features(num_nodes, FEATURE_DIM)

        # Load labels if available
        target_path = "/kaggle/input/facebook-large-page-page-network/musae_facebook_target.csv"
        if os.path.exists(target_path):
            target_df = pd.read_csv(target_path)
            y = torch.zeros(num_nodes, dtype=torch.long)
            # page_type as label: map to 0-3
            type_map = {t: i for i, t in enumerate(target_df["page_type"].unique())}
            for _, row in target_df.iterrows():
                y[row["id"]] = type_map.get(row["page_type"], 0)
        else:
            y = torch.randint(0, NUM_CLASSES, (num_nodes,))

        return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    else:
        logger.warning("Kaggle dataset not found. Using synthetic graph for testing.")
        return _generate_synthetic_graph(num_nodes=5000, num_edges=50000)


def _generate_synthetic_graph(num_nodes: int = 5000, num_edges: int = 50000) -> Data:
    """Generate a synthetic scale-free-like graph for testing."""
    logger.info(f"Generating synthetic graph: {num_nodes} nodes, {num_edges} edges")
    # Preferential attachment (simplified)
    edges_src, edges_dst = [], []
    degrees = np.ones(num_nodes)

    for _ in range(num_edges):
        probs = degrees / degrees.sum()
        src = np.random.choice(num_nodes, p=probs)
        dst = np.random.choice(num_nodes)
        if src != dst:
            edges_src.append(src)
            edges_dst.append(dst)
            degrees[src] += 1
            degrees[dst] += 1

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    x = build_structural_features(num_nodes, FEATURE_DIM)
    y = torch.randint(0, NUM_CLASSES, (num_nodes,))

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_epoch(model, optimizer, data, device):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    # Use train edges for message passing
    train_edge_index = data.train_pos_edge_index.to(device)

    # Positive and negative edges for link prediction loss
    pos_edge = data.train_pos_edge_index.to(device)
    neg_edge = get_negative_samples(train_edge_index, data.num_nodes, pos_edge.size(1))
    neg_edge = neg_edge.to(device)

    z, node_logits, link_probs = model(x, train_edge_index, pos_edge, neg_edge)

    # Node classification loss
    node_labels = data.y.to(device)
    node_loss = F.cross_entropy(node_logits, node_labels)

    # Link prediction loss
    link_labels = torch.zeros(link_probs.size(0), device=device)
    link_labels[:pos_edge.size(1)] = 1.0
    link_loss = F.binary_cross_entropy(link_probs, link_labels)

    total_loss = (1 - LINK_LOSS_WEIGHT) * node_loss + LINK_LOSS_WEIGHT * link_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "node_loss": node_loss.item(),
        "link_loss": link_loss.item(),
    }


@torch.no_grad()
def evaluate(model, data, device, split="val"):
    model.eval()

    x = data.x.to(device)
    train_edge = data.train_pos_edge_index.to(device)
    z = model.encode(x, train_edge)

    # Link prediction AUC
    if split == "val":
        pos_edge = data.val_pos_edge_index.to(device)
        neg_edge = data.val_neg_edge_index.to(device)
    else:
        pos_edge = data.test_pos_edge_index.to(device)
        neg_edge = data.test_neg_edge_index.to(device)

    pos_pred = model.predict_link(z, pos_edge).cpu().numpy()
    neg_pred = model.predict_link(z, neg_edge).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    y_pred = np.concatenate([pos_pred, neg_pred])
    auc = roc_auc_score(y_true, y_pred)

    # Node classification accuracy
    node_logits = model.classify_node(z)
    pred_labels = node_logits.argmax(dim=-1).cpu().numpy()
    true_labels = data.y.numpy()
    f1 = f1_score(true_labels, pred_labels, average="macro")

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
    logger.info(f"Using device: {device}")

    # ── Load data ──
    logger.info("Loading dataset...")
    raw_data = load_facebook_dataset()
    logger.info(f"Graph: {raw_data.num_nodes} nodes, {raw_data.edge_index.size(1)} edges")

    # Split edges
    data = train_test_split_edges(raw_data, val_ratio=0.05, test_ratio=0.1)
    data.num_nodes = raw_data.num_nodes
    data.y = raw_data.y

    # ── Model ──
    model = SocialGraphGNN(
        in_channels=FEATURE_DIM,
        hidden_channels=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    logger.info(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = f"{args.output_dir}/model_weights_{DATASET_NAME}_best.pth"
    early_stopper = EarlyStopping(patience=PATIENCE)

    # ── Training loop ──
    best_val_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = train_epoch(model, optimizer, data, device)
        scheduler.step()

        if epoch % 10 == 0:
            metrics = evaluate(model, data, device, split="val")
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch:03d} | Loss: {losses['total_loss']:.4f} | "
                f"Val AUC: {metrics['link_auc']:.4f} | "
                f"Node F1: {metrics['node_f1']:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

            if early_stopper(metrics["link_auc"], model, best_model_path):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if metrics["link_auc"] > best_val_auc:
                best_val_auc = metrics["link_auc"]

    # ── Final evaluation ──
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, data, device, split="test")
    logger.info(
        f"TEST | Link AUC: {test_metrics['link_auc']:.4f} | "
        f"Node F1: {test_metrics['node_f1']:.4f}"
    )

    # ── Save final model + embeddings ──
    final_weights_path = f"{args.output_dir}/model_weights_{DATASET_NAME}.pth"
    torch.save(model.state_dict(), final_weights_path)

    with torch.no_grad():
        model.eval()
        x = data.x.to(device)
        edge = data.train_pos_edge_index.to(device)
        embeddings = model.encode(x, edge).cpu().numpy()

    embeddings_path = f"{args.output_dir}/embeddings_{DATASET_NAME}.npy"
    np.save(embeddings_path, embeddings)

    logger.info(f"Saved weights: {final_weights_path}")
    logger.info(f"Saved embeddings: {embeddings_path} shape={embeddings.shape}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
