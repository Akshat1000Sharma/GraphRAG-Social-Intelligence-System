"""
Self-contained Twitter Graph ML training script (Kaggle-ready)
"""

import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

# Install PyG if needed
try:
    import torch_geometric
except:
    os.system("pip install torch-geometric -q")

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, train_test_split_edges
from torch_geometric.datasets import SNAPDataset
from torch_geometric.nn import GATConv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ───────────────── CONFIG ─────────────────

FEATURE_DIM = 128
HIDDEN_DIM = 128
EMBEDDING_DIM = 64
NUM_CLASSES = 4
HEADS = 4
LR = 0.005
EPOCHS = 150
PATIENCE = 20
OUTPUT_DIR = "weights"
SEED = 42

# ───────────────── UTILS ─────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_features(num_nodes, dim):
    return torch.randn(num_nodes, dim) * 0.1

def get_negative_samples(edge_index, num_nodes, num_samples):
    neg_src = torch.randint(0, num_nodes, (num_samples,))
    neg_dst = torch.randint(0, num_nodes, (num_samples,))
    return torch.stack([neg_src, neg_dst], dim=0)

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = 0
        self.counter = 0

    def __call__(self, metric, model, path):
        if metric > self.best:
            self.best = metric
            self.counter = 0
            torch.save(model.state_dict(), path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# ───────────────── MODEL ─────────────────

class GATSocialGNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gat1 = GATConv(in_dim, HIDDEN_DIM, heads=HEADS, dropout=0.4)
        self.gat2 = GATConv(HIDDEN_DIM * HEADS, EMBEDDING_DIM, heads=1, dropout=0.4)
        self.classifier = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)

    def encode(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return x

    def classify_node(self, z):
        return self.classifier(z)

    def predict_link(self, z, edge_index):
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))

    def forward(self, x, edge_index, pos_edge, neg_edge):
        z = self.encode(x, edge_index)
        node_logits = self.classify_node(z)
        pos_pred = self.predict_link(z, pos_edge)
        neg_pred = self.predict_link(z, neg_edge)
        return z, node_logits, torch.cat([pos_pred, neg_pred])

# ───────────────── DATA ─────────────────

def load_twitter():
    logger.info("Loading Twitter SNAP dataset...")

    dataset = SNAPDataset(root="/tmp/snap", name="ego-Twitter")

    edges_src, edges_dst = [], []
    offset = 0

    for d in dataset:
        if d.edge_index.size(1) == 0:
            continue

        # Limit size (important safety)
        if d.num_nodes > 2000:
            idx = torch.randperm(d.num_nodes)[:2000]
            mask = [(u.item() in idx.tolist() and v.item() in idx.tolist())
                    for u, v in zip(d.edge_index[0], d.edge_index[1])]
            edge_index = d.edge_index[:, mask]

            mapping = {old.item(): i for i, old in enumerate(idx)}
            src = torch.tensor([mapping[u.item()] for u in edge_index[0]])
            dst = torch.tensor([mapping[v.item()] for v in edge_index[1]])

            edges_src.append(src + offset)
            edges_dst.append(dst + offset)
            offset += len(idx)

        else:
            edges_src.append(d.edge_index[0] + offset)
            edges_dst.append(d.edge_index[1] + offset)
            offset += d.num_nodes

    edge_index = torch.stack([torch.cat(edges_src), torch.cat(edges_dst)])
    edge_index = to_undirected(edge_index)

    num_nodes = offset
    logger.info(f"Graph: {num_nodes} nodes, {edge_index.size(1)} edges")

    x = build_features(num_nodes, FEATURE_DIM)

    # Degree-based labels
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))

    q1, q2, q3 = deg.quantile(0.25), deg.quantile(0.5), deg.quantile(0.75)
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[deg > q1] = 1
    y[deg > q2] = 2
    y[deg > q3] = 3

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

# ───────────────── TRAIN ─────────────────

def train_epoch(model, optimizer, data, device):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    edge = data.train_pos_edge_index.to(device)

    pos = data.train_pos_edge_index.to(device)
    neg = get_negative_samples(edge, data.num_nodes, pos.size(1)).to(device)

    z, node_logits, link_probs = model(x, edge, pos, neg)

    node_loss = F.cross_entropy(node_logits, data.y.to(device))
    labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(device)
    link_loss = F.binary_cross_entropy(link_probs, labels)

    loss = 0.4 * node_loss + 0.6 * link_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()

    return loss.item()

@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device))

    pos = data.test_pos_edge_index.to(device)
    neg = data.test_neg_edge_index.to(device)

    pos_pred = model.predict_link(z, pos).cpu().numpy()
    neg_pred = model.predict_link(z, neg).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    auc = roc_auc_score(y_true, np.concatenate([pos_pred, neg_pred]))

    preds = model.classify_node(z).argmax(dim=1).cpu().numpy()
    f1 = f1_score(data.y.numpy(), preds, average="macro")

    return auc, f1

# ───────────────── MAIN ─────────────────

def main():
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data = load_twitter()
    data = train_test_split_edges(data)

    model = GATSocialGNN(data.x.shape[1]).to(device)
    logger.info(f"Params: {count_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    early = EarlyStopping(PATIENCE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_path = f"{OUTPUT_DIR}/best.pth"

    for epoch in range(EPOCHS):
        loss = train_epoch(model, optimizer, data, device)

        if epoch % 10 == 0:
            auc, f1 = evaluate(model, data, device)
            logger.info(f"Epoch {epoch} | Loss {loss:.4f} | AUC {auc:.4f} | F1 {f1:.4f}")

            if early(auc, model, best_path):
                break

    model.load_state_dict(torch.load(best_path))

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_weights_twitter.pth")

    with torch.no_grad():
        z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device)).cpu().numpy()

    np.save(f"{OUTPUT_DIR}/embeddings_twitter.npy", z)

    logger.info("Training complete.")

if __name__ == "__main__":
    main()