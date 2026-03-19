"""
Self-contained Reddit Graph ML training script (Kaggle-ready)
"""

import os
import time
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

from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, train_test_split_edges
from torch_geometric.nn import GCNConv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ───────────────── CONFIG ─────────────────

HIDDEN_DIM = 256
EMBEDDING_DIM = 128
NUM_CLASSES = 4
NUM_LAYERS = 3
LR = 0.001
EPOCHS = 100
PATIENCE = 15
OUTPUT_DIR = "weights"
SEED = 42

# ───────────────── UTILS ─────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalize_features(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best = 0

    def __call__(self, metric, model, path):
        if metric > self.best:
            self.best = metric
            self.counter = 0
            torch.save(model.state_dict(), path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def get_negative_samples(edge_index, num_nodes, num_samples):
    neg_src = torch.randint(0, num_nodes, (num_samples,))
    neg_dst = torch.randint(0, num_nodes, (num_samples,))
    return torch.stack([neg_src, neg_dst], dim=0)

# ───────────────── MODEL ─────────────────

class SocialGraphGNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, HIDDEN_DIM))
        for _ in range(NUM_LAYERS - 2):
            self.convs.append(GCNConv(HIDDEN_DIM, HIDDEN_DIM))
        self.convs.append(GCNConv(HIDDEN_DIM, EMBEDDING_DIM))
        self.classifier = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
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

def load_reddit():
    from torch_geometric.datasets import Reddit2
    from torch_geometric.utils import subgraph

    logger.info("Loading Reddit2 (memory-safe)...")
    dataset = Reddit2(root="/tmp/reddit2")
    data = dataset[0]

    logger.info(f"Original: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

    # HARD LIMIT (VERY IMPORTANT)
    max_nodes = 20000

    idx = torch.randperm(data.num_nodes)[:max_nodes]
    idx = idx.sort()[0]

    edge_index, _ = subgraph(idx, data.edge_index, relabel_nodes=True)

    data = Data(
        x=data.x[idx],
        edge_index=edge_index,
        y=data.y[idx],
        num_nodes=max_nodes
    )

    data.edge_index = to_undirected(data.edge_index)

    # Normalize features
    data.x = (data.x - data.x.mean(0)) / (data.x.std(0) + 1e-6)

    # Reduce to 4 classes
    data.y = (data.y * 4 // (data.y.max().item() + 1)).clamp(0, 3)

    logger.info(f"Final: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

    return data

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

    loss = 0.5 * node_loss + 0.5 * link_loss
    loss.backward()
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

    data = load_reddit()
    data = train_test_split_edges(data)

    model = SocialGraphGNN(data.x.shape[1]).to(device)
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

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_weights_reddit.pth")

    with torch.no_grad():
        z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device)).cpu().numpy()

    np.save(f"{OUTPUT_DIR}/embeddings_reddit.npy", z)

    logger.info("Training complete.")

if __name__ == "__main__":
    main()