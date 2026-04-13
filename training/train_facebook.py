# train_facebook.py
# Mirrors all Jupyter notebook cells end-to-end in a single script.

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import csv
import json
import zipfile
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
HIDDEN_DIM   = 128
EPOCHS       = 101
LR           = 0.01
WD           = 5e-4
DROPOUT      = 0.3
OUTPUT_DIR   = "weights"
DATA_PATH    = "/kaggle/working/facebook_data/facebook_large"
DATASET_URL  = "https://snap.stanford.edu/data/facebook_large.zip"
EXTRACT_PATH = "/kaggle/working/facebook_data"


# ══════════════════════════════════════════════════════════════════════════════
# Cell 1 equivalent: install check reminder
# (packages assumed pre-installed; run:
#   pip install torch torchvision torchaudio torch-geometric sentence-transformers)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Cell 2: Download and extract dataset
# ══════════════════════════════════════════════════════════════════════════════
def download_and_extract():
    zip_path = "facebook_large.zip"
    if not os.path.exists(zip_path):
        logger.info("Downloading dataset...")
        os.system(f"wget -q {DATASET_URL}")
    else:
        logger.info("Zip already exists, skipping download.")

    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(EXTRACT_PATH)

    logger.info("Dataset ready at: %s", EXTRACT_PATH)
    logger.info("Files: %s", os.listdir(EXTRACT_PATH))


# ══════════════════════════════════════════════════════════════════════════════
# Cell 3: Resolve file paths
# ══════════════════════════════════════════════════════════════════════════════
def get_paths():
    edges_path    = os.path.join(DATA_PATH, "musae_facebook_edges.csv")
    features_path = os.path.join(DATA_PATH, "musae_facebook_features.json")
    target_path   = os.path.join(DATA_PATH, "musae_facebook_target.csv")
    return edges_path, features_path, target_path


# ══════════════════════════════════════════════════════════════════════════════
# Cell 4: Load edges
# ══════════════════════════════════════════════════════════════════════════════
def load_edges(edges_path):
    edges = pd.read_csv(edges_path)
    logger.info("Edges head:\n%s", edges.head())
    logger.info("Total edges: %d", len(edges))
    return edges


# ══════════════════════════════════════════════════════════════════════════════
# Cell 5: Load labels
# ══════════════════════════════════════════════════════════════════════════════
def load_labels(target_path):
    labels_dict = {}
    with open(target_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_dict[int(row["id"])] = row["page_type"]

    nodes = sorted(labels_dict.keys())
    le    = LabelEncoder()
    Y     = le.fit_transform([labels_dict[n] for n in nodes])
    logger.info("Classes: %d", len(set(Y)))
    return nodes, Y, le


# ══════════════════════════════════════════════════════════════════════════════
# Cell 6: Build one-hot features from JSON feature file
# ══════════════════════════════════════════════════════════════════════════════
def build_onehot_features(features_path, nodes):
    with open(features_path, "r") as f:
        features_dict = json.load(f)

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    max_feature = 0
    for feats in features_dict.values():
        if feats:
            max_feature = max(max_feature, max(feats))

    X = np.zeros((len(nodes), max_feature + 1))
    for node_str, feats in features_dict.items():
        node = int(node_str)          # JSON keys are strings
        if node in node_to_idx and feats:
            X[node_to_idx[node], feats] = 1

    logger.info("One-hot feature shape: %s", X.shape)
    return X, node_to_idx


# ══════════════════════════════════════════════════════════════════════════════
# Cell 7: Build text documents for each node
# ══════════════════════════════════════════════════════════════════════════════
def build_documents(nodes):
    documents = []
    for node in nodes:
        doc = f"""
    Node ID: {node}
    This is a Facebook page in a social network graph.
    It is connected to other pages.
    """
        documents.append(doc)
    logger.info("Sample document: %s", documents[0])
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# Cell 8: Encode documents with SentenceTransformer
# ══════════════════════════════════════════════════════════════════════════════
def encode_text(documents):
    encoder         = SentenceTransformer("all-MiniLM-L6-v2")
    text_embeddings = encoder.encode(documents, show_progress_bar=True, batch_size=64)
    text_embeddings = np.array(text_embeddings)
    logger.info("Text embedding shape: %s", text_embeddings.shape)
    return text_embeddings


# ══════════════════════════════════════════════════════════════════════════════
# Cell 9: Build edge_index tensor
# ══════════════════════════════════════════════════════════════════════════════
def build_edge_index(edges, node_to_idx):
    edge_index = []
    for _, row in edges.iterrows():
        src_id = int(row["id_1"])
        dst_id = int(row["id_2"])
        if src_id in node_to_idx and dst_id in node_to_idx:
            src = node_to_idx[src_id]
            dst = node_to_idx[dst_id]
            edge_index.append([src, dst])
            edge_index.append([dst, src])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    logger.info("Edge index shape: %s", edge_index.shape)
    return edge_index


# ══════════════════════════════════════════════════════════════════════════════
# Cell 10: Combine features and build PyG Data object
# ══════════════════════════════════════════════════════════════════════════════
def build_data(X, text_embeddings, edge_index, Y):
    X_combined = np.concatenate([X, text_embeddings], axis=1)
    x    = torch.tensor(X_combined, dtype=torch.float)
    y    = torch.tensor(Y, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    logger.info("PyG Data: %s", data)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Cell 11: Train/test split masks
# ══════════════════════════════════════════════════════════════════════════════
def apply_masks(data, Y):
    indices = list(range(len(Y)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=Y
    )
    train_mask = torch.zeros(len(Y), dtype=torch.bool)
    test_mask  = torch.zeros(len(Y), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx]   = True
    data.train_mask = train_mask
    data.test_mask  = test_mask
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Cell 12: Model definition
# ══════════════════════════════════════════════════════════════════════════════
class GNN_RAG_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1      = SAGEConv(in_channels, hidden_channels)
        self.bn1        = nn.BatchNorm1d(hidden_channels)
        self.conv2      = SAGEConv(hidden_channels, hidden_channels)
        self.bn2        = nn.BatchNorm1d(hidden_channels)
        self.conv3      = SAGEConv(hidden_channels, hidden_channels)
        self.bn3        = nn.BatchNorm1d(hidden_channels)
        self.fusion     = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode(self, x, edge_index):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        h = F.relu(self.fusion(h))
        return h

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.classifier(h)


# ══════════════════════════════════════════════════════════════════════════════
# Cells 13-16: Build model, train, evaluate, save weights
# ══════════════════════════════════════════════════════════════════════════════
def train_and_save(data, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = GNN_RAG_Model(
        in_channels=data.num_features,
        hidden_channels=HIDDEN_DIM,
        num_classes=num_classes,
    ).to(device)

    data      = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_acc = 0.0

    # ── Training loop (Cell 14) ───────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out  = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                acc  = (
                    (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
                    / data.test_mask.sum().item()
                )
            logger.info(
                "Epoch %4d | Loss: %.4f | LR: %.6f | Val Acc: %.4f",
                epoch, loss.item(), optimizer.param_groups[0]["lr"], acc,
            )
            # Save best checkpoint
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"{OUTPUT_DIR}/best.pth")

    # ── Cell 15: Final test accuracy ──────────────────────────────────────────
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best.pth"))
    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = (
            (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
            / data.test_mask.sum().item()
        )
    logger.info("Test Accuracy: %.4f", test_acc)

    # ── Cell 16: Train + test accuracy breakdown ──────────────────────────────
    with torch.no_grad():
        train_acc = (
            (pred[data.train_mask] == data.y[data.train_mask]).sum().item()
            / data.train_mask.sum().item()
        )
    logger.info("Train Acc: %.4f | Test Acc: %.4f", train_acc, test_acc)

    # ── Save final weights (mirrors train_facebook.py) ────────────────────────
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_weights_facebook.pth")
    logger.info("Saved model_weights_facebook.pth")

    # ── Save embeddings ───────────────────────────────────────────────────────
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
    np.save(f"{OUTPUT_DIR}/embeddings_facebook.npy", embeddings.cpu().numpy())
    logger.info("Saved embeddings_facebook.npy")

    print("Training successfully completed!")
    print(f"  Best Val Acc : {best_acc:.4f}")
    print(f"  Train Acc    : {train_acc:.4f}")
    print(f"  Test  Acc    : {test_acc:.4f}")
    print(f"  Weights      : {OUTPUT_DIR}/model_weights_facebook.pth")
    print(f"  Embeddings   : {OUTPUT_DIR}/embeddings_facebook.npy")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Cell 2
    download_and_extract()

    # Cell 3
    edges_path, features_path, target_path = get_paths()

    # Cell 4
    edges = load_edges(edges_path)

    # Cell 5
    nodes, Y, le = load_labels(target_path)

    # Cell 6
    X, node_to_idx = build_onehot_features(features_path, nodes)

    # Cell 7
    documents = build_documents(nodes)

    # Cell 8
    text_embeddings = encode_text(documents)

    # Cell 9
    edge_index = build_edge_index(edges, node_to_idx)

    # Cell 10
    data = build_data(X, text_embeddings, edge_index, Y)

    # Cell 11
    data = apply_masks(data, Y)

    # Cells 13-16
    train_and_save(data, num_classes=len(set(Y)))


if __name__ == "__main__":
    main()