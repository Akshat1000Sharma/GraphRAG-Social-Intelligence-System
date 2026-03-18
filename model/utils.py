"""
Utility functions for GNN training and inference.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional
import random


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_link_prediction_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute AUC for link prediction."""
    return roc_auc_score(y_true, y_pred)


def compute_node_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute accuracy and macro F1 for node classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def prepare_graph_splits(
    data: Data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
) -> Data:
    """Split graph edges into train/val/test for link prediction."""
    return train_test_split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio)


def build_synthetic_features(num_nodes: int, feature_dim: int = 128) -> torch.Tensor:
    """
    Create synthetic node features when real features are unavailable.
    Uses random Gaussian features as baseline.
    """
    return torch.randn(num_nodes, feature_dim)


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    """Row-normalize node feature matrix."""
    norm = x.norm(p=1, dim=1, keepdim=True).clamp(min=1.0)
    return x / norm


def compute_degree_features(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute degree-based node features."""
    degrees = torch.zeros(num_nodes)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
    return degrees.unsqueeze(1)


def build_structural_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    feature_dim: int = 64,
) -> torch.Tensor:
    """
    Build structural node features combining:
    - Degree features
    - Random walk positional encodings (approx)
    - Random projection for remaining dims
    """
    degree_feat = compute_degree_features(edge_index, num_nodes)
    # Normalize degree
    degree_norm = (degree_feat - degree_feat.mean()) / (degree_feat.std() + 1e-8)

    # Random features for remaining dimensions
    remaining = feature_dim - 1
    random_feat = torch.randn(num_nodes, remaining) * 0.1

    return torch.cat([degree_norm, random_feat], dim=1)


def get_negative_samples(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_samples: Optional[int] = None,
) -> torch.Tensor:
    """Sample negative edges (non-existing connections)."""
    if num_samples is None:
        num_samples = edge_index.size(1)
    return negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_samples,
    )


def save_model_and_embeddings(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    dataset_name: str,
    output_dir: str = "weights",
) -> Dict[str, str]:
    """Save model weights and embeddings to disk."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    weights_path = f"{output_dir}/model_weights_{dataset_name}.pth"
    embeddings_path = f"{output_dir}/embeddings_{dataset_name}.npy"

    torch.save(model.state_dict(), weights_path)
    np.save(embeddings_path, embeddings)

    return {"weights": weights_path, "embeddings": embeddings_path}


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load model from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_score: float, model: torch.nn.Module, path: str) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self._save(model, path)
        elif val_score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_score
            self._save(model, path)
            self.counter = 0
        return self.should_stop

    def _save(self, model: torch.nn.Module, path: str):
        torch.save(model.state_dict(), path)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
