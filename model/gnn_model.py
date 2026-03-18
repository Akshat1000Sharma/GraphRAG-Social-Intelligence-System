"""
GNN Model: GraphSAGE-based architecture for link prediction and node classification.
Supports CPU inference after GPU training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.utils import negative_sampling
from typing import Optional, Tuple


class GraphSAGEEncoder(nn.Module):
    """
    Multi-layer GraphSAGE encoder that produces node embeddings.
    Used for both link prediction and node classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class LinkPredictor(nn.Module):
    """
    MLP-based link predictor that takes two node embeddings and predicts edge probability.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_channels * 2, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, 1))

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_u, z_v], dim=-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=0.2, training=self.training)
        return self.layers[-1](x).squeeze(-1)


class NodeClassifier(nn.Module):
    """
    MLP classifier on top of node embeddings for influencer/role detection.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z))
        x = self.dropout(x)
        return self.fc2(x)


class SocialGraphGNN(nn.Module):
    """
    Full GNN model combining encoder, link predictor, and node classifier.
    Single model supports both tasks simultaneously.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        embedding_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.link_predictor = LinkPredictor(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels // 2,
        )

        self.node_classifier = NodeClassifier(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels // 2,
            num_classes=num_classes,
        )

        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.num_classes = num_classes

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        return self.encoder(x, edge_index)

    def predict_link(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict link probability for given edges."""
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        return torch.sigmoid(self.link_predictor(z_u, z_v))

    def classify_node(self, z: torch.Tensor) -> torch.Tensor:
        """Classify nodes (e.g., influencer roles)."""
        return self.node_classifier(z)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: Optional[torch.Tensor] = None,
        neg_edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Full forward pass.
        Returns: (node_embeddings, node_logits, link_probs)
        """
        z = self.encode(x, edge_index)
        node_logits = self.classify_node(z)

        link_probs = None
        if pos_edge_index is not None:
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=x.size(0),
                    num_neg_samples=pos_edge_index.size(1),
                )
            all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            link_probs = self.predict_link(z, all_edges)

        return z, node_logits, link_probs

    def compute_loss(
        self,
        node_logits: torch.Tensor,
        node_labels: torch.Tensor,
        link_probs: Optional[torch.Tensor],
        pos_edge_count: Optional[int],
        link_weight: float = 0.5,
    ) -> torch.Tensor:
        """Combined loss for multi-task training."""
        node_loss = F.cross_entropy(node_logits, node_labels)

        if link_probs is not None and pos_edge_count is not None:
            link_labels = torch.zeros(link_probs.size(0), device=link_probs.device)
            link_labels[:pos_edge_count] = 1.0
            link_loss = F.binary_cross_entropy(link_probs, link_labels)
            return (1 - link_weight) * node_loss + link_weight * link_loss

        return node_loss

    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "embedding_dim": self.embedding_dim,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
        }


class GATSocialGNN(nn.Module):
    """
    Alternative GAT-based model for Twitter dataset (attention-heavy networks).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_classes: int = 4,
        heads: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, embedding_dim, heads=1, dropout=dropout)

        self.link_predictor = LinkPredictor(embedding_dim, hidden_channels)
        self.node_classifier = NodeClassifier(embedding_dim, hidden_channels // 2, num_classes)

        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.num_classes = num_classes

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)
        return self.conv2(x, edge_index)

    def predict_link(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.link_predictor(z[edge_index[0]], z[edge_index[1]]))

    def classify_node(self, z: torch.Tensor) -> torch.Tensor:
        return self.node_classifier(z)

    def forward(self, x, edge_index, pos_edge_index=None, neg_edge_index=None):
        z = self.encode(x, edge_index)
        node_logits = self.classify_node(z)
        link_probs = None
        if pos_edge_index is not None:
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(edge_index, x.size(0), pos_edge_index.size(1))
            all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            link_probs = self.predict_link(z, all_edges)
        return z, node_logits, link_probs
