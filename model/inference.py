"""
Inference Engine: CPU-only inference for the pretrained GNN models.
Loads weights, runs predictions, returns structured results.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from model.gnn_model import SocialGraphGNN, GATSocialGNN

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

DATASET_CONFIG = {
    "facebook": {
        "model_class": SocialGraphGNN,
        "in_channels": 128,
        "hidden_channels": 256,
        "embedding_dim": 128,
        "num_classes": 4,
        "weights_file": "model_weights_facebook.pth",
        "embeddings_file": "embeddings_facebook.npy",
    },
    "twitter": {
        "model_class": GATSocialGNN,
        "in_channels": 128,
        "hidden_channels": 128,
        "embedding_dim": 64,
        "num_classes": 4,
        "weights_file": "model_weights_twitter.pth",
        "embeddings_file": "embeddings_twitter.npy",
    },
    "reddit": {
        "model_class": SocialGraphGNN,
        "in_channels": 300,
        "hidden_channels": 256,
        "embedding_dim": 128,
        "num_classes": 4,
        "weights_file": "model_weights_reddit.pth",
        "embeddings_file": "embeddings_reddit.npy",
    },
}

# Node class labels
NODE_CLASS_LABELS = {
    0: "regular_user",
    1: "influencer",
    2: "content_creator",
    3: "community_hub",
}


class GNNInferenceEngine:
    """
    CPU-optimized inference engine for pretrained GNN models.
    Supports multiple datasets and lazy loading.
    """

    def __init__(self, dataset: str = "facebook"):
        self.dataset = dataset
        self.config = DATASET_CONFIG[dataset]
        self.model: Optional[torch.nn.Module] = None
        self.embeddings: Optional[np.ndarray] = None
        self.device = torch.device("cpu")
        self._loaded = False

    def load(self) -> bool:
        """Load pretrained weights and embeddings."""
        weights_path = WEIGHTS_DIR / self.config["weights_file"]
        embeddings_path = WEIGHTS_DIR / self.config["embeddings_file"]

        if not weights_path.exists():
            logger.warning(f"Weights not found at {weights_path}. Using untrained model.")
            self._init_untrained_model()
            return False

        try:
            model_class = self.config["model_class"]
            self.model = model_class(
                in_channels=self.config["in_channels"],
                hidden_channels=self.config["hidden_channels"],
                embedding_dim=self.config["embedding_dim"],
                num_classes=self.config["num_classes"],
            )

            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            if embeddings_path.exists():
                self.embeddings = np.load(str(embeddings_path))
                logger.info(f"Loaded embeddings shape: {self.embeddings.shape}")

            self._loaded = True
            logger.info(f"Model loaded successfully for dataset: {self.dataset}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._init_untrained_model()
            return False

    def _init_untrained_model(self):
        """Initialize a random-weight model for demo/fallback purposes."""
        model_class = self.config["model_class"]
        self.model = model_class(
            in_channels=self.config["in_channels"],
            hidden_channels=self.config["hidden_channels"],
            embedding_dim=self.config["embedding_dim"],
            num_classes=self.config["num_classes"],
        )
        self.model.eval()
        logger.info("Using untrained model (demo mode)")

    @torch.no_grad()
    def get_node_embeddings(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> np.ndarray:
        """Compute node embeddings for the given graph."""
        if self.model is None:
            self.load()
        embeddings = self.model.encode(node_features, edge_index)
        return embeddings.numpy()

    @torch.no_grad()
    def predict_link_probability(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        candidate_pairs: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """
        Predict friendship/connection probability for candidate pairs.
        Returns sorted results with probabilities.
        """
        if self.model is None:
            self.load()

        z = self.model.encode(node_features, edge_index)

        if not candidate_pairs:
            return []

        pairs_tensor = torch.tensor(candidate_pairs, dtype=torch.long).t()
        probs = self.model.predict_link(z, pairs_tensor)

        results = []
        for (src, dst), prob in zip(candidate_pairs, probs.tolist()):
            results.append({"source_id": src, "target_id": dst, "probability": round(prob, 4)})

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    @torch.no_grad()
    def classify_nodes(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify nodes (influencer detection).
        Returns predicted class and confidence for each node.
        """
        if self.model is None:
            self.load()

        z = self.model.encode(node_features, edge_index)
        logits = self.model.classify_node(z)
        probs = torch.softmax(logits, dim=-1)
        predicted_classes = probs.argmax(dim=-1)

        results = []
        for i in range(z.size(0)):
            node_id = node_ids[i] if node_ids else i
            class_idx = predicted_classes[i].item()
            results.append(
                {
                    "node_id": node_id,
                    "predicted_class": NODE_CLASS_LABELS.get(class_idx, f"class_{class_idx}"),
                    "class_index": class_idx,
                    "confidence": round(probs[i, class_idx].item(), 4),
                    "class_probabilities": {
                        NODE_CLASS_LABELS.get(j, f"class_{j}"): round(probs[i, j].item(), 4)
                        for j in range(probs.size(1))
                    },
                }
            )

        return results

    @torch.no_grad()
    def get_influence_score(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
    ) -> Dict[str, Any]:
        """
        Compute influence score for a specific node.
        Combines GNN classification with structural features.
        """
        if self.model is None:
            self.load()

        z = self.model.encode(node_features, edge_index)
        logits = self.model.classify_node(z)
        probs = torch.softmax(logits, dim=-1)

        node_emb = z[node_idx]
        # Influence score: weighted combination of influencer/creator probs
        influence_score = (probs[node_idx, 1] * 0.5 + probs[node_idx, 2] * 0.3 + probs[node_idx, 3] * 0.2).item()

        # Embedding norm as proxy for centrality
        emb_magnitude = node_emb.norm().item()

        return {
            "node_idx": node_idx,
            "influence_score": round(influence_score, 4),
            "embedding_magnitude": round(emb_magnitude, 4),
            "role": NODE_CLASS_LABELS.get(probs[node_idx].argmax().item(), "unknown"),
            "role_probabilities": {
                NODE_CLASS_LABELS.get(j, f"class_{j}"): round(probs[node_idx, j].item(), 4)
                for j in range(probs.size(1))
            },
        }

    def embedding_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find most similar nodes by embedding cosine similarity.
        Uses precomputed embeddings if available.
        """
        if self.embeddings is None:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        corpus_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        similarities = corpus_norms @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {"node_idx": int(idx), "similarity": round(float(similarities[idx]), 4)}
            for idx in top_indices
        ]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


class MultiDatasetInferenceManager:
    """
    Manages multiple GNN models (one per dataset).
    Provides unified interface for inference.
    """

    def __init__(self):
        self.engines: Dict[str, GNNInferenceEngine] = {}
        self._primary = "facebook"

    def load_dataset(self, dataset: str) -> GNNInferenceEngine:
        if dataset not in self.engines:
            engine = GNNInferenceEngine(dataset)
            engine.load()
            self.engines[dataset] = engine
        return self.engines[dataset]

    def get_engine(self, dataset: str = "facebook") -> GNNInferenceEngine:
        if dataset not in self.engines:
            return self.load_dataset(dataset)
        return self.engines[dataset]

    def load_all(self):
        for dataset in DATASET_CONFIG:
            try:
                self.load_dataset(dataset)
            except Exception as e:
                logger.warning(f"Could not load {dataset}: {e}")

    def status(self) -> Dict[str, bool]:
        return {ds: self.engines[ds].is_loaded for ds in self.engines}


# Singleton manager
inference_manager = MultiDatasetInferenceManager()
