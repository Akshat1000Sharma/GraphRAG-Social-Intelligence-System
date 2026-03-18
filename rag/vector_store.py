"""
Vector Store: Embedding management and similarity search.
Supports both Neo4j vector index and in-memory FAISS for hybrid retrieval.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
import json

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dim
GNN_EMBEDDING_DIM = 128


class TextEmbeddingStore:
    """
    Manages text embeddings for users, posts, and queries.
    Uses sentence-transformers for semantic embedding.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, np.ndarray] = {}

    def load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if text in self._cache:
            return self._cache[text]
        self.load_model()
        emb = self._model.encode(text, normalize_embeddings=True)
        self._cache[text] = emb
        return emb

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        self.load_model()
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32)

    def embed_user_profile(self, user: Dict[str, Any]) -> np.ndarray:
        """Create a rich text representation of a user for embedding."""
        parts = [
            f"User: {user.get('name', '')}",
            f"Bio: {user.get('bio', '')}",
        ]
        if user.get("friends"):
            parts.append(f"Connected to: {', '.join([f.get('name', '') for f in user.get('friends', [])[:5]])}")
        if user.get("posts"):
            topics = [p.get("title", "") for p in user.get("posts", [])[:3]]
            parts.append(f"Posts about: {', '.join(topics)}")
        return self.embed_text(" ".join(parts))

    def embed_post(self, post: Dict[str, Any]) -> np.ndarray:
        """Embed a post for semantic retrieval."""
        text = f"Post: {post.get('title', '')} Content: {post.get('content', '')} Topic: {post.get('topic', '')}"
        return self.embed_text(text)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


class InMemoryVectorIndex:
    """
    Lightweight in-memory vector index using numpy.
    Falls back when FAISS is unavailable.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty = True

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]):
        self.vectors.append(vector.astype(np.float32))
        self.metadata.append(metadata)
        self._dirty = True

    def add_batch(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        for v, m in zip(vectors, metadatas):
            self.add(v, m)

    def _rebuild_matrix(self):
        if self.vectors:
            self._matrix = np.stack(self.vectors, axis=0)
            # Normalize rows
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self._matrix = self._matrix / norms
            self._dirty = False

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for most similar vectors."""
        if not self.vectors:
            return []

        if self._dirty:
            self._rebuild_matrix()

        query_norm = query.astype(np.float32)
        qn = np.linalg.norm(query_norm)
        if qn > 0:
            query_norm = query_norm / qn

        scores = self._matrix @ query_norm
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {**self.metadata[i], "similarity_score": float(scores[i])}
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self.vectors)


class FAISSVectorIndex:
    """
    FAISS-backed vector index for high-performance similarity search.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.metadata: List[Dict[str, Any]] = []
        self._index = None
        self._load_faiss()

    def _load_faiss(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)  # Inner product (cosine after normalization)
            logger.info("FAISS index initialized")
        except ImportError:
            logger.warning("FAISS not available, falling back to numpy index")
            self._index = None

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]):
        if self._index is None:
            return
        import faiss
        v = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(v)
        self._index.add(v)
        self.metadata.append(metadata)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import faiss
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({**self.metadata[idx], "similarity_score": float(score)})
        return results

    def __len__(self) -> int:
        return self._index.ntotal if self._index else 0


def build_vector_index(dim: int = VECTOR_DIM) -> "InMemoryVectorIndex | FAISSVectorIndex":
    """Build the best available vector index."""
    try:
        import faiss
        return FAISSVectorIndex(dim)
    except ImportError:
        return InMemoryVectorIndex(dim)


# Singleton stores
_text_store: Optional[TextEmbeddingStore] = None
_user_index: Optional[Any] = None
_post_index: Optional[Any] = None


def get_text_store() -> TextEmbeddingStore:
    global _text_store
    if _text_store is None:
        _text_store = TextEmbeddingStore()
    return _text_store


def get_user_index() -> Any:
    global _user_index
    if _user_index is None:
        _user_index = build_vector_index(VECTOR_DIM)
    return _user_index


def get_post_index() -> Any:
    global _post_index
    if _post_index is None:
        _post_index = build_vector_index(VECTOR_DIM)
    return _post_index
