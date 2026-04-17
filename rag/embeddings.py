"""
embeddings.py  (REFACTORED — delegates to Neo4jEmbeddingPopulator)
====================================================================
Previously: populated in-memory FAISS indexes from Neo4j data.
Now: writes embeddings directly to Neo4j node properties.

FAISS operations removed:
  - self.user_index.add_batch(...)   → removed
  - self.post_index.add_batch(...)   → removed
  - InMemoryVectorIndex              → removed
  - FAISSVectorIndex                 → removed

Neo4j operations added:
  - UNWIND batch upsert → User.text_embedding
  - UNWIND batch upsert → Post.text_embedding
  - UNWIND batch upsert → User.gnn_embedding (for GNN structural search)
  - Neo4j vector indexes auto-index the new property values
"""

import logging
import numpy as np
from typing import Dict, List, Optional

from db.neo4j_client import Neo4jClient
from rag.neo4j_vector_store import (
    Neo4jEmbeddingPopulator,
    TextEmbeddingEngine,
    get_text_engine,
)

logger = logging.getLogger(__name__)

# ── Main entrypoint (same interface as before) ─────────────────────────────────

class EmbeddingPopulator(Neo4jEmbeddingPopulator):
    """
    Drop-in replacement for the old EmbeddingPopulator.
    Extends Neo4jEmbeddingPopulator — all logic delegated there.

    The old class populated FAISS indexes. This class writes to Neo4j.
    """

    def __init__(self, neo4j: Neo4jClient, text_store: Optional[TextEmbeddingEngine] = None):
        super().__init__(neo4j_client=neo4j, text_engine=text_store or get_text_engine())

    # populate_all(), populate_users(), populate_posts() all inherited.
    # _store_embeddings_neo4j() is now the primary write path (not a side-effect).


def build_gnn_embedding_index(
    gnn_embeddings: np.ndarray,
    user_ids: List[str],
    neo4j_client=None,
) -> None:
    """
    Store GNN structural embeddings on User nodes in Neo4j.

    OLD behaviour: added to in-memory FAISS/numpy index.
    NEW behaviour: writes to User.gnn_embedding property, auto-indexed by
                   the 'user_gnn_embeddings' vector index.
    """
    if neo4j_client is None:
        logger.warning(
            "build_gnn_embedding_index: neo4j_client not provided. "
            "GNN embeddings not stored. Pass a Neo4jClient instance."
        )
        return

    populator = Neo4jEmbeddingPopulator(neo4j_client=neo4j_client)
    populator.store_gnn_embeddings(user_ids=user_ids, gnn_embeddings=gnn_embeddings)
