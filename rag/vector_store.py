"""
vector_store.py  (COMPATIBILITY SHIM — FAISS REMOVED)
======================================================
This file previously contained FAISSVectorIndex, InMemoryVectorIndex,
build_vector_index(), get_user_index(), get_post_index().

ALL OF THAT IS NOW REMOVED.

Neo4j native vector indexes replace FAISS entirely.
See: rag/neo4j_vector_store.py

This file is kept as a shim so any code that still imports from
rag.vector_store continues to work during the migration period.
It re-exports the Neo4j equivalents under the old names.

After full migration, this file can be deleted.
"""

# Re-export the Neo4j-native equivalents
from rag.neo4j_vector_store import (
    TextEmbeddingEngine as TextEmbeddingStore,  # same interface
    get_text_engine as get_text_store,
    Neo4jEmbeddingPopulator,
    Neo4jVectorRetriever,
)

import logging
logger = logging.getLogger(__name__)
logger.warning(
    "rag.vector_store is deprecated. "
    "Use rag.neo4j_vector_store directly. FAISS has been removed."
)

# Stubs for old functions that no longer exist
def get_user_index():
    raise RuntimeError(
        "get_user_index() has been removed. "
        "User vector search is now done via Neo4j vector indexes. "
        "Use Neo4jVectorRetriever.search_users_by_text() instead."
    )

def get_post_index():
    raise RuntimeError(
        "get_post_index() has been removed. "
        "Post vector search is now done via Neo4j vector indexes. "
        "Use Neo4jVectorRetriever.search_posts_by_text() instead."
    )

def build_vector_index(*args, **kwargs):
    raise RuntimeError(
        "build_vector_index() has been removed (was FAISS). "
        "Neo4j vector indexes are created via Neo4jVectorSchemaManager.create_all_indexes()."
    )
