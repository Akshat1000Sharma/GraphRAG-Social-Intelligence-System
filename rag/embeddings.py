"""
Embeddings: Population of vector indexes from Neo4j data.
Runs on startup to build searchable semantic indexes for GraphRAG.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional

from db.neo4j_client import Neo4jClient
from rag.vector_store import TextEmbeddingStore, get_text_store, get_user_index, get_post_index

logger = logging.getLogger(__name__)


class EmbeddingPopulator:
    """Populates in-memory vector indexes from Neo4j data."""

    def __init__(self, neo4j: Neo4jClient, text_store: Optional[TextEmbeddingStore] = None):
        self.neo4j = neo4j
        self.text_store = text_store or get_text_store()
        self.user_index = get_user_index()
        self.post_index = get_post_index()

    def populate_all(self) -> Dict[str, int]:
        counts = {}
        try:
            counts["users"] = self.populate_users()
        except Exception as e:
            logger.warning(f"User embedding population failed: {e}")
            counts["users"] = 0
        try:
            counts["posts"] = self.populate_posts()
        except Exception as e:
            logger.warning(f"Post embedding population failed: {e}")
            counts["posts"] = 0
        logger.info(f"Vector indexes populated: {counts}")
        return counts

    def populate_users(self) -> int:
        if not self.neo4j.is_connected:
            return 0
        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:FRIEND]->(f:User)
        OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
        WITH u,
             collect(DISTINCT f.name)[..5] AS friend_names,
             collect(DISTINCT p.title)[..3] AS post_titles
        RETURN u.id AS id, u.name AS name, u.bio AS bio,
               u.influence_score AS influence,
               friend_names, post_titles
        LIMIT 500
        """
        records = self.neo4j.run_query(query)
        if not records:
            return 0
        texts, metadatas = [], []
        for r in records:
            parts = [f"User: {r.get('name', '')}", f"Bio: {r.get('bio', '')}"]
            friends = [f for f in (r.get("friend_names") or []) if f]
            if friends:
                parts.append(f"Friends: {', '.join(friends[:5])}")
            posts = [p for p in (r.get("post_titles") or []) if p]
            if posts:
                parts.append(f"Posts: {', '.join(posts[:3])}")
            texts.append(" | ".join(parts))
            metadatas.append({
                "id": r.get("id", ""),
                "user_id": r.get("id", ""),
                "name": r.get("name", ""),
                "bio": r.get("bio", ""),
                "influence": r.get("influence", 0.0),
                "type": "user",
            })
        if texts:
            embeddings = self.text_store.embed_batch(texts)
            self.user_index.add_batch(embeddings, metadatas)
            self._store_embeddings_neo4j(records, embeddings)
        logger.info(f"Indexed {len(texts)} user embeddings")
        return len(texts)

    def populate_posts(self) -> int:
        if not self.neo4j.is_connected:
            return 0
        query = """
        MATCH (p:Post)
        OPTIONAL MATCH (u:User)-[:POSTED]->(p)
        RETURN p.id AS id, p.title AS title, p.content AS content,
               p.topic AS topic, p.like_count AS likes,
               u.id AS author_id, u.name AS author_name
        LIMIT 1000
        """
        records = self.neo4j.run_query(query)
        if not records:
            return 0
        texts, metadatas = [], []
        for r in records:
            text = (
                f"Post: {r.get('title', '')} "
                f"Content: {r.get('content', '')} "
                f"Topic: {r.get('topic', '')} "
                f"By: {r.get('author_name', 'unknown')}"
            )
            texts.append(text)
            metadatas.append({
                "id": r.get("id", ""),
                "post_id": r.get("id", ""),
                "title": r.get("title", ""),
                "topic": r.get("topic", ""),
                "likes": r.get("likes", 0),
                "author_id": r.get("author_id", ""),
                "type": "post",
            })
        if texts:
            embeddings = self.text_store.embed_batch(texts)
            self.post_index.add_batch(embeddings, metadatas)
        logger.info(f"Indexed {len(texts)} post embeddings")
        return len(texts)

    def _store_embeddings_neo4j(self, records, embeddings):
        for record, emb in zip(records, embeddings):
            uid = record.get("id")
            if not uid:
                continue
            try:
                self.neo4j.run_write_query(
                    "MATCH (u:User {id: $id}) SET u.embedding = $emb",
                    {"id": uid, "emb": emb.tolist()},
                )
            except Exception:
                pass


def build_gnn_embedding_index(gnn_embeddings: np.ndarray, user_ids: List[str]) -> None:
    user_index = get_user_index()
    for emb, uid in zip(gnn_embeddings, user_ids):
        user_index.add(emb, {"id": uid, "user_id": uid, "type": "user_gnn"})
    logger.info(f"Loaded {len(user_ids)} GNN embeddings into user index")
