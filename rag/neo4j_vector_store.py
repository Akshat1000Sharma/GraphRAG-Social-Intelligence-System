"""
neo4j_vector_store.py
=====================
Neo4j-only unified backend for BOTH graph storage and vector similarity search.
Completely replaces FAISS and the in-memory InMemoryVectorIndex.

Architecture:
  - Embeddings stored as node properties: User.embedding, Post.embedding
  - Neo4j native vector indexes (v5+) for ANN (approximate nearest neighbor) search
  - Cypher-native hybrid queries combining graph constraints + vector similarity
  - Single connection pool for all operations (no external vector DB)

Design decisions:
  1. Embeddings stored on the node itself → no join needed at query time
  2. Separate vector indexes per label (User, Post) → smaller index = faster search
  3. Cosine similarity chosen over Euclidean → normalized embeddings, better for NLP
  4. Batch upsert with MERGE → idempotent startup population
  5. Query-time hybrid scoring done in Cypher → no Python-side fusion for simple cases
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TEXT_EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 output dimension
GNN_EMBEDDING_DIM  = 128   # GraphSAGE encoder output dimension

# Neo4j vector index names (must match schema setup queries)
USER_TEXT_INDEX = "user_text_embeddings"
USER_GNN_INDEX  = "user_gnn_embeddings"
POST_TEXT_INDEX = "post_text_embeddings"


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EMBEDDING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TextEmbeddingEngine:
    """
    Manages SentenceTransformer model for encoding text → dense vectors.
    Lazy-loaded singleton. Includes LRU-style cache for repeated queries.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        # Simple in-process cache for repeated identical queries
        self._cache: Dict[str, List[float]] = {}
        self._cache_limit = 1000

    def _load(self):
        if self._model is None:
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded")

    def encode(self, text: str) -> List[float]:
        """Encode a single string → normalized float list for Neo4j storage."""
        if text in self._cache:
            return self._cache[text]
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True)
        result = vec.tolist()
        if len(self._cache) < self._cache_limit:
            self._cache[text] = result
        return result

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts → (N, dim) numpy array, L2-normalized."""
        self._load()
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32)

    def encode_user_profile(self, user: Dict[str, Any]) -> List[float]:
        """
        Build a rich text representation of a user node and embed it.
        Concatenates name, bio, friend names, and post titles for context.
        """
        parts = [
            f"User: {user.get('name', '')}",
            f"Bio: {user.get('bio', '')}",
        ]
        friends = user.get("friends") or []
        if friends:
            names = [f.get("name", "") for f in friends[:5] if f.get("name")]
            if names:
                parts.append(f"Connected to: {', '.join(names)}")
        posts = user.get("posts") or []
        if posts:
            titles = [p.get("title", "") for p in posts[:3] if p.get("title")]
            if titles:
                parts.append(f"Posts about: {', '.join(titles)}")
        return self.encode(" | ".join(parts))

    def encode_post(self, post: Dict[str, Any]) -> List[float]:
        """Embed a post node (title + content + topic + author)."""
        text = (
            f"Post: {post.get('title', '')} "
            f"Content: {post.get('content', '')} "
            f"Topic: {post.get('topic', '')} "
            f"By: {post.get('author_name', 'unknown')}"
        )
        return self.encode(text)


# ══════════════════════════════════════════════════════════════════════════════
# NEO4J VECTOR SCHEMA MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jVectorSchemaManager:
    """
    Manages Neo4j vector index lifecycle:
    - Create indexes (idempotent, IF NOT EXISTS)
    - Drop and recreate if dimension changes
    - Verify index status

    Neo4j 5.x vector index API used throughout.
    """

    # ── Index definitions ─────────────────────────────────────────────────────
    VECTOR_INDEXES = [
        {
            "name": USER_TEXT_INDEX,
            "label": "User",
            "property": "text_embedding",
            "dimensions": TEXT_EMBEDDING_DIM,
            "similarity": "cosine",
            "description": "Text-based user profile embeddings (SentenceTransformer)",
        },
        {
            "name": USER_GNN_INDEX,
            "label": "User",
            "property": "gnn_embedding",
            "dimensions": GNN_EMBEDDING_DIM,
            "similarity": "cosine",
            "description": "GNN structural embeddings from GraphSAGE encoder",
        },
        {
            "name": POST_TEXT_INDEX,
            "label": "Post",
            "property": "text_embedding",
            "dimensions": TEXT_EMBEDDING_DIM,
            "similarity": "cosine",
            "description": "Text-based post content embeddings",
        },
    ]

    def __init__(self, neo4j_client):
        self.client = neo4j_client

    def create_all_indexes(self):
        """
        Create all vector indexes if they don't exist.
        Safe to call on every startup — uses IF NOT EXISTS.
        """
        for idx in self.VECTOR_INDEXES:
            self._create_index(idx)

    def _create_index(self, idx: Dict[str, Any]):
        """
        Create a single vector index.

        Cypher (Neo4j 5.x):
            CREATE VECTOR INDEX <name> IF NOT EXISTS
            FOR (n:<Label>) ON (n.<property>)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: <dim>,
                `vector.similarity_function`: 'cosine'
              }
            }
        """
        cypher = f"""
        CREATE VECTOR INDEX {idx['name']} IF NOT EXISTS
        FOR (n:{idx['label']}) ON (n.{idx['property']})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {idx['dimensions']},
            `vector.similarity_function`: '{idx['similarity']}'
          }}
        }}
        """
        try:
            self.client.run_write_query(cypher.strip())
            logger.info(
                f"Vector index '{idx['name']}' ready "
                f"({idx['label']}.{idx['property']}, "
                f"dim={idx['dimensions']}, sim={idx['similarity']})"
            )
        except Exception as e:
            logger.warning(f"Vector index '{idx['name']}' creation: {e}")

    def get_index_status(self) -> List[Dict[str, Any]]:
        """Query Neo4j for current vector index status."""
        try:
            records = self.client.run_query(
                "SHOW VECTOR INDEXES YIELD name, state, populationPercent, labelsOrTypes, properties"
            )
            return records
        except Exception as e:
            logger.warning(f"Could not query vector index status: {e}")
            return []

    def wait_for_indexes_online(self, timeout_seconds: int = 60):
        """
        Poll until all vector indexes are ONLINE.
        Important: indexes are populated asynchronously in Neo4j.
        """
        import time
        start = time.time()
        target_names = {idx["name"] for idx in self.VECTOR_INDEXES}

        while time.time() - start < timeout_seconds:
            statuses = self.get_index_status()
            online = {r["name"] for r in statuses if r.get("state") == "ONLINE"}
            if target_names.issubset(online):
                logger.info("All vector indexes are ONLINE")
                return True
            pending = target_names - online
            logger.debug(f"Waiting for indexes: {pending}")
            time.sleep(2)

        logger.warning("Timed out waiting for vector indexes to come online")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING POPULATOR  (replaces rag/embeddings.py)
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jEmbeddingPopulator:
    """
    Populates embedding properties on Neo4j nodes at startup.
    Writes embeddings DIRECTLY to Neo4j — no in-memory FAISS index.

    Workflow:
      1. Query Neo4j for nodes without embeddings (or all nodes on full refresh)
      2. Build text representation for each node
      3. Batch-encode with SentenceTransformer
      4. MERGE embedding back into Neo4j node property
      5. Neo4j vector index auto-indexes the new properties

    Performance note:
      - Batch upserts in chunks of BATCH_SIZE to avoid transaction timeouts
      - Only re-embed nodes where embedding IS NULL (incremental updates)
    """

    BATCH_SIZE = 50  # nodes per transaction

    def __init__(self, neo4j_client, text_engine: Optional[TextEmbeddingEngine] = None):
        self.client = neo4j_client
        self.engine = text_engine or TextEmbeddingEngine()

    def populate_all(self, force_refresh: bool = False) -> Dict[str, int]:
        """
        Populate embeddings for all node types.
        Set force_refresh=True to re-embed everything (e.g., after model change).
        """
        counts = {}
        try:
            counts["users"] = self.populate_users(force_refresh=force_refresh)
        except Exception as e:
            logger.warning(f"User embedding population failed: {e}")
            counts["users"] = 0
        try:
            counts["posts"] = self.populate_posts(force_refresh=force_refresh)
        except Exception as e:
            logger.warning(f"Post embedding population failed: {e}")
            counts["posts"] = 0
        logger.info(f"Neo4j embedding population complete: {counts}")
        return counts

    def populate_users(self, force_refresh: bool = False) -> int:
        """
        Embed user profile text and store as User.text_embedding.
        Fetches users with their social context (friends, posts) for richer embeddings.
        """
        if not self.client.is_connected:
            return 0

        # Only fetch users without embeddings unless force_refresh
        null_filter = "" if force_refresh else "WHERE u.text_embedding IS NULL"
        query = f"""
        MATCH (u:User)
        {null_filter}
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
        records = self.client.run_query(query)
        if not records:
            return 0

        # Build text representations
        texts, record_ids = [], []
        for r in records:
            parts = [
                f"User: {r.get('name', '')}",
                f"Bio: {r.get('bio', '')}",
            ]
            friends = [f for f in (r.get("friend_names") or []) if f]
            if friends:
                parts.append(f"Friends: {', '.join(friends[:5])}")
            posts = [p for p in (r.get("post_titles") or []) if p]
            if posts:
                parts.append(f"Posts: {', '.join(posts[:3])}")
            texts.append(" | ".join(parts))
            record_ids.append(r.get("id", ""))

        # Batch encode and write back to Neo4j
        embeddings = self.engine.encode_batch(texts)
        self._batch_write_embeddings(
            node_label="User",
            property_name="text_embedding",
            node_ids=record_ids,
            embeddings=embeddings,
        )
        logger.info(f"Populated text_embedding for {len(records)} users")
        return len(records)

    def populate_posts(self, force_refresh: bool = False) -> int:
        """
        Embed post content and store as Post.text_embedding.
        """
        if not self.client.is_connected:
            return 0

        null_filter = "" if force_refresh else "WHERE p.text_embedding IS NULL"
        query = f"""
        MATCH (p:Post)
        {null_filter}
        OPTIONAL MATCH (u:User)-[:POSTED]->(p)
        RETURN p.id AS id, p.title AS title, p.content AS content,
               p.topic AS topic, p.like_count AS likes,
               u.id AS author_id, u.name AS author_name
        LIMIT 1000
        """
        records = self.client.run_query(query)
        if not records:
            return 0

        texts, record_ids = [], []
        for r in records:
            text = (
                f"Post: {r.get('title', '')} "
                f"Content: {r.get('content', '')} "
                f"Topic: {r.get('topic', '')} "
                f"By: {r.get('author_name', 'unknown')}"
            )
            texts.append(text)
            record_ids.append(r.get("id", ""))

        embeddings = self.engine.encode_batch(texts)
        self._batch_write_embeddings(
            node_label="Post",
            property_name="text_embedding",
            node_ids=record_ids,
            embeddings=embeddings,
        )
        logger.info(f"Populated text_embedding for {len(records)} posts")
        return len(records)

    def store_gnn_embeddings(
        self,
        user_ids: List[str],
        gnn_embeddings: np.ndarray,
    ) -> int:
        """
        Store GNN structural embeddings (from GraphSAGE encoder) on User nodes.
        Called after GNN inference to persist learned representations.
        """
        if not self.client.is_connected or len(user_ids) == 0:
            return 0

        self._batch_write_embeddings(
            node_label="User",
            property_name="gnn_embedding",
            node_ids=user_ids,
            embeddings=gnn_embeddings,
        )
        logger.info(f"Stored GNN embeddings for {len(user_ids)} users")
        return len(user_ids)

    def _batch_write_embeddings(
        self,
        node_label: str,
        property_name: str,
        node_ids: List[str],
        embeddings: np.ndarray,
    ):
        """
        Write embeddings to Neo4j in batches using UNWIND for efficiency.

        Uses MERGE + SET so it's idempotent — safe to re-run.
        UNWIND avoids N round-trips (one transaction per batch).
        """
        for start in range(0, len(node_ids), self.BATCH_SIZE):
            batch_ids  = node_ids[start : start + self.BATCH_SIZE]
            batch_embs = embeddings[start : start + self.BATCH_SIZE]

            batch_data = [
                {"id": nid, "embedding": emb.tolist()}
                for nid, emb in zip(batch_ids, batch_embs)
                if nid  # skip empty IDs
            ]
            if not batch_data:
                continue

            cypher = f"""
            UNWIND $batch AS item
            MATCH (n:{node_label} {{id: item.id}})
            SET n.{property_name} = item.embedding,
                n.embedding_updated_at = datetime()
            """
            try:
                self.client.run_write_query(cypher, {"batch": batch_data})
            except Exception as e:
                logger.error(
                    f"Batch embedding write failed ({node_label}.{property_name}): {e}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# NEO4J VECTOR RETRIEVER  (replaces FAISS VectorRetriever)
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jVectorRetriever:
    """
    Performs ALL vector similarity searches through Neo4j vector indexes.
    Completely replaces FAISS / InMemoryVectorIndex.

    Two search modes:
    1. Pure vector search (vector index only)
    2. Hybrid search (vector + graph constraints combined in single Cypher query)

    Neo4j vector search Cypher syntax (v5+):
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
    """

    def __init__(self, neo4j_client, text_engine: Optional[TextEmbeddingEngine] = None):
        self.client = neo4j_client
        self.engine = text_engine or TextEmbeddingEngine()

    # ── Pure Vector Search ────────────────────────────────────────────────────

    def search_users_by_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar users by text embedding.
        Searches USER_TEXT_INDEX in Neo4j — no FAISS.

        Cypher:
            CALL db.index.vector.queryNodes('user_text_embeddings', $top_k, $vec)
            YIELD node AS u, score
            RETURN u.id, u.name, u.bio, score
        """
        query_vector = self.engine.encode(query)
        return self._vector_query_users(
            index_name=USER_TEXT_INDEX,
            query_vector=query_vector,
            top_k=top_k,
            label="text",
        )

    def search_users_by_gnn_embedding(
        self,
        gnn_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find structurally similar users using GNN embeddings stored in Neo4j.
        Used to find users with similar graph-structural positions.
        """
        return self._vector_query_users(
            index_name=USER_GNN_INDEX,
            query_vector=gnn_embedding.tolist(),
            top_k=top_k,
            label="gnn",
        )

    def search_posts_by_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar posts by text embedding.
        Searches POST_TEXT_INDEX in Neo4j.
        """
        query_vector = self.engine.encode(query)
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node AS p, score
        OPTIONAL MATCH (author:User)-[:POSTED]->(p)
        RETURN p.id           AS id,
               p.title        AS title,
               p.content      AS content,
               p.topic        AS topic,
               p.like_count   AS likes,
               p.created_at   AS created_at,
               author.id      AS author_id,
               author.name    AS author_name,
               score          AS similarity_score,
               'neo4j_vector' AS source
        ORDER BY score DESC
        """
        try:
            return self.client.run_query(cypher, {
                "index_name": POST_TEXT_INDEX,
                "top_k": top_k,
                "query_vector": query_vector,
            })
        except Exception as e:
            logger.error(f"Post vector search failed: {e}")
            return []

    # ── Hybrid Search (Vector + Graph Constraints) ────────────────────────────

    def search_users_hybrid(
        self,
        query: str,
        graph_filter: Optional[str] = None,
        filter_params: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        HYBRID SEARCH: combine vector similarity + graph constraints in one query.

        This is the key capability replacing FAISS + separate graph traversal.
        Everything happens inside Neo4j — no Python-side fusion needed.

        Example graph_filter:
            "AND (u)-[:FRIEND]->(ref:User {id: $ref_user_id})"

        Returns nodes ranked by cosine similarity that also satisfy the graph pattern.
        """
        query_vector = self.engine.encode(query)
        params = {
            "index_name": USER_TEXT_INDEX,
            "top_k": top_k * 3,  # fetch more to allow graph filtering
            "query_vector": query_vector,
            **(filter_params or {}),
        }

        extra_filter = graph_filter or ""

        # Hybrid Cypher: vector search + optional graph constraints
        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node AS u, score
        WHERE u.id IS NOT NULL
        {extra_filter}
        RETURN u.id           AS id,
               u.name         AS name,
               u.bio          AS bio,
               u.influence_score AS influence_score,
               u.follower_count  AS follower_count,
               score             AS similarity_score,
               'neo4j_hybrid'    AS source
        ORDER BY score DESC
        LIMIT {top_k}
        """
        try:
            return self.client.run_query(cypher, params)
        except Exception as e:
            logger.error(f"Hybrid user search failed: {e}")
            return []

    def search_friends_of_friends_by_similarity(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        HYBRID QUERY: Find 2-hop connections semantically similar to the query.

        "Who are semantically relevant friends-of-friends of user X?"

        Combines:
        - Graph constraint: must be 2-hop FRIEND of user_id (not a direct friend)
        - Vector constraint: must be semantically similar to query text

        Single Cypher query — no Python-side merging.
        """
        query_vector = self.engine.encode(query)

        cypher = """
        // Step 1: anchor user by canonical id or source_id (matches Graph Explorer / ingest)
        MATCH (u:User)
        WHERE u.id = $user_id OR u.source_id = $user_id
        MATCH (u)-[:FRIEND]->(friend:User)-[:FRIEND]->(candidate:User)
        WHERE candidate <> u
          AND NOT (u)-[:FRIEND]->(candidate)
          AND candidate.text_embedding IS NOT NULL
        WITH candidate, count(friend) AS mutual_count

        // Step 2: score by vector similarity within that subgraph
        WITH candidate, mutual_count,
             vector.similarity.cosine(candidate.text_embedding, $query_vector) AS vec_score

        // Step 3: combined ranking (graph signal + vector signal)
        WITH candidate, mutual_count, vec_score,
             (mutual_count * 0.4 + vec_score * 0.6) AS combined_score

        ORDER BY combined_score DESC
        LIMIT $top_k

        RETURN candidate.id       AS id,
               candidate.name     AS name,
               candidate.bio      AS bio,
               candidate.influence_score AS influence_score,
               mutual_count,
               vec_score          AS similarity_score,
               combined_score     AS fusion_score,
               'neo4j_hybrid'     AS source
        """
        try:
            return self.client.run_query(cypher, {
                "user_id": user_id,
                "query_vector": query_vector,
                "top_k": top_k,
            })
        except Exception as e:
            logger.error(f"Friend-of-friend hybrid search failed: {e}")
            return []

    def search_influencers_by_topic(
        self,
        topic_query: str,
        min_followers: int = 0,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        HYBRID QUERY: Find influential users who post about a specific topic.

        Combines:
        - Graph constraint: user has posted content + follower count threshold
        - Vector constraint: user's posts are semantically similar to topic_query

        Demonstrates the power of Neo4j hybrid over FAISS:
        FAISS alone can't apply the graph structural constraints.
        """
        query_vector = self.engine.encode(topic_query)

        cypher = """
        // Vector search over Post embeddings to find topic-relevant posts
        CALL db.index.vector.queryNodes($post_index, $top_k_posts, $query_vector)
        YIELD node AS p, score AS post_similarity

        // Join to the post's author via graph relationship
        MATCH (author:User)-[:POSTED]->(p)
        WHERE author.follower_count >= $min_followers

        // Aggregate: for each author, take max post similarity
        WITH author,
             max(post_similarity) AS best_post_score,
             count(p)             AS matching_post_count

        // Final ranking: combine influence + topic relevance
        ORDER BY (best_post_score * 0.6 + author.influence_score * 0.4) DESC
        LIMIT $top_k

        RETURN author.id              AS id,
               author.name            AS name,
               author.bio             AS bio,
               author.follower_count  AS follower_count,
               author.influence_score AS influence_score,
               best_post_score        AS similarity_score,
               matching_post_count,
               'neo4j_hybrid'         AS source
        """
        try:
            return self.client.run_query(cypher, {
                "post_index": POST_TEXT_INDEX,
                "top_k_posts": top_k * 5,
                "query_vector": query_vector,
                "min_followers": min_followers,
                "top_k": top_k,
            })
        except Exception as e:
            logger.error(f"Influencer-by-topic hybrid search failed: {e}")
            return []

    def search_trending_by_engagement_and_similarity(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        HYBRID QUERY: Trending posts scored by engagement velocity + semantic similarity.

        Replaces: separate FAISS vector search + manual Python RRF fusion.
        Now done entirely inside Neo4j with a single Cypher query.
        """
        query_vector = self.engine.encode(query)

        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k_vec, $query_vector)
        YIELD node AS p, score AS vec_score

        // Apply engagement scoring inline
        WITH p, vec_score,
             p.like_count + (p.comment_count * 2.0) AS raw_engagement,
             duration.between(datetime(p.created_at), datetime()).hours AS age_hours

        WITH p, vec_score,
             raw_engagement / (toFloat(age_hours) + 1.0) AS engagement_velocity

        // Combined score: engagement + semantic relevance
        WITH p, vec_score, engagement_velocity,
             (vec_score * 0.4 + (engagement_velocity / 1000.0) * 0.6) AS combined_score

        ORDER BY combined_score DESC
        LIMIT $top_k

        OPTIONAL MATCH (author:User)-[:POSTED]->(p)
        RETURN p.id          AS id,
               p.title       AS title,
               p.content     AS content,
               p.topic       AS topic,
               p.like_count  AS likes,
               p.created_at  AS created_at,
               vec_score     AS similarity_score,
               engagement_velocity,
               combined_score AS fusion_score,
               author.name   AS author_name,
               'neo4j_hybrid' AS source
        """
        try:
            return self.client.run_query(cypher, {
                "index_name": POST_TEXT_INDEX,
                "top_k_vec": top_k * 3,
                "query_vector": query_vector,
                "top_k": top_k,
            })
        except Exception as e:
            logger.error(f"Trending hybrid search failed: {e}")
            return []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _vector_query_users(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int,
        label: str,
    ) -> List[Dict[str, Any]]:
        """Execute a pure user vector search against a named index."""
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node AS u, score
        RETURN u.id              AS id,
               u.name            AS name,
               u.bio             AS bio,
               u.influence_score AS influence_score,
               u.follower_count  AS follower_count,
               score             AS similarity_score,
               'neo4j_vector'    AS source
        ORDER BY score DESC
        """
        try:
            return self.client.run_query(cypher, {
                "index_name": index_name,
                "top_k": top_k,
                "query_vector": query_vector,
            })
        except Exception as e:
            logger.error(f"User vector search ({label}) failed: {e}")
            return []


# ── Module-level singletons ───────────────────────────────────────────────────

_text_engine: Optional[TextEmbeddingEngine] = None


def get_text_engine() -> TextEmbeddingEngine:
    global _text_engine
    if _text_engine is None:
        _text_engine = TextEmbeddingEngine()
    return _text_engine
