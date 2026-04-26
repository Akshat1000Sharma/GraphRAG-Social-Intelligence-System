"""
hybrid_retrieval.py  (REFACTORED — Neo4j-only)
================================================
Replaces the old dual-backend system (Neo4j graph + FAISS vectors).
Now uses Neo4j as the SINGLE unified backend for all retrieval.

Key architectural changes:
  OLD: GraphRetriever(Neo4j) + VectorRetriever(FAISS) → Python-side RRF
  NEW: GraphRetriever(Neo4j) + Neo4jVectorRetriever(Neo4j) → Cypher-side fusion
       where possible, Python-side RRF only for cross-index results

Design decisions:
  1. For HYBRID mode: use Neo4j hybrid Cypher queries (single round-trip)
  2. For pure VECTOR mode: use Neo4j vector index procedures
  3. For pure GRAPH mode: unchanged Cypher traversal queries
  4. Python RRF only used when combining GRAPH + VECTOR results from
     two separate queries (e.g., friend traversal + semantic search)
  5. No external dependencies beyond neo4j driver
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from rag.neo4j_vector_store import Neo4jVectorRetriever, TextEmbeddingEngine, get_text_engine

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    GRAPH  = "graph"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class GraphContext:
    """Structured context retrieved from Neo4j graph traversal."""
    query_type: str
    primary_entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]]    = field(default_factory=list)
    paths: List[List[str]]                 = field(default_factory=list)
    raw_records: List[Dict[str, Any]]      = field(default_factory=list)
    cypher_used: str = ""


@dataclass
class VectorContext:
    """Context retrieved from Neo4j vector index. (Previously FAISS.)"""
    query: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    model_used: str = ""
    index_used: str = ""   # NEW: which Neo4j vector index was queried


@dataclass
class HybridContext:
    """Fused context from Neo4j graph + Neo4j vector retrieval."""
    graph_context:  Optional[GraphContext]  = None
    vector_context: Optional[VectorContext] = None
    fused_entities: List[Dict[str, Any]]    = field(default_factory=list)
    fusion_scores:  Dict[str, float]        = field(default_factory=dict)
    retrieval_mode: RetrievalMode           = RetrievalMode.HYBRID
    metadata:       Dict[str, Any]          = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH RETRIEVER  (unchanged — Cypher traversal)
# ══════════════════════════════════════════════════════════════════════════════

class GraphRetriever:
    """Executes structured Cypher queries against Neo4j for graph traversal."""

    QUERY_TEMPLATES = {
        "friend_recommendation": """
            MATCH (u:User)
            WHERE u.id = $user_id OR u.source_id = $user_id
            MATCH (u)-[:FRIEND]->(friend)-[:FRIEND]->(fof:User)
            WHERE fof <> u AND fof.id <> u.id AND NOT (u)-[:FRIEND]->(fof)
            WITH fof, count(friend) AS mutual_count
            ORDER BY mutual_count DESC LIMIT $top_k
            RETURN fof.id AS id, fof.name AS name,
                   mutual_count AS mutual_friends, fof.influence_score AS influence_score,
                   fof.follower_count AS follower_count
        """,
        "user_profile": """
            MATCH (u:User)
            WHERE u.id = $user_id OR u.source_id = $user_id
            OPTIONAL MATCH (u)-[:FRIEND]->(f:User)
            OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
            RETURN u.id AS id, u.name AS name, u.bio AS bio,
                   u.influence_score AS influence,
                   collect(DISTINCT {id: f.id, name: f.name}) AS friends,
                   collect(DISTINCT {id: p.id, title: p.title, likes: p.like_count}) AS posts
        """,
        "user_profile_ds": """
            MATCH (u:User {dataset: $dataset})
            WHERE u.id = $user_id OR u.source_id = $user_id
            OPTIONAL MATCH (u)-[:FRIEND]->(f:User {dataset: $dataset})
            OPTIONAL MATCH (u)-[:POSTED]->(p:Post {dataset: $dataset})
            RETURN u.id AS id, u.name AS name, u.bio AS bio,
                   u.influence_score AS influence, u.dataset AS dataset,
                   collect(DISTINCT {id: f.id, name: f.name}) AS friends,
                   collect(DISTINCT {id: p.id, title: p.title, likes: p.like_count}) AS posts
        """,
        "trending_posts": """
            MATCH (p:Post)
            WHERE p.created_at IS NOT NULL
            WITH p, p.like_count + (p.comment_count * 2) AS engagement
            ORDER BY engagement DESC LIMIT $top_k
            RETURN p.id AS id, p.title AS title, p.content AS content,
                   p.like_count AS likes, p.comment_count AS comments,
                   p.topic AS topic, p.created_at AS created_at, engagement
        """,
        "explain_connection": """
            MATCH path = shortestPath(
              (a:User {id: $user_a})-[*..6]-(b:User {id: $user_b})
            )
            RETURN [n IN nodes(path) | n.name] AS node_names,
                   [r IN relationships(path) | type(r)] AS rel_types,
                   length(path) AS hops
        """,
        "common_context": """
            MATCH (a:User {id: $user_a})-[:FRIEND]->(common)<-[:FRIEND]-(b:User {id: $user_b})
            RETURN common.id AS id, common.name AS name
        """,
        "influence_stats": """
            MATCH (u:User {id: $user_id})
            OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
            OPTIONAL MATCH (u)-[:FRIEND]-(f:User)
            WITH u, count(DISTINCT p) AS posts, count(DISTINCT f) AS friends,
                 coalesce(sum(p.like_count), 0) AS total_likes
            RETURN u.id AS id, u.name AS name, u.follower_count AS followers,
                   u.influence_score AS gnn_score, posts, friends, total_likes
        """,
        "link_candidates": """
            MATCH (u:User {id: $user_id})-[:FRIEND*2..3]->(candidate:User)
            WHERE candidate.id <> $user_id AND NOT (u)-[:FRIEND]->(candidate)
            WITH candidate, count(*) AS path_count
            ORDER BY path_count DESC LIMIT $top_k
            RETURN candidate.id AS id, candidate.name AS name, path_count AS graph_score
        """,
        "all_users": """
            MATCH (u:User)
            RETURN u.id AS id, u.name AS name, u.bio AS bio,
                   u.influence_score AS influence LIMIT $top_k
        """,
        "fulltext_search": """
            CALL db.index.fulltext.queryNodes('post_content', $query)
            YIELD node AS p, score
            RETURN p.id AS id, p.title AS title, p.content AS content,
                   p.topic AS topic, p.like_count AS likes, score
            ORDER BY score DESC LIMIT $top_k
        """,
        # ── Dataset-scoped versions of key queries (R3) ───────────────────────
        # Dataset in chat context scopes results when the property exists; if nodes have no
        # `dataset` (legacy / unlabeled import), still match so behavior matches
        # GraphQueryService.get_friend_recommendations and /graph/friend-recommendations.
        "friend_recommendation_ds": """
            MATCH (u:User)
            WHERE (u.id = $user_id OR u.source_id = $user_id)
              AND (u.dataset IS NULL OR u.dataset = $dataset)
            MATCH (u)-[:FRIEND]->(friend:User)
            WHERE (friend.dataset IS NULL OR friend.dataset = $dataset)
            MATCH (friend)-[:FRIEND]->(fof:User)
            WHERE fof <> u
              AND NOT (u)-[:FRIEND]->(fof)
              AND (fof.dataset IS NULL OR fof.dataset = $dataset)
            WITH fof, count(friend) AS mutual_count
            ORDER BY mutual_count DESC LIMIT $top_k
            RETURN fof.id AS id, fof.name AS name,
                   mutual_count AS mutual_friends, fof.influence_score AS influence_score,
                   fof.follower_count AS follower_count,
                   fof.dataset AS dataset
        """,
        "influence_stats_ds": """
            MATCH (u:User {dataset: $dataset})
            WHERE u.source_id = $user_id OR u.id = $user_id
            OPTIONAL MATCH (u)-[:POSTED]->(p:Post {dataset: $dataset})
            OPTIONAL MATCH (u)-[:FRIEND]-(f:User {dataset: $dataset})
            WITH u, count(DISTINCT p) AS posts, count(DISTINCT f) AS friends,
                 coalesce(sum(p.like_count), 0) AS total_likes
            RETURN u.id AS id, u.name AS name, u.follower_count AS followers,
                   u.influence_score AS gnn_score, posts, friends, total_likes,
                   u.dataset AS dataset
        """,
        "trending_posts_ds": """
            MATCH (p:Post {dataset: $dataset})
            WHERE p.created_at IS NOT NULL
            WITH p, p.like_count + (p.comment_count * 2) AS engagement
            ORDER BY engagement DESC LIMIT $top_k
            RETURN p.id AS id, p.title AS title, p.content AS content,
                   p.like_count AS likes, p.comment_count AS comments,
                   p.topic AS topic, p.created_at AS created_at, engagement,
                   p.dataset AS dataset
        """,
        "all_users_ds": """
            MATCH (u:User {dataset: $dataset})
            RETURN u.id AS id, u.name AS name, u.bio AS bio,
                   u.influence_score AS influence, u.dataset AS dataset
            LIMIT $top_k
        """,
        "link_candidates_ds": """
            MATCH (u:User {dataset: $dataset})-[:FRIEND*2..3]->(candidate:User {dataset: $dataset})
            WHERE (u.source_id = $user_id OR u.id = $user_id)
              AND candidate.id <> $user_id
              AND candidate.source_id <> $user_id
              AND NOT (u)-[:FRIEND]->(candidate)
            WITH candidate, count(*) AS path_count
            ORDER BY path_count DESC LIMIT $top_k
            RETURN candidate.id AS id, candidate.name AS name,
                   path_count AS graph_score, candidate.dataset AS dataset
        """,
    }

    def __init__(self, neo4j_client):
        self.client = neo4j_client

    def retrieve(self, query_type: str, params: Dict[str, Any], top_k: int = 10) -> GraphContext:
        params = dict(params)
        params["top_k"] = top_k

        # R3: use dataset-scoped Cypher templates when dataset is set
        # Router may already pass query_type ending in _ds — still need $dataset in params.
        dataset = params.pop("dataset", None)
        effective_type = query_type
        # "all" = no dataset filter; facebook/twitter/reddit/demo must pass $dataset into *_ds Cypher
        if dataset and dataset != "all":
            if not query_type.endswith("_ds"):
                ds_key = f"{query_type}_ds"
                if ds_key in self.QUERY_TEMPLATES:
                    effective_type = ds_key
            if effective_type.endswith("_ds"):
                params["dataset"] = dataset

        if effective_type not in self.QUERY_TEMPLATES:
            logger.warning(f"Unknown query type: {effective_type}")
            return GraphContext(query_type=query_type)

        cypher = self.QUERY_TEMPLATES[effective_type]

        try:
            records = self.client.run_query(cypher, params)
        except Exception as e:
            logger.error(f"Graph retrieval failed [{query_type}]: {e}")
            records = []

        ctx = GraphContext(
            query_type=query_type,
            raw_records=records,
            cypher_used=cypher.strip(),
        )
        for r in records:
            if "id" in r or "user_id" in r or "post_id" in r:
                ctx.primary_entities.append(r)
            if "rel_types" in r:
                ctx.relationships.append(r)
            if "node_names" in r:
                ctx.paths.append(r.get("node_names", []))
        return ctx

    def custom_query(self, cypher: str, params: Dict[str, Any]) -> GraphContext:
        try:
            records = self.client.run_query(cypher, params)
        except Exception as e:
            logger.error(f"Custom query failed: {e}")
            records = []
        return GraphContext(
            query_type="custom",
            raw_records=records,
            primary_entities=records,
            cypher_used=cypher,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR RETRIEVER  (REFACTORED — wraps Neo4jVectorRetriever, no FAISS)
# ══════════════════════════════════════════════════════════════════════════════

class VectorRetriever:
    """
    Semantic similarity search via Neo4j vector indexes.

    BEFORE: FAISS IndexFlatIP + in-memory InMemoryVectorIndex
    AFTER:  Neo4jVectorRetriever (Cypher + db.index.vector.queryNodes)

    Same external interface — pipeline is unaffected.
    """

    def __init__(self, neo4j_client, text_engine: Optional[TextEmbeddingEngine] = None):
        self.neo4j_retriever = Neo4jVectorRetriever(
            neo4j_client=neo4j_client,
            text_engine=text_engine or get_text_engine(),
        )
        self.model_name = self.neo4j_retriever.engine.model_name

    def search_users(self, query: str, top_k: int = 10) -> VectorContext:
        results = self.neo4j_retriever.search_users_by_text(query, top_k=top_k)
        return VectorContext(
            query=query,
            results=results,
            model_used=self.model_name,
            index_used="user_text_embeddings",
        )

    def search_posts(self, query: str, top_k: int = 10) -> VectorContext:
        results = self.neo4j_retriever.search_posts_by_text(query, top_k=top_k)
        return VectorContext(
            query=query,
            results=results,
            model_used=self.model_name,
            index_used="post_text_embeddings",
        )

    def search_by_gnn_embedding(self, gnn_embedding, top_k: int = 10) -> VectorContext:
        import numpy as np
        if not isinstance(gnn_embedding, np.ndarray):
            gnn_embedding = np.array(gnn_embedding)
        results = self.neo4j_retriever.search_users_by_gnn_embedding(gnn_embedding, top_k=top_k)
        return VectorContext(
            query="gnn_embedding_search",
            results=results,
            model_used="gnn_encoder",
            index_used="user_gnn_embeddings",
        )

    # ── Enhanced hybrid methods (new — not possible with FAISS) ───────────────

    def search_friends_semantically(self, user_id: str, query: str, top_k: int = 10) -> VectorContext:
        """Hybrid: friends-of-friends ranked by semantic similarity (single Cypher)."""
        results = self.neo4j_retriever.search_friends_of_friends_by_similarity(
            user_id=user_id, query=query, top_k=top_k
        )
        return VectorContext(query=query, results=results,
                             model_used=self.model_name, index_used="neo4j_hybrid")

    def search_influencers_by_topic(self, topic_query: str, min_followers: int = 0, top_k: int = 10) -> VectorContext:
        """Hybrid: influencers who post about a topic (POSTED relationship + vector)."""
        results = self.neo4j_retriever.search_influencers_by_topic(
            topic_query=topic_query, min_followers=min_followers, top_k=top_k
        )
        return VectorContext(query=topic_query, results=results,
                             model_used=self.model_name, index_used="neo4j_hybrid")

    def search_trending_hybrid(self, query: str, top_k: int = 10) -> VectorContext:
        """Hybrid: trending posts by engagement velocity + semantic relevance."""
        results = self.neo4j_retriever.search_trending_by_engagement_and_similarity(
            query=query, top_k=top_k
        )
        return VectorContext(query=query, results=results,
                             model_used=self.model_name, index_used="neo4j_hybrid")


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER  (REFACTORED — Neo4j-only, simplified fusion)
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Orchestrates all retrieval through Neo4j.

    Fusion strategy (simplified from original RRF-always approach):
    - friend_recommendation + HYBRID → Neo4j native hybrid Cypher (1 round-trip)
    - trending_posts + HYBRID        → Neo4j native hybrid Cypher (1 round-trip)
    - All other HYBRID               → graph + vector separately → Python RRF
    - GRAPH only                     → Cypher traversal only
    - VECTOR only                    → Neo4j vector index only
    """

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_retriever: VectorRetriever,
        graph_weight: float = 0.6,
        vector_weight: float = 0.4,
    ):
        self.graph  = graph_retriever
        self.vector = vector_retriever
        self.graph_weight  = graph_weight
        self.vector_weight = vector_weight

    def retrieve(
        self,
        query_type: str,
        params: Dict[str, Any],
        nl_query: str = "",
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 10,
    ) -> HybridContext:
        ctx = HybridContext(retrieval_mode=mode)

        # Locked graph intents: never blend unrelated vector hits when the UI sends mode=hybrid
        if (
            query_type.startswith("user_profile")
            or query_type.startswith("influence_stats")
            or query_type.startswith("explain_connection")
        ):
            mode = RetrievalMode.GRAPH
            ctx.retrieval_mode = mode

        # Friend recommendations: same mutual-friend Cypher as Graph Explorer (graph-only).
        # The previous "native hybrid" path ranked FoF by embedding similarity to the *whole*
        # NL string, which produced unrelated users; hybrid+RRF with vector search_users()
        # on that string had the same problem.
        if query_type.startswith("friend_recommendation"):
            mode = RetrievalMode.GRAPH
            ctx.retrieval_mode = mode

        # ── Neo4j-native hybrid: trending posts ───────────────────────────────
        if mode == RetrievalMode.HYBRID and query_type == "trending_posts" and nl_query:
            vc = self.vector.search_trending_hybrid(nl_query, top_k=top_k)
            ctx.vector_context = vc
            ctx.fused_entities = vc.results[:top_k]
            ctx.fusion_scores  = {
                r.get("id", str(i)): r.get("fusion_score", r.get("similarity_score", 0))
                for i, r in enumerate(ctx.fused_entities)
            }
            ctx.metadata["fusion_method"] = "neo4j_native_hybrid"
            return ctx

        # ── Standard path: separate queries + Python RRF ─────────────────────
        if mode in (RetrievalMode.GRAPH, RetrievalMode.HYBRID):
            ctx.graph_context = self.graph.retrieve(query_type, params, top_k=top_k)

        if mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID) and nl_query:
            is_post = any(kw in nl_query.lower() for kw in ["post","content","trending","topic"])
            ctx.vector_context = (
                self.vector.search_posts(nl_query, top_k=top_k)
                if is_post
                else self.vector.search_users(nl_query, top_k=top_k)
            )

        ctx.fused_entities, ctx.fusion_scores = self._rrf_fuse(ctx, top_k)
        ctx.metadata["fusion_method"] = "python_rrf"
        return ctx

    def _rrf_fuse(self, ctx: HybridContext, top_k: int, k: int = 60):
        """Reciprocal Rank Fusion. Source is now Neo4j vector, not FAISS."""
        fusion_scores: Dict[str, float] = {}

        def _row_key(entity: Dict[str, Any], rank: int) -> str:
            """Stable string key: Neo4j may return int ids; None id must not desync rank vs entity_map."""
            for key in ("id", "user_id", "post_id"):
                v = entity.get(key)
                if v is not None and v != "":
                    return str(v)
            return f"__row_{rank}__"

        if ctx.graph_context and ctx.graph_context.primary_entities:
            for rank, entity in enumerate(ctx.graph_context.primary_entities[:top_k]):
                eid = _row_key(entity, rank)
                fusion_scores[eid] = (
                    fusion_scores.get(eid, 0)
                    + self.graph_weight * (1.0 / (k + rank + 1))
                )

        if ctx.vector_context and ctx.vector_context.results:
            for rank, result in enumerate(ctx.vector_context.results[:top_k]):
                eid = _row_key(result, rank)
                sim_boost = result.get("similarity_score", 0.5)
                fusion_scores[eid] = (
                    fusion_scores.get(eid, 0)
                    + self.vector_weight * (1.0 / (k + rank + 1)) * (1 + sim_boost)
                )

        sorted_ids = sorted(fusion_scores, key=lambda x: fusion_scores[x], reverse=True)

        entity_map: Dict[str, Dict] = {}
        if ctx.graph_context:
            for rank, e in enumerate(ctx.graph_context.primary_entities):
                eid = _row_key(e, rank)
                entity_map[eid] = {**e, "source": "graph"}
        if ctx.vector_context:
            for rank, e in enumerate(ctx.vector_context.results):
                eid = _row_key(e, rank)
                if eid not in entity_map:
                    entity_map[eid] = {**e, "source": "neo4j_vector"}
                else:
                    entity_map[eid]["similarity_score"] = e.get("similarity_score")
                    entity_map[eid]["source"] = "neo4j_hybrid"

        fused = [
            {**entity_map[eid], "fusion_score": round(fusion_scores[eid], 6)}
            for eid in sorted_ids[:top_k]
            if eid in entity_map
        ]
        return fused, fusion_scores
