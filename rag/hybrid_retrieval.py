"""
Hybrid Retrieval: Fuses Cypher (graph) and vector (semantic) retrieval.
Core of the GraphRAG system.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    GRAPH = "graph"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class GraphContext:
    """Structured context retrieved from Neo4j graph traversal."""
    query_type: str
    primary_entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    raw_records: List[Dict[str, Any]] = field(default_factory=list)
    cypher_used: str = ""


@dataclass
class VectorContext:
    """Context retrieved from semantic vector similarity search."""
    query: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    model_used: str = ""


@dataclass
class HybridContext:
    """Fused context combining graph and vector retrieval."""
    graph_context: Optional[GraphContext] = None
    vector_context: Optional[VectorContext] = None
    fused_entities: List[Dict[str, Any]] = field(default_factory=list)
    fusion_scores: Dict[str, float] = field(default_factory=dict)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphRetriever:
    """
    Executes Cypher queries against Neo4j and structures results.
    """

    QUERY_TEMPLATES = {
        "friend_recommendation": """
            MATCH (u:User {id: $user_id})-[:FRIEND]->(friend)-[:FRIEND]->(fof:User)
            WHERE fof.id <> $user_id AND NOT (u)-[:FRIEND]->(fof)
            WITH fof, count(friend) AS mutual_count
            ORDER BY mutual_count DESC LIMIT $top_k
            RETURN fof.id AS id, fof.name AS name,
                   mutual_count AS mutual_friends, fof.influence_score AS influence_score
        """,
        "user_profile": """
            MATCH (u:User {id: $user_id})
            OPTIONAL MATCH (u)-[:FRIEND]->(f:User)
            OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
            RETURN u.id AS id, u.name AS name, u.bio AS bio,
                   u.influence_score AS influence,
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
    }

    def __init__(self, neo4j_client):
        self.client = neo4j_client

    def retrieve(
        self,
        query_type: str,
        params: Dict[str, Any],
        top_k: int = 10,
    ) -> GraphContext:
        """Execute a structured graph query and return context."""
        if query_type not in self.QUERY_TEMPLATES:
            logger.warning(f"Unknown query type: {query_type}")
            return GraphContext(query_type=query_type)

        cypher = self.QUERY_TEMPLATES[query_type]
        params["top_k"] = top_k

        try:
            records = self.client.run_query(cypher, params)
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            records = []

        ctx = GraphContext(
            query_type=query_type,
            raw_records=records,
            cypher_used=cypher.strip(),
        )

        # Extract primary entities and relationships
        for r in records:
            if "id" in r or "user_id" in r or "post_id" in r:
                ctx.primary_entities.append(r)
            if "rel_types" in r:
                ctx.relationships.append(r)
            if "node_names" in r:
                ctx.paths.append(r.get("node_names", []))

        return ctx

    def custom_query(self, cypher: str, params: Dict[str, Any]) -> GraphContext:
        """Execute a custom Cypher query."""
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


class VectorRetriever:
    """
    Performs semantic similarity search over embedded users and posts.
    """

    def __init__(self, text_store, user_index, post_index):
        self.text_store = text_store
        self.user_index = user_index
        self.post_index = post_index

    def search_users(self, query: str, top_k: int = 10) -> VectorContext:
        """Find semantically similar users."""
        query_emb = self.text_store.embed_text(query)
        results = self.user_index.search(query_emb, top_k=top_k)
        return VectorContext(
            query=query,
            results=results,
            model_used=self.text_store.model_name,
        )

    def search_posts(self, query: str, top_k: int = 10) -> VectorContext:
        """Find semantically similar posts."""
        query_emb = self.text_store.embed_text(query)
        results = self.post_index.search(query_emb, top_k=top_k)
        return VectorContext(
            query=query,
            results=results,
            model_used=self.text_store.model_name,
        )

    def search_by_gnn_embedding(
        self,
        gnn_embedding: "np.ndarray",
        top_k: int = 10,
    ) -> VectorContext:
        """Find users by GNN embedding similarity (uses user index if GNN embs stored)."""
        results = self.user_index.search(gnn_embedding, top_k=top_k)
        return VectorContext(query="gnn_embedding_search", results=results)


class HybridRetriever:
    """
    Combines graph traversal and vector retrieval with score fusion.
    """

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_retriever: VectorRetriever,
        graph_weight: float = 0.6,
        vector_weight: float = 0.4,
    ):
        self.graph = graph_retriever
        self.vector = vector_retriever
        self.graph_weight = graph_weight
        self.vector_weight = vector_weight

    def retrieve(
        self,
        query_type: str,
        params: Dict[str, Any],
        nl_query: str = "",
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 10,
    ) -> HybridContext:
        """
        Perform hybrid retrieval and fuse results.
        """
        ctx = HybridContext(retrieval_mode=mode)

        # ── Graph retrieval ──
        if mode in (RetrievalMode.GRAPH, RetrievalMode.HYBRID):
            ctx.graph_context = self.graph.retrieve(query_type, params, top_k=top_k)

        # ── Vector retrieval ──
        if mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID) and nl_query:
            # Determine whether to search users or posts
            if any(kw in nl_query.lower() for kw in ["post", "content", "trending", "topic"]):
                ctx.vector_context = self.vector.search_posts(nl_query, top_k=top_k)
            else:
                ctx.vector_context = self.vector.search_users(nl_query, top_k=top_k)

        # ── Fusion ──
        ctx.fused_entities, ctx.fusion_scores = self._fuse(ctx, top_k)

        return ctx

    def _fuse(
        self,
        ctx: HybridContext,
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Reciprocal Rank Fusion of graph and vector results.
        """
        fusion_scores: Dict[str, float] = {}

        # Graph results
        if ctx.graph_context and ctx.graph_context.primary_entities:
            for rank, entity in enumerate(ctx.graph_context.primary_entities[:top_k]):
                entity_id = entity.get("id") or entity.get("user_id") or entity.get("post_id") or str(rank)
                rrf_score = 1.0 / (60 + rank + 1)
                fusion_scores[entity_id] = fusion_scores.get(entity_id, 0) + self.graph_weight * rrf_score

        # Vector results
        if ctx.vector_context and ctx.vector_context.results:
            for rank, result in enumerate(ctx.vector_context.results[:top_k]):
                entity_id = result.get("id") or result.get("user_id") or str(rank)
                rrf_score = 1.0 / (60 + rank + 1)
                # Boost by actual similarity score
                sim_boost = result.get("similarity_score", 0.5)
                fusion_scores[entity_id] = (
                    fusion_scores.get(entity_id, 0) + self.vector_weight * rrf_score * (1 + sim_boost)
                )

        # Sort by fused score
        sorted_ids = sorted(fusion_scores, key=lambda x: fusion_scores[x], reverse=True)

        # Collect entities in fused order
        entity_map: Dict[str, Dict] = {}
        if ctx.graph_context:
            for e in ctx.graph_context.primary_entities:
                eid = e.get("id") or e.get("user_id") or e.get("post_id", "")
                if eid:
                    entity_map[eid] = {**e, "source": "graph"}
        if ctx.vector_context:
            for e in ctx.vector_context.results:
                eid = e.get("id") or e.get("user_id", "")
                if eid and eid not in entity_map:
                    entity_map[eid] = {**e, "source": "vector"}
                elif eid:
                    entity_map[eid]["similarity_score"] = e.get("similarity_score")
                    entity_map[eid]["source"] = "hybrid"

        fused = []
        for eid in sorted_ids[:top_k]:
            if eid in entity_map:
                fused.append({**entity_map[eid], "fusion_score": round(fusion_scores[eid], 6)})

        return fused, fusion_scores
