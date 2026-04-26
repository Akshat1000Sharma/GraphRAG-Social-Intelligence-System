"""
router.py  (REFACTORED — Neo4j-only routing)
=============================================
Routes analyzed queries to the appropriate Neo4j retrieval strategy.

Key changes from FAISS version:
  - REMOVED: FAISS-specific routing logic
  - ADDED: Neo4j-native hybrid routing for friend_recommendation and trending_posts
  - All retrieval modes (GRAPH / VECTOR / HYBRID) route through Neo4j

Routing table:
  FRIEND_RECOMMENDATION  → friend_recommendation → HYBRID  → graph+vector Cypher
  INFLUENCER_DETECTION   → influence_stats        → GRAPH   → Cypher traversal
  TRENDING_POSTS         → trending_posts         → HYBRID  → engagement+vector Cypher
  EXPLAIN_CONNECTION     → explain_connection     → GRAPH   → shortestPath (locked)
  LINK_PREDICTION        → link_candidates        → HYBRID  → multi-hop+vector
  USER_PROFILE           → user_profile           → GRAPH   → Cypher traversal
  CONTENT_SEARCH         → fulltext_search        → VECTOR  → post vector index
  GENERAL_GRAPH          → all_users              → GRAPH   → Cypher scan
"""

import logging
from typing import Any, Dict, Tuple

from api.agents.analyzer import AnalyzedQuery, QueryIntent, RetrievalStrategy
from rag.hybrid_retrieval import HybridContext, HybridRetriever, RetrievalMode

logger = logging.getLogger(__name__)


class RouterAgent:
    """
    Agent 2: Routes analyzed queries to the correct Neo4j retrieval strategy.
    All paths route through Neo4j — no FAISS, no external vector DB.
      GRAPH  → Cypher traversal
      VECTOR → db.index.vector.queryNodes procedures
      HYBRID → combined Cypher+vector (single Neo4j round-trip)
    """

    INTENT_ROUTING = {
        QueryIntent.FRIEND_RECOMMENDATION: {
            "query_type": "friend_recommendation",
            "preferred_mode": RetrievalMode.HYBRID,
            "lock_mode": False,
            "neo4j_capability": "graph_traversal + vector_index (native hybrid)",
        },
        QueryIntent.INFLUENCER_DETECTION: {
            "query_type": "influence_stats",
            "preferred_mode": RetrievalMode.GRAPH,
            "lock_mode": False,
            "neo4j_capability": "cypher_aggregation",
        },
        QueryIntent.TRENDING_POSTS: {
            "query_type": "trending_posts",
            "preferred_mode": RetrievalMode.HYBRID,
            "lock_mode": False,
            "neo4j_capability": "engagement_scoring + vector_index (native hybrid)",
        },
        QueryIntent.EXPLAIN_CONNECTION: {
            "query_type": "explain_connection",
            "preferred_mode": RetrievalMode.GRAPH,
            "lock_mode": True,  # always needs graph path
            "neo4j_capability": "shortestPath_cypher",
        },
        QueryIntent.LINK_PREDICTION: {
            "query_type": "link_candidates",
            "preferred_mode": RetrievalMode.HYBRID,
            "lock_mode": False,
            "neo4j_capability": "multi_hop_traversal + vector_index",
        },
        QueryIntent.USER_PROFILE: {
            "query_type": "user_profile",
            "preferred_mode": RetrievalMode.GRAPH,
            "lock_mode": True,
            "neo4j_capability": "cypher_aggregation",
        },
        QueryIntent.CONTENT_SEARCH: {
            "query_type": "fulltext_search",
            "preferred_mode": RetrievalMode.VECTOR,
            "lock_mode": False,
            "neo4j_capability": "post_text_vector_index",
        },
        QueryIntent.GENERAL_GRAPH: {
            "query_type": "all_users",
            "preferred_mode": RetrievalMode.GRAPH,
            "lock_mode": False,
            "neo4j_capability": "cypher_scan",
        },
        QueryIntent.UNKNOWN: {
            "query_type": "all_users",
            "preferred_mode": RetrievalMode.HYBRID,
            "lock_mode": False,
            "neo4j_capability": "graph_scan + vector_index",
        },
    }

    STRATEGY_TO_MODE = {
        RetrievalStrategy.GRAPH_ONLY:  RetrievalMode.GRAPH,
        RetrievalStrategy.VECTOR_ONLY: RetrievalMode.VECTOR,
        RetrievalStrategy.HYBRID:      RetrievalMode.HYBRID,
    }

    def route(
        self,
        analyzed: AnalyzedQuery,
        retriever: HybridRetriever,
    ) -> Tuple[str, Dict[str, Any], RetrievalMode]:
        """
        Returns (query_type, params, mode). All modes route through Neo4j.
        """
        intent   = analyzed.intent
        entities = analyzed.entities
        strategy = analyzed.retrieval_strategy

        routing    = self.INTENT_ROUTING.get(intent, self.INTENT_ROUTING[QueryIntent.UNKNOWN])
        query_type = routing["query_type"]
        params     = self._build_params(intent, entities, analyzed.constraints)

        # Lock mode for certain intents; otherwise honour analyzer strategy
        if routing["lock_mode"]:
            mode = routing["preferred_mode"]
        else:
            mode = self.STRATEGY_TO_MODE.get(strategy, routing["preferred_mode"])

        # R3: switch to dataset-scoped query template when dataset is specified
        dataset_in_params = params.get("dataset")
        if dataset_in_params and dataset_in_params not in ("all", None):
            ds_variant = query_type + "_ds"
            # Use _ds variant if it exists in GraphRetriever templates
            from rag.hybrid_retrieval import GraphRetriever
            if ds_variant in GraphRetriever.QUERY_TEMPLATES:
                query_type = ds_variant

        logger.debug(
            f"Router [Neo4j-only]: intent={intent}, query_type={query_type}, "
            f"mode={mode}, capability={routing['neo4j_capability']}, "
            f"dataset={params.get('dataset', 'all')}"
        )
        return query_type, params, mode

    def _build_params(self, intent, entities, constraints) -> Dict[str, Any]:
        def _uid(v, default="user_1"):
            if v is None:
                return default
            s = str(v).strip()
            return s if s else default

        raw_uid = entities.get("user_id")
        # Do not invent user_1 for profile-style queries — empty graph is better than wrong user
        if intent == QueryIntent.USER_PROFILE:
            user_id = _uid(raw_uid, default="")
        else:
            user_id = _uid(raw_uid, default="user_1")

        user_a = _uid(entities.get("user_a"), default=user_id or "user_1")
        user_b = _uid(entities.get("user_b"), default="user_2")
        # R3: thread dataset through to retrieval params
        dataset = entities.get("dataset") or constraints.get("dataset")

        base = {
            QueryIntent.FRIEND_RECOMMENDATION: {"user_id": user_id},
            QueryIntent.INFLUENCER_DETECTION:  {"user_id": user_id},
            QueryIntent.TRENDING_POSTS:        {},
            QueryIntent.EXPLAIN_CONNECTION:    {"user_a": user_a, "user_b": user_b},
            QueryIntent.LINK_PREDICTION:       {"user_id": user_id},
            QueryIntent.USER_PROFILE:          {"user_id": user_id},
            QueryIntent.CONTENT_SEARCH:        {"query": entities.get("search_query", "")},
            QueryIntent.GENERAL_GRAPH:         {},
            QueryIntent.UNKNOWN:               {},
        }.get(intent, {}).copy()

        if constraints.get("topic"):
            base["topic"] = constraints["topic"]
        if constraints.get("min_influence"):
            base["min_influence"] = constraints["min_influence"]

        # Pass dataset into params so GraphRetriever can use _ds query variants
        if dataset and dataset not in ("all", None):
            base["dataset"] = dataset

        return base
