"""
Router Agent: Routes queries to the appropriate retrieval strategy.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from api.agents.analyzer import AnalyzedQuery, QueryIntent, RetrievalStrategy
from rag.hybrid_retrieval import HybridContext, HybridRetriever, RetrievalMode

logger = logging.getLogger(__name__)


class RouterAgent:
    """
    Agent 2: Routes analyzed queries to graph, vector, or hybrid retrievers.
    Decides execution path based on intent + strategy.
    """

    def route(
        self,
        analyzed: AnalyzedQuery,
        retriever: HybridRetriever,
    ) -> Tuple[str, Dict[str, Any], RetrievalMode]:
        """
        Returns: (query_type, params, retrieval_mode)
        """
        intent = analyzed.intent
        entities = analyzed.entities
        strategy = analyzed.retrieval_strategy

        # Map intent to Cypher query type
        INTENT_TO_QUERY = {
            QueryIntent.FRIEND_RECOMMENDATION: ("friend_recommendation", {"user_id": entities.get("user_id", "user_1")}),
            QueryIntent.INFLUENCER_DETECTION: ("influence_stats", {"user_id": entities.get("user_id", "user_1")}),
            QueryIntent.TRENDING_POSTS: ("trending_posts", {}),
            QueryIntent.EXPLAIN_CONNECTION: ("explain_connection", {
                "user_a": entities.get("user_a", entities.get("user_id", "user_1")),
                "user_b": entities.get("user_b", "user_2"),
            }),
            QueryIntent.LINK_PREDICTION: ("link_candidates", {"user_id": entities.get("user_id", "user_1")}),
            QueryIntent.USER_PROFILE: ("user_profile", {"user_id": entities.get("user_id", "user_1")}),
            QueryIntent.CONTENT_SEARCH: ("trending_posts", {}),
            QueryIntent.GENERAL_GRAPH: ("all_users", {}),
            QueryIntent.UNKNOWN: ("all_users", {}),
        }

        query_type, params = INTENT_TO_QUERY.get(intent, ("all_users", {}))

        # Merge additional constraints
        if analyzed.constraints.get("topic"):
            params["topic"] = analyzed.constraints["topic"]

        # Determine retrieval mode
        mode_map = {
            RetrievalStrategy.GRAPH_ONLY: RetrievalMode.GRAPH,
            RetrievalStrategy.VECTOR_ONLY: RetrievalMode.VECTOR,
            RetrievalStrategy.HYBRID: RetrievalMode.HYBRID,
        }
        mode = mode_map.get(strategy, RetrievalMode.HYBRID)

        logger.debug(f"Router: query_type={query_type}, mode={mode}, params={params}")
        return query_type, params, mode
