"""
Retrievers Agent: Executes both graph and vector retrieval.
Wraps the hybrid retrieval engine with agent-level logic.
"""

import logging
from typing import Any, Dict, List, Optional
from rag.hybrid_retrieval import HybridContext, HybridRetriever, RetrievalMode, GraphRetriever, VectorRetriever

logger = logging.getLogger(__name__)


class RetrieversAgent:
    """
    Agent 3a/3b: Executes graph + vector retrieval in parallel (logically).
    Delegates to HybridRetriever for the actual execution.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def execute(
        self,
        query_type: str,
        params: Dict[str, Any],
        nl_query: str,
        mode: RetrievalMode,
        top_k: int = 10,
    ) -> HybridContext:
        """Execute retrieval and return hybrid context."""
        logger.debug(f"Retrievers: executing {query_type} in {mode} mode")
        return self.retriever.retrieve(
            query_type=query_type,
            params=params,
            nl_query=nl_query,
            mode=mode,
            top_k=top_k,
        )
