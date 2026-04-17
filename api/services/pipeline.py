"""
Multi-Agent Pipeline Orchestrator
Wires together: Analyzer → Router → Retrievers → Synthesizer → Validator
"""

import time
import logging
from typing import Any, Dict, Optional

from rag.hybrid_retrieval import RetrievalMode

from api.agents.analyzer import QueryAnalyzerAgent, AnalyzedQuery
from api.agents.router import RouterAgent
from api.agents.retrievers import RetrieversAgent
from api.agents.synthesizer import SynthesizerAgent, SynthesizedResponse
from api.agents.validator import ValidatorAgent

logger = logging.getLogger(__name__)


class MultiAgentPipeline:
    """
    Orchestrates the full multi-agent reasoning pipeline.

    Flow:
    User Query
      → QueryAnalyzerAgent  (parse intent)
      → RouterAgent         (determine retrieval strategy)
      → RetrieversAgent     (execute graph + vector retrieval)
      → SynthesizerAgent    (merge GNN + context + LLM)
      → ValidatorAgent      (verify + format output)
      → Structured Response
    """

    def __init__(
        self,
        retriever,             # HybridRetriever instance
        inference_engine=None, # GNNInferenceEngine instance
    ):
        self.analyzer = QueryAnalyzerAgent()
        self.router = RouterAgent()
        self.retrievers = RetrieversAgent(retriever)
        self.synthesizer = SynthesizerAgent()
        self.validator = ValidatorAgent()
        self.inference_engine = inference_engine

    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline for a given query.
        Returns final structured + NL response.
        """
        t0 = time.time()
        context = context or {}

        # ── Step 1: Analyze query ──
        t1 = time.time()
        analyzed = self.analyzer.analyze(query, context)
        t1_elapsed = (time.time() - t1) * 1000

        # ── Step 2: Route query ──
        t2 = time.time()
        query_type, params, mode = self.router.route(analyzed, None)
        # Honor explicit retrieval mode from API (/chat, /query context)
        req_mode = (context.get("mode") or "").strip().lower()
        if req_mode == "graph":
            mode = RetrievalMode.GRAPH
        elif req_mode == "vector":
            mode = RetrievalMode.VECTOR
        elif req_mode == "hybrid":
            mode = RetrievalMode.HYBRID
        t2_elapsed = (time.time() - t2) * 1000

        # ── Step 3: Execute retrieval ──
        t3 = time.time()
        hybrid_ctx = self.retrievers.execute(
            query_type=query_type,
            params=params,
            nl_query=query,
            mode=mode,
            top_k=top_k,
        )
        t3_elapsed = (time.time() - t3) * 1000

        # ── Step 4: GNN inference (if engine available) ──
        t4 = time.time()
        gnn_predictions = self._run_gnn_inference(analyzed, hybrid_ctx)
        t4_elapsed = (time.time() - t4) * 1000

        # ── Step 5: Synthesize ──
        t5 = time.time()
        synthesized = self.synthesizer.synthesize(
            analyzed=analyzed,
            hybrid_ctx=hybrid_ctx,
            gnn_predictions=gnn_predictions,
            top_k=top_k,
        )
        t5_elapsed = (time.time() - t5) * 1000

        # ── Step 6: Validate ──
        t6 = time.time()
        report = self.validator.validate(analyzed, synthesized)
        t6_elapsed = (time.time() - t6) * 1000

        # ── Step 7: Format final response ──
        final = self.validator.format_final_response(analyzed, synthesized, report)

        total_ms = (time.time() - t0) * 1000
        final["pipeline_timing_ms"] = {
            "analyzer": round(t1_elapsed, 2),
            "router": round(t2_elapsed, 2),
            "retrieval": round(t3_elapsed, 2),
            "gnn_inference": round(t4_elapsed, 2),
            "synthesizer": round(t5_elapsed, 2),
            "validator": round(t6_elapsed, 2),
            "total": round(total_ms, 2),
        }

        logger.info(
            f"Pipeline complete: intent={analyzed.intent}, "
            f"results={len(final['results'])}, total={total_ms:.1f}ms"
        )

        return final

    def _run_gnn_inference(
        self,
        analyzed: AnalyzedQuery,
        hybrid_ctx,
    ) -> list:
        """Run GNN inference if engine is available and context permits."""
        if self.inference_engine is None:
            return []

        try:
            import torch
            import numpy as np
            from api.agents.analyzer import QueryIntent

            # Build minimal graph from retrieved entities
            entities = hybrid_ctx.fused_entities
            if not entities:
                return []

            num_nodes = max(len(entities), 10)
            # Synthetic features for demo inference
            x = torch.randn(num_nodes, self.inference_engine.config["in_channels"])
            # Fully connected demo graph
            src = list(range(num_nodes)) * (num_nodes - 1)
            dst = [j for i in range(num_nodes) for j in range(num_nodes) if i != j]
            edge_index = torch.tensor([src[:num_nodes*2], dst[:num_nodes*2]], dtype=torch.long)

            if analyzed.intent == QueryIntent.INFLUENCER_DETECTION:
                preds = self.inference_engine.classify_nodes(x, edge_index)
                return preds[:5]
            elif analyzed.intent in (QueryIntent.FRIEND_RECOMMENDATION, QueryIntent.LINK_PREDICTION):
                pairs = [(i, j) for i in range(min(5, num_nodes)) for j in range(min(5, num_nodes)) if i != j]
                preds = self.inference_engine.predict_link_probability(x, edge_index, pairs[:10])
                return preds[:5]
            else:
                return []
        except Exception as e:
            logger.debug(f"GNN inference skipped: {e}")
            return []
