"""
Multi-Agent Pipeline Orchestrator
Wires together: Analyzer → Router → Retrievers → Synthesizer → Validator
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from rag.hybrid_retrieval import GraphContext, HybridContext, RetrievalMode

from api.agents.analyzer import QueryAnalyzerAgent, AnalyzedQuery, QueryIntent
from api.agents.router import RouterAgent
from api.agents.retrievers import RetrieversAgent
from api.agents.synthesizer import SynthesizerAgent, SynthesizedResponse
from api.agents.validator import ValidatorAgent

logger = logging.getLogger(__name__)

# Which pretrained GNN to run for a request (from /chat `dataset` context).
_GNN_DATASETS = frozenset({"facebook", "twitter", "reddit"})


def resolve_gnn_dataset_from_context(context: Optional[Dict[str, Any]]) -> str:
    """
    Choose GNN weights: explicit ``gnn_dataset`` in context wins; else map from ``dataset``;
    ``all`` / ``demo`` / missing → facebook.
    """
    ctx = context or {}
    gnn = ctx.get("gnn_dataset")
    if isinstance(gnn, str) and gnn in _GNN_DATASETS:
        return gnn
    ds = ctx.get("dataset")
    if isinstance(ds, str) and ds in _GNN_DATASETS:
        return ds
    return "facebook"


def _looks_like_friend_recommendation_query(query: str) -> bool:
    """
    Heuristic: detect friend-recommendation requests even if the rule-based intent is UNKNOWN
    (e.g. phrasing that does not match ``recommend.*friend`` order) or a tie in intent scoring.
    """
    if not (query and query.strip()):
        return False
    q = query.lower()
    if "friend" in q and "recommend" in q:
        return True
    if "recommendation" in q and "user" in q and (
        "friend" in q or "mutual" in q
    ):
        return True
    if re.search(r"\buser_id\s*=", q, re.I):
        return True
    if re.search(r"^recommend \d+ new friends for user ", q.strip(), re.I):
        return True
    if re.search(r"\bfor user\b", q) and ("friend" in q or "recommend" in q or "recommendation" in q):
        return True
    return False


def _single_user_friend_recommendation_context(
    query: str, context: Optional[Dict[str, Any]]
) -> bool:
    """
    If we have a concrete ``user_id`` in context and the query is about friend recommendations,
    do not expand ``dataset: all`` into three parallel platform runs (which triples work and
    can return empty/merged results incorrectly).
    """
    ctx = context or {}
    if not (ctx.get("user_id") and str(ctx.get("user_id", "")).strip()):
        return False
    return _looks_like_friend_recommendation_query(query)


if TYPE_CHECKING:
    from model.inference import MultiDatasetInferenceManager
    from api.services.graph_service import GraphQueryService


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
        retriever,  # HybridRetriever instance
        inference_manager: Optional["MultiDatasetInferenceManager"] = None,
        inference_engine=None,  # legacy: single GNNInferenceEngine
        graph_query_service: Optional["GraphQueryService"] = None,
    ):
        self.analyzer = QueryAnalyzerAgent()
        self.router = RouterAgent()
        self.retrievers = RetrieversAgent(retriever)
        self.synthesizer = SynthesizerAgent()
        self.validator = ValidatorAgent()
        self.inference_manager = inference_manager
        self.inference_engine = inference_engine
        # Same Neo4j logic as GET /graph/friend-recommendations — used when RAG Cypher returns no rows
        self.graph_query_service = graph_query_service

    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline for a given query.
        Dynamically selects datasets based on query keywords or context.
        If multiple datasets are selected, runs pipeline for each and combines results.
        """
        query_lower = query.lower()
        datasets_to_run = []

        # Platform names in the question (avoid treating substring "all" in words like "call" as "all datasets")
        if "facebook" in query_lower:
            datasets_to_run.append("facebook")
        if "twitter" in query_lower:
            datasets_to_run.append("twitter")
        if "reddit" in query_lower:
            datasets_to_run.append("reddit")

        # Fallback to context
        if not datasets_to_run:
            ctx = context or {}
            if (ctx.get("dataset") == "all" or ctx.get("gnn_dataset") == "all") and not _single_user_friend_recommendation_context(
                query, context
            ):
                datasets_to_run = ["facebook", "twitter", "reddit"]
            else:
                datasets_to_run = [resolve_gnn_dataset_from_context(context)]
                
        # Deduplicate preserving order
        seen = set()
        datasets_to_run = [x for x in datasets_to_run if not (x in seen or seen.add(x))]
        
        if len(datasets_to_run) == 1:
            ctx = dict(context or {})
            ctx["gnn_dataset"] = datasets_to_run[0]
            ctx["dataset"] = datasets_to_run[0]
            return self._run_single(query, ctx, top_k)
            
        # Multiple datasets
        combined_results = []
        combined_predictions = []
        combined_insights = []
        combined_timing = {}
        base_res = None
        
        for ds in datasets_to_run:
            ctx = dict(context or {})
            ctx["gnn_dataset"] = ds
            ctx["dataset"] = ds
            res = self._run_single(query, ctx, top_k)
            if not base_res:
                base_res = dict(res)
                
            combined_insights.append(f"[{ds.capitalize()}] {res.get('insight', '')}")
            
            for item in res.get("results", []):
                item_copy = dict(item)
                if "name" in item_copy:
                    item_copy["name"] = f"{item_copy['name']} ({ds})"
                elif "title" in item_copy:
                    item_copy["title"] = f"{item_copy['title']} ({ds})"
                elif "user_id" in item_copy:
                    item_copy["user_id"] = f"{item_copy['user_id']} ({ds})"
                combined_results.append(item_copy)
                
            for pred in res.get("gnn_predictions", []):
                pred_copy = dict(pred)
                if "node_id" in pred_copy:
                    pred_copy["node_id"] = f"{pred_copy['node_id']} ({ds})"
                elif "source_id" in pred_copy:
                    pred_copy["source_id"] = f"{pred_copy['source_id']} ({ds})"
                combined_predictions.append(pred_copy)
                
            for k, v in res.get("pipeline_timing_ms", {}).items():
                combined_timing[k] = round(combined_timing.get(k, 0) + v, 2)
                
        final = base_res or {}
        final["results"] = combined_results
        final["gnn_predictions"] = combined_predictions
        final["insight"] = "\n\n".join(combined_insights)
        final["gnn_dataset_used"] = ", ".join(datasets_to_run)
        final["pipeline_timing_ms"] = combined_timing
        return final

    def _run_single(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Internal: Execute the pipeline for a single dataset.
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
        # ONLY if the router hasn't locked the mode for this intent
        routing = self.router.INTENT_ROUTING.get(analyzed.intent, {})
        is_locked = routing.get("lock_mode", False)

        if not is_locked:
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
        use_graph_friends = self.graph_query_service and (
            analyzed.intent == QueryIntent.FRIEND_RECOMMENDATION
            or _looks_like_friend_recommendation_query(query)
        )
        if use_graph_friends:
            # Same rows as GET /graph/friend-recommendations/{user_id} (GraphQueryService)
            uid = self._resolve_friend_rec_user_id(analyzed, context, params, query)
            if uid:
                rows = self._fetch_friend_recommendations_like_rest(
                    uid, context, top_k=top_k
                )
                hybrid_ctx = self._hybrid_from_friend_recommendation_rows(rows, top_k=top_k)
            else:
                hybrid_ctx = self.retrievers.execute(
                    query_type=query_type,
                    params=params,
                    nl_query=query,
                    mode=mode,
                    top_k=top_k,
                )
                hybrid_ctx = self._friend_recommendation_service_fallback(
                    analyzed, hybrid_ctx, context, params, query, top_k=top_k
                )
        else:
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
        gnn_predictions, gnn_dataset_used = self._run_gnn_inference(
            analyzed, hybrid_ctx, context
        )
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
        final["gnn_dataset_used"] = gnn_dataset_used

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

    def _resolve_friend_rec_user_id(
        self,
        analyzed: AnalyzedQuery,
        context: Dict[str, Any],
        params: Dict[str, Any],
        query: str,
    ) -> str:
        """
        1) Explicit ``user_id = X`` in the message
        2) ``user_id`` from API (``/recommend-friends/{id}``, /chat)
        3) Entity / router params, skipping bogus ``user_1`` default
        4) ``for user X`` / "Recommend N new friends for user X" in the text
        """
        m = re.search(r"\buser_id\s*=\s*([a-zA-Z0-9_]+)", query, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        for src in (context.get("user_id"),):
            if src is not None and str(src).strip() and str(src).strip() != "user_1":
                return str(src).strip()
        for src in (analyzed.entities.get("user_id"), params.get("user_id")):
            if src is None or str(src).strip() == "":
                continue
            s = str(src).strip()
            if s and s != "user_1":
                return s
        for pat in (
            r"^recommend \d+ new friends for user ([a-zA-Z0-9_]+)(?:\b|\s)",
            r"\bnew friends for user\s*([a-zA-Z0-9_]+)\b",
            r"\bfor user\s*#?\s*([a-zA-Z0-9_]+)\b",
        ):
            m2 = re.search(pat, query, re.IGNORECASE)
            if m2:
                return m2.group(1).strip()
        for src in (analyzed.entities.get("user_id"), context.get("user_id"), params.get("user_id")):
            if src is not None and str(src).strip():
                return str(src).strip()
        return ""

    def _fetch_friend_recommendations_like_rest(
        self, uid: str, context: Dict[str, Any], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Call the same service as ``GET /graph/friend-recommendations/{id}``;
        try id variants (plain id vs fb_/tw_/rd_ prefix) if the first call is empty.
        """
        if not self.graph_query_service or not uid:
            return []

        def _get(u: str) -> List[Dict[str, Any]]:
            try:
                return (
                    self.graph_query_service.friend_recommendations_for_llm(
                        u, top_k=top_k
                    )
                    or []
                )
            except Exception as e:
                logger.warning("friend_recommendations_for_llm(%r): %s", u, e)
                return []

        r = _get(uid)
        if r:
            return r
        if uid.isdigit():
            ds = resolve_gnn_dataset_from_context(context)
            pref = {"facebook": "fb_", "twitter": "tw_", "reddit": "rd_"}.get(
                ds, "fb_"
            )
            r = _get(f"{pref}{uid}")
            if r:
                return r
            for p in ("fb_", "tw_", "rd_"):
                if p == pref:
                    continue
                r = _get(f"{p}{uid}")
                if r:
                    return r
        return []

    def _hybrid_from_friend_recommendation_rows(
        self, rows: List[Dict[str, Any]], top_k: int
    ) -> HybridContext:
        rows = (rows or [])[:top_k]
        gc = GraphContext(
            query_type="friend_recommendation",
            raw_records=rows,
            primary_entities=list(rows),
            cypher_used="GraphQueryService.get_friend_recommendations",
        )
        h = HybridContext(
            graph_context=gc,
            retrieval_mode=RetrievalMode.GRAPH,
            fused_entities=[{**r, "source": "graph", "fusion_score": 1.0} for r in rows],
        )
        h.metadata["fusion_method"] = "graph_service"
        return h

    def _friend_recommendation_service_fallback(
        self,
        analyzed: AnalyzedQuery,
        hybrid_ctx: HybridContext,
        context: Dict[str, Any],
        params: Dict[str, Any],
        query: str,
        top_k: int,
    ) -> HybridContext:
        """
        When the retriever has no friend candidates, use ``GraphQueryService`` as for GET.
        """
        if analyzed.intent != QueryIntent.FRIEND_RECOMMENDATION:
            return hybrid_ctx
        if hybrid_ctx.fused_entities:
            return hybrid_ctx
        if not self.graph_query_service:
            return hybrid_ctx
        uid = self._resolve_friend_rec_user_id(analyzed, context, params, query)
        if not uid:
            return hybrid_ctx
        rows = self._fetch_friend_recommendations_like_rest(uid, context, top_k=top_k)
        if not rows:
            return hybrid_ctx
        return self._hybrid_from_friend_recommendation_rows(rows, top_k=top_k)

    def _engine_for_gnn(
        self, context: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[Any], str]:
        """Return (GNN engine or None, dataset key for API `gnn_dataset_used`)."""
        ds = resolve_gnn_dataset_from_context(context)
        if self.inference_manager is not None:
            try:
                return self.inference_manager.get_engine(ds), ds
            except Exception as e:
                logger.warning("GNN engine %s: %s", ds, e)
            if ds != "facebook":
                try:
                    return self.inference_manager.get_engine("facebook"), "facebook"
                except Exception as e2:
                    logger.warning("GNN facebook fallback: %s", e2)
            return None, ds
        if self.inference_engine is not None:
            return self.inference_engine, "facebook"
        return None, "none"

    def _run_gnn_inference(
        self,
        analyzed: AnalyzedQuery,
        hybrid_ctx,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], str]:
        """Run GNN for the dataset implied by `context` (from /chat `dataset` field)."""
        engine, gnn_label = self._engine_for_gnn(context)
        if engine is None:
            return [], gnn_label

        try:
            import torch
            from api.agents.analyzer import QueryIntent

            # Build minimal graph from retrieved entities
            entities = hybrid_ctx.fused_entities
            if not entities:
                return [], gnn_label

            num_nodes = max(len(entities), 10)
            in_ch = int(engine.config["in_channels"])
            # Synthetic features for demo inference
            x = torch.randn(num_nodes, in_ch)
            # Fully connected demo graph
            src = list(range(num_nodes)) * (num_nodes - 1)
            dst = [j for i in range(num_nodes) for j in range(num_nodes) if i != j]
            edge_index = torch.tensor(
                [src[: num_nodes * 2], dst[: num_nodes * 2]], dtype=torch.long
            )

            if analyzed.intent == QueryIntent.INFLUENCER_DETECTION:
                preds = engine.classify_nodes(x, edge_index)
                return preds[:5], gnn_label
            if analyzed.intent in (
                QueryIntent.FRIEND_RECOMMENDATION,
                QueryIntent.LINK_PREDICTION,
            ):
                pairs = [
                    (i, j)
                    for i in range(min(5, num_nodes))
                    for j in range(min(5, num_nodes))
                    if i != j
                ]
                preds = engine.predict_link_probability(
                    x, edge_index, pairs[:10]
                )
                return preds[:5], gnn_label
            return [], gnn_label
        except Exception as e:
            logger.debug("GNN inference skipped: %s", e)
            return [], gnn_label
