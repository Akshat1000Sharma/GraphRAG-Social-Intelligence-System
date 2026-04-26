"""
Synthesizer Agent: Combines GNN predictions + RAG context + LLM generation.
Core of the KAG (Knowledge-Augmented Generation) system.

LLM backend: Google Gemini via LangChain (langchain-google-genai).
Uses LangChain PromptTemplate + PydanticOutputParser for structured outputs.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from api.agents.analyzer import AnalyzedQuery, QueryIntent
from rag.hybrid_retrieval import HybridContext

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"


# ─── Pydantic Output Schemas ──────────────────────────────────────────────────

class GraphInsight(BaseModel):
    """Structured insight for friend recommendations and general queries."""
    summary: str = Field(description="A concise 2-3 sentence summary directly answering the user's query")
    key_findings: List[str] = Field(description="List of 2-4 specific findings grounded in the graph data")
    confidence_assessment: str = Field(description="Brief assessment of result confidence based on GNN scores and data quality")
    recommended_action: str = Field(description="One actionable recommendation for the user")


class InfluencerInsight(BaseModel):
    """Structured insight for influencer detection queries."""
    summary: str = Field(description="Summary of the user's influence in the network")
    influence_factors: List[str] = Field(description="List of factors driving their influence score")
    network_role: str = Field(description="The user's role in the network: regular_user, influencer, content_creator, or community_hub")
    comparison: str = Field(description="How this user compares to the average network participant")


class ConnectionExplanation(BaseModel):
    """Structured explanation for connection path queries."""
    relationship_summary: str = Field(description="Natural language summary of how the two users are connected")
    connection_strength: str = Field(description="Assessment of connection strength: strong, moderate, or weak")
    common_ground: List[str] = Field(description="List of shared interests, friends, or communities")
    recommendation: str = Field(description="Whether connecting is recommended and why")


class TrendingInsight(BaseModel):
    """Structured insight for trending post queries."""
    summary: str = Field(description="Overview of current trending topics")
    top_themes: List[str] = Field(description="List of dominant themes across trending posts")
    engagement_pattern: str = Field(description="Description of the engagement pattern observed")
    peak_topic: str = Field(description="The single most engaging topic right now")


# Intent → Pydantic schema mapping
INTENT_SCHEMA_MAP = {
    QueryIntent.FRIEND_RECOMMENDATION: GraphInsight,
    QueryIntent.LINK_PREDICTION: GraphInsight,
    QueryIntent.USER_PROFILE: GraphInsight,
    QueryIntent.CONTENT_SEARCH: GraphInsight,
    QueryIntent.GENERAL_GRAPH: GraphInsight,
    QueryIntent.UNKNOWN: GraphInsight,
    QueryIntent.INFLUENCER_DETECTION: InfluencerInsight,
    QueryIntent.EXPLAIN_CONNECTION: ConnectionExplanation,
    QueryIntent.TRENDING_POSTS: TrendingInsight,
}


# ─── Prompt Templates ─────────────────────────────────────────────────────────

KAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a Social Network Intelligence assistant with access to live graph data and GNN model predictions.

User Query: {user_query}
Detected Intent: {intent}

--- Graph Retrieval Context ---
{graph_context}

--- Top Retrieved Entities (fused graph + vector) ---
{top_results}

--- GNN Model Predictions ---
{gnn_predictions}

--- Instructions ---
Analyze the data above and generate a structured response.
Be specific: reference actual entity names, IDs, and scores from the data.
Do NOT invent entities or relationships not present in the context above.
Keep each field concise and factual.

{format_instructions}""",
    input_variables=["user_query", "intent", "graph_context", "top_results", "gnn_predictions"],
    partial_variables={},  # format_instructions injected per-call
)

FALLBACK_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a Social Network Intelligence assistant.

User Query: {user_query}
Context: {graph_context}
Data: {top_results}

Provide a concise, factual insight (under 100 words) grounded only in the data above.""",
    input_variables=["user_query", "graph_context", "top_results"],
)


# ─── Dataclass for internal response ──────────────────────────────────────────

@dataclass
class SynthesizedResponse:
    """Final synthesized output combining GNN + RAG + LLM."""
    intent: str
    structured_data: List[Dict[str, Any]] = field(default_factory=list)
    gnn_predictions: List[Dict[str, Any]] = field(default_factory=list)
    natural_language_insight: str = ""
    parsed_insight: Optional[Dict[str, Any]] = field(default=None)
    graph_context_summary: str = ""
    retrieval_mode: str = ""
    confidence: float = 0.8
    sources: List[str] = field(default_factory=list)


# ─── Synthesizer Agent ────────────────────────────────────────────────────────

class SynthesizerAgent:
    """
    Agent 4: Synthesizes GNN predictions with retrieved context using Gemini + LangChain.

    Pipeline per intent:
      PromptTemplate (KAG) | ChatGoogleGenerativeAI | PydanticOutputParser
                                                    ↕ (fallback)
                                                StrOutputParser
    """

    def __init__(self):
        self._model: Optional[ChatGoogleGenerativeAI] = None

    def _get_model(self) -> Optional[ChatGoogleGenerativeAI]:
        """Lazy-load the Gemini model."""
        if self._model is None and USE_LLM:
            try:
                self._model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
                logger.info(f"Gemini model loaded: {GEMINI_MODEL}")
            except Exception as e:
                logger.warning(f"Could not load Gemini model: {e}")
        return self._model

    # ── Public interface ───────────────────────────────────────────────────────

    def synthesize(
        self,
        analyzed: AnalyzedQuery,
        hybrid_ctx: HybridContext,
        gnn_predictions: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 10,
    ) -> SynthesizedResponse:
        """Combine GNN predictions + RAG context → structured + NL output."""
        response = SynthesizedResponse(intent=analyzed.intent)
        response.retrieval_mode = hybrid_ctx.retrieval_mode

        # Structured data from hybrid retrieval
        response.structured_data = hybrid_ctx.fused_entities[:top_k]

        # GNN predictions
        if gnn_predictions:
            response.gnn_predictions = gnn_predictions

        # Merge GNN scores into retrieved entities
        response.structured_data = self._merge_gnn_with_retrieved(
            response.structured_data, gnn_predictions or []
        )

        # Graph context summary (for prompt injection)
        response.graph_context_summary = self._summarize_graph_context(hybrid_ctx)
        if analyzed.intent == QueryIntent.USER_PROFILE and response.structured_data:
            response.graph_context_summary += (
                " The first JSON object is the focal user (id/name/bio/friends/posts); "
                "summarize only that record, not other users."
            )
        response.sources = self._identify_sources(hybrid_ctx)

        # LLM-generated insight (Pydantic-parsed or fallback)
        needs_llm = analyzed.requires_explanation or analyzed.intent in (
            QueryIntent.EXPLAIN_CONNECTION,
            QueryIntent.FRIEND_RECOMMENDATION,
            QueryIntent.INFLUENCER_DETECTION,
            QueryIntent.TRENDING_POSTS,
            QueryIntent.USER_PROFILE,
        )

        if needs_llm:
            parsed, insight_text = self._generate_structured_insight(analyzed, response, hybrid_ctx)
            response.parsed_insight = parsed
            response.natural_language_insight = insight_text
        else:
            response.natural_language_insight = self._template_insight(analyzed, response)

        return response

    # ── LangChain pipeline ────────────────────────────────────────────────────

    def _generate_structured_insight(
        self,
        analyzed: AnalyzedQuery,
        response: SynthesizedResponse,
        ctx: HybridContext,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        """
        Run the full LangChain pipeline:
          PromptTemplate | Gemini | PydanticOutputParser
        Returns (parsed_dict, insight_string).
        Falls back to StrOutputParser on parse failure.
        """
        model = self._get_model()
        if not model:
            return None, self._template_insight(analyzed, response)

        # Select the right Pydantic schema for this intent
        schema_class = INTENT_SCHEMA_MAP.get(analyzed.intent, GraphInsight)
        pydantic_parser = PydanticOutputParser(pydantic_object=schema_class)

        # Build prompt with format instructions injected
        prompt = KAG_PROMPT_TEMPLATE.partial(
            format_instructions=pydantic_parser.get_format_instructions()
        )

        # Render input variables
        top_results_str = json.dumps(response.structured_data[:5], indent=2, default=str)
        gnn_str = json.dumps(response.gnn_predictions[:3], indent=2, default=str) if response.gnn_predictions else "[]"

        # ── Primary chain: PromptTemplate | Gemini | PydanticOutputParser ──
        chain = prompt | model | pydantic_parser

        try:
            result = chain.invoke({
                "user_query": analyzed.raw_query,
                "intent": analyzed.intent,
                "graph_context": response.graph_context_summary,
                "top_results": top_results_str,
                "gnn_predictions": gnn_str,
            })

            parsed_dict = result.model_dump()
            # Flatten to readable string: join all string fields
            insight_text = self._flatten_pydantic_to_text(result)
            logger.debug(f"Pydantic-parsed insight ({schema_class.__name__}): {parsed_dict}")
            return parsed_dict, insight_text

        except Exception as e:
            logger.warning(f"PydanticOutputParser failed ({e}), falling back to StrOutputParser")
            return self._fallback_str_chain(model, analyzed, response)

    def _fallback_str_chain(
        self,
        model: ChatGoogleGenerativeAI,
        analyzed: AnalyzedQuery,
        response: SynthesizedResponse,
    ) -> tuple[None, str]:
        """
        Fallback chain when Pydantic parsing fails:
          FALLBACK_PROMPT_TEMPLATE | Gemini | StrOutputParser
        """
        str_parser = StrOutputParser()
        chain = FALLBACK_PROMPT_TEMPLATE | model | str_parser

        top_results_str = json.dumps(response.structured_data[:3], indent=2, default=str)

        try:
            text = chain.invoke({
                "user_query": analyzed.raw_query,
                "graph_context": response.graph_context_summary,
                "top_results": top_results_str,
            })
            return None, text.strip()
        except Exception as e:
            logger.warning(f"Fallback chain also failed: {e}")
            return None, self._template_insight(analyzed, response)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _flatten_pydantic_to_text(self, result: BaseModel) -> str:
        """Convert a parsed Pydantic object to a readable natural language string."""
        parts = []
        for field_name, value in result.model_dump().items():
            if isinstance(value, str) and value:
                parts.append(value)
            elif isinstance(value, list):
                parts.extend([str(v) for v in value if v])
        return " | ".join(parts)

    def _merge_gnn_with_retrieved(
        self,
        retrieved: List[Dict[str, Any]],
        gnn_preds: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Augment retrieved entities with GNN prediction scores."""
        gnn_map = {
            str(p.get("node_id", p.get("source_id", ""))): p
            for p in gnn_preds
        }
        result = []
        for entity in retrieved:
            eid = entity.get("id", entity.get("user_id", ""))
            if eid in gnn_map:
                entity = {
                    **entity,
                    "gnn_score": gnn_map[eid].get(
                        "probability", gnn_map[eid].get("confidence", 0.5)
                    ),
                }
            result.append(entity)
        result.sort(
            key=lambda x: x.get("gnn_score", x.get("fusion_score", 0)),
            reverse=True,
        )
        return result

    def _summarize_graph_context(self, ctx: HybridContext) -> str:
        """Create a text summary of graph retrieval results for the prompt."""
        parts = []
        if ctx.graph_context and ctx.graph_context.raw_records:
            n = len(ctx.graph_context.raw_records)
            parts.append(f"Graph query '{ctx.graph_context.query_type}' returned {n} records.")
            if ctx.graph_context.paths:
                paths_desc = "; ".join(
                    [" -> ".join(p) for p in ctx.graph_context.paths[:3]]
                )
                parts.append(f"Paths found: {paths_desc}")
        if ctx.vector_context and ctx.vector_context.results:
            n = len(ctx.vector_context.results)
            top_score = (
                ctx.vector_context.results[0].get("similarity_score", 0) if n > 0 else 0
            )
            parts.append(
                f"Vector search returned {n} results (top similarity: {top_score:.3f})."
            )
        return " ".join(parts) if parts else "No graph context available."

    def _identify_sources(self, ctx: HybridContext) -> List[str]:
        sources = []
        if ctx.graph_context and ctx.graph_context.raw_records:
            sources.append("neo4j_graph")
        if ctx.vector_context and ctx.vector_context.results:
            sources.append("vector_index")
        return sources

    def _template_insight(self, analyzed: AnalyzedQuery, response: SynthesizedResponse) -> str:
        """Generate a deterministic template insight when LLM is unavailable."""
        n = len(response.structured_data)
        intent_templates = {
            QueryIntent.FRIEND_RECOMMENDATION: (
                f"Found {n} friend recommendations based on mutual connections and network patterns."
            ),
            QueryIntent.TRENDING_POSTS: (
                f"Retrieved {n} trending posts ranked by engagement velocity."
            ),
            QueryIntent.INFLUENCER_DETECTION: (
                f"Identified {n} users with high influence scores across the network."
            ),
            QueryIntent.LINK_PREDICTION: (
                f"Predicted {n} potential new connections using GNN link prediction."
            ),
            QueryIntent.USER_PROFILE: "Retrieved user profile with network context.",
            QueryIntent.CONTENT_SEARCH: f"Found {n} relevant posts matching the query.",
            QueryIntent.EXPLAIN_CONNECTION: (
                "Connection path analysis completed. See results for path details."
            ),
        }
        return intent_templates.get(
            analyzed.intent, f"Retrieved {n} results for: {analyzed.raw_query}"
        )
