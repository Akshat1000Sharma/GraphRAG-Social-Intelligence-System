"""
Query Analyzer Agent: Parses natural language queries into structured intent.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FRIEND_RECOMMENDATION = "friend_recommendation"
    INFLUENCER_DETECTION = "influencer_detection"
    TRENDING_POSTS = "trending_posts"
    EXPLAIN_CONNECTION = "explain_connection"
    LINK_PREDICTION = "link_prediction"
    USER_PROFILE = "user_profile"
    CONTENT_SEARCH = "content_search"
    GENERAL_GRAPH = "general_graph"
    UNKNOWN = "unknown"


class RetrievalStrategy(str, Enum):
    GRAPH_ONLY = "graph"
    VECTOR_ONLY = "vector"
    HYBRID = "hybrid"


@dataclass
class AnalyzedQuery:
    """Structured output of the query analyzer."""
    raw_query: str
    intent: QueryIntent
    retrieval_strategy: RetrievalStrategy
    entities: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 10
    requires_explanation: bool = False
    confidence: float = 1.0
    reasoning: str = ""


# ─── Intent keyword mappings ───────────────────────────────────────────────────

INTENT_PATTERNS = {
    QueryIntent.FRIEND_RECOMMENDATION: [
        r"recommend.*friend",
        r"friend.*recommend",  # "Friend recommendations for …" (noun-first phrasing)
        r"recommendations?\s+for",
        r"suggest.*friend",
        r"who should.*connect",
        r"friend.*suggest",
        r"people.*know",
        r"new.*connection",
        r"who.*follow",
        r"similar.*user",
        r"\bfof\b",
        r"friend-of-friend",
        r"friends?\s+of\s+friends?",
    ],
    QueryIntent.INFLUENCER_DETECTION: [
        r"influencer", r"influence", r"top.*user", r"popular.*user",
        r"most.*follower", r"best.*poster", r"viral", r"reach",
        r"who.*power", r"key.*player",
    ],
    QueryIntent.TRENDING_POSTS: [
        r"trending", r"popular.*post", r"hot.*post", r"viral.*post",
        r"most.*liked", r"top.*post", r"what.*people.*talk",
        r"buzz", r"viral.*content",
    ],
    QueryIntent.EXPLAIN_CONNECTION: [
        r"how.*connect", r"connection.*between", r"relation.*between",
        r"path.*between", r"why.*recommend", r"explain.*why",
        r"how.*know", r"link.*between",
    ],
    QueryIntent.LINK_PREDICTION: [
        r"predict.*link", r"will.*connect", r"likely.*friend",
        r"future.*connection", r"potential.*friend",
    ],
    QueryIntent.USER_PROFILE: [
        r"who is", r"tell.*about.*user", r"user.*info",
        r"profile.*of", r"about.*person", r"\bbio\b",
    ],
    QueryIntent.CONTENT_SEARCH: [
        r"find.*post", r"search.*post", r"posts.*about",
        r"content.*about", r"articles.*about",
    ],
    QueryIntent.GENERAL_GRAPH: [
        r"graph", r"network", r"statistics", r"overview",
        r"summary", r"analyze",
    ],
}

# Retrieval strategy hints
GRAPH_KEYWORDS = {"friend", "connection", "network", "path", "mutual", "shortest", "recommend"}
VECTOR_KEYWORDS = {"similar", "semantic", "related", "like", "about", "topic", "content"}
EXPLANATION_KEYWORDS = {"why", "explain", "how", "reason", "because", "understand"}


class QueryAnalyzerAgent:
    """
    Agent 1: Parses user queries into structured intents and extraction parameters.
    """

    def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> AnalyzedQuery:
        """
        Analyze a natural language query and return structured intent.
        """
        q = query.lower().strip()
        intent = self._classify_intent(q)
        strategy = self._determine_strategy(q, intent)
        entities = self._extract_entities(query, context or {})
        constraints = self._extract_constraints(q)
        requires_explanation = any(kw in q for kw in EXPLANATION_KEYWORDS)
        top_k = self._extract_top_k(q)

        result = AnalyzedQuery(
            raw_query=query,
            intent=intent,
            retrieval_strategy=strategy,
            entities=entities,
            constraints=constraints,
            top_k=top_k,
            requires_explanation=requires_explanation,
            reasoning=f"Classified as '{intent}' based on keyword matching",
        )

        logger.debug(f"Query analyzed: intent={intent}, strategy={strategy}, entities={entities}")
        return result

    def _classify_intent(self, query: str) -> QueryIntent:
        """Match query against intent patterns."""
        scores = {intent: 0 for intent in QueryIntent}

        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    scores[intent] += 1

        best_intent = max(scores, key=lambda x: scores[x])
        if scores[best_intent] == 0:
            return QueryIntent.UNKNOWN
        return best_intent

    def _determine_strategy(self, query: str, intent: QueryIntent) -> RetrievalStrategy:
        """Determine whether to use graph, vector, or hybrid retrieval."""
        words = set(query.split())
        has_graph = bool(words & GRAPH_KEYWORDS)
        has_vector = bool(words & VECTOR_KEYWORDS)

        # Intent-based defaults
        intent_strategy_map = {
            QueryIntent.FRIEND_RECOMMENDATION: RetrievalStrategy.HYBRID,
            QueryIntent.INFLUENCER_DETECTION: RetrievalStrategy.GRAPH_ONLY,
            QueryIntent.TRENDING_POSTS: RetrievalStrategy.GRAPH_ONLY,
            QueryIntent.EXPLAIN_CONNECTION: RetrievalStrategy.GRAPH_ONLY,
            QueryIntent.LINK_PREDICTION: RetrievalStrategy.HYBRID,
            QueryIntent.USER_PROFILE: RetrievalStrategy.GRAPH_ONLY,
            QueryIntent.CONTENT_SEARCH: RetrievalStrategy.VECTOR_ONLY,
            QueryIntent.GENERAL_GRAPH: RetrievalStrategy.GRAPH_ONLY,
            QueryIntent.UNKNOWN: RetrievalStrategy.HYBRID,
        }

        base = intent_strategy_map.get(intent, RetrievalStrategy.HYBRID)

        # Override if explicit signals
        if has_graph and has_vector:
            return RetrievalStrategy.HYBRID
        if has_vector and not has_graph:
            return RetrievalStrategy.VECTOR_ONLY

        return base

    def _extract_entities(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user IDs, names, and other entities from query and context."""
        entities = {}

        def _nz(key: str) -> Optional[str]:
            v = context.get(key)
            if v is None:
                return None
            s = str(v).strip()
            return s or None

        # From context (API-level parameters) — only non-empty strings
        uid_ctx = _nz("user_id")
        if uid_ctx:
            entities["user_id"] = uid_ctx
        ua = _nz("user_a")
        if ua:
            entities["user_a"] = ua
        ub = _nz("user_b")
        if ub:
            entities["user_b"] = ub
        pid = _nz("post_id")
        if pid:
            entities["post_id"] = pid
        # R3: dataset scoping threaded from context
        if "dataset" in context and context["dataset"] is not None:
            ds = str(context["dataset"]).strip()
            if ds:
                entities["dataset"] = ds

        # Match specific dataset IDs (fb_0, tw_123, rd_456, etc.) or user_X
        specific_id_matches = re.findall(r"\b((?:fb|tw|rd|user)_[a-zA-Z0-9]+)\b", query, re.IGNORECASE)
        if specific_id_matches and not entities.get("user_id"):
            entities["user_id"] = specific_id_matches[0].lower()

        # user_id = 1 / user_id=fb_1 (word boundary: plain \bid would not match inside "user_id")
        uid_eq = re.search(r"\buser_id\s*=\s*([a-zA-Z0-9_]+)", query, re.IGNORECASE)
        if uid_eq and not entities.get("user_id"):
            entities["user_id"] = uid_eq.group(1).strip()

        # Match explicit IDs like 'id 4224', 'id=4224', 'id: fb_4224'
        id_matches = re.findall(r"\bid[:\s=_]+([a-zA-Z0-9_]+)", query, re.IGNORECASE)
        if id_matches and not entities.get("user_id"):
            matched = id_matches[0]
            if matched.isdigit():
                entities["user_id"] = matched
            else:
                entities["user_id"] = matched

        # "user 4224", "user #4224"
        if not entities.get("user_id"):
            user_num = re.search(
                r"\buser\s*#?\s*([a-zA-Z0-9_]+)\b", query, re.IGNORECASE
            )
            if user_num:
                entities["user_id"] = user_num.group(1)

        # Match standalone numbers if still no user_id (prefer last digit group — "user id 4224")
        if not entities.get("user_id"):
            number_matches = re.findall(r"\b(\d+)\b", query)
            if number_matches:
                ds = entities.get("dataset", "")
                prefix = ""
                if ds == "facebook":
                    prefix = "fb_"
                elif ds == "twitter":
                    prefix = "tw_"
                elif ds == "reddit":
                    prefix = "rd_"
                entities["user_id"] = f"{prefix}{number_matches[-1]}"

        return entities

    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract filters, date ranges, topics, etc."""
        constraints = {}

        # Topic extraction
        topic_match = re.search(r"about\s+(\w+)", query)
        if topic_match:
            constraints["topic"] = topic_match.group(1)

        # Time-based
        if any(t in query for t in ["today", "this week", "recent", "latest", "new"]):
            constraints["recency"] = "recent"

        # Role/type constraints
        if "influencer" in query:
            constraints["min_influence"] = 0.5
        if "verified" in query:
            constraints["verified_only"] = True

        return constraints

    def _extract_top_k(self, query: str) -> int:
        """Extract requested count from query."""
        # "top 5", "5 recommendations", etc.
        match = re.search(r"(?:top|best|show me)\s+(\d+)", query, re.IGNORECASE)
        if match:
            return min(int(match.group(1)), 50)

        match2 = re.search(r"(\d+)\s+(?:results|users|posts|friends)", query, re.IGNORECASE)
        if match2:
            return min(int(match2.group(1)), 50)

        return 10
