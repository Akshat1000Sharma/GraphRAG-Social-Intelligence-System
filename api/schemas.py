"""
API Schemas: Pydantic models for request/response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Request Models ────────────────────────────────────────────────────────────

class NLQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Natural language query")
    user_id: Optional[str] = Field(None, description="Optional user context")
    mode: Optional[str] = Field("hybrid", description="Retrieval mode: hybrid | graph | vector")
    top_k: int = Field(default=10, ge=1, le=50)

    model_config = {"json_schema_extra": {
        "example": {
            "query": "Who are the top influencers in the tech community?",
            "user_id": "user_1",
            "mode": "hybrid",
            "top_k": 10,
        }
    }}


class LinkPredictionRequest(BaseModel):
    user_id: str = Field(..., description="Source user for prediction")
    pairs: Optional[List[List[str]]] = Field(
        None, description="Explicit pairs to score: [['user_1', 'user_2'], ...]"
    )
    top_k: int = Field(default=10, ge=1, le=50)

    model_config = {"json_schema_extra": {
        "example": {"user_id": "user_1", "top_k": 10}
    }}


# ─── Response Models ───────────────────────────────────────────────────────────

class ValidationInfo(BaseModel):
    is_valid: bool
    confidence: float
    warnings: List[str] = []
    issues: List[str] = []


class PipelineTiming(BaseModel):
    analyzer: float
    router: float
    retrieval: float
    gnn_inference: float
    synthesizer: float
    validator: float
    total: float


class BaseGraphResponse(BaseModel):
    intent: str
    query: str
    results: List[Dict[str, Any]] = []
    gnn_predictions: List[Dict[str, Any]] = []
    insight: str = ""
    graph_context: str = ""
    retrieval_mode: str
    sources: List[str] = []
    validation: ValidationInfo
    pipeline_timing_ms: Optional[PipelineTiming] = None


class FriendRecommendationResponse(BaseGraphResponse):
    pass


class TrendingPostsResponse(BaseGraphResponse):
    pass


class InfluenceResponse(BaseGraphResponse):
    user_stats: Optional[Dict[str, Any]] = None


class ConnectionExplanationResponse(BaseGraphResponse):
    common_friends: List[Dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    gnn_loaded: bool
    gnn_datasets: List[str]
    pipeline_ready: bool
    version: str
