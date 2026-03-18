"""
Routes: Friend Recommendations & Link Prediction
These thin route modules delegate entirely to the multi-agent pipeline.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

router = APIRouter(prefix="/v1", tags=["Recommendations"])


# ─── Request/Response Models ──────────────────────────────────────────────────

class LinkPredictionRequest(BaseModel):
    user_id: str
    candidate_ids: Optional[List[str]] = None
    top_k: int = 10


class FriendRecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    gnn_predictions: List[Dict[str, Any]]
    insight: str
    validation: Dict[str, Any]
    pipeline_timing_ms: Dict[str, float]
