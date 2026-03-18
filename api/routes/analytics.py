"""
Routes: Analytics — Influencer Detection & Trending Posts
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

router = APIRouter(prefix="/v1/analytics", tags=["Analytics"])


class InfluencerResponse(BaseModel):
    user_id: str
    influence_score: float
    role: str
    role_probabilities: Dict[str, float]
    graph_stats: Optional[Dict[str, Any]] = None
    insight: str


class TrendingPostsResponse(BaseModel):
    posts: List[Dict[str, Any]]
    insight: str
    topic_filter: Optional[str] = None
