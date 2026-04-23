"""
api/schemas.py  (EXTENDED with chat, insert, dataset endpoints)
"""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import re

VALID_DATASETS = ["facebook", "twitter", "reddit", "demo", "all"]
VALID_MODES    = ["hybrid", "graph", "vector"]
VALID_GNN_DATASETS = ["facebook", "twitter", "reddit"]

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
    validation: Optional[ValidationInfo] = None
    pipeline_timing_ms: Optional[PipelineTiming] = None

class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    gnn_loaded: bool
    gnn_datasets: List[str]
    pipeline_ready: bool
    version: str
    dataset_counts: Optional[Dict[str, Dict]] = None

class DatasetStatus(BaseModel):
    name: str
    on_disk: bool
    files: Dict[str, bool] = {}
    last_ingest: Optional[str] = None
    ingest_version: Optional[str] = None
    neo4j_counts: Optional[Dict[str, int]] = None

class DatasetsStatusResponse(BaseModel):
    datasets: Dict[str, DatasetStatus]
    neo4j_connected: bool

class IngestResponse(BaseModel):
    triggered: List[str]
    results: Dict[str, Any]

class ChatRequest(BaseModel):
    """R4: NL question over Neo4j graph data."""
    message: str = Field(..., min_length=1, max_length=2000)
    dataset: Optional[str] = Field(default="all")
    """Override which pretrained GNN weights to run (optional). If unset, GNN follows `dataset` (all/demo → facebook)."""
    gnn_dataset: Optional[str] = Field(default=None)
    session_id: Optional[str] = None
    mode: Optional[str] = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=50)
    user_id: Optional[str] = None

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v):
        if v and v not in VALID_DATASETS:
            raise ValueError(f"dataset must be one of {VALID_DATASETS}")
        return v or "all"

    @field_validator("gnn_dataset")
    @classmethod
    def validate_gnn_dataset(cls, v: Optional[str]):
        if v is None or v == "":
            return None
        if v not in VALID_GNN_DATASETS:
            raise ValueError(f"gnn_dataset must be one of {VALID_GNN_DATASETS} or omitted")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v and v not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        return v or "hybrid"

    model_config = {"json_schema_extra": {"example": {
        "message": "Who are the most influential users in the facebook dataset?",
        "dataset": "facebook", "mode": "hybrid", "top_k": 10,
    }}}

class ChatResponse(BaseModel):
    message: str
    dataset_queried: str
    mode: str
    intent: str
    results: List[Dict[str, Any]] = []
    insight: str = ""
    datasets_cited: List[str] = []
    graph_context_summary: str = ""
    pipeline_timing_ms: Optional[Dict[str, float]] = None
    session_id: Optional[str] = None
    gnn_dataset_used: Optional[str] = Field(
        default=None,
        description="Pretrained GNN used (facebook|twitter|reddit), aligned to dataset scope for GNN",
    )

class InsertUserRequest(BaseModel):
    dataset: str = Field(...)
    name: str = Field(..., min_length=1, max_length=200)
    bio: Optional[str] = Field(default="", max_length=1000)
    source_id: Optional[str] = None
    follower_count: int = Field(default=0, ge=0)
    influence_score: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v):
        allowed = ["facebook", "twitter", "reddit", "demo"]
        if v not in allowed:
            raise ValueError(f"dataset must be one of {allowed}")
        return v

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v):
        return re.sub(r'[\x00-\x1f\x7f]', '', v).strip()

class InsertEdgeRequest(BaseModel):
    dataset: str
    from_user_id: str
    to_user_id: str
    rel_type: Literal["FRIEND", "FOLLOWS", "LIKED"] = Field(default="FRIEND")
    bidirectional: bool = Field(default=True)

class InsertPostRequest(BaseModel):
    dataset: str
    author_source_id: str
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(default="", max_length=5000)
    topic: str = Field(default="general")
    source_id: Optional[str] = None

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v):
        if v not in ["facebook", "twitter", "reddit", "demo"]:
            raise ValueError("dataset must be one of: facebook, twitter, reddit, demo")
        return v

class NLInsertRequest(BaseModel):
    """R5: Natural language insert command."""
    nl_command: str = Field(..., min_length=5, max_length=2000)
    dataset: Optional[str] = None
    confirm: bool = Field(default=False)

    model_config = {"json_schema_extra": {"example": {
        "nl_command": "Add user Alice who is friends with Bob in the facebook dataset",
        "confirm": True,
    }}}

class InsertResult(BaseModel):
    ok: bool
    operation: str
    nodes_created: int = 0
    edges_created: int = 0
    cypher_summary: str = ""
    detail: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    requires_confirm: bool = False
    preview: Optional[Dict[str, Any]] = None


# ── Direct graph (Neo4j) read API — structured rows from GraphQueryService ──

class NLInsertParseRequest(BaseModel):
    """Body for /chat/insert/parse — same fields as NLInsertRequest except confirm is unused."""
    nl_command: str = Field(..., min_length=5, max_length=2000)
    dataset: Optional[str] = None


class NLInsertParseResponse(BaseModel):
    ok: bool
    dataset: str = ""
    operations: List[Dict[str, Any]] = []
    parsed_names: List[str] = []
    error: Optional[str] = None
