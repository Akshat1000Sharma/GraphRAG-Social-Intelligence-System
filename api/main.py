"""
main.py  (v3 — datasets + chat + chat/insert)
=============================================
Startup order:
  1. Neo4j connect + base schema
  2. Dataset presence check + download (R1)
  3. Dataset schema extensions + bulk ingest (R2)
  4. Neo4j vector indexes
  5. Embedding population
  6. GNN models
  7. Wire retriever + pipeline + chat services

New endpoints: /datasets/status, /datasets/ingest, /chat, /chat/insert
Existing endpoints: unchanged (preserved)
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

# Load env before db.neo4j_client reads os.environ: project root first, then api/ (overrides).
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from db.neo4j_client import get_neo4j_client
from model.inference import inference_manager
from rag.neo4j_vector_store import (
    Neo4jVectorSchemaManager,
    Neo4jEmbeddingPopulator,
    get_text_engine,
)
from rag.hybrid_retrieval import (
    HybridRetriever,
    GraphRetriever,
    VectorRetriever,
    RetrievalMode,
)
from api.services.pipeline import MultiAgentPipeline
from api.services.chat_service import ChatService, InsertService
from api.services.graph_service import GraphQueryService
from api.schemas import (
    ChatRequest, ChatResponse,
    NLInsertRequest, InsertResult,
    InsertUserRequest, InsertEdgeRequest, InsertPostRequest,
    NLInsertParseRequest, NLInsertParseResponse,
    DatasetsStatusResponse, DatasetStatus, IngestResponse,
)
from api.bootstrap.config import ALL_DATASETS, AUTO_INGEST, FORCE_REINGEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AppState:
    neo4j              = None
    pipeline           = None
    retriever          = None
    schema_manager     = None
    embedding_populator = None
    chat_service       = None
    insert_service     = None
    graph_service      = None
    dataset_statuses   = {}
    ingest_results     = {}


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Social Graph Intelligence v3 starting up ===")

    # ── Step 1: Neo4j base connection + schema ────────────────────────────────
    try:
        app_state.neo4j = get_neo4j_client()
        if app_state.neo4j.is_connected:
            app_state.neo4j.setup_schema()
            logger.info("Neo4j connected and base schema initialized")
        else:
            logger.warning("Neo4j not connected — degraded mode")
    except Exception as e:
        logger.warning(f"Neo4j init error: {e}")

    # ── Step 2: Dataset download / ensure presence (R1) ───────────────────────
    try:
        from api.bootstrap.datasets import ensure_all_datasets
        app_state.dataset_statuses = ensure_all_datasets()
    except Exception as e:
        logger.warning(f"Dataset bootstrap error: {e}")

    # ── Step 3: Dataset schema + bulk ingest into Neo4j (R2) ─────────────────
    if app_state.neo4j and app_state.neo4j.is_connected and AUTO_INGEST:
        try:
            from db.ingest.ingest_all import ingest_all_if_needed
            app_state.ingest_results = ingest_all_if_needed(
                app_state.neo4j,
                force=FORCE_REINGEST,
            )
        except Exception as e:
            logger.warning(f"Dataset ingest error: {e}")
    elif app_state.neo4j and app_state.neo4j.is_connected:
        # Always seed demo data if no real datasets loaded
        try:
            app_state.neo4j.seed_demo_data()
        except Exception as e:
            logger.debug(f"Demo seed (non-critical): {e}")

    # ── Step 4: Neo4j vector indexes ──────────────────────────────────────────
    if app_state.neo4j and app_state.neo4j.is_connected:
        try:
            app_state.schema_manager = Neo4jVectorSchemaManager(app_state.neo4j)
            app_state.schema_manager.create_all_indexes()
        except Exception as e:
            logger.warning(f"Vector index setup: {e}")

    # ── Step 5: Embedding population ──────────────────────────────────────────
    if app_state.neo4j and app_state.neo4j.is_connected:
        try:
            text_engine = get_text_engine()
            app_state.embedding_populator = Neo4jEmbeddingPopulator(
                neo4j_client=app_state.neo4j,
                text_engine=text_engine,
            )
            app_state.embedding_populator.populate_all(force_refresh=False)
        except Exception as e:
            logger.warning(f"Embedding population: {e}")

    # ── Step 6: GNN models ────────────────────────────────────────────────────
    try:
        inference_manager.load_dataset("facebook")
    except Exception as e:
        logger.warning(f"GNN model load: {e}")

    # ── Step 7: Wire retriever + pipeline + services ──────────────────────────
    text_engine = get_text_engine()

    class _NullGraph:
        def retrieve(self, *a, **kw):
            from rag.hybrid_retrieval import GraphContext
            return GraphContext(query_type="null", raw_records=[])

    graph_ret = (
        GraphRetriever(app_state.neo4j)
        if app_state.neo4j and app_state.neo4j.is_connected
        else _NullGraph()
    )
    vec_ret = (
        VectorRetriever(neo4j_client=app_state.neo4j, text_engine=text_engine)
        if app_state.neo4j and app_state.neo4j.is_connected
        else None
    )

    class _NullVector:
        model_name = "none"
        def search_users(self, *a, **kw):
            from rag.hybrid_retrieval import VectorContext
            return VectorContext(query="", results=[])
        def search_posts(self, *a, **kw):
            from rag.hybrid_retrieval import VectorContext
            return VectorContext(query="", results=[])
        def search_by_gnn_embedding(self, *a, **kw):
            from rag.hybrid_retrieval import VectorContext
            return VectorContext(query="", results=[])

    if vec_ret is None:
        vec_ret = _NullVector()

    app_state.retriever = HybridRetriever(graph_retriever=graph_ret, vector_retriever=vec_ret)
    gnn_engine = inference_manager.get_engine("facebook") if inference_manager.engines else None
    app_state.pipeline  = MultiAgentPipeline(retriever=app_state.retriever, inference_engine=gnn_engine)
    app_state.chat_service   = ChatService(pipeline=app_state.pipeline, neo4j_client=app_state.neo4j)
    app_state.insert_service = InsertService(neo4j_client=app_state.neo4j)
    app_state.graph_service  = GraphQueryService(app_state.neo4j)

    logger.info("=== API ready ===")
    yield

    if app_state.neo4j:
        app_state.neo4j.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Social Graph Intelligence API",
    description="GNN + GraphRAG + Multi-Agent — Neo4j unified backend + dataset ingest + chat",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH (extended with per-dataset counts)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health():
    neo4j_ok = app_state.neo4j.is_connected if app_state.neo4j else False
    dataset_counts = {}
    if neo4j_ok:
        try:
            from db.ingest.ingest_all import get_dataset_counts
            for ds in ["facebook", "twitter", "reddit", "demo"]:
                dataset_counts[ds] = get_dataset_counts(app_state.neo4j, ds)
        except Exception:
            pass

    vector_indexes = []
    if app_state.schema_manager:
        try:
            vector_indexes = app_state.schema_manager.get_index_status()
        except Exception:
            pass

    return {
        "status": "healthy",
        "neo4j_connected": neo4j_ok,
        "vector_backend": "neo4j",
        "vector_indexes": [{"name": i.get("name"), "state": i.get("state")} for i in vector_indexes],
        "gnn_loaded": len(inference_manager.engines) > 0,
        "gnn_datasets": list(inference_manager.engines.keys()),
        "pipeline_ready": app_state.pipeline is not None,
        "version": "3.0.0",
        "dataset_counts": dataset_counts,
        "ingest_results": {k: v.get("status") for k, v in app_state.ingest_results.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATASET MANAGEMENT (R1/R2)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/datasets/status", tags=["Datasets"])
async def datasets_status():
    """R1/R2: Which datasets are on disk + last ingest time + Neo4j counts."""
    neo4j_ok = app_state.neo4j.is_connected if app_state.neo4j else False
    out = {}
    for name, manifest in ALL_DATASETS.items():
        files_present = {f.filename: (manifest.dir / f.filename).exists() for f in manifest.files}
        neo4j_counts = None
        if neo4j_ok:
            try:
                from db.ingest.ingest_all import get_dataset_counts
                neo4j_counts = get_dataset_counts(app_state.neo4j, name)
            except Exception:
                pass
        marker = manifest.marker_path()
        out[name] = DatasetStatus(
            name=name,
            on_disk=manifest.all_required_present(),
            files=files_present,
            last_ingest=marker.read_text().strip() if marker.exists() else None,
            ingest_version=manifest.ingest_version,
            neo4j_counts=neo4j_counts,
        )
    return DatasetsStatusResponse(datasets=out, neo4j_connected=neo4j_ok)


@app.post("/datasets/ingest", tags=["Datasets"])
async def trigger_ingest(dataset: Optional[str] = Query(default=None), force: bool = Query(default=False)):
    """Admin: trigger re-ingest of one or all datasets."""
    if not app_state.neo4j or not app_state.neo4j.is_connected:
        raise HTTPException(503, "Neo4j not connected")
    from db.ingest.ingest_all import ingest_dataset, ingest_all_if_needed
    if dataset:
        result = ingest_dataset(app_state.neo4j, dataset, force=force)
        return IngestResponse(triggered=[dataset], results={dataset: result})
    else:
        results = ingest_all_if_needed(app_state.neo4j, force=force)
        return IngestResponse(triggered=list(results.keys()), results=results)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT (R4)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", tags=["Chat"], response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    R4: Natural language Q&A over Neo4j-stored graph data.
    Accepts dataset scope: facebook | twitter | reddit | demo | all.
    """
    if not app_state.chat_service:
        raise HTTPException(503, "Chat service not initialized")
    return app_state.chat_service.query(req)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INSERT (R5)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat/insert", tags=["Chat"], response_model=InsertResult)
async def chat_insert(req: NLInsertRequest):
    """
    R5: Natural language graph insertion.
    Example: {"nl_command": "Add user Alice who is friends with Bob in facebook", "confirm": true}
    Set confirm=false (default) to preview without executing.
    """
    if not app_state.insert_service:
        raise HTTPException(503, "Insert service not initialized")
    return app_state.insert_service.execute_nl_insert(req)


@app.post("/chat/insert/user", tags=["Chat"], response_model=InsertResult)
async def insert_user_structured(req: InsertUserRequest, confirm: bool = Query(default=False)):
    """Structured user insert (skip NL parsing). confirm=true to execute."""
    if not app_state.insert_service:
        raise HTTPException(503, "Insert service not initialized")
    return app_state.insert_service.insert_user(req, preview_only=not confirm)


@app.post("/chat/insert/edge", tags=["Chat"], response_model=InsertResult)
async def insert_edge_structured(req: InsertEdgeRequest, confirm: bool = Query(default=False)):
    """Structured edge insert. confirm=true to execute."""
    if not app_state.insert_service:
        raise HTTPException(503, "Insert service not initialized")
    return app_state.insert_service.insert_edge(req, preview_only=not confirm)


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING ENDPOINTS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/recommend-friends/{user_id}", tags=["Recommendations"])
async def recommend_friends(user_id: str, top_k: int = Query(default=10, ge=1, le=50),
                            dataset: Optional[str] = Query(default="all")):
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    if not app_state.neo4j or not app_state.neo4j.is_connected:
        raise HTTPException(
            503,
            "Neo4j is not connected, so the social graph cannot be queried. "
            "Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in api/.env (or project .env) "
            "to match your Docker Neo4j credentials, restart the API, then run ingest if needed.",
        )
    return app_state.pipeline.run(
        query=f"Recommend {top_k} new friends for user {user_id}",
        context={"user_id": user_id, "dataset": dataset},
        top_k=top_k,
    )


@app.post("/predict-links", tags=["Predictions"])
async def predict_links(body: dict):
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return app_state.pipeline.run(
        query=f"Predict potential connections for {body.get('user_id', 'user_1')}",
        context={"user_id": body.get("user_id", "user_1"), "dataset": body.get("dataset", "all")},
        top_k=body.get("top_k", 10),
    )


@app.get("/user-influence/{user_id}", tags=["Analytics"])
async def user_influence(user_id: str, dataset: Optional[str] = Query(default="all")):
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return app_state.pipeline.run(
        query=f"What is the influence and role of user {user_id}?",
        context={"user_id": user_id, "dataset": dataset},
        top_k=1,
    )


@app.get("/trending-posts", tags=["Analytics"])
async def trending_posts(top_k: int = Query(default=10, ge=1, le=50),
                         topic: Optional[str] = Query(default=None),
                         dataset: Optional[str] = Query(default="all")):
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    query = f"Show me the top {top_k} trending posts"
    if topic:
        query += f" about {topic}"
    return app_state.pipeline.run(
        query=query,
        context={"topic": topic, "dataset": dataset} if topic else {"dataset": dataset},
        top_k=top_k,
    )


@app.get("/explain-connection", tags=["Explainability"])
async def explain_connection(user_a: str = Query(...), user_b: str = Query(...),
                              dataset: Optional[str] = Query(default="all")):
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return app_state.pipeline.run(
        query=f"Explain the connection between {user_a} and {user_b}",
        context={"user_a": user_a, "user_b": user_b, "dataset": dataset},
        top_k=5,
    )


@app.post("/query", tags=["GraphRAG"])
async def natural_language_query(body: dict):
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    if not body.get("query"):
        raise HTTPException(400, "Query field is required")
    return app_state.pipeline.run(
        query=body["query"],
        context={k: v for k, v in body.items() if k not in ("query", "top_k")},
        top_k=body.get("top_k", 10),
    )


@app.get("/vector-indexes", tags=["System"])
async def list_vector_indexes():
    if not app_state.schema_manager:
        raise HTTPException(503, "Schema manager not initialized")
    return {"indexes": app_state.schema_manager.get_index_status()}


@app.post("/refresh-embeddings", tags=["System"])
async def refresh_embeddings(force: bool = False):
    if not app_state.embedding_populator:
        raise HTTPException(503, "Embedding populator not initialized")
    counts = app_state.embedding_populator.populate_all(force_refresh=force)
    return {"status": "ok", "counts": counts}


# ══════════════════════════════════════════════════════════════════════════════
# DIRECT GRAPH QUERIES (GraphQueryService — structured Neo4j rows, no LLM pipeline)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/graph/friend-recommendations/{user_id}", tags=["Graph (direct)"])
async def graph_friend_recommendations(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=100),
):
    """Friend-of-friend suggestions from Cypher (mutual friend counts)."""
    if not app_state.graph_service:
        raise HTTPException(503, "Graph service not initialized")
    rows = app_state.graph_service.get_friend_recommendations(user_id, top_k=top_k)
    return {"user_id": user_id, "top_k": top_k, "recommendations": rows}


@app.get("/graph/trending-posts", tags=["Graph (direct)"])
async def graph_trending_posts(
    top_k: int = Query(default=10, ge=1, le=100),
    topic: Optional[str] = Query(default=None),
    hours_window: int = Query(default=48, ge=1, le=24 * 30),
):
    """Trending posts by engagement score (likes + 2×comments)."""
    if not app_state.graph_service:
        raise HTTPException(503, "Graph service not initialized")
    rows = app_state.graph_service.get_trending_posts(
        top_k=top_k, topic=topic, hours_window=hours_window
    )
    return {"top_k": top_k, "topic": topic, "hours_window": hours_window, "posts": rows}


@app.get("/graph/users/{user_id}/influence-stats", tags=["Graph (direct)"])
async def graph_user_influence_stats(user_id: str):
    """Per-user activity and influence fields from Neo4j (not the multi-agent pipeline)."""
    if not app_state.graph_service:
        raise HTTPException(503, "Graph service not initialized")
    return app_state.graph_service.get_user_influence_stats(user_id)


@app.get("/graph/connection-path", tags=["Graph (direct)"])
async def graph_connection_path(
    user_a: str = Query(..., description="First user id or source_id"),
    user_b: str = Query(..., description="Second user id or source_id"),
):
    """Shortest path, mutual friends, and common liked posts between two users."""
    if not app_state.graph_service:
        raise HTTPException(503, "Graph service not initialized")
    return app_state.graph_service.get_connection_path(user_a, user_b)


@app.get("/graph/link-prediction/candidates/{user_id}", tags=["Graph (direct)"])
async def graph_link_prediction_candidates(
    user_id: str,
    top_k: int = Query(default=20, ge=1, le=200),
):
    """2–3 hop non-friend candidates scored by path count (graph-side candidates for GNN)."""
    if not app_state.graph_service:
        raise HTTPException(503, "Graph service not initialized")
    rows = app_state.graph_service.get_link_prediction_candidates(user_id, top_k=top_k)
    return {"user_id": user_id, "top_k": top_k, "candidates": rows}


@app.get("/graph/top-influencers", tags=["Graph (direct)"])
async def graph_top_influencers(top_k: int = Query(default=20, ge=1, le=200)):
    """Network-wide ranked users by a composite of followers, avg likes, and post count."""
    if not app_state.graph_service:
        raise HTTPException(503, "Graph service not initialized")
    rows = app_state.graph_service.get_all_top_influencers(top_k=top_k)
    return {"top_k": top_k, "influencers": rows}


# ══════════════════════════════════════════════════════════════════════════════
# GNN / MODEL STATUS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/gnn/status", tags=["System"])
async def gnn_status():
    """Which pretrained GNN engines are loaded in memory."""
    from model.inference import DATASET_CONFIG

    return {
        "loaded_datasets": list(inference_manager.engines.keys()),
        "all_configured_datasets": list(DATASET_CONFIG.keys()),
        "load_state": inference_manager.status(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INSERT — extra helpers (parse preview, structured post)
# ══════════════════════════════════════════════════════════════════════════════

def _nl_parse_to_response(parsed: dict) -> NLInsertParseResponse:
    if not parsed.get("ok"):
        return NLInsertParseResponse(
            ok=False,
            error=parsed.get("error"),
            dataset=(parsed.get("dataset") or ""),
        )
    ops_out = []
    for op in parsed.get("operations", []):
        payload = op.get("payload")
        pl = payload.model_dump() if hasattr(payload, "model_dump") else payload
        ops_out.append({"type": op["type"], "payload": pl})
    return NLInsertParseResponse(
        ok=True,
        dataset=parsed.get("dataset") or "",
        operations=ops_out,
        parsed_names=list(parsed.get("parsed_names") or []),
    )


@app.post("/chat/insert/parse", tags=["Chat"], response_model=NLInsertParseResponse)
async def chat_insert_parse(req: NLInsertParseRequest):
    """
    Parse a natural-language insert command into structured operations (preview only).
    Does not write to the database. Use for UI step-before-confirm.
    """
    if not app_state.insert_service:
        raise HTTPException(503, "Insert service not initialized")
    from api.schemas import NLInsertRequest
    inner = NLInsertRequest(nl_command=req.nl_command, dataset=req.dataset, confirm=False)
    parsed = app_state.insert_service.parse_nl_insert(inner)
    return _nl_parse_to_response(parsed)


@app.post("/chat/insert/post", tags=["Chat"], response_model=InsertResult)
async def insert_post_structured(req: InsertPostRequest, confirm: bool = Query(default=False)):
    """
    Create a Post linked to an existing User (by dataset + author source_id).
    The author User must already exist (e.g. from /chat/insert/user). confirm=true to execute.
    """
    if not app_state.insert_service:
        raise HTTPException(503, "Insert service not initialized")
    return app_state.insert_service.insert_post(req, preview_only=not confirm)


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
        workers=1,
    )
