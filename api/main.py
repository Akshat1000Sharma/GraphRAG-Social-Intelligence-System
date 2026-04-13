"""
FastAPI Application: Social Network Intelligence API
Endpoints: friend recommendations, influencer detection, trending posts,
           link prediction, connection explanation, NL graph query.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load api/.env before any module reads os.environ (e.g. db.neo4j_client).
load_dotenv(Path(__file__).resolve().parent / ".env")
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Internal imports
from db.neo4j_client import get_neo4j_client
from model.inference import inference_manager
from rag.vector_store import get_text_store, get_user_index, get_post_index
from rag.hybrid_retrieval import (
    HybridRetriever,
    GraphRetriever,
    VectorRetriever,
    RetrievalMode,
)
from api.services.pipeline import MultiAgentPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App-level state ──────────────────────────────────────────────────────────

class AppState:
    pipeline: Optional[MultiAgentPipeline] = None
    neo4j = None
    retriever: Optional[HybridRetriever] = None


app_state = AppState()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, clean up on shutdown."""
    logger.info("Starting Social Graph Intelligence API...")

    # ── Neo4j ──
    try:
        app_state.neo4j = get_neo4j_client()
        if app_state.neo4j.is_connected:
            app_state.neo4j.setup_schema()
            app_state.neo4j.seed_demo_data()
            logger.info("Neo4j connected and schema initialized")
        else:
            logger.warning("Neo4j not connected — graph queries will fail gracefully")
    except Exception as e:
        logger.warning(f"Neo4j initialization error: {e}")

    # ── GNN Models ──
    try:
        inference_manager.load_dataset("facebook")
        logger.info("GNN model loaded")
    except Exception as e:
        logger.warning(f"GNN model load error: {e}")

    # ── Vector stores ──
    try:
        text_store = get_text_store()
        user_index = get_user_index()
        post_index = get_post_index()
        logger.info("Vector stores initialized")
    except Exception as e:
        logger.warning(f"Vector store error: {e}")
        text_store = get_text_store()
        user_index = get_user_index()
        post_index = get_post_index()

    # ── Hybrid retriever ──
    graph_retriever = GraphRetriever(app_state.neo4j) if app_state.neo4j else None
    vector_retriever = VectorRetriever(text_store, user_index, post_index)

    class NullGraphRetriever:
        def retrieve(self, *args, **kwargs):
            from rag.hybrid_retrieval import GraphContext
            return GraphContext(query_type="null", raw_records=[])

    app_state.retriever = HybridRetriever(
        graph_retriever=graph_retriever or NullGraphRetriever(),
        vector_retriever=vector_retriever,
    )

    # ── Multi-agent pipeline ──
    gnn_engine = inference_manager.get_engine("facebook") if inference_manager.engines else None
    app_state.pipeline = MultiAgentPipeline(
        retriever=app_state.retriever,
        inference_engine=gnn_engine,
    )

    logger.info("All components initialized. API ready.")
    yield

    # Shutdown
    if app_state.neo4j:
        app_state.neo4j.close()
    logger.info("Shutdown complete.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Social Graph Intelligence API",
    description="GNN + GraphRAG + Multi-Agent Pipeline for Social Network Analysis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """System health check."""
    neo4j_ok = app_state.neo4j.is_connected if app_state.neo4j else False
    gnn_ok = len(inference_manager.engines) > 0

    return {
        "status": "healthy",
        "neo4j_connected": neo4j_ok,
        "gnn_loaded": gnn_ok,
        "gnn_datasets": list(inference_manager.engines.keys()),
        "pipeline_ready": app_state.pipeline is not None,
        "version": "1.0.0",
    }


# ─── Friend Recommendations ───────────────────────────────────────────────────

@app.get("/recommend-friends/{user_id}", tags=["Recommendations"])
async def recommend_friends(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
):
    """
    Recommend friends for a user using GNN link prediction + graph traversal.
    Uses multi-agent pipeline with hybrid GraphRAG.
    """
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    result = app_state.pipeline.run(
        query=f"Recommend {top_k} new friends for user {user_id}",
        context={"user_id": user_id},
        top_k=top_k,
    )
    return result


# ─── Link Prediction ──────────────────────────────────────────────────────────

@app.post("/predict-links", tags=["Predictions"])
async def predict_links(body: dict):
    """
    Predict connection probability between user pairs.
    Body: {"pairs": [["user_1", "user_2"], ...]}
    """
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    pairs = body.get("pairs", [])
    user_id = body.get("user_id", "user_1")

    result = app_state.pipeline.run(
        query=f"Predict potential connections for {user_id}",
        context={"user_id": user_id, "pairs": pairs},
        top_k=len(pairs) if pairs else 10,
    )
    return result


# ─── User Influence ───────────────────────────────────────────────────────────

@app.get("/user-influence/{user_id}", tags=["Analytics"])
async def user_influence(user_id: str):
    """
    Compute influence score for a user combining GNN classification + graph metrics.
    """
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    result = app_state.pipeline.run(
        query=f"What is the influence and role of user {user_id}?",
        context={"user_id": user_id},
        top_k=1,
    )

    # If Neo4j is connected, augment with direct graph stats
    if app_state.neo4j and app_state.neo4j.is_connected:
        try:
            stats = app_state.neo4j.run_query(
                """
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
                OPTIONAL MATCH (u)-[:FRIEND]-(f:User)
                WITH u, count(DISTINCT p) AS posts, count(DISTINCT f) AS friends,
                     coalesce(sum(p.like_count), 0) AS total_likes
                RETURN u.id AS id, u.name AS name, u.follower_count AS followers,
                       u.influence_score AS gnn_score, posts, friends, total_likes
                """,
                {"user_id": user_id},
            )
            if stats:
                result["user_stats"] = stats[0]
        except Exception as e:
            logger.warning(f"User stats query failed: {e}")

    return result


# ─── Trending Posts ───────────────────────────────────────────────────────────

@app.get("/trending-posts", tags=["Analytics"])
async def trending_posts(
    top_k: int = Query(default=10, ge=1, le=50),
    topic: Optional[str] = Query(default=None),
):
    """
    Get trending posts ranked by engagement velocity.
    """
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    query = f"Show me the top {top_k} trending posts"
    if topic:
        query += f" about {topic}"

    result = app_state.pipeline.run(
        query=query,
        context={"topic": topic} if topic else {},
        top_k=top_k,
    )
    return result


# ─── Explain Connection ───────────────────────────────────────────────────────

@app.get("/explain-connection", tags=["Explainability"])
async def explain_connection(
    user_a: str = Query(..., description="First user ID"),
    user_b: str = Query(..., description="Second user ID"),
):
    """
    Explain the connection between two users using graph path analysis + LLM.
    """
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    result = app_state.pipeline.run(
        query=f"Explain the connection between {user_a} and {user_b}",
        context={"user_a": user_a, "user_b": user_b},
        top_k=5,
    )

    # Add shortest path info
    if app_state.neo4j and app_state.neo4j.is_connected:
        try:
            common_friends = app_state.neo4j.run_query(
                """
                MATCH (a:User {id: $user_a})-[:FRIEND]->(common)<-[:FRIEND]-(b:User {id: $user_b})
                RETURN common.id AS id, common.name AS name
                """,
                {"user_a": user_a, "user_b": user_b},
            )
            result["common_friends"] = common_friends
        except Exception as e:
            logger.warning(f"Common friends query failed: {e}")

    return result


# ─── Natural Language Query (GraphRAG) ────────────────────────────────────────

@app.post("/query", tags=["GraphRAG"])
async def natural_language_query(body: dict):
    """
    Natural language query over the social graph using GraphRAG + multi-agent pipeline.

    Body: {
      "query": "Who are the top influencers in the tech space?",
      "user_id": "user_1" (optional context),
      "mode": "hybrid" | "graph" | "vector",
      "top_k": 10
    }
    """
    if not app_state.pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    query = body.get("query", "")
    if not query:
        raise HTTPException(400, "Query field is required")

    context = {k: v for k, v in body.items() if k not in ("query", "top_k")}
    top_k = body.get("top_k", 10)

    result = app_state.pipeline.run(
        query=query,
        context=context,
        top_k=top_k,
    )
    return result


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
        workers=1,
    )
