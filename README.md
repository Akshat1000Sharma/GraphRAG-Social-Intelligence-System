# Social Network Intelligence System

**GNN + GraphRAG + Multi-Agent KAG Pipeline**

A production-ready social network intelligence system combining:
- **Graph Neural Networks** (GraphSAGE + GAT) for link prediction & node classification
- **GraphRAG** — hybrid Neo4j Cypher + vector similarity retrieval
- **Multi-Agent Pipeline** — Analyzer → Router → Retrievers → Synthesizer → Validator
- **KAG** (Knowledge-Augmented Generation) — GNN predictions fused with LLM reasoning
- **FastAPI** backend with 7 endpoints, CPU-only inference

---

## Architecture Overview

```
      User Query
          │
          ▼
  ┌─────────────────┐
  │  Query Analyzer │  ← Parse intent, extract entities, pick strategy
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Router Agent   │  ← Map intent → query_type + retrieval mode
  └────────┬────────┘
           │
      ┌────┴────┐
      ▼         ▼
 ┌────────┐ ┌────────┐
 │ Graph  │ │ Vector │  ← Cypher (Neo4j) + Embedding similarity (FAISS)
 │Retriev │ │Retriev │
 └────┬───┘ └───┬────┘
      └────┬────┘
           │ Reciprocal Rank Fusion
           ▼
  ┌──────────────────┐
  │ GNN Inference    │  ← Link prediction / Node classification (CPU)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Synthesizer     │  ← KAG: merge GNN + RAG context → LLM prompt
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Validator       │  ← Grounding checks, deduplication, confidence
  └────────┬─────────┘
           │
           ▼
   Structured JSON Response
   + Natural Language Insight
```

---

## Project Structure

```
project_root/
├── training/
│   ├── train_facebook.py      ← Facebook Large Page-Page Network
│   ├── train_twitter.py       ← Twitter ego-network (GAT model)
│   └── train_reddit.py        ← Reddit community graph
│
├── model/
│   ├── gnn_model.py           ← SocialGraphGNN, GATSocialGNN, LinkPredictor, NodeClassifier
│   ├── inference.py           ← CPU-only inference engine, MultiDatasetInferenceManager
│   └── utils.py               ← Training utils, metrics, EarlyStopping
│
├── api/
│   ├── main.py                ← FastAPI app, lifespan, all endpoints
│   ├── routes/
│   │   ├── recommendations.py ← Pydantic models for rec endpoints
│   │   └── analytics.py       ← Pydantic models for analytics endpoints
│   ├── services/
│   │   ├── pipeline.py        ← Multi-agent orchestrator
│   │   └── graph_service.py   ← Business logic / Neo4j query service
│   └── agents/
│       ├── analyzer.py        ← QueryAnalyzerAgent (intent, entities, strategy)
│       ├── router.py          ← RouterAgent (query_type → retrieval mode)
│       ├── retrievers.py      ← RetrieversAgent (delegates to HybridRetriever)
│       ├── synthesizer.py     ← SynthesizerAgent (KAG: GNN + RAG + LLM)
│       └── validator.py       ← ValidatorAgent (grounding, dedup, confidence)
│
├── db/
│   ├── neo4j_client.py        ← Thread-safe Neo4j driver, schema setup, seed data
│   └── cypher_queries.cql     ← Full Cypher query library
│
├── rag/
│   ├── vector_store.py        ← SentenceTransformer + FAISS/numpy vector index
│   ├── embeddings.py          ← Embedding manager (GNN + text, Neo4j sync)
│   └── hybrid_retrieval.py    ← GraphRetriever + VectorRetriever + RRF fusion
│
├── weights/                   ← Pretrained model weights (git-ignored)
│   ├── model_weights_facebook.pth
│   ├── model_weights_twitter.pth
│   ├── model_weights_reddit.pth
│   ├── embeddings_facebook.npy
│   ├── embeddings_twitter.npy
│   └── embeddings_reddit.npy
│
├── tests/
│   └── test_all.py            ← Full test suite (unit + integration)
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── prometheus.yml
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone & Configure

```bash
git clone <repo-url>
cd social_graph_intelligence
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY
```

### 2. Train Models (Kaggle GPU — separate from API)

Upload training scripts to Kaggle. Each script runs independently:

```bash
# On Kaggle (GPU enabled):
python training/train_facebook.py --epochs 200 --output_dir weights/
python training/train_twitter.py  --epochs 300 --output_dir weights/
python training/train_reddit.py   --epochs 200 --output_dir weights/
```

Kaggle datasets to attach:
- `facebook-large-page-page-network` → for `train_facebook.py`
- No extra dataset needed for Twitter/Reddit (auto-downloads via PyG)

Download output files: `model_weights_*.pth`, `embeddings_*.npy` → place in `weights/`

### 3. Launch with Docker

```bash
cd docker
docker-compose up -d

# Check logs
docker-compose logs -f api

# With monitoring stack
docker-compose --profile monitoring up -d
```

### 4. Run API Locally (without Docker)

```bash
pip install -r requirements.txt

# Start Neo4j separately (or use Docker for just Neo4j):
docker run -d -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5.13-community

# Start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### Health Check
```http
GET /health
```

### Friend Recommendations
```http
GET /recommend-friends/{user_id}?top_k=10
```
Returns: GNN-ranked friend recommendations with mutual connection counts and influence scores.

### Link Prediction
```http
POST /predict-links
{
  "user_id": "user_1",
  "pairs": [["user_1", "user_5"], ["user_1", "user_12"]]
}
```

### User Influence
```http
GET /user-influence/{user_id}
```
Returns: GNN-predicted role (influencer/regular/creator/hub), confidence, and graph stats.

### Trending Posts
```http
GET /trending-posts?top_k=10&topic=AI
```
Returns: Posts ranked by engagement velocity (likes + 2×comments / age).

### Explain Connection
```http
GET /explain-connection?user_a=user_1&user_b=user_5
```
Returns: Shortest path, common friends, common liked posts, LLM-generated explanation.

### Natural Language Query (GraphRAG)
```http
POST /query
{
  "query": "Who are the top influencers in the tech space?",
  "user_id": "user_1",
  "mode": "hybrid",
  "top_k": 10
}
```
Full multi-agent pipeline. Returns structured results + NL insight + validation report.

---

## Response Format

All endpoints return:

```json
{
  "intent": "friend_recommendation",
  "results": [
    {
      "id": "user_5",
      "name": "Alice",
      "mutual_friends": 4,
      "influence_score": 0.82,
      "gnn_score": 0.91,
      "fusion_score": 0.0312,
      "source": "hybrid"
    }
  ],
  "gnn_predictions": [...],
  "insight": "Based on 4 mutual connections and network centrality...",
  "graph_context": "Graph query 'friend_recommendation' returned 8 records.",
  "retrieval_mode": "hybrid",
  "sources": ["neo4j_graph", "vector_index"],
  "validation": {
    "is_valid": true,
    "confidence": 0.87,
    "warnings": [],
    "issues": []
  },
  "query": "Recommend friends for user_1",
  "pipeline_timing_ms": {
    "analyzer": 0.4,
    "router": 0.1,
    "retrieval": 12.3,
    "gnn_inference": 8.7,
    "synthesizer": 45.2,
    "validator": 0.9,
    "total": 67.6
  }
}
```

---

## Neo4j Schema

```
(:User {id, name, email, bio, follower_count, influence_score, embedding})
(:Post {id, title, content, topic, like_count, comment_count, created_at})
(:Comment {id, text, created_at})
(:Group {id, name, description})

(:User)-[:FRIEND]->(:User)
(:User)-[:POSTED]->(:Post)
(:User)-[:LIKED]->(:Post)
(:User)-[:COMMENTED]->(:Comment)
(:Comment)-[:ON]->(:Post)
(:User)-[:MEMBER_OF]->(:Group)
```

Indexes:
- `UNIQUE` constraints on all `id` properties
- Full-text index on `Post(content, title)`
- **Vector index** on `User(embedding)` — 128-dim cosine similarity

---

## Model Architecture

### SocialGraphGNN (Facebook, Reddit)
```
GraphSAGEEncoder
  └── 3× SAGEConv + BatchNorm + ReLU + Dropout
  └── Output: 128-dim node embeddings

LinkPredictor (MLP)
  └── Concatenate (z_u, z_v) → 2 FC layers → sigmoid

NodeClassifier (MLP)
  └── z → 2 FC layers → 4-class softmax
```

### GATSocialGNN (Twitter)
```
GATConv(in, 128, heads=4) → ELU
GATConv(128×4, 64, heads=1)

Same LinkPredictor + NodeClassifier heads
```

### Node Classes
| Class | Label | Description |
|---|---|---|
| 0 | regular_user | Low engagement, small network |
| 1 | influencer | High follower count, viral posts |
| 2 | content_creator | Frequent posting, moderate reach |
| 3 | community_hub | High connectivity, bridge nodes |

---

## Evaluation Metrics

| Task | Metric | Target |
|---|---|---|
| Link Prediction | AUC-ROC | > 0.85 |
| Node Classification | Macro F1 | > 0.75 |
| API Latency | P95 | < 200ms |
| Hallucination | Validator pass rate | > 95% |

---

## Running Tests

```bash
# Unit + integration tests
pytest tests/test_all.py -v

# With coverage
pytest tests/test_all.py -v --cov=. --cov-report=html

# Single test class
pytest tests/test_all.py::TestGNNModel -v
pytest tests/test_all.py::TestAPIEndpoints -v
```

---

## Key Design Decisions

| Concern | Decision | Rationale |
|---|---|---|
| Train/Inference separation | No FastAPI/Neo4j in training scripts | Kaggle compatibility, clean boundaries |
| GPU/CPU split | `CUDA` in training, hard `cpu` in inference | API must run on cheap VMs |
| Hybrid retrieval | Reciprocal Rank Fusion (RRF) | Robust, parameter-free fusion |
| Hallucination reduction | 6-step validator pipeline | Grounding + dedup + confidence checks |
| LLM integration | Optional (degrades gracefully) | Works without API key |
| Fallback data | Mock data when Neo4j unavailable | Development without infrastructure |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection string |
| `NEO4J_PASSWORD` | `password123` | Neo4j auth |
| `ANTHROPIC_API_KEY` | — | LLM for NL insights (optional) |
| `USE_LLM` | `true` | Enable/disable LLM generation |
| `LLM_MODEL` | `claude-3-haiku-20240307` | LLM model for KAG |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `PORT` | `8000` | API port |
