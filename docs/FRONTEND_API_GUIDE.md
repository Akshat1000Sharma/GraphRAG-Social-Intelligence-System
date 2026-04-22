# Frontend integration guide — Social Graph Intelligence API

This document describes the **FastAPI** backend (`api/main.py`) so a frontend (React, Vue, mobile, etc.) can call it safely and predictably. Default server: `http://localhost:8000` (override with `PORT` in the environment).

**OpenAPI / Swagger UI:** `GET http://localhost:8000/docs` (interactive try-it-out UI).  
**ReDoc:** `GET http://localhost:8000/redoc`.

**CORS:** The API allows all origins (`*`) for development. Tighten this in production.

---

## Environment the backend expects

| Variable | Purpose |
|----------|---------|
| `NEO4J_URI` | e.g. `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `NEO4J_DATABASE` | Optional, default `neo4j` |
| `PORT` | API port, default `8000` |
| `ALLOW_CHAT_INSERT` | If `true`, enables writes from `/chat/insert` and related insert routes. Default is often `false` in production — check `api/bootstrap/config.py`. |
| `CHAT_INSERT_MAX_NODES` | Max operations per NL insert |

If Neo4j is unavailable, several routes return **503** with a short explanation. Some code paths still return **mock data** (see `GraphQueryService`) so the UI may get sample rows without a live graph.

---

## Conventions

### `dataset` parameter

Many pipeline-style endpoints accept an optional query parameter `dataset`:

- Allowed values: `facebook`, `twitter`, `reddit`, `demo`, or `all` (default).
- This scopes **retrieval** for the multi-agent pipeline to the correct subgraph when data was ingested with a `dataset` property on nodes.

**User identifiers:** Users can be looked up by canonical `id` (e.g. `user_1` from demo seed) or by `source_id` (used for chat-created users, e.g. `chat_alice`). The backend’s Cypher uses `id` **or** `source_id` where relevant.

### Pipeline vs direct graph endpoints

| Style | Prefix / routes | Behavior |
|-------|------------------|----------|
| **Multi-agent pipeline** | `/recommend-friends/*`, `/predict-links`, `/user-influence/*`, `/trending-posts`, `/explain-connection`, `/query`, `/chat` | Runs analyzers, retrievers (graph + vector), optional GNN, synthesizer, validator. Response includes `insight`, `validation`, timings, etc. |
| **Direct Neo4j (structured rows)** | `/graph/*` | No LLM narrative; returns raw lists/dicts from `GraphQueryService`. Good for tables, charts, and deterministic UI. |

Use **both** as needed: pipeline for explanations; `/graph/*` for fast tabular data.

---

## System

### `GET /health`

**Response (example):**

```json
{
  "status": "healthy",
  "neo4j_connected": true,
  "vector_backend": "neo4j",
  "vector_indexes": [{ "name": "user_embeddings", "state": "ONLINE" }],
  "gnn_loaded": true,
  "gnn_datasets": ["facebook"],
  "pipeline_ready": true,
  "version": "3.0.0",
  "dataset_counts": {
    "facebook": { "users": 100, "posts": 42 },
    "demo": { "users": 20, "posts": 40 }
  },
  "ingest_results": { "facebook": "ok" }
}
```

Use this for status dashboards and “can I run graph features?” checks.

### `GET /vector-indexes`

Returns Neo4j vector index metadata (admin/diagnostics).

### `POST /refresh-embeddings?force=false`

Rebuilds text embeddings in Neo4j. Can be slow; use for admin tools.

**Response (example):**

```json
{ "status": "ok", "counts": { "User": 120, "Post": 400 } }
```

### `GET /gnn/status`

Lists which GNN engines are loaded and which datasets exist in config.

**Response (example):**

```json
{
  "loaded_datasets": ["facebook"],
  "all_configured_datasets": ["facebook", "twitter", "reddit"],
  "load_state": { "facebook": true }
}
```

---

## Datasets (admin)

### `GET /datasets/status`

Per-dataset files on disk, ingest markers, and Neo4j counts.

### `POST /datasets/ingest?dataset=&force=false`

- No `dataset` query param: ingest all configured datasets (as implemented in `ingest_all_if_needed`).
- `dataset=facebook` (etc.): ingest one dataset.
- `force=true`: re-ingest even if a marker already exists (when supported by ingest logic).

**Response (shape):** `{ "triggered": ["facebook"], "results": { ... } }`

---

## Chat (NL Q&A over the graph)

### `POST /chat`

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | required | User question (1–2000 chars) |
| `dataset` | string | `all` | `facebook` \| `twitter` \| `reddit` \| `demo` \| `all` |
| `mode` | string | `hybrid` | `hybrid` \| `graph` \| `vector` — retrieval mode |
| `top_k` | int | 10 | 1–50 |
| `session_id` | string | null | Optional client id to track history server-side |
| `user_id` | string | null | Optional focus user for the question |

**Example:**

```http
POST /chat
Content-Type: application/json

{
  "message": "Who are the most active users in the facebook dataset?",
  "dataset": "facebook",
  "mode": "hybrid",
  "top_k": 10
}
```

**Response (example):**

```json
{
  "message": "Who are the most active users in the facebook dataset?",
  "dataset_queried": "facebook",
  "mode": "hybrid",
  "intent": "trending_or_influence",
  "results": [{ "id": "user_1", "name": "User 1", "dataset": "facebook" }],
  "insight": "Top accounts by post volume include …",
  "datasets_cited": ["facebook"],
  "graph_context_summary": "…",
  "pipeline_timing_ms": { "analyzer": 1.2, "router": 0.4, "retrieval": 15.0, "total": 45.0 },
  "session_id": null
}
```

---

## Graph mutations (insert)

Chat-driven inserts are **gated** by `ALLOW_CHAT_INSERT`. If disabled, NL execute returns an error.

### `POST /chat/insert`

Natural-language insert. **`confirm: false` (default)** = preview only; **`true`** = execute.

**Request:**

```json
{
  "nl_command": "Add user Alice who is friends with Bob in the facebook dataset",
  "dataset": "facebook",
  "confirm": true
}
```

**Response (success execute, example):**

```json
{
  "ok": true,
  "operation": "inserted",
  "nodes_created": 2,
  "edges_created": 1,
  "cypher_summary": "…",
  "detail": { "operations": [] },
  "error": null,
  "requires_confirm": false,
  "preview": null
}
```

### `POST /chat/insert/parse`

Parses the same NL as `/chat/insert` into **structured operations** without writing to the database. Use for a two-step “preview → confirm” UI.

**Request:**

```json
{
  "nl_command": "Add user Alice who is friends with Bob in the facebook dataset",
  "dataset": null
}
```

**Response (example):**

```json
{
  "ok": true,
  "dataset": "facebook",
  "operations": [
    {
      "type": "insert_user",
      "payload": {
        "dataset": "facebook",
        "name": "Alice",
        "bio": "User added via chat: Alice",
        "source_id": "chat_alice",
        "follower_count": 0,
        "influence_score": 0.3
      }
    }
  ],
  "parsed_names": ["Alice", "Bob"],
  "error": null
}
```

### `POST /chat/insert/user?confirm=false`

Structured user create/update (MERGE by `dataset` + `source_id`).

**Body (JSON):**

```json
{
  "dataset": "facebook",
  "name": "Carol",
  "bio": "Data scientist",
  "source_id": "carol_01",
  "follower_count": 100,
  "influence_score": 0.4
}
```

- `confirm=false` (default): returns `operation: "preview"`, `requires_confirm: true`.
- `confirm=true`: runs the Cypher write.

The canonical user `id` is derived as `"{dataset[:2]}_{source_id}"` (e.g. `fb_carol_01` for `facebook`).

### `POST /chat/insert/edge?confirm=false`

**Body (JSON):**

```json
{
  "dataset": "facebook",
  "from_user_id": "carol_01",
  "to_user_id": "chat_alice",
  "rel_type": "FRIEND",
  "bidirectional": true
}
```

`from_user_id` / `to_user_id` are **source_id** values in that dataset, not necessarily the public `u.id` field.

### `POST /chat/insert/post?confirm=false`

Creates a `Post` and a `(:User)-[:POSTED]->(:Post)` relationship. The **author** must already exist for that `dataset` + `author_source_id`.

**Body (JSON):**

```json
{
  "dataset": "facebook",
  "author_source_id": "carol_01",
  "title": "Hello from the app",
  "content": "Post body text",
  "topic": "tech",
  "source_id": "my_post_1"
}
```

- Omit `source_id` to let the server generate one.
- If the author is missing, the API returns `ok: false` with a clear `error` message (no partial relationship).

---

## Multi-agent pipeline (GraphRAG-style) — `POST /query` and feature GET routes

### `POST /query`

**Body (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | **Required.** Natural language task |
| `top_k` | int | Default 10 |
| Any other fields | any | Passed through to pipeline `context` (e.g. `user_id`, `dataset`, `user_a` / `user_b`, `mode`) |

**Example:**

```json
{
  "query": "Summarize community structure around user_5",
  "dataset": "all",
  "user_id": "user_5",
  "top_k": 5
}
```

**Response shape (pipeline output):**

```json
{
  "intent": "free_form",
  "query": "Summarize community structure around user_5",
  "results": [],
  "gnn_predictions": [],
  "insight": "… natural language answer …",
  "graph_context": "…",
  "retrieval_mode": "hybrid",
  "sources": [],
  "validation": {
    "is_valid": true,
    "confidence": 0.85,
    "warnings": [],
    "issues": []
  },
  "pipeline_timing_ms": {
    "analyzer": 0.0,
    "router": 0.0,
    "retrieval": 0.0,
    "gnn_inference": 0.0,
    "synthesizer": 0.0,
    "validator": 0.0,
    "total": 0.0
  }
}
```

### `GET /recommend-friends/{user_id}`

- Query: `top_k` (1–50, default 10), `dataset` (optional, default `all`).

**Example:** `GET /recommend-friends/user_3?top_k=5&dataset=demo`

Uses the same pipeline as `/query` with a fixed natural-language task.

### `POST /predict-links`

**Body (example):**

```json
{ "user_id": "user_1", "dataset": "all", "top_k": 10 }
```

### `GET /user-influence/{user_id}`

- Query: `dataset` (default `all`).

### `GET /trending-posts`

- Query: `top_k` (1–50), `topic` (optional), `dataset` (default `all`).

### `GET /explain-connection`

- Query: `user_a`, `user_b` (required), `dataset` (default `all`).

**Example:** `GET /explain-connection?user_a=user_1&user_b=user_5&dataset=demo`

---

## Direct graph API (`/graph/*`) — tabular / deterministic

These endpoints return **JSON wrappers** around lists of Cypher results. They do **not** run the LLM synthesis step. When Neo4j is down, `GraphQueryService` may return **mock** rows (see `source: "mock"` in objects).

### `GET /graph/friend-recommendations/{user_id}`

- Query: `top_k` (1–100, default 10)

**Response (example):**

```json
{
  "user_id": "user_1",
  "top_k": 10,
  "recommendations": [
    {
      "id": "user_4",
      "name": "User 4",
      "mutual_friends": 3,
      "influence_score": 0.55,
      "follower_count": 1200
    }
  ]
}
```

### `GET /graph/trending-posts`

- Query: `top_k`, `topic` (optional), `hours_window` (default 48 hours — parameter is passed through; the underlying query may order by engagement).

**Response (example):**

```json
{
  "top_k": 10,
  "topic": "AI",
  "hours_window": 48,
  "posts": [
    {
      "id": "post_2",
      "title": "…",
      "content": "…",
      "likes": 100,
      "comments": 10,
      "topic": "AI",
      "created_at": "2025-01-01T12:00:00",
      "engagement": 120
    }
  ]
}
```

### `GET /graph/users/{user_id}/influence-stats`

Returns aggregate stats: posts count, friends, total likes, `gnn_score` (stored property), etc.

**Response (example):**

```json
{
  "id": "user_1",
  "name": "User 1",
  "bio": "…",
  "followers": 500,
  "gnn_score": 0.65,
  "posts": 3,
  "friends": 8,
  "total_likes": 1200,
  "avg_likes": 400.0
}
```

### `GET /graph/connection-path?user_a=&user_b=`

**Response (example):**

```json
{
  "shortest_path": {
    "node_names": ["User 1", "User 2", "User 3"],
    "rel_types": ["FRIEND", "FRIEND"],
    "hops": 2
  },
  "common_friends": [{ "id": "user_9", "name": "User 9" }],
  "common_liked_posts": [{ "id": "post_1", "title": "…", "topic": "AI" }]
}
```

If no path is found, `shortest_path` may be `null`.

### `GET /graph/link-prediction/candidates/{user_id}`

- Query: `top_k` (1–200)

**Response (example):**

```json
{
  "user_id": "user_1",
  "top_k": 20,
  "candidates": [
    { "id": "user_7", "name": "User 7", "graph_score": 4 }
  ]
}
```

`graph_score` is a path-count–style score from the graph, not a neural link probability (see GNN / pipeline for combined reasoning).

### `GET /graph/top-influencers`

- Query: `top_k` (1–200)

**Response (example):**

```json
{
  "top_k": 20,
  "influencers": [
    {
      "id": "user_2",
      "name": "User 2",
      "followers": 8000,
      "gnn_score": 0.9,
      "post_count": 5,
      "avg_likes": 300.0,
      "composite_score": 1203.4
    }
  ]
}
```

---

## Suggested UI flows

1. **Landing / health:** `GET /health` → show connection status, dataset sizes, GNN load state.
2. **Explore user:** `GET /graph/users/{id}/influence-stats` + `GET /recommend-friends/{id}` or `GET /graph/friend-recommendations/{id}`.
3. **Network explain:** `GET /graph/connection-path` for a side panel; optional narrative via `GET /explain-connection` or `/chat`.
4. **Global leaderboard:** `GET /graph/top-influencers`.
5. **Assisted content:** `POST /chat` for Q&A; `POST /query` for custom prompts.
6. **Safe writes:** `POST /chat/insert/parse` → user reviews operations → `POST /chat/insert` with `confirm: true` or call structured `insert/user`, `insert/edge`, `insert/post` with `confirm=true`.

---

## Error handling tips for the frontend

- **503:** Neo4j or service not ready — show a “backend unavailable” state; do not treat as user error.
- **400:** On `/query`, missing `query` field.
- **4xx/5xx from Pydantic:** Invalid enum (`dataset` / `mode` on `/chat`).

Display `insight` and `validation.warnings` from pipeline responses to power transparency UI (“how confident is this answer?”).

---

## Version

API `version` field in `/health` and OpenAPI title reflect **3.0.0**-series behavior at the time this document was written. If you upgrade the backend, re-check `/docs` for new fields and routes.
