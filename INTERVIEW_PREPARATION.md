# Interview preparation — GraphRAG Social Intelligence System

This document is organized **by topic**. For each topic you get a **concise deep dive** (what it is, why it matters), then **interview-style prompts** you can practice: *What is this?*, *What does it do?*, *Why this?*, *How does it happen?*, *Project relevance*, *Inputs / outputs / where*.

Use it alongside [`README.md`](README.md) (overview, API) and [`ARCHITECTURE.md`](ARCHITECTURE.md) (Neo4j-only design and performance).

---

## 1. Social graph intelligence (problem domain)

### Topic in depth

A **social graph** models people, content, and interactions as **nodes** and **relationships** (friendships, likes, posts, groups). “Social intelligence” here means answering questions that need both **structure** (who is connected to whom, paths, communities) and **semantics** (what posts or bios are “about” a topic). Pure keyword search misses graph structure; pure graph algorithms miss text meaning. This project combines **GNNs** (learned representations from topology), **retrieval** (Cypher + vectors), and **LLM synthesis** (natural language grounded in retrieved facts).

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What is social graph intelligence in this project? | Using Neo4j to store Users, Posts, Comments, Groups and relationships (e.g. `FRIEND`, `LIKED`, `POSTED`), then running recommendations, influence analysis, trending content, and NL Q&A over that graph. |
| **What does it do?** | What business questions does the system address? | Friend recommendations, link prediction, influencer/trending analysis, explaining how two users are connected, and open-ended queries via `/query` or chat-style endpoints. |
| **Why this?** | Why use a graph instead of only a document store? | Many questions are inherently relational (mutual friends, shortest path, multi-hop reach). A graph database expresses those queries naturally and efficiently; documents would require expensive joins or denormalization. |
| **How is this happening?** | How is the graph stored and queried? | **Neo4j** holds the graph; **Cypher** traverses and aggregates; optional **vector indexes** on node properties support semantic similarity in the same database. |
| **Project relevance** | Where does this show up in your architecture? | Every retrieval path ultimately reads from Neo4j; the multi-agent pipeline’s retriever step is built around graph + vector access patterns defined for each intent. |
| **Input / output / where** | What goes in and what comes out at the “graph” layer? | **Input:** user id, query text, `top_k`, sometimes `mode` (graph / vector / hybrid). **Output:** node/relationship records, scores, and text context passed to synthesis. **Where:** `db/neo4j_client.py`, Cypher in schema/query modules, `api/services/graph_service.py`, `rag/hybrid_retrieval.py`. |

---

## 2. GraphRAG (retrieval-augmented generation on graphs)

### Topic in depth

**GraphRAG** here means: for a user question, **retrieve** relevant subgraphs, paths, or semantically similar nodes/posts from the graph (and vector indexes), then **augment** an LLM prompt with that grounded context so answers cite **actual graph evidence** instead of hallucinating. It is **not** “only vector RAG over chunks”; graph structure is first-class (e.g. friend-of-friend constraints, shortest paths, engagement-weighted trending).

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What is GraphRAG? | RAG where the **retrieval source** is a knowledge graph (or property graph): structured retrieval + optional vector similarity, then generation conditioned on retrieved facts. |
| **What does it do?** | What does GraphRAG do in this codebase? | The pipeline retrieves via `HybridRetriever` (graph / vector / hybrid Cypher), then the synthesizer builds an LLM prompt from retrieval + GNN signals. |
| **Why this?** | Why GraphRAG vs plain RAG over text chunks? | Social questions mix **topology** (paths, degrees, communities) with **text** (bios, posts). Graph retrieval encodes relationships explicitly; chunk-only RAG often loses that structure. |
| **How is this happening?** | Walk through the retrieval → generation path. | **Analyzer** infers intent → **Router** picks query type and mode → **Retrievers** run Neo4j (and vector procedures where needed) → **GNN inference** enriches candidates → **Synthesizer** (Gemini + LangChain) produces structured + NL output → **Validator** checks grounding and confidence. |
| **Project relevance** | Which endpoint exercises full GraphRAG? | `POST /query` and chat flows that call `MultiAgentPipeline.run()` — full multi-agent orchestration with timing breakdowns. |
| **Input / output / where** | What are the inputs and outputs of GraphRAG here? | **Input:** natural language `query`, optional `user_id`, `context` (e.g. dataset, `mode`). **Output:** structured results, `insight`, `graph_context`, `validation`, `pipeline_timing_ms`. **Where:** `api/services/pipeline.py`, `api/agents/*.py`, `rag/hybrid_retrieval.py`. |

---

## 3. Multi-agent pipeline (orchestration)

### Topic in depth

Instead of one monolithic function, the system splits work into **agents** with clear roles: analyze → route → retrieve → synthesize → validate. Benefits: **testability**, **explicit intent handling**, and **different retrieval strategies per intent** without entangling everything in one prompt.

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What is the multi-agent pipeline? | A staged orchestrator: `QueryAnalyzerAgent` → `RouterAgent` → `RetrieversAgent` → `SynthesizerAgent` → `ValidatorAgent`. |
| **What does it do?** | What does each stage own? | Analyzer: intent + entities + strategy. Router: maps intent to Neo4j query type and `RetrievalMode` (GRAPH / VECTOR / HYBRID). Retrievers: executes hybrid retrieval. Synthesizer: KAG-style merge of GNN + context + LLM. Validator: grounding, dedup, confidence. |
| **Why this?** | Why not a single LLM call end-to-end? | A single call would mix planning, retrieval, and generation; harder to enforce **deterministic retrieval**, **audit trails**, and **latency control**. Stages allow caching, fallbacks, and metrics per step. |
| **How is this happening?** | How does routing interact with the API? | `MultiAgentPipeline.run()` analyzes the query, then `RouterAgent.route()` chooses strategy; client can override with `context.mode` (`graph` / `vector` / `hybrid`). GNN dataset can be resolved from context (`facebook`, `twitter`, `reddit`). |
| **Project relevance** | Where is the pipeline wired? | Instantiated in app lifespan in `api/main.py`, passed into services that handle `/query` and chat. |
| **Input / output / where** | Inputs/outputs? | **Input:** `query`, `context`, `top_k`. **Output:** dict with results, insights, validation, timings. **Where:** `api/services/pipeline.py`. |

---

## 4. Query analyzer and router (intent → strategy)

### Topic in depth

The **analyzer** turns raw text into an `AnalyzedQuery`: intent enum, entities (e.g. user ids), and retrieval hints. The **router** maps each `QueryIntent` to a **concrete Neo4j query type** and a default **retrieval mode** (e.g. friend recommendations → hybrid native Cypher; explain connection → graph-only, mode often locked).

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What is the router’s job? | Decide *how* to retrieve: pure graph traversal, vector ANN on Neo4j indexes, or **hybrid** (combined in one Cypher for hot paths). |
| **What does it do?** | Give examples of intent → query_type. | Friend recommendation → `friend_recommendation` + HYBRID. Influencer → `influence_stats` + GRAPH. Trending → `trending_posts` + HYBRID. Explain connection → `explain_connection` + GRAPH (locked). Link prediction → `link_candidates` + HYBRID. Content search → `fulltext_search` / vector-oriented paths + VECTOR. |
| **Why this?** | Why different modes per intent? | Some tasks are purely structural (shortest path). Others need semantic similarity (posts/users). Hybrid reduces round-trips when graph filters and vector scores belong in one query. |
| **How is this happening?** | Where is the routing table defined? | `RouterAgent.INTENT_ROUTING` in `api/agents/router.py` documents Neo4j capabilities per intent. |
| **Project relevance** | What breaks if routing is wrong? | Wrong mode → missed semantic matches (graph-only) or noisy results without structural constraints (vector-only where graph matters). |
| **Input / output / where** | I/O? | **Input:** `AnalyzedQuery`, optional hints. **Output:** `query_type`, params, `RetrievalMode`. **Where:** `api/agents/analyzer.py`, `api/agents/router.py`. |

---

## 5. Neo4j-only backend (graph + native vector indexes)

### Topic in depth

The system was refactored from **Neo4j + FAISS + Python RRF** to **Neo4j only**: embeddings live as **node properties**, and Neo4j **vector indexes** answer approximate nearest-neighbor queries. **Hybrid** workloads often use **one Cypher** that applies graph patterns and vector scoring together, avoiding two round-trips and Python-side fusion for those paths.

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What does “Neo4j-only” mean here? | All retrieval — structural and vector — goes through Neo4j; no separate in-process FAISS index for serving. |
| **What does it do?** | What vector indexes exist? | Typically separate indexes per label/property (e.g. user text embeddings 384-d, user GNN embeddings 128-d, post text embeddings 384-d) with cosine similarity — dimensions must match per index. |
| **Why this?** | Why drop FAISS for this use case? | **Operational simplicity** (one system), **data consistency** (no sync drift), **native hybrid Cypher** (filter + vector in one query). Pure ANN may be slightly slower than in-process FAISS, but hybrid latency and cold start often improve overall. |
| **How is this happening?** | How are embeddings written? | Batched `UNWIND` updates setting `text_embedding` / `gnn_embedding` on existing nodes; incremental population skips nodes that already have embeddings unless `force_refresh`. |
| **Project relevance** | Where is this documented? | [`ARCHITECTURE.md`](ARCHITECTURE.md) — design decisions, batch sizes, trade-off tables, tuning notes. |
| **Input / output / where** | I/O for vector layer? | **Input:** embedding vectors, index names, `top_k`. **Output:** nodes + scores from `db.index.vector.queryNodes` or hybrid Cypher. **Where:** `rag/neo4j_vector_store.py`, schema Cypher files, startup in `api/main.py`. |

---

## 6. Hybrid retrieval and Reciprocal Rank Fusion (RRF)

### Topic in depth

**Hybrid retrieval** combines rankings from **graph-based** retrieval (e.g. multi-hop candidates, mutual friends) and **vector similarity** (similar bios/posts). When two lists must be merged from **separate** queries, **RRF** fuses ranked lists with a simple, robust score that does not require calibrated probabilities. When possible, this project pushes fusion **into a single Cypher** instead of Python RRF for hot intents.

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What is hybrid retrieval? | Using both structural graph signals and embedding similarity to produce a single ranked candidate set. |
| **What does it do?** | What does RRF do? | Merges multiple ranked lists by boosting items that appear high in *any* list, reducing dependence on raw score scales. |
| **Why this?** | Why RRF instead of weighted linear blend? | Graph scores and cosine scores are on different scales; RRF is **parameter-light** and often robust without tuning. |
| **How is this happening?** | When is fusion in Cypher vs Python? | **Cypher-native hybrid** for e.g. friend recommendation and trending posts (one round-trip). **Python RRF** kept where two genuinely different Cypher patterns must run separately (e.g. some link-candidate flows). |
| **Project relevance** | Why does this matter for product quality? | Captures “friends of friends who are semantically similar” better than either modality alone. |
| **Input / output / where** | I/O? | **Input:** query embedding, user id, `top_k`, mode. **Output:** fused ranked records + provenance (`sources`). **Where:** `rag/hybrid_retrieval.py`, Neo4j query modules. |

---

## 7. GNNs (GraphSAGE / GAT), link prediction, node classification

### Topic in depth

**GNNs** learn **node embeddings** by message passing over edges: each node aggregates neighbor features to produce a vector useful for downstream tasks. This project trains **SocialGraphGNN** (GraphSAGE-style encoder) for Facebook/Reddit and **GATSocialGNN** (attention) for Twitter. **Link prediction** scores whether an edge should exist (e.g. potential friendship). **Node classification** assigns roles (e.g. influencer vs regular). **Inference** runs on **CPU** in the API for cheap deployment; training can use GPU elsewhere (e.g. Kaggle).

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What is a GNN doing here? | Encoding graph structure into vectors that improve ranking and understanding of users (link likelihood, role). |
| **What does it do?** | Link prediction vs node classification? | Link prediction: score pairs. Node classification: label single nodes (e.g. hub/influencer). Both use shared encoder embeddings. |
| **Why this?** | Why GNN vs classical features only? | Classical features (degree, mutual friends) are useful but **limited**; GNNs can combine multi-hop patterns and feature interactions non-linearly. |
| **How is this happening?** | How does training differ from inference? | Training scripts in `training/` produce `weights/*.pth` (and possibly numpy embeddings). `model/inference.py` loads weights and runs forward passes inside the pipeline after retrieval. |
| **Project relevance** | Where do GNN outputs feed the rest of the system? | Into the **synthesizer** as “knowledge” alongside retrieved graph rows — **KAG**: knowledge-augmented generation. |
| **Input / output / where** | I/O? | **Input:** node ids, candidate pairs, selected dataset weights. **Output:** scores, predicted classes, embeddings used for explanation/ranking. **Where:** `model/gnn_model.py`, `model/inference.py`, called from `MultiAgentPipeline`. |

---

## 8. KAG and the synthesizer (Gemini + LangChain)

### Topic in depth

**KAG (Knowledge-Augmented Generation)** here means: the LLM is given **explicit structured knowledge** — retrieved graph facts and GNN outputs — before generation. The **Synthesizer** uses **Google Gemini** via **LangChain** (`ChatGoogleGenerativeAI`), with **Pydantic output parsers** for structured insight types (e.g. influencer vs connection explanation schemas).

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What is KAG in one line? | Conditioning generation on **external knowledge** (graph + model predictions), not just the user prompt. |
| **What does it do?** | What does the synthesizer output? | Structured objects (e.g. summary, key findings, confidence) and natural language insight aligned to intent-specific schemas. |
| **Why this?** | Why structured outputs with Pydantic? | Easier validation, UI rendering, and **grounding checks** than free-form prose alone. |
| **How is this happening?** | Which LLM and how is it configured? | Gemini (default model from `GEMINI_MODEL` env); `USE_LLM` can disable LLM path for degraded operation. |
| **Project relevance** | How does this connect to the validator? | The validator checks whether claims align with retrieved evidence and assigns confidence / warnings — closing the loop on hallucination risk. |
| **Input / output / where** | I/O? | **Input:** analyzed query, retrieval context, GNN predictions. **Output:** structured insight + strings for the API response. **Where:** `api/agents/synthesizer.py`. |

---

## 9. Validator (grounding, quality, confidence)

### Topic in depth

The **validator** post-processes the synthesized response: deduplication, consistency checks, and a **confidence** assessment tied to evidence. This is the main **guardrail** layer when an LLM is involved — the system aims to flag weak grounding rather than silently inventing facts.

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | What is the validator? | The final agent that sanity-checks outputs against retrieval and internal consistency before returning JSON to the client. |
| **What does it do?** | What might it flag? | Low confidence, warnings, duplicate entities, mismatches between insight and retrieved ids — exact rules live in `api/agents/validator.py`. |
| **Why this?** | Why validate if we already use RAG? | RAG **reduces** hallucinations; it does not **eliminate** them. Validation adds an explicit **quality gate** for production APIs. |
| **How is this happening?** | When does validation run? | After synthesis, before the pipeline returns the combined payload (see `pipeline.py` flow). |
| **Project relevance** | Interview tie-in for “reliability”? | Mention metrics like validator pass rate / confidence reporting from README evaluation table as targets, not guarantees. |
| **Input / output / where** | I/O? | **Input:** synthesized response + context. **Output:** `validation` object in API JSON. **Where:** `api/agents/validator.py`. |

---

## 10. Embeddings (text vs GNN)

### Topic in depth

**Text embeddings** (e.g. from SentenceTransformers, often 384-d) capture semantic similarity of bios/posts. **GNN embeddings** (e.g. 128-d) capture **structural** role in the graph. They serve different retrieval and ranking roles; Neo4j stores both on `User` where applicable so **one node** can be filtered structurally and scored semantically or by learned structure.

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | Difference between text and GNN embeddings? | Text: language meaning. GNN: topology-informed representation from message passing. |
| **What does it do?** | How does the project use each? | Text: vector ANN on user/post text; GNN: link/node tasks and additional vector index for users. |
| **Why this?** | Why not only text embeddings? | Text misses **latent graph position** (e.g. bridge nodes); GNN misses **out-of-graph text** unless features include it. |
| **How is this happening?** | Where are they persisted? | As properties on Neo4j nodes, indexed by separate vector indexes where needed. |
| **Project relevance** | End-to-end story? | Ingest → embed → index → retrieve → GNN → synthesize → validate. |
| **Input / output / where** | I/O? | **Input:** raw text, graph features, model checkpoints. **Output:** float vectors on nodes; query vectors at request time. **Where:** `rag/embeddings.py`, `rag/neo4j_vector_store.py`, `get_text_engine()`. |

---

## 11. FastAPI service, lifecycle, datasets

### Topic in depth

**FastAPI** exposes REST endpoints for health, recommendations, analytics, **NL query**, **chat**, dataset status/ingest, and embedding maintenance (`/vector-indexes`, `/refresh-embeddings`). **Lifespan** startup connects Neo4j, ensures datasets, ingests if configured, creates vector indexes, populates embeddings, loads GNN models, and wires the pipeline and services.

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **What is this?** | How is the API structured? | Routers/services separate HTTP from agent logic; `main.py` owns lifespan and global `app_state`. |
| **What does it do?** | What happens on cold start? | Ordered bootstrap: DB → data presence → ingest/seed → vector schema → embeddings → GNN → pipeline (see comments in `api/main.py`). |
| **Why this?** | Why async lifespan + explicit ordering? | Failures at one stage can degrade gracefully (e.g. Neo4j down → warnings, mock/demo paths where implemented). |
| **How is this happening?** | How do datasets get into Neo4j? | Bootstrap modules download/ensure datasets and `ingest_all_if_needed` bulk-loads; flags like `AUTO_INGEST` / `FORCE_REINGEST` control behavior. |
| **Project relevance** | DevOps talking point? | Single deployable API; Neo4j can be Dockerized; optional monitoring stack under `docker/`. |
| **Input / output / where** | I/O? | **Input:** HTTP JSON/query params. **Output:** JSON with timings and validation. **Where:** `api/main.py`, `api/services/*`, `api/schemas.py`. |

---

## 12. System design trade-offs (interview favorite)

### Topic in depth

Be ready to compare: **latency vs operational cost**, **pure ANN vs hybrid**, **LLM optional degradation**, **CPU inference vs GPU training**, **single DB vs polyglot stores**. This project explicitly chose **Neo4j-only serving** to simplify ops and enable **Cypher-native hybrid**, accepting that raw vector QPS may be lower than in-process FAISS.

### Interview prompts

| Style | Sample question | Answer sketch |
|--------|-----------------|---------------|
| **Why this?** | Why not microservices per agent? | Simpler deployment and lower latency for now; agents are logical modules in one process. Could split if teams/scale demand. |
| **How is this happening?** | How would you scale reads? | Neo4j read replicas, caching query embeddings, tuning HNSW params, batching writes — see `ARCHITECTURE.md`. |
| **Project relevance** | One sentence summary for “design decisions”? | Graph-native retrieval + native vectors + multi-agent KAG + CPU GNN inference for a unified social intelligence API. |

---

## How to practice

1. Read one topic’s **deep dive** aloud.
2. Close the doc and answer each **interview prompt** without looking — especially **Why this?** and **How is this happening?**
3. Sketch the **data flow** from HTTP request to Neo4j to GNN to Gemini to validator on a whiteboard.
4. Prepare **two failure stories**: Neo4j unavailable; LLM unavailable — what degrades and what still works?

Good luck.
