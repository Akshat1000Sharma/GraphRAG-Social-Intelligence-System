# Neo4j-Only Architecture: Design Decisions & Performance Analysis

## Refactoring Summary

**Before:** Neo4j (graph) + FAISS (vector) + Python-side RRF fusion
**After:**  Neo4j only — graph traversal + native vector indexes + Cypher-side fusion

---

## Updated Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                  MULTI-AGENT PIPELINE                        │
│                                                             │
│  QueryAnalyzer → Router → Retrievers → Synthesizer → Validator
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              NEO4J UNIFIED BACKEND                           │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │  Graph Retrieval │    │    Vector Retrieval           │  │
│  │  (Cypher queries)│    │    (db.index.vector.*)        │  │
│  │                  │    │                              │  │
│  │  - FRIEND paths  │    │  user_text_embeddings (384d) │  │
│  │  - shortestPath  │    │  user_gnn_embeddings  (128d) │  │
│  │  - aggregations  │    │  post_text_embeddings (384d) │  │
│  └──────────────────┘    └──────────────────────────────┘  │
│            │                           │                    │
│            └───────────────────────────┘                    │
│                         │                                   │
│              ┌──────────▼──────────┐                        │
│              │   Hybrid Cypher     │                        │
│              │   (combined in      │                        │
│              │   single query)     │                        │
│              └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
GNN Inference → Synthesizer (Gemini KAG) → Validator → JSON
```

### What changed in each component

| Component | Old | New |
|-----------|-----|-----|
| `rag/vector_store.py` | FAISS + InMemoryVectorIndex | Compatibility shim → removed |
| `rag/neo4j_vector_store.py` | (new file) | TextEmbeddingEngine + Neo4jVectorSchemaManager + Neo4jEmbeddingPopulator + Neo4jVectorRetriever |
| `rag/embeddings.py` | Populated FAISS indexes | Writes embeddings to Neo4j node properties (UNWIND batch) |
| `rag/hybrid_retrieval.py` | VectorRetriever(FAISS) | VectorRetriever(Neo4jVectorRetriever) |
| `api/agents/router.py` | Routes to FAISS or Neo4j | Routes all to Neo4j (graph/vector/hybrid Cypher) |
| `api/main.py` | Initializes FAISS indexes | Creates Neo4j vector indexes, populates node embeddings |
| `requirements.txt` | faiss-cpu==1.7.4 | REMOVED |

---

## Design Decisions

### 1. Embeddings stored as node properties

**Decision:** Store `User.text_embedding`, `User.gnn_embedding`, `Post.text_embedding` directly on nodes.

**Why:**
- Zero join cost — embedding is co-located with node data
- Neo4j vector index indexes the property automatically
- GNN embeddings and text embeddings can coexist on the same node
- Easy to query both structural and semantic info in one traversal

**Alternative considered:** Separate embedding nodes linked by `:HAS_EMBEDDING`. Rejected because it adds a join and complicates queries with no benefit.

### 2. Three separate vector indexes

**Decision:** One index per (label, property) pair:
- `user_text_embeddings` on `User.text_embedding` (384d, cosine)
- `user_gnn_embeddings` on `User.gnn_embedding` (128d, cosine)
- `post_text_embeddings` on `Post.text_embedding` (384d, cosine)

**Why:**
- Smaller indexes = faster ANN search (IVF builds better clusters)
- Allows mixing text and GNN search independently
- Different dimensions require different indexes (Neo4j constraint)
- Cosine similarity chosen: embeddings are L2-normalized from SentenceTransformer, cosine = dot product at that point → equivalent to Euclidean on normalized vectors but more numerically stable

**Alternative considered:** Single combined index. Rejected because Neo4j requires uniform dimensions per index.

### 3. Cypher-native hybrid queries for hot paths

**Decision:** For `friend_recommendation` and `trending_posts`, use a single Cypher query that combines graph constraints AND vector scoring. Python-side RRF only for paths that require two separate queries.

**Why (performance):**
- 1 round-trip to Neo4j vs 2 (graph query + vector query) + Python fusion
- Neo4j can filter the vector result set using graph patterns before returning
- Eliminates Python-side list manipulation for the most common intents

**Trade-off:** More complex Cypher queries. Mitigated by having all queries in `neo4j_vector_schema.cql` and `neo4j_vector_store.py` with comments.

### 4. UNWIND batch writes for embeddings

**Decision:** Use `UNWIND $batch AS item MATCH ... SET` for embedding writes.

**Why:**
- 1 transaction per batch (50 nodes) vs N transactions
- MERGE-less MATCH + SET is safe since nodes already exist when embeddings are written
- Avoids Neo4j transaction timeout on large graphs

**Batch size:** 50 nodes. Tunable via `Neo4jEmbeddingPopulator.BATCH_SIZE`.

### 5. Incremental embedding population

**Decision:** `populate_all(force_refresh=False)` only embeds nodes where `text_embedding IS NULL`.

**Why:**
- Startup is fast — only new nodes get embedded
- `force_refresh=True` available for model changes
- `/refresh-embeddings?force=true` endpoint for ops use

### 6. Python RRF retained for cross-query fusion

**Decision:** Keep Python RRF for queries that run separate graph + vector queries.

**Why:**
- Some intents (e.g., `link_candidates` + semantic search) genuinely need two different Cypher patterns
- RRF is cheap (list sorting, no DB calls)
- Allows precise weight tuning per intent

---

## Performance Considerations

### Neo4j Vector Index Performance

| Metric | FAISS (old) | Neo4j Vector Index (new) |
|--------|-------------|--------------------------|
| Index type | HNSW or Flat | HNSW (approximate NN) |
| Similarity | Cosine (Flat IP) | Cosine (configurable) |
| Throughput | ~10k QPS (in-process) | ~500-2000 QPS (over network) |
| Latency | 1-5ms | 5-20ms (local) / 20-50ms (remote) |
| Memory | Separate process | Unified with graph |
| Persistence | Requires serialization | Native (survives restart) |
| Cold start | Must rebuild from Neo4j | Already there |
| Hybrid capability | None (Python only) | Native (Cypher + vector) |

**Key insight:** Neo4j vector search is ~3-5x slower than in-process FAISS for pure ANN queries. However, it eliminates the Python-side RRF step for hybrid queries, which saves 1 round-trip. For hybrid workloads (the common case here), the total latency difference is small (10-30ms).

### Optimization Strategies for Large Scale

#### Index-level optimizations
```cypher
-- Tune HNSW parameters for recall vs speed trade-off
-- (Set at index creation time, cannot change after)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine',
    `vector.hnsw.m`: 16,           -- connections per node (default 16)
    `vector.hnsw.ef_construction`: 100  -- build-time precision (default 100)
  }
}
-- Higher m + ef_construction = better recall but slower build + more memory
-- For production: m=32, ef_construction=200 for better recall at 10-30% perf cost
```

#### Query-level optimizations
1. **Limit candidate set early:** Use `WHERE ... AND candidate.text_embedding IS NOT NULL` before vector scoring to avoid null-check overhead
2. **Two-phase retrieval:** For large graphs, first filter by graph constraints (cheap, reduces candidate set), then apply vector scoring (expensive)
3. **Warm-up queries:** Hit the vector index once at startup so HNSW graph is loaded into Neo4j's page cache
4. **Read replicas:** Route vector queries to read replicas in Neo4j Cluster to separate from write traffic

#### Application-level optimizations
1. **Query embedding cache:** `TextEmbeddingEngine._cache` caches repeated identical queries (LRU, 1000 entries)
2. **Batch embedding writes:** 50-node batches in `UNWIND` transactions reduce round-trips by ~50x
3. **Async embedding population:** Run `populate_all()` in a background thread so startup is non-blocking
4. **GNN embedding freshness:** GNN embeddings only need refresh after model retraining (not every startup)

### Benchmarking Neo4j vs FAISS

To benchmark the replacement empirically:

```python
import time
import numpy as np

# FAISS baseline (if keeping for comparison)
def benchmark_faiss(query_vec, top_k=10, n_trials=100):
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        faiss_index.search(query_vec, top_k)  # your old FAISS index
        times.append(time.perf_counter() - t0)
    return {"mean_ms": np.mean(times) * 1000, "p99_ms": np.percentile(times, 99) * 1000}

# Neo4j vector search
def benchmark_neo4j_vector(neo4j_client, query_vec, top_k=10, n_trials=100):
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        neo4j_client.run_query(
            "CALL db.index.vector.queryNodes('user_text_embeddings', $k, $vec) YIELD node, score",
            {"k": top_k, "vec": query_vec.tolist()}
        )
        times.append(time.perf_counter() - t0)
    return {"mean_ms": np.mean(times) * 1000, "p99_ms": np.percentile(times, 99) * 1000}

# Neo4j hybrid query (no FAISS equivalent)
def benchmark_neo4j_hybrid(neo4j_client, user_id, query_vec, top_k=10, n_trials=100):
    # Uses friend-of-friend hybrid Cypher — impossible with FAISS alone
    ...
```

**Expected results:**
- Pure vector search: FAISS wins by ~5-10ms (in-process)
- Hybrid search: Neo4j wins by ~20-40ms (eliminates Python RRF + second query)
- Cold start: Neo4j wins significantly (no rebuild needed)
- Operational complexity: Neo4j wins (one system to maintain)

### Trade-offs Summary

| Concern | FAISS | Neo4j Vector Index |
|---------|-------|-------------------|
| Pure ANN latency | Better (~5ms) | Acceptable (~15ms) |
| Hybrid query latency | Poor (2 queries + Python) | Better (1 query) |
| Operational complexity | High (2 systems) | Low (1 system) |
| Data consistency | Manual sync risk | Always consistent |
| Cold start | Slow (rebuild from DB) | None (persisted) |
| Scalability | Single process | Neo4j Cluster |
| Hybrid capabilities | None | Rich (Cypher + vector) |
| Dimensionality | Any | Up to 4096 (v5.13+) |
| Update cost | O(n) rebuild or incremental | UNWIND batch SET |

**Conclusion:** For this use case (social graph intelligence with frequent hybrid queries), Neo4j-only is the right choice. The ~10ms latency penalty for pure vector queries is more than offset by the elimination of the Python-side fusion step for hybrid queries and the massive reduction in operational complexity.

---

## Migration Checklist

- [x] Remove `faiss-cpu` from `requirements.txt`
- [x] Remove `FAISSVectorIndex` class
- [x] Remove `InMemoryVectorIndex` class
- [x] Remove `build_vector_index()` function
- [x] Remove `get_user_index()`, `get_post_index()` singletons
- [x] Create `Neo4jVectorSchemaManager` (index creation)
- [x] Create `Neo4jEmbeddingPopulator` (UNWIND batch writes)
- [x] Create `Neo4jVectorRetriever` (vector search via Cypher)
- [x] Refactor `VectorRetriever` to wrap `Neo4jVectorRetriever`
- [x] Add 3 hybrid Cypher queries (friends, influencers, trending)
- [x] Update `HybridRetriever` to use Neo4j-native hybrid paths
- [x] Update `RouterAgent` to remove FAISS routing references
- [x] Update `main.py` startup sequence (no FAISS init)
- [x] Add `/vector-indexes` and `/refresh-embeddings` endpoints
- [x] Update `cypher_queries.cql` → `neo4j_vector_schema.cql`
- [x] Update `requirements.txt`
- [x] Replace `rag/embeddings.py` to delegate to `Neo4jEmbeddingPopulator`
- [x] Add compatibility shim in `rag/vector_store.py`
