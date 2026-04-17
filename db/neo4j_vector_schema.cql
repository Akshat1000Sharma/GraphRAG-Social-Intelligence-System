// ============================================================
// neo4j_vector_schema.cql
// Neo4j Vector Index Schema for Social Graph Intelligence
// Replaces all FAISS operations with native Neo4j vector search
// Neo4j 5.x+ required
// ============================================================


// ──────────────────────────────────────────────────────────────
// PART 1: VECTOR INDEX CREATION
// ──────────────────────────────────────────────────────────────
// All three indexes use cosine similarity (appropriate for L2-normalized
// SentenceTransformer and GNN embeddings).
// Run once on schema setup — idempotent (IF NOT EXISTS).


// Index 1: User text profile embeddings (384-dim, all-MiniLM-L6-v2)
CREATE VECTOR INDEX user_text_embeddings IF NOT EXISTS
FOR (u:User) ON (u.text_embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
};

// Index 2: User GNN structural embeddings (128-dim, GraphSAGE encoder)
CREATE VECTOR INDEX user_gnn_embeddings IF NOT EXISTS
FOR (u:User) ON (u.gnn_embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 128,
    `vector.similarity_function`: 'cosine'
  }
};

// Index 3: Post text embeddings (384-dim, all-MiniLM-L6-v2)
CREATE VECTOR INDEX post_text_embeddings IF NOT EXISTS
FOR (p:Post) ON (p.text_embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
};

// Verify indexes are ONLINE before querying
// SHOW VECTOR INDEXES YIELD name, state, populationPercent
// WHERE state = 'ONLINE';


// ──────────────────────────────────────────────────────────────
// PART 2: EMBEDDING STORAGE (write to node properties)
// ──────────────────────────────────────────────────────────────
// Pattern: UNWIND batch → MATCH node → SET property
// Uses UNWIND for batch efficiency (1 transaction per batch, not N).


// Store text embeddings on users (batch upsert)
UNWIND $batch AS item
MATCH (u:User {id: item.id})
SET u.text_embedding        = item.embedding,
    u.embedding_updated_at  = datetime();

// Store text embeddings on posts (batch upsert)
UNWIND $batch AS item
MATCH (p:Post {id: item.id})
SET p.text_embedding        = item.embedding,
    p.embedding_updated_at  = datetime();

// Store GNN structural embeddings on users
UNWIND $batch AS item
MATCH (u:User {id: item.id})
SET u.gnn_embedding         = item.embedding,
    u.embedding_updated_at  = datetime();


// ──────────────────────────────────────────────────────────────
// PART 3: PURE VECTOR SIMILARITY SEARCH
// ──────────────────────────────────────────────────────────────
// Replaces: FAISS.search(query_vector, top_k)
// Uses: db.index.vector.queryNodes procedure (Neo4j 5.x)


// 3a. Find similar users by text query embedding
CALL db.index.vector.queryNodes('user_text_embeddings', $top_k, $query_vector)
YIELD node AS u, score
RETURN u.id              AS id,
       u.name            AS name,
       u.bio             AS bio,
       u.influence_score AS influence_score,
       u.follower_count  AS follower_count,
       score             AS similarity_score,
       'neo4j_vector'    AS source
ORDER BY score DESC;


// 3b. Find similar users by GNN embedding (structural similarity)
CALL db.index.vector.queryNodes('user_gnn_embeddings', $top_k, $query_vector)
YIELD node AS u, score
RETURN u.id              AS id,
       u.name            AS name,
       u.bio             AS bio,
       u.influence_score AS influence_score,
       score             AS similarity_score,
       'neo4j_gnn_vector' AS source
ORDER BY score DESC;


// 3c. Find similar posts by text query embedding
CALL db.index.vector.queryNodes('post_text_embeddings', $top_k, $query_vector)
YIELD node AS p, score
OPTIONAL MATCH (author:User)-[:POSTED]->(p)
RETURN p.id           AS id,
       p.title        AS title,
       p.content      AS content,
       p.topic        AS topic,
       p.like_count   AS likes,
       p.created_at   AS created_at,
       author.id      AS author_id,
       author.name    AS author_name,
       score          AS similarity_score,
       'neo4j_vector' AS source
ORDER BY score DESC;


// 3d. Inline cosine similarity (no index — for single pair scoring)
// Useful for scoring specific known pairs without a full ANN search.
MATCH (u:User {id: $user_a}), (v:User {id: $user_b})
WHERE u.text_embedding IS NOT NULL AND v.text_embedding IS NOT NULL
RETURN vector.similarity.cosine(u.text_embedding, v.text_embedding) AS cosine_score;


// ──────────────────────────────────────────────────────────────
// PART 4: HYBRID QUERIES (graph constraints + vector similarity)
// ──────────────────────────────────────────────────────────────
// This is the KEY advantage over FAISS:
// Graph structural constraints are applied INSIDE the vector query.
// FAISS cannot filter by relationship patterns.


// 4a. Friend-of-friend recommendation with semantic ranking
//
// Combines:
//   - Graph: 2-hop FRIEND traversal (structural proximity)
//   - Vector: cosine similarity of text embeddings
//   - Fusion: weighted combined score in Cypher
//
MATCH (u:User {id: $user_id})-[:FRIEND]->(friend:User)-[:FRIEND]->(candidate:User)
WHERE candidate.id <> $user_id
  AND NOT (u)-[:FRIEND]->(candidate)
  AND candidate.text_embedding IS NOT NULL
WITH candidate, count(friend) AS mutual_count

// Score by vector similarity within the candidate set
WITH candidate, mutual_count,
     vector.similarity.cosine(candidate.text_embedding, $query_vector) AS vec_score

// Combine: 40% mutual friends (graph), 60% semantic similarity (vector)
WITH candidate, mutual_count, vec_score,
     (mutual_count * 0.4 + vec_score * 0.6) AS combined_score

ORDER BY combined_score DESC
LIMIT $top_k

RETURN candidate.id       AS id,
       candidate.name     AS name,
       candidate.bio      AS bio,
       candidate.influence_score AS influence_score,
       mutual_count,
       vec_score          AS similarity_score,
       combined_score     AS fusion_score,
       'neo4j_hybrid'     AS source;


// 4b. Influencer discovery by topic expertise
//
// Combines:
//   - Graph: :POSTED relationship + follower threshold
//   - Vector: post content similarity to topic query
//   - Fusion: best post score * topic weight + influence weight
//
CALL db.index.vector.queryNodes('post_text_embeddings', $top_k_posts, $query_vector)
YIELD node AS p, score AS post_similarity

MATCH (author:User)-[:POSTED]->(p)
WHERE author.follower_count >= $min_followers

WITH author,
     max(post_similarity)  AS best_post_score,
     count(p)              AS matching_post_count

ORDER BY (best_post_score * 0.6 + author.influence_score * 0.4) DESC
LIMIT $top_k

RETURN author.id              AS id,
       author.name            AS name,
       author.bio             AS bio,
       author.follower_count  AS follower_count,
       author.influence_score AS influence_score,
       best_post_score        AS similarity_score,
       matching_post_count,
       'neo4j_hybrid'         AS source;


// 4c. Trending posts by engagement velocity + semantic relevance
//
// Combines:
//   - Graph: engagement score (likes + comments, time-weighted)
//   - Vector: semantic similarity to query topic
//   - Fusion: 40% vector score + 60% engagement velocity
//
CALL db.index.vector.queryNodes('post_text_embeddings', $top_k_vec, $query_vector)
YIELD node AS p, score AS vec_score

WITH p, vec_score,
     p.like_count + (p.comment_count * 2.0) AS raw_engagement,
     duration.between(datetime(p.created_at), datetime()).hours AS age_hours

WITH p, vec_score,
     raw_engagement / (toFloat(age_hours) + 1.0) AS engagement_velocity

WITH p, vec_score, engagement_velocity,
     (vec_score * 0.4 + (engagement_velocity / 1000.0) * 0.6) AS combined_score

ORDER BY combined_score DESC
LIMIT $top_k

OPTIONAL MATCH (author:User)-[:POSTED]->(p)
RETURN p.id          AS id,
       p.title       AS title,
       p.content     AS content,
       p.topic       AS topic,
       p.like_count  AS likes,
       p.created_at  AS created_at,
       vec_score     AS similarity_score,
       engagement_velocity,
       combined_score AS fusion_score,
       author.name  AS author_name,
       'neo4j_hybrid' AS source;


// 4d. "People similar to X who are also in the same community"
//
// Example use-case from the refactoring spec:
// "Find people similar to X who also work in Y company"
//
// Combines:
//   - Graph: shared community membership (:MEMBER_OF)
//   - Vector: profile embedding similarity
//
MATCH (ref:User {id: $ref_user_id})-[:MEMBER_OF]->(group:Group)
MATCH (candidate:User)-[:MEMBER_OF]->(group)
WHERE candidate.id <> $ref_user_id
  AND candidate.text_embedding IS NOT NULL
WITH candidate,
     count(DISTINCT group) AS shared_groups,
     vector.similarity.cosine(candidate.text_embedding, $query_vector) AS vec_score
WHERE vec_score >= $min_similarity
ORDER BY (shared_groups * 0.3 + vec_score * 0.7) DESC
LIMIT $top_k
RETURN candidate.id   AS id,
       candidate.name AS name,
       shared_groups,
       vec_score      AS similarity_score,
       'neo4j_hybrid' AS source;


// ──────────────────────────────────────────────────────────────
// PART 5: MAINTENANCE QUERIES
// ──────────────────────────────────────────────────────────────


// Check index status
SHOW VECTOR INDEXES
YIELD name, state, populationPercent, labelsOrTypes, properties, indexConfig;

// Count nodes with/without embeddings
MATCH (u:User)
RETURN
  count(u)                                        AS total_users,
  count(u.text_embedding)                         AS users_with_text_emb,
  count(u.gnn_embedding)                          AS users_with_gnn_emb,
  count(u) - count(u.text_embedding)              AS users_missing_text_emb;

MATCH (p:Post)
RETURN
  count(p)                                        AS total_posts,
  count(p.text_embedding)                         AS posts_with_text_emb,
  count(p) - count(p.text_embedding)              AS posts_missing_text_emb;

// Re-index nodes missing embeddings (run via Python EmbeddingPopulator)
// MATCH (u:User) WHERE u.text_embedding IS NULL RETURN u.id LIMIT 100;

// Drop and recreate index (when changing embedding model/dimensions)
DROP INDEX user_text_embeddings IF EXISTS;
CREATE VECTOR INDEX user_text_embeddings ...;  -- see Part 1

// Warm up vector index for production (force population)
// CALL db.index.vector.queryNodes('user_text_embeddings', 1, $dummy_vector) YIELD node, score;
