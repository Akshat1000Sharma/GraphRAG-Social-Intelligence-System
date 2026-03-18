// ============================================================
// cypher_queries.cql
// Neo4j Cypher Query Library for Social Graph Intelligence
// ============================================================


// ──────────────────────────────────────────────────────────────
// SCHEMA SETUP
// ──────────────────────────────────────────────────────────────

CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT post_id IF NOT EXISTS
FOR (p:Post) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT comment_id IF NOT EXISTS
FOR (c:Comment) REQUIRE c.id IS UNIQUE;

CREATE INDEX user_name IF NOT EXISTS FOR (u:User) ON (u.name);
CREATE INDEX post_created IF NOT EXISTS FOR (p:Post) ON (p.created_at);
CREATE INDEX post_likes IF NOT EXISTS FOR (p:Post) ON (p.like_count);

CREATE FULLTEXT INDEX post_content IF NOT EXISTS
FOR (p:Post) ON EACH [p.content, p.title];

CREATE VECTOR INDEX user_embeddings IF NOT EXISTS
FOR (u:User) ON (u.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 128,
    `vector.similarity_function`: 'cosine'
  }
};


// ──────────────────────────────────────────────────────────────
// FRIEND RECOMMENDATION
// ──────────────────────────────────────────────────────────────

// Friends-of-friends recommendation (2-hop paths)
MATCH (u:User {id: $user_id})-[:FRIEND]->(friend)-[:FRIEND]->(fof:User)
WHERE fof.id <> $user_id
  AND NOT (u)-[:FRIEND]->(fof)
WITH fof, count(friend) AS mutual_count
ORDER BY mutual_count DESC
LIMIT $top_k
RETURN fof.id AS recommended_id,
       fof.name AS name,
       mutual_count AS mutual_friends,
       fof.influence_score AS influence_score;


// Get user's direct friends
MATCH (u:User {id: $user_id})-[:FRIEND]->(f:User)
RETURN f.id AS friend_id, f.name AS name, f.influence_score AS influence;


// ──────────────────────────────────────────────────────────────
// INFLUENCER DETECTION
// ──────────────────────────────────────────────────────────────

// Top influencers by follower + post engagement
MATCH (u:User)
OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
WITH u, count(p) AS post_count, coalesce(avg(p.like_count), 0) AS avg_likes
RETURN u.id AS user_id,
       u.name AS name,
       u.follower_count AS followers,
       post_count,
       avg_likes,
       u.influence_score AS gnn_score,
       (u.follower_count * 0.4 + avg_likes * 0.4 + post_count * 0.2) AS composite_score
ORDER BY composite_score DESC
LIMIT $top_k;


// User influence details
MATCH (u:User {id: $user_id})
OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
OPTIONAL MATCH (u)-[:FRIEND]-(f:User)
WITH u,
     count(DISTINCT p) AS post_count,
     count(DISTINCT f) AS friend_count,
     coalesce(sum(p.like_count), 0) AS total_likes
RETURN u.id AS user_id,
       u.name AS name,
       u.follower_count AS followers,
       u.influence_score AS gnn_influence_score,
       post_count,
       friend_count,
       total_likes,
       u.bio AS bio;


// ──────────────────────────────────────────────────────────────
// TRENDING POSTS
// ──────────────────────────────────────────────────────────────

// Trending posts (recent + high engagement)
MATCH (p:Post)
WHERE p.created_at > $since
WITH p,
     p.like_count + (p.comment_count * 2) AS engagement_score,
     duration.between(datetime(p.created_at), datetime()).hours AS age_hours
WHERE age_hours <= 48
RETURN p.id AS post_id,
       p.title AS title,
       p.content AS content,
       p.like_count AS likes,
       p.comment_count AS comments,
       p.topic AS topic,
       p.created_at AS created_at,
       engagement_score,
       age_hours
ORDER BY engagement_score / (age_hours + 1) DESC
LIMIT $top_k;


// Posts by topic
MATCH (p:Post {topic: $topic})
RETURN p.id AS post_id, p.title AS title, p.like_count AS likes, p.created_at AS created_at
ORDER BY p.like_count DESC
LIMIT $top_k;


// ──────────────────────────────────────────────────────────────
// CONNECTION EXPLANATION
// ──────────────────────────────────────────────────────────────

// Shortest path between two users
MATCH path = shortestPath(
  (a:User {id: $user_a})-[*..6]-(b:User {id: $user_b})
)
RETURN path,
       [n IN nodes(path) | n.name] AS node_names,
       [r IN relationships(path) | type(r)] AS rel_types,
       length(path) AS hops;


// Common friends between two users
MATCH (a:User {id: $user_a})-[:FRIEND]->(common)<-[:FRIEND]-(b:User {id: $user_b})
RETURN common.id AS friend_id, common.name AS friend_name;


// Common liked posts
MATCH (a:User {id: $user_a})-[:LIKED]->(p:Post)<-[:LIKED]-(b:User {id: $user_b})
RETURN p.id AS post_id, p.title AS title, p.topic AS topic;


// Common groups
MATCH (a:User {id: $user_a})-[:MEMBER_OF]->(g:Group)<-[:MEMBER_OF]-(b:User {id: $user_b})
RETURN g.id AS group_id, g.name AS group_name;


// ──────────────────────────────────────────────────────────────
// GRAPH CONTEXT FOR RAG
// ──────────────────────────────────────────────────────────────

// User neighborhood context (1-hop) for RAG
MATCH (u:User {id: $user_id})
OPTIONAL MATCH (u)-[:FRIEND]->(f:User)
OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
OPTIONAL MATCH (u)-[:LIKED]->(lp:Post)
RETURN u.id AS user_id,
       u.name AS name,
       u.bio AS bio,
       u.influence_score AS influence,
       collect(DISTINCT {id: f.id, name: f.name}) AS friends,
       collect(DISTINCT {id: p.id, title: p.title, likes: p.like_count}) AS posts,
       collect(DISTINCT {id: lp.id, title: lp.title}) AS liked_posts;


// Subgraph for two users (explain connection)
MATCH (a:User {id: $user_a}), (b:User {id: $user_b})
OPTIONAL MATCH (a)-[:FRIEND]->(af:User)
OPTIONAL MATCH (b)-[:FRIEND]->(bf:User)
OPTIONAL MATCH (a)-[:POSTED]->(ap:Post)
OPTIONAL MATCH (b)-[:POSTED]->(bp:Post)
RETURN a.name AS user_a_name,
       b.name AS user_b_name,
       collect(DISTINCT af.name) AS a_friends,
       collect(DISTINCT bf.name) AS b_friends,
       [f IN collect(DISTINCT af) WHERE f IN collect(DISTINCT bf) | f.name] AS mutual_friends,
       collect(DISTINCT {topic: ap.topic}) AS a_topics,
       collect(DISTINCT {topic: bp.topic}) AS b_topics;


// ──────────────────────────────────────────────────────────────
// VECTOR SIMILARITY QUERY (for hybrid retrieval)
// ──────────────────────────────────────────────────────────────

// Vector similarity search on user embeddings
CALL db.index.vector.queryNodes('user_embeddings', $top_k, $query_vector)
YIELD node AS u, score
RETURN u.id AS user_id,
       u.name AS name,
       u.bio AS bio,
       u.influence_score AS influence,
       score AS similarity_score
ORDER BY score DESC;


// ──────────────────────────────────────────────────────────────
// LINK PREDICTION CANDIDATES
// ──────────────────────────────────────────────────────────────

// Candidate pairs for link prediction (non-connected users with shared context)
MATCH (u:User {id: $user_id})-[:FRIEND*2..3]->(candidate:User)
WHERE candidate.id <> $user_id
  AND NOT (u)-[:FRIEND]->(candidate)
WITH candidate, count(*) AS path_count
ORDER BY path_count DESC
LIMIT $top_k
RETURN candidate.id AS candidate_id,
       candidate.name AS name,
       path_count AS graph_score;


// ──────────────────────────────────────────────────────────────
// WRITE QUERIES
// ──────────────────────────────────────────────────────────────

// Create/update user with embedding
MERGE (u:User {id: $user_id})
SET u.name = $name,
    u.bio = $bio,
    u.embedding = $embedding,
    u.influence_score = $influence_score,
    u.updated_at = datetime();

// Create friendship
MATCH (a:User {id: $user_a}), (b:User {id: $user_b})
MERGE (a)-[:FRIEND]->(b)
MERGE (b)-[:FRIEND]->(a);

// Create post
MERGE (p:Post {id: $post_id})
SET p.title = $title,
    p.content = $content,
    p.created_at = datetime(),
    p.like_count = 0,
    p.comment_count = 0,
    p.topic = $topic
WITH p
MATCH (u:User {id: $user_id})
MERGE (u)-[:POSTED]->(p);

// Like a post
MATCH (u:User {id: $user_id}), (p:Post {id: $post_id})
MERGE (u)-[:LIKED]->(p)
WITH p
SET p.like_count = p.like_count + 1;
