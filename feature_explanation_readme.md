# GraphRAG Social Intelligence System - Feature & Architecture Explanation

This document provides a comprehensive explanation of every frontend screen, the UI components within them, and the underlying backend operations triggered by user interactions.

## 1. Dashboard (Home)
**Route:** `/`

*   **Components:**
    *   **System Health Cards:** Displays real-time status for the Neo4j database connection, GNN Model loaded state, Multi-Agent Pipeline readiness, and the active Vector Backend.
    *   **Dataset Statistics:** Visual cards showing node/edge counts for ingested datasets (e.g., Facebook, Twitter, Reddit, Demo).
    *   **Pipeline Stage Visualization:** A dynamic flow showing the 6 stages of the Multi-Agent architecture.
*   **Backend Interaction:**
    *   On load, the frontend hits `/health` and `/datasets/status`.
    *   The backend validates the Neo4j client connection (`app_state.neo4j.is_connected`), polls the vector index schema manager, and checks the in-memory state of the GNN engines and LangChain pipeline. It executes lightweight Cypher count queries to gather dataset statistics.

## 2. Chat Query (AI)
**Route:** `/chat`

*   **Components:**
    *   **Control Panel:** Dropdowns for Dataset selection, Retrieval Mode (Auto, Graph, Vector, Hybrid), and Top-K results.
    *   **Suggested Queries:** Quick-select buttons for common complex questions.
    *   **Chat Interface:** Message history, user input textarea, and a "Send" button.
    *   **Analytics Panel:** Shows pipeline execution timing per stage and a toggle to view the "raw JSON" response.
*   **Backend Interaction:**
    *   When a user sends a message, a POST request is made to `/chat`.
    *   The backend invokes the **Multi-Agent Pipeline** (`MultiAgentPipeline.run()`):
        1.  **Analyzer:** Uses an LLM to detect user intent, extract entities, and format the query.
        2.  **Router:** Determines whether to use Vector, Graph, or Hybrid retrieval based on the intent.
        3.  **Retrieval:** The `HybridRetriever` searches Neo4j vector indexes for semantic similarity and executes Cypher queries to fetch subgraph contexts.
        4.  **GNN Inference:** If applicable, the retrieved nodes are passed to the loaded PyTorch/DGL GNN engine to compute influence or link prediction scores.
        5.  **Synthesizer:** The LLM receives the raw context and synthesizes a natural language response.
        6.  **Validator:** Checks the generated response against the context to prevent hallucinations.
    *   The compiled response, including insights and millisecond timings, is returned to the UI.

## 3. Graph Explorer
**Route:** `/explore`

*   **Components:**
    *   **Query Type Toggles:** Switch between "Friend Recommendations", "Top Influencers", and "Link Candidates".
    *   **Inputs:** User ID field (e.g., `user_1`), Top-K slider/input, and Dataset selector.
    *   **Results Grid:** Displays user cards with relevant graph scores, mutual friend counts, and influence metrics.
*   **Backend Interaction:**
    *   Triggered by "Query", the frontend calls direct graph API endpoints like `/graph/friend-recommendations/{user_id}` or `/graph/top-influencers`.
    *   These bypass the LLM pipeline for speed. The backend `GraphQueryService` translates these into optimized **Cypher Queries**.
        *   *Friend Recs:* Finds friends-of-friends and aggregates mutual connections (`MATCH (u)-[:FRIEND]->(friend)-[:FRIEND]->(fof) ... count(friend) AS mutual_count`).
        *   *Top Influencers:* Computes a composite score using follower counts, average likes, and post counts.

## 4. Analytics
**Route:** `/analytics`

*   **Components:**
    *   **Load Trending Button:** Triggers data fetch.
    *   **Chart Area:** Renders bar charts and line graphs showing trending posts and engagement over time.
    *   **Stats Cards:** Summarizes total engagement, top topics, etc.
*   **Backend Interaction:**
    *   Calls `/graph/trending-posts` with a specified time window.
    *   The backend queries Neo4j for posts, calculates an engagement score dynamically in Cypher (`p.like_count + (p.comment_count * 2) AS engagement`), sorts them, and returns the structured data for charting.

## 5. Users
**Route:** `/users`

*   **Components:**
    *   **Search:** User ID input field.
    *   **Graph Stats Button:** Fetches user data.
    *   **Profile Card:** Displays a comprehensive view of followers, posts, friends, GNN scores, and total/average likes.
*   **Backend Interaction:**
    *   Hits `/graph/users/{user_id}/influence-stats`.
    *   The backend executes an aggregation query on Neo4j, grouping a user's `[:POSTED]` and `[:FRIEND]` relationships to compute their specific network footprint and retrieving their cached GNN influence score.

## 6. Connections
**Route:** `/connections`

*   **Components:**
    *   **Inputs:** "From" and "To" User ID fields.
    *   **Find Path Button:** Retrieves structured shortest path data.
    *   **AI Explain Button:** Uses the LLM to interpret the connection.
    *   **Visualizations:** A linear node-edge-node path visualization, lists of mutual friends, and shared liked posts.
*   **Backend Interaction:**
    *   **Find Path:** Calls `/graph/connection-path`. The backend runs `shortestPath((a)-[*..6]-(b))` in Cypher, alongside queries for common `[:FRIEND]` nodes and common `[:LIKED]` Post nodes.
    *   **AI Explain:** Calls `/explain-connection`. This routes through the Multi-Agent pipeline, providing the graph path as context to the LLM so it can generate a human-readable narrative explaining *how* and *why* the users are connected.

## 7. Admin
**Route:** `/admin`

*   **Components:**
    *   **Tabs:** Switch between "Datasets", "NL Insert", and "Insert User".
    *   **NL Insert Area:** A textarea for natural language commands (e.g., "Add user Alice who is friends with Bob") and an "Execute" button.
*   **Backend Interaction:**
    *   **Datasets Tab:** Allows triggering `/datasets/ingest` to download and bulk-load raw SNAP/Reddit data into Neo4j using the `Neo4jEmbeddingPopulator`.
    *   **NL Insert Parse & Execute:** 
        *   When previewed, calls `/chat/insert/parse`. The backend uses an LLM to translate the English sentence into a structured JSON payload detailing node and edge creation.
        *   When executed, calls `/chat/insert`. The backend dynamically maps the structured LLM output into transactional write Cypher queries (`run_write_query`) to mutate the graph database in real-time.
