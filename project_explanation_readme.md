# GraphRAG Social Intelligence System: Complete Project Explanation

Welcome to the **GraphRAG Social Intelligence System**! This document is designed to be the ultimate, single source of truth for understanding how this entire project works from the ground up. Whether you are a new developer joining the team or someone with zero prior context, reading this file will give you a comprehensive understanding of the project's aims, architecture, logic, and APIs.

---

## 1. Project Aim & What It Does

**The Goal:**
The aim of this project is to build an advanced, intelligent social network analysis tool. Imagine having a massive dataset of social interactions (who friends whom, who likes what posts). Traditional databases can only answer simple questions like "How many followers does X have?" 

This project uses **GraphRAG (Retrieval-Augmented Generation over a Graph Database)** and **GNNs (Graph Neural Networks)** to answer incredibly complex, contextual questions like:
*   *"Who are the rising influencers in the AI space?"*
*   *"Why might User A and User B become friends?"*
*   *"Recommend 5 new friends for User X based on their semantic interests and network topology."*

**What it Does:**
It ingests massive real-world datasets (like Facebook, Twitter, and Reddit snaps), stores them in a highly interconnected format (Neo4j), calculates deep mathematical predictions using PyTorch GNNs, and allows a human to converse with all this data naturally using an AI LLM (Large Language Model) Multi-Agent Pipeline.

---

## 2. The Core Technologies (The Tech Stack)

1.  **Backend Framework:** `FastAPI` (Python) - Handles all API requests incredibly fast.
2.  **Database:** `Neo4j` - A Graph Database. Instead of tables and rows, it stores **Nodes** (Users, Posts) and **Edges** (FRIEND, POSTED, LIKED). It also natively stores **Vector Embeddings** for semantic search.
3.  **Machine Learning / AI:** 
    *   `PyTorch` & `Deep Graph Library (DGL)`: Used for the Graph Neural Networks (GNN).
    *   `LangChain`: Orchestrates the Multi-Agent LLM pipeline.
    *   `SentenceTransformers`: Generates semantic text embeddings for posts and users.
    *   `Gemini LLM`: Provides the intelligence for understanding user intent and summarizing data.
4.  **Frontend:** `Next.js` (React) & `TailwindCSS` - A modern, dynamic, responsive web interface.

---

## 3. High-Level Architecture Flow

When a user interacts with the system, data flows through three main layers:

### Layer A: Data Ingestion & Storage (The Foundation)
On startup, the system downloads raw CSV/TSV datasets (Facebook, Twitter, Reddit). It runs them through an ingestion pipeline:
1.  **Node Creation:** Creates Users and Posts in Neo4j.
2.  **Edge Creation:** Links them (e.g., User-[FRIEND]->User).
3.  **Embedding Generation:** It reads the text of posts/bios, passes it through an ML model (`all-MiniLM-L6-v2`) to turn words into numbers (vectors), and stores these in Neo4j. This allows the system to understand "meaning" (e.g., knowing that a post about "puppies" is similar to a post about "dogs").

### Layer B: The Multi-Agent Pipeline (The Brain)
When a user asks a natural language question (e.g., *"Recommend friends for Alice"*), it triggers the Multi-Agent Pipeline:
1.  **Analyzer Agent:** Reads the query, identifies the intent (`FRIEND_RECOMMENDATION`), and extracts entities ("Alice").
2.  **Router Agent:** Decides how to fetch the data. Should it use the Graph structure (connections), Vector semantics (meanings), or a Hybrid of both?
3.  **Retrievers Agent:** Executes the queries against Neo4j.
4.  **GNN Inference Engine:** Takes the retrieved users and passes them through the PyTorch Neural Network. The GNN predicts things like "Is this user a Community Hub?" or computes the exact mathematical probability that two users will become friends.
5.  **Synthesizer Agent:** The LLM takes all the raw data (Database rows + GNN probabilities) and writes a human-readable response.
6.  **Validator Agent:** Double-checks the LLM's response to ensure it didn't hallucinate fake data.

### Layer C: The Frontend (The Interface)
The React application consumes the FastAPI endpoints to render Dashboards, Chats, Network Graphs, and Analytics charts.

---

## 4. Extreme Detail: How Specific Features Work

### A. Hybrid Search & Retrieval (GraphRAG)
RAG (Retrieval-Augmented Generation) usually just searches text. **GraphRAG** searches text *and* relationships.
If you search for *"Trending AI posts"*:
1.  **Vector Search:** Neo4j looks at the vector embeddings of all posts to find ones semantically similar to "AI".
2.  **Graph Search (Cypher):** Neo4j simultaneously calculates the "engagement velocity" of the posts using graph mathematics (`p.like_count + (p.comment_count * 2)`).
3.  **Reciprocal Rank Fusion (RRF):** The system merges the Vector score and Graph score to return the absolute best results. 

### B. Graph Neural Networks (GNN)
Located in `model/inference.py`. The system has pre-trained models for different datasets.
*   **Influencer Detection (Node Classification):** The GNN analyzes the entire network shape. It doesn't just look at follower count; it looks at the *quality* of followers. It classifies nodes into: `regular_user`, `influencer`, `content_creator`, `community_hub`.
*   **Link Prediction:** If User A and User B aren't friends, the GNN looks at their shared neighborhood and calculates a decimal probability (e.g., 0.89) that they *should* be friends.

### C. Natural Language Graph Mutations (NL Insert)
On the Admin page, a user can type: *"Add user Charlie who is friends with Alice"*.
1.  The text goes to `/chat/insert/parse`.
2.  An LLM breaks this down into JSON: `[{"type": "CREATE_USER", "name": "Charlie"}, {"type": "CREATE_EDGE", "source": "Charlie", "target": "Alice"}]`.
3.  The `InsertService` executes write-transactions (`run_write_query`) directly in Neo4j to mutate the database live.

---

## 5. Backend API Reference

The backend exposes several routes. Here is a breakdown of the core APIs:

### System & Datasets
*   `GET /health`: Checks if Neo4j is online, which GNN models are loaded, and returns vector index status.
*   `GET /datasets/status`: Checks the local disk to see if the massive data files (Facebook, Twitter, Reddit) are present and ingested.
*   `POST /datasets/ingest`: Triggers the massive background job to download missing data and bulk-load it into Neo4j.

### GraphRAG & AI Chat
*   `POST /chat`: The main interface for the Multi-Agent Pipeline. Takes a natural language query, runs it through the 6 agents, and returns a synthesized insight along with pipeline execution timings (in milliseconds).

### Direct Graph Queries (Fast UI Endpoints)
These endpoints bypass the slow LLM pipeline. They use direct Cypher queries for instantaneous UI rendering.
*   `GET /graph/friend-recommendations/{user_id}`: Traverses the graph to find friends-of-friends, counting mutual connections to rank suggestions.
*   `GET /graph/trending-posts`: Calculates an engagement score for recent posts.
*   `GET /graph/users/{user_id}/influence-stats`: Aggregates a user's total likes, posts, and returns their GNN influence score.
*   `GET /graph/connection-path`: Finds the exact shortest path (up to 6 hops) between two users, plus mutual friends and common liked posts.
*   `GET /graph/top-influencers`: Ranks the whole network using a composite score of followers, average likes, and post counts.

### AI Explanations & Mutations
*   `GET /explain-connection`: Passes two users to the LLM pipeline so the AI can explain *why* they are connected based on their shared subgraph.
*   `POST /chat/insert`: Executes the Natural Language to Cypher mutations.

---

## 6. Frontend Screen Breakdown

Located in `frontend/app/`.

1.  **Dashboard:** The command center. Shows Neo4j connectivity, loaded ML models, and how many nodes/edges exist per dataset. Features a live animation of the 6-stage Multi-Agent pipeline.
2.  **Chat (AI):** A ChatGPT-like interface. You can ask complex questions. A unique feature is the "Raw JSON" toggle, allowing developers to see the exact context the Graph returned to the LLM before synthesis.
3.  **Explore:** A grid layout to browse raw Graph queries instantly. You can toggle between viewing Friend Recommendations, Top Influencers, or Link Candidates (GNN predictions).
4.  **Analytics:** Visualizes the network. Uses chart libraries to graph trending topics and engagement scores over time.
5.  **Users:** A profile lookup tool. Type an ID to get a beautifully rendered card of that user's footprint (their GNN assigned role, likes, friends).
6.  **Connections:** The pathfinding visualizer. You enter two User IDs, and the UI draws a literal path (User A -> FRIEND -> User C -> FRIEND -> User B). It also renders commonalities (shared posts).
7.  **Admin:** The database mutation hub. Here you can execute Natural Language Inserts to alter the database without knowing Cypher.

---

## Summary

The GraphRAG Social Intelligence System is a state-of-the-art fusion of **Graph Databases (topology)**, **Vector Search (semantics)**, **Graph Neural Networks (deep learning)**, and **Large Language Models (reasoning)**. 

By combining these four pillars, the system doesn't just store social data—it truly *understands* the social fabric, predicting relationships, identifying structural influencers, and explaining complex network paths in plain English.
