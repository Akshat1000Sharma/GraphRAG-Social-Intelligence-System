# GraphRAG Social Intelligence System — 20-Slide Presentation Guide

> This document provides slide-by-slide content, speaker notes, and image specifications for a 20-slide project presentation.

---

## Slide 1: Title Slide

**Title:** Social Network Intelligence System: GNN + GraphRAG + Multi-Agent KAG Pipeline

**Subtitle:** Enhancing Graph Neural Networks with Retrieval-Augmented Generation for Node Classification

**Authors:** Neeraj Yadav (2310110199), Dev Singh (2310110099), Gourav Varanasi

**Affiliation:** Department of Computer Science, Shiv Nadar University

**Date:** April 2026

**Image:** University logo (top-right corner). Use Shiv Nadar University logo from official resources.

---

## Slide 2: Problem Statement & Motivation

**Content:**
- Traditional databases answer simple queries: "How many followers does X have?"
- Cannot answer complex questions like:
  - "Who are the rising influencers in the AI space?"
  - "Why might User A and User B become friends?"
  - "Recommend friends based on semantic interests AND network topology"
- **Gap:** Pure text retrieval ignores topology; pure graph algorithms miss semantics
- **Our Solution:** Fuse GNNs (structure) + RAG (semantics) + LLMs (reasoning)

**Image:** A split diagram showing "Text-only search" vs "Graph-only search" vs "Our Hybrid approach" with checkmarks. *Create this as a simple comparison diagram in PowerPoint or export from draw.io.*

---

## Slide 3: Project Overview & Tech Stack

**Content:**
| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python) |
| Database | Neo4j 5.13 (Graph + Vector) |
| ML/AI | PyTorch, PyG, SentenceTransformers, Gemini LLM, LangChain |
| Frontend | Next.js (React) + TailwindCSS |
| Training | Kaggle GPU (Tesla T4) |
| Deployment | Docker + Docker Compose |

**Image:** Tech stack logos arranged in a layered architecture diagram. *Collect logos of FastAPI, Neo4j, PyTorch, Next.js, Docker, Kaggle, and Gemini. Arrange in a horizontal or layered layout.*

---

## Slide 4: Datasets

**Content:**

| Property | Facebook (SNAP MUSAE) | Twitter (SNAP ego) |
|----------|----------------------|-------------------|
| Nodes | 22,470 | 133,857 |
| Edges | 171,002 | 1,774,636 |
| Features | 4,714 (sparse binary) | 385 (proj. BOW + structural) |
| Classes | 4 (politician, govt, tvshow, company) | 4 (lurker, regular, active, influencer) |
| Split | 80/20 stratified | 80/20 stratified |

**Image:** Two side-by-side network graph visualizations.

**How to get the image:** Open Neo4j Browser → run `MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100` → screenshot the graph visualization.

---

## Slide 5: Methodology Overview — Three Stages

**Content:**
```
Stage 1: RAG-Only     → Text embedding + FAISS retrieval + LLM Q&A
                         (No graph structure used)

Stage 2: GNN-Only     → GraphSAGE on node features + edge structure
                         (No text semantics)

Stage 3: GNN + RAG    → Concatenate text embeddings with graph features
                         → Train GNN on enriched features
                         (Best of both worlds)
```

- Each stage evaluated on BOTH Facebook and Twitter datasets
- Total: 6 Kaggle notebooks

**Image:** A horizontal timeline/arrow showing the three stages with icons. *Create in PowerPoint with arrow shapes and stage labels.*

---

## Slide 6: Stage 1 — RAG-Only Pipeline

**Content:**
- Convert each node → natural language document
  - Facebook: "Node ID: X, Page: [name], Category: [type]"
  - Twitter: "Node ID: X, Ego network, Activity tier: [tier]"
- Embed with `all-MiniLM-L6-v2` → 384-d vectors
- Index with FAISS (IndexFlatL2)
- Query: encode question → retrieve top-k → Gemini generates answer

**Key Output (Facebook):** Embedding shape: (22,470 × 384)

**Limitation:** Cannot classify nodes — no graph structure used

**Image:** Screenshot from `rag-only-pipeline-facebook.ipynb` showing a RAG query and Gemini's answer. *Open the notebook on Kaggle → screenshot the output cell with "🔍 Query: What type of facebook pages exist..." and the Gemini answer.*

---

## Slide 7: Stage 2 — GNN-Only Pipeline (Facebook)

**Content:**
- **Architecture:** 2-layer GraphSAGE (4,714 → 64 → 4)
- **Training:** 21 epochs, batch size 1024, CUDA, lr=0.005

**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | **93.06%** |
| F1 (macro) | **0.9278** |
| AUC (macro OVR) | **0.9880** |

**Image:** Screenshot from `gnn-only-facebook.ipynb` showing the final test results: `Test Accuracy: 0.9306 | F1 (macro): 0.9278 | AUC (macro OVR): 0.9880`. *Kaggle notebook → screenshot the final output cell.*

---

## Slide 8: Stage 2 — GNN-Only Pipeline (Twitter)

**Content:**
- **Architecture:** 3-layer GraphSAGE + LayerNorm + BatchNorm + Dropout(0.4) + Fusion
- **Training:** NeighborLoader [12,10,8], batch 256, AMP, early stopping
- **Data:** 133,857 nodes, 3,549,272 edges

**Results (best validation checkpoint):**
| Metric | Value |
|--------|-------|
| Best Val Accuracy | **87.94%** |
| Test Accuracy | **87.27%** |
| Test F1 (macro) | **0.8573** |
| Test AUC (macro OVR) | **0.9857** |

**Image:** Screenshot from `gnn-only-twitter.ipynb` showing `Loaded best val checkpoint: val acc 0.8794, test acc at save 0.8727`.

---

## Slide 9: Stage 3 — GNN+RAG Hybrid (Facebook)

**Content:**
- **Innovation:** Concatenate MiniLM text embeddings (384-d) with sparse features (4,714-d) → 5,098-d input
- **Architecture:** 2-layer GraphSAGE (5,098 → 64 → 4)

**Results:**
| Metric | GNN-Only | GNN+RAG | Improvement |
|--------|----------|---------|-------------|
| Accuracy | 93.06% | **94.37%** | +1.31% |
| F1 | 0.9278 | **0.9412** | +1.34% |
| AUC | 0.9880 | **0.9918** | +0.38% |

**Image:** Screenshot from `gnn-rag-facebook-best.ipynb` showing: `Test Accuracy: 0.9437 | F1 (macro): 0.9412 | AUC (macro OVR): 0.9918`.

---

## Slide 10: Stage 3 — GNN+RAG Hybrid (Twitter)

**Content:**
- **Innovation:** Feature-grounded per-node text with degree, circle strength, top BOW projections
- **Architecture:** GNN_RAG_Model with text adapter (bottleneck=96), 3 SAGEConv + BN, fusion head

**Results:**
| Metric | GNN-Only | GNN+RAG | Improvement |
|--------|----------|---------|-------------|
| Accuracy | 87.27% | **97.46%** | **+10.19%** |
| F1 | 0.8573 | **0.9704** | **+11.31%** |
| AUC | 0.9857 | **0.9920** | +0.63% |

**Image:** Screenshot from `gnn-rag-twitter-best.ipynb` showing: `Test Accuracy: 0.9746 | F1 (macro): 0.9704 | AUC (macro OVR): 0.9920`.

---

## Slide 11: Results Comparison — All 6 Notebooks

**Content:**

| Pipeline | Facebook Acc | Facebook F1 | Twitter Acc | Twitter F1 |
|----------|-------------|-------------|-------------|------------|
| RAG-Only | Qualitative | — | Qualitative | — |
| GNN-Only | 93.06% | 0.9278 | 87.27% | 0.8573 |
| GNN+RAG | **94.37%** | **0.9412** | **97.46%** | **0.9704** |

**Key insight:** GNN+RAG gives the biggest improvement on Twitter (+10.2%) where per-node text is feature-grounded and diverse.

**Image:** Grouped bar chart comparing accuracy. Generate with:
```python
import matplotlib.pyplot as plt
labels = ['FB GNN-Only', 'FB GNN+RAG', 'TW GNN-Only', 'TW GNN+RAG']
acc = [93.06, 94.37, 87.27, 97.46]
colors = ['#4A90D9', '#2ECC71', '#E67E22', '#E74C3C']
plt.bar(labels, acc, color=colors)
plt.ylabel('Test Accuracy (%)')
plt.title('Node Classification Results')
plt.savefig('results_chart.png', dpi=200, bbox_inches='tight')
```

---

## Slide 12: System Architecture — Multi-Agent Pipeline

**Content:**
```
User Query → Analyzer → Router
→ Graph + Vector Retrievers (RRF)
→ GNN Inference (CPU)
→ Synthesizer (KAG/LLM) → Validator
→ Structured JSON + NL Insight
```

6 agents: Analyzer, Router, Retrievers, GNN, Synthesizer, Validator

**Image:** Polished flowchart recreating the architecture from README.md. *Create in draw.io or PowerPoint with colored boxes and arrows. Export as PNG.*

---

## Slide 13: Neo4j Unified Backend

**Content:**
- **Before:** Neo4j + FAISS + Python RRF (2 systems)
- **After:** Neo4j only — graph + vector + Cypher-side fusion (1 system)
- Three vector indexes: user_text (384-d), user_gnn (128-d), post_text (384-d)
- Hybrid queries save 20-40ms by eliminating Python-side RRF

**Image:** Screenshot of Neo4j Browser showing graph schema. *Run `CALL db.schema.visualization()` → screenshot.*

---

## Slide 14: Backend — FastAPI API Endpoints

**Content:** 15+ endpoints across System, AI/Chat, Graph API, and Mutations categories.

**Image:** Screenshot of Swagger UI at `http://localhost:8000/docs`. *Start API → open `/docs` → screenshot.*

---

## Slide 15: Frontend — Dashboard & Chat

**Content:** Dashboard with health cards and dataset stats. Chat with ChatGPT-style interface and timing analytics.

**Image:** Screenshots of Dashboard page and Chat page. *Run frontend (`npm run dev`) → screenshot `localhost:3000` and `localhost:3000/chat`.*

---

## Slide 16: Frontend — Explorer, Analytics & Users

**Content:** Graph Explorer (friend recs, influencers, link candidates), Analytics (trending charts), Users (profile lookup).

**Image:** Screenshots of Explorer, Analytics, and Users pages. *Navigate to `/explore`, `/analytics`, `/users` → screenshot each.*

---

## Slide 17: Frontend — Connections & Admin

**Content:** Connections pathfinding visualizer with AI Explain. Admin NL insert for database mutations.

**Image:** Screenshots of Connections page (path visualization) and Admin page (NL insert). *Navigate to `/connections` and `/admin` → screenshot.*

---

## Slide 18: Deployment — Docker & Kaggle

**Content:**
- Docker Compose: 2 services (neo4j + api), health checks, one-command startup
- Kaggle: GPU training pipeline, push script, output download

**Image:** Terminal screenshot of `docker-compose up` and Kaggle notebook with GPU indicator.

---

## Slide 19: Key Design Decisions & Performance

**Content:**

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Train/Inference | GPU/CPU split | Cheap VM deployment |
| Vector Store | Neo4j-native | Single system |
| Fusion | RRF | Parameter-free |
| Hallucination | 6-step validator | Grounding checks |

Performance: AUC 0.99, F1 0.97, API latency ~68ms

---

## Slide 20: Conclusion & Future Work

**Key Achievements:**
- GNN+RAG outperforms GNN-only: +1.3% Facebook, +10.2% Twitter
- Full production: 6 notebooks → FastAPI → Neo4j → Next.js → Docker
- 15+ APIs, 7 frontend screens, multi-agent KAG pipeline

**Future Work:** Reddit dataset, GAT for all datasets, real-time streaming, confidence calibration

**Thank You — Questions?**

**Image:** Summary infographic: "3 Datasets → 6 Notebooks → 97.46% Best → 15+ APIs → 7 Screens → Docker"

---

## Image Sourcing Summary

| Slide | Image Type | Source |
|-------|-----------|--------|
| 1 | University logo | Official SNU resources |
| 2 | Comparison diagram | Create in PowerPoint/draw.io |
| 3 | Tech stack logos | Download official logos |
| 4 | Graph visualization | Neo4j Browser screenshot |
| 6 | Notebook output | Kaggle `rag-only-pipeline-facebook.ipynb` |
| 7 | Notebook output | Kaggle `gnn-only-facebook.ipynb` |
| 8 | Notebook output | Kaggle `gnn-only-twitter.ipynb` |
| 9 | Notebook output | Kaggle `gnn-rag-facebook-best.ipynb` |
| 10 | Notebook output | Kaggle `gnn-rag-twitter-best.ipynb` |
| 11 | Bar chart | Generate with matplotlib (code above) |
| 12 | Architecture flowchart | Create in draw.io / PowerPoint |
| 13 | Neo4j schema | Neo4j Browser: `CALL db.schema.visualization()` |
| 14 | Swagger UI | Browser: `http://localhost:8000/docs` |
| 15-17 | Frontend screenshots | Browser: `http://localhost:3000/*` |
| 18 | Docker + Kaggle | Terminal + Kaggle web UI |
| 20 | Summary infographic | Create in Canva/PowerPoint |

---

## Presentation Timing

| Section | Slides | Time |
|---------|--------|------|
| Introduction & Setup | 1-5 | 5 min |
| Notebook Results | 6-11 | 8 min |
| System Implementation | 12-17 | 5 min |
| Deployment & Conclusion | 18-20 | 2 min |
| **Total** | **20** | **20 min + 5 min Q&A** |
