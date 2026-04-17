# Facebook Graph Classification — Project Documentation
### From Exploration to Production-Grade Pipeline

> **Dataset:** SNAP Facebook Large Page-Page Network — ~22,470 nodes, ~171,002 edges, 4 classes (`politician`, `government`, `tvshow`, `company`)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [File 1 — RAG-Only Pipeline](#file-1--rag-only-pipeline)
3. [File 2 — GNN-Only](#file-2--gnn-only)
4. [File 3 — GNN + RAG Hybrid (Training)](#file-3--gnn--rag-hybrid-training)
5. [File 4 — Final Testing Pipeline](#file-4--final-testing-pipeline-testing_pipeline_accuracypy)
6. [Progression Summary](#progression-summary)
7. [Accuracy Results](#accuracy-results)
8. [Key Lessons Learned](#key-lessons-learned)

---

## Project Overview

This project explores node classification on a Facebook social graph — given a Facebook page and its connections, predict whether it belongs to one of four categories. The work evolved through four distinct stages, each improving on the last in methodology, architecture, and rigour.

```
RAG-Only (NB1)
    │  ❌ No graph structure, no training, just retrieval + LLM
    │
    ▼
GNN-Only (NB2)
    │  ✅ Graph structure + training, but weak architecture
    │  📈 Accuracy: 93.52%
    ▼
GNN + RAG Hybrid (NB3)
    │  ✅ Richer features (sparse + semantic), deeper model
    │  📈 Test Accuracy: 94.62%
    ▼
Final Testing Pipeline (testing_pipeline_accuracy.py)
    │  ✅ Graph-RAG (label propagation), strict val/test split,
    │     vectorised hyperparameter search, 94%+ threshold gate
    └─ 🏆 Production-grade evaluation & regression test
```

---

## File 1 — RAG-Only Pipeline
**File:** `rag-only-pipeline.ipynb`

### What It Does

This was the **exploration phase** — treating graph node classification as a pure text retrieval and question-answering problem. The graph structure was ignored entirely. Each node was converted to a short text document, embedded using a sentence encoder, stored in a FAISS vector index, and queried via Google Gemini.

### Architecture

```
Node → Text Document → SentenceTransformer Embedding → FAISS Index
                                                              │
Query → Encode Query → Top-5 Similar Docs ──────────────────┘
                                │
                                ▼
                        Gemini 2.5 Flash (LLM) → Answer
```

### Key Numbers

| Component | Detail |
|---|---|
| Nodes | ~22,470 Facebook pages |
| Embedding model | `all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Vector index | FAISS `IndexFlatL2` |
| Retrieval top-k | 5 |
| LLM | Google Gemini 2.5 Flash |
| Training | ❌ None |
| Graph structure used | ❌ No |

### Document Format

Each node became a text document:
```
Node ID: 1234
This Facebook page belongs to category: politician.
This page is part of a social network graph dataset.
```

### What It Got Right and Wrong

**✅ Got right:**
- Proved that semantic retrieval can find relevant nodes
- Fast to set up, no training required
- Good for exploratory Q&A ("Which nodes belong to politician category?")

**❌ Got wrong:**
- Completely ignores graph structure (edges, connectivity, neighbourhood)
- Is a Q&A chatbot, not a node classifier — it doesn't produce per-node predictions
- No train/test split, no accuracy metric, no reproducibility
- Retrieval is based on label leakage — documents already contain the category label, so the system is essentially looking up answers rather than predicting them
- Does not scale to unseen/unlabeled nodes

### Why We Moved On

RAG-only treats nodes as isolated text blobs. A Facebook page's category is strongly determined by *who it is connected to* — a politician page links to other politician pages. Ignoring edges throws away the most valuable signal. We needed a model that could learn from the graph topology.

---

## File 2 — GNN-Only
**File:** `gnn-only.ipynb`

### What It Does

This was the **baseline phase** — training a real Graph Neural Network on the Facebook graph. The GNN learns node representations by aggregating information from each node's neighbours, which is exactly what the graph structure provides.

### Architecture

```
Sparse Node Features (binary, bag-of-words)
        │
        ▼
  SAGEConv Layer 1 → ReLU     (in_features → 64)
        │
  SAGEConv Layer 2             (64 → 4 classes)
        │
        ▼
    Predictions (argmax)
```

**Model:** 2-layer GraphSAGE
- Hidden dimension: 64
- Optimizer: Adam, lr = 0.005
- Epochs: 20
- Batch size: 1,024 (mini-batch over training nodes)
- Split: 80% train / 20% test (stratified)

### Data Pipeline

```
musae_facebook_edges.csv   →  edge_index tensor [2, ~342,004]
musae_facebook_features.json → X feature matrix [22,470 × N_features] (binary)
musae_facebook_target.csv  →  y labels [22,470] (encoded 0–3)
```

- Edges are made bidirectional (undirected graph)
- Features are sparse: page description words encoded as binary one-hot
- Labels: `company=0, government=1, politician=2, tvshow=3`

### Training Loop

```python
# Mini-batch training on full graph
for epoch in range(20):
    perm = shuffle(train_indices)
    for batch in chunks(perm, batch_size=1024):
        out = model(data.x, data.edge_index)  # full-graph forward pass
        loss = cross_entropy(out[batch], y[batch])
        loss.backward(); optimizer.step()
```

### Key Numbers

| Metric | Value |
|---|---|
| Train nodes | ~17,976 |
| Test nodes | ~4,494 |
| Feature dimension | Variable (max feature index + 1) |
| GNN layers | 2 |
| Hidden dim | 64 |
| Epochs | 20 |
| **Test Accuracy** | **93.52%** |
| Classes predicted | All 4 (confirmed: `tensor([0, 1, 2, 3])`) |

### What It Got Right and Wrong

**✅ Got right:**
- Uses graph structure — nodes learn from their neighbours
- Proper train/test split with stratification
- Measurable accuracy on a held-out set
- Proves GNNs work well on this problem

**❌ Got wrong:**
- Only 20 epochs of training — model is underfit
- Small hidden dimension (64) — limited representational capacity
- No BatchNorm or Dropout — no regularisation
- No learning rate scheduling
- No semantic/text features — sparse binary features only
- No validation set — hyperparameters not properly tuned

### Why We Moved On

93.52% is a solid baseline, but the model was shallow and undertrained. More importantly, the sparse binary features from the dataset don't capture *semantic meaning* — two pages about different politicians might have very different word indices despite being the same category. We needed richer features.

---

## File 3 — GNN + RAG Hybrid (Training)
**File:** `gnn-rag.ipynb`

### What It Does

This was the **main training phase** — combining the best ideas from both previous files. Semantic text embeddings (from File 1's RAG approach) were concatenated with the sparse graph features, and a deeper, better-regularised GNN was trained on the combined representation.

### The Core Idea

> The RAG component enriches each node's *feature vector* before GNN training — rather than retrieving at query time, semantic knowledge is baked into the input features.

```
Sparse Features [22,470 × N_feat]  ──┐
                                      ├─ Concatenate → Combined X [22,470 × (N_feat + 384)]
Text Embeddings [22,470 × 384]    ──┘
                                              │
                                              ▼
                                    3-layer GraphSAGE + BN + Dropout
                                              │
                                              ▼
                                         4-class output
```

### Text Embedding Strategy

Each node gets a natural-language document:
```
Node ID: 1234
This is a Facebook page in a social network graph.
It is connected to other pages.
```

This is encoded using `all-MiniLM-L6-v2` → 384-dimensional dense vector. This embedding captures *semantic similarity* between nodes regardless of their specific word features. Then it is concatenated with the sparse structural features to form a much richer input.

### Architecture

```
Input: X_combined = [sparse_features | text_embeddings]
       Dimension: N_feat + 384

SAGEConv(in → 128) → BatchNorm1d(128) → ReLU → Dropout(0.3)
SAGEConv(128 → 128) → BatchNorm1d(128) → ReLU → Dropout(0.3)
SAGEConv(128 → 128) → BatchNorm1d(128) → ReLU
Linear(128 → 128) [fusion] → ReLU
Linear(128 → 4) [classifier]
```

### Training Configuration

| Parameter | Value |
|---|---|
| Hidden dimension | 128 (vs 64 in NB2) |
| GNN layers | 3 (vs 2 in NB2) |
| Fusion layer | ✅ Additional linear layer |
| BatchNorm | ✅ After each conv |
| Dropout | ✅ p=0.3 after conv 1 & 2 |
| Optimizer | Adam, lr=0.01, weight_decay=5e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=50) |
| Epochs | 101 (vs 20 in NB2) |
| Split | 80/20 stratified |

### Outputs Saved

```
weights/
├── best.pth                    ← best checkpoint by val accuracy
├── model_weights_facebook.pth  ← final weights (used by File 4)
└── embeddings_facebook.npy     ← node embedding vectors [22,470 × 128]
```

### Key Numbers

| Metric | Value |
|---|---|
| **Train Accuracy** | **99.84%** |
| **Test Accuracy** | **94.62%** |
| Improvement over GNN-only | +1.10% on test |
| Feature input size | N_feat + 384 |
| Parameters | Significantly more than NB2 |

### What It Got Right and Wrong

**✅ Got right:**
- Richer features: semantic + structural
- Deeper, better-regularised model
- Longer training with LR scheduling
- Saves checkpoint for downstream use
- Meaningful accuracy improvement

**❌ Got wrong:**
- Still only 80/20 split — hyperparameters were not tuned on a separate validation set
- The graph RAG at inference time is absent — retrieval is only used at feature construction time
- High train accuracy (99.84%) vs test (94.62%) suggests some overfitting
- No systematic search for optimal blend weights or graph-propagation parameters

### Why We Moved On

The training notebook produces a good model, but *evaluation* was not rigorous. There was no regression gate, no validation-based hyperparameter selection, and no graph-aware inference strategy. The final file addresses all of this.

---

## File 4 — Final Testing Pipeline
**File:** `testing_pipeline_accuracy.py`

### What It Does

This is the **production evaluation and regression testing phase**. It loads the pre-trained checkpoint from File 3, then asks: *"Can we do even better at inference time by blending GNN logits with graph neighbourhood information?"* It enforces a hard accuracy threshold and writes structured results.

### The Fundamental Difference from File 3

File 3's "RAG" enriches input features *before training*. File 4's "graph-RAG" operates *at inference time* on the GNN's output logits, using multi-hop label propagation through the graph — a completely different and complementary mechanism.

### Inference-Time Graph-RAG

```
Train labels ──→ M (one-hot seed matrix, train nodes only)
                      │
               A @ M = h1  (1-hop propagation)
               A @ h1 = h2 (2-hop)
               A @ h2 = h3 (3-hop)
               A @ h3 = h4 (4-hop)
               A @ h4 = h5 (5-hop)
                      │
h_graph = h1 + γ·h2 + δ·h3 + ε·h4 + ζ·h5 + prior
                      │
GNN logits ──→ p_gnn = softmax(logits / temperature)
                      │
pipeline prediction = (1-α)·p_gnn + α·(h_graph / sum)
                      │
    [homophily gate: if neighbours are confident → use their vote]
```

Where `A` is the row-normalised adjacency matrix. This is analogous to Neo4j neighbourhood retrieval but done efficiently via sparse matrix multiplication.

### Three-Way Data Split (Critical Improvement)

```
All labeled nodes (22,470)
        │
   80/20 stratified split
        │
   ┌────┴────────────────────┐
Train (80%)            Test (20%)  ← NEVER touched until final eval
   │
   15% of train = Validation
   │
   ├─ train_fit (68% of total) ← GNN fine-tune + propagation seeds
   └─ val (12% of total)       ← Hyperparameter tuning ONLY
```

This prevents the data leakage present in all three notebooks, where models were effectively evaluated on data that influenced their configuration.

### Hyperparameter Grid Search (Validation-Only)

All hyperparameters are tuned on the **validation set only**, then applied once to the test set:

| Parameter | What It Controls | Fast Grid | Heavy Grid |
|---|---|---|---|
| α (alpha) | GNN vs graph-RAG blend | 21 values | 51 values |
| τ (tau) | Min neighbour vote mass | 10 values | 17 values |
| min_conf | Min neighbour confidence | 13 values | 22 values |
| γ, δ, ε, ζ | Multi-hop weights (h2–h5) | 4×3×3×1 | 19×13×11×5 |
| η (eta) | 1-hop histogram weight | 14 values | 41 values |
| temperature | Logit sharpening | 12 values | 22 values |
| LP steps | Label propagation depth | 14 snapshots | 21 snapshots |
| prior strength | Dirichlet class smoothing | 5 values | 10 values |

All inner loops are **vectorised** with PyTorch batching — no Python loops over nodes.

### Pipeline Predict Logic

```python
# Homophily gate: trust neighbours when they strongly agree
if neighbour_vote_mass >= tau AND neighbour_confidence >= min_conf:
    prediction = argmax(neighbour_histogram)   # graph-RAG wins
else:
    prediction = argmax((1-α)*p_gnn + α*p_graph)  # blended
```

### Two Operation Modes

**Fast mode** (`PIPELINE_ACCURACY_FAST=1`)
- Smaller hyperparameter grid
- No GNN fine-tuning
- Runs on CPU in under 1 minute

**Heavy mode** (`PIPELINE_ACCURACY_HEAVY=1`, recommended on Kaggle GPU)
- Full grid search
- GNN fine-tuned on `train_fit` with OneCycleLR + early stopping on validation CE
- Label smoothing (default 0.03) during fine-tune only

### Hard Assertions (Regression Gates)

```python
# Gate 1: pipeline must not be worse than plain GNN
assert acc_pipe_test >= acc_gnn_test - 1e-6

# Gate 2: pipeline must exceed 94% on test set
assert acc_pipe_test > 235/250  # = 0.94
```

These make the script usable as a **CI/CD regression test** — if a new model checkpoint is saved, this test must pass before it is accepted.

### Outputs

```json
{
  "heavy_search": true,
  "device": "cuda",
  "acc_gnn_test": 0.9352,
  "acc_pipeline_test": 0.9462,
  "best_val_tuning": 0.9481,
  "best_graph_tag": "add",
  "best_eta_hist": 1.5,
  "best_logit_temp": 0.95,
  "best_alpha": 0.35,
  "best_tau": 3.0,
  "best_min_conf": 0.35,
  "seconds_wall": 42.7
}
```

---

## Progression Summary

### How Each File Led to the Next

**NB1 → NB2 (RAG-Only → GNN-Only)**

The RAG-only notebook proved that the dataset could be explored with text retrieval, but it was fundamentally not solving node classification — it was a Q&A system with label leakage. The decision to move to a GNN was motivated by the key insight that graph edges carry classification signal: Facebook pages cluster by type, and neighbours strongly predict a node's class. NB2 established a proper measurable baseline with train/test split and real accuracy metrics.

**NB2 → NB3 (GNN-Only → GNN+RAG Hybrid)**

The GNN-only model achieved 93.52% but used only sparse binary features. The insight from NB1 was that semantic text embeddings can capture meaning that raw feature indices cannot — two politician pages may share no common feature indices yet be semantically similar. NB3 combined both: semantic embeddings (from the RAG-style encoder) were concatenated into the node feature matrix, giving the GNN richer input signals. The architecture was also deepened (2→3 layers), widened (64→128 hidden), and properly regularised (BatchNorm, Dropout, LR scheduling, 101 epochs). This yielded the saved checkpoint used in File 4.

**NB3 → File 4 (Training → Evaluation)**

NB3 trained a good model but evaluated it naively. File 4 takes that checkpoint and asks whether inference-time graph-RAG (label propagation) can squeeze out additional accuracy. It introduces: (1) a proper 3-way split with a validation set for hyperparameter tuning, (2) multi-hop label propagation as a graph-native retrieval mechanism, (3) a homophily gate that dynamically decides when to trust neighbours vs the GNN, (4) a full vectorised grid search over all blending parameters, and (5) hard regression assertions. This transforms the project from a notebook experiment into a testable, reproducible system.

---

## Accuracy Results

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| RAG-Only (NB1) | N/A | N/A | No classifier, no split |
| GNN-Only (NB2) | ~93–94% (estimated) | **93.52%** | 2-layer SAGE, hidden=64, 20 epochs |
| GNN+RAG Hybrid (NB3) | **99.84%** | **94.62%** | 3-layer SAGE, hidden=128, 101 epochs, semantic features |
| Final Pipeline (File 4) | — | **≥ 94.00%** (threshold) | Loads NB3 checkpoint + label propagation RAG |

### Accuracy Progression Chart

```
Test Accuracy
  95.0% │                                              ┌──── ≥94% gate
        │                                  ┌───────────┤
  94.0% │                        ┌─────────┘ 94.62%    │
        │                        │  NB3                 │
  93.5% │              ┌─────────┘                      │
        │              │  92.52%                        │
  93.0% │              │  NB2                           │
        │              │                                │
   N/A  │──────────────┤                                │
        │  NB1         │                                │
        └──────────────┴────────────────────────────────┘
           RAG-Only    GNN-Only   GNN+RAG    Final Pipeline
```

### The Gap Explained

The +1.10% improvement from NB2 → NB3 comes from:
1. **Richer features**: text embeddings capture semantic similarity
2. **Deeper model**: 3 layers can aggregate 3-hop neighbourhoods
3. **Better regularisation**: BatchNorm + Dropout reduce overfitting
4. **Longer training**: 101 epochs with LR decay vs 20 flat epochs

The NB3 train accuracy of 99.84% vs test of 94.62% indicates mild overfitting — the gap is real but not catastrophic, and the test accuracy still comfortably clears the 94% threshold in File 4.

---

## Key Lessons Learned

### 1. Graph Structure Is the Most Valuable Signal
The biggest single jump was from RAG-Only (no graph) to GNN-Only (with graph). Edges encode category information directly — connected pages tend to be the same type.

### 2. Semantic Features Complement Structural Features
Concatenating text embeddings to sparse bag-of-words features gave a meaningful +1.10% improvement. The two feature types capture different aspects: structural features encode *what words appear on the page*, semantic embeddings encode *what the page is about*.

### 3. A Separate Validation Set Is Non-Negotiable
Notebooks 1–3 had no validation set — any configuration choice (epochs, learning rate, architecture) was implicitly tuned on the test set. File 4 corrects this with a strict 3-way split.

### 4. Inference-Time Graph-RAG Is Complementary to Training-Time Enrichment
File 3 bakes semantic knowledge into features before training. File 4 applies label propagation after training. These are different operations that stack: the GNN learns from rich features, then the graph-RAG corrects its predictions using the known label neighbourhood at test time.

### 5. Vectorisation Makes Grid Search Feasible
Searching thousands of hyperparameter combinations over 22,470 nodes would be prohibitively slow with Python loops. File 4's batch-tensor approach (`pipeline_predict_batch_atm`) makes this run in under a minute on CPU and seconds on GPU.

### 6. Regression Gates Enable Confident Iteration
Without a hard accuracy threshold, there is no safe way to update the model (e.g., retrain with new data) and know whether quality degraded. File 4's `assert acc_pipe_test > 0.94` is a safety net for future development.

---

*Documentation covers: `rag-only-pipeline.ipynb`, `gnn-only.ipynb`, `gnn-rag.ipynb`, `testing_pipeline_accuracy.py`*
*Dataset: SNAP Facebook Large Page-Page Network | Framework: PyTorch Geometric | Language: Python 3*
