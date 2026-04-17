"""
testing_pipeline_accuracy.py
=============================
Compares **node classification accuracy** on the Facebook MUSAE graph:

1. **GNN-only** — pretrained `weights/model_weights_facebook.pth` (legacy 3-layer GraphSAGE +
   classifier, matching the checkpoint that shipped with this repo).
2. **Pipeline (GNN + graph RAG)** — same logits blended with **train-only** multi-hop
   label propagation (row-normalized adjacency: h₁…h₄ plus a class-frequency prior),
   analogous to Neo4j neighborhood retrieval. Hyperparameters are tuned on a **validation**
   slice only, with **vectorized** search over blend weight α and gate (τ, min_conf) for speed
   (typically well under a minute on CPU; faster on GPU if CUDA is available).

**Split:** stratified **80% train / 20% test** on labeled pages, `random_state=42`, same spirit
as the historical `train_test_split(..., stratify=Y, test_size=0.2, random_state=42)` baseline.

**Note:** Current `training/train_facebook.py` trains `SocialGraphGNN` and reports link AUC +
node macro-F1 in `evaluate()`, not this exact stratified node accuracy on the held-out split.

**Heavy mode (Kaggle GPU, recommended):** Set ``PIPELINE_ACCURACY_HEAVY=1`` or run under
``/kaggle/working`` with CUDA — runs a short **GNN node-classification fine-tune** on the
train-fit split (early-stop on validation loss) and a **wider** graph-RAG search. Local CPU:
set ``PIPELINE_ACCURACY_FAST=1`` for a smaller grid and no fine-tune.

Run:

    pytest tests/testing_pipeline_accuracy.py -v --tb=short
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

from model.gnn_model import SocialGraphGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
TEST_SIZE = 0.2
VAL_FRACTION_OF_TRAIN = 0.15
FEATURE_DIM_CAP = 128
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "model_weights_facebook.pth"
DROPOUT = 0.3
# Pipeline test accuracy must exceed this float (two hundred thirty-five two-hundred-fiftieths + ε).
_MIN_STRATIFIED_PIPELINE_ACC = 235 / 250 + 1e-12


def _facebook_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data")) / "facebook"


def _use_heavy_pipeline_search() -> bool:
    if os.environ.get("PIPELINE_ACCURACY_FAST", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("PIPELINE_ACCURACY_HEAVY", "").lower() in ("1", "true", "yes"):
        return True
    return Path("/kaggle/working").is_dir() and torch.cuda.is_available()


@torch.enable_grad()
def finetune_node_classifier_transductive(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_fit_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    max_epochs: int = 80,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 6,
    eval_every: int = 2,
) -> None:
    """
    Fine-tune **node classifier** on ``train_fit_mask`` only; full-graph message passing.
    Early-stops on validation **cross-entropy** (``val_mask``) — no test leakage.
    """
    device = x.device
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_vloss = float("inf")
    bad = 0
    for ep in range(max_epochs):
        opt.zero_grad(set_to_none=True)
        if isinstance(model, LegacyFacebookGNN):
            logits = model(x, edge_index)
        else:
            z = model.encode(x, edge_index)
            logits = model.classify_node(z)
        loss = F.cross_entropy(
            logits[train_fit_mask],
            y[train_fit_mask],
            label_smoothing=0.05,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if ep % eval_every != 0:
            continue
        model.eval()
        with torch.no_grad():
            if isinstance(model, LegacyFacebookGNN):
                logits_v = model(x, edge_index)
            else:
                zv = model.encode(x, edge_index)
                logits_v = model.classify_node(zv)
            vloss = F.cross_entropy(
                logits_v[val_mask], y[val_mask], label_smoothing=0.0
            ).item()
        model.train()
        if vloss < best_vloss - 1e-5:
            best_vloss = vloss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    del opt


def _grids_for_mode(heavy: bool, device: torch.device) -> dict:
    if not heavy:
        return {
            "gammas": [0.0, 1.0, 2.0, 3.0, 4.0],
            "deltas": [0.0, 1.0, 2.5],
            "epsilons": [0.0, 0.75, 2.0],
            "priors_mh": [0.0, 0.03, 0.08, 0.12],
            "priors_lp": [0.0, 0.02, 0.05, 0.10, 0.15],
            "lam_add": torch.linspace(0, 6.0, 31, device=device),
            "w_cvx": torch.linspace(0, 1, 21, device=device),
            "etas": (
                0.0,
                0.25,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ),
            "temps": (0.5, 0.65, 0.8, 0.95, 1.0, 1.1, 1.25, 1.4, 1.6, 1.85, 2.1, 2.5),
            "alphas_step": 0.05,
            "taus_extra": (),
            "mcs_extra": (),
        }
    g = [round(i * 0.25, 2) for i in range(0, 19)]
    d = [round(i * 0.25, 2) for i in range(0, 13)]
    e = [round(i * 0.25, 2) for i in range(0, 11)]
    pmh = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22]
    plp = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22]
    return {
        "gammas": g,
        "deltas": d,
        "epsilons": e,
        "priors_mh": pmh,
        "priors_lp": plp,
        "lam_add": torch.linspace(0, 7.0, 43, device=device),
        "w_cvx": torch.linspace(0, 1, 41, device=device),
        "etas": tuple(round(i * 0.25, 2) for i in range(0, 41)),
        "temps": tuple(
            round(float(t), 3) for t in torch.linspace(0.4, 2.8, 22).tolist()
        ),
        "alphas_step": 0.025,
        "taus_extra": (0.25, 0.35, 1.5, 7.0, 9.0, 11.0, 14.0),
        "mcs_extra": (0.06, 0.07, 0.09, 0.11, 0.13, 0.48, 0.52, 0.58, 0.62),
    }


# ─── Legacy architecture (matches checkpoint keys: conv1..3, fusion, classifier) ─────────


class LegacyFacebookGNN(nn.Module):
    """Same layout as older project training (GraphSAGE ×3 + fusion + linear classifier)."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.fusion = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        h = F.relu(self.fusion(h))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, edge_index))


def load_musae_facebook(
    data_dir: Path,
    in_channels_required: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Returns x, edge_index, y, labeled_mask, num_classes (from CSV types)."""
    edges_path = data_dir / "musae_facebook_edges.csv"
    target_path = data_dir / "musae_facebook_target.csv"
    features_path = data_dir / "musae_facebook_features.json"

    if not edges_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Need {edges_path} and {target_path}")

    edges_src, edges_dst = [], []
    with open(edges_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges_src.append(int(row["id_1"]))
            edges_dst.append(int(row["id_2"]))

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)

    types: list[str] = []
    rows: list[tuple[int, str]] = []
    with open(target_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["id"])
            ptype = row.get("page_type", "unknown")
            rows.append((pid, ptype))
            types.append(ptype)

    unique_types = sorted(set(types))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    num_classes = len(unique_types)

    max_id = max(edge_index.max().item(), max(pid for pid, _ in rows))
    num_nodes = max_id + 1

    y = torch.zeros(num_nodes, dtype=torch.long)
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for pid, ptype in rows:
        y[pid] = type_to_idx[ptype]
        labeled_mask[pid] = True

    if features_path.exists():
        with open(features_path, encoding="utf-8") as f:
            feat_dict = json.load(f)
        max_feat = 0
        for feats in feat_dict.values():
            if feats:
                max_feat = max(max_feat, max(feats))
        feat_width = min(max_feat + 1, FEATURE_DIM_CAP)
        x = torch.zeros(num_nodes, feat_width)
        for node_id_str, feats in feat_dict.items():
            nid = int(node_id_str)
            if nid >= num_nodes:
                continue
            for fi in feats or []:
                if fi < feat_width:
                    x[nid, fi] = 1.0
        if feat_width < in_channels_required:
            x = torch.cat([x, torch.zeros(num_nodes, in_channels_required - feat_width)], dim=1)
        elif x.size(1) > in_channels_required:
            x = x[:, :in_channels_required]
    else:
        from model.utils import build_structural_features

        x = build_structural_features(edge_index, num_nodes, in_channels_required)

    assert x.size(1) == in_channels_required
    return x, edge_index, y, labeled_mask, num_classes


def stratified_train_test_masks(
    y: torch.Tensor,
    labeled_mask: torch.Tensor,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    labeled_idx = torch.where(labeled_mask)[0].numpy()
    y_l = y[labeled_idx].numpy()
    try:
        train_idx, test_idx, _, _ = train_test_split(
            labeled_idx,
            y_l,
            test_size=test_size,
            random_state=seed,
            stratify=y_l,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            labeled_idx,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    train_mask[torch.tensor(train_idx)] = True
    test_mask[torch.tensor(test_idx)] = True
    return train_mask, test_mask


def train_neighbor_histogram_fast(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    source_mask: torch.Tensor,
    num_nodes: int,
    num_classes: int,
) -> torch.Tensor:
    """1-hop histogram of **train-labeled** neighbor class counts (vectorized)."""
    src, dst = edge_index[0], edge_index[1]
    hist = torch.zeros(num_nodes, num_classes, dtype=torch.float32)
    oh = F.one_hot(y.clamp(min=0, max=num_classes - 1), num_classes).float()

    m = source_mask[dst]
    if m.any():
        hist.index_add_(0, src[m], oh[dst[m]])
    m2 = source_mask[src]
    if m2.any():
        hist.index_add_(0, dst[m2], oh[src[m2]])
    return hist


def row_normalized_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Symmetric row-normalized sparse adjacency (for graph propagation)."""
    row, col = edge_index[0], edge_index[1]
    deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1.0)
    vals = 1.0 / deg[row]
    return torch.sparse_coo_tensor(
        torch.stack([row, col]),
        vals,
        (num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()


def precompute_train_propagation_bases(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    num_nodes: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fixed **train-only** seeds M; row-normalized A. Precompute h1=A@M … h4=A@h3 once.
    Any combo h1 + γ h2 + δ h3 + ε h4 + prior is cheap (no extra sparse-mm in the grid).
    """
    A = row_normalized_adj(edge_index, num_nodes)
    M = torch.zeros(num_nodes, num_classes, dtype=torch.float32)
    tm = train_mask.nonzero(as_tuple=True)[0]
    M[tm] = F.one_hot(y[tm].clamp(min=0, max=num_classes - 1), num_classes).float()
    h1 = torch.sparse.mm(A, M)
    h2 = torch.sparse.mm(A, h1)
    h3 = torch.sparse.mm(A, h2)
    h4 = torch.sparse.mm(A, h3)
    counts = torch.bincount(y[train_mask], minlength=num_classes).float()
    freq = counts / counts.sum().clamp(min=1.0)
    return h1, h2, h3, h4, freq


def combine_propagation_histogram(
    h1: torch.Tensor,
    h2: torch.Tensor,
    h3: torch.Tensor,
    h4: torch.Tensor,
    train_class_freq: torch.Tensor,
    gamma: float,
    delta: float,
    epsilon: float,
    prior_strength: float,
) -> torch.Tensor:
    """h1 + γ·h2 + δ·h3 + ε·h4 + prior_strength * class frequency (Dirichlet smoothing)."""
    prior = prior_strength * train_class_freq.unsqueeze(0).expand_as(h1)
    return h1 + gamma * h2 + delta * h3 + epsilon * h4 + prior


def add_class_prior(h: torch.Tensor, train_class_freq: torch.Tensor, prior_strength: float) -> torch.Tensor:
    if prior_strength <= 0:
        return h
    return h + prior_strength * train_class_freq.unsqueeze(0).expand_as(h)


LP_SNAPSHOT_STEPS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)
LP_SNAPSHOT_STEPS_HEAVY = tuple(
    sorted(set(LP_SNAPSHOT_STEPS) | {30, 32, 34, 36, 38, 40})
)


def reseed_lp_snapshots(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    num_nodes: int,
    num_classes: int,
    steps: tuple[int, ...] | None = None,
) -> dict[int, torch.Tensor]:
    """
    Iterative label propagation: ``F <- A @ F`` then re-impose **train** one-hot labels.
    One pass up to ``max(steps)`` sparse MMs; returns F at each listed step.
    """
    step_t = steps if steps is not None else LP_SNAPSHOT_STEPS
    want = frozenset(step_t)
    mx = max(step_t)
    A = row_normalized_adj(edge_index, num_nodes)
    Y0 = torch.zeros(num_nodes, num_classes, dtype=torch.float32, device=y.device)
    tm = train_mask.nonzero(as_tuple=True)[0]
    Y0[tm] = F.one_hot(y[tm].clamp(min=0, max=num_classes - 1), num_classes).float()
    prop = Y0.clone()
    snaps: dict[int, torch.Tensor] = {}
    for t in range(1, mx + 1):
        prop = torch.sparse.mm(A, prop)
        prop[tm] = Y0[tm]
        if t in want:
            snaps[t] = prop.clone()
    return snaps


@torch.no_grad()
def accuracy_on_mask(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return 0.0
    correct = (pred[mask] == y[mask]).sum().item()
    return correct / int(mask.sum().item())


def pipeline_predict(
    gnn_logits: torch.Tensor,
    h_graph: torch.Tensor,
    alpha: float,
    tau_votes: float,
    min_conf: float,
) -> torch.Tensor:
    """
    GNN + graph RAG with an extra **homophily gate**: when enough train-labeled neighbors
    agree (mass + confidence), trust the retrieved label histogram (Neo4j-style neighborhood).
    Hyperparameters tuned on validation only.
    """
    dev = gnn_logits.device
    return pipeline_predict_batch_atm(
        gnn_logits,
        h_graph,
        torch.tensor([alpha], dtype=torch.float32, device=dev),
        torch.tensor([tau_votes], dtype=torch.float32, device=dev),
        torch.tensor([min_conf], dtype=torch.float32, device=dev),
    )[0, 0, 0, :]


def pipeline_predict_batch_atm(
    gnn_logits: torch.Tensor,
    h_graph: torch.Tensor,
    alphas: torch.Tensor,
    taus: torch.Tensor,
    mcs: torch.Tensor,
    node_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Vectorized pipeline predictions: ``alphas`` [K], ``taus`` [T], ``mcs`` [M].

    If ``node_idx`` is ``None``, returns ``[K, T, M, N]`` for all nodes.
    If ``node_idx`` is ``[Nv]`` long indices, returns ``[K, T, M, Nv]`` (faster grid search).
    """
    n_full, c = gnn_logits.shape
    if node_idx is None:
        n = n_full
        idx = slice(None)
    else:
        n = int(node_idx.numel())
        idx = node_idx

    k, tt, mm = int(alphas.numel()), int(taus.numel()), int(mcs.numel())
    dtype = gnn_logits.dtype
    dev = gnn_logits.device

    p_gnn = F.softmax(gnn_logits, dim=-1)[idx]
    row = h_graph.sum(dim=1, keepdim=True).clamp(min=1e-6)
    p1 = (h_graph / row)[idx]
    mass = h_graph.sum(dim=1)[idx]
    conf = h_graph.max(dim=1).values[idx] / mass.clamp(min=1e-6)
    pure = h_graph.argmax(dim=-1)[idx]
    no_nei = mass < 1e-6
    gna = gnn_logits.argmax(dim=-1)[idx]

    a = alphas.to(device=dev, dtype=dtype).view(k, 1, 1, 1, 1)
    pge = p_gnn.view(1, 1, 1, n, c).expand(k, 1, 1, n, c)
    p1e = p1.view(1, 1, 1, n, c).expand(k, 1, 1, n, c)
    mixed = (1.0 - a) * pge + a * p1e
    nn5 = no_nei.view(1, 1, 1, n, 1).expand(k, 1, 1, n, 1)
    mixed = torch.where(nn5, p_gnn.view(1, 1, 1, n, c).expand(k, 1, 1, n, c), mixed)
    mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    base = mixed.argmax(dim=-1).squeeze(1).squeeze(1)

    taus = taus.to(device=dev, dtype=dtype)
    mcs = mcs.to(device=dev, dtype=dtype)
    mass_ge = mass.unsqueeze(0) >= taus.unsqueeze(1)
    conf_ge = conf.unsqueeze(0) >= mcs.unsqueeze(1)
    use_graph = mass_ge.view(tt, 1, n) & conf_ge.view(1, mm, n)

    out = base.view(k, 1, 1, n).expand(k, tt, mm, n).clone()
    pure_exp = pure.view(1, 1, 1, n).expand(k, tt, mm, n)
    ug = use_graph.view(1, tt, mm, n).expand(k, tt, mm, n)
    out = torch.where(ug, pure_exp, out)
    nn_exp = no_nei.view(1, 1, 1, n).expand(k, tt, mm, n)
    gna_exp = gna.view(1, 1, 1, n).expand(k, tt, mm, n)
    out = torch.where(nn_exp, gna_exp, out)
    return out


def _load_model_for_checkpoint(state: dict) -> tuple[nn.Module, int]:
    """Return (model, num_classes) for either legacy or SocialGraphGNN checkpoints."""
    if "conv1.lin_l.weight" in state:
        in_ch = int(state["conv1.lin_l.weight"].shape[1])
        hidden = int(state["conv1.lin_l.weight"].shape[0])
        n_cls = int(state["classifier.weight"].shape[0])
        m = LegacyFacebookGNN(in_ch, hidden, n_cls)
        m.load_state_dict(state, strict=True)
        return m, n_cls
    if "encoder.convs.0.lin_l.weight" in state:
        n_cls = int(state["node_classifier.fc2.weight"].shape[0])
        m = SocialGraphGNN(
            in_channels=int(state["encoder.convs.0.lin_l.weight"].shape[1]),
            hidden_channels=int(state["encoder.convs.0.lin_l.weight"].shape[0]),
            embedding_dim=int(state["encoder.convs.2.lin_l.weight"].shape[0]),
            num_classes=n_cls,
            num_layers=3,
            dropout=DROPOUT,
        )
        m.load_state_dict(state, strict=True)
        return m, n_cls
    raise ValueError("Unrecognized checkpoint format")


@pytest.mark.integration
def test_facebook_pipeline_beats_gnn_stratified_target():
    data_dir = _facebook_dir()
    if not data_dir.is_dir():
        pytest.skip(f"Facebook data directory missing: {data_dir}")
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"No weights at {WEIGHTS_PATH}")

    try:
        try:
            state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(WEIGHTS_PATH, map_location="cpu")
    except Exception as e:
        pytest.skip(f"Could not load weights: {e}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Infer input dim from weights before building x
    if "conv1.lin_l.weight" in state:
        in_ch = int(state["conv1.lin_l.weight"].shape[1])
    elif "encoder.convs.0.lin_l.weight" in state:
        in_ch = int(state["encoder.convs.0.lin_l.weight"].shape[1])
    else:
        pytest.skip("Checkpoint missing expected first-layer keys")

    x, edge_index, y, labeled_mask, num_classes_data = load_musae_facebook(data_dir, in_ch)
    model, num_classes_ckpt = _load_model_for_checkpoint(state)
    if num_classes_ckpt != num_classes_data:
        pytest.skip(
            f"Checkpoint num_classes={num_classes_ckpt} vs data={num_classes_data} — label space mismatch"
        )

    num_classes = num_classes_ckpt
    model.eval()

    train_mask, test_mask = stratified_train_test_masks(y, labeled_mask)

    train_idx = torch.where(train_mask)[0].numpy()
    y_tr = y[train_mask].numpy()
    try:
        tr_sub, val_sub, _, _ = train_test_split(
            train_idx,
            y_tr,
            test_size=VAL_FRACTION_OF_TRAIN,
            random_state=SEED + 1,
            stratify=y_tr,
        )
    except ValueError:
        tr_sub, val_sub = train_test_split(
            train_idx,
            test_size=VAL_FRACTION_OF_TRAIN,
            random_state=SEED + 1,
        )
    val_mask = torch.zeros_like(train_mask)
    train_fit_mask = torch.zeros_like(train_mask)
    val_mask[torch.tensor(val_sub)] = True
    train_fit_mask[torch.tensor(tr_sub)] = True

    t0 = time.time()
    heavy = _use_heavy_pipeline_search()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    val_mask = val_mask.to(device)
    train_fit_mask = train_fit_mask.to(device)
    model = model.to(device)

    logger.info(
        "Pipeline accuracy: heavy_search=%s device=%s (set PIPELINE_ACCURACY_HEAVY=1 or "
        "PIPELINE_ACCURACY_FAST=1 to override)",
        heavy,
        device,
    )

    if heavy:
        ft_epochs = int(os.getenv("PIPELINE_FINETUNE_EPOCHS", "120"))
        ft_lr = float(os.getenv("PIPELINE_FINETUNE_LR", "5e-4"))
        logger.info(
            "Fine-tuning GNN on train_fit (%d epochs max, lr=%s, early-stop on val CE)...",
            ft_epochs,
            ft_lr,
        )
        ft_t0 = time.time()
        finetune_node_classifier_transductive(
            model,
            x,
            edge_index,
            y,
            train_fit_mask,
            val_mask,
            max_epochs=ft_epochs,
            lr=ft_lr,
        )
        logger.info("Fine-tune done in %.1fs", time.time() - ft_t0)

    num_nodes = x.size(0)
    with torch.no_grad():
        if isinstance(model, LegacyFacebookGNN):
            logits = model(x, edge_index)
        else:
            z = model.encode(x, edge_index)
            logits = model.classify_node(z)

    pred_gnn = logits.argmax(dim=-1)
    acc_gnn_test = accuracy_on_mask(pred_gnn, y, test_mask)
    logger.info("GNN-only TEST accuracy: %.4f (device=%s)", acc_gnn_test, device)

    lp_steps = LP_SNAPSHOT_STEPS_HEAVY if heavy else LP_SNAPSHOT_STEPS
    h1_fit, h2_fit, h3_fit, h4_fit, freq_fit = precompute_train_propagation_bases(
        edge_index, y, train_fit_mask, num_nodes, num_classes
    )
    h1_full, h2_full, h3_full, h4_full, freq_full = precompute_train_propagation_bases(
        edge_index, y, train_mask, num_nodes, num_classes
    )
    lp_fit = reseed_lp_snapshots(
        edge_index, y, train_fit_mask, num_nodes, num_classes, steps=lp_steps
    )
    lp_full = reseed_lp_snapshots(
        edge_index, y, train_mask, num_nodes, num_classes, steps=lp_steps
    )
    hist1_fit = train_neighbor_histogram_fast(
        edge_index, y, train_fit_mask, num_nodes, num_classes
    ).to(dtype=torch.float32)
    hist1_full = train_neighbor_histogram_fast(
        edge_index, y, train_mask, num_nodes, num_classes
    ).to(dtype=torch.float32)

    grids = _grids_for_mode(heavy, device)
    gammas = grids["gammas"]
    deltas = grids["deltas"]
    epsilons = grids["epsilons"]
    priors_mh = grids["priors_mh"]
    priors_lp = grids["priors_lp"]

    st = float(grids["alphas_step"])
    n_alpha = int(round(1.0 / st)) + 1
    alphas_t = torch.linspace(0, 1, n_alpha, device=device, dtype=torch.float32)

    base_taus = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    base_mcs = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    taus_t = torch.tensor(
        sorted(set(base_taus + list(grids["taus_extra"]))),
        dtype=torch.float32,
        device=device,
    )
    mcs_t = torch.tensor(
        sorted(set(base_mcs + list(grids["mcs_extra"]))),
        dtype=torch.float32,
        device=device,
    )
    val_idx = val_mask.nonzero(as_tuple=True)[0]
    y_val = y[val_idx]
    nt = int(taus_t.numel())
    nm = int(mcs_t.numel())

    def eval_on_val(
        hg: torch.Tensor,
        logits_scaled: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float]:
        lg = logits if logits_scaled is None else logits_scaled
        preds = pipeline_predict_batch_atm(
            lg, hg, alphas_t, taus_t, mcs_t, node_idx=val_idx
        )
        vaccs = (preds == y_val.view(1, 1, 1, -1)).to(dtype=torch.float32).mean(dim=-1)
        flat_idx = int(vaccs.argmax().item())
        va = float(vaccs.view(-1)[flat_idx].item())
        ki = flat_idx // (nt * nm)
        rem = flat_idx % (nt * nm)
        ti = rem // nm
        mi = rem % nm
        return va, float(alphas_t[ki].item()), float(taus_t[ti].item()), float(mcs_t[mi].item())

    mh_best_val = -1.0
    mh_st: dict[str, float] = {
        "a": 0.35,
        "tau": 3.0,
        "mc": 0.35,
        "g": 1.0,
        "d": 0.0,
        "e": 0.0,
        "ps": 0.05,
    }
    for gam in gammas:
        for delta in deltas:
            for eps in epsilons:
                for ps in priors_mh:
                    h_fit = combine_propagation_histogram(
                        h1_fit, h2_fit, h3_fit, h4_fit, freq_fit, gam, delta, eps, ps
                    )
                    va, ba, bt, bm = eval_on_val(h_fit)
                    if va > mh_best_val:
                        mh_best_val = va
                        mh_st.update(
                            a=ba, tau=bt, mc=bm, g=gam, d=delta, e=eps, ps=float(ps)
                        )

    lp_best_val = -1.0
    lp_st: dict[str, float | int] = {"a": 0.35, "tau": 3.0, "mc": 0.35, "step": 12, "ps": 0.05}
    for step in lp_steps:
        for ps in priors_lp:
            h_fit = add_class_prior(lp_fit[step], freq_fit, ps)
            va, ba, bt, bm = eval_on_val(h_fit)
            if va > lp_best_val:
                lp_best_val = va
                lp_st.update(a=ba, tau=bt, mc=bm, step=step, ps=float(ps))

    h_mh_fit = combine_propagation_histogram(
        h1_fit,
        h2_fit,
        h3_fit,
        h4_fit,
        freq_fit,
        float(mh_st["g"]),
        float(mh_st["d"]),
        float(mh_st["e"]),
        float(mh_st["ps"]),
    )
    h_lp_fit = add_class_prior(
        lp_fit[int(lp_st["step"])], freq_fit, float(lp_st["ps"])
    )
    mix_best_val = -1.0
    mix_st: dict[str, float | str] = {
        "a": 0.35,
        "tau": 3.0,
        "mc": 0.35,
        "lam": 0.0,
        "kind": "add",
    }
    for lam in grids["lam_add"]:
        va, ba, bt, bm = eval_on_val(h_mh_fit + float(lam.item()) * h_lp_fit)
        if va > mix_best_val:
            mix_best_val = va
            mix_st.update(a=ba, tau=bt, mc=bm, lam=float(lam.item()), kind="add")
    for w in grids["w_cvx"]:
        wf = float(w.item())
        va, ba, bt, bm = eval_on_val((1.0 - wf) * h_mh_fit + wf * h_lp_fit)
        if va > mix_best_val:
            mix_best_val = va
            mix_st.update(a=ba, tau=bt, mc=bm, lam=wf, kind="cvx")

    def build_mh_full() -> torch.Tensor:
        return combine_propagation_histogram(
            h1_full,
            h2_full,
            h3_full,
            h4_full,
            freq_full,
            float(mh_st["g"]),
            float(mh_st["d"]),
            float(mh_st["e"]),
            float(mh_st["ps"]),
        )

    def build_lp_full() -> torch.Tensor:
        return add_class_prior(
            lp_full[int(lp_st["step"])], freq_full, float(lp_st["ps"])
        )

    lam_star = float(mix_st["lam"]) if mix_st["kind"] == "add" else 0.0
    w_star = float(mix_st["lam"]) if mix_st["kind"] == "cvx" else 0.0

    graph_candidates: list[tuple[str, torch.Tensor, object]] = [
        ("mh", h_mh_fit, build_mh_full),
        ("lp", h_lp_fit, build_lp_full),
        (
            "add",
            h_mh_fit + lam_star * h_lp_fit,
            lambda: build_mh_full() + lam_star * build_lp_full(),
        ),
        (
            "cvx",
            (1.0 - w_star) * h_mh_fit + w_star * h_lp_fit,
            lambda: (1.0 - w_star) * build_mh_full() + w_star * build_lp_full(),
        ),
        (
            "mean",
            0.5 * (h_mh_fit + h_lp_fit),
            lambda: 0.5 * (build_mh_full() + build_lp_full()),
        ),
    ]

    best_val = -1.0
    best_a, best_tau, best_mc = 0.35, 3.0, 0.35
    best_eta = 0.0
    best_tag = "mh"
    best_builder = build_mh_full
    best_h_f = h_mh_fit
    for tag, h_f, builder in graph_candidates:
        for eta in grids["etas"]:
            va, ba, bt, bm = eval_on_val(h_f + float(eta) * hist1_fit)
            if va > best_val:
                best_val = va
                best_a, best_tau, best_mc = ba, bt, bm
                best_eta = float(eta)
                best_tag = tag
                best_builder = builder
                best_h_f = h_f

    h_combo_fit = best_h_f + best_eta * hist1_fit
    best_temp = 1.0
    for temp in grids["temps"]:
        va, ba, bt, bm = eval_on_val(h_combo_fit, logits_scaled=logits / float(temp))
        if va > best_val:
            best_val = va
            best_a, best_tau, best_mc = ba, bt, bm
            best_temp = float(temp)

    h_full = best_builder() + best_eta * hist1_full

    logger.info(
        "Validation best acc=%.4f | graph=%s eta_hist=%.2f logit_temp=%.3f "
        "alpha=%.2f tau=%.2f min_conf=%.2f",
        best_val,
        best_tag,
        best_eta,
        best_temp,
        best_a,
        best_tau,
        best_mc,
    )

    pred_pipe = pipeline_predict(logits / best_temp, h_full, best_a, best_tau, best_mc)
    acc_pipe_test = accuracy_on_mask(pred_pipe, y, test_mask)

    logger.info("Pipeline (GNN + graph-RAG) TEST accuracy: %.4f", acc_pipe_test)

    elapsed = time.time() - t0
    logger.info("Total wall time: %.1fs (heavy=%s)", elapsed, heavy)
    kaggle_out = Path("/kaggle/working") / "pipeline_accuracy_results.json"
    if kaggle_out.parent.is_dir():
        kaggle_out.write_text(
            json.dumps(
                {
                    "heavy_search": heavy,
                    "device": str(device),
                    "acc_gnn_test": acc_gnn_test,
                    "acc_pipeline_test": acc_pipe_test,
                    "best_val_tuning": best_val,
                    "best_graph_tag": best_tag,
                    "best_eta_hist": best_eta,
                    "best_logit_temp": best_temp,
                    "best_alpha": best_a,
                    "best_tau": best_tau,
                    "best_min_conf": best_mc,
                    "seconds_wall": elapsed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote %s", kaggle_out)

    assert acc_pipe_test >= acc_gnn_test - 1e-6, "Pipeline should be >= GNN-only (same split)"
    assert acc_pipe_test > _MIN_STRATIFIED_PIPELINE_ACC, (
        f"Pipeline stratified test accuracy too low ({acc_pipe_test:.4f}); "
        f"GNN-only was {acc_gnn_test:.4f}. Retrain or verify data/facebook matches weights."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
