"""
db/ingest/ingest_all.py
========================
R2: Idempotent bulk ingest of Facebook, Twitter, Reddit datasets into Neo4j.

Architecture:
  - Each dataset has a parser that yields rows as Python dicts
  - A shared `_batch_upsert` function does UNWIND + MERGE in Neo4j
  - Ingest is gated by a marker file on disk (skipped if already done)
  - FORCE_REINGEST=true bypasses the gate
  - Nodes get `dataset` and `ingest_version` properties for filtering
  - Labels are normalized to User / Post + FRIEND / POSTED / LIKED
    so GraphRetriever Cypher works unchanged

Schema additions (beyond existing):
  - User.dataset  (str) — "facebook"|"twitter"|"reddit"|"demo"
  - Post.dataset  (str) — same
  - CREATE INDEX user_dataset IF NOT EXISTS FOR (u:User) ON (u.dataset)
  - CREATE INDEX post_dataset IF NOT EXISTS FOR (p:Post) ON (p.dataset)
  - Composite unique constraint: (dataset + source_id) via MERGE key
"""

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from api.bootstrap.config import (
    ALL_DATASETS,
    FORCE_REINGEST,
    INGEST_BATCH_SIZE,
    DatasetManifest,
)


# ── Data file resolution (data/facebook, data/twitter, data/reddit) ─────────

def _resolve_twitter_edge_file(manifest: DatasetManifest) -> Optional[Path]:
    """Find twitter edge list: env TWITTER_COMBINED_PATH, then data/twitter/*."""
    explicit = os.getenv("TWITTER_COMBINED_PATH", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    d = manifest.dir
    for name in ("twitter_combined.txt", "twitter_combined", "twitter_edges.txt", "edges.txt"):
        p = d / name
        if p.is_file():
            return p
    for p in sorted(d.glob("*.txt")):
        if p.is_file() and not p.name.startswith(".") and p.stat().st_size > 0:
            logger.info("Twitter: using %s (first non-empty .txt in %s)", p.name, d)
            return p
    return None


def _resolve_reddit_tsv(manifest: DatasetManifest) -> Optional[Path]:
    explicit = os.getenv("REDDIT_TSV_PATH", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    d = manifest.dir
    for name in (
        "soc-redditHyperlinks-title.tsv",
        "soc-redditHyperlinks-title.txt",
        "reddit_hyperlinks.tsv",
    ):
        p = d / name
        if p.is_file():
            return p
    for p in d.glob("*.tsv"):
        if p.is_file() and p.stat().st_size > 0:
            logger.info("Reddit: using %s (first .tsv in %s)", p.name, d)
            return p
    return None


def dataset_data_files_ready(manifest: DatasetManifest) -> bool:
    """True if the expected on-disk files exist so a real Neo4j ingest can run."""
    d = manifest.dir
    if manifest.name == "facebook":
        edges = d / "musae_facebook_edges.csv"
        target = d / "musae_facebook_target.csv"
        if edges.is_file() and target.is_file():
            return True
        # edges alone are enough to derive users (see _parse_facebook_nodes)
        return edges.is_file()
    if manifest.name == "twitter":
        return _resolve_twitter_edge_file(manifest) is not None
    if manifest.name == "reddit":
        return _resolve_reddit_tsv(manifest) is not None
    return manifest.all_required_present()


def _ingest_had_data(dataset_name: str, result: Dict[str, Any]) -> bool:
    if result.get("ingest_stub"):
        return False
    if result.get("ok") is False:
        return False
    u = int(result.get("users", 0) or 0)
    p = int(result.get("posts", 0) or 0)
    e = int(result.get("edges", 0) or 0)
    if dataset_name == "reddit":
        return (u + p) > 0 and e >= 0
    if dataset_name == "facebook":
        return u > 0
    if dataset_name == "twitter":
        return u > 0 and e > 0
    return u > 0

# ── Schema extensions ─────────────────────────────────────────────────────────

DATASET_SCHEMA_QUERIES = [
    # Dataset-scoped composite uniqueness (dataset + id avoids cross-dataset collisions)
    "CREATE CONSTRAINT user_dataset_id IF NOT EXISTS FOR (u:User) REQUIRE (u.dataset, u.source_id) IS UNIQUE",
    "CREATE CONSTRAINT post_dataset_id IF NOT EXISTS FOR (p:Post) REQUIRE (p.dataset, p.source_id) IS UNIQUE",

    # Lookup indexes for dataset filtering
    "CREATE INDEX user_dataset IF NOT EXISTS FOR (u:User) ON (u.dataset)",
    "CREATE INDEX post_dataset IF NOT EXISTS FOR (p:Post) ON (p.dataset)",
    "CREATE INDEX user_source_id IF NOT EXISTS FOR (u:User) ON (u.source_id)",
]

# ── UNWIND batch helpers ──────────────────────────────────────────────────────

def _run_batch(neo4j_client, cypher: str, batch: List[Dict], description: str = ""):
    """Execute a batched UNWIND query. Logs counters."""
    if not batch:
        return
    try:
        neo4j_client.run_write_query(cypher, {"batch": batch})
    except Exception as e:
        logger.error(f"Batch write failed [{description}]: {e}")


def _chunked(lst: List, size: int) -> Generator:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ── Facebook ingest ───────────────────────────────────────────────────────────

def _parse_facebook_nodes(manifest: DatasetManifest) -> Tuple[List[Dict], bool]:
    """Parse musae_facebook_target.csv, or derive user ids from musae_facebook_edges.csv, or tiny stub."""
    target_path = manifest.dir / "musae_facebook_target.csv"
    edges_path = manifest.dir / "musae_facebook_edges.csv"
    page_type_influence = {"politician": 0.8, "government": 0.7, "tvshow": 0.6, "company": 0.5}

    if target_path.exists():
        nodes: List[Dict] = []
        with open(target_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("id", row.get("Id", ""))
                ptype = row.get("page_type", "unknown")
                pname = row.get("page_name", row.get("name", f"Page_{pid}"))
                nodes.append({
                    "source_id":      str(pid),
                    "name":           pname,
                    "page_type":      ptype,
                    "bio":            f"Facebook {ptype}: {pname}",
                    "follower_count": 0,
                    "influence_score": page_type_influence.get(ptype, 0.3),
                })
        return nodes, False

    if edges_path.exists():
        ids: Set[str] = set()
        with open(edges_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.add(str(row["id_1"]))
                ids.add(str(row["id_2"]))
        nodes = [
            {
                "source_id":      pid,
                "name":           f"Page_{pid}",
                "page_type":      "unknown",
                "bio":            f"Facebook page {pid} (from edge list; add target CSV for names)",
                "follower_count": 0,
                "influence_score": 0.3,
            }
            for pid in sorted(ids, key=lambda x: int(x) if str(x).isdigit() else x)
        ]
        logger.info("Facebook: built %d user nodes from edge list (no target CSV)", len(nodes))
        return nodes, False

    logger.warning("Facebook: no target or edges CSV under %s — cannot ingest", manifest.dir)
    return [], True


def _parse_facebook_edges(manifest: DatasetManifest) -> List[Dict]:
    """Parse musae_facebook_edges.csv → list of (id_1, id_2) dicts."""
    edges_path = manifest.dir / "musae_facebook_edges.csv"
    if not edges_path.exists():
        return []
    edges = []
    with open(edges_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append({"src": str(row["id_1"]), "dst": str(row["id_2"])})
    return edges


def ingest_facebook(neo4j_client, manifest: DatasetManifest, batch_size: int = INGEST_BATCH_SIZE):
    logger.info("Facebook: parsing nodes...")
    nodes, ingest_stub = _parse_facebook_nodes(manifest)
    logger.info(f"Facebook: {len(nodes)} pages found (stub={ingest_stub})")
    if ingest_stub:
        logger.error(
            "Facebook: refusing to load 10 stub nodes into Neo4j — add musae_facebook_edges.csv "
            "(and ideally musae_facebook_target.csv) under %s",
            manifest.dir,
        )
        return {"users": 0, "edges": 0, "ingest_stub": True, "ok": False, "error": "no_real_facebook_files"}

    # Upsert users
    upsert_user_cypher = """
    UNWIND $batch AS item
    MERGE (u:User {dataset: 'facebook', source_id: item.source_id})
    SET u.id            = 'fb_' + item.source_id,
        u.name          = item.name,
        u.bio           = item.bio,
        u.page_type     = item.page_type,
        u.follower_count = item.follower_count,
        u.influence_score = item.influence_score,
        u.dataset       = 'facebook',
        u.source_id     = item.source_id,
        u.created_at    = datetime().epochSeconds
    """
    total_nodes = 0
    for chunk in _chunked(nodes, batch_size):
        _run_batch(neo4j_client, upsert_user_cypher, chunk, "facebook_nodes")
        total_nodes += len(chunk)
    logger.info(f"Facebook: upserted {total_nodes} users")

    # Upsert edges (FRIEND)
    logger.info("Facebook: parsing edges...")
    edges = _parse_facebook_edges(manifest)
    logger.info(f"Facebook: {len(edges)} edges found")

    upsert_edge_cypher = """
    UNWIND $batch AS item
    MATCH (a:User {dataset: 'facebook', source_id: item.src})
    MATCH (b:User {dataset: 'facebook', source_id: item.dst})
    MERGE (a)-[:FRIEND]->(b)
    MERGE (b)-[:FRIEND]->(a)
    """
    total_edges = 0
    for chunk in _chunked(edges, batch_size):
        _run_batch(neo4j_client, upsert_edge_cypher, chunk, "facebook_edges")
        total_edges += len(chunk)

    logger.info(f"Facebook: ingest complete — {total_nodes} users, {total_edges} edges")
    return {"users": total_nodes, "edges": total_edges, "ingest_stub": ingest_stub}


# ── Twitter ingest ────────────────────────────────────────────────────────────

def _parse_twitter_edges(manifest: DatasetManifest, max_edges: int = 50000) -> Tuple[List[str], List[Dict]]:
    """Parse combined edge list (twitter_combined.txt or first *.txt) → user IDs and directed edges."""
    txt_path = _resolve_twitter_edge_file(manifest)
    if txt_path is None or not txt_path.is_file():
        return [], []

    user_ids = set()
    edges = []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_edges:
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = parts[0], parts[1]
                user_ids.add(src)
                user_ids.add(dst)
                edges.append({"src": src, "dst": dst})

    return list(user_ids), edges


def ingest_twitter(neo4j_client, manifest: DatasetManifest, batch_size: int = INGEST_BATCH_SIZE):
    max_edges = int(os.getenv("TWITTER_MAX_EDGES", "50000"))
    logger.info(f"Twitter: parsing (max_edges={max_edges})...")
    user_ids, edges = _parse_twitter_edges(manifest, max_edges=max_edges)
    logger.info(f"Twitter: {len(user_ids)} unique users, {len(edges)} edges")

    # Upsert users (no metadata in this dataset — structural only)
    nodes = [{"source_id": uid, "name": f"TwitterUser_{uid}"} for uid in user_ids]

    upsert_user_cypher = """
    UNWIND $batch AS item
    MERGE (u:User {dataset: 'twitter', source_id: item.source_id})
    SET u.id            = 'tw_' + item.source_id,
        u.name          = item.name,
        u.bio           = 'Twitter user ' + item.source_id,
        u.follower_count = 0,
        u.influence_score = 0.3,
        u.dataset       = 'twitter',
        u.source_id     = item.source_id,
        u.created_at    = datetime().epochSeconds
    """
    total_nodes = 0
    for chunk in _chunked(nodes, batch_size):
        _run_batch(neo4j_client, upsert_user_cypher, chunk, "twitter_nodes")
        total_nodes += len(chunk)

    upsert_edge_cypher = """
    UNWIND $batch AS item
    MATCH (a:User {dataset: 'twitter', source_id: item.src})
    MATCH (b:User {dataset: 'twitter', source_id: item.dst})
    MERGE (a)-[:FRIEND]->(b)
    """
    total_edges = 0
    for chunk in _chunked(edges, batch_size):
        _run_batch(neo4j_client, upsert_edge_cypher, chunk, "twitter_edges")
        total_edges += len(chunk)

    # Compute degree-based influence score
    logger.info("Twitter: computing influence scores from degree...")
    neo4j_client.run_write_query("""
        MATCH (u:User {dataset: 'twitter'})
        WITH u, size([(u)-[:FRIEND]->() | 1]) AS out_degree
        SET u.follower_count = out_degree,
            u.influence_score = CASE
                WHEN out_degree > 1000 THEN 0.9
                WHEN out_degree > 500  THEN 0.7
                WHEN out_degree > 100  THEN 0.5
                WHEN out_degree > 10   THEN 0.3
                ELSE 0.1
            END
    """)

    logger.info(f"Twitter: ingest complete — {total_nodes} users, {total_edges} edges")
    if total_nodes == 0:
        return {"users": 0, "edges": 0, "ingest_stub": True, "ok": False, "error": "no_edges_parsed"}
    return {"users": total_nodes, "edges": total_edges, "ingest_stub": False}


# ── Reddit ingest ─────────────────────────────────────────────────────────────

def _parse_reddit_edges(manifest: DatasetManifest, max_rows: int = 30000) -> Tuple[List[str], List[Dict], List[Dict]]:
    """Parse Reddit hyperlinks TSV → subreddits + edges + posts."""
    tsv_path = _resolve_reddit_tsv(manifest)
    if tsv_path is None or not tsv_path.is_file():
        return [], [], []

    subreddits = set()
    edges = []
    posts = []

    with open(tsv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            src = row.get("SOURCE_SUBREDDIT", "").strip().lower()
            dst = row.get("TARGET_SUBREDDIT", "").strip().lower()
            post_id = row.get("POST_ID", f"post_{i}").strip()
            sentiment = int(row.get("LINK_SENTIMENT", "0") or "0")
            timestamp = row.get("TIMESTAMP", "").strip()

            if not src or not dst:
                continue

            subreddits.add(src)
            subreddits.add(dst)
            edges.append({"src": src, "dst": dst, "post_id": post_id, "sentiment": sentiment})

            # Each cross-post becomes a Post node
            posts.append({
                "source_id":   post_id,
                "title":       f"Cross-post from r/{src} to r/{dst}",
                "content":     f"Subreddit {src} linked to {dst}",
                "topic":       src,
                "like_count":  max(0, sentiment),
                "comment_count": 0,
                "created_at":  timestamp or "2024-01-01T00:00:00",
                "author_src":  src,
            })

    return list(subreddits), edges, posts


def ingest_reddit(neo4j_client, manifest: DatasetManifest, batch_size: int = INGEST_BATCH_SIZE):
    max_rows = int(os.getenv("REDDIT_MAX_ROWS", "30000"))
    logger.info(f"Reddit: parsing (max_rows={max_rows})...")
    subreddits, edges, posts = _parse_reddit_edges(manifest, max_rows=max_rows)
    logger.info(f"Reddit: {len(subreddits)} subreddits, {len(edges)} edges, {len(posts)} posts")

    # Subreddits → User nodes
    nodes = [{"source_id": sr, "name": f"r/{sr}"} for sr in subreddits]

    upsert_user_cypher = """
    UNWIND $batch AS item
    MERGE (u:User {dataset: 'reddit', source_id: item.source_id})
    SET u.id            = 'rd_' + item.source_id,
        u.name          = item.name,
        u.bio           = 'Reddit subreddit: ' + item.source_id,
        u.follower_count = 0,
        u.influence_score = 0.3,
        u.dataset       = 'reddit',
        u.source_id     = item.source_id,
        u.created_at    = datetime().epochSeconds
    """
    total_users = 0
    for chunk in _chunked(nodes, batch_size):
        _run_batch(neo4j_client, upsert_user_cypher, chunk, "reddit_users")
        total_users += len(chunk)

    # Posts
    upsert_post_cypher = """
    UNWIND $batch AS item
    MERGE (p:Post {dataset: 'reddit', source_id: item.source_id})
    SET p.id           = 'rd_' + item.source_id,
        p.title        = item.title,
        p.content      = item.content,
        p.topic        = item.topic,
        p.like_count   = item.like_count,
        p.comment_count = item.comment_count,
        p.created_at   = item.created_at,
        p.dataset      = 'reddit',
        p.source_id    = item.source_id
    WITH p, item
    MATCH (author:User {dataset: 'reddit', source_id: item.author_src})
    MERGE (author)-[:POSTED]->(p)
    """
    total_posts = 0
    for chunk in _chunked(posts, batch_size):
        _run_batch(neo4j_client, upsert_post_cypher, chunk, "reddit_posts")
        total_posts += len(chunk)

    # Edges (subreddit cross-links → FRIEND)
    upsert_edge_cypher = """
    UNWIND $batch AS item
    MATCH (a:User {dataset: 'reddit', source_id: item.src})
    MATCH (b:User {dataset: 'reddit', source_id: item.dst})
    MERGE (a)-[:FRIEND]->(b)
    """
    total_edges = 0
    for chunk in _chunked(edges, batch_size):
        _run_batch(neo4j_client, upsert_edge_cypher, chunk, "reddit_edges")
        total_edges += len(chunk)

    # Compute influence by out-degree
    neo4j_client.run_write_query("""
        MATCH (u:User {dataset: 'reddit'})
        WITH u, size([(u)-[:FRIEND]->() | 1]) AS out_degree
        SET u.follower_count = out_degree,
            u.influence_score = CASE
                WHEN out_degree > 500 THEN 0.9
                WHEN out_degree > 100 THEN 0.7
                WHEN out_degree > 20  THEN 0.5
                ELSE 0.3
            END
    """)

    logger.info(f"Reddit: ingest complete — {total_users} users, {total_posts} posts, {total_edges} edges")
    if total_users == 0 and total_posts == 0:
        return {
            "users": 0,
            "posts": 0,
            "edges": 0,
            "ingest_stub": True,
            "ok": False,
            "error": "no_reddit_rows_parsed",
        }
    return {"users": total_users, "posts": total_posts, "edges": total_edges, "ingest_stub": False}


# ── Orchestrator ──────────────────────────────────────────────────────────────

DATASET_INGESTORS = {
    "facebook": ingest_facebook,
    "twitter":  ingest_twitter,
    "reddit":   ingest_reddit,
}


def setup_dataset_schema(neo4j_client):
    """Add dataset-scoped indexes and constraints to Neo4j schema."""
    for q in DATASET_SCHEMA_QUERIES:
        try:
            neo4j_client.run_write_query(q)
        except Exception as e:
            logger.warning(f"Schema query skipped ({e}): {q[:60]}")


def get_dataset_counts(neo4j_client, dataset: str) -> Dict[str, int]:
    """Return node/edge counts for a specific dataset."""
    try:
        rows = neo4j_client.run_query(
            "MATCH (u:User {dataset: $ds}) RETURN count(u) AS users",
            {"ds": dataset}
        )
        users = rows[0]["users"] if rows else 0
        rows2 = neo4j_client.run_query(
            "MATCH (p:Post {dataset: $ds}) RETURN count(p) AS posts",
            {"ds": dataset}
        )
        posts = rows2[0]["posts"] if rows2 else 0
        rows3 = neo4j_client.run_query(
            "MATCH (a:User {dataset: $ds})-[:FRIEND]->() RETURN count(*) AS edges",
            {"ds": dataset}
        )
        edges = rows3[0]["edges"] if rows3 else 0
        return {"users": users, "posts": posts, "edges": edges}
    except Exception as e:
        logger.warning(f"Count query failed for {dataset}: {e}")
        return {"users": 0, "posts": 0, "edges": 0}


def ingest_dataset(neo4j_client, dataset_name: str, force: bool = False) -> Dict[str, Any]:
    """
    Ingest a single dataset. Idempotent by default (gated by marker file).
    Set force=True or FORCE_REINGEST=true env to bypass.
    """
    manifest = ALL_DATASETS.get(dataset_name)
    if not manifest:
        return {"dataset": dataset_name, "status": "unknown", "ok": False}

    should_force = force or FORCE_REINGEST

    if manifest.is_ingested() and not should_force:
        counts = get_dataset_counts(neo4j_client, dataset_name)
        # Marker can exist from a past run while DB was wiped or ingest failed silently
        if counts.get("users", 0) == 0 and manifest.all_required_present():
            logger.warning(
                f"Dataset [{dataset_name}]: ingest marker present but Neo4j has 0 users — re-ingesting"
            )
            try:
                manifest.marker_path().unlink(missing_ok=True)
            except OSError:
                pass
        else:
            logger.info(
                f"Dataset [{dataset_name}]: already ingested (use FORCE_REINGEST=true to redo) | {counts}"
            )
            return {
                "dataset": dataset_name,
                "status": "skipped_already_ingested",
                "ok": True,
                "counts": counts,
            }

    if not dataset_data_files_ready(manifest):
        logger.warning(
            "Dataset [%s]: no loadable data files under %s — skipping. "
            "See data/facebook, data/twitter, data/reddit in README.",
            dataset_name,
            manifest.dir,
        )
        return {"dataset": dataset_name, "status": "skipped_files_missing", "ok": False}

    ingestor = DATASET_INGESTORS.get(dataset_name)
    if not ingestor:
        return {"dataset": dataset_name, "status": "no_ingestor", "ok": False}

    logger.info(f"=== Ingesting dataset: {dataset_name} ===")
    t0 = time.time()
    try:
        result = ingestor(neo4j_client, manifest)
        if not _ingest_had_data(dataset_name, result):
            st = "ingest_stub_or_empty" if result.get("ingest_stub") else "ingest_empty"
            logger.error(
                "Dataset [%s]: Neo4j ingest produced no usable data: %s. Not writing .ingest marker. "
                "Add dataset files, then: POST /datasets/ingest?dataset=%s&force=true",
                dataset_name,
                result,
                dataset_name,
            )
            return {
                "dataset": dataset_name,
                "status": st,
                "ok": False,
                "result": result,
            }
        manifest.mark_ingested()
        elapsed = round(time.time() - t0, 1)
        counts = get_dataset_counts(neo4j_client, dataset_name)
        logger.info(f"Dataset [{dataset_name}]: ingest complete in {elapsed}s | {counts}")
        return {
            "dataset":  dataset_name,
            "status":   "ingested",
            "ok":       True,
            "elapsed_s": elapsed,
            "result":   result,
            "counts":   counts,
        }
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        logger.error(f"Dataset [{dataset_name}]: ingest FAILED after {elapsed}s: {e}")
        return {"dataset": dataset_name, "status": f"error: {e}", "ok": False}


def ingest_all_if_needed(neo4j_client, force: bool = False) -> Dict[str, Dict]:
    """
    Ingest all datasets. Skips any that are already ingested.
    Called once at startup after ensure_all_datasets().
    """
    if not neo4j_client.is_connected:
        logger.warning("Neo4j not connected — skipping all dataset ingest")
        return {name: {"status": "skipped_no_neo4j", "ok": False} for name in ALL_DATASETS}

    logger.info("=== Neo4j Dataset Ingest: starting ===")
    setup_dataset_schema(neo4j_client)

    results = {}
    for name in ALL_DATASETS:
        results[name] = ingest_dataset(neo4j_client, name, force=force)

    ok_count = sum(1 for r in results.values() if r.get("ok"))
    logger.info(f"=== Neo4j Dataset Ingest: {ok_count}/{len(ALL_DATASETS)} datasets ok ===")
    return results
