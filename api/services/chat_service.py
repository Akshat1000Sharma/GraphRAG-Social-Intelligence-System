"""
api/services/chat_service.py
=============================
Services for /chat (R4) and /chat/insert (R5).

/chat:
  Wraps the existing MultiAgentPipeline with dataset scoping.
  Passes dataset filter into the graph retrieval context.

/chat/insert:
  LLM-assisted extraction: parses natural language → InsertUserRequest/InsertEdgeRequest
  Then runs parameterized MERGE Cypher (never raw user text in Cypher).
  Rate-limit and node-cap enforced.

Security:
  - All Cypher uses parameterized queries (no string concatenation from user input)
  - insert_user / insert_edge validators run before Cypher
  - ALLOW_CHAT_INSERT env gate
  - CHAT_INSERT_MAX_NODES per-request cap
"""

import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from api.bootstrap.config import (
    ALLOW_CHAT_INSERT,
    CHAT_INSERT_MAX_NODES,
    VALID_DATASET_NAMES,
)
from api.services.connection_path_nl import (
    connection_path_result_row,
    extract_two_user_ids,
    format_connection_path_insight,
    looks_like_connection_path_query,
)
from api.schemas import (
    ChatRequest, ChatResponse,
    InsertUserRequest, InsertEdgeRequest, InsertPostRequest,
    NLInsertRequest, InsertResult,
)

logger = logging.getLogger(__name__)

# ── Dataset filter helpers ─────────────────────────────────────────────────────

def build_dataset_filter(dataset: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Return a Cypher WHERE clause fragment and params dict for dataset scoping.
    dataset="all" or None → no filter.
    """
    if not dataset or dataset == "all":
        return "", {}
    # Validate (belt-and-suspenders — validator already ran on Pydantic)
    allowed = ["facebook", "twitter", "reddit", "demo"]
    if dataset not in allowed:
        return "", {}
    return "AND u.dataset = $dataset_filter", {"dataset_filter": dataset}


def inject_dataset_context(context: Dict[str, Any], dataset: Optional[str]) -> Dict[str, Any]:
    """Merge dataset into the pipeline context dict passed to the agents."""
    updated = dict(context)
    if dataset and dataset != "all":
        updated["dataset"] = dataset
    return updated


# ── Chat handler (R4) ─────────────────────────────────────────────────────────

class ChatService:
    """
    Wraps the multi-agent pipeline for the /chat endpoint.
    Adds dataset scoping, session tracking, and response formatting.
    """

    def __init__(self, pipeline, neo4j_client=None, graph_query_service=None):
        self.pipeline   = pipeline
        self.neo4j      = neo4j_client
        self.graph_query_service = graph_query_service
        self._sessions: Dict[str, List[Dict]] = {}  # session_id → history

    def query(self, req: ChatRequest) -> ChatResponse:
        """
        R4: Run NL question through the pipeline, scoped to a dataset.
        """
        if self.pipeline is None:
            return ChatResponse(
                message=req.message,
                dataset_queried=req.dataset or "all",
                mode=req.mode or "hybrid",
                intent="error",
                insight="Pipeline not initialized.",
                datasets_cited=[],
            )

        # Build context for the pipeline (omit user_id when unset so the analyzer
        # can parse IDs from the message; None/"" would block extraction)
        base_ctx: Dict[str, Any] = {"mode": req.mode}
        if req.user_id and str(req.user_id).strip():
            base_ctx["user_id"] = str(req.user_id).strip()
        context = inject_dataset_context(context=base_ctx, dataset=req.dataset)
        if req.gnn_dataset:
            context["gnn_dataset"] = req.gnn_dataset

        # Two-user shortest path / connection: same as GET /graph/connection-path (no RAG)
        if self.graph_query_service and looks_like_connection_path_query(req.message):
            t0 = time.time()
            user_a, user_b = extract_two_user_ids(req.message)
            if user_a and user_b:
                try:
                    path = self.graph_query_service.get_connection_path(
                        str(user_a).strip(), str(user_b).strip()
                    )
                except Exception as e:
                    logger.warning("get_connection_path from chat: %s", e)
                    path = {
                        "shortest_path": None,
                        "common_friends": [],
                        "common_liked_posts": [],
                    }
                row = connection_path_result_row(
                    str(user_a).strip(), str(user_b).strip(), path
                )
                insight = format_connection_path_insight(row)
                elapsed = (time.time() - t0) * 1000
                resp = ChatResponse(
                    message=req.message,
                    dataset_queried=req.dataset or "all",
                    mode=req.mode or "hybrid",
                    intent="connection_path",
                    results=[row],
                    insight=insight,
                    datasets_cited=([req.dataset] if req.dataset and req.dataset != "all" else []),
                    graph_context_summary=insight,
                    pipeline_timing_ms={
                        "connection_path": round(elapsed, 2),
                        "total": round(elapsed, 2),
                    },
                    session_id=req.session_id,
                    gnn_dataset_used=req.gnn_dataset
                    or (req.dataset if req.dataset in ("facebook", "twitter", "reddit") else "facebook"),
                )
            else:
                resp = ChatResponse(
                    message=req.message,
                    dataset_queried=req.dataset or "all",
                    mode=req.mode or "hybrid",
                    intent="connection_path",
                    results=[],
                    insight=(
                        "I could not extract two user ids (e.g. `user id = 1` and `user id = 2`). "
                        "Rephrase with both ids, or set USE_LLM=true for LLM-based extraction when ids are not explicit."
                    ),
                    datasets_cited=[],
                    graph_context_summary="",
                    session_id=req.session_id,
                    gnn_dataset_used=req.gnn_dataset,
                )
            if req.session_id:
                if req.session_id not in self._sessions:
                    self._sessions[req.session_id] = []
                self._sessions[req.session_id].append({
                    "message": req.message,
                    "intent": "connection_path",
                    "dataset": req.dataset,
                })
            return resp

        # Run pipeline (dataset filter passed as context, agents thread it through)
        result = self.pipeline.run(
            query=req.message,
            context=context,
            top_k=req.top_k,
        )

        # Determine which datasets were actually cited
        datasets_cited = self._extract_cited_datasets(result.get("results", []))

        # Store in session history
        if req.session_id:
            if req.session_id not in self._sessions:
                self._sessions[req.session_id] = []
            self._sessions[req.session_id].append({
                "message": req.message,
                "intent": result.get("intent"),
                "dataset": req.dataset,
            })

        return ChatResponse(
            message=req.message,
            dataset_queried=req.dataset or "all",
            mode=req.mode or "hybrid",
            intent=result.get("intent", "unknown"),
            results=result.get("results", [])[:req.top_k],
            insight=result.get("insight", ""),
            datasets_cited=datasets_cited or ([req.dataset] if req.dataset != "all" else []),
            graph_context_summary=result.get("graph_context", ""),
            pipeline_timing_ms=result.get("pipeline_timing_ms"),
            session_id=req.session_id,
            gnn_dataset_used=result.get("gnn_dataset_used"),
        )

    def _extract_cited_datasets(self, results: List[Dict]) -> List[str]:
        """Extract unique dataset values from result nodes."""
        datasets = set()
        for r in results:
            ds = r.get("dataset")
            if ds:
                datasets.add(ds)
        return sorted(datasets)


# ── Insert service (R5) ───────────────────────────────────────────────────────

class InsertService:
    """
    R5: Chat-driven graph insertions into Neo4j.

    Safety guarantees:
    1. All Cypher uses parameterized queries (no string concat from user input)
    2. ALLOW_CHAT_INSERT env gate — disabled by default in prod
    3. Max nodes per request enforced
    4. confirm=False returns preview without executing
    5. Source ID sanitized and auto-generated if missing
    """

    def __init__(self, neo4j_client):
        self.neo4j = neo4j_client

    # ── User insert ───────────────────────────────────────────────────────────

    def insert_user(self, req: InsertUserRequest, preview_only: bool = False) -> InsertResult:
        """Insert or merge a User node. Returns preview if preview_only=True."""
        source_id = (req.source_id or f"chat_{uuid.uuid4().hex[:8]}").strip()

        node_id = f"{req.dataset[:2]}_{source_id}"
        preview = {
            "type":         "MERGE User",
            "dataset":      req.dataset,
            "source_id":    source_id,
            "id":           node_id,
            "name":         req.name,
            "bio":          req.bio or "",
            "follower_count": req.follower_count,
            "influence_score": req.influence_score,
        }

        if preview_only:
            return InsertResult(
                ok=True,
                operation="preview",
                nodes_created=0,
                cypher_summary=f"MERGE (:User {{dataset:'{req.dataset}', source_id:'{source_id}'}}) SET ...",
                preview=preview,
                requires_confirm=True,
            )

        cypher = """
        MERGE (u:User {dataset: $dataset, source_id: $source_id})
        SET u.id            = $id,
            u.name          = $name,
            u.bio           = $bio,
            u.follower_count = $follower_count,
            u.influence_score = $influence_score,
            u.created_at    = datetime().epochSeconds,
            u.inserted_via  = 'chat'
        RETURN u.id AS created_id
        """
        try:
            self.neo4j.run_write_query(cypher, {
                "dataset":        req.dataset,
                "source_id":      source_id,
                "id":             node_id,
                "name":           req.name,
                "bio":            req.bio or "",
                "follower_count": req.follower_count,
                "influence_score": req.influence_score,
            })
            return InsertResult(
                ok=True,
                operation="inserted",
                nodes_created=1,
                cypher_summary=f"MERGE User(dataset={req.dataset}, source_id={source_id})",
                detail=preview,
            )
        except Exception as e:
            logger.error(f"insert_user failed: {e}")
            return InsertResult(ok=False, operation="failed", error=str(e))

    # ── Edge insert ───────────────────────────────────────────────────────────

    def insert_edge(self, req: InsertEdgeRequest, preview_only: bool = False) -> InsertResult:
        """Insert a relationship between two existing User nodes."""
        preview = {
            "type":         f"MERGE :{req.rel_type}",
            "dataset":      req.dataset,
            "from_user_id": req.from_user_id,
            "to_user_id":   req.to_user_id,
            "bidirectional": req.bidirectional,
        }

        if preview_only:
            return InsertResult(
                ok=True,
                operation="preview",
                edges_created=0,
                cypher_summary=f"MERGE (:{req.rel_type}) between {req.from_user_id} → {req.to_user_id}",
                preview=preview,
                requires_confirm=True,
            )

        rel = req.rel_type  # already validated by Literal
        cypher = f"""
        MATCH (a:User {{dataset: $dataset, source_id: $from_id}})
        MATCH (b:User {{dataset: $dataset, source_id: $to_id}})
        MERGE (a)-[:{rel}]->(b)
        """
        edges_created = 1
        try:
            self.neo4j.run_write_query(cypher, {
                "dataset": req.dataset,
                "from_id": req.from_user_id,
                "to_id":   req.to_user_id,
            })
            if req.bidirectional and rel in ("FRIEND", "FOLLOWS"):
                cypher2 = f"""
                MATCH (a:User {{dataset: $dataset, source_id: $from_id}})
                MATCH (b:User {{dataset: $dataset, source_id: $to_id}})
                MERGE (b)-[:{rel}]->(a)
                """
                self.neo4j.run_write_query(cypher2, {
                    "dataset": req.dataset,
                    "from_id": req.from_user_id,
                    "to_id":   req.to_user_id,
                })
                edges_created = 2

            return InsertResult(
                ok=True,
                operation="inserted",
                edges_created=edges_created,
                cypher_summary=f"MERGE :{rel} {req.from_user_id} ↔ {req.to_user_id} in {req.dataset}",
                detail=preview,
            )
        except Exception as e:
            logger.error(f"insert_edge failed: {e}")
            return InsertResult(ok=False, operation="failed", error=str(e))

    def insert_post(self, req: InsertPostRequest, preview_only: bool = False) -> InsertResult:
        """Create or merge a Post and (:User)-[:POSTED]->(:Post) for a chat/API insert."""
        source_id = (req.source_id or f"post_{uuid.uuid4().hex[:8]}").strip()
        post_key_id = f"{req.dataset[:2]}_{source_id}"
        author_id = (req.author_source_id or "").strip()
        if not author_id:
            return InsertResult(
                ok=False, operation="failed",
                error="author_source_id is required (user source_id in this dataset).",
            )

        preview = {
            "type": "MERGE Post + POSTED",
            "dataset": req.dataset,
            "post_source_id": source_id,
            "id": post_key_id,
            "title": req.title,
            "author_source_id": author_id,
        }

        if preview_only:
            return InsertResult(
                ok=True,
                operation="preview",
                nodes_created=0,
                edges_created=0,
                cypher_summary="MERGE Post + MERGE (User)-[:POSTED]->(Post)",
                preview=preview,
                requires_confirm=True,
            )

        found = self.neo4j.run_query(
            """
            MATCH (u:User {dataset: $dataset, source_id: $author_source_id})
            RETURN u.id AS id LIMIT 1
            """,
            {"dataset": req.dataset, "author_source_id": author_id},
        )
        if not found:
            return InsertResult(
                ok=False,
                operation="failed",
                error=f"No User with dataset={req.dataset!r} and source_id={author_id!r}.",
            )

        cypher = """
        MATCH (u:User {dataset: $dataset, source_id: $author_source_id})
        MERGE (p:Post {dataset: $dataset, source_id: $post_source_id})
        SET p.id = $id,
            p.title = $title,
            p.content = $content,
            p.topic = $topic,
            p.like_count = coalesce(p.like_count, 0),
            p.comment_count = coalesce(p.comment_count, 0),
            p.created_at = coalesce(p.created_at, toString(datetime())),
            p.inserted_via = 'api'
        MERGE (u)-[:POSTED]->(p)
        RETURN p.id AS post_id
        """
        try:
            self.neo4j.run_write_query(
                cypher,
                {
                    "dataset": req.dataset,
                    "post_source_id": source_id,
                    "id": post_key_id,
                    "title": req.title,
                    "content": req.content or "",
                    "topic": req.topic,
                    "author_source_id": author_id,
                },
            )
            return InsertResult(
                ok=True,
                operation="inserted",
                nodes_created=1,
                edges_created=1,
                cypher_summary=f"MERGE Post({post_key_id}) POSTED by {author_id} in {req.dataset}",
                detail=preview,
            )
        except Exception as e:
            logger.error(f"insert_post failed: {e}")
            return InsertResult(ok=False, operation="failed", error=str(e))

    # ── NL insert (LLM-assisted extraction) ───────────────────────────────────

    def parse_nl_insert(self, req: NLInsertRequest) -> Dict[str, Any]:
        """
        Parse a natural language insert command into structured operations.

        Uses regex + heuristics for robustness (no LLM dependency in hot path).
        Falls back to error if intent is ambiguous.

        Examples handled:
          "Add user Alice who is friends with Bob in the facebook dataset"
          "Create a user named Carol with bio 'Data scientist' in twitter"
          "Connect user_123 and user_456 as friends in reddit"
        """
        cmd = req.nl_command.strip()
        dataset = req.dataset

        # Detect dataset from command if not specified
        if not dataset:
            for ds in ["facebook", "twitter", "reddit", "demo"]:
                if ds in cmd.lower():
                    dataset = ds
                    break
            if not dataset:
                dataset = "demo"

        # Detect operation type
        is_add_user  = bool(re.search(r'\b(add|create|insert)\b.*\buser\b', cmd, re.I))
        is_add_edge  = bool(re.search(r'\b(friend|follow|connect|link)\b', cmd, re.I))
        is_add_post  = bool(re.search(r'\b(post|article|publish)\b', cmd, re.I))

        extracted_names = re.findall(r'\b(?:user\s+)?([A-Z][a-z]+)\b', cmd)
        extracted_names = [n for n in extracted_names if n.lower() not in
                          ('add','create','who','the','and','in','on','at','to','is','are')]

        operations = []

        if is_add_user and extracted_names:
            # Bio extraction
            bio_match = re.search(r"bio[: ]+['\"]?([^'\"]+)['\"]?", cmd, re.I)
            bio = bio_match.group(1).strip() if bio_match else ""

            for name in extracted_names[:CHAT_INSERT_MAX_NODES]:
                operations.append({
                    "type": "insert_user",
                    "payload": InsertUserRequest(
                        dataset=dataset,
                        name=name,
                        bio=bio or f"User added via chat: {name}",
                        source_id=f"chat_{name.lower()}",
                    )
                })

            # Also add FRIEND edge if "friends with" pattern found
            if is_add_edge and len(extracted_names) >= 2:
                operations.append({
                    "type": "insert_edge",
                    "payload": InsertEdgeRequest(
                        dataset=dataset,
                        from_user_id=f"chat_{extracted_names[0].lower()}",
                        to_user_id=f"chat_{extracted_names[1].lower()}",
                        rel_type="FRIEND",
                        bidirectional=True,
                    )
                })

        elif is_add_edge and len(extracted_names) >= 2:
            # Pure edge insertion between existing or new users
            operations.append({
                "type": "insert_edge",
                "payload": InsertEdgeRequest(
                    dataset=dataset,
                    from_user_id=extracted_names[0].lower(),
                    to_user_id=extracted_names[1].lower(),
                    rel_type="FRIEND",
                    bidirectional=True,
                )
            })

        if not operations:
            return {
                "ok": False,
                "error": "Could not extract a clear insert intent from the command. "
                         "Try: 'Add user Alice who is friends with Bob in the facebook dataset'",
                "operations": [],
                "dataset": dataset,
            }

        return {
            "ok": True,
            "operations": operations,
            "dataset": dataset,
            "parsed_names": extracted_names,
        }

    def execute_nl_insert(self, req: NLInsertRequest) -> InsertResult:
        """
        R5 main entry point: parse NL command → validate → optional preview → execute.
        """
        if not ALLOW_CHAT_INSERT:
            return InsertResult(
                ok=False,
                operation="rejected",
                error="Chat-driven inserts are disabled (ALLOW_CHAT_INSERT=false).",
            )

        parsed = self.parse_nl_insert(req)
        if not parsed["ok"]:
            return InsertResult(
                ok=False,
                operation="parse_failed",
                error=parsed.get("error", "Parse failed"),
            )

        ops = parsed["operations"]
        if len(ops) > CHAT_INSERT_MAX_NODES:
            return InsertResult(
                ok=False,
                operation="rejected",
                error=f"Too many operations ({len(ops)}). Max per request: {CHAT_INSERT_MAX_NODES}",
            )

        preview_mode = not req.confirm
        total_nodes, total_edges = 0, 0
        summaries = []
        details = []

        for op in ops:
            op_type = op["type"]
            payload = op["payload"]

            if op_type == "insert_user":
                result = self.insert_user(payload, preview_only=preview_mode)
            elif op_type == "insert_edge":
                result = self.insert_edge(payload, preview_only=preview_mode)
            else:
                continue

            total_nodes += result.nodes_created
            total_edges += result.edges_created
            summaries.append(result.cypher_summary)
            details.append(result.dict())

        if preview_mode:
            return InsertResult(
                ok=True,
                operation="preview",
                nodes_created=0,
                edges_created=0,
                cypher_summary="; ".join(summaries),
                preview={"operations": details, "parsed": parsed},
                requires_confirm=True,
            )

        return InsertResult(
            ok=True,
            operation="inserted",
            nodes_created=total_nodes,
            edges_created=total_edges,
            cypher_summary="; ".join(summaries),
            detail={"operations": details, "parsed": parsed},
        )
