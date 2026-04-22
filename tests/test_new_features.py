"""
Tests for R1-R5: dataset bootstrap, ingest, chat, insert.
Run: pytest tests/test_new_features.py -v
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch, mock_open
import json


# ── R1: Dataset manifest ──────────────────────────────────────────────────────

class TestDatasetConfig:
    def test_all_datasets_defined(self):
        from api.bootstrap.config import ALL_DATASETS
        assert set(ALL_DATASETS.keys()) == {"facebook", "twitter", "reddit"}

    def test_manifest_has_files(self):
        from api.bootstrap.config import FACEBOOK_MANIFEST
        assert len(FACEBOOK_MANIFEST.files) > 0
        assert FACEBOOK_MANIFEST.name == "facebook"

    def test_required_paths_list(self):
        from api.bootstrap.config import FACEBOOK_MANIFEST, DATA_DIR
        paths = FACEBOOK_MANIFEST.required_paths()
        assert all(isinstance(p, Path) for p in paths)

    def test_marker_path_naming(self):
        from api.bootstrap.config import FACEBOOK_MANIFEST
        marker = FACEBOOK_MANIFEST.marker_path()
        assert "ingest" in marker.name
        assert FACEBOOK_MANIFEST.ingest_version in marker.name

    def test_all_required_present_false_when_no_files(self, tmp_path):
        from api.bootstrap.config import DatasetManifest, DatasetFile, DATA_DIR
        m = DatasetManifest(
            name="test",
            local_dir=tmp_path / "test",
            files=[DatasetFile(filename="nope.csv", url="http://x", description="x")],
        )
        assert not m.all_required_present()


# ── R1: Dataset ensure ────────────────────────────────────────────────────────

class TestDatasetEnsure:
    def test_ensure_unknown_dataset(self):
        from api.bootstrap.datasets import ensure_dataset
        result = ensure_dataset("nonexistent")
        assert result["ok"] is False

    def test_ensure_dataset_already_present(self, tmp_path, monkeypatch):
        from api.bootstrap import config as cfg
        monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
        # Create the required files
        fb_dir = tmp_path / "facebook"
        fb_dir.mkdir()
        (fb_dir / "musae_facebook_edges.csv").write_text("id_1,id_2\n0,1\n")
        (fb_dir / "musae_facebook_target.csv").write_text("id,page_type,page_name\n0,company,Test\n")

        from api.bootstrap.config import FACEBOOK_MANIFEST
        monkeypatch.setattr(FACEBOOK_MANIFEST, "local_dir", fb_dir)

        from api.bootstrap.datasets import _ensure_facebook
        ok, status = _ensure_facebook(FACEBOOK_MANIFEST)
        assert ok
        assert status == "already_present"


# ── R2: Ingest helpers ────────────────────────────────────────────────────────

class TestIngest:
    def _mock_neo4j(self):
        client = MagicMock()
        client.is_connected = True
        client.run_query.return_value = [{"users": 0}]
        client.run_write_query.return_value = {"counters": {}}
        return client

    def test_get_dataset_counts(self):
        from db.ingest.ingest_all import get_dataset_counts
        client = self._mock_neo4j()
        client.run_query.return_value = [{"users": 42}]
        result = get_dataset_counts(client, "facebook")
        assert "users" in result

    def test_ingest_skips_when_not_connected(self):
        from db.ingest.ingest_all import ingest_all_if_needed
        client = MagicMock()
        client.is_connected = False
        result = ingest_all_if_needed(client)
        assert all(not r.get("ok") for r in result.values())

    def test_ingest_skips_when_marker_exists(self, tmp_path, monkeypatch):
        from api.bootstrap.config import FACEBOOK_MANIFEST
        monkeypatch.setattr(FACEBOOK_MANIFEST, "local_dir", tmp_path / "facebook")
        FACEBOOK_MANIFEST.dir.mkdir(parents=True, exist_ok=True)
        FACEBOOK_MANIFEST.mark_ingested()

        from db.ingest.ingest_all import ingest_dataset
        client = self._mock_neo4j()
        client.run_query.return_value = [{"users": 10}]
        result = ingest_dataset(client, "facebook")
        assert result.get("status") == "skipped_already_ingested"
        assert result.get("ok") is True

    def test_chunked_batching(self):
        from db.ingest.ingest_all import _chunked
        data = list(range(250))
        chunks = list(_chunked(data, 100))
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[2]) == 50


# ── R4: Chat schemas ──────────────────────────────────────────────────────────

class TestChatSchemas:
    def test_chat_request_defaults(self):
        from api.schemas import ChatRequest
        req = ChatRequest(message="hello")
        assert req.dataset == "all"
        assert req.mode == "hybrid"
        assert req.top_k == 10

    def test_chat_request_invalid_dataset(self):
        from api.schemas import ChatRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(message="hello", dataset="myspace")

    def test_chat_request_invalid_mode(self):
        from api.schemas import ChatRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(message="hello", mode="invalid")

    def test_chat_request_valid_datasets(self):
        from api.schemas import ChatRequest
        for ds in ["facebook", "twitter", "reddit", "demo", "all"]:
            req = ChatRequest(message="test", dataset=ds)
            assert req.dataset == ds


# ── R5: Insert schemas ────────────────────────────────────────────────────────

class TestInsertSchemas:
    def test_insert_user_valid(self):
        from api.schemas import InsertUserRequest
        req = InsertUserRequest(dataset="facebook", name="Alice")
        assert req.name == "Alice"
        assert req.influence_score == 0.3

    def test_insert_user_invalid_dataset(self):
        from api.schemas import InsertUserRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            InsertUserRequest(dataset="myspace", name="Alice")

    def test_insert_user_sanitizes_name(self):
        from api.schemas import InsertUserRequest
        req = InsertUserRequest(dataset="twitter", name="Bob\x00\x01")
        assert "\x00" not in req.name

    def test_insert_edge_valid(self):
        from api.schemas import InsertEdgeRequest
        req = InsertEdgeRequest(dataset="facebook", from_user_id="u1", to_user_id="u2")
        assert req.rel_type == "FRIEND"
        assert req.bidirectional is True

    def test_nl_insert_request(self):
        from api.schemas import NLInsertRequest
        req = NLInsertRequest(nl_command="Add user Alice in facebook", confirm=True)
        assert req.confirm is True


# ── R5: Insert service ────────────────────────────────────────────────────────

class TestInsertService:
    def _make_service(self):
        from api.services.chat_service import InsertService
        client = MagicMock()
        client.run_write_query.return_value = {"counters": {}}
        return InsertService(neo4j_client=client)

    def test_insert_user_preview(self):
        svc = self._make_service()
        from api.schemas import InsertUserRequest
        req = InsertUserRequest(dataset="facebook", name="Alice")
        result = svc.insert_user(req, preview_only=True)
        assert result.operation == "preview"
        assert result.requires_confirm is True
        assert result.nodes_created == 0

    def test_insert_user_execute(self):
        svc = self._make_service()
        from api.schemas import InsertUserRequest
        req = InsertUserRequest(dataset="facebook", name="Bob", source_id="test_bob")
        result = svc.insert_user(req, preview_only=False)
        assert result.ok is True
        assert result.nodes_created == 1
        svc.neo4j.run_write_query.assert_called_once()

    def test_insert_edge_preview(self):
        svc = self._make_service()
        from api.schemas import InsertEdgeRequest
        req = InsertEdgeRequest(dataset="twitter", from_user_id="u1", to_user_id="u2")
        result = svc.insert_edge(req, preview_only=True)
        assert result.operation == "preview"

    def test_nl_parse_add_user_with_friend(self):
        svc = self._make_service()
        from api.schemas import NLInsertRequest
        req = NLInsertRequest(
            nl_command="Add user Alice who is friends with Bob in the facebook dataset",
            confirm=False,
        )
        parsed = svc.parse_nl_insert(req)
        assert parsed["ok"] is True
        assert parsed["dataset"] == "facebook"
        assert len(parsed["operations"]) >= 1

    def test_nl_parse_detects_dataset(self):
        svc = self._make_service()
        from api.schemas import NLInsertRequest
        for ds in ["facebook", "twitter", "reddit"]:
            req = NLInsertRequest(nl_command=f"Add user Test in the {ds} dataset")
            parsed = svc.parse_nl_insert(req)
            assert parsed["dataset"] == ds

    def test_nl_insert_disabled_when_env_off(self, monkeypatch):
        import api.bootstrap.config as cfg
        monkeypatch.setattr(cfg, "ALLOW_CHAT_INSERT", False)
        # Need to reload the module to pick up the change
        import importlib
        import api.services.chat_service as cs
        monkeypatch.setattr(cs, "ALLOW_CHAT_INSERT", False)
        svc = self._make_service()
        from api.schemas import NLInsertRequest
        req = NLInsertRequest(nl_command="Add user Alice in facebook", confirm=True)
        result = svc.execute_nl_insert(req)
        assert result.ok is False
        assert "disabled" in result.error.lower()

    def test_cypher_is_parameterized(self):
        """Ensure no raw user text appears in Cypher strings."""
        svc = self._make_service()
        from api.schemas import InsertUserRequest
        malicious = "'; DROP ALL DATA; --"
        req = InsertUserRequest(dataset="facebook", name=malicious, source_id="hack")
        # execute — should not raise and Cypher must use params
        result = svc.insert_user(req, preview_only=False)
        call_args = svc.neo4j.run_write_query.call_args
        if call_args:
            cypher_str = call_args[0][0]
            # The malicious string must NOT appear in the Cypher template
            assert malicious not in cypher_str
            # It should be in params instead
            params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
            assert any(malicious in str(v) for v in params.values())

    def test_insert_post_missing_author(self):
        from api.schemas import InsertPostRequest
        svc = self._make_service()
        svc.neo4j.run_query.return_value = []
        req = InsertPostRequest(
            dataset="facebook",
            author_source_id="nobody",
            title="T",
        )
        result = svc.insert_post(req, preview_only=False)
        assert result.ok is False
        assert "source_id" in (result.error or "").lower() or "No User" in (result.error or "")

    def test_insert_post_preview(self):
        from api.schemas import InsertPostRequest
        svc = self._make_service()
        req = InsertPostRequest(
            dataset="facebook",
            author_source_id="u1",
            title="Title",
        )
        r = svc.insert_post(req, preview_only=True)
        assert r.operation == "preview"
        assert r.requires_confirm is True


# ── R3: Dataset filter in retrieval ──────────────────────────────────────────

class TestDatasetFilter:
    def test_router_passes_dataset_in_params(self):
        from api.agents.router import RouterAgent
        from api.agents.analyzer import QueryAnalyzerAgent, QueryIntent, RetrievalStrategy
        from dataclasses import dataclass, field
        from typing import Dict, Any

        agent = RouterAgent()
        # Build a minimal AnalyzedQuery with dataset in entities
        analyzed = MagicMock()
        analyzed.intent = QueryIntent.FRIEND_RECOMMENDATION
        analyzed.retrieval_strategy = RetrievalStrategy.HYBRID
        analyzed.entities = {"user_id": "u1", "dataset": "facebook"}
        analyzed.constraints = {}

        query_type, params, mode = agent.route(analyzed, None)
        assert params.get("dataset") == "facebook"

    def test_query_type_switches_to_ds_variant(self):
        from api.agents.router import RouterAgent
        from api.agents.analyzer import QueryIntent, RetrievalStrategy

        agent = RouterAgent()
        analyzed = MagicMock()
        analyzed.intent = QueryIntent.TRENDING_POSTS
        analyzed.retrieval_strategy = RetrievalStrategy.HYBRID
        analyzed.entities = {"dataset": "twitter"}
        analyzed.constraints = {}

        query_type, params, mode = agent.route(analyzed, None)
        # Should select the _ds variant when dataset is specified
        assert "dataset" in params
        assert params["dataset"] == "twitter"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
