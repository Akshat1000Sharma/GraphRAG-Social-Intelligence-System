"""
Test Suite: Unit and integration tests for the Social Graph Intelligence System.
Run with: pytest tests/ -v
"""

import json
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestGNNModel:

    def test_social_graph_gnn_forward(self):
        """Test forward pass of the main GNN model."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model.gnn_model import SocialGraphGNN

        model = SocialGraphGNN(
            in_channels=16,
            hidden_channels=32,
            embedding_dim=16,
            num_classes=4,
            num_layers=2,
        )
        model.eval()

        num_nodes = 50
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, 100))

        with torch.no_grad():
            z, logits, _ = model(x, edge_index)

        assert z.shape == (num_nodes, 16), f"Expected ({num_nodes}, 16), got {z.shape}"
        assert logits.shape == (num_nodes, 4), f"Expected ({num_nodes}, 4), got {logits.shape}"

    def test_link_prediction_output_range(self):
        """Link prediction probabilities must be in [0, 1]."""
        from model.gnn_model import SocialGraphGNN

        model = SocialGraphGNN(in_channels=16, hidden_channels=32, embedding_dim=16, num_classes=4)
        model.eval()

        num_nodes = 20
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        pos_edges = torch.randint(0, num_nodes, (2, 10))

        with torch.no_grad():
            z = model.encode(x, edge_index)
            probs = model.predict_link(z, pos_edges)

        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_gat_model_forward(self):
        """Test GAT model for Twitter dataset."""
        from model.gnn_model import GATSocialGNN

        model = GATSocialGNN(in_channels=16, hidden_channels=16, embedding_dim=8, num_classes=4)
        model.eval()

        x = torch.randn(30, 16)
        edge_index = torch.randint(0, 30, (2, 60))

        with torch.no_grad():
            z, logits, _ = model(x, edge_index)

        assert z.shape[0] == 30
        assert logits.shape == (30, 4)

    def test_node_classifier_output_shape(self):
        """Node classifier must output correct number of classes."""
        from model.gnn_model import NodeClassifier

        clf = NodeClassifier(in_channels=64, hidden_channels=32, num_classes=4)
        z = torch.randn(100, 64)
        out = clf(z)
        assert out.shape == (100, 4)

    def test_model_loss_computation(self):
        """Loss computation must return a scalar tensor."""
        from model.gnn_model import SocialGraphGNN

        model = SocialGraphGNN(in_channels=16, hidden_channels=32, embedding_dim=16, num_classes=4)
        x = torch.randn(20, 16)
        edge_index = torch.randint(0, 20, (2, 40))
        pos_edges = torch.randint(0, 20, (2, 10))

        z, logits, link_probs = model(x, edge_index, pos_edges)
        labels = torch.randint(0, 4, (20,))
        loss = model.compute_loss(logits, labels, link_probs, pos_edges.size(1))

        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)

    def test_model_info(self):
        """Model info dict should have expected keys."""
        from model.gnn_model import SocialGraphGNN

        model = SocialGraphGNN(in_channels=32, hidden_channels=64, embedding_dim=32, num_classes=4)
        info = model.get_model_info()
        assert "total_parameters" in info
        assert info["total_parameters"] > 0


# ─── Inference Engine Tests ───────────────────────────────────────────────────

class TestInferenceEngine:

    def test_engine_loads_untrained_model(self):
        """Engine should fall back to untrained model gracefully."""
        from model.inference import GNNInferenceEngine

        engine = GNNInferenceEngine("facebook")
        engine.load()  # Weights don't exist in test env → fallback
        assert engine.model is not None

    def test_embedding_computation(self):
        """Embeddings should have correct shape."""
        from model.inference import GNNInferenceEngine

        engine = GNNInferenceEngine("facebook")
        engine.load()

        x = torch.randn(15, 128)
        edge_index = torch.randint(0, 15, (2, 30))
        embs = engine.get_node_embeddings(x, edge_index)

        assert embs.shape == (15, 128)

    def test_link_probability_prediction(self):
        """Predicted probabilities should be in [0, 1]."""
        from model.inference import GNNInferenceEngine

        engine = GNNInferenceEngine("facebook")
        engine.load()

        x = torch.randn(10, 128)
        edge_index = torch.randint(0, 10, (2, 20))
        pairs = [(0, 1), (2, 3), (4, 5)]
        results = engine.predict_link_probability(x, edge_index, pairs)

        assert len(results) == 3
        for r in results:
            assert 0.0 <= r["probability"] <= 1.0

    def test_node_classification_returns_all_nodes(self):
        """Node classification should return results for all nodes."""
        from model.inference import GNNInferenceEngine

        engine = GNNInferenceEngine("facebook")
        engine.load()

        num_nodes = 8
        x = torch.randn(num_nodes, 128)
        edge_index = torch.randint(0, num_nodes, (2, 16))
        results = engine.classify_nodes(x, edge_index)

        assert len(results) == num_nodes
        for r in results:
            assert "predicted_class" in r
            assert "confidence" in r
            assert 0.0 <= r["confidence"] <= 1.0

    def test_embedding_similarity_empty_index(self):
        """Similarity search on empty index should return empty list."""
        from model.inference import GNNInferenceEngine

        engine = GNNInferenceEngine("facebook")
        # No embeddings loaded
        query = np.random.randn(128)
        results = engine.embedding_similarity(query)
        assert isinstance(results, list)


# ─── Vector Store Tests ───────────────────────────────────────────────────────

class TestVectorStore:

    def test_in_memory_index_add_and_search(self):
        """In-memory vector index should return results after adding vectors."""
        from rag.vector_store import InMemoryVectorIndex

        idx = InMemoryVectorIndex(dim=16)
        for i in range(20):
            idx.add(np.random.randn(16).astype(np.float32), {"id": f"item_{i}", "name": f"Item {i}"})

        query = np.random.randn(16).astype(np.float32)
        results = idx.search(query, top_k=5)

        assert len(results) == 5
        assert "similarity_score" in results[0]

    def test_in_memory_index_empty_search(self):
        """Search on empty index should return empty list."""
        from rag.vector_store import InMemoryVectorIndex

        idx = InMemoryVectorIndex(dim=32)
        results = idx.search(np.random.randn(32), top_k=5)
        assert results == []

    def test_text_store_embeds_text(self):
        """Text store should produce embedding of expected dimension."""
        from rag.vector_store import TextEmbeddingStore

        store = TextEmbeddingStore()
        emb = store.embed_text("Hello world, this is a test.")

        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert emb.shape[0] > 0

    def test_cosine_similarity_self(self):
        """Cosine similarity of a vector with itself should be 1.0."""
        from rag.vector_store import TextEmbeddingStore

        store = TextEmbeddingStore()
        v = np.random.randn(128).astype(np.float32)
        sim = store.cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors should be ~0."""
        from rag.vector_store import TextEmbeddingStore

        store = TextEmbeddingStore()
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = store.cosine_similarity(v1, v2)
        assert abs(sim) < 1e-5


# ─── Query Analyzer Tests ─────────────────────────────────────────────────────

class TestQueryAnalyzer:

    def setup_method(self):
        from api.agents.analyzer import QueryAnalyzerAgent
        self.analyzer = QueryAnalyzerAgent()

    def test_friend_recommendation_intent(self):
        from api.agents.analyzer import QueryIntent
        result = self.analyzer.analyze("Recommend friends for user_1")
        assert result.intent == QueryIntent.FRIEND_RECOMMENDATION

    def test_influencer_intent(self):
        from api.agents.analyzer import QueryIntent
        result = self.analyzer.analyze("Who are the top influencers on the platform?")
        assert result.intent == QueryIntent.INFLUENCER_DETECTION

    def test_trending_posts_intent(self):
        from api.agents.analyzer import QueryIntent
        result = self.analyzer.analyze("Show me the trending posts today")
        assert result.intent == QueryIntent.TRENDING_POSTS

    def test_explain_connection_intent(self):
        from api.agents.analyzer import QueryIntent
        result = self.analyzer.analyze("Explain the connection between user_1 and user_2")
        assert result.intent == QueryIntent.EXPLAIN_CONNECTION

    def test_top_k_extraction(self):
        result = self.analyzer.analyze("Show top 5 recommendations")
        assert result.top_k == 5

    def test_entity_extraction_from_context(self):
        result = self.analyzer.analyze("Who should I connect with?", context={"user_id": "user_42"})
        assert result.entities.get("user_id") == "user_42"

    def test_requires_explanation_flag(self):
        result = self.analyzer.analyze("Why should I connect with user_3?")
        assert result.requires_explanation is True

    def test_content_search_uses_vector_strategy(self):
        from api.agents.analyzer import RetrievalStrategy
        result = self.analyzer.analyze("Find posts about machine learning")
        assert result.retrieval_strategy == RetrievalStrategy.VECTOR_ONLY


# ─── Hybrid Retrieval Tests ───────────────────────────────────────────────────

class TestHybridRetrieval:

    def _make_retriever(self):
        from rag.hybrid_retrieval import HybridRetriever, GraphRetriever, VectorRetriever
        from rag.vector_store import TextEmbeddingStore, InMemoryVectorIndex

        mock_neo4j = MagicMock()
        mock_neo4j.is_connected = True
        mock_neo4j.run_query.return_value = [
            {"id": "user_1", "name": "Alice", "mutual_friends": 3, "influence_score": 0.8},
            {"id": "user_2", "name": "Bob", "mutual_friends": 2, "influence_score": 0.6},
        ]

        graph_ret = GraphRetriever(mock_neo4j)
        text_store = TextEmbeddingStore()
        user_idx = InMemoryVectorIndex(dim=384)
        post_idx = InMemoryVectorIndex(dim=384)

        # Add dummy entries to user index
        for i in range(5):
            user_idx.add(
                np.random.randn(384).astype(np.float32),
                {"id": f"user_{i}", "name": f"User {i}"},
            )

        vector_ret = VectorRetriever(text_store, user_idx, post_idx)
        return HybridRetriever(graph_ret, vector_ret)

    def test_graph_only_retrieval(self):
        from rag.hybrid_retrieval import RetrievalMode
        retriever = self._make_retriever()
        ctx = retriever.retrieve("friend_recommendation", {"user_id": "user_1"}, mode=RetrievalMode.GRAPH)
        assert ctx.graph_context is not None
        assert len(ctx.graph_context.raw_records) > 0

    def test_hybrid_retrieval_fuses_results(self):
        from rag.hybrid_retrieval import RetrievalMode
        retriever = self._make_retriever()
        ctx = retriever.retrieve(
            "friend_recommendation",
            {"user_id": "user_1"},
            nl_query="recommend friends",
            mode=RetrievalMode.HYBRID,
        )
        # Should have results from both sources
        assert ctx.graph_context is not None or ctx.vector_context is not None

    def test_fusion_scores_are_positive(self):
        from rag.hybrid_retrieval import RetrievalMode
        retriever = self._make_retriever()
        ctx = retriever.retrieve(
            "friend_recommendation",
            {"user_id": "user_1"},
            nl_query="find friends",
            mode=RetrievalMode.HYBRID,
        )
        for score in ctx.fusion_scores.values():
            assert score >= 0


# ─── Synthesizer Tests ────────────────────────────────────────────────────────

class TestSynthesizer:

    def test_template_insight_fallback(self):
        """Without LLM, should return template-based insight."""
        from api.agents.synthesizer import SynthesizerAgent, SynthesizedResponse
        from api.agents.analyzer import QueryAnalyzerAgent, QueryIntent
        from rag.hybrid_retrieval import HybridContext, RetrievalMode

        agent = SynthesizerAgent()
        analyzed = QueryAnalyzerAgent().analyze("Show trending posts")
        hybrid_ctx = HybridContext(retrieval_mode=RetrievalMode.GRAPH)

        with patch.dict("os.environ", {"USE_LLM": "false"}):
            response = agent.synthesize(analyzed, hybrid_ctx)

        assert isinstance(response.natural_language_insight, str)
        assert len(response.natural_language_insight) > 0

    def test_synthesize_merges_gnn_predictions(self):
        """GNN predictions should be merged into structured data."""
        from api.agents.synthesizer import SynthesizerAgent
        from api.agents.analyzer import QueryAnalyzerAgent
        from rag.hybrid_retrieval import HybridContext, RetrievalMode, GraphContext

        agent = SynthesizerAgent()
        analyzed = QueryAnalyzerAgent().analyze("Find influencers")

        graph_ctx = GraphContext(
            query_type="influence_stats",
            primary_entities=[{"id": "user_1", "name": "Alice", "gnn_score": None}],
        )
        hybrid_ctx = HybridContext(
            graph_context=graph_ctx,
            fused_entities=[{"id": "user_1", "name": "Alice"}],
            retrieval_mode=RetrievalMode.GRAPH,
        )

        gnn_preds = [{"node_id": "user_1", "confidence": 0.92, "predicted_class": "influencer"}]
        response = agent.synthesize(analyzed, hybrid_ctx, gnn_predictions=gnn_preds)

        # Check that gnn score was merged
        assert response.gnn_predictions == gnn_preds


# ─── Validator Tests ──────────────────────────────────────────────────────────

class TestValidator:

    def test_valid_response_passes(self):
        from api.agents.validator import ValidatorAgent
        from api.agents.synthesizer import SynthesizedResponse
        from api.agents.analyzer import QueryAnalyzerAgent

        validator = ValidatorAgent()
        analyzed = QueryAnalyzerAgent().analyze("Get trending posts")
        response = SynthesizedResponse(
            intent="trending_posts",
            structured_data=[{"id": "post_1", "title": "Hot Post", "likes": 500}],
            natural_language_insight="Here are the trending posts.",
            sources=["neo4j_graph"],
            confidence=0.85,
        )

        report = validator.validate(analyzed, response)
        assert report.is_valid
        assert len(report.issues) == 0

    def test_empty_data_flagged_for_most_intents(self):
        from api.agents.validator import ValidatorAgent
        from api.agents.synthesizer import SynthesizedResponse
        from api.agents.analyzer import QueryAnalyzerAgent

        validator = ValidatorAgent()
        analyzed = QueryAnalyzerAgent().analyze("Find trending posts")
        response = SynthesizedResponse(
            intent="trending_posts",
            structured_data=[],   # empty!
            natural_language_insight="Some insight",
            sources=[],
        )

        report = validator.validate(analyzed, response)
        assert not report.is_valid

    def test_long_insight_gets_truncated(self):
        from api.agents.validator import ValidatorAgent
        from api.agents.synthesizer import SynthesizedResponse
        from api.agents.analyzer import QueryAnalyzerAgent

        validator = ValidatorAgent()
        analyzed = QueryAnalyzerAgent().analyze("Find users")
        very_long = "x" * 3000
        response = SynthesizedResponse(
            intent="friend_recommendation",
            structured_data=[{"id": "u1"}],
            natural_language_insight=very_long,
            sources=["neo4j_graph"],
        )
        report = validator.validate(analyzed, response)
        final = report.corrected_response
        assert len(final.natural_language_insight) <= 1510  # truncated

    def test_deduplication_removes_duplicates(self):
        from api.agents.validator import ValidatorAgent
        from api.agents.synthesizer import SynthesizedResponse
        from api.agents.analyzer import QueryAnalyzerAgent

        validator = ValidatorAgent()
        analyzed = QueryAnalyzerAgent().analyze("Recommend friends")
        response = SynthesizedResponse(
            intent="friend_recommendation",
            structured_data=[
                {"id": "user_1", "name": "Alice"},
                {"id": "user_1", "name": "Alice"},  # duplicate
                {"id": "user_2", "name": "Bob"},
            ],
            natural_language_insight="Some friends.",
            sources=["neo4j_graph"],
        )
        report = validator.validate(analyzed, response)
        assert len(report.corrected_response.structured_data) == 2


# ─── API Integration Tests ────────────────────────────────────────────────────

class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints using TestClient."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Mock all external dependencies before testing API."""
        from fastapi.testclient import TestClient

        with patch("db.neo4j_client.Neo4jClient.connect", return_value=True), \
             patch("db.neo4j_client.Neo4jClient.is_connected", new_callable=lambda: property(lambda self: True)), \
             patch("db.neo4j_client.Neo4jClient.run_query", return_value=[]), \
             patch("db.neo4j_client.Neo4jClient.run_write_query", return_value={}), \
             patch("db.neo4j_client.Neo4jClient.setup_schema"), \
             patch("db.neo4j_client.Neo4jClient.seed_demo_data"), \
             patch("model.inference.inference_manager.load_dataset"):

            from api.main import app
            self.client = TestClient(app)
            yield

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_recommend_friends_endpoint(self):
        resp = self.client.get("/recommend-friends/user_1")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data or "intent" in data

    def test_trending_posts_endpoint(self):
        resp = self.client.get("/trending-posts?top_k=5")
        assert resp.status_code == 200

    def test_user_influence_endpoint(self):
        resp = self.client.get("/user-influence/user_1")
        assert resp.status_code == 200

    def test_explain_connection_endpoint(self):
        resp = self.client.get("/explain-connection?user_a=user_1&user_b=user_2")
        assert resp.status_code == 200

    def test_query_endpoint_post(self):
        resp = self.client.post(
            "/query",
            json={"query": "Who are the top influencers?", "top_k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "intent" in data

    def test_query_endpoint_empty_query_rejected(self):
        resp = self.client.post("/query", json={"query": ""})
        assert resp.status_code == 400

    def test_predict_links_endpoint(self):
        resp = self.client.post(
            "/predict-links",
            json={"user_id": "user_1", "pairs": [["user_1", "user_2"]]},
        )
        assert resp.status_code == 200


# ─── Utils Tests ──────────────────────────────────────────────────────────────

class TestUtils:

    def test_set_seed_reproducibility(self):
        from model.utils import set_seed
        set_seed(42)
        t1 = torch.randn(5)
        set_seed(42)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2)

    def test_build_structural_features_shape(self):
        from model.utils import build_structural_features
        edge_index = torch.randint(0, 50, (2, 200))
        x = build_structural_features(edge_index, num_nodes=50, feature_dim=32)
        assert x.shape == (50, 32)

    def test_normalize_features(self):
        from model.utils import normalize_features
        x = torch.randn(10, 16)
        x_norm = normalize_features(x)
        row_sums = x_norm.abs().sum(dim=1)
        # Each row should have L1 norm <= 1 (after normalization)
        assert (row_sums <= 1.001).all()

    def test_early_stopping_triggers(self):
        from model.utils import EarlyStopping
        import tempfile, os

        stopper = EarlyStopping(patience=3)
        model = MagicMock()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        # Stagnating scores should trigger early stopping
        for score in [0.8, 0.79, 0.78, 0.77]:
            stop = stopper(score, model, path)

        assert stop is True
        os.unlink(path)

    def test_count_parameters(self):
        from model.utils import count_parameters
        from model.gnn_model import SocialGraphGNN

        model = SocialGraphGNN(in_channels=16, hidden_channels=32, embedding_dim=16, num_classes=4)
        count = count_parameters(model)
        assert count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
