"""
Graph Query Service: Business logic layer between API routes and Neo4j/GNN.
Handles query construction, result formatting, and error handling.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GraphQueryService:
    """
    Service layer for graph queries.
    Abstracts Neo4j details from API routes.
    """

    def __init__(self, neo4j_client):
        self.db = neo4j_client

    def get_friend_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get friend-of-friend recommendations from Neo4j."""
        if not self.db or not self.db.is_connected:
            return self._mock_recommendations(user_id, top_k)

        try:
            results = self.db.run_query(
                """
                MATCH (u:User)
                WHERE (u.id = $user_id OR u.source_id = $user_id)
                MATCH (u)-[:FRIEND]->(friend)-[:FRIEND]->(fof:User)
                WHERE fof <> u AND NOT (u)-[:FRIEND]->(fof)
                WITH fof, count(friend) AS mutual_count
                ORDER BY mutual_count DESC LIMIT $top_k
                RETURN fof.id AS id, fof.name AS name,
                       mutual_count AS mutual_friends,
                       fof.influence_score AS influence_score,
                       fof.follower_count AS follower_count
                """,
                {"user_id": user_id, "top_k": top_k},
            )
            return results
        except Exception as e:
            logger.error(f"Friend recommendation query failed: {e}")
            return self._mock_recommendations(user_id, top_k)

    def friend_recommendations_for_llm(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Same rows as GET /graph/friend-recommendations/{user_id} — use for /chat, /recommend-friends, and LLM context.
        """
        return self.get_friend_recommendations(user_id, top_k=top_k)

    def get_trending_posts(
        self,
        top_k: int = 10,
        topic: Optional[str] = None,
        hours_window: int = 48,
        dataset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get trending posts from Neo4j with engagement scoring."""
        if not self.db or not self.db.is_connected:
            return self._mock_trending_posts(top_k)

        since = (datetime.now() - timedelta(hours=hours_window)).isoformat()

        dataset_filter = " {dataset: $dataset}" if dataset and dataset != "all" else ""
        base_query = f"""
            MATCH (p:Post{dataset_filter})
            WITH p, p.like_count + (p.comment_count * 2) AS engagement
            {{topic_filter}}
            ORDER BY engagement DESC
            LIMIT $top_k
            RETURN p.id AS id, p.title AS title, p.content AS content,
                   p.like_count AS likes, p.comment_count AS comments,
                   p.topic AS topic, p.created_at AS created_at, engagement
        """

        topic_filter = "WHERE p.topic = $topic" if topic else ""
        query = base_query.format(topic_filter=topic_filter)
        params: Dict[str, Any] = {"top_k": top_k, "since": since}
        if topic:
            params["topic"] = topic
        if dataset and dataset != "all":
            params["dataset"] = dataset

        try:
            return self.db.run_query(query, params)
        except Exception as e:
            logger.error(f"Trending posts query failed: {e}")
            return self._mock_trending_posts(top_k)

    def get_user_influence_stats(self, user_id: str) -> Dict[str, Any]:
        """Get detailed influence statistics for a user."""
        if not self.db or not self.db.is_connected:
            return self._mock_user_stats(user_id)

        try:
            results = self.db.run_query(
                """
                MATCH (u:User)
                WHERE u.id = $user_id OR u.source_id = $user_id
                OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
                OPTIONAL MATCH (u)-[:FRIEND]-(f:User)
                WITH u, count(DISTINCT p) AS posts,
                     count(DISTINCT f) AS friends,
                     coalesce(sum(p.like_count), 0) AS total_likes,
                     coalesce(avg(p.like_count), 0) AS avg_likes
                RETURN u.id AS id, u.name AS name, u.bio AS bio,
                       u.follower_count AS followers,
                       u.influence_score AS gnn_score,
                       posts, friends, total_likes, avg_likes
                """,
                {"user_id": user_id},
            )
            return results[0] if results else self._mock_user_stats(user_id)
        except Exception as e:
            logger.error(f"User influence query failed: {e}")
            return self._mock_user_stats(user_id)

    def get_connection_path(
        self,
        user_a: str,
        user_b: str,
    ) -> Dict[str, Any]:
        """Find shortest path and common context between two users."""
        if not self.db or not self.db.is_connected:
            return {
                "shortest_path": None,
                "common_friends": [],
                "common_liked_posts": []
            }

        result: Dict[str, Any] = {}

        # Shortest path
        try:
            paths = self.db.run_query(
                """
                MATCH (a:User), (b:User)
                WHERE (a.id = $user_a OR a.source_id = $user_a)
                  AND (b.id = $user_b OR b.source_id = $user_b)
                MATCH path = shortestPath((a)-[*..6]-(b))
                RETURN [n IN nodes(path) | n.name] AS node_names,
                       [r IN relationships(path) | type(r)] AS rel_types,
                       length(path) AS hops
                LIMIT 1
                """,
                {"user_a": user_a, "user_b": user_b},
            )
            result["shortest_path"] = paths[0] if paths else None
        except Exception as e:
            logger.warning(f"Shortest path query failed: {e}")
            result["shortest_path"] = None

        # Common friends
        try:
            common = self.db.run_query(
                """
                MATCH (a:User), (b:User)
                WHERE (a.id = $user_a OR a.source_id = $user_a)
                  AND (b.id = $user_b OR b.source_id = $user_b)
                MATCH (a)-[:FRIEND]->(c)<-[:FRIEND]-(b)
                RETURN c.id AS id, c.name AS name
                """,
                {"user_a": user_a, "user_b": user_b},
            )
            result["common_friends"] = common
        except Exception as e:
            logger.warning(f"Common friends query failed: {e}")
            result["common_friends"] = []

        # Common liked posts
        try:
            common_posts = self.db.run_query(
                """
                MATCH (a:User), (b:User)
                WHERE (a.id = $user_a OR a.source_id = $user_a)
                  AND (b.id = $user_b OR b.source_id = $user_b)
                MATCH (a)-[:LIKED]->(p:Post)<-[:LIKED]-(b)
                RETURN p.id AS id, p.title AS title, p.topic AS topic
                LIMIT 5
                """,
                {"user_a": user_a, "user_b": user_b},
            )
            result["common_liked_posts"] = common_posts
        except Exception:
            result["common_liked_posts"] = []

        return result

    def get_link_prediction_candidates(
        self,
        user_id: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get candidate users for link prediction (2-3 hop neighbors)."""
        if not self.db or not self.db.is_connected:
            return [{"id": f"user_{i}", "name": f"User {i}", "graph_score": 1} for i in range(top_k)]

        try:
            return self.db.run_query(
                """
                MATCH (u:User)
                WHERE (u.id = $user_id OR u.source_id = $user_id)
                MATCH (u)-[:FRIEND*2..3]->(candidate:User)
                WHERE candidate <> u
                  AND NOT (u)-[:FRIEND]->(candidate)
                WITH candidate, count(*) AS path_count
                ORDER BY path_count DESC LIMIT $top_k
                RETURN candidate.id AS id, candidate.name AS name,
                       path_count AS graph_score
                """,
                {"user_id": user_id, "top_k": top_k},
            )
        except Exception as e:
            logger.error(f"Link candidate query failed: {e}")
            return []

    def get_all_top_influencers(self, top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve top influencers across the whole network."""
        if not self.db or not self.db.is_connected:
            return [{"id": f"user_{i}", "name": f"Influencer {i}", "composite_score": 1.0 - i * 0.05}
                    for i in range(top_k)]
        try:
            return self.db.run_query(
                """
                MATCH (u:User)
                OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
                WITH u, count(p) AS post_count, coalesce(avg(p.like_count), 0) AS avg_likes
                RETURN u.id AS id, u.name AS name,
                       u.follower_count AS followers,
                       u.influence_score AS gnn_score,
                       post_count, avg_likes,
                       (coalesce(u.follower_count,0)*0.4 + avg_likes*0.4 + post_count*0.2) AS composite_score
                ORDER BY composite_score DESC
                LIMIT $top_k
                """,
                {"top_k": top_k},
            )
        except Exception as e:
            logger.error(f"Influencer query failed: {e}")
            return []

    # ─── Mock data (fallback when Neo4j unavailable) ──────────────────────────

    def _mock_recommendations(self, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        return [
            {
                "id": f"user_{i}",
                "name": f"Suggested User {i}",
                "mutual_friends": max(1, 5 - i),
                "influence_score": round(0.9 - i * 0.05, 2),
                "source": "mock",
            }
            for i in range(1, top_k + 1)
        ]

    def _mock_trending_posts(self, top_k: int) -> List[Dict[str, Any]]:
        topics = ["AI", "Sports", "Tech", "Music", "Travel"]
        return [
            {
                "id": f"post_{i}",
                "title": f"Trending Post {i}",
                "content": f"This post about {topics[i % len(topics)]} is going viral.",
                "likes": max(100, 1000 - i * 50),
                "comments": max(10, 100 - i * 5),
                "topic": topics[i % len(topics)],
                "engagement": max(100, 1200 - i * 60),
                "source": "mock",
            }
            for i in range(1, top_k + 1)
        ]

    def _mock_user_stats(self, user_id: str) -> Dict[str, Any]:
        return {
            "id": user_id,
            "name": f"User {user_id}",
            "bio": "Social network user",
            "followers": 500,
            "gnn_score": 0.65,
            "posts": 12,
            "friends": 45,
            "total_likes": 3400,
            "avg_likes": 283.0,
            "source": "mock",
        }
