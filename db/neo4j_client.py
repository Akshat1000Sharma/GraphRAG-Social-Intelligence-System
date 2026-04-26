"""
Neo4j Client: Connection pool and query execution for the social graph.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


def _neo4j_uri() -> str:
    return os.getenv("NEO4J_URI", "bolt://localhost:7687")


def _neo4j_user() -> str:
    return os.getenv("NEO4J_USER", "neo4j")


def _neo4j_password() -> str:
    return os.getenv("NEO4J_PASSWORD", "password123")


def _neo4j_database() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


# Legacy exports (values fixed at import time; prefer Neo4jClient() which re-reads env)
NEO4J_URI = _neo4j_uri()
NEO4J_USER = _neo4j_user()
NEO4J_PASSWORD = _neo4j_password()
NEO4J_DATABASE = _neo4j_database()


class Neo4jClient:
    """
    Thread-safe Neo4j client with connection pooling.
    """

    _instance: Optional["Neo4jClient"] = None
    _driver: Optional[Driver] = None

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri if uri is not None else _neo4j_uri()
        self.user = user if user is not None else _neo4j_user()
        self.password = password if password is not None else _neo4j_password()
        self._connected = False

    def connect(self) -> bool:
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=50,
                connection_acquisition_timeout=30,
            )
            self._driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j: {self.uri}")
            return True
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Neo4j connection failed: {e}")
            self._connected = False
            return False

    def close(self):
        if self._driver:
            self._driver.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._driver is not None

    @contextmanager
    def session(self, database: Optional[str] = None):
        db = database if database is not None else _neo4j_database()
        if not self.is_connected:
            self.connect()
        session = self._driver.session(database=db)
        try:
            yield session
        finally:
            session.close()

    def run_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as list of dicts."""
        if not self.is_connected:
            self.connect()
        db = database if database is not None else _neo4j_database()
        with self.session(db) as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def run_write_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a write Cypher query in a transaction."""
        if not self.is_connected:
            self.connect()
        with self.session() as session:
            result = session.execute_write(
                lambda tx: tx.run(query, params or {}).consume()
            )
            return {"counters": result.counters.__dict__}

    # ─── Schema Setup ─────────────────────────────────────────────────────────

    def setup_schema(self):
        """Create constraints, btree, and fulltext indexes (not vector indexes; those run at API startup)."""
        schema_queries = [
            # Uniqueness constraints
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT comment_id IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT group_id IF NOT EXISTS FOR (g:Group) REQUIRE g.id IS UNIQUE",

            # Lookup indexes
            "CREATE INDEX user_name IF NOT EXISTS FOR (u:User) ON (u.name)",
            "CREATE INDEX post_created IF NOT EXISTS FOR (p:Post) ON (p.created_at)",
            "CREATE INDEX post_likes IF NOT EXISTS FOR (p:Post) ON (p.like_count)",

            # Full-text search
            """CREATE FULLTEXT INDEX post_content IF NOT EXISTS
               FOR (p:Post) ON EACH [p.content, p.title]""",
            """CREATE FULLTEXT INDEX user_search IF NOT EXISTS
               FOR (u:User) ON EACH [u.name, u.bio]""",
        ]

        for q in schema_queries:
            try:
                self.run_write_query(q)
                logger.debug(f"Schema: {q[:60]}...")
            except Exception as e:
                logger.warning(f"Schema query skipped ({e}): {q[:60]}")

    # ─── Data Population ──────────────────────────────────────────────────────

    def seed_demo_data(self, num_users: int = 20, num_posts: int = 40):
        """Populate Neo4j with demo social graph data."""
        import random
        from datetime import datetime, timedelta

        logger.info("Seeding demo data...")

        # Create users
        for i in range(1, num_users + 1):
            self.run_write_query(
                """
                MERGE (u:User {id: $id})
                SET u.name = $name,
                    u.email = $email,
                    u.bio = $bio,
                    u.follower_count = $followers,
                    u.created_at = $created_at,
                    u.influence_score = $influence
                """,
                {
                    "id": f"user_{i}",
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "bio": f"Bio of user {i}",
                    "followers": random.randint(10, 10000),
                    "created_at": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                    "influence": round(random.random(), 3),
                },
            )

        # Create friend relationships (random graph)
        for i in range(1, num_users + 1):
            friends = random.sample(range(1, num_users + 1), k=min(5, num_users - 1))
            for j in friends:
                if i != j:
                    self.run_write_query(
                        """
                        MATCH (a:User {id: $a}), (b:User {id: $b})
                        MERGE (a)-[:FRIEND]->(b)
                        """,
                        {"a": f"user_{i}", "b": f"user_{j}"},
                    )

        # Create posts
        topics = ["AI", "Sports", "Music", "Politics", "Tech", "Food", "Travel"]
        for i in range(1, num_posts + 1):
            user_id = f"user_{random.randint(1, num_users)}"
            self.run_write_query(
                """
                MERGE (p:Post {id: $id})
                SET p.title = $title,
                    p.content = $content,
                    p.like_count = $likes,
                    p.comment_count = $comments,
                    p.created_at = $created_at,
                    p.topic = $topic
                WITH p
                MATCH (u:User {id: $user_id})
                MERGE (u)-[:POSTED]->(p)
                """,
                {
                    "id": f"post_{i}",
                    "title": f"Post about {random.choice(topics)}",
                    "content": f"Interesting content about {random.choice(topics)} from post {i}",
                    "likes": random.randint(0, 5000),
                    "comments": random.randint(0, 500),
                    "created_at": (datetime.now() - timedelta(hours=random.randint(1, 720))).isoformat(),
                    "topic": random.choice(topics),
                    "user_id": user_id,
                },
            )

        # Create likes
        for i in range(1, num_posts + 1):
            likers = random.sample(range(1, num_users + 1), k=random.randint(0, min(5, num_users)))
            for liker in likers:
                self.run_write_query(
                    """
                    MATCH (u:User {id: $uid}), (p:Post {id: $pid})
                    MERGE (u)-[:LIKED]->(p)
                    """,
                    {"uid": f"user_{liker}", "pid": f"post_{i}"},
                )

        logger.info(f"Demo data seeded: {num_users} users, {num_posts} posts")


# Singleton instance
_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get the singleton Neo4j client (creates and connects if needed)."""
    global _client
    if _client is None:
        _client = Neo4jClient()
        _client.connect()
    return _client
