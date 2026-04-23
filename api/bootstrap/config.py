"""
api/bootstrap/config.py
=======================
Central configuration for all dataset sources: local paths, download URLs,
required files, and ingest versioning.

All paths and URLs are configurable via environment variables so no secrets
are hardcoded and production deployments can point to internal mirrors.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Base directories ──────────────────────────────────────────────────────────

# Root of the data directory (configurable so tests and CI can override)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# Whether to attempt downloads when data is missing
AUTO_DOWNLOAD = os.getenv("AUTO_DOWNLOAD_DATASETS", "true").lower() == "true"

# Whether to ingest into Neo4j at startup
AUTO_INGEST = os.getenv("AUTO_INGEST_NEO4J", "true").lower() == "true"

# When AUTO_INGEST is false, optionally seed small demo users/posts (not Snap datasets)
SEED_DEMO_NEO4J = os.getenv("SEED_DEMO_NEO4J", "false").lower() == "true"

# Force reingest even if marker exists
FORCE_REINGEST = os.getenv("FORCE_REINGEST", "false").lower() == "true"

# Whether chat-driven inserts are allowed
ALLOW_CHAT_INSERT = os.getenv("ALLOW_CHAT_INSERT", "true").lower() == "true"

# Max nodes a single chat insert can create
CHAT_INSERT_MAX_NODES = int(os.getenv("CHAT_INSERT_MAX_NODES", "10"))

# Ingest batch size for UNWIND transactions
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "500"))

# ── Dataset manifest ──────────────────────────────────────────────────────────

@dataclass
class DatasetFile:
    """A single file required for a dataset."""
    filename: str
    url: str
    description: str
    required: bool = True


@dataclass
class DatasetManifest:
    """Everything needed to locate, download, and identify a dataset."""
    name: str               # "facebook" | "twitter" | "reddit"
    local_dir: Path         # DATA_DIR / name
    files: List[DatasetFile] = field(default_factory=list)
    ingest_version: str = "v1"   # bump to force re-ingest after schema changes
    description: str = ""

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.name

    def marker_path(self) -> Path:
        """Path to the ingest-complete marker file."""
        return self.dir / f".ingest_{self.ingest_version}.done"

    def is_ingested(self) -> bool:
        return self.marker_path().exists()

    def mark_ingested(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.marker_path().write_text("ok")

    def required_paths(self) -> List[Path]:
        return [self.dir / f.filename for f in self.files if f.required]

    def all_required_present(self) -> bool:
        return all(p.exists() for p in self.required_paths())


# ── Facebook Large Page-Page Network (SNAP/MUSAE) ─────────────────────────────
# Source: https://snap.stanford.edu/data/facebook-large-page-page-network.html
# Nodes: Facebook pages; Edges: mutual likes
# ~22k nodes, ~171k edges

FACEBOOK_MANIFEST = DatasetManifest(
    name="facebook",
    local_dir=DATA_DIR / "facebook",
    description="Facebook Large Page-Page Network (MUSAE). Pages as users, mutual likes as FRIEND edges.",
    ingest_version="v2",
    files=[
        DatasetFile(
            filename="musae_facebook_edges.csv",
            url=os.getenv(
                "FACEBOOK_EDGES_URL",
                "https://snap.stanford.edu/data/facebook_large.zip"
            ),
            description="Page-page edge list (id_1, id_2)",
        ),
        DatasetFile(
            filename="musae_facebook_target.csv",
            url=os.getenv(
                "FACEBOOK_TARGET_URL",
                "https://snap.stanford.edu/data/facebook_large.zip"
            ),
            description="Page metadata (id, page_type, page_name)",
            required=True,
        ),
        DatasetFile(
            filename="musae_facebook_features.json",
            url=os.getenv(
                "FACEBOOK_FEATURES_URL",
                "https://snap.stanford.edu/data/facebook_large.zip"
            ),
            description="Node feature vectors (optional, used by GNN)",
            required=False,  # not required for graph ingest
        ),
    ],
)

# ── Twitter Social Circles (SNAP ego-Twitter) ─────────────────────────────────
# Source: https://snap.stanford.edu/data/ego-Twitter.html
# Nodes: Twitter users; Edges: follow relationships
# Multiple ego-networks bundled in one archive

TWITTER_MANIFEST = DatasetManifest(
    name="twitter",
    local_dir=DATA_DIR / "twitter",
    description="Twitter Ego Networks (SNAP). Follow relationships as FRIEND edges.",
    ingest_version="v2",
    files=[
        DatasetFile(
            filename="twitter_combined.txt",
            url=os.getenv(
                "TWITTER_EDGES_URL",
                "https://snap.stanford.edu/data/twitter_combined.txt.gz"
            ),
            description="Combined ego network edge list (src dst)",
        ),
    ],
)

# ── Reddit Hyperlink Network ───────────────────────────────────────────────────
# Source: http://snap.stanford.edu/data/soc-RedditHyperlinks.html
# Nodes: Subreddits; Edges: cross-posting hyperlinks with sentiment
# ~35k nodes, ~858k edges

REDDIT_MANIFEST = DatasetManifest(
    name="reddit",
    local_dir=DATA_DIR / "reddit",
    description="Reddit Hyperlink Network (SNAP). Subreddits as users, hyperlinks as FRIEND edges.",
    ingest_version="v2",
    files=[
        DatasetFile(
            filename="soc-redditHyperlinks-title.tsv",
            url=os.getenv(
                "REDDIT_TITLE_URL",
                "http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
            ),
            description="Subreddit hyperlinks (SOURCE_SUBREDDIT, TARGET_SUBREDDIT, LINK_SENTIMENT, ...)",
        ),
    ],
)

# ── Registry ──────────────────────────────────────────────────────────────────

ALL_DATASETS: Dict[str, DatasetManifest] = {
    "facebook": FACEBOOK_MANIFEST,
    "twitter":  TWITTER_MANIFEST,
    "reddit":   REDDIT_MANIFEST,
}

VALID_DATASET_NAMES = list(ALL_DATASETS.keys()) + ["all", "demo"]
