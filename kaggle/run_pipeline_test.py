"""
Kaggle script kernel: clone this repo, ensure Facebook (MUSAE) data the same way as
``api/main.py`` (``ensure_dataset`` / ``api.bootstrap.datasets``), then run
``tests/testing_pipeline_accuracy.py``.

Environment mirrors ``.env.example`` / API startup where relevant:
``DATA_DIR``, ``AUTO_DOWNLOAD_DATASETS``, ``FACEBOOK_EDGES_URL`` (optional override).
Neo4j is not started here; the pipeline accuracy test reads graph files from disk only.

Data resolution order:
1. If ``musae_facebook_edges.csv`` + ``musae_facebook_target.csv`` already exist under
   the repo ``data/facebook`` directory, nothing is fetched.
2. Else, if Kaggle mounted input under ``/kaggle/input``, copy that tree (recursive
   search for the edges file).
3. Else, download via the same ZIP logic as ``api/bootstrap/datasets.py`` (URL from
   ``FACEBOOK_EDGES_URL`` or SNAP default in ``api/bootstrap/config.py``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = os.environ.get(
    "GRAPH_RAG_REPO_URL",
    "https://github.com/Akshat1000Sharma/GraphRAG-Social-Intelligence-System.git",
)
REPO_DIR = WORK / "GraphRAG-Social-Intelligence-System"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _copy_musae_from_kaggle_input(dest: Path) -> bool:
    """If an attached dataset lives under /kaggle/input (any depth), mirror it into dest."""
    inp = Path("/kaggle/input")
    if not inp.is_dir():
        return False
    for edges in inp.rglob("musae_facebook_edges.csv"):
        if not edges.is_file():
            continue
        src = edges.parent
        tgt = src / "musae_facebook_target.csv"
        if not tgt.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_dir():
            shutil.rmtree(dest)
        elif dest.exists():
            dest.unlink()
        shutil.copytree(src, dest)
        print(f"Copied MUSAE from Kaggle input: {src} -> {dest}", flush=True)
        return True
    return False


def _configure_runtime_env() -> None:
    """Match api/main.py load order: repo root .env then api/.env; then fixed DATA_DIR."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_DIR / ".env", override=False)
        load_dotenv(REPO_DIR / "api" / ".env", override=True)
    except ImportError:
        pass

    # Same semantics as .env.example / api/bootstrap/config.py
    os.environ["DATA_DIR"] = str(REPO_DIR / "data")
    os.environ.setdefault("AUTO_DOWNLOAD_DATASETS", "true")
    os.environ.setdefault("AUTO_INGEST_NEO4J", "false")
    # FACEBOOK_*_URL overrides from Kaggle env / secrets: leave os.environ as-is;
    # api/bootstrap/config.py reads them when first imported below.


def _ensure_facebook_dataset() -> None:
    """Use the same bootstrap as FastAPI lifespan (R1), facebook only for this kernel."""
    fb_dir = Path(os.environ["DATA_DIR"]) / "facebook"
    edges = fb_dir / "musae_facebook_edges.csv"
    target = fb_dir / "musae_facebook_target.csv"
    if not (edges.is_file() and target.is_file()):
        _copy_musae_from_kaggle_input(fb_dir)

    # Import only after DATA_DIR / AUTO_DOWNLOAD_* are set (config reads os.environ once).
    from api.bootstrap.datasets import ensure_dataset

    result = ensure_dataset("facebook")
    if not result.get("ok"):
        print(
            f"ERROR: Facebook dataset bootstrap failed: {result!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Facebook dataset: {result}", flush=True)


def main() -> None:
    if not REPO_DIR.is_dir():
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                REPO_URL,
                str(REPO_DIR),
            ],
            cwd=WORK,
        )

    sys.path.insert(0, str(REPO_DIR))
    _configure_runtime_env()
    _ensure_facebook_dataset()

    # GPU kernel + /kaggle/working enables heavy search + GNN fine-tune in the test file.
    os.environ.setdefault("PIPELINE_ACCURACY_HEAVY", "1")

    run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")],
        cwd=REPO_DIR,
    )

    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            str(REPO_DIR / "tests" / "testing_pipeline_accuracy.py"),
            "-v",
            "--tb=short",
        ],
        cwd=str(REPO_DIR),
        env=os.environ.copy(),
    )
    summary = WORK / "kaggle_kernel_exit_code.txt"
    summary.write_text(f"pytest_exit_code={rc}\n", encoding="utf-8")
    print(f"Wrote {summary}", flush=True)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
