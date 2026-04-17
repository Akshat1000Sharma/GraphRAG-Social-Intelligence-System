#!/usr/bin/env python3
"""
Load repo-root ``.env`` (``KAGGLE_USERNAME``, ``KAGGLE_KEY``) and run
``kaggle kernels push`` for ``kaggle/kernel-metadata.json``.

Usage (from repo root):

    pip install kaggle python-dotenv
    python scripts/push_kaggle_kernel.py

The kernel ``id`` is set to ``<KAGGLE_USERNAME>/graphrag-pipeline-accuracy`` on each push
(template slug comes from ``kaggle/kernel-metadata.json``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Install python-dotenv: pip install python-dotenv", file=sys.stderr)
        return 1
    load_dotenv(env_path)
    user = (os.environ.get("KAGGLE_USERNAME") or "").strip()
    key = (os.environ.get("KAGGLE_KEY") or "").strip()
    if not user or not key:
        print(
            "Missing KAGGLE_USERNAME or KAGGLE_KEY (set in .env at repo root).",
            file=sys.stderr,
        )
        return 1
    kaggle_dir = root / "kaggle"
    meta = kaggle_dir / "kernel-metadata.json"
    if not meta.is_file():
        print(f"Missing {meta}", file=sys.stderr)
        return 1
    orig_text = meta.read_text(encoding="utf-8")
    data = json.loads(orig_text)
    slug = data.get("id", "").split("/")[-1] or "graphrag-pipeline-accuracy"
    data["id"] = f"{user}/{slug}"
    print(f"Kernel id for push: {data['id']} (from KAGGLE_USERNAME).", flush=True)

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = user
    env["KAGGLE_KEY"] = key

    kaggle_home = Path.home() / ".kaggle"
    kaggle_home.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_home / "kaggle.json"
    creds = json.dumps({"username": user, "key": key}, indent=2)
    had_json = kaggle_json.is_file()
    prev = kaggle_json.read_text(encoding="utf-8") if had_json else ""
    kaggle_json.write_text(creds + "\n", encoding="utf-8")
    if os.name == "nt":
        try:
            os.chmod(kaggle_json, 0o600)
        except OSError:
            pass

    cmd = ["kaggle", "kernels", "push", "-p", str(kaggle_dir)]
    print("+", " ".join(cmd), flush=True)
    code = 1
    try:
        meta.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        code = subprocess.call(cmd, env=env, cwd=str(root))
    finally:
        meta.write_text(orig_text, encoding="utf-8")
        if had_json:
            kaggle_json.write_text(prev, encoding="utf-8")
        else:
            try:
                kaggle_json.unlink()
            except OSError:
                pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
