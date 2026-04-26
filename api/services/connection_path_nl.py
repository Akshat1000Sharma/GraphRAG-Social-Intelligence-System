"""
Natural-language handling for two-user connection / shortest-path questions.
Uses the same graph logic as GET /graph/connection-path.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"


def looks_like_connection_path_query(message: str) -> bool:
    """Heuristic: user wants shortest path / connection between two users."""
    if not (message and message.strip()):
        return False
    m = message.lower()
    if "shortest path" in m:
        return True
    if "connection path" in m:
        return True
    if "path between" in m or "path of friends" in m:
        return True
    if "between" in m and "path" in m and ("user" in m or "friend" in m):
        return True
    if "how are" in m and "connected" in m and ("user" in m or "two" in m):
        return True
    return False


def extract_two_user_ids_regex(message: str) -> Tuple[Optional[str], Optional[str]]:
    """Deterministic ids from phrasing like `user id = 1` … `user id = 2` or `between 1 and 2`."""
    # "user id = 1" (with spaces) or user_id=1
    for pat in (r"user_id\s*=\s*([a-zA-Z0-9_]+)", r"user\s+id\s*=\s*([a-zA-Z0-9_]+)"):
        found = re.findall(pat, message, re.IGNORECASE)
        if len(found) >= 2:
            return found[0].strip(), found[1].strip()

    m = re.search(
        r"between\s+(?:user\s+)?#?([a-zA-Z0-9_]+)\s+and\s+(?:user\s+)?#?([a-zA-Z0-9_]+)",
        message,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    m2 = re.search(
        r"from\s+(?:user\s+)?#?([a-zA-Z0-9_]+)\s+to\s+(?:user\s+)?#?([a-zA-Z0-9_]+)",
        message,
        re.IGNORECASE,
    )
    if m2:
        return m2.group(1).strip(), m2.group(2).strip()

    return None, None


class TwoPathUserIds(BaseModel):
    user_a: str = Field(
        default="",
        description="First user id or source_id as in the graph (e.g. 1, fb_1). Empty if unknown.",
    )
    user_b: str = Field(
        default="",
        description="Second user id or source_id. Empty if unknown.",
    )


_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract exactly two user identifiers for a social graph query. "
            "Use the same id format the user used (1, 2, fb_1, user_1). "
            "user_a is the first user mentioned (or the smaller id if order is clear). user_b is the other user.\n"
            "{format_instructions}",
        ),
        ("human", "Question:\n{message}"),
    ]
)


def extract_two_user_ids_with_llm(message: str) -> Tuple[Optional[str], Optional[str]]:
    """LLM-based extraction when regex fails (requires API key and USE_LLM)."""
    if not USE_LLM:
        return None, None
    try:
        model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)
        parser = PydanticOutputParser(pydantic_object=TwoPathUserIds)
        chain = _EXTRACT_PROMPT.partial(
            format_instructions=parser.get_format_instructions()
        ) | model | parser
        out: TwoPathUserIds = chain.invoke({"message": message})
        a, b = (out.user_a or "").strip(), (out.user_b or "").strip()
        if a and b and a != b:
            return a, b
    except Exception as e:
        logger.warning("LLM two-user id extract failed: %s", e)
    return None, None


def extract_two_user_ids(message: str) -> Tuple[Optional[str], Optional[str]]:
    """Regex first, then LLM."""
    a, b = extract_two_user_ids_regex(message)
    if a and b:
        return a, b
    return extract_two_user_ids_with_llm(message)


def connection_path_result_row(
    user_a: str, user_b: str, path_payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Single `results` entry mirroring the connection-path API for chat responses."""
    return {
        "user_a": user_a,
        "user_b": user_b,
        "shortest_path": path_payload.get("shortest_path"),
        "common_friends": path_payload.get("common_friends", []),
        "common_liked_posts": path_payload.get("common_liked_posts", []),
    }


def format_connection_path_insight(row: Dict[str, Any]) -> str:
    """Short human-readable summary (no second LLM call by default)."""
    sp = row.get("shortest_path") or {}
    if not sp:
        return "No path found between the two users in the graph (or one user is missing)."

    names = sp.get("node_names") or []
    rels = sp.get("rel_types") or []
    hops = sp.get("hops")
    if names:
        chain = " → ".join(str(n) for n in names)
        hop_str = f" ({hops} hops)" if hops is not None else ""
        rel_str = f" [relationships: {', '.join(str(r) for r in rels)}]" if rels else ""
        return f"Shortest path{hop_str}: {chain}{rel_str}."
    return json.dumps(
        {k: row[k] for k in ("user_a", "user_b", "shortest_path", "common_friends", "common_liked_posts") if k in row},
        indent=2,
        default=str,
    )
