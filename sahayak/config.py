"""Central settings. Every tunable number lives here so you can experiment in ONE place."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY (and any overrides) from the .env file

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = str(ROOT / "chroma_db")
SQLITE_PATH = str(ROOT / "data" / "sahayak_memory.db")

# Model name is configurable because OpenAI retires models over time.
# (The curriculum says GPT-4o; if your account still serves it, set OPENAI_MODEL="gpt-4o".)
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
EMBED_MODEL = "text-embedding-3-small"

# ---- RAG ----
TOP_K = 4
# Cosine distance: LOWER = more similar. Tune these after running inspect_kb.py.
STRONG_MATCH = 0.55
WEAK_MATCH = 0.75

# ---- Memory ----
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "sqlite")  # "sqlite" or "redis"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

BUFFER_TURNS = 4              # "buffer" strategy: only the last N turns are remembered
KEEP_RECENT_TURNS = 3         # summary/hybrid: this many latest turns stay word-for-word
SUMMARY_TRIGGER_TOKENS = 700  # when un-summarised history passes this, compress the old part
                              # (deliberately SMALL so you can watch it happen in a demo;
                              #  in production this would be thousands)
MAX_HISTORY_TOKENS = 3000     # hard safety cap: drop oldest turns if still above this

STRATEGIES = ("buffer", "summary", "entity", "hybrid")


def estimate_tokens(text: str) -> int:
    """Rough rule of thumb: 1 token ~ 4 characters of English. Good enough for budgeting."""
    return max(1, len(text) // 4)
