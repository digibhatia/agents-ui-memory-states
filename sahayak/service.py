"""Wires everything together so the UI, the demo script and the comparison script share one setup."""
from . import config
from .agent import build_agent, build_single_tool_agent, make_fact_extractor, make_summarizer, run_turn
from .kb import KnowledgeBase
from .memory import MemoryManager
from .store import get_store


def get_backend():
    return get_store(config.MEMORY_BACKEND, config.SQLITE_PATH, config.REDIS_URL)


def make_manager(store, strategy):
    return MemoryManager(store, summarize_fn=make_summarizer(), extract_fn=make_fact_extractor(), strategy=strategy)


__all__ = ["get_backend", "make_manager", "build_agent", "build_single_tool_agent", "KnowledgeBase", "run_turn"]
