"""OFFLINE test of the memory logic - no API key, no network. Proves overflow handling works.

    python selftest_memory.py
"""
import os
import tempfile

from sahayak import config
from sahayak.memory import MemoryManager
from sahayak.store import SQLiteStore


def fake_summarize(prev, transcript):
    return f"SUMMARY[{len(transcript.splitlines())} lines]"


def fake_extract(text, existing):
    return {"name": text.split("I'm ")[1].split(",")[0]} if "I'm " in text else {}


def run(strategy, turns=20):
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    mgr = MemoryManager(SQLiteStore(path), fake_summarize, fake_extract, strategy)
    for i in range(turns):
        user = "Hi I'm Priya, hello." if i == 0 else f"question {i} " + "x" * 300
        mgr.record_turn("s", "u", user, "answer " + "y" * 300)
    return mgr.build_context("s", "u"), mgr


for s in config.STRATEGIES:
    ctx, mgr = run(s)
    print(f"{s:<8} replayed={ctx.stats['replayed_messages']:>2}  tokens={ctx.stats['total_memory_tokens']:>4}  "
          f"summary={ctx.stats['has_summary']}  block={'name: Priya' in ctx.memory_block}")

# assertions = the lesson in code
ctx, _ = run("buffer");  assert ctx.stats["replayed_messages"] == config.BUFFER_TURNS * 2   # window
ctx, _ = run("buffer");  assert "Priya" not in str(ctx.history)                              # forgot turn 1
ctx, _ = run("summary"); assert ctx.stats["has_summary"] and ctx.stats["replayed_messages"] <= 2 * 20
ctx, _ = run("entity");  assert "name: Priya" in ctx.memory_block
ctx, _ = run("hybrid");  assert "name: Priya" in ctx.memory_block and "SUMMARY[" in ctx.memory_block
ctx, _ = run("hybrid", turns=200); assert ctx.stats["history_tokens"] <= config.MAX_HISTORY_TOKENS  # never overflows
print("\nAll memory self-tests passed.")
