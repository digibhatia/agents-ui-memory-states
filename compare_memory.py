"""Run the SAME 12-turn script under all four memory strategies and print a trade-off table:
tokens sent (cost), average latency, and whether the final RECALL question was answered correctly.

    python compare_memory.py

This makes ~12 turns x 4 strategies of real API calls - expect a couple of minutes.
Judge 'recall ok' yourself: the last answer should contain 'Priya', '5G Core' and 'laptop'.
"""
from demo_conversation import TURNS
from sahayak.service import KnowledgeBase, build_agent, get_backend, make_manager, run_turn
from sahayak import config

store, kb, agent = get_backend(), KnowledgeBase(), build_agent()
rows = []
for strat in config.STRATEGIES:
    session, user = f"cmp-{strat}", f"cmp-user-{strat}"
    store.clear_session(session)
    for k in list(store.get_facts(user)):
        store.delete_fact(user, k)
    mgr = make_manager(store, strat)
    toks = lat = 0
    for q in TURNS:
        r = run_turn(agent, mgr, kb, session, user, q)
        toks += (r["input_tokens"] or 0) + (r["output_tokens"] or 0)
        lat += r["latency_s"]
    a = r["answer"].answer.lower()
    ok = all(w in a for w in ("priya", "5g", "laptop"))
    rows.append((strat, toks, round(lat / len(TURNS), 2), "YES" if ok else "NO", r["memory"]["total_memory_tokens"]))

print(f"\n{'strategy':<10}{'total tokens':>14}{'avg latency s':>15}{'recall ok':>11}{'final memory tok':>18}")
for r in rows:
    print(f"{r[0]:<10}{r[1]:>14}{r[2]:>15}{r[3]:>11}{r[4]:>18}")
print("\nDiscuss: which strategy would you ship for 5 turns? For 500? Who pays for the extra LLM calls?")
