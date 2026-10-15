"""STEP 3 (Terminal 2) - a hard-coded 12-turn conversation, run from a SEPARATE process.

    python demo_conversation.py                 # uses session "demo", strategy hybrid
    python demo_conversation.py buffer          # same script, buffer-only memory

Because memory lives in SQLite (not in RAM), open the UI with Session ID = demo (user = priya)
and you will see this entire conversation, the facts and the summary that were built.
Edit TURNS below, then re-run - that is the exercise.
"""
import sys

from sahayak.service import KnowledgeBase, build_agent, get_backend, make_manager, run_turn

TURNS = [
    "Hi! I'm Priya, I work in the 5G Core team in Pune.",                            # 1  plant facts
    "How many casual leave days do I get in a year?",                               # 2  RAG
    "Can I carry them forward to next year?",                                       # 3  follow-up (needs turn 2)
    "What about sick leave - do I need a certificate?",                             # 4
    "How many days a week can I work from home?",                                   # 5
    "My laptop was stolen yesterday. What should I do?",                            # 6  IT doc
    "How long before I get a replacement?",                                         # 7
    "I want to take leave from 2026-12-21 to 2026-12-31. How many working days is that?",  # 8  multi-tool
    "What is the reimbursement limit for domestic hotel stays?",                     # 9
    "Do I need manager approval for travel?",                                       # 10
    "How often do I have to change my password?",                                   # 11
    "Quick check: what's my name, which team am I on, and what did I ask about my laptop?",  # 12 RECALL TEST
]

if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    session, user = "demo", "priya"
    store, kb, agent = get_backend(), KnowledgeBase(), build_agent()
    store.clear_session(session)
    manager = make_manager(store, strategy)
    print(f"Strategy: {strategy}\n")
    for i, q in enumerate(TURNS, 1):
        r = run_turn(agent, manager, kb, session, user, q)
        m = r["memory"]
        print(f"--- Turn {i} ---\nUSER : {q}\nAGENT: {r['answer'].answer}")
        print(f"       [sources={r['answer'].sources} conf={r['answer'].confidence} "
              f"tools={[t['tool'] for t in r['trace']]} {r['latency_s']}s "
              f"| memory: {m['replayed_messages']} msgs, {m['total_memory_tokens']} tok, summary={m['has_summary']}]\n")
    print("Facts remembered:", store.get_facts(user))
    print("Summary:", store.get_summary(session)[0] or "(none)")
