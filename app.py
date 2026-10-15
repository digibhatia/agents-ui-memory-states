"""STEP 2 (Terminal 1) - the Sahayak chat UI.

    streamlit run app.py

Left sidebar  : choose user, session, memory strategy, single/multi-tool agent. Memory inspector.
Main area     : chat. Under every answer: sources, confidence, tool trace, tokens, latency.
Streamlit re-runs this whole file on every click, so anything that must survive lives in
st.session_state (UI objects) or in the memory store (conversation data).
"""
import streamlit as st

from sahayak import config
from sahayak.service import (KnowledgeBase, build_agent, build_single_tool_agent, get_backend,
                             make_manager, run_turn)

st.set_page_config(page_title="Sahayak", page_icon="🧭", layout="wide")


@st.cache_resource  # created once per server, reused across clicks
def get_shared():
    return get_backend(), KnowledgeBase()


@st.cache_resource
def get_agents():
    return {"multi-tool (RAG + dates)": build_agent(), "single-tool (RAG only)": build_single_tool_agent()}


store, kb = get_shared()
agents = get_agents()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧭 Sahayak")
user_id = st.sidebar.text_input("User ID", "priya")
session_id = st.sidebar.text_input("Session ID", "session-1",
                                   help="Use 'demo' to see the conversation run by demo_conversation.py")
strategy = st.sidebar.selectbox("Memory strategy", config.STRATEGIES, index=3,
                                help="buffer / summary / entity / hybrid - same question, different memory")
agent_kind = st.sidebar.radio("Agent", list(agents))
st.sidebar.caption(f"Model: {config.MODEL} · Backend: {config.MEMORY_BACKEND}")

if st.sidebar.button("🗑 Clear this session's chat"):
    store.clear_session(session_id)
    st.session_state.pop("traces", None)
    st.rerun()

manager = make_manager(store, strategy)
ctx = manager.build_context(session_id, user_id)  # NOTE: may trigger a summary - that's intended

st.sidebar.subheader("Memory inspector")
c1, c2 = st.sidebar.columns(2)
c1.metric("Replayed msgs", ctx.stats["replayed_messages"])
c2.metric("Memory tokens", ctx.stats["total_memory_tokens"])
st.sidebar.progress(min(1.0, ctx.stats["history_tokens"] / config.SUMMARY_TRIGGER_TOKENS),
                    text=f"History vs summary trigger ({config.SUMMARY_TRIGGER_TOKENS} tok)")
facts = store.get_facts(user_id)
st.sidebar.markdown("**Remembered facts** (entity memory)")

if facts:
    st.sidebar.json(facts)
else:
    st.sidebar.caption("none yet - tell Sahayak your name and team")

facts = store.get_facts(user_id)

st.sidebar.markdown("**Remembered facts** (entity memory)")

if facts:
    st.sidebar.json(facts)
else:
    st.sidebar.caption("none yet - tell Sahayak your name and team")

# ------------------------------------------------------------------ chat area
st.title("Sahayak - enterprise assistant")
st.caption("Ask HR / IT policy questions. Try telling it your name and team, then ask 10+ follow-ups.")

traces = st.session_state.setdefault("traces", {})  # message id -> trace info, for display only
for m in store.get_messages(session_id):  # full transcript from the DB (survives refresh/restart)
    with st.chat_message(m["role"]):
        st.write(m["content"])
        t = traces.get(m["id"])
        if t:
            st.caption(f"Sources: {', '.join(t['sources']) or '-'} · confidence: {t['confidence']} · "
                       f"{t['latency_s']}s · {t['llm_calls']} LLM call(s) · {t['input_tokens']} in / {t['output_tokens']} out tokens")
            if t["trace"]:
                with st.expander(f"🔧 Tool trace ({len(t['trace'])} call(s))"):
                    st.json(t["trace"])

if prompt := st.chat_input("Ask Sahayak..."):
    with st.chat_message("user"):
        st.write(prompt)
    with st.spinner("Thinking..."):
        r = run_turn(agents[agent_kind], manager, kb, session_id, user_id, prompt)
    last_id = store.get_messages(session_id)[-1]["id"]
    traces[last_id] = {"sources": r["answer"].sources, "confidence": r["answer"].confidence, **{
        k: r[k] for k in ("latency_s", "llm_calls", "input_tokens", "output_tokens", "trace")}}
    st.rerun()
