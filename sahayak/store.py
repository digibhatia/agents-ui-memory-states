#store.py
"""Memory persistence: where conversation memory physically lives.
 
Two interchangeable backends with the SAME methods:
  * SQLiteStore - a file on disk. Zero setup. Survives restarts. Default.
  * RedisStore  - an in-memory server. Faster, shared across many app servers.
 
Three kinds of data are stored:
  messages : every user/assistant message (the raw transcript)
  summary  : ONE running summary per session (older turns compressed)
  facts    : key/value facts about a USER (entity memory) - shared across sessions
"""
import json
import sqlite3
import time
from pathlib import Path
 
 
class SQLiteStore:
    def __init__(self, path: str):
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
 
        # check_same_thread=False: Streamlit may call us from different threads.
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, ts REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS summaries(
                session_id TEXT PRIMARY KEY, summary TEXT NOT NULL,
                summarized_upto INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS facts(
                user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                updated REAL NOT NULL, PRIMARY KEY(user_id, key));
            """
        )
        self.db.commit()
 
    # ---- messages ----
    def add_message(self, session_id, role, content):
        cur = self.db.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?,?,?,?)",
            (session_id, role, content, time.time()),
        )
        self.db.commit()
        return cur.lastrowid
 
    def get_messages(self, session_id, after_id=0):
        rows = self.db.execute(
            "SELECT id, role, content FROM messages WHERE session_id=? AND id>? ORDER BY id",
            (session_id, after_id),
        ).fetchall()
        return [{"id": r[0], "role": r[1], "content": r[2]} for r in rows]
 
    # ---- summary ----
    def get_summary(self, session_id):
        """Returns (summary_text, summarized_upto_message_id)."""
        r = self.db.execute(
            "SELECT summary, summarized_upto FROM summaries WHERE session_id=?", (session_id,)
        ).fetchone()
        return (r[0], r[1]) if r else ("", 0)
 
    def set_summary(self, session_id, summary, upto_id):
        self.db.execute(
            "INSERT INTO summaries(session_id, summary, summarized_upto) VALUES (?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, "
            "summarized_upto=excluded.summarized_upto",
            (session_id, summary, upto_id),
        )
        self.db.commit()
 
    # ---- facts (entity memory) ----
    def upsert_fact(self, user_id, key, value):
        self.db.execute(
            "INSERT INTO facts(user_id, key, value, updated) VALUES (?,?,?,?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (user_id, key, value, time.time()),
        )
        self.db.commit()
 
    def get_facts(self, user_id):
        rows = self.db.execute("SELECT key, value FROM facts WHERE user_id=? ORDER BY key", (user_id,))
        return {k: v for k, v in rows.fetchall()}
 
    def delete_fact(self, user_id, key):
        self.db.execute("DELETE FROM facts WHERE user_id=? AND key=?", (user_id, key))
        self.db.commit()
 
    # ---- housekeeping ----
    def list_sessions(self):
        rows = self.db.execute(
            "SELECT session_id, COUNT(*) FROM messages GROUP BY session_id ORDER BY MAX(id) DESC"
        ).fetchall()
        return [{"session_id": r[0], "messages": r[1]} for r in rows]
 
    def clear_session(self, session_id):
        self.db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        self.db.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
        self.db.commit()
 
 
class RedisStore:
    """Same interface, backed by Redis. Needs `pip install redis` and a running server.
 
    Key layout:
      sahayak:{session}:msgs     LIST of JSON messages (RPUSH keeps order)
      sahayak:{session}:counter  INCR counter for message ids
      sahayak:{session}:summary  HASH {summary, upto}
      sahayak:user:{user}:facts  HASH {key: value}
    Trade-off vs SQLite: much faster and shareable between servers, but data lives in RAM
    (persistence depends on Redis AOF/RDB settings) and there is no SQL to query it with.
    """
 
    def __init__(self, url: str):
        import redis  # lazy import: only needed when this backend is chosen
 
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.r.ping()  # fail fast with a clear error if the server is down
 
    def _k(self, s, name):
        return f"sahayak:{s}:{name}"
 
    def add_message(self, session_id, role, content):
        mid = self.r.incr(self._k(session_id, "counter"))
        self.r.rpush(self._k(session_id, "msgs"), json.dumps({"id": mid, "role": role, "content": content}))
        return mid
 
    def get_messages(self, session_id, after_id=0):
        raw = self.r.lrange(self._k(session_id, "msgs"), 0, -1)
        return [m for m in map(json.loads, raw) if m["id"] > after_id]
 
    def get_summary(self, session_id):
        h = self.r.hgetall(self._k(session_id, "summary"))
        return (h.get("summary", ""), int(h.get("upto", 0))) if h else ("", 0)
 
    def set_summary(self, session_id, summary, upto_id):
        self.r.hset(self._k(session_id, "summary"), mapping={"summary": summary, "upto": upto_id})
 
    def upsert_fact(self, user_id, key, value):
        self.r.hset(f"sahayak:user:{user_id}:facts", key, value)
 
    def get_facts(self, user_id):
        return dict(sorted(self.r.hgetall(f"sahayak:user:{user_id}:facts").items()))
 
    def delete_fact(self, user_id, key):
        self.r.hdel(f"sahayak:user:{user_id}:facts", key)
 
    def list_sessions(self):
        out = []
        for k in self.r.scan_iter("sahayak:*:msgs"):
            out.append({"session_id": k.split(":")[1], "messages": self.r.llen(k)})
        return out
 
    def clear_session(self, session_id):
        self.r.delete(self._k(session_id, "msgs"), self._k(session_id, "counter"), self._k(session_id, "summary"))
 
 
def get_store(backend: str, sqlite_path: str, redis_url: str):
    return RedisStore(redis_url) if backend == "redis" else SQLiteStore(sqlite_path)

has context menu
