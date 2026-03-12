import sqlite3
import sqlite_vec
import os
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NEW: Path for Long-Term Markdown Summaries ---
SUMMARY_DIR = "session_summaries"
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers model...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

DB_PATH = "chat_memory.sqlite"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_msg TEXT,
                assistant_msg TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding FLOAT[384]
            );
        """)
        conn.commit()

def embed_text(text: str) -> bytes:
    model = get_embedding_model()
    vector = model.encode(text) 
    return sqlite_vec.serialize_float32(vector)

def add_memory(session_id: str, user_msg: str, assistant_msg: str):
    if not session_id: return
    embedding_bytes = embed_text(user_msg)
    
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO chat_memory (session_id, user_msg, assistant_msg, timestamp, embedding)
            VALUES (?, ?, ?, datetime('now'), ?)
        """, (session_id, user_msg, assistant_msg, embedding_bytes))
        conn.commit()

def get_memories(session_id: str, query: str, top_k: int = 3) -> str:
    if not session_id or not query: return ""
    query_embedding_bytes = embed_text(query)

    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT user_msg, assistant_msg, timestamp, 
                   vec_distance_L2(embedding, ?) as distance
            FROM chat_memory
            WHERE session_id = ?
            ORDER BY distance ASC LIMIT ?;
        """, (query_embedding_bytes, session_id, top_k))
        rows = cursor.fetchall()
        
    if not rows: return ""
        
    memories_str = "--- PREVIOUS RELEVANT MEMORIES (Short-Term) ---\n"
    for row in rows:
        memories_str += f"[Time: {row['timestamp']}]\nUser: {row['user_msg']}\nAssistant: {row['assistant_msg']}\n---\n"
    return memories_str

# ==========================================
#  NEW RECURSIVE SUMMARY ENGINE
# ==========================================

def get_summary_path(session_id):
    return os.path.join(SUMMARY_DIR, f"{session_id}_summary.md")

def get_existing_summary(session_id):
    path = get_summary_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "No prior long-term summary exists for this session."

def summarize_and_rotate(session_id, llm):
    """Summarizes SQL logs into Markdown and clears the SQL table."""
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT user_msg, assistant_msg FROM chat_memory 
            WHERE session_id = ? ORDER BY timestamp ASC
        """, (session_id,))
        logs = cursor.fetchall()
    
    if not logs: return

    new_chat_segment = "\n".join([f"User: {l['user_msg']}\nAI: {l['assistant_msg']}" for l in logs])
    old_summary = get_existing_summary(session_id)

    prompt = f"""You are a Memory Manager. Update the existing session summary with the new conversation details.
    
    EXISTING LONG-TERM SUMMARY:
    {old_summary}

    NEW RECENT CONVERSATION:
    {new_chat_segment}

    TASK:
    Create a refined, unified Markdown summary. 
    1. Retain key facts, user preferences, and project names.
    2. Remove redundant or trivial interactions.
    3. Output ONLY the updated summary in bullet points. Do not include introductory text.
    """
    
    try:
        updated_summary = llm.invoke(prompt).content

        with open(get_summary_path(session_id), "w", encoding="utf-8") as f:
            f.write(f"# Recursive Session Summary: {session_id}\n")
            f.write(f"Last Rotation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(updated_summary)

        # Clear the SQL logs we just summarized to prevent duplicates
        with get_db_connection() as conn:
            conn.execute("DELETE FROM chat_memory WHERE session_id = ?", (session_id,))
            conn.commit()
        logger.info(f"Rotated memory to Markdown for session: {session_id}")
    except Exception as e:
        logger.error(f"Summarization failed: {e}")