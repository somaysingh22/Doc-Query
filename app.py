import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import json
import tempfile
import traceback
import re
import sqlite3
import pathlib
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# streamlit rerun problem solver
def safe_rerun():
    """Attempt rerun safely across Streamlit versions. Uses st.experimental_rerun if available,
    otherwise toggles st.query_params to force a rerun. Falls back to st.stop()."""
    # Try official API first (older Streamlit still exposes experimental_rerun)
    try:
        if hasattr(st, "experimental_rerun"):
            return st.experimental_rerun()
    except Exception:
        pass

    # Newer Streamlit: toggle st.query_params
    try:
        # Read current params (copy to regular dict)
        params = dict(st.query_params) if hasattr(st, "query_params") else {}

        # Compute toggle value (string values expected)
        current_toggle = params.get("_r", ["0"])[0] if isinstance(params.get("_r"), list) else params.get("_r", "0")
        toggle = "0" if current_toggle == "1" else "1"

        # Assign back to st.query_params (setter)
        # Ensure all values are string or list-of-strings to be safe
        new_params = {}
        for k, v in params.items():
            # keep existing params, convert single values to list-of-strings
            if isinstance(v, list):
                new_params[k] = v
            else:
                new_params[k] = [str(v)]
        new_params["_r"] = [toggle]
        st.query_params = new_params
        return
    except Exception:
        pass

    # Final fallback: stop app (user can manually refresh)
    try:
        st.stop()
    except Exception:
        return
# --------------------------------------------------------------------------------------------


# ---------------- SQLite-backed session storage ----------------
DB_DIR = pathlib.Path("./sessions")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "sessions.db"

def get_db_conn():
    conn = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TEXT,
        updated_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        seq INTEGER,
        role TEXT,
        text TEXT,
        ts TEXT,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages(session_id, seq)")
    conn.commit()
    cur.close()
    conn.close()

init_db()

def list_saved_sessions() -> List[str]:
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT session_id FROM sessions ORDER BY COALESCE(updated_at, '') DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r["session_id"] for r in rows]

def load_session(session_id: str) -> List[Dict]:
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT seq, role, text, ts FROM messages WHERE session_id = ? ORDER BY seq ASC", (session_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    msgs = []
    for r in rows:
        msgs.append({"role": r["role"], "text": r["text"], "ts": r["ts"]})
    return msgs

def save_session(session_id: str, messages: List[Dict]):
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    conn = get_db_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    # upsert session metadata
    cur.execute("""
        INSERT OR REPLACE INTO sessions(session_id, created_at, updated_at)
        VALUES (?, COALESCE((SELECT created_at FROM sessions WHERE session_id = ?), ?), ?)
    """, (session_id, session_id, now, now))
    # replace messages atomically
    cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    seq = 1
    for m in messages:
        role = m.get("role", "")
        text = m.get("text", "")
        ts = m.get("ts", datetime.utcnow().isoformat())
        cur.execute("INSERT INTO messages(session_id, seq, role, text, ts) VALUES (?, ?, ?, ?, ?)",
                    (session_id, seq, role, text, ts))
        seq += 1
    cur.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (datetime.utcnow().isoformat(), session_id))
    conn.commit()
    cur.close()
    conn.close()

def save_message(session_id: str, role: str, text: str, ts: str = None):
    if ts is None:
        ts = datetime.utcnow().isoformat()
    conn = get_db_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("INSERT OR IGNORE INTO sessions(session_id, created_at, updated_at) VALUES (?, ?, ?)", (session_id, now, now))
    cur.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM messages WHERE session_id = ?", (session_id,))
    next_seq = cur.fetchone()["next_seq"]
    cur.execute("INSERT INTO messages(session_id, seq, role, text, ts) VALUES (?, ?, ?, ?, ?)", (session_id, next_seq, role, text, ts))
    cur.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (datetime.utcnow().isoformat(), session_id))
    conn.commit()
    cur.close()
    conn.close()

def delete_session_file(session_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    cur.close()
    conn.close()

def export_session_json(session_id: str) -> str:
    msgs = load_session(session_id)
    return json.dumps(msgs, ensure_ascii=False, indent=2)

# ---------------- End DB helpers ----------------

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up Streamlit 
st.set_page_config(page_title="Conversational RAG With PDF", layout="wide")
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# --- Sidebar UI controls ---
with st.sidebar:
    demo_mode = st.checkbox("Demo mode (no API keys, offline)", value=False)
    top_k = st.number_input("Retriever top_k (how many chunks to retrieve)", min_value=1, max_value=20, value=6)
    chunk_size = st.number_input("Chunk size", min_value=256, max_value=8000, value=5000)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=500)
    show_sentence_source = st.checkbox("Show sentence → source mapping", value=False)
    show_retrieved_ui = st.checkbox("Show retrieved evidence (top results)", value=True)
    show_sources_ui = st.checkbox("Show sources (from retrieved chunks)", value=True)

# API key only required when NOT in demo mode
api_key = st.text_input("Enter your Groq API key:", type="password") if not demo_mode else ""

# make session_id available early
session_id = st.text_input("Session ID", value="default_session")

# session manager in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### Session manager")
    saved_sessions = list_saved_sessions()
    options = ["(none)"] + saved_sessions
    selected_session = st.selectbox("View saved session", options=options, index=0)
    if selected_session != "(none)":
        msgs = load_session(selected_session)
        with st.expander(f"Session: {selected_session} (messages: {len(msgs)})", expanded=False):
            if not msgs:
                st.write("_No messages in this session._")
            else:
                for m in msgs:
                    ts = m.get("ts", "")
                    ts_hr = ts
                    try:
                        ts_hr = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
                    prefix = f"[{ts_hr}] "
                    if m.get("role") == "user":
                        st.write(f"**{prefix}User:** {m.get('text')}")
                    else:
                        st.write(f"**{prefix}Assistant:** {m.get('text')}")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.download_button(
                label="Download session JSON",
                data=export_session_json(selected_session),
                file_name=f"{selected_session}.json",
                mime="application/json",
            )
        with col2:
            if st.button("Delete session", key=f"delete_{selected_session}"):
                delete_session_file(selected_session)
                safe_rerun()
    st.markdown("---")

# ---------------- Helpers ----------------
def llm_text(res):
    try:
        if hasattr(res, "content"):
            c = getattr(res, "content")
            return c if isinstance(c, str) else str(c)
        if hasattr(res, "generations"):
            gens = getattr(res, "generations")
            if isinstance(gens, list) and gens:
                first = gens[0]
                if isinstance(first, list):
                    first = first[0]
                if hasattr(first, "text"):
                    return first.text
        return str(res)
    except Exception:
        return str(res)

def extractive_summary(text: str, max_sentences: int = 5) -> str:
    if not text or len(text.strip()) == 0:
        return ""
    sents = re.split(r'(?<=[.!?])\s+', text)
    if len(sents) <= max_sentences:
        return " ".join(sents)
    words = re.findall(r'\w+', text.lower())
    stopwords = set(["the","is","in","and","to","of","a","for","on","with","that","this","as","are","be","by","an","or","it","from","we","which","these"])
    freq = {}
    for w in words:
        if w in stopwords or len(w) < 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    scored = []
    for idx, s in enumerate(sents):
        s_words = re.findall(r'\w+', s.lower())
        score = sum(freq.get(w,0) for w in s_words)
        scored.append((score, idx, s))
    scored_top = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_sentences]
    scored_top_sorted = sorted(scored_top, key=lambda x: x[1])
    summary = " ".join([t[2].strip() for t in scored_top_sorted])
    return summary

# Demo helpers
def simple_generate_questions_for_doc(docs, max_q=5) -> List[str]:
    text = "\n\n".join([d.page_content for d in docs])
    text_low = text.lower()
    qs = []
    if "compatib" in text_low or "python" in text_low:
        qs.append("What versions of Python does this document say are supported or planned?")
    if "deprecat" in text_low or "old_func" in text_low or "new_func" in text_low:
        qs.append("Which functions are deprecated and what are their suggested replacements?")
    if "community" in text_low or "build" in text_low or "ci" in text_low:
        qs.append("What do community test results say about compatibility or stability?")
    if "migration" in text_low:
        qs.append("Are there migration instructions mentioned for breaking changes?")
    generic = [
        "What is the main purpose of this document?",
        "List important notes or warnings mentioned in the document.",
        "What change is anticipated in the next version?"
    ]
    for g in generic:
        if len(qs) >= max_q: break
        qs.append(g)
    return qs[:max_q] if qs else generic[:max_q]

def simple_synthesize(retrieved_docs: List, question: str, top_n_sentences: int = 3) -> Dict:
    q_tokens = set(re.findall(r"\w+", question.lower()))
    candidates = []
    for doc in retrieved_docs:
        content = doc.page_content
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "")
        sents = re.split(r'(?<=[.!?])\s+', content)
        for s in sents:
            stokens = set(re.findall(r"\w+", s.lower()))
            score = len(q_tokens & stokens)
            if score > 0:
                candidates.append({"sent": s.strip(), "score": score, "source": source, "page": page})
    if not candidates:
        for doc in retrieved_docs[:top_n_sentences]:
            s = (doc.page_content.split(".")[0] + ".").strip()
            m = getattr(doc, "metadata", {}) or {}
            candidates.append({"sent": s, "score": 0, "source": m.get("source", "unknown"), "page": m.get("page", "")})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    picked = []
    used_text = set()
    for c in candidates:
        txt = re.sub(r'\s+', ' ', c["sent"]).strip()
        if txt and txt not in used_text:
            used_text.add(txt)
            picked.append(c)
        if len(picked) >= top_n_sentences:
            break
    answer_parts = []
    for p in picked:
        tag = f"(source: {p['source']}{' p.'+str(p['page']) if p['page'] else ''})"
        answer_parts.append(f"{p['sent']} {tag}")
    answer = " ".join(answer_parts)
    sources = []
    for p in picked:
        if p["source"] not in sources:
            sources.append(p["source"])
    return {"answer": answer, "sources": sources, "picked": picked}

# ---------------- Main app logic ----------------
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'auto_questions' not in st.session_state:
    st.session_state.auto_questions = {}
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

# demo sample pdfs (if you provide them in /mnt/data they'll auto-load in demo mode)
demo_files_paths = ["/mnt/data/test_pdf_A.pdf", "/mnt/data/test_pdf_B.pdf", "/mnt/data/test_pdf_C.pdf"]

uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)

# auto-load demo files if demo_mode on and nothing uploaded
if demo_mode and not uploaded_files:
    uploaded_files = []
    for p in demo_files_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                class SimpleUpload:
                    def __init__(self, name, data):
                        self.name = name
                        self._data = data
                    def getvalue(self):
                        return self._data
                uploaded_files.append(SimpleUpload(os.path.basename(p), f.read()))

# ---------------- Process uploaded PDFs ----------------
if uploaded_files:
    documents = []
    file_docs_map = {}
    with st.spinner("Processing uploaded PDF(s) — extracting text and building embeddings..."):
        try:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1] or ".pdf") as tf:
                    tf.write(uploaded_file.getvalue())
                    tmp_path = tf.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for i, d in enumerate(docs):
                    if not d.metadata:
                        d.metadata = {}
                    if "page" not in d.metadata:
                        d.metadata["page"] = i + 1
                    d.metadata["source"] = uploaded_file.name
                documents.extend(docs)
                file_docs_map[uploaded_file.name] = docs

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": int(top_k)})

            # Generate 5 q per file
            st.session_state.auto_questions = {}
            for fname, docs_for_file in file_docs_map.items():
                combined_text = "\n\n".join([d.page_content for d in docs_for_file])
                safe_text = combined_text[:15000]

                # Summarize (LLM when available, else extractive fallback)
                if not demo_mode:
                    summary_prompt = f"""
You are an expert at summarizing documents. Summarize the following text into 5-7 very short bullet points (one sentence each). Keep it concise.

Text:
{safe_text}
"""
                    try:
                        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
                        summary_res = llm.invoke(summary_prompt)
                        summary_text = llm_text(summary_res)
                        st.write(f"**[DEBUG] Summary for {fname} (first 300 chars):**")
                        st.write(summary_text[:300])
                    except Exception as e:
                        st.warning(f"[DEBUG] Summary generation failed for {fname}: {e}")
                        st.text(traceback.format_exc())
                        summary_text = extractive_summary(safe_text, max_sentences=5)
                        st.write(f"**[DEBUG] Fallback summary for {fname}:**")
                        st.write(summary_text[:500])
                else:
                    summary_text = extractive_summary(safe_text, max_sentences=5)

                # Question generation
                if demo_mode:
                    st.session_state.auto_questions[fname] = simple_generate_questions_for_doc(docs_for_file, max_q=5)
                else:
                    question_prompt = f"""
You are a helpful assistant that generates exam-style or study questions.
Based on the summary below, generate EXACTLY 5 high-quality, diverse, and meaningful questions someone should ask to understand this document.
Return ONLY a valid JSON array of 5 strings and nothing else. Example:
["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]

Summary:
{summary_text}
"""
                    try:
                        q_res = llm.invoke(question_prompt)
                        q_text = llm_text(q_res).strip()
                        st.write(f"**[DEBUG] Raw question-gen response for {fname}:**")
                        st.write(q_text[:1000])

                        parsed_questions = None
                        try:
                            parsed_questions = json.loads(q_text)
                        except Exception:
                            start = q_text.find('[')
                            end = q_text.rfind(']') + 1
                            if start != -1 and end != -1:
                                snippet = q_text[start:end]
                                try:
                                    parsed_questions = json.loads(snippet)
                                except Exception:
                                    parsed_questions = None

                        if not isinstance(parsed_questions, list):
                            lines = [ln.strip() for ln in q_text.splitlines() if ln.strip()]
                            cleaned_lines = []
                            for ln in lines:
                                ln2 = ln.lstrip('-•0123456789. ').strip().strip('"').strip("'")
                                if ln2:
                                    cleaned_lines.append(ln2)
                            if cleaned_lines:
                                parsed_questions = cleaned_lines[:5]

                        if isinstance(parsed_questions, list) and parsed_questions:
                            cleaned = [str(x).strip() for x in parsed_questions][:5]
                            if len(cleaned) < 5:
                                cleaned += ["(Not generated)"] * (5 - len(cleaned))
                            st.session_state.auto_questions[fname] = cleaned
                        else:
                            st.warning(f"[DEBUG] Could not parse questions from LLM for {fname}. Falling back to heuristic generator.")
                            text_low = combined_text.lower()
                            fallback = []
                            if "python" in text_low or "compatib" in text_low:
                                fallback.append("What Python versions are discussed or supported?")
                            if "deprecat" in text_low or "old_func" in text_low:
                                fallback.append("Which functions are deprecated and what are the replacements?")
                            if "model" in text_low or "diffusion" in text_low or "transformer" in text_low:
                                fallback.append("What model/architecture is the document about and what are its key traits?")
                            generic_qs = [
                                "What is the primary purpose of this document?",
                                "What are the main conclusions or recommendations?",
                                "List any important limitations or warnings mentioned."
                            ]
                            for g in generic_qs:
                                if len(fallback) >= 5: break
                                fallback.append(g)
                            st.session_state.auto_questions[fname] = fallback[:5]

                    except Exception as e:
                        st.error(f"[DEBUG] Question generation LLM call failed for {fname}: {e}")
                        st.text(traceback.format_exc())
                        text_low = combined_text.lower()
                        fallback = []
                        if "python" in text_low or "compatib" in text_low:
                            fallback.append("What Python versions are discussed or supported?")
                        if "deprecat" in text_low or "old_func" in text_low:
                            fallback.append("Which functions are deprecated and what are the replacements?")
                        generic_qs = [
                            "What is the primary purpose of this document?",
                            "What are the main conclusions or recommendations?",
                            "List any important limitations or warnings mentioned."
                        ]
                        for g in generic_qs:
                            if len(fallback) >= 5: break
                            fallback.append(g)
                        st.session_state.auto_questions[fname] = fallback[:5]

        except Exception:
            st.error("There was an error processing uploaded PDF(s). See traceback below.")
            st.text(traceback.format_exc())
else:
    st.info("Upload one or more PDFs (or enable Demo mode to auto-load sample PDFs).")

# ---------------- Display generated questions ----------------
st.subheader("🔎 Suggested Questions From Each Uploaded PDF")
if st.session_state.auto_questions:
    for fname, qlist in st.session_state.auto_questions.items():
        st.markdown(f"**File:** {fname}")
        for i, q in enumerate(qlist, start=1):
            col1, col2 = st.columns([12, 1])
            col1.write(f"**Q{i}:** {q}")
            if col2.button("Ask", key=f"ask_{fname}_{i}"):
                st.session_state.user_question = q
else:
    st.write("No suggested questions available for the uploaded files.")

# ---------------- History-aware retriever & QA chain ----------------
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# helper: reconstruct ChatMessageHistory from DB (so LLM sees prior turns)
def get_session_history(session: str) -> BaseChatMessageHistory:
    # prefer cached in-memory instance to avoid re-creating repeatedly
    key = f"_chat_history_obj__{session}"
    if key in st.session_state:
        return st.session_state[key]

    # create fresh ChatMessageHistory and populate from DB
    hist = ChatMessageHistory()
    msgs = load_session(session)
    # try to use add_user_message / add_assistant_message if present
    added_via_api = False
    try:
        add_user = getattr(hist, "add_user_message", None)
        add_assistant = getattr(hist, "add_assistant_message", None)
        if callable(add_user) and callable(add_assistant):
            for m in msgs:
                role = m.get("role", "user")
                text = m.get("text", "")
                if role == "user":
                    add_user(text)
                else:
                    add_assistant(text)
            added_via_api = True
    except Exception:
        added_via_api = False

    if not added_via_api:
        # fallback: try other method names or set .messages directly
        try:
            # some implementations use add_message or append
            add_msg = getattr(hist, "add_message", None)
            if callable(add_msg):
                for m in msgs:
                    add_msg({"role": m.get("role", "user"), "content": m.get("text", "")})
                added_via_api = True
        except Exception:
            added_via_api = False

    if not added_via_api:
        # last resort: set a .messages list (langchain internals may differ — best-effort)
        try:
            hist.messages = []
            for m in msgs:
                # store a minimal representation
                hist.messages.append({"role": m.get("role", "user") if m.get("role") == "user" else "assistant", "content": m.get("text", "")})
        except Exception:
            pass

    st.session_state[key] = hist
    return hist

# instantiate history_aware_retriever only when retriever exists & API used
# we'll create history_aware_retriever below after we have retriever and llm

# Build QA chain later when retriever exists (done in flow below)
conversational_rag_chain = None

# ---------------- User question input and response flow ----------------
# Single text input (key user_question)
user_input = st.text_input("Your question:", key="user_question")

if user_input:
    # persist the user message to DB immediately (so it's part of history)
    try:
        save_message(session_id, "user", user_input, datetime.utcnow().isoformat())
    except Exception:
        # non-fatal: log but continue
        st.warning("Warning: failed to save user message to DB.")

    # make sure history object is updated (invalidate cached history for session so it's rebuilt)
    hist_key = f"_chat_history_obj__{session_id}"
    if hist_key in st.session_state:
        del st.session_state[hist_key]

    # if we have a retriever + LLM path available use it
    if 'retriever' in locals() and api_key:
        # rebuild llm & retriever-aware chain
        try:
            llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
            # build QA chain
            if show_sentence_source:
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "You will be given several retrieved context chunks from possibly multiple documents (each chunk includes a source). "
                    "Synthesize information across the retrieved chunks to answer the user's question. "
                    "For every sentence in your final answer, APPEND a source tag in the exact format: (source: filename.pdf p.X). "
                    "If a sentence is supported by multiple sources, list them separated by semicolons inside the parentheses. "
                    "Place the source tag immediately after the sentence with one space before the '(' character. "
                    "At the end, if you can, include a short 'Sources:' line listing unique filenames used. Keep answers concise (max three sentences). "
                    "\n\n"
                    "{context}"
                )
            else:
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "You will be given several retrieved context chunks from possibly multiple documents (each chunk includes a source). "
                    "Synthesize information across the retrieved chunks to answer the user's question. "
                    "If your answer uses information from multiple documents, you may cite the sources in a final 'Sources:' line using the filename and page number where possible. "
                    "If the retrieved context is insufficient or contradictory, say so. Be concise (max three sentences)."
                    "\n\n"
                    "{context}"
                )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # invoke the chain
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
            assistant_text = response.get('answer') if isinstance(response, dict) else str(response)

            # show assistant
            st.markdown("### Assistant:")
            st.write(assistant_text)

            # persist assistant response to DB
            try:
                save_message(session_id, "assistant", assistant_text, datetime.utcnow().isoformat())
            except Exception:
                st.warning("Warning: failed to save assistant message to DB.")

            # display retrieved evidence if requested
            try:
                retrieved_docs = retriever.get_relevant_documents(user_input)
            except Exception:
                retrieved_docs = []
            if show_retrieved_ui:
                st.markdown("#### Retrieved evidence (top results)")
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs, start=1):
                        meta = getattr(doc, "metadata", {}) or {}
                        src = meta.get("source", "unknown")
                        page = meta.get("page", "")
                        preview = doc.page_content[:400].replace("\n", " ")
                        st.write(f"**{i}. Source:** {src} {f'| page: {page}' if page else ''}")
                        st.write(preview + ("..." if len(doc.page_content) > 400 else ""))
                else:
                    st.write("No retrieved documents to display.")
            if show_sources_ui:
                sources_set = []
                for d in retrieved_docs:
                    m = getattr(d, "metadata", {}) or {}
                    s = m.get("source", "unknown")
                    if s not in sources_set:
                        sources_set.append(s)
                st.markdown("#### Sources (from retrieved chunks)")
                st.write(", ".join(sources_set) if sources_set else "No explicit sources found.")

        except Exception:
            st.error("There was an error invoking the RAG chain. See traceback:")
            st.text(traceback.format_exc())

    else:
        # Demo/fallback synthesizer path — use local synthesizer and persist assistant reply
        try:
            retrieved_docs = retriever.get_relevant_documents(user_input) if 'retriever' in locals() else []
        except Exception:
            retrieved_docs = []

        if show_retrieved_ui:
            st.markdown("#### Retrieved evidence (top results)")
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs, start=1):
                    meta = getattr(doc, "metadata", {}) or {}
                    src = meta.get("source", "unknown")
                    page = meta.get("page", "")
                    preview = doc.page_content[:400].replace("\n", " ")
                    st.write(f"**{i}. Source:** {src} {f'| page: {page}' if page else ''}")
                    st.write(preview + ("..." if len(doc.page_content) > 400 else ""))

        synth = simple_synthesize(retrieved_docs, user_input, top_n_sentences=3)
        assistant_text = synth["answer"]
        st.markdown("### Assistant (Demo synthesizer):")
        st.write(assistant_text)

        # persist assistant message
        try:
            save_message(session_id, "assistant", assistant_text, datetime.utcnow().isoformat())
        except Exception:
            st.warning("Warning: failed to save assistant message to DB.")

        if show_sentence_source:
            st.markdown("#### Sentence → Source mapping (parsed)")
            pattern = re.compile(r'(.*?)(\(\s*source:[^)]+\))', flags=re.DOTALL)
            matches = list(pattern.finditer(assistant_text))
            if matches:
                for idx, m in enumerate(matches, start=1):
                    sentence = m.group(1).strip()
                    source_tag = m.group(2).strip().lstrip('(').rstrip(')')
                    sentence = re.sub(r'\s+', ' ', sentence)
                    st.write(f"**Sentence {idx}:** {sentence}")
                    st.write(f"- {source_tag}")
            else:
                st.write("Could not parse sentence→source tags from the assistant response. Showing raw assistant text above.")

        if show_sources_ui:
            st.markdown("#### Sources (from retrieved chunks)")
            st.write(", ".join(synth["sources"]) if synth["sources"] else "No explicit sources found.")

else:
    st.info("Enter a question (or click a suggested question).")
