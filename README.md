### Doc-Query — Conversational RAG over PDF Documents

A production-grade **Retrieval-Augmented Generation (RAG)** application built with **LangChain** and **Streamlit**. Upload one or more PDFs, ask natural language questions, and get source-cited answers grounded entirely in the uploaded documents — with full conversation memory, automatic question generation, and persistent session history.

---

### Features

- **PDF-grounded answers** — every response retrieved from uploaded documents only (no hallucination)
- **History-aware retrieval** — LangChain's `create_history_aware_retriever` reformulates follow-up questions into standalone queries before retrieval
- **Full conversational RAG chain** — `history_aware_retriever` → `create_stuff_documents_chain` → `create_retrieval_chain` → `RunnableWithMessageHistory`
- **Automatic question generation** — LLM generates 5 exam-style questions per uploaded document with 3-tier JSON parsing fallback
- **SQLite session persistence** — save, load, delete, and export conversations; WAL journal mode for concurrent read safety
- **Source attribution** — retrieved evidence shown with filename and page number per answer
- **LangSmith tracing** — end-to-end chain observability: prompts, retrieved chunks, token usage, latency
- **Demo mode** — fully offline fallback using extractive summarisation (`extractive_summary`) and keyword-overlap QA (`simple_synthesize`)
- **Configurable retrieval** — sidebar controls for chunk size, overlap, and top-k tunable per document type

---

## Project Structure

```
doc-query/
│
├── app.py                   # Main Streamlit application — all UI + chain logic
│
├── sessions/
│   └── sessions.db          # SQLite database (auto-created on first run)
│
├── .env                     # API keys (not committed)
├── .gitignore
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/somaysingh22/doc-query.git
cd doc-query
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here     # optional — for tracing
LANGCHAIN_TRACING_V2=true                          # optional
LANGCHAIN_PROJECT=doc-query                        # optional
```

- Get your Groq API key at: https://console.groq.com
- Get your HuggingFace token at: https://huggingface.co/settings/tokens
- Get your LangSmith key at: https://smith.langchain.com

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## How to Use

### Querying Documents

1. Enter your **Groq API key** in the sidebar
2. Enter a **Session ID** (e.g. `my-session-1`) to name your conversation
3. Upload one or more **PDF files** using the file uploader
4. Wait for processing — chunks are embedded and indexed into ChromaDB
5. Click any **suggested question** (auto-generated per document) or type your own
6. Ask **follow-up questions** freely — conversational history is handled automatically
7. Toggle **Show retrieved evidence** in the sidebar to see which chunks were used

### Session Management (Sidebar)

- **View** a saved session — see all past messages with timestamps
- **Download** a session as JSON export
- **Delete** a session permanently from the database

### Sidebar Controls

| Control | Description | Default |
|---|---|---|
| `Demo Mode` | Run fully offline without API keys | Off |
| `Retriever top_k` | Number of chunks retrieved per query | 6 |
| `Chunk size` | Max characters per text chunk | 5000 |
| `Chunk overlap` | Overlapping characters between chunks | 500 |
| `Show sentence → source` | Append `(source: file.pdf p.X)` per sentence | Off |
| `Show retrieved evidence` | Display top-k retrieved chunks below answer | On |
| `Show sources` | Display source filenames used | On |

---

## System Architecture

```
User uploads PDF(s)
        ↓
Ingestion Pipeline
├── PyPDFLoader          → Extract text + inject source/page metadata
├── RecursiveCharacterTextSplitter → Split on ¶ → line → sentence → char
├── HuggingFaceEmbeddings (all-MiniLM-L6-v2) → 384-dim vectors
└── Chroma.from_documents → In-memory vector store

        ↓
Automatic Question Generation (per document)
├── LLM summarises document (first 15,000 chars)
├── LLM generates 5 questions → JSON array
└── 3-tier parsing: json.loads → bracket extraction → line-split fallback

        ↓
Query Pipeline (per user message)
├── create_history_aware_retriever
│       └── LLM reformulates follow-up → standalone query
├── ChromaDB similarity search → top-k chunks
├── create_stuff_documents_chain → chunks + history → LLaMA-3.1 via Groq
└── RunnableWithMessageHistory → auto-injects + saves session history

        ↓
SQLite Persistence (WAL mode)
├── sessions table  → session_id, created_at, updated_at
└── messages table  → session_id, seq, role, text, ts

        ↓
Answer + Source Citations displayed in Streamlit UI
        ↓
LangSmith traces chain execution (optional)
```

---

## RAG Pipeline Details

### Ingestion — Chunking Strategy

Uses `RecursiveCharacterTextSplitter` which splits text attempting natural boundaries in priority order:

| Priority | Separator | Behaviour |
|---|---|---|
| 1st | `\n\n` | Split on paragraph breaks |
| 2nd | `\n` | Split on line breaks |
| 3rd | `. ` | Split on sentence ends |
| Last | character | Hard cut only if no boundary found |

This preserves semantic coherence within each chunk — a sentence is never split mid-way unless no boundary exists.

### Retrieval — History-Aware Reformulation

Core challenge: a follow-up question like *"What did it say about that?"* sent directly to the vector store returns irrelevant chunks — the store has no memory.

**Solution:** `create_history_aware_retriever` runs a dedicated LLM step before every retrieval:

```
User: "What does Section 3 cover?"
History: [previous turns...]
         ↓ LLM reformulates ↓
Standalone query: "What topics and content are covered in Section 3 of the document?"
         ↓ sent to ChromaDB ↓
Top-k semantically relevant chunks returned
```

The system prompt includes: *"Do NOT answer the question, just reformulate it if needed and otherwise return it as is"* — preventing the LLM from answering from history alone and bypassing retrieval.

### Generation — LLM Chain

- **Model:** `llama-3.1-8b-instant` via Groq API (128k context window)
- **Chain type:** `stuff` — all retrieved chunks concatenated into one prompt
- **Source citation mode (optional):** LLM instructed to append `(source: filename.pdf p.X)` after every sentence
- **Grounding instruction:** *"If the retrieved context is insufficient or contradictory, say so. Be concise (max three sentences)."*

---

## Database Schema

```sql
CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT,
    updated_at  TEXT
);

CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    seq         INTEGER,    -- explicit per-session ordering (independent of global AUTOINCREMENT)
    role        TEXT,       -- 'user' | 'assistant'
    text        TEXT,
    ts          TEXT        -- ISO 8601 UTC timestamp
);

CREATE INDEX idx_messages_session_seq ON messages(session_id, seq);
```

**Design decisions:**
- `PRAGMA journal_mode=WAL` — concurrent reads during writes; prevents "database is locked" errors during Streamlit reruns
- `seq` column — explicit per-session message ordering; AUTOINCREMENT alone is unreliable after deletions
- Parameterised queries (`%s` / `?`) throughout — no raw string interpolation
- `conn.row_factory = sqlite3.Row` — results accessible as named dictionaries

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| RAG / LLM Framework | LangChain |
| LLM Inference | Groq API — LLaMA-3.1-8b-instant |
| Vector Store | ChromaDB (in-memory) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace Sentence Transformers) |
| Database | SQLite (WAL mode) |
| Observability | LangSmith |
| PDF Parsing | PyPDFLoader (LangChain Community) |
| Environment | python-dotenv |

---

## Known Limitations

- **ChromaDB is in-memory** — vector store is rebuilt on every PDF upload; server restart requires re-uploading documents
- **Stuff chain type** — very long documents (combined chunks exceeding ~100k tokens) may hit the LLM context limit
- **Temp file cleanup** — `NamedTemporaryFile(delete=False)` writes to disk but `os.remove()` is not explicitly called after loading
- **Demo mode heuristics** — keyword-based question generation was tuned for the test documents used during development; may produce generic questions for arbitrary document types
- **Single-instance Streamlit** — not tested under concurrent multi-user load

---

## 🔮 Future Work

- Add `persist_directory` to ChromaDB so vector store survives server restarts
- Switch to `map_reduce` chain type for documents exceeding context window limits
- Per-user document namespacing for multi-tenant deployment
- Add `@st.cache_resource` for LLM instantiation to avoid per-query recreation
- Docker containerisation for one-command deployment
- Document deduplication by content hash to avoid re-embedding the same file
- OCR support for scanned PDFs

---

