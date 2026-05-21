import os
import streamlit as st


def _get(key: str) -> str:
    """Read from Streamlit secrets (deployed) or env vars (local)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")


# ── PDF limits ────────────────────────────────────────────────────────────────
MAX_PDF_SIZE_MB = 10
MAX_PDF_PAGES   = 50

# ── Chunking (matched to CLIP's 77-token limit) ───────────────────────────────
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 60

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_TOTAL             = 5
IMAGE_INCLUDE_THRESHOLD = 0.15   # soft floor: include best image if sim > this

# ── Embedding ─────────────────────────────────────────────────────────────────
CLIP_MODEL    = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512

# ── Qdrant (pre-loaded sample PDFs) ──────────────────────────────────────────
QDRANT_COLLECTION = "sample_docs"

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER = "groq"
LLM_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Suggested questions per PDF ───────────────────────────────────────────────
SUGGESTED_Q_COUNT = 3

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_JUDGE_MODEL        = "llama-3.1-8b-instant"
RETRIEVAL_CONFIDENCE_OK = 0.50   # below this → warn user answer may be weak


# ── Credentials (resolved at runtime) ─────────────────────────────────────────
def get_groq_api_key()    -> str: return _get("GROQ_API_KEY")
def get_qdrant_url()      -> str: return _get("QDRANT_URL")
def get_qdrant_api_key()  -> str: return _get("QDRANT_API_KEY")
