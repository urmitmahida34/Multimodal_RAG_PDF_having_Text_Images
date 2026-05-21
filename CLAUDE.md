# CLAUDE.md — Multimodal RAG PDF Assistant

This file captures every architectural decision, constraint, and reasoning from the
original design conversation so future sessions have full context without re-deriving anything.

---

## What This Project Is

A production Streamlit app that answers questions about research papers using both
text and images. Started as a local notebook (Ollama + LLaVA + FAISS in-memory),
rearchitected into a fully deployed free-tier app.

**Live app:** https://multimodal-pdf-rag-with-images.streamlit.app/
**Prototype:** `notebook_prototype.ipynb` — original Ollama/LLaVA local version

---

## Final Tech Stack (and why each was chosen)

| Layer | Choice | Reason |
|---|---|---|
| LLM | Groq — LLaMA 4 Scout 17B Vision | Fastest free multimodal option (~1-3s); LPU inference |
| Embeddings | CLIP clip-vit-base-patch32 | Joint text+image vector space — handles multimodal retrieval without separate pipelines |
| Vector DB | Qdrant Cloud free tier | Persistent storage for pre-loaded sample PDFs; 1GB free |
| Session store | FAISS in-memory | User-uploaded PDFs live only for the browser session — no need to persist |
| Frontend + hosting | Streamlit Community Cloud | Free, no sleep on public apps, zero config |
| PDF parsing | PyMuPDF | No LangChain dependency, direct control |

---

## Architecture — Two Storage Paths

```
Pre-loaded sample PDFs → ingest_samples.py (run once locally) → Qdrant Cloud (persistent)
User-uploaded PDFs    → processed at upload time              → st.session_state FAISS (ephemeral)
```

Query time: unified search across both stores, results merged and sorted by cosine similarity.

Images are stored as base64 JPEG in:
- Qdrant payload (for sample PDFs)
- `st.session_state["user_images"]` dict keyed by image_id (for user uploads)

Images are passed to Groq as vision context but are **never displayed in the UI**.
The LLM reads them and answers in text only.

---

## Key Design Decisions

### Chunking: 300 chars / 60 overlap
- CLIP text encoder has a hard **77-token limit**
- 300 chars ≈ 65 tokens — fits within limit without silent truncation
- The notebook used chunk_size=500 which was silently truncated by CLIP (bug)
- Overlap = 20% of chunk size — prevents boundary sentences being missed

### Top-k = 5, type-agnostic retrieval
- No hardcoded text/image split — CLIP's joint embedding space naturally
  floats relevant images up for visual queries, text for text queries
- Soft floor: if 0 images in top-5 but best image scores > 0.25, swap rank-5 for it
- Rationale: some answers benefit from visual context even when not explicitly asked

### Groq model names (verified against live API)
- `LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"` — vision capable
- `EVAL_JUDGE_MODEL = "llama-3.1-8b-instant"` — cheaper, text-only, for question generation

### Image compression
- Images extracted as PNG, re-saved as JPEG quality=85
- Saves ~60% storage vs PNG
- Images smaller than 80×80px skipped (decorative icons/bullets)

### Batched CLIP embedding
- `embed_texts_batch`: processes 64 texts per forward pass (not one per chunk)
- `embed_images_batch`: processes 32 images per forward pass
- Critical for upload speed on large PDFs

---

## Streamlit-Specific Decisions

### Suggestion chip buttons use `on_click` callbacks, not return value checks
```python
# Wrong — unreliable inside conditional blocks:
if st.button(q, key=...):
    st.session_state.pending_query = q

# Correct — callback runs atomically before rerun starts:
st.button(q, key=..., on_click=_ask, args=(q,))
```
The `on_click` callback is guaranteed to run before any script code in the next rerun.
Using return value checks inside `if uploaded_file is not None:` caused silent misses.

### CLIP warm-up on session start
```python
if "model_ready" not in st.session_state:
    with st.spinner("Loading embedding model — one-time setup, ~15 seconds…"):
        from services.embedder import _load
        _load()
    st.session_state.model_ready = True
```
Without this, first query freezes for 15-30s with no feedback.
The module-level singleton means `_load()` only downloads/loads once per process.

### `TOKENIZERS_PARALLELISM=false` set inside app.py
```python
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```
Cannot rely on shell environment on Streamlit Cloud — must be set programmatically
before any HuggingFace import.

---

## PDF Limits

- **10 MB file size** — safe within Streamlit Cloud's 1GB RAM
- **50 pages** — peak RAM during processing ≈ 570MB (within 1GB limit)
- RAM budget: CLIP model ~350MB + Streamlit ~80MB + processing ~140MB = ~570MB

---

## Credentials

Stored in `.streamlit/secrets.toml` locally (gitignored).
On Streamlit Cloud: added via dashboard → Advanced settings → Secrets.

Keys needed (all uppercase, TOML format):
```
GROQ_API_KEY
QDRANT_URL
QDRANT_API_KEY
```

Config reads them via `config._get()` which tries `st.secrets` first,
falls back to `os.environ` — works in both Streamlit and local script contexts.

---

## Deployment Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python packages (pip) |
| `packages.txt` | System packages — `libgomp1` for OpenMP (torch on Linux) |
| `runtime.txt` | Pins Python 3.11 — 3.14 default on Streamlit Cloud breaks ML package wheels |
| `.streamlit/config.toml` | 10MB upload limit, dark theme, disables file watcher |

---

## Sample PDFs (pre-loaded into Qdrant)

| Source key | Paper | File |
|---|---|---|
| `facenet` | FaceNet: Face Recognition & Clustering | `pdf_data/1503.03832v3.pdf` |
| `tp-gan` | TP-GAN: Frontal Face View Synthesis | `pdf_data/1704.04086v2.pdf` |
| `sketch2photo` | Sketch2Photo: Internet Image Montage | `pdf_data/SiggraphAsia_2009_sketch2photo.pdf` |

PDFs are gitignored (`*.pdf`). They only need to exist locally to run `ingest_samples.py`.
Once ingested into Qdrant, the PDFs are not needed by the app at runtime.

`suggested_questions.json` is committed to git — generated by `ingest_samples.py`,
loaded at app startup to show clickable question chips for each sample paper.

---

## Re-ingestion

Running `ingest_samples.py` twice is safe — it deletes existing points for each
source before upserting, preventing duplicate vectors that would skew retrieval scores.

```bash
source .venv/bin/activate
python ingest_samples.py
```

---

## Local Run Command (macOS)

```bash
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false .venv/bin/streamlit run app.py
```

`KMP_DUPLICATE_LIB_OK=TRUE` is macOS-only — suppresses a duplicate OpenMP conflict
between PyTorch and FAISS dylibs. Not needed on Linux/Streamlit Cloud.

---

## Known Behaviours

- Qdrant free cluster goes idle after inactivity; first query after idle has ~2s extra latency
- Streamlit Community Cloud sleeps apps after ~7 days of no traffic — visitor sees a "wake up" button
- CLIP model re-downloads on every new Streamlit Cloud deploy (~400MB, ~30s)
- User-uploaded PDFs are lost on browser tab close (session-only by design)
- Only one user-uploaded PDF is tracked at a time (`uploaded_source` in session state)

---

## Evaluation

- **Offline** (pre-loaded PDFs only): `eval/run_eval.py` using RAGAS against `eval/golden_*.json`
  Install: `pip install ragas==0.2.15` (not in main requirements, eval-only)
- **Online** (every query, any PDF): retrieval confidence score (avg cosine similarity of top-5 chunks)
  Shown live in the UI as two metrics below each answer (Retrieval confidence + Text chunks used)
