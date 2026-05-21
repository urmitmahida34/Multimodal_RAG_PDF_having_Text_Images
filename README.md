# Multimodal RAG — PDF Text & Image Assistant

A production-grade Retrieval-Augmented Generation system that answers questions about research papers using both **text and visual content**. Upload any PDF and ask questions — the system retrieves relevant text chunks and figures, then streams a grounded answer.

**Live demo:** https://multimodal-pdf-rag-with-images.streamlit.app/

---

## How It Works

```
PDF uploaded
     │
     ▼
PyMuPDF extracts text chunks + images
     │
     ▼
CLIP (clip-vit-base-patch32) embeds both into
the same 512-dim vector space
     │
     ├── Text vectors + Image vectors
     │         │
     │         ▼
     │   Qdrant Cloud          ←── pre-loaded sample PDFs (persistent)
     │   FAISS session store   ←── user-uploaded PDFs (in-memory)
     │
User asks a question
     │
     ▼
CLIP embeds the query → unified similarity search across both stores
     │
     ▼
Top-5 chunks retrieved (text + images ranked by cosine similarity)
     │
     ▼
Groq — LLaMA 4 Scout 17B Vision
reads the text context + figures → streams a text answer
```

---

## Features

- **Multimodal retrieval** — CLIP's joint text-image embedding space means visual queries naturally surface relevant figures without any special handling
- **Streaming answers** — tokens stream live via Groq's LPU inference (~1-3s first token)
- **Pre-loaded sample papers** — three research papers ready to query with suggested questions; no upload needed
- **Upload your own PDF** — 10 MB / 50 page limit; auto-generates suggested questions after processing
- **Per-query quality metrics** — retrieval confidence score, number of text chunks and images used shown below every answer
- **Graceful degradation** — if Qdrant is unreachable, session FAISS results still return

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) — unified text + image vectors |
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Vector DB (persistent) | [Qdrant Cloud](https://cloud.qdrant.io/) — free tier |
| Vector DB (session) | [FAISS](https://github.com/facebookresearch/faiss) — in-memory |
| LLM | [Groq](https://groq.com/) — LLaMA 4 Scout 17B Vision, free tier |
| Frontend | [Streamlit](https://streamlit.io/) |
| Hosting | [Streamlit Community Cloud](https://share.streamlit.io/) — free |

---

## Project Structure

```
├── app.py                        # Streamlit UI
├── config.py                     # All constants and credentials
├── ingest_samples.py             # One-time script to index sample PDFs into Qdrant
├── suggested_questions.json      # Auto-generated questions for sample papers
│
├── services/
│   ├── pdf_processor.py          # PDF validation, text chunking, image extraction
│   ├── embedder.py               # CLIP singleton — embed_text, embed_image, batched
│   ├── vector_store.py           # Qdrant + FAISS session store, unified search
│   └── llm.py                    # Groq streaming, multimodal prompt builder
│
├── eval/                         # Offline evaluation scripts (see Evaluation section)
│
├── notebook_prototype.ipynb      # Original R&D prototype (Ollama + LLaVA, local)
│
├── requirements.txt
├── packages.txt                  # System packages for Streamlit Cloud (libgomp1)
└── runtime.txt                   # Python version pin (3.11)
```

---

## Chunking & Retrieval Design

**Chunk size = 300 characters** — matched to CLIP's 77-token hard limit. At ~4-5 chars/token, 300 chars ≈ 65 tokens, ensuring no silent truncation during embedding.

**Chunk overlap = 60 characters (20%)** — prevents key sentences that fall at chunk boundaries from being missed during retrieval.

**Top-k = 5, type-agnostic** — CLIP's joint embedding space naturally floats relevant images to the top for visual queries and text for text queries. A soft floor ensures at least one image is included when a relevant figure scores above 0.25 cosine similarity.

---

## Sample Papers (Pre-loaded)

| Paper | Topics |
|---|---|
| **FaceNet** (Schroff et al., 2015) | Face recognition, triplet loss, embedding space |
| **TP-GAN** (Huang et al., 2017) | Face synthesis, GANs, identity preservation |
| **Sketch2Photo** (Chen et al., 2009) | Image montage, sketch-to-photo, internet retrieval |

---

## Local Setup

```bash
# 1. Clone and enter project
git clone <your-repo-url>
cd multimodal-rag-pdf

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add credentials — create .env
GROQ_API_KEY=your_groq_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

# 5. Ingest sample PDFs into Qdrant (run once)
python ingest_samples.py

# 6. Run the app
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/streamlit run app.py
```

> `KMP_DUPLICATE_LIB_OK=TRUE` is macOS-only — suppresses a duplicate OpenMP conflict between PyTorch and FAISS. Not needed on Linux.

---

## Evaluation

Offline evaluation runs against a golden test set on the pre-loaded sample papers:

```bash
pip install -r requirements.txt  # ragas included
python eval/run_eval.py
```

Online (per-query) evaluation is shown live in the app — retrieval confidence score is the average cosine similarity of the top-5 retrieved chunks. A score below 0.50 triggers a low-confidence warning.

---

## Prototype

`notebook_prototype.ipynb` contains the original local implementation using **Ollama (LLaVA)** and **FAISS in-memory**. The production app replaces these with Groq (cloud inference, no GPU needed) and Qdrant (persistent vector storage), while keeping the same CLIP embedding approach.
