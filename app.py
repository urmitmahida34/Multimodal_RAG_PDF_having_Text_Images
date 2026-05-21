import base64
import io
import json
import os
from pathlib import Path

# Must be set before any HuggingFace import to suppress fork warnings on Linux
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import streamlit as st
from PIL import Image

import config
from services.embedder import embed_images_batch, embed_text, embed_texts_batch
from services.llm import generate_suggested_questions, stream_answer
from services.pdf_processor import process_pdf, validate_pdf
from services.vector_store import init_session_store, unified_search, upsert_to_session

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal PDF RAG",
    page_icon="📄",
    layout="wide",
)

# ── Suggestion chip callback ──────────────────────────────────────────────────
# on_click runs atomically BEFORE the rerun starts, guaranteeing pending_query
# is set by the time the main area reads it — unlike checking button return values
# inside conditional blocks, which can silently miss the click.
def _ask(q: str) -> None:
    st.session_state.pending_query = q

# ── Session state init ────────────────────────────────────────────────────────
init_session_store(st.session_state)

if "messages"           not in st.session_state: st.session_state.messages           = []
if "uploaded_source"    not in st.session_state: st.session_state.uploaded_source    = None
if "pending_query"      not in st.session_state: st.session_state.pending_query      = None

# Warm up CLIP once per session — prevents a silent 15-30s freeze on first query
if "model_ready" not in st.session_state:
    with st.spinner("Loading embedding model — one-time setup, ~15 seconds…"):
        from services.embedder import _load
        _load()
    st.session_state.model_ready = True
if "suggested_questions" not in st.session_state:
    sq_path = Path("suggested_questions.json")
    st.session_state.suggested_questions = (
        json.loads(sq_path.read_text()) if sq_path.exists() else {}
    )

# ── Friendly display names ────────────────────────────────────────────────────
FRIENDLY = {
    "facenet":      "FaceNet: Face Recognition & Clustering",
    "tp-gan":       "TP-GAN: Frontal Face Synthesis",
    "sketch2photo": "Sketch2Photo: Image Montage",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Sources")

    # ── Sample PDFs ───────────────────────────────────────────────────────────
    st.subheader("Sample Papers")
    st.caption("Pre-loaded & ready to query")

    sq: dict = st.session_state.suggested_questions

    for source, friendly in FRIENDLY.items():
        with st.expander(friendly, expanded=False):
            questions = sq.get(source, [])
            if questions:
                st.markdown("**Suggested questions — click to ask:**")
                for q in questions:
                    st.button(
                        q,
                        key=f"sq_{source}_{q[:40]}",
                        on_click=_ask,
                        args=(q,),
                        use_container_width=True,
                    )
            else:
                st.caption("Run ingest_samples.py to populate questions.")

    st.divider()

    # ── User PDF upload ───────────────────────────────────────────────────────
    st.subheader("Upload Your PDF")
    st.caption(f"Max {config.MAX_PDF_SIZE_MB} MB · {config.MAX_PDF_PAGES} pages")

    uploaded_file = st.file_uploader(
        label="Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_bytes  = uploaded_file.read()
        source_name = Path(uploaded_file.name).stem.lower().replace(" ", "_")

        # Process only when a new file is uploaded
        if st.session_state.uploaded_source != source_name:
            valid, err = validate_pdf(file_bytes, uploaded_file.name)
            if not valid:
                st.error(err)
            else:
                with st.spinner("Processing your PDF…"):
                    text_chunks, image_chunks = process_pdf(file_bytes, source_name)

                    text_vecs = embed_texts_batch(
                        [c.content for c in text_chunks]
                    )
                    pil_images = [
                        Image.open(io.BytesIO(base64.b64decode(ic.base64_data))).convert("RGB")
                        for ic in image_chunks
                    ]
                    image_vecs = (
                        embed_images_batch(pil_images)
                        if pil_images
                        else np.empty((0, config.EMBEDDING_DIM), dtype="float32")
                    )

                    upsert_to_session(
                        st.session_state,
                        text_chunks, text_vecs,
                        image_chunks, image_vecs,
                    )

                    first_text = " ".join(
                        c.content for c in text_chunks if c.page == 0
                    )[:1200]
                    user_qs = generate_suggested_questions(source_name, first_text)
                    sq[source_name] = user_qs
                    st.session_state.suggested_questions = sq
                    st.session_state.uploaded_source     = source_name

                st.success(
                    f"✓ {uploaded_file.name} — "
                    f"{len(text_chunks)} text chunks, {len(image_chunks)} images indexed."
                )

        # Show suggested questions on EVERY rerun — not just after first upload
        upload_questions = sq.get(source_name, [])
        if upload_questions:
            st.markdown("**Suggested questions:**")
            for q in upload_questions:
                st.button(
                    q,
                    key=f"uq_{source_name}_{q[:40]}",
                    on_click=_ask,
                    args=(q,),
                    use_container_width=True,
                )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Multimodal PDF RAG")
st.caption(
    "Ask questions about FaceNet, TP-GAN, or Sketch2Photo — "
    "or upload your own research paper."
)
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metrics" in msg:
            m    = msg["metrics"]
            cols = st.columns(2)
            cols[0].metric("Retrieval confidence", f"{m['avg_score']:.2f}")
            cols[1].metric("Text chunks used",     m["n_text"])

# ── Pick up pending query (set by suggestion chip callback) ───────────────────
query: str | None = st.session_state.pending_query
st.session_state.pending_query = None

# ── Chat input ────────────────────────────────────────────────────────────────
typed = st.chat_input("Ask a question about the papers…")
query = query or typed

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve
    query_vec = embed_text(query)
    try:
        results = unified_search(query_vec, st.session_state)
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        st.stop()

    text_chunks = results["text_chunks"]
    image_b64   = results["image_b64"]
    avg_score   = results["avg_score"]

    if avg_score < config.RETRIEVAL_CONFIDENCE_OK:
        st.warning(
            f"Low retrieval confidence ({avg_score:.2f}). "
            "The answer may not be well-grounded in the documents."
        )

    # Stream answer
    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(
                stream_answer(query, text_chunks, image_b64)
            )
        except Exception as e:
            response_text = f"Generation failed: {e}"
            st.error(response_text)

        metrics = {
            "avg_score": avg_score,
            "n_text":    len(text_chunks),
        }
        cols = st.columns(2)
        cols[0].metric("Retrieval confidence", f"{avg_score:.2f}")
        cols[1].metric("Text chunks used",     len(text_chunks))

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response_text,
        "metrics": metrics,
    })
