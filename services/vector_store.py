"""
Two-store retrieval layer:

  Qdrant   — persistent cloud store for the 3 pre-loaded sample PDFs.
  FAISS    — in-memory session store for PDFs uploaded by the user.
             Stored inside a plain dict (pass st.session_state from app.py).

Unified search merges both stores and returns the top-k results by cosine
similarity, with a soft floor that ensures at least one image is included
when a relevant figure exists.
"""
import uuid
from typing import Any

import faiss
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

import config
from services.pdf_processor import ImageChunk, TextChunk

# ── Qdrant client (module-level singleton) ────────────────────────────────────

_qdrant: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=config.get_qdrant_url(),
            api_key=config.get_qdrant_api_key(),
            check_compatibility=False,
        )
    return _qdrant


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def ensure_collection() -> None:
    """Create the sample_docs collection if it does not exist yet."""
    client = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


def upsert_to_qdrant(
    text_chunks:  list[TextChunk],
    text_vectors: np.ndarray,
    image_chunks: list[ImageChunk],
    image_vectors: np.ndarray,
) -> None:
    """Push text + image points to the Qdrant sample collection."""
    client = _get_qdrant()
    points: list[PointStruct] = []

    for chunk, vec in zip(text_chunks, text_vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={
                "type":    "text",
                "source":  chunk.source,
                "page":    chunk.page,
                "content": chunk.content,
            },
        ))

    for chunk, vec in zip(image_chunks, image_vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={
                "type":       "image",
                "source":     chunk.source,
                "page":       chunk.page,
                "image_id":   chunk.image_id,
                "base64_data": chunk.base64_data,
            },
        ))

    # batch upsert in chunks of 100 to stay within Qdrant limits
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=points[i : i + batch_size],
        )


def search_qdrant(query_vector: np.ndarray, k: int) -> list[dict]:
    """Search the sample_docs collection. Returns list of payload dicts with 'score'."""
    client = _get_qdrant()
    hits = client.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_vector.tolist(),
        limit=k,
        with_payload=True,
    )
    results = []
    for h in hits:
        item = dict(h.payload)
        item["score"] = h.score
        results.append(item)
    return results


# ── FAISS session store ───────────────────────────────────────────────────────

SESSION_DOCS_KEY   = "user_docs"
SESSION_EMBEDS_KEY = "user_embeddings"
SESSION_IMAGES_KEY = "user_images"   # image_id → base64_data


def init_session_store(session: dict) -> None:
    """Initialise session keys if missing. Call once at app startup."""
    session.setdefault(SESSION_DOCS_KEY,   [])
    session.setdefault(SESSION_EMBEDS_KEY, None)   # np.ndarray or None
    session.setdefault(SESSION_IMAGES_KEY, {})


def upsert_to_session(
    session:       dict,
    text_chunks:   list[TextChunk],
    text_vectors:  np.ndarray,
    image_chunks:  list[ImageChunk],
    image_vectors: np.ndarray,
) -> None:
    """
    Add a newly processed PDF to the session store.
    Vectors are stacked into a single numpy array; images are stored by ID.
    """
    new_docs: list[dict] = []
    new_vecs: list[np.ndarray] = []

    for chunk, vec in zip(text_chunks, text_vectors):
        new_docs.append({
            "type":    "text",
            "source":  chunk.source,
            "page":    chunk.page,
            "content": chunk.content,
        })
        new_vecs.append(vec)

    for chunk, vec in zip(image_chunks, image_vectors):
        new_docs.append({
            "type":     "image",
            "source":   chunk.source,
            "page":     chunk.page,
            "image_id": chunk.image_id,
        })
        new_vecs.append(vec)
        session[SESSION_IMAGES_KEY][chunk.image_id] = chunk.base64_data

    if not new_vecs:
        return

    stacked = np.array(new_vecs, dtype=np.float32)

    if session[SESSION_EMBEDS_KEY] is None:
        session[SESSION_EMBEDS_KEY] = stacked
        session[SESSION_DOCS_KEY]   = new_docs
    else:
        session[SESSION_EMBEDS_KEY] = np.vstack(
            [session[SESSION_EMBEDS_KEY], stacked]
        )
        session[SESSION_DOCS_KEY].extend(new_docs)


def search_session(
    session:      dict,
    query_vector: np.ndarray,
    k:            int,
) -> list[dict]:
    """
    FAISS cosine search over the session FAISS store.
    Returns list of payload dicts with 'score'.
    """
    embeds: np.ndarray | None = session.get(SESSION_EMBEDS_KEY)
    docs:   list[dict]        = session.get(SESSION_DOCS_KEY, [])

    if embeds is None or len(docs) == 0:
        return []

    # FAISS IndexFlatIP works on unit vectors → equivalent to cosine similarity
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(embeds)

    q = query_vector.reshape(1, -1).astype(np.float32)
    k_actual = min(k, len(docs))
    scores, indices = index.search(q, k_actual)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = dict(docs[idx])
        item["score"] = float(score)
        results.append(item)
    return results


# ── Unified search ────────────────────────────────────────────────────────────

def unified_search(
    query_vector: np.ndarray,
    session:      dict,
    k:            int = config.TOP_K_TOTAL,
) -> dict[str, Any]:
    """
    Search both Qdrant (sample PDFs) and the session FAISS store (user upload).
    Merges and sorts by score, then applies a soft-floor so at least one image
    is present when a relevant figure exists.

    Returns:
        {
          "text_chunks": [{"content": ..., "page": ..., "source": ..., "score": ...}],
          "image_b64":   ["base64str", ...],
          "avg_score":   float,
          "sources":     set of source names,
        }
    """
    # Fetch 2k so that after slicing top-k, the remaining k are available
    # as soft-floor candidates for image inclusion. With exactly k results,
    # all_results[k:] is always empty and the soft floor never fires.
    search_k = k * 2

    try:
        qdrant_results = search_qdrant(query_vector, k=search_k)
    except Exception:
        qdrant_results = []   # degrade gracefully if Qdrant is unreachable

    session_results = search_session(session, query_vector, k=search_k)

    all_results = qdrant_results + session_results
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top = all_results[:k]

    text_chunks: list[dict] = []
    image_b64:   list[str]  = []
    sources:     set[str]   = set()

    for item in top:
        sources.add(item.get("source", ""))
        if item["type"] == "text":
            text_chunks.append({
                "content": item["content"],
                "page":    item["page"],
                "source":  item.get("source", ""),
                "score":   item["score"],
            })
        else:
            # images: base64 comes from Qdrant payload OR session image store
            b64 = item.get("base64_data") or session.get(SESSION_IMAGES_KEY, {}).get(
                item.get("image_id", ""), None
            )
            if b64:
                image_b64.append(b64)

    # ── soft floor: add best image if none retrieved but one is close enough ──
    if not image_b64:
        candidates = [r for r in all_results[k:] if r["type"] == "image"]
        if candidates:
            best_img = candidates[0]
            if best_img["score"] >= config.IMAGE_INCLUDE_THRESHOLD:
                b64 = best_img.get("base64_data") or session.get(
                    SESSION_IMAGES_KEY, {}
                ).get(best_img.get("image_id", ""), None)
                if b64:
                    image_b64.append(b64)

    avg_score = float(np.mean([r["score"] for r in top])) if top else 0.0

    return {
        "text_chunks":  text_chunks,
        "image_b64":    image_b64,
        "avg_score":    avg_score,
        "sources":      sources,
        "debug_all":    all_results,   # full ranked list with scores — for debug panel
    }
