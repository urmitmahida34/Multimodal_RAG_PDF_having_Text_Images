"""
One-time ingestion script — run this ONCE locally before deploying.

    source .venv/bin/activate
    python ingest_samples.py

Reads the 3 sample PDFs from pdf_data/, embeds them with CLIP, and
pushes all vectors + payloads to the Qdrant sample_docs collection.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env so credentials are available outside Streamlit

from qdrant_client.models import FieldCondition, Filter, MatchValue

from services.pdf_processor import process_pdf
from services.embedder import embed_texts_batch, embed_images_batch
from services.vector_store import ensure_collection, upsert_to_qdrant, _get_qdrant
from services.llm import generate_suggested_questions
import config
import io
from PIL import Image
import base64

# ── PDF registry ──────────────────────────────────────────────────────────────
# (source_name shown in UI, file path)
SAMPLE_PDFS = [
    ("facenet",      "pdf_data/1503.03832v3.pdf"),
    ("tp-gan",       "pdf_data/1704.04086v2.pdf"),
    ("sketch2photo", "pdf_data/SiggraphAsia_2009_sketch2photo.pdf"),
]

FRIENDLY_NAMES = {
    "facenet":      "FaceNet: Face Recognition & Clustering",
    "tp-gan":       "TP-GAN: Frontal Face View Synthesis",
    "sketch2photo": "Sketch2Photo: Internet Image Montage",
}


def ingest_pdf(source_name: str, pdf_path: str) -> list[str]:
    """Process, embed, and upsert one PDF. Returns generated suggested questions."""
    print(f"\n{'─'*60}")
    print(f"  Ingesting: {FRIENDLY_NAMES[source_name]}")
    print(f"{'─'*60}")

    file_bytes = Path(pdf_path).read_bytes()

    # 0. Delete existing points for this source (prevents duplicates on re-run)
    print("  [0/4] Removing any existing points for this source …")
    _get_qdrant().delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_name))]
        ),
    )

    # 1. Extract text chunks + images
    print("  [1/4] Extracting text and images …")
    text_chunks, image_chunks = process_pdf(file_bytes, source_name)
    print(f"        {len(text_chunks)} text chunks, {len(image_chunks)} images")

    # 2. Embed text
    print("  [2/4] Embedding text chunks …")
    text_vecs = embed_texts_batch([c.content for c in text_chunks])

    # 3. Embed images
    print("  [3/4] Embedding images …")
    pil_images = []
    for ic in image_chunks:
        img_bytes = base64.b64decode(ic.base64_data)
        pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    image_vecs = embed_images_batch(pil_images) if pil_images else []

    # 4. Push to Qdrant
    print("  [4/4] Upserting to Qdrant …")
    upsert_to_qdrant(text_chunks, text_vecs, image_chunks, image_vecs)

    # 5. Generate suggested questions from first-page text
    first_page_text = " ".join(
        c.content for c in text_chunks if c.page == 0
    )[:1200]
    questions = generate_suggested_questions(
        FRIENDLY_NAMES[source_name], first_page_text
    )
    print(f"  Suggested questions:")
    for q in questions:
        print(f"    • {q}")

    return questions


def main() -> None:
    print("\n====  Multimodal RAG — Sample PDF Ingestion  ====\n")

    print("Creating Qdrant collection if needed …")
    ensure_collection()

    all_questions: dict[str, list[str]] = {}

    for source_name, pdf_path in SAMPLE_PDFS:
        if not Path(pdf_path).exists():
            print(f"  WARNING: {pdf_path} not found — skipping.")
            continue
        questions = ingest_pdf(source_name, pdf_path)
        all_questions[source_name] = questions

    # Write suggested questions to a JSON file so app.py can load them
    import json
    out_path = "suggested_questions.json"
    with open(out_path, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"\n✓ Suggested questions saved to {out_path}")
    print("\n====  Ingestion complete  ====\n")


if __name__ == "__main__":
    main()
