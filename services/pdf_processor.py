import io
import base64
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image

import config


@dataclass
class TextChunk:
    content:     str
    page:        int
    chunk_index: int
    source:      str   # human-readable PDF name (e.g. "facenet")


@dataclass
class ImageChunk:
    image_id:   str   # unique: "{source}_p{page}_i{idx}"
    page:       int
    base64_data: str  # JPEG-compressed base64
    source:     str


# ── Validation ────────────────────────────────────────────────────────────────

def validate_pdf(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """Return (is_valid, error_message). Empty error means valid."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > config.MAX_PDF_SIZE_MB:
        return False, f"File is {size_mb:.1f} MB — maximum allowed is {config.MAX_PDF_SIZE_MB} MB."

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        n_pages = len(doc)
        doc.close()
    except Exception:
        return False, "Could not open file. Make sure it is a valid PDF."

    if n_pages > config.MAX_PDF_PAGES:
        return False, f"PDF has {n_pages} pages — maximum allowed is {config.MAX_PDF_PAGES} pages."

    return True, ""


# ── Text chunking (no LangChain) ──────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursive character splitter that tries paragraph → line → sentence →
    word boundaries before hard-splitting. Keeps chunks within chunk_size.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in ["\n\n", "\n", ". ", " "]:
        if sep not in text:
            continue

        chunks: list[str] = []
        current = ""

        for part in text.split(sep):
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # carry overlap into next chunk
                overlap = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                current = (overlap + sep if overlap else "") + part

        if current.strip():
            chunks.append(current.strip())

        if chunks:
            return chunks

    # hard split fallback
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks


# ── Main processor ────────────────────────────────────────────────────────────

def process_pdf(
    file_bytes: bytes,
    source_name: str,
) -> tuple[list[TextChunk], list[ImageChunk]]:
    """
    Extract text chunks and images from PDF bytes.
    Returns (text_chunks, image_chunks).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    text_chunks:  list[TextChunk]  = []
    image_chunks: list[ImageChunk] = []
    chunk_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # ── Text ──────────────────────────────────────────────────────────────
        raw_text = page.get_text().strip()
        if raw_text:
            for chunk_text in _split_text(raw_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                if len(chunk_text.strip()) < 20:   # skip noise / single-word lines
                    continue
                text_chunks.append(TextChunk(
                    content=chunk_text,
                    page=page_num,
                    chunk_index=chunk_counter,
                    source=source_name,
                ))
                chunk_counter += 1

        # ── Images ────────────────────────────────────────────────────────────
        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # skip tiny decorative images
                if pil_img.width < 80 or pil_img.height < 80:
                    continue

                # JPEG-compress to save ~60 % vs PNG
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()

                image_chunks.append(ImageChunk(
                    image_id=f"{source_name}_p{page_num}_i{img_idx}",
                    page=page_num,
                    base64_data=b64,
                    source=source_name,
                ))
            except Exception:
                continue   # skip corrupt / unsupported image formats

    doc.close()
    return text_chunks, image_chunks
