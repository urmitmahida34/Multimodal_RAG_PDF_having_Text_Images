"""
Groq LLM service.

  stream_answer()              — streams tokens for a RAG query
  generate_suggested_questions() — generates 3 question suggestions for a PDF
"""
from collections.abc import Generator

from groq import Groq

import config

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.get_groq_api_key())
    return _client


# ── Answer streaming ──────────────────────────────────────────────────────────

def stream_answer(
    query:       str,
    text_chunks: list[dict],
    image_b64:   list[str],
) -> Generator[str, None, None]:
    """
    Build a multimodal prompt from retrieved context and stream tokens back.
    Images are passed to the vision model as base64 JPEG data URLs.
    The model is instructed to answer in text only.
    """
    content: list[dict] = []

    # System instruction
    content.append({
        "type": "text",
        "text": (
            "You are a precise research assistant. Answer the user's question "
            "using only the provided text context and visual figures. "
            "Respond in clear text — do not describe that you are looking at images, "
            "just use them to inform your answer. "
            "Give a thorough answer of 4-6 sentences, covering the key details and context."
        ),
    })

    # Text context
    if text_chunks:
        ctx = "\n\n".join(
            f"[Page {c['page'] + 1}, {c['source']}]: {c['content']}"
            for c in text_chunks
        )
        content.append({"type": "text", "text": f"Context:\n{ctx}"})

    # Images (vision context, not displayed in UI)
    for b64 in image_b64:
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    # User question
    content.append({"type": "text", "text": f"Question: {query}"})

    stream = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": content}],
        stream=True,
        temperature=0.1,
        max_tokens=1024,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Suggested question generation ─────────────────────────────────────────────

def generate_suggested_questions(source_name: str, sample_text: str) -> list[str]:
    """
    Ask a cheap Groq model to generate SUGGESTED_Q_COUNT questions
    a user might ask about this PDF.
    """
    prompt = (
        f"A research paper titled '{source_name}' contains the following excerpt:\n\n"
        f"{sample_text[:1200]}\n\n"
        f"Generate exactly {config.SUGGESTED_Q_COUNT} concise, specific questions "
        "that someone reading this paper might want answered. "
        "Return only the questions, one per line, no numbering or bullets."
    )

    response = _get_client().chat.completions.create(
        model=config.EVAL_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip().split("\n")
    return [q.strip() for q in raw if q.strip()][: config.SUGGESTED_Q_COUNT]
