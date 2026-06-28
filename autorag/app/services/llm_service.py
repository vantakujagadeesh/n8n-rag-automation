# app/services/llm_service.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/services/llm_service.py
"""
AutoRAG — LLM Service
======================
Provides two public async functions for the query pipeline:

  1. generate_hypothetical_doc(question) → str
     HyDE (Hypothetical Document Embeddings) expansion.
     Asks the LLM to write a short passage that would answer the question,
     then embeds THAT passage instead of the raw query for better recall.

  2. generate_answer(question, context_chunks, model) → dict
     Builds a grounded, citation-aware answer from reranked chunks.
     Returns: {answer, model_used, token_count, citations}

Internal helpers:
  - _call_openai(messages, model)   → (text, tokens)
  - _call_anthropic(messages, model) → (text, tokens)
  - extract_citations(answer, chunks) → list[str]

Fallback strategy:
  PRIMARY_LLM (gpt-4o) → on ANY exception → FALLBACK_LLM (claude-sonnet-4-6)

All functions are async and production-safe.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import anthropic
import openai

from app.core.config import get_settings

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── System Prompts ────────────────────────────────────────────────────────────
_HYDE_SYSTEM_PROMPT: str = (
    "Write a 2-3 sentence passage from a document that directly answers this question. "
    "Write only the passage, no preamble."
)

_QA_SYSTEM_PROMPT: str = (
    "You are a helpful assistant answering questions from a document knowledge base. "
    "Answer ONLY from the provided context. "
    "If the context does not contain enough information to answer, say exactly: "
    "'I could not find relevant information in the knowledge base.' "
    "Always cite which document your answer comes from by mentioning the filename."
)

# Maximum characters of context to inject into the LLM prompt.
# Prevents exceeding context windows on very large reranked sets.
_MAX_CONTEXT_CHARS: int = 12_000


# ── OpenAI Client Factory ────────────────────────────────────────────────────
def _get_openai_client() -> openai.AsyncOpenAI:
    """Returns a fresh AsyncOpenAI client using the configured API key."""
    settings = get_settings()
    return openai.AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=60.0,
        max_retries=2,
    )


# ── Anthropic Client Factory ─────────────────────────────────────────────────
def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    """Returns a fresh AsyncAnthropic client using the configured API key."""
    settings = get_settings()
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        timeout=60.0,
        max_retries=2,
    )


# ── Internal LLM Callers ──────────────────────────────────────────────────────

async def _call_openai(
    messages: list[dict[str, str]],
    model: str,
) -> tuple[str, int]:
    """
    Calls OpenAI Chat Completions API.

    Args:
        messages: OpenAI-format message list [{"role": ..., "content": ...}].
        model:    Model name e.g. "gpt-4o".

    Returns:
        (response_text, total_tokens) tuple.

    Raises:
        openai.OpenAIError: On API errors (rate limit, auth, network, etc.)
    """
    client = _get_openai_client()
    logger.debug("Calling OpenAI model='%s' with %d messages.", model, len(messages))

    response = await client.chat.completions.create(
        model=model,
        messages=messages,          # type: ignore[arg-type]
        temperature=0.2,            # low temperature for factual grounding
        max_tokens=1024,
    )

    text: str = response.choices[0].message.content or ""
    tokens: int = response.usage.total_tokens if response.usage else 0

    logger.debug(
        "OpenAI response: model='%s' tokens=%d chars=%d.",
        response.model,
        tokens,
        len(text),
    )
    return text.strip(), tokens


async def _call_anthropic(
    messages: list[dict[str, str]],
    model: str,
) -> tuple[str, int]:
    """
    Calls Anthropic Messages API.

    Converts OpenAI-format messages to Anthropic format:
      - "system" role is extracted as top-level system param.
      - Remaining messages become the messages list.

    Args:
        messages: OpenAI-format message list [{"role": ..., "content": ...}].
        model:    Model name e.g. "claude-sonnet-4-6".

    Returns:
        (response_text, total_tokens) tuple.

    Raises:
        anthropic.AnthropicError: On API errors.
    """
    client = _get_anthropic_client()

    # Separate system prompt from conversation messages
    system_prompt: str = ""
    anthropic_messages: list[dict[str, str]] = []

    for msg in messages:
        if msg["role"] == "system":
            # Anthropic takes system as a separate top-level param
            system_prompt = msg["content"]
        else:
            anthropic_messages.append(
                {"role": msg["role"], "content": msg["content"]}
            )

    logger.debug(
        "Calling Anthropic model='%s' with %d messages (system=%d chars).",
        model,
        len(anthropic_messages),
        len(system_prompt),
    )

    response = await client.messages.create(
        model=model,
        system=system_prompt if system_prompt else anthropic.NOT_GIVEN,  # type: ignore[arg-type]
        messages=anthropic_messages,          # type: ignore[arg-type]
        max_tokens=1024,
        temperature=0.2,
    )

    text: str = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    tokens: int = (
        response.usage.input_tokens + response.usage.output_tokens
        if response.usage
        else 0
    )

    logger.debug(
        "Anthropic response: model='%s' tokens=%d chars=%d.",
        response.model,
        tokens,
        len(text),
    )
    return text.strip(), tokens


# ── Citation Extractor ────────────────────────────────────────────────────────

def extract_citations(answer: str, chunks: list[dict[str, Any]]) -> list[str]:
    """
    Scans the answer text for filenames mentioned from the retrieved chunks.

    Strategy:
      For each unique filename across all chunks, check if the base name
      (without extension) or the full filename appears in the answer (case-
      insensitive). Deduplicates and preserves first-mention order.

    Args:
        answer: LLM-generated answer string.
        chunks: Reranked chunk dicts — each must have metadata.filename
                OR a top-level "filename" key (handles both shapes).

    Returns:
        Ordered, deduplicated list of filenames cited in the answer.
        Returns [] if no filenames are detected.
    """
    if not answer or not chunks:
        return []

    # Collect unique filenames from chunks (handle both dict shapes)
    seen_filenames: list[str] = []
    for chunk in chunks:
        filename: str | None = None
        if isinstance(chunk.get("metadata"), dict):
            filename = chunk["metadata"].get("filename")
        if not filename:
            filename = chunk.get("filename")
        if filename and filename not in seen_filenames:
            seen_filenames.append(filename)

    answer_lower = answer.lower()
    cited: list[str] = []

    for filename in seen_filenames:
        # Check full filename match (e.g. "policy.pdf")
        if filename.lower() in answer_lower:
            cited.append(filename)
            continue

        # Check base name without extension (e.g. "policy")
        base = re.sub(r"\.[^.]+$", "", filename)   # strip last extension
        if base and base.lower() in answer_lower:
            cited.append(filename)

    logger.debug(
        "extract_citations: scanned %d filenames, found %d cited.",
        len(seen_filenames),
        len(cited),
    )
    return cited


# ── Context Builder ───────────────────────────────────────────────────────────

def _build_context_string(chunks: list[dict[str, Any]]) -> str:
    """
    Formats reranked chunks into a numbered context block for the LLM prompt.

    Each chunk is rendered as:
        [1] Source: <filename> (chunk <index>)
        <text>

    Args:
        chunks: Reranked chunk dicts (each has "text" and metadata).

    Returns:
        Formatted context string, truncated to _MAX_CONTEXT_CHARS if needed.
    """
    parts: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        # Resolve filename from either shape
        filename: str = "unknown"
        chunk_index: int = 0

        if isinstance(chunk.get("metadata"), dict):
            filename = chunk["metadata"].get("filename", "unknown")
            chunk_index = chunk["metadata"].get("chunk_index", 0)
        filename = chunk.get("filename", filename)
        chunk_index = chunk.get("chunk_index", chunk_index)

        text = chunk.get("text", "").strip()
        parts.append(
            f"[{i}] Source: {filename} (chunk {chunk_index})\n{text}"
        )

    context = "\n\n---\n\n".join(parts)

    # Truncate to avoid exceeding context window
    if len(context) > _MAX_CONTEXT_CHARS:
        logger.warning(
            "Context string truncated from %d to %d chars.",
            len(context),
            _MAX_CONTEXT_CHARS,
        )
        context = context[:_MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"

    return context


# ── Public API ─────────────────────────────────────────────────────────────────

async def generate_hypothetical_doc(question: str) -> str:
    """
    HyDE (Hypothetical Document Embeddings) expansion.

    Asks the PRIMARY_LLM to write a short document passage that would answer
    the question. Embedding this synthetic passage and using it as the query
    vector typically improves retrieval recall by ~10-20% on factual queries.

    On any exception, falls back to returning the original question so the
    pipeline continues uninterrupted.

    Args:
        question: The user's original question.

    Returns:
        A 2-3 sentence hypothetical document excerpt, or the original
        question if the LLM call fails.
    """
    settings = get_settings()
    model = settings.primary_llm

    messages = [
        {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    logger.info(
        "HyDE expansion: model='%s' question='%.80s…'", model, question
    )

    try:
        text, tokens = await _call_openai(messages, model)
        logger.info(
            "HyDE expanded successfully: %d tokens, %d chars.", tokens, len(text)
        )
        return text if text else question

    except Exception as exc:
        # Non-fatal — log and fall through to original question
        logger.warning(
            "HyDE expansion failed (model='%s'): %s. "
            "Falling back to original question.",
            model,
            exc,
        )
        return question


async def generate_answer(
    question: str,
    context_chunks: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, Any]:
    """
    Generates a grounded, citation-aware answer using the configured LLM.

    Attempts PRIMARY_LLM first; on any exception automatically retries with
    FALLBACK_LLM. If both fail, raises the fallback exception.

    Args:
        question:       The user's original (or HyDE-expanded) question.
        context_chunks: Reranked chunk dicts from reranker.rerank().
        model:          Override model name. Defaults to settings.primary_llm.

    Returns:
        dict with keys:
          "answer"      : str   — LLM-generated answer grounded in context.
          "model_used"  : str   — Actual model that produced the answer.
          "token_count" : int   — Total tokens consumed by the LLM call.
          "citations"   : list[str] — Filenames cited in the answer.

    Raises:
        Exception: If both PRIMARY and FALLBACK LLM calls fail.
    """
    settings = get_settings()
    primary_model = model or settings.primary_llm
    fallback_model = settings.fallback_llm

    if not context_chunks:
        logger.warning(
            "generate_answer called with no context chunks. "
            "Returning 'not found' response without LLM call."
        )
        return {
            "answer": "I could not find relevant information in the knowledge base.",
            "model_used": primary_model,
            "token_count": 0,
            "citations": [],
        }

    context_str = _build_context_string(context_chunks)

    user_content = (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": _QA_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # ── Try primary LLM ───────────────────────────────────────────────────────
    answer_text: str = ""
    token_count: int = 0
    model_used: str = primary_model

    logger.info(
        "generate_answer: primary='%s' | %d chunks | question='%.80s…'",
        primary_model,
        len(context_chunks),
        question,
    )

    try:
        answer_text, token_count = await _call_openai(messages, primary_model)
        model_used = primary_model
        logger.info(
            "generate_answer: OpenAI '%s' succeeded (%d tokens).",
            primary_model,
            token_count,
        )

    except Exception as primary_exc:
        logger.warning(
            "Primary LLM '%s' failed: %s. Attempting fallback '%s'.",
            primary_model,
            primary_exc,
            fallback_model,
        )

        # ── Fallback to Anthropic ──────────────────────────────────────────
        try:
            answer_text, token_count = await _call_anthropic(messages, fallback_model)
            model_used = fallback_model
            logger.info(
                "generate_answer: Anthropic fallback '%s' succeeded (%d tokens).",
                fallback_model,
                token_count,
            )

        except Exception as fallback_exc:
            logger.exception(
                "Both primary ('%s') and fallback ('%s') LLMs failed. "
                "Primary error: %s | Fallback error: %s",
                primary_model,
                fallback_model,
                primary_exc,
                fallback_exc,
            )
            raise fallback_exc

    # ── Extract citations ─────────────────────────────────────────────────────
    citations = extract_citations(answer_text, context_chunks)

    logger.info(
        "generate_answer complete: model='%s' tokens=%d citations=%s",
        model_used,
        token_count,
        citations,
    )

    return {
        "answer":       answer_text,
        "model_used":   model_used,
        "token_count":  token_count,
        "citations":    citations,
    }
