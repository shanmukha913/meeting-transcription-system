import math
import re
from typing import Any

import ollama

from services.settings import EMBEDDING_MODEL


CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_SENTENCES = 1
TOP_K = 4
MIN_SIMILARITY = 0.20


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    words = sentence.split()
    pieces = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                word[index:index + max_chars]
                for index in range(0, len(word), max_chars)
            )
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_transcript(
    transcript: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> list[str]:
    """Create readable, sentence-aware transcript chunks with light overlap."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative.")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Transcript is empty.")

    sentences = []
    for paragraph in re.split(r"\n\s*\n", transcript.strip()):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            sentences.extend(
                _split_long_sentence(sentence.strip(), max_chars)
                for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
                if sentence.strip()
            )
    flat_sentences = [sentence for group in sentences for sentence in group]
    chunks = []
    current: list[str] = []
    current_length = 0
    for sentence in flat_sentences:
        candidate_length = current_length + len(sentence) + (
            1 if current else 0
        )
        if current and candidate_length > max_chars:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            overlap_length = sum(map(len, overlap)) + max(0, len(overlap) - 1)
            if overlap and overlap_length + len(sentence) + 1 > max_chars:
                overlap = []
            current = list(overlap)
            current_length = overlap_length if overlap else 0
        current.append(sentence)
        current_length += len(sentence) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _validate_embedding(embedding: list[float]) -> list[float]:
    if not embedding or any(not math.isfinite(value) for value in embedding):
        raise ValueError("The local embedding model returned an invalid vector.")
    return embedding


def build_transcript_index(transcript: str) -> list[dict[str, Any]]:
    """Create transcript embeddings using the local Ollama embedding model."""
    chunks = chunk_transcript(transcript)
    response = ollama.embed(model=EMBEDDING_MODEL, input=chunks)
    embeddings = response.embeddings
    if len(embeddings) != len(chunks):
        raise ValueError("The embedding model returned an unexpected result count.")

    index = []
    for chunk_id, (text, embedding) in enumerate(
        zip(chunks, embeddings), start=1
    ):
        index.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "embedding": _validate_embedding(embedding),
            }
        )
    if len({len(item["embedding"]) for item in index}) != 1:
        raise ValueError("The embedding model returned inconsistent vector sizes.")
    return index


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity for two embedding vectors."""
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cannot compare a zero-length embedding vector.")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def retrieve_relevant_chunks(
    question: str,
    index: list[dict[str, Any]],
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
) -> list[dict[str, Any]]:
    """Embed a question locally and return relevant transcript excerpts."""
    if not question.strip():
        raise ValueError("Question is empty.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between zero and one.")
    if not index:
        return []

    response = ollama.embed(model=EMBEDDING_MODEL, input=question.strip())
    if len(response.embeddings) != 1:
        raise ValueError("The question embedding model returned an unexpected result.")
    question_embedding = _validate_embedding(response.embeddings[0])

    ranked = []
    for item in index:
        similarity = cosine_similarity(question_embedding, item["embedding"])
        if similarity >= min_similarity:
            ranked.append(
                {
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "similarity": similarity,
                }
            )
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked[:top_k]
