# modelsAI/search_engine.py
import re
from typing import List, Dict, Any

from .model_loader import (
    KNOWLEDGE_CORPUS,
    MODEL_LOADED,
    FAISS_INDEX,
    RETRIEVER,
    GENERATOR,
)

# -----------------------------
# Fallback keyword search (your original logic, slightly refactored)
# -----------------------------
def _keyword_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not KNOWLEDGE_CORPUS:
        return []

    q = query.lower()
    search_terms = set(q.split())

    if any(w in q for w in ["nervous", "anxious", "worried", "stress"]):
        search_terms.update(["anxiety", "calm", "breathing", "relax", "stressed"])
    if any(w in q for w in ["conference", "presentation", "work", "meeting"]):
        search_terms.update(["professional", "confidence", "performance"])
    if any(w in q for w in ["doctor", "medical", "health", "appointment"]):
        search_terms.update(["care", "treatment"])
    if any(w in q for w in ["week", "busy", "schedule", "time"]):
        search_terms.update(["overwhelmed", "planning", "organization"])

    results = []
    for i, entry in enumerate(KNOWLEDGE_CORPUS):
        text_content = str(entry).lower()
        if not text_content:
            continue
        words = set(re.findall(r"\b\w+\b", text_content))
        overlap = len(search_terms & words)
        semantic_score = sum(1 for t in search_terms if t in text_content)
        total = overlap + semantic_score
        if total > 0:
            results.append({
                "index": i,
                "content": text_content,
                "score": total / (len(search_terms) or 1),
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# -----------------------------
# Vector search (if FAISS + retriever available)
# -----------------------------
def _vector_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not (FAISS_INDEX and RETRIEVER and KNOWLEDGE_CORPUS):
        return []

    try:
        import numpy as np
        import faiss  # type: ignore
        query_emb = RETRIEVER.encode([query])
        # normalize for cosine similarity if index was built with IP
        faiss.normalize_L2(query_emb)
        scores, idxs = FAISS_INDEX.search(query_emb.astype("float32"), top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(KNOWLEDGE_CORPUS):
                out.append({
                    "index": int(idx),
                    "content": str(KNOWLEDGE_CORPUS[idx]),
                    "score": float(score),
                })
        # Sort descending by score
        out.sort(key=lambda x: x["score"], reverse=True)
        return out
    except Exception as e:
        print(f"[search_engine] vector search failed, falling back: {e}")
        return []

def improved_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Try vector search first (if available). If not, or if it returns empty,
    use the keyword fallback.
    """
    if not MODEL_LOADED or not KNOWLEDGE_CORPUS:
        return []

    vec_hits = _vector_search(query, top_k)
    if vec_hits:
        return vec_hits
    return _keyword_search(query, top_k)

# -----------------------------
# Response construction
# -----------------------------
def _intro_for(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["nervous", "anxious", "conference", "presentation"]):
        return "I understand that nervousness before important events. "
    if any(w in q for w in ["doctor", "medical", "waiting"]):
        return "Waiting for medical news can be really stressful. "
    if any(w in q for w in ["week", "busy", "ahead"]):
        return "Big weeks can feel overwhelming. "
    return "I hear what you're going through. "

def _generate_with_llm(query: str, context_docs: List[str]) -> str:
    """
    If a transformers generator pipeline is available, use it to produce
    an empathetic response grounded in context. Otherwise, fall back.
    """
    # If NO generator, fallback
    if GENERATOR is None:
        intro = _intro_for(query)
        content = " ".join(context_docs)
        if len(content) > 400:
            parts = content.split(".")
            content = ". ".join(parts[:2]) + "."
        return intro + content

    # With generator available
    context = " ".join(context_docs)
    prompt = (
        "You are MAI, a caring mental health buddy.\n\n"
        f"Context: {context}\n\n"
        f"User: {query}\n\n"
        "Respond with empathy and practical tips in 1-3 sentences:"
    )
    try:
        gen = GENERATOR(prompt, max_length=120, do_sample=True, temperature=0.7)
        text = gen[0]["generated_text"]
        return text.strip()
    except Exception as e:
        print(f"[search_engine] generation failed, fallback: {e}")
        return _intro_for(query) + (context_docs[0] if context_docs else "")

def use_best_training_content(user_input: str, relevant_docs: List[dict], quiz_summary: str = "") -> str:
    """
    Compose final response using retrieved docs and optional generator model.
    """
    if not relevant_docs:
        return (
            "I want to help you with that. Can you tell me more about what's "
            "specifically concerning you so I can provide better guidance?"
        )

    # Collect top doc texts
    docs = [d["content"] for d in relevant_docs if d.get("content")]
    if not docs:
        return (
            "I want to help you with that. Can you share a bit more detail so I "
            "can offer something more specific?"
        )

    # Optionally prepend quiz summary as soft context
    if quiz_summary:
        docs = [f"(User profile hints: {quiz_summary})"] + docs

    # Generate or fallback
    reply = _generate_with_llm(user_input, docs[:3])
    # Light de-duplication of repeated intros
    reply = re.sub(
        r"^(I understand|I hear|That sounds|Thank you for sharing).*?\. (I understand|I hear|That sounds)",
        r"\1",
        reply,
    )
    return reply
