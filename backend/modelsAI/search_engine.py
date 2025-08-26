# backend/modelsAI/search_engine.py
import re
import unicodedata
from typing import List, Dict, Any, Iterable

# ✅ Import the MODULE so we always see the live globals
import modelsAI.model_loader as ml

# Gemini service (RAG generator)
from services.gemini import generate_with_gemini, is_configured as gemini_ready

# -----------------------------
# Helpers
# -----------------------------
_TEXT_KEYS = (
    "text","content","response","answer","advice","data","message","body",
    "notes","summary","snippet","tip","label","title","question","explanation"
)

def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_str(x) -> str:
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""

def _iter_strings(entry) -> Iterable[str]:
    if entry is None:
        return
    if isinstance(entry, (str, bytes, int, float)):
        s = _to_str(entry)
        if s:
            yield s
        return
    if isinstance(entry, dict):
        for k in _TEXT_KEYS:
            if k in entry and entry[k]:
                for s in _iter_strings(entry[k]):
                    yield s
        for k, v in entry.items():
            if k in _TEXT_KEYS:
                continue
            for s in _iter_strings(v):
                yield s
        return
    if isinstance(entry, (list, tuple, set)):
        for v in list(entry)[:50]:
            for s in _iter_strings(v):
                yield s
        return
    s = _to_str(entry)
    if s:
        yield s

def _extract_text(entry, max_chars: int = 4000) -> str:
    parts: List[str] = []
    total = 0
    for s in _iter_strings(entry):
        s = _normalize_text(s)
        if not s:
            continue
        parts.append(s)
        total += len(s)
        if total >= max_chars:
            break
    return " ".join(parts)[:max_chars]

def _longest_entries(top_k: int = 3) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for i, entry in enumerate(ml.KNOWLEDGE_CORPUS):
        txt = _extract_text(entry)
        if not txt:
            continue
        candidates.append({"index": i, "content": txt, "score": len(txt)})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

# -----------------------------
# Keyword fallback search (robust)
# -----------------------------
def _keyword_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not ml.KNOWLEDGE_CORPUS:
        return []

    q = _normalize_text(query).lower()
    search_terms = set(q.split())

    if any(w in q for w in [
        "nervous","anxious","anxiety","worried","worry","stress","stressed","panic","panic attack","overthinking"
    ]):
        search_terms.update([
            "anxiety","anxious","worry","worried","stress","stressed","panic","calm","breathing","relax",
            "grounding","coping","mindfulness","visualization","soothe","reassure"
        ])

    if any(w in q for w in ["conference","presentation","presenting","work","meeting","interview","performance"]):
        search_terms.update([
            "work","professional","meeting","presentation","confidence","performance","rehearse","practice","slides",
            "prepare","timing","delivery","public speaking"
        ])

    if any(w in q for w in ["doctor","medical","health","appointment","results","lab","clinic","diagnosis"]):
        search_terms.update([
            "health","medical","doctor","appointment","care","treatment","lab","results","prepare","questions",
            "support person","self-care","follow-up"
        ])

    if any(w in q for w in ["week","busy","schedule","time","overwhelmed","overwhelm","deadline","too much"]):
        search_terms.update([
            "time","schedule","busy","overwhelmed","planning","organization","prioritize","break tasks","small steps",
            "pomodoro","rest"
        ])

    results: List[Dict[str, Any]] = []

    for i, entry in enumerate(ml.KNOWLEDGE_CORPUS):
        text = _extract_text(entry)
        if not text or len(text) < 6:
            continue
        text_l = text.lower()

        words = set(re.findall(r"\b\w+\b", text_l))
        overlap = len(search_terms & words)
        semantic = sum(1 for t in search_terms if t in text_l)
        total = overlap + semantic
        if total <= 0:
            continue

        score = total / (len(search_terms) or 1)
        results.append({
            "index": i,
            "content": text,
            "score": score,
            "overlap_count": overlap,
            "semantic_count": semantic,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# -----------------------------
# Vector search (if FAISS + retriever available)
# -----------------------------
def _vector_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not (ml.FAISS_INDEX and ml.RETRIEVER and ml.KNOWLEDGE_CORPUS):
        return []
    try:
        import numpy as np
        import faiss  # type: ignore
        qemb = ml.RETRIEVER.encode([query])
        faiss.normalize_L2(qemb)
        scores, idxs = ml.FAISS_INDEX.search(qemb.astype("float32"), top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(ml.KNOWLEDGE_CORPUS):
                out.append({
                    "index": int(idx),
                    "content": _extract_text(ml.KNOWLEDGE_CORPUS[idx]),
                    "score": float(score),
                })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out
    except Exception as e:
        print(f"[search_engine] vector search failed, fallback: {e}")
        return []

def improved_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Vector first, then robust keyword; if still empty, return longest entries as a last resort."""
    if not ml.MODEL_LOADED or not ml.KNOWLEDGE_CORPUS:
        return []
    hits = _vector_search(query, top_k)
    if not hits:
        hits = _keyword_search(query, top_k)
    if not hits:
        hits = _longest_entries(top_k)
    return hits

# -----------------------------
# Generation
# -----------------------------
def _intro_for(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["nervous","anxious","conference","presentation","presenting"]):
        return "I understand that nervousness before important events. "
    if any(w in q for w in ["doctor","medical","waiting","results","lab"]):
        return "Waiting for medical news can be really stressful. "
    if any(w in q for w in ["week","busy","ahead","overwhelmed","deadline"]):
        return "Big weeks can feel overwhelming. "
    return "I hear what you're going through. "

def _generate_with_llm(query: str, context_docs: List[str], quiz_summary: str = "") -> str:
    if gemini_ready():
        g = generate_with_gemini(query, context_docs, quiz_summary=quiz_summary)
        if g:
            return g
    if ml.GENERATOR is not None:
        try:
            context = " ".join(context_docs)
            prompt = (
                "You are MAI, a caring mental health buddy.\n\n"
                f"Context: {context}\n\n"
                f"User: {query}\n\n"
                "Respond with empathy and practical tips in 1-3 sentences:"
            )
            out = ml.GENERATOR(prompt, max_length=120, do_sample=True, temperature=0.7)
            return out[0]["generated_text"].strip()
        except Exception as e:
            print(f"[search_engine] local generation failed: {e}")
    intro = _intro_for(query)
    content = " ".join(context_docs)
    if len(content) > 400:
        parts = content.split(".")
        content = ". ".join(parts[:2]) + "."
    return intro + content

def use_best_training_content(user_input: str, relevant_docs: List[dict], quiz_summary: str = "") -> str:
    if not relevant_docs:
        return (
            "I want to help you with that. Can you tell me more about what's "
            "specifically concerning you so I can provide better guidance?"
        )
    docs = [d.get("content") for d in relevant_docs if d.get("content")]
    docs = [d for d in docs if isinstance(d, str) and d.strip()]
    if not docs:
        return (
            "I want to help you with that. Can you share a bit more detail so I "
            "can offer something more specific?"
        )
    return _generate_with_llm(user_input, docs[:3], quiz_summary=quiz_summary)

# === DEBUG HELPERS ===
def _debug_is_running() -> str:
    return "search_engine.py debug marker v3 (module import + recursive extractor + longest fallback)"

def debug_extract_entry(i: int) -> dict:
    if not (0 <= i < len(ml.KNOWLEDGE_CORPUS)):
        return {"error": f"index {i} out of range", "total": len(ml.KNOWLEDGE_CORPUS)}
    txt = _extract_text(ml.KNOWLEDGE_CORPUS[i])
    return {
        "index": i,
        "type": type(ml.KNOWLEDGE_CORPUS[i]).__name__,
        "extracted_text_preview": txt[:500],
        "extracted_text_len": len(txt),
    }

def debug_corpus_summary(max_items: int = 10) -> dict:
    kinds = {}
    samples = []
    for i, e in enumerate(ml.KNOWLEDGE_CORPUS[:max_items]):
        k = type(e).__name__
        kinds[k] = kinds.get(k, 0) + 1
        preview = _extract_text(e)[:180]
        keys = list(e.keys())[:10] if isinstance(e, dict) else None
        samples.append({"index": i, "type": k, "keys": keys, "preview": preview})
    return {"total": len(ml.KNOWLEDGE_CORPUS), "kinds": kinds, "samples": samples}
