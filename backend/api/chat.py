from fastapi import APIRouter
from datetime import datetime

from pydantic_schemas.schema import ChatRequest, ChatResponse 
import modelsAI.model_loader as ml
from modelsAI.search_engine import improved_search, use_best_training_content

router = APIRouter()

# Ensure artifacts are loaded in this process
ml.load_your_trained_models()

@router.get("/health")
async def health_check():
    return {
        "status": "operational",
        "models_loaded": ml.MODEL_LOADED,
        "knowledge_entries": len(ml.KNOWLEDGE_CORPUS),
        "timestamp": datetime.now().isoformat(),
    }

@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    user_input = request.user_input.strip()
    if not user_input:
        return ChatResponse(response="I'm here to listen. What's on your mind?")

    if ml.MODEL_LOADED:
        relevant_docs = improved_search(user_input, top_k=3)
        # pass quiz_summary so Gemini can adapt tone/strategy
        response_text = use_best_training_content(user_input, relevant_docs, request.quiz_summary)
    else:
        response_text = "I'm having trouble accessing my knowledge. Can you tell me more?"

    # Additive suggestions (not mutually exclusive) with dedupe
    suggestions = []
    user_lower = user_input.lower()

    # medical/doctor related
    if any(w in user_lower for w in ["doctor", "medical", "appointment", "results", "health"]):
        suggestions += ["Prepare questions", "Bring support person", "Practice self-care"]

    # anxiety/stress related
    if any(w in user_lower for w in ["anxious", "nervous", "worry", "worried", "stress", "stressed"]):
        suggestions += ["Try deep breathing", "Practice positive visualization", "Take breaks"]

    # dedupe preserving order
    seen = set()
    suggestions = [s for s in suggestions if not (s in seen or seen.add(s))]

    return ChatResponse(response=response_text, suggestions=suggestions)

@router.get("/debug/corpus")
async def debug_corpus():
    return {
        "total_entries": len(ml.KNOWLEDGE_CORPUS),
        "sample_entries": ml.KNOWLEDGE_CORPUS[:5],
    }

@router.get("/debug/search/{query}")
async def debug_search(query: str):
    results = improved_search(query, top_k=5)
    return {"query": query, "results_found": len(results), "results": results}

# ===== EXTRA DEBUG ROUTES =====
from fastapi import Query
from modelsAI import search_engine as se

@router.get("/debug/version")
async def debug_version():
    return {
        "module_marker": se._debug_is_running(),
        "models_loaded": ml.MODEL_LOADED,
        "knowledge_entries": len(ml.KNOWLEDGE_CORPUS),
    }

@router.get("/debug/corpus/summary")
async def debug_corpus_summary(max_items: int = 10):
    return se.debug_corpus_summary(max_items=max_items)

@router.get("/debug/corpus/extract/{i}")
async def debug_corpus_extract(i: int):
    return se.debug_extract_entry(i)

@router.get("/debug/raw_search")
async def debug_raw_search(q: str = Query(..., min_length=1), limit: int = 5):
    """
    Brute-force substring search across extracted texts (case-insensitive).
    This bypasses the keyword-scoring logic to confirm whether the corpus contains the term at all.
    """
    ql = q.lower()
    hits = []
    for i, entry in enumerate(ml.KNOWLEDGE_CORPUS):
        txt = se._extract_text(entry)
        if not txt:
            continue
        if ql in txt.lower():
            hits.append({"index": i, "content_preview": txt[:300]})
            if len(hits) >= limit:
                break
    return {"query": q, "hits_found": len(hits), "hits": hits}
