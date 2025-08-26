from fastapi import APIRouter
from datetime import datetime

from pydantic_schemas.schema import ChatRequest, ChatResponse

# Import the module so we always see the current globals
import modelsAI.model_loader as ml
from modelsAI.search_engine import improved_search, use_best_training_content

router = APIRouter()

# Ensure models/artifacts are loaded in THIS process
ml.load_your_trained_models()

@router.get("/health")
async def health_check():
    return {
        "status": "operational",
        "models_loaded": ml.MODEL_LOADED,
        "knowledge_entries": len(ml.KNOWLEDGE_CORPUS),
        "timestamp": datetime.now().isoformat()
    }

@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    user_input = request.user_input.strip()
    if not user_input:
        return ChatResponse(response="I'm here to listen. What's on your mind?")

    if ml.MODEL_LOADED:
        relevant_docs = improved_search(user_input, top_k=3)
        # If your function accepts quiz_summary, pass it:
        response_text = use_best_training_content(user_input, relevant_docs, request.quiz_summary)
    else:
        response_text = "I'm having trouble accessing my knowledge. Can you tell me more?"

    # --- suggestions (additive, deduped) ---
    suggestions = []
    user_lower = user_input.lower()

    # medical/doctor related
    if any(w in user_lower for w in ["doctor", "medical", "appointment", "test results", "health"]):
        suggestions += ["Prepare questions", "Bring support person", "Practice self-care"]

    # anxiety/stress related
    if any(w in user_lower for w in ["anxious", "nervous", "worry", "worried", "stress", "stressed"]):
        suggestions += ["Try deep breathing", "Practice positive visualization", "Take breaks"]

    # dedupe while preserving order
    seen = set()
    suggestions = [s for s in suggestions if not (s in seen or seen.add(s))]


    return ChatResponse(response=response_text, suggestions=suggestions)

@router.get("/debug/corpus")
async def debug_corpus():
    return {
        "total_entries": len(ml.KNOWLEDGE_CORPUS),
        "sample_entries": ml.KNOWLEDGE_CORPUS[:5]
    }

@router.get("/debug/search/{query}")
async def debug_search(query: str):
    results = improved_search(query, top_k=5)
    return {"query": query, "results_found": len(results), "results": results}
