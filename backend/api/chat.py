from fastapi import APIRouter
from datetime import datetime
from models.schema import ChatRequest, ChatResponse
from core.model_loader import load_your_trained_models, KNOWLEDGE_CORPUS, MODEL_LOADED
from core.search_engine import improved_search, use_best_training_content

router = APIRouter()

models_loaded = load_your_trained_models()

@router.get("/health")
async def health_check():
    return {
        "status": "operational",
        "models_loaded": models_loaded,
        "knowledge_entries": len(KNOWLEDGE_CORPUS),
        "timestamp": datetime.now().isoformat()
    }

@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    user_input = request.user_input.strip()
    if not user_input:
        return ChatResponse(response="I'm here to listen. What's on your mind?")

    if models_loaded:
        relevant_docs = improved_search(user_input, top_k=3)
        response_text = use_best_training_content(user_input, relevant_docs)
    else:
        response_text = "I'm having trouble accessing my knowledge. Can you tell me more?"

    suggestions = []
    user_lower = user_input.lower()
    if "anxious" in user_lower or "nervous" in user_lower:
        suggestions = ["Try deep breathing", "Practice positive visualization"]
    elif "doctor" in user_lower:
        suggestions = ["Prepare questions", "Bring support person"]

    return ChatResponse(response=response_text, suggestions=suggestions)

@router.get("/debug/corpus")
async def debug_corpus():
    return {
        "total_entries": len(KNOWLEDGE_CORPUS),
        "sample_entries": KNOWLEDGE_CORPUS[:5]
    }

@router.get("/debug/search/{query}")
async def debug_search(query: str):
    results = improved_search(query, top_k=5)
    return {"query": query, "results_found": len(results), "results": results}
