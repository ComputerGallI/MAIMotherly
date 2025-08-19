# mai fastapi - use retrieved content instead of templates
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import numpy as np
from datetime import datetime

# Try to import AI libraries
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS library available")
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS library not available, using fallback search")

# request/response models
class ChatRequest(BaseModel):
    user_input: str
    quiz_summary: Optional[str] = ""
    subscription_tier: Optional[str] = "free"

class ChatResponse(BaseModel):
    response: str
    suggestions: Optional[List[str]] = []

print("Starting MAI with your trained models...")

app = FastAPI(title="MAI AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables for RAG system
KNOWLEDGE_CORPUS = []
FAISS_INDEX = None
CONFIG = None
MODEL_LOADED = False

def load_your_trained_models():
    """Load your actual trained models"""
    global KNOWLEDGE_CORPUS, FAISS_INDEX, CONFIG, MODEL_LOADED
    
    try:
        artifacts_path = os.getenv('ARTIFACTS_PATH', './mai_artifacts')
        print(f"Looking for your trained models in: {artifacts_path}")
        
        # Load your knowledge corpus
        corpus_path = f"{artifacts_path}/knowledge_corpus.pkl"
        if os.path.exists(corpus_path):
            with open(corpus_path, 'rb') as f:
                KNOWLEDGE_CORPUS = pickle.load(f)
            print(f"SUCCESS: Loaded knowledge corpus with {len(KNOWLEDGE_CORPUS)} entries")
        else:
            print(f"ERROR: Knowledge corpus not found: {corpus_path}")
            return False
        
        # Load your config
        config_path = f"{artifacts_path}/config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                CONFIG = json.load(f)
            print(f"SUCCESS: Loaded config: {CONFIG}")
        
        # Load your FAISS index
        if FAISS_AVAILABLE:
            faiss_path = f"{artifacts_path}/faiss_index.bin"
            if os.path.exists(faiss_path):
                FAISS_INDEX = faiss.read_index(faiss_path)
                print(f"SUCCESS: Loaded FAISS index with {FAISS_INDEX.ntotal} vectors")
        
        MODEL_LOADED = True
        print("SUCCESS: Your trained models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR loading your trained models: {e}")
        import traceback
        traceback.print_exc()
        return False

def search_your_knowledge(query, top_k=3):
    """Search your knowledge using keyword matching"""
    if not KNOWLEDGE_CORPUS:
        print("ERROR: No knowledge corpus available")
        return []
    
    query_words = set(query.lower().split())
    results = []
    
    for i, entry in enumerate(KNOWLEDGE_CORPUS):
        text_content = ""
        
        if isinstance(entry, dict):
            for key in ['text', 'content', 'response', 'answer', 'question', 'data']:
                if key in entry and entry[key]:
                    text_content = str(entry[key])
                    break
        elif isinstance(entry, str):
            text_content = entry
        elif isinstance(entry, list) and len(entry) > 0:
            text_content = str(entry[0])
        
        if text_content and len(text_content) > 10:
            text_words = set(text_content.lower().split())
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                score = overlap / len(query_words.union(text_words))
                results.append({
                    'content': text_content,
                    'score': score,
                    'index': i,
                    'original_entry': entry
                })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"INFO: Found {len(results)} relevant entries for query: {query}")
    return results[:top_k]

def actually_use_retrieved_content(user_input, relevant_docs, quiz_summary=""):
    """ACTUALLY use the retrieved content instead of generating templates"""
    
    if not relevant_docs:
        return "I want to help you with that. Can you tell me more about what's going on so I can better understand your situation?"
    
    # Get the BEST matching content from your training data
    best_match = relevant_docs[0]
    retrieved_content = best_match['content']
    
    print(f"USING RETRIEVED CONTENT: {retrieved_content}")
    
    # Clean up the content if needed
    if len(retrieved_content) > 500:
        # If it's very long, take first few sentences
        sentences = retrieved_content.split('.')
        retrieved_content = '. '.join(sentences[:3]) + '.'
    
    # Add a natural introduction to make it conversational
    introduction_phrases = [
        "I understand what you're going through. ",
        "That sounds really challenging. ",
        "I hear you. ",
        "Thank you for sharing that with me. "
    ]
    
    # Choose introduction based on user input sentiment
    user_lower = user_input.lower()
    if any(word in user_lower for word in ["baby", "crying", "overwhelmed"]):
        intro = "Being overwhelmed with a crying baby is so exhausting. "
    elif any(word in user_lower for word in ["anxious", "worried", "stress"]):
        intro = "I understand how anxiety can feel overwhelming. "
    elif any(word in user_lower for word in ["sad", "depressed", "down"]):
        intro = "I hear that you're going through a tough time. "
    else:
        intro = "I understand what you're experiencing. "
    
    # Combine intro with retrieved content
    final_response = intro + retrieved_content
    
    # Add personality context if available
    if quiz_summary:
        if any(personality in quiz_summary for personality in ['INFP', 'ISFP', 'introvert']):
            final_response += " Take some quiet time to process this - that's perfectly okay for someone like you."
        elif any(personality in quiz_summary for personality in ['ENFP', 'ESFP', 'extrovert']):
            final_response += " Consider talking this through with someone you trust - that often helps people like you."
    
    print(f"FINAL RESPONSE USING YOUR TRAINING DATA: {final_response}")
    return final_response

# Load your trained models on startup
models_loaded = load_your_trained_models()

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "your_models_loaded": models_loaded,
        "knowledge_corpus_entries": len(KNOWLEDGE_CORPUS),
        "faiss_index_available": FAISS_INDEX is not None,
        "faiss_index_size": FAISS_INDEX.ntotal if FAISS_INDEX else 0,
        "config_loaded": CONFIG is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        user_input = request.user_input.strip()
        print(f"PROCESSING query: {user_input}")
        
        if not user_input:
            return ChatResponse(response="I'm here to listen. What would you like to talk about?")
        
        # Search your trained knowledge
        if models_loaded and KNOWLEDGE_CORPUS:
            print("SEARCHING your trained knowledge corpus...")
            relevant_docs = search_your_knowledge(user_input, top_k=3)
            
            if relevant_docs and len(relevant_docs) > 0:
                print(f"FOUND {len(relevant_docs)} relevant entries from your training")
                # ACTUALLY USE the retrieved content
                response_text = actually_use_retrieved_content(user_input, relevant_docs, request.quiz_summary)
            else:
                print("NO matches found in your trained knowledge")
                response_text = "I understand you're looking for support. While I don't have specific guidance for this situation in my training, I'm here to listen. Can you tell me more about what you're experiencing?"
        else:
            print("ERROR: Your trained models are not loaded")
            response_text = "I'm having trouble accessing my trained knowledge right now. Can you tell me more about what's on your mind?"
        
        # Generate suggestions based on input (keep these generic)
        suggestions = []
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["baby", "crying", "sleep", "infant"]):
            suggestions = ["Try white noise", "Check if baby needs feeding", "Consider swaddling"]
        elif any(word in user_lower for word in ["anxious", "anxiety", "worried"]):
            suggestions = ["Practice deep breathing", "Try grounding techniques", "Talk to someone"]
        elif any(word in user_lower for word in ["sad", "depressed", "down"]):
            suggestions = ["Reach out to support", "Practice self-care", "Consider professional help"]
        
        return ChatResponse(
            response=response_text,
            suggestions=suggestions
        )
        
    except Exception as e:
        print(f"ERROR in generate_response: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            response="I'm experiencing some technical difficulties, but I'm still here for you. Can you tell me more about what's on your mind?"
        )

@app.get("/debug/corpus")
async def debug_corpus():
    """Debug endpoint to check your knowledge corpus"""
    if not KNOWLEDGE_CORPUS:
        return {"error": "No knowledge corpus loaded"}
    
    sample_entries = KNOWLEDGE_CORPUS[:3] if len(KNOWLEDGE_CORPUS) >= 3 else KNOWLEDGE_CORPUS
    return {
        "total_entries": len(KNOWLEDGE_CORPUS),
        "sample_entries": sample_entries,
        "faiss_index_size": FAISS_INDEX.ntotal if FAISS_INDEX else 0,
        "config": CONFIG
    }

@app.get("/debug/search/{query}")
async def debug_search(query: str):
    """Debug endpoint to test search with your knowledge"""
    results = search_your_knowledge(query, top_k=5)
    return {
        "query": query,
        "results_found": len(results),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)