# mai fastapi - RAG implementation for your actual trained files
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
        
        # Check if artifacts directory exists
        if not os.path.exists(artifacts_path):
            print(f"ERROR: Artifacts directory not found: {artifacts_path}")
            return False
        
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
        else:
            print("WARNING: No config.json found, using defaults")
        
        # Load your FAISS index
        if FAISS_AVAILABLE:
            faiss_path = f"{artifacts_path}/faiss_index.bin"
            if os.path.exists(faiss_path):
                FAISS_INDEX = faiss.read_index(faiss_path)
                print(f"SUCCESS: Loaded FAISS index with {FAISS_INDEX.ntotal} vectors")
            else:
                print(f"WARNING: FAISS index not found: {faiss_path}")
        else:
            print("INFO: FAISS not available, will use keyword search")
        
        MODEL_LOADED = True
        print("SUCCESS: Your trained models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR loading your trained models: {e}")
        import traceback
        traceback.print_exc()
        return False

def search_your_knowledge(query, top_k=3):
    """Search your knowledge using FAISS or fallback to keyword search"""
    if not KNOWLEDGE_CORPUS:
        print("ERROR: No knowledge corpus available")
        return []
    
    # Try FAISS search first if available
    if FAISS_INDEX is not None and FAISS_AVAILABLE:
        try:
            # This would require the sentence transformer model to encode the query
            # For now, fall back to keyword search
            print("INFO: FAISS index available but need encoder, using keyword search")
        except Exception as e:
            print(f"FAISS search failed: {e}")
    
    # Keyword-based search as fallback
    query_words = set(query.lower().split())
    results = []
    
    for i, entry in enumerate(KNOWLEDGE_CORPUS):
        # Handle different possible formats of your knowledge corpus
        text_content = ""
        
        if isinstance(entry, dict):
            # Try different possible keys
            for key in ['text', 'content', 'response', 'answer', 'question', 'data']:
                if key in entry and entry[key]:
                    text_content = str(entry[key])
                    break
        elif isinstance(entry, str):
            text_content = entry
        elif isinstance(entry, list) and len(entry) > 0:
            text_content = str(entry[0])  # Use first element if it's a list
        
        if text_content and len(text_content) > 10:  # Only consider substantial content
            text_words = set(text_content.lower().split())
            # Calculate similarity score
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                score = overlap / len(query_words.union(text_words))
                results.append({
                    'content': text_content,
                    'score': score,
                    'index': i,
                    'original_entry': entry
                })
    
    # Sort by score and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"INFO: Found {len(results)} relevant entries for query: {query}")
    return results[:top_k]

def create_response_from_knowledge(user_input, relevant_docs, quiz_summary=""):
    """Create response using your retrieved knowledge"""
    
    if not relevant_docs:
        print("WARNING: No relevant documents found in your knowledge base")
        return "I want to help you with that. Can you tell me more about what's going on so I can better understand your situation?"
    
    print(f"INFO: Using {len(relevant_docs)} documents from your trained knowledge")
    
    # Extract the most relevant content
    best_content = relevant_docs[0]['content']
    
    # Clean up the response
    if len(best_content) > 400:
        # Truncate very long responses at sentence boundaries
        sentences = best_content.split('.')
        truncated = '. '.join(sentences[:3])
        if not truncated.endswith('.'):
            truncated += '.'
        best_content = truncated
    
    # Add personality context if available
    if quiz_summary:
        if 'introvert' in quiz_summary.lower() or 'INFP' in quiz_summary or 'ISFP' in quiz_summary:
            best_content += " Take some quiet time to process this if you need to."
        elif 'extrovert' in quiz_summary.lower() or 'ENFP' in quiz_summary or 'ESFP' in quiz_summary:
            best_content += " Consider talking this through with someone you trust."
    
    print(f"RESPONSE: Generated from your training data: {best_content[:100]}...")
    return best_content

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
        print(f"PROCESSING query with your trained models: {user_input}")
        
        if not user_input:
            return ChatResponse(response="I'm here to listen. What would you like to talk about?")
        
        # Use your trained knowledge
        if models_loaded and KNOWLEDGE_CORPUS:
            print("INFO: Searching your trained knowledge corpus...")
            relevant_docs = search_your_knowledge(user_input, top_k=3)
            
            if relevant_docs:
                print(f"SUCCESS: Found {len(relevant_docs)} relevant entries from your training")
                response_text = create_response_from_knowledge(user_input, relevant_docs, request.quiz_summary)
            else:
                print("WARNING: No matches found in your trained knowledge")
                response_text = "I understand you're looking for support. While I don't have specific guidance for this situation in my training, I'm here to listen. Can you tell me more about what you're experiencing?"
        else:
            print("ERROR: Your trained models are not loaded")
            response_text = "I'm having trouble accessing my trained knowledge right now. Can you tell me more about what's on your mind?"
        
        # Generate suggestions based on input
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