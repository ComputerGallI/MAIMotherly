# mai fastapi - Better search to actually find your training content
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import numpy as np
from datetime import datetime
import re

# request/response models
class ChatRequest(BaseModel):
    user_input: str
    quiz_summary: Optional[str] = ""
    subscription_tier: Optional[str] = "free"

class ChatResponse(BaseModel):
    response: str
    suggestions: Optional[List[str]] = []

print("Starting MAI with improved search...")

app = FastAPI(title="MAI AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables
KNOWLEDGE_CORPUS = []
MODEL_LOADED = False

def load_your_trained_models():
    """Load your actual trained models"""
    global KNOWLEDGE_CORPUS, MODEL_LOADED
    
    try:
        artifacts_path = os.getenv('ARTIFACTS_PATH', './mai_artifacts')
        print(f"Looking for your trained models in: {artifacts_path}")
        
        corpus_path = f"{artifacts_path}/knowledge_corpus.pkl"
        if os.path.exists(corpus_path):
            with open(corpus_path, 'rb') as f:
                KNOWLEDGE_CORPUS = pickle.load(f)
            print(f"SUCCESS: Loaded {len(KNOWLEDGE_CORPUS)} knowledge entries")
            
            # Print first few entries to see their structure
            for i, entry in enumerate(KNOWLEDGE_CORPUS[:3]):
                print(f"Entry {i}: {entry}")
                
            MODEL_LOADED = True
            return True
        else:
            print(f"ERROR: Knowledge corpus not found: {corpus_path}")
            return False
        
    except Exception as e:
        print(f"ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def improved_search(query, top_k=3):
    """Improved search that's more likely to find matches"""
    if not KNOWLEDGE_CORPUS:
        return []
    
    query_lower = query.lower()
    print(f"SEARCHING for: '{query}'")
    
    # Create broader search terms
    search_terms = set()
    
    # Add original words
    search_terms.update(query_lower.split())
    
    # Add emotional/mental health related terms
    if any(word in query_lower for word in ["nervous", "anxious", "worried", "stress"]):
        search_terms.update(["anxiety", "anxious", "stress", "worried", "nervous", "calm", "breathing", "relax"])
    
    if any(word in query_lower for word in ["conference", "presentation", "work", "meeting"]):
        search_terms.update(["work", "professional", "meeting", "presentation", "confidence", "performance"])
    
    if any(word in query_lower for word in ["doctor", "medical", "health", "appointment"]):
        search_terms.update(["health", "medical", "doctor", "appointment", "care", "treatment"])
    
    if any(word in query_lower for word in ["week", "busy", "schedule", "time"]):
        search_terms.update(["time", "schedule", "busy", "overwhelmed", "planning", "organization"])
    
    print(f"Expanded search terms: {search_terms}")
    
    results = []
    
    for i, entry in enumerate(KNOWLEDGE_CORPUS):
        # Extract text from entry
        text_content = ""
        if isinstance(entry, dict):
            for key in ['text', 'content', 'response', 'answer', 'advice', 'data', 'message']:
                if key in entry and entry[key]:
                    text_content = str(entry[key]).lower()
                    break
        elif isinstance(entry, str):
            text_content = entry.lower()
        elif isinstance(entry, list) and len(entry) > 0:
            text_content = str(entry[0]).lower()
        
        if text_content and len(text_content) > 5:
            # Check for any overlap with search terms
            text_words = set(re.findall(r'\b\w+\b', text_content))
            
            overlap = len(search_terms.intersection(text_words))
            
            # Also check for partial matches and semantic similarity
            semantic_score = 0
            for search_term in search_terms:
                if search_term in text_content:
                    semantic_score += 1
            
            total_score = overlap + semantic_score
            
            if total_score > 0:
                final_score = total_score / len(search_terms) if search_terms else 0
                results.append({
                    'content': text_content,
                    'original_entry': entry,
                    'score': final_score,
                    'overlap_count': overlap,
                    'semantic_count': semantic_score,
                    'index': i
                })
                print(f"Match found: {text_content[:100]}... (score: {final_score})")
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Found {len(results)} total matches")
    return results[:top_k]

def use_best_training_content(user_input, relevant_docs, quiz_summary=""):
    """Use the best matching content from training data"""
    
    if not relevant_docs or len(relevant_docs) == 0:
        print("NO RELEVANT CONTENT FOUND - using generic fallback")
        return "I want to help you with that. Can you tell me more about what's specifically concerning you so I can provide better guidance?"
    
    # Use the best match
    best_match = relevant_docs[0]
    content = best_match['content']
    
    print(f"USING TRAINING CONTENT: {content}")
    
    # If the content is very short, combine with second best
    if len(content) < 50 and len(relevant_docs) > 1:
        content += " " + relevant_docs[1]['content']
    
    # Clean up the content
    if len(content) > 400:
        sentences = content.split('.')
        content = '. '.join(sentences[:2]) + '.'
    
    # Make it more conversational
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["nervous", "anxious", "conference"]):
        intro = "I understand that nervousness before important events. "
    elif any(word in user_lower for word in ["doctor", "medical", "waiting"]):
        intro = "Waiting for medical news can be really stressful. "
    elif any(word in user_lower for word in ["week", "busy", "ahead"]):
        intro = "Big weeks can feel overwhelming. "
    else:
        intro = "I hear what you're going through. "
    
    final_response = intro + content
    
    # Remove any duplicate intro phrases
    final_response = re.sub(r'^(I understand|I hear|That sounds|Thank you for sharing).*?\. (I understand|I hear|That sounds)', r'\1', final_response)
    
    print(f"FINAL RESPONSE: {final_response}")
    return final_response

# Load models
models_loaded = load_your_trained_models()

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "models_loaded": models_loaded,
        "knowledge_entries": len(KNOWLEDGE_CORPUS),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        user_input = request.user_input.strip()
        print(f"\n=== PROCESSING: {user_input} ===")
        
        if not user_input:
            return ChatResponse(response="I'm here to listen. What's on your mind?")
        
        if models_loaded and KNOWLEDGE_CORPUS:
            # Use improved search
            relevant_docs = improved_search(user_input, top_k=3)
            response_text = use_best_training_content(user_input, relevant_docs, request.quiz_summary)
        else:
            response_text = "I'm having trouble accessing my knowledge. Can you tell me more about what's on your mind?"
        
        # Simple suggestions
        suggestions = []
        user_lower = user_input.lower()
        if any(word in user_lower for word in ["nervous", "anxious"]):
            suggestions = ["Try deep breathing", "Practice positive visualization", "Take breaks"]
        elif any(word in user_lower for word in ["doctor", "medical"]):
            suggestions = ["Prepare questions", "Bring support person", "Practice self-care"]
        
        print(f"=== RETURNING: {response_text[:100]}... ===\n")
        
        return ChatResponse(
            response=response_text,
            suggestions=suggestions
        )
        
    except Exception as e:
        print(f"ERROR: {e}")
        return ChatResponse(response="I'm here to help. Can you tell me more about what's on your mind?")

@app.get("/debug/corpus")
async def debug_corpus():
    if not KNOWLEDGE_CORPUS:
        return {"error": "No knowledge corpus loaded"}
    
    return {
        "total_entries": len(KNOWLEDGE_CORPUS),
        "sample_entries": KNOWLEDGE_CORPUS[:5]
    }

@app.get("/debug/search/{query}")
async def debug_search(query: str):
    results = improved_search(query, top_k=5)
    return {
        "query": query,
        "results_found": len(results),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)