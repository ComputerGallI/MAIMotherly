# mai fastapi - ACTUALLY use your specific training data
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

print("Starting MAI with your specific training data...")

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
            
            # Print actual content to understand structure
            print("Sample training content:")
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

def smart_search_your_content(query, top_k=3):
    """Smart search specifically for your training data format"""
    if not KNOWLEDGE_CORPUS:
        return []
    
    query_lower = query.lower()
    print(f"SEARCHING your training data for: '{query}'")
    
    # Based on your corpus, it looks like entries are strings
    # Let's create a comprehensive search
    results = []
    
    # Emotional keywords mapping
    emotion_keywords = {
        'sad': ['sad', 'sadness', 'down', 'depressed', 'low', 'feelings', 'valid', 'pass'],
        'anxious': ['anxious', 'anxiety', 'worry', 'nervous', 'breathing', 'calm'],
        'overwhelmed': ['overwhelmed', 'tasks', 'break', 'pieces', 'progress', 'small'],
        'tired': ['tired', 'energy', 'low', 'valid', 'rest', 'water', 'self-care'],
        'relationships': ['relationships', 'boundaries', 'saying', 'no', 'protects', 'mental', 'health'],
        'stress': ['stress', 'overwhelmed', 'tasks', 'break', 'breathing', 'calm']
    }
    
    # Expand search terms based on emotional context
    search_terms = set(query_lower.split())
    
    # Add related terms based on emotional keywords
    for emotion, keywords in emotion_keywords.items():
        if any(term in query_lower for term in [emotion]):
            search_terms.update(keywords)
    
    print(f"Expanded search terms: {search_terms}")
    
    # Search through your training content
    for i, entry in enumerate(KNOWLEDGE_CORPUS):
        # Your entries appear to be strings based on the debug output
        if isinstance(entry, str):
            content = entry.lower()
        elif isinstance(entry, dict):
            # Handle if some entries are dictionaries
            content = str(entry.get('content', entry.get('text', str(entry)))).lower()
        else:
            content = str(entry).lower()
        
        # Calculate relevance score
        score = 0
        content_words = set(re.findall(r'\b\w+\b', content))
        
        # Direct word matches
        matches = search_terms.intersection(content_words)
        score += len(matches) * 2
        
        # Partial matches and semantic similarity
        for search_term in search_terms:
            if search_term in content:
                score += 1
        
        # Boost score for exact emotional matches
        if any(emotion in query_lower for emotion in ['sad', 'anxious', 'overwhelmed', 'tired']) and \
           any(emotion in content for emotion in ['valid', 'feeling', 'pass', 'breathing', 'care']):
            score += 3
        
        if score > 0:
            results.append({
                'content': entry if isinstance(entry, str) else str(entry),
                'score': score,
                'matches': matches,
                'index': i
            })
            print(f"MATCH FOUND (score {score}): {entry[:100]}...")
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(results)} total matches from your training data")
    
    return results[:top_k]

def use_your_training_content(user_input, relevant_docs, quiz_summary=""):
    """Use YOUR specific training content directly"""
    
    if not relevant_docs or len(relevant_docs) == 0:
        print("NO MATCHES in your training data - this shouldn't happen with good content")
        return "I want to help you with that. Can you tell me more about what you're experiencing?"
    
    # Get the best match from YOUR training data
    best_match = relevant_docs[0]
    your_content = best_match['content']
    
    print(f"USING YOUR TRAINING CONTENT: {your_content}")
    
    # Use your training content directly with minimal modification
    # Your content already sounds natural and helpful
    
    # Add a brief natural intro if needed
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["sad", "down"]):
        if not your_content.lower().startswith('i'):
            response = f"I hear that you're feeling sad. {your_content}"
        else:
            response = your_content
    elif any(word in user_lower for word in ["anxious", "worried"]):
        if not your_content.lower().startswith('i'):
            response = f"I understand you're feeling anxious. {your_content}"
        else:
            response = your_content
    elif any(word in user_lower for word in ["overwhelmed", "tasks"]):
        if not your_content.lower().startswith('i'):
            response = f"I hear that you're feeling overwhelmed. {your_content}"
        else:
            response = your_content
    else:
        # Use your content directly - it's already well-written
        response = your_content
    
    # Clean up any weird formatting
    response = response.strip()
    if not response.endswith('.') and not response.endswith('!') and not response.endswith('?'):
        response += '.'
    
    print(f"FINAL RESPONSE from your training: {response}")
    return response

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
        print(f"\n=== PROCESSING with your training data: {user_input} ===")
        
        if not user_input:
            return ChatResponse(response="I'm here to listen. What's on your mind?")
        
        if models_loaded and KNOWLEDGE_CORPUS:
            # Use improved search on YOUR content
            relevant_docs = smart_search_your_content(user_input, top_k=3)
            response_text = use_your_training_content(user_input, relevant_docs, request.quiz_summary)
        else:
            response_text = "I'm having trouble accessing my knowledge. Can you tell me more about what's on your mind?"
        
        # Simple suggestions
        suggestions = []
        user_lower = user_input.lower()
        if any(word in user_lower for word in ["sad", "down"]):
            suggestions = ["Practice self-compassion", "Reach out to someone", "Take things one step at a time"]
        elif any(word in user_lower for word in ["anxious", "worried"]):
            suggestions = ["Try deep breathing", "Practice grounding", "Focus on what you can control"]
        
        print(f"=== RETURNING your training content: {response_text[:100]}... ===\n")
        
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
    results = smart_search_your_content(query, top_k=5)
    return {
        "query": query,
        "results_found": len(results),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)