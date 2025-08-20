# mai fastapi - Semantic understanding with sentence embeddings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import numpy as np

# Try to import AI libraries for semantic understanding
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SEMANTIC_AVAILABLE = True
    print("Semantic understanding libraries available")
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Semantic libraries not available - using keyword fallback")

class ChatRequest(BaseModel):
    user_input: str
    quiz_summary: Optional[str] = ""
    subscription_tier: Optional[str] = "free"

class ChatResponse(BaseModel):
    response: str
    suggestions: Optional[List[str]] = []

print("Starting MAI with semantic understanding...")

app = FastAPI(title="MAI AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables
KNOWLEDGE_CORPUS = []
FAISS_INDEX = None
ENCODER_MODEL = None
MODEL_LOADED = False

def load_semantic_model():
    """Load semantic understanding components"""
    global KNOWLEDGE_CORPUS, FAISS_INDEX, ENCODER_MODEL, MODEL_LOADED
    
    try:
        artifacts_path = os.getenv('ARTIFACTS_PATH', './mai_artifacts')
        print(f"Loading semantic model from: {artifacts_path}")
        
        # Load knowledge corpus
        corpus_path = f"{artifacts_path}/knowledge_corpus.pkl"
        if os.path.exists(corpus_path):
            with open(corpus_path, 'rb') as f:
                KNOWLEDGE_CORPUS = pickle.load(f)
            print(f"SUCCESS: Loaded {len(KNOWLEDGE_CORPUS)} knowledge entries")
        else:
            print("ERROR: Knowledge corpus not found")
            return False
        
        if SEMANTIC_AVAILABLE:
            # Load sentence transformer for encoding new queries
            ENCODER_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            print("SUCCESS: Loaded sentence transformer")
            
            # Load FAISS index for fast similarity search
            faiss_path = f"{artifacts_path}/faiss_index.bin"
            if os.path.exists(faiss_path):
                FAISS_INDEX = faiss.read_index(faiss_path)
                print(f"SUCCESS: Loaded FAISS index with {FAISS_INDEX.ntotal} vectors")
            else:
                print("WARNING: FAISS index not found - using keyword fallback")
        
        MODEL_LOADED = True
        return True
        
    except Exception as e:
        print(f"ERROR loading semantic model: {e}")
        return False

def is_physical_health(query):
    """Check if query is about physical health"""
    physical_keywords = [
        'knee', 'back', 'shoulder', 'neck', 'pain', 'hurt', 'ache', 'injury',
        'muscle', 'joint', 'bone', 'headache', 'migraine', 'stomach', 'nausea',
        'fever', 'cold', 'flu', 'sick', 'illness', 'medical', 'doctor'
    ]
    
    query_lower = query.lower()
    physical_count = sum(1 for keyword in physical_keywords if keyword in query_lower)
    
    # If multiple physical health indicators, it's probably not mental health
    if physical_count >= 1:
        print(f"PHYSICAL HEALTH DETECTED: {physical_count} indicators")
        return True
    
    return False

def semantic_search(user_input, top_k=3):
    """Use semantic similarity to find relevant content"""
    if not SEMANTIC_AVAILABLE or not ENCODER_MODEL or not FAISS_INDEX:
        print("SEMANTIC SEARCH NOT AVAILABLE - using keyword fallback")
        return keyword_search(user_input, top_k)
    
    try:
        print(f"SEMANTIC SEARCH for: {user_input}")
        
        # Encode the user's query into the same vector space as training data
        query_embedding = ENCODER_MODEL.encode([user_input])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        # Search for most similar content
        similarities, indices = FAISS_INDEX.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity > 0.4:  # Similarity threshold (0.4 = moderately similar)
                content = KNOWLEDGE_CORPUS[idx]
                results.append({
                    'content': content,
                    'similarity': float(similarity),
                    'index': idx
                })
                print(f"SEMANTIC MATCH {i+1} (similarity: {similarity:.3f}): {content[:80]}...")
        
        print(f"Found {len(results)} semantic matches")
        return results
        
    except Exception as e:
        print(f"SEMANTIC SEARCH ERROR: {e}")
        return keyword_search(user_input, top_k)

def keyword_search(user_input, top_k=3):
    """Fallback keyword search when semantic search isn't available"""
    print(f"KEYWORD SEARCH for: {user_input}")
    
    user_lower = user_input.lower()
    user_words = set(user_lower.split())
    
    # Contextual keyword mapping
    context_keywords = {
        'work_stress': {
            'triggers': ['work', 'job', 'workplace', 'boss', 'career', 'deadline', 'meeting'],
            'related': ['stress', 'anxiety', 'burnout', 'overwhelmed', 'pressure']
        },
        'relationship_issues': {
            'triggers': ['relationship', 'partner', 'boyfriend', 'girlfriend', 'marriage'],
            'related': ['fight', 'argument', 'communication', 'love', 'breakup']
        },
        'anxiety_stress': {
            'triggers': ['anxious', 'anxiety', 'nervous', 'worried', 'panic'],
            'related': ['breathing', 'calm', 'relax', 'ground']
        },
        'depression_sadness': {
            'triggers': ['sad', 'depressed', 'down', 'depression'],
            'related': ['feelings', 'valid', 'motivation', 'energy']
        },
        'overwhelm': {
            'triggers': ['overwhelmed', 'too much', 'busy', 'stressed'],
            'related': ['tasks', 'break', 'pieces', 'small', 'progress']
        }
    }
    
    # Find context matches
    context_scores = {}
    for context, keywords in context_keywords.items():
        score = 0
        # Check for trigger words (higher weight)
        for trigger in keywords['triggers']:
            if trigger in user_lower:
                score += 3
        # Check for related words
        for related in keywords['related']:
            if related in user_lower:
                score += 1
        
        if score > 0:
            context_scores[context] = score
            print(f"CONTEXT MATCH: {context} (score: {score})")
    
    # Find best matching content based on context
    results = []
    if context_scores:
        # Get the top context
        best_context = max(context_scores.keys(), key=lambda x: context_scores[x])
        print(f"BEST CONTEXT: {best_context}")
        
        # Search for content related to this context
        context_terms = context_keywords[best_context]['triggers'] + context_keywords[best_context]['related']
        
        for i, entry in enumerate(KNOWLEDGE_CORPUS):
            if isinstance(entry, str):
                entry_lower = entry.lower()
                entry_words = set(entry_lower.split())
                
                # Calculate contextual relevance
                relevance = 0
                for term in context_terms:
                    if term in entry_lower:
                        relevance += 2 if term in context_keywords[best_context]['triggers'] else 1
                
                if relevance > 0:
                    results.append({
                        'content': entry,
                        'similarity': relevance / 10,  # Normalize to 0-1 scale
                        'index': i
                    })
    
    # Sort by relevance and return top results
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def create_contextual_response(user_input, search_results):
    """Create response with context awareness"""
    if not search_results:
        if is_physical_health(user_input):
            return "I'm specifically trained for mental health support. For physical health concerns, I'd recommend speaking with a healthcare professional. Is there anything about how this is affecting you emotionally that I can help with?"
        else:
            return "I want to help you with that. Can you tell me more about what you're experiencing? I'm here to support your mental wellbeing."
    
    # Use the best match
    best_match = search_results[0]
    content = best_match['content']
    similarity = best_match['similarity']
    
    print(f"USING CONTENT (similarity: {similarity:.3f}): {content}")
    
    # Add contextual introduction based on user's emotional tone
    user_lower = user_input.lower()
    
    # Detect emotional context and respond appropriately
    if any(word in user_lower for word in ['work', 'job', 'workplace']):
        if similarity > 0.6:
            return content  # High confidence, use content directly
        else:
            return f"Work stress can be really challenging. {content}"
    
    elif any(word in user_lower for word in ['sad', 'depressed', 'down']):
        if 'valid' in content.lower() or 'feeling' in content.lower():
            return content
        else:
            return f"I hear that you're feeling down. {content}"
    
    elif any(word in user_lower for word in ['anxious', 'nervous', 'worried']):
        return content
    
    elif any(word in user_lower for word in ['overwhelmed', 'stressed']):
        return content
    
    else:
        # General case
        if similarity > 0.7:
            return content
        else:
            return f"I understand what you're going through. {content}"

# Load the semantic model
models_loaded = load_semantic_model()

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "semantic_available": SEMANTIC_AVAILABLE,
        "knowledge_entries": len(KNOWLEDGE_CORPUS),
        "faiss_index_size": FAISS_INDEX.ntotal if FAISS_INDEX else 0,
        "understanding_method": "semantic" if SEMANTIC_AVAILABLE else "contextual_keywords"
    }

@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        user_input = request.user_input.strip()
        print(f"\nPROCESSING: {user_input}")
        
        if not user_input:
            return ChatResponse(response="I'm here to listen. What's on your mind?")
        
        if not models_loaded:
            return ChatResponse(response="I'm having trouble accessing my knowledge. Can you tell me more about what's on your mind?")
        
        # Check for physical health first
        if is_physical_health(user_input):
            response_text = "I'm specifically trained for mental health support. For physical health concerns, I'd recommend speaking with a healthcare professional. Is there anything about how this is affecting you emotionally that I can help with?"
        else:
            # Use semantic search if available, otherwise contextual keywords
            search_results = semantic_search(user_input, top_k=3)
            response_text = create_contextual_response(user_input, search_results)
        
        # Context-aware suggestions
        suggestions = []
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["anxious", "nervous", "worried"]):
            suggestions = ["Try deep breathing", "Practice grounding", "Take breaks"]
        elif any(word in user_lower for word in ["sad", "depressed", "down"]):
            suggestions = ["Reach out to someone", "Practice self-care", "Be patient with yourself"]
        elif any(word in user_lower for word in ["work", "job", "workplace"]):
            suggestions = ["Set work boundaries", "Take breaks", "Talk to supervisor if needed"]
        elif any(word in user_lower for word in ["overwhelmed", "stressed"]):
            suggestions = ["Break tasks into steps", "Prioritize self-care", "Ask for help"]
        
        print(f"RESPONSE: {response_text[:100]}...")
        
        return ChatResponse(
            response=response_text,
            suggestions=suggestions
        )
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(response="I'm here to help. Can you tell me more about what's on your mind?")

@app.get("/debug/corpus")
async def debug_corpus():
    return {
        "total_entries": len(KNOWLEDGE_CORPUS),
        "sample_entries": KNOWLEDGE_CORPUS[:3] if KNOWLEDGE_CORPUS else [],
        "semantic_available": SEMANTIC_AVAILABLE
    }

@app.get("/debug/search/{query}")
async def debug_search(query: str):
    if is_physical_health(query):
        return {
            "query": query,
            "physical_health_detected": True,
            "results": []
        }
    
    results = semantic_search(query, top_k=5)
    return {
        "query": query,
        "search_method": "semantic" if SEMANTIC_AVAILABLE else "contextual_keywords",
        "results_found": len(results),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)