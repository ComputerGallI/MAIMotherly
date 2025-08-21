# mai fastapi - Reliable system using your actual training data
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import re
import random

app = FastAPI(title="MAI Mental Health AI - Using Real Training Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    quiz_summary: Optional[str] = ""
    subscription_tier: Optional[str] = "free"

class ChatResponse(BaseModel):
    response: str
    suggestions: List[str] = []

# Global variables
knowledge_corpus = []
config = {}

# Enhanced keyword mapping that will find your training content
KEYWORD_TO_CONTENT = {
    # Work/Career stress
    'work': ['work', 'job', 'career', 'workplace', 'office', 'boss', 'colleague', 'professional', 'deadline', 'meeting', 'project'],
    'stress': ['stress', 'stressed', 'pressure', 'overwhelmed', 'busy', 'burnout', 'exhausted'],
    
    # Anxiety and worry
    'anxiety': ['anxious', 'anxiety', 'nervous', 'worried', 'panic', 'fear', 'scared'],
    'worry': ['worry', 'worrying', 'concern', 'concerned', 'afraid'],
    
    # Sadness and depression
    'sad': ['sad', 'sadness', 'down', 'low', 'blue', 'depressed', 'depression'],
    'hurt': ['hurt', 'pain', 'upset', 'disappointed', 'heartbroken'],
    
    # Relationships
    'relationship': ['relationship', 'partner', 'boyfriend', 'girlfriend', 'husband', 'wife', 'dating'],
    'fight': ['fight', 'argument', 'conflict', 'disagree', 'angry', 'mad'],
    'family': ['family', 'parent', 'mother', 'father', 'sibling', 'child'],
    'friend': ['friend', 'friendship', 'social', 'lonely', 'alone'],
    
    # Self-esteem and confidence
    'confidence': ['confidence', 'self-esteem', 'self-worth', 'insecure', 'doubt'],
    'failure': ['failure', 'failed', 'not good enough', 'inadequate', 'worthless'],
    
    # Sleep and energy
    'sleep': ['sleep', 'sleeping', 'insomnia', 'tired', 'exhausted', 'rest'],
    'energy': ['energy', 'motivation', 'drive', 'focus', 'lazy', 'unmotivated'],
    
    # General emotional support
    'feeling': ['feeling', 'feel', 'emotion', 'emotional', 'mood'],
    'help': ['help', 'support', 'advice', 'guidance', 'assistance']
}

def load_models():
    """Load the trained knowledge base"""
    global knowledge_corpus, config
    
    try:
        print("Starting MAI - Using your real training data...")
        artifacts_dir = "./mai_artifacts"
        print(f"Looking for trained models in: {artifacts_dir}")
        
        # Load knowledge corpus
        with open(os.path.join(artifacts_dir, "knowledge_corpus.pkl"), "rb") as f:
            knowledge_corpus = pickle.load(f)
        print(f"SUCCESS: Loaded {len(knowledge_corpus)} real mental health knowledge entries")
        
        # Show sample of your training data
        if knowledge_corpus:
            print("Sample of your training data:")
            for i, entry in enumerate(knowledge_corpus[:3]):
                print(f"  Entry {i+1}: {entry[:60]}...")
        
        # Load config
        try:
            with open(os.path.join(artifacts_dir, "config.json"), "r") as f:
                config = json.load(f)
            print(f"SUCCESS: Loaded training config")
        except:
            print("Config file not found, continuing without it")
        
        print("AI system ready to use YOUR training data!")
        
    except Exception as e:
        print(f"ERROR loading models: {e}")
        raise

def find_best_training_content(user_input: str) -> str:
    """Find the best match from your actual training data"""
    if not knowledge_corpus:
        print("ERROR: No training data loaded")
        return None
        
    user_lower = user_input.lower()
    user_words = set(user_lower.split())
    
    print(f"SEARCHING your {len(knowledge_corpus)} training entries for: '{user_input}'")
    
    # Find matches by checking how many keywords overlap
    matches = []
    
    for i, entry in enumerate(knowledge_corpus):
        if isinstance(entry, str):
            entry_lower = entry.lower()
            entry_words = set(entry_lower.split())
            
            # Count word overlaps
            overlap_count = len(user_words.intersection(entry_words))
            
            # Also check for keyword matches
            keyword_matches = 0
            for keyword_group in KEYWORD_TO_CONTENT.values():
                for keyword in keyword_group:
                    if keyword in user_lower and keyword in entry_lower:
                        keyword_matches += 2
            
            total_score = overlap_count + keyword_matches
            
            if total_score > 0:
                matches.append({
                    'entry': entry,
                    'score': total_score,
                    'index': i
                })
    
    # Sort by score and return best match
    if matches:
        matches.sort(key=lambda x: x['score'], reverse=True)
        best_match = matches[0]
        
        print(f"BEST MATCH (score {best_match['score']}): {best_match['entry'][:80]}...")
        print(f"Found {len(matches)} total matches in your training data")
        
        return best_match['entry']
    
    print("NO MATCHES found in your training data")
    return None

def is_physical_health_question(user_input: str) -> bool:
    """Check if this is about physical health (outside mental health scope)"""
    physical_terms = ['knee', 'back', 'shoulder', 'pain', 'hurt', 'ache', 'headache', 'sick', 'fever', 'cold', 'injury', 'medical', 'doctor visit']
    user_lower = user_input.lower()
    
    physical_count = sum(1 for term in physical_terms if term in user_lower)
    
    if physical_count >= 1:
        print(f"PHYSICAL HEALTH DETECTED: {physical_count} indicators")
        return True
    return False

def generate_response(user_input: str, quiz_summary: str = "") -> dict:
    """Generate response using your actual training data"""
    print(f"\n=== PROCESSING: {user_input} ===")
    
    # Check for physical health first
    if is_physical_health_question(user_input):
        return {
            "response": "I'm specifically trained for mental health support. For physical health concerns, I'd recommend speaking with a healthcare professional. Is there anything about how this situation is affecting you emotionally that I can help with?",
            "suggestions": ["How are you feeling about this?", "Any emotional impact?", "Want to talk about stress?"]
        }
    
    # Find the best content from your training data
    best_content = find_best_training_content(user_input)
    
    if best_content:
        # Use your training content directly
        response = best_content
        
        # Add context-appropriate suggestions that work as calendar reminders
        user_lower = user_input.lower()
        suggestions = []
        
        if any(word in user_lower for word in ['work', 'job', 'workplace', 'boss', 'colleague']):
            suggestions = [
                'Take a 10-minute break every hour',
                'Practice deep breathing before meetings', 
                'Set work-life boundaries',
                'Schedule time for lunch away from desk'
            ]
        elif any(word in user_lower for word in ['anxious', 'nervous', 'worried', 'panic']):
            suggestions = [
                'Practice 4-7-8 breathing technique',
                'Do 5-minute grounding exercise',
                'Take a short walk outside',
                'Listen to calming music'
            ]
        elif any(word in user_lower for word in ['sad', 'down', 'depressed', 'low']):
            suggestions = [
                'Call a friend or family member',
                'Do one small self-care activity',
                'Write in a gratitude journal',
                'Get some sunlight or fresh air'
            ]
        elif any(word in user_lower for word in ['relationship', 'partner', 'friend', 'family']):
            suggestions = [
                'Practice active listening',
                'Express appreciation to someone',
                'Set healthy communication boundaries',
                'Schedule quality time together'
            ]
        elif any(word in user_lower for word in ['stress', 'overwhelmed', 'pressure']):
            suggestions = [
                'Break large tasks into smaller steps',
                'Practice 5-minute meditation',
                'Take three deep breaths',
                'Prioritize your top 3 tasks'
            ]
        elif any(word in user_lower for word in ['sleep', 'tired', 'exhausted']):
            suggestions = [
                'Set a consistent bedtime routine',
                'Avoid screens 1 hour before bed',
                'Try progressive muscle relaxation',
                'Keep bedroom cool and dark'
            ]
        else:
            suggestions = [
                'Take 5 deep breaths mindfully',
                'Do something kind for yourself',
                'Check in with your feelings',
                'Practice one minute of gratitude'
            ]
        
        print(f"USING YOUR TRAINING DATA: {response[:100]}...")
        
        return {
            "response": response,
            "suggestions": suggestions
        }
    
    # Fallback when no good match found
    fallback_response = "I want to help you with that. Can you tell me more about what you're experiencing? I'm here to support your mental health and wellbeing."
    
    # Provide general wellness suggestions even for fallback
    fallback_suggestions = [
        'Take three deep mindful breaths',
        'Do a quick body scan for tension',
        'Practice self-compassion',
        'Take a moment to acknowledge your feelings'
    ]
    
    print("USING FALLBACK: No good match found in training data")
    
    return {
        "response": fallback_response,
        "suggestions": fallback_suggestions
    }

# Load models on startup
load_models()

@app.get("/health")
async def health_check():
    return {
        "status": "operational - using your real training data",
        "knowledge_entries": len(knowledge_corpus),
        "system": "Direct content matching from your training",
        "ready": len(knowledge_corpus) > 0
    }

@app.post("/generate", response_model=ChatResponse)
async def generate_response_endpoint(request: ChatRequest):
    try:
        user_input = request.user_input.strip()
        print(f"\nRECEIVED REQUEST: {user_input}")
        
        if not user_input:
            return ChatResponse(response="I'm here to listen. What's on your mind?")
        
        if not knowledge_corpus:
            return ChatResponse(response="I'm having trouble accessing my knowledge. Let me try to help anyway - what's going on?")
        
        # Generate response using your training data
        result = generate_response(user_input, request.quiz_summary)
        
        print(f"SENDING RESPONSE: {result['response'][:100]}...")
        
        return ChatResponse(
            response=result['response'],
            suggestions=result.get('suggestions', [])
        )
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(response="I'm here to help. Can you tell me more about what's on your mind?")

@app.get("/debug/corpus")
async def debug_corpus():
    return {
        "total_entries": len(knowledge_corpus),
        "sample_entries": knowledge_corpus[:5] if knowledge_corpus else [],
        "entry_types": [type(entry).__name__ for entry in knowledge_corpus[:5]]
    }

@app.get("/debug/search/{query}")
async def debug_search(query: str):
    if is_physical_health_question(query):
        return {
            "query": query,
            "physical_health_detected": True,
            "results": "Redirected to physical health response"
        }
    
    best_content = find_best_training_content(query)
    return {
        "query": query,
        "best_match": best_content[:200] + "..." if best_content else None,
        "total_training_entries": len(knowledge_corpus),
        "match_found": best_content is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)