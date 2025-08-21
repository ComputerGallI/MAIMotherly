# mai fastapi - Your training data + Gemini LLM fallback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import re
import random

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("Gemini AI available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini AI not available - install google-generativeai package")

app = FastAPI(title="MAI Mental Health AI - Your Training + Gemini Fallback")

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
gemini_model = None

# Enhanced keyword mapping for your training content
KEYWORD_TO_CONTENT = {
    # Mental health topics that your model handles
    'work': ['work', 'job', 'career', 'workplace', 'office', 'boss', 'colleague', 'professional', 'deadline', 'meeting', 'project'],
    'stress': ['stress', 'stressed', 'pressure', 'overwhelmed', 'busy', 'burnout', 'exhausted'],
    'anxiety': ['anxious', 'anxiety', 'nervous', 'worried', 'panic', 'fear', 'scared'],
    'worry': ['worry', 'worrying', 'concern', 'concerned', 'afraid'],
    'sad': ['sad', 'sadness', 'down', 'low', 'blue', 'depressed', 'depression'],
    'hurt': ['hurt', 'pain', 'upset', 'disappointed', 'heartbroken'],
    'relationship': ['relationship', 'partner', 'boyfriend', 'girlfriend', 'husband', 'wife', 'dating'],
    'fight': ['fight', 'argument', 'conflict', 'disagree', 'angry', 'mad'],
    'family': ['family', 'parent', 'mother', 'father', 'sibling', 'child'],
    'friend': ['friend', 'friendship', 'social', 'lonely', 'alone'],
    'confidence': ['confidence', 'self-esteem', 'self-worth', 'insecure', 'doubt'],
    'failure': ['failure', 'failed', 'not good enough', 'inadequate', 'worthless'],
    'sleep': ['sleep', 'sleeping', 'insomnia', 'tired', 'exhausted', 'rest'],
    'energy': ['energy', 'motivation', 'drive', 'focus', 'lazy', 'unmotivated'],
    'feeling': ['feeling', 'feel', 'emotion', 'emotional', 'mood'],
    'help': ['help', 'support', 'advice', 'guidance', 'assistance']
}

def setup_gemini():
    """Initialize Gemini AI model"""
    global gemini_model
    
    if not GEMINI_AVAILABLE:
        print("Gemini not available - skipping setup")
        return False
    
    try:
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("GEMINI_API_KEY not found in environment")
            return False
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model with system instructions for MAI
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 300,
        }
        
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction="""You are MAI, a caring mental health assistant. When answering questions outside of mental health topics, be helpful and informative while maintaining a warm, supportive tone. Keep responses concise (under 250 words) and always try to gently connect back to wellbeing when appropriate. If someone asks about serious medical issues, recommend they consult healthcare professionals."""
        )
        
        print("SUCCESS: Gemini AI model initialized")
        return True
        
    except Exception as e:
        print(f"ERROR setting up Gemini: {e}")
        return False

def load_models():
    """Load the trained knowledge base and Gemini"""
    global knowledge_corpus, config
    
    try:
        print("Starting MAI - Your training data + Gemini fallback...")
        artifacts_dir = "./mai_artifacts"
        print(f"Looking for trained models in: {artifacts_dir}")
        
        # Load knowledge corpus
        with open(os.path.join(artifacts_dir, "knowledge_corpus.pkl"), "rb") as f:
            knowledge_corpus = pickle.load(f)
        print(f"SUCCESS: Loaded {len(knowledge_corpus)} mental health knowledge entries")
        
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
        
        # Setup Gemini
        gemini_setup = setup_gemini()
        if gemini_setup:
            print("SUCCESS: Gemini AI ready for fallback responses")
        else:
            print("WARNING: Gemini AI not available - using basic fallbacks")
        
        print("AI system ready - Your training data FIRST, Gemini for everything else!")
        
    except Exception as e:
        print(f"ERROR loading models: {e}")
        raise

def is_mental_health_related(user_input: str) -> bool:
    """Check if the question is related to mental health topics that your training covers"""
    user_lower = user_input.lower()
    
    # Count mental health keyword matches
    mental_health_matches = 0
    for keyword_group in KEYWORD_TO_CONTENT.values():
        for keyword in keyword_group:
            if keyword in user_lower:
                mental_health_matches += 1
    
    # If we have mental health indicators, it's probably in scope
    if mental_health_matches >= 1:
        print(f"MENTAL HEALTH TOPIC DETECTED: {mental_health_matches} indicators")
        return True
    
    # Check for emotional language
    emotional_words = ['feel', 'feeling', 'emotions', 'emotional', 'mental', 'psychology', 'therapy', 'counseling', 'wellness']
    emotional_matches = sum(1 for word in emotional_words if word in user_lower)
    
    if emotional_matches >= 1:
        print(f"EMOTIONAL TOPIC DETECTED: {emotional_matches} indicators")
        return True
    
    print(f"NON-MENTAL-HEALTH TOPIC: Will use Gemini fallback")
    return False

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

def ask_gemini(user_input: str, quiz_summary: str = "") -> dict:
    """Ask Gemini AI for responses to questions outside your training scope"""
    if not GEMINI_AVAILABLE or not gemini_model:
        return {
            "response": "I'm specifically trained for mental health support. For questions outside that scope, I'd recommend searching online or consulting relevant professionals. Is there anything about your mental health or wellbeing I can help with?",
            "suggestions": ["Tell me how you're feeling", "Any stress or anxiety?", "Want to talk about relationships?"]
        }
    
    try:
        print(f"ASKING GEMINI: {user_input}")
        
        # Create context for Gemini
        context = f"User question: {user_input}"
        if quiz_summary:
            context += f"\nUser personality context: {quiz_summary}"
        
        # Get response from Gemini
        response = gemini_model.generate_content(context)
        
        if response.text:
            gemini_response = response.text.strip()
            print(f"GEMINI RESPONSE: {gemini_response[:100]}...")
            
            # Generate suggestions based on the topic
            suggestions = [
                "Is there anything else I can help with?",
                "How are you feeling about this?",
                "Want to talk about anything else?"
            ]
            
            # Try to detect topic for better suggestions
            user_lower = user_input.lower()
            if any(word in user_lower for word in ['learn', 'study', 'school', 'education']):
                suggestions = ["Need study tips?", "Feeling stressed about learning?", "Want to talk about school pressure?"]
            elif any(word in user_lower for word in ['weather', 'temperature', 'rain', 'sun']):
                suggestions = ["How does weather affect your mood?", "Feeling seasonal changes?", "Want tips for weather-related mood?"]
            elif any(word in user_lower for word in ['food', 'recipe', 'cooking', 'eating']):
                suggestions = ["How's your relationship with food?", "Feeling stressed about eating?", "Want to talk about nutrition and mood?"]
            
            return {
                "response": gemini_response,
                "suggestions": suggestions
            }
        else:
            raise Exception("Empty response from Gemini")
            
    except Exception as e:
        print(f"GEMINI ERROR: {e}")
        return {
            "response": "I'm having trouble accessing additional information right now. Is there anything about your mental health, stress, relationships, or wellbeing I can help you with instead?",
            "suggestions": ["Tell me about your day", "Any stress or worries?", "Want to talk about feelings?"]
        }

def is_physical_health_question(user_input: str) -> bool:
    """Check if this is about physical health (outside mental health scope)"""
    physical_terms = ['knee', 'back', 'shoulder', 'pain', 'hurt', 'ache', 'headache', 'sick', 'fever', 'cold', 'injury', 'medical', 'doctor visit', 'surgery', 'medication', 'pills', 'prescription']
    user_lower = user_input.lower()
    
    physical_count = sum(1 for term in physical_terms if term in user_lower)
    
    if physical_count >= 1:
        print(f"PHYSICAL HEALTH DETECTED: {physical_count} indicators")
        return True
    return False

def generate_response(user_input: str, quiz_summary: str = "") -> dict:
    """Generate response using your training data first, then Gemini fallback"""
    print(f"\n=== PROCESSING: {user_input} ===")
    
    # Check for physical health first
    if is_physical_health_question(user_input):
        return {
            "response": "I'm specifically trained for mental health support. For physical health concerns, I'd recommend speaking with a healthcare professional. Is there anything about how this situation is affecting you emotionally that I can help with?",
            "suggestions": ["How are you feeling about this?", "Any emotional impact?", "Want to talk about stress?"]
        }
    
    # Check if it's a mental health topic that your training should handle
    if is_mental_health_related(user_input):
        print("MENTAL HEALTH TOPIC: Searching your training data...")
        
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
        else:
            print("NO MATCH in training data for mental health topic - using Gemini...")
            return ask_gemini(user_input, quiz_summary)
    
    else:
        # Not a mental health topic - use Gemini for general questions
        print("GENERAL TOPIC: Using Gemini...")
        return ask_gemini(user_input, quiz_summary)

# Load models on startup
load_models()

@app.get("/health")
async def health_check():
    return {
        "status": "operational - your training data + Gemini fallback",
        "knowledge_entries": len(knowledge_corpus),
        "gemini_available": GEMINI_AVAILABLE,
        "system": "Mental health training FIRST, Gemini for everything else",
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
        
        # Generate response using your training data first, Gemini fallback
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
        "entry_types": [type(entry).__name__ for entry in knowledge_corpus[:5]],
        "gemini_available": GEMINI_AVAILABLE
    }

@app.get("/debug/search/{query}")
async def debug_search(query: str):
    if is_physical_health_question(query):
        return {
            "query": query,
            "physical_health_detected": True,
            "results": "Redirected to physical health response"
        }
    
    is_mental_health = is_mental_health_related(query)
    best_content = None
    
    if is_mental_health:
        best_content = find_best_training_content(query)
    
    return {
        "query": query,
        "is_mental_health_topic": is_mental_health,
        "best_match": best_content[:200] + "..." if best_content else None,
        "total_training_entries": len(knowledge_corpus),
        "will_use_gemini": not is_mental_health or (is_mental_health and not best_content),
        "gemini_available": GEMINI_AVAILABLE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)