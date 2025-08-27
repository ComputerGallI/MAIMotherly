# mai fastapi - Simple and reliable Gemini integration
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import re
import random

# Import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✓ Gemini AI available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("✗ Gemini AI not installed - run: pip install google-generativeai")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("✓ No .env file loader (install with: pip install python-dotenv)")

app = FastAPI(title="MAI Mental Health AI - Your training data + Gemini fallback")

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

def setup_gemini():
    """Set up Gemini AI model"""
    global gemini_model
    
    if not GEMINI_AVAILABLE:
        print("❌ Gemini not available - google-generativeai not installed")
        return False
        
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        print("   Create a .env file with: GEMINI_API_KEY=your_key_here")
        print("   Or set environment variable: set GEMINI_API_KEY=your_key_here")
        return False
    
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-pro')
        print(f"✅ Gemini initialized successfully (key: {api_key[:8]}...)")
        
        # Test Gemini with a simple call
        test_response = gemini_model.generate_content("Say hello")
        print(f"✅ Gemini test successful: {test_response.text[:30]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini setup failed: {e}")
        return False

def call_gemini(user_input: str, quiz_summary: str = "") -> dict:
    """Call Gemini AI for response"""
    if not gemini_model:
        print("⚠️  Gemini not available - using fallback")
        return {
            "response": "I'd like to help you with that. Can you tell me more about what you're experiencing?",
            "suggestions": [
                "Take a moment to breathe deeply",
                "Notice your current feelings",
                "Practice self-compassion",
                "Do something kind for yourself"
            ]
        }
    
    try:
        # Create mental health-focused prompt
        prompt = f"""You are MAI, a caring mental health AI assistant. Be warm and supportive.

User background: {quiz_summary or 'No background info available'}
User question: "{user_input}"

Respond with:
1. A caring, supportive response (1-3 sentences)
2. Then list exactly 4 practical wellness suggestions that could be calendar reminders

Format like this:
[Your caring response here]

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2] 
- [Suggestion 3]
- [Suggestion 4]

Keep it warm, practical, and mental health focused."""

        print(f"🤖 Calling Gemini for: {user_input[:30]}...")
        
        response = gemini_model.generate_content(prompt)
        gemini_text = response.text.strip()
        
        print(f"✅ Gemini responded: {gemini_text[:50]}...")
        
        # Split response and suggestions
        if "SUGGESTIONS:" in gemini_text:
            parts = gemini_text.split("SUGGESTIONS:")
            main_response = parts[0].strip()
            suggestions_text = parts[1].strip()
            
            # Extract suggestions
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    suggestion = line[1:].strip()
                    if suggestion and len(suggestion) > 3:
                        suggestions.append(suggestion)
            
            # Ensure we have 4 suggestions
            while len(suggestions) < 4:
                suggestions.extend([
                    "Take three deep breaths mindfully",
                    "Check in with your emotions",
                    "Practice gratitude for one minute",
                    "Do something kind for yourself"
                ])
            
            suggestions = suggestions[:4]  # Limit to 4
            
        else:
            # If format wasn't followed, use the whole response
            main_response = gemini_text
            suggestions = [
                "Take three deep breaths",
                "Practice mindful awareness", 
                "Show yourself compassion",
                "Take care of your needs"
            ]
        
        return {
            "response": main_response,
            "suggestions": suggestions
        }
        
    except Exception as e:
        print(f"❌ Gemini call failed: {e}")
        return {
            "response": "I want to support you through this. Can you share more about how you're feeling?",
            "suggestions": [
                "Take slow, deep breaths",
                "Ground yourself in the present",
                "Be gentle with yourself", 
                "Remember that you matter"
            ]
        }

def find_training_data_match(user_input: str) -> str:
    """Find match in training data"""
    if not knowledge_corpus:
        return None
        
    user_lower = user_input.lower()
    
    # Simple keyword matching
    for entry in knowledge_corpus:
        if isinstance(entry, str):
            entry_lower = entry.lower()
            
            # Count matching words
            user_words = set(user_lower.split())
            entry_words = set(entry_lower.split())
            matches = len(user_words.intersection(entry_words))
            
            # If we have good overlap, use it
            if matches >= 2:
                print(f"📚 Using training data (match score: {matches})")
                return entry
    
    print("📚 No good training data match found")
    return None

def generate_suggestions_for_topic(user_input: str) -> List[str]:
    """Generate topic-appropriate suggestions"""
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ['work', 'job', 'office', 'boss']):
        return [
            'Take a 10-minute break',
            'Practice desk breathing exercises',
            'Set a boundary with work time',
            'Connect with a supportive colleague'
        ]
    elif any(word in user_lower for word in ['anxious', 'anxiety', 'worried', 'nervous']):
        return [
            'Try 4-7-8 breathing technique',
            'Do a 5-minute grounding exercise',
            'Take a short mindful walk',
            'Listen to calming music'
        ]
    elif any(word in user_lower for word in ['sad', 'depressed', 'down', 'low']):
        return [
            'Reach out to someone who cares',
            'Do one small self-care activity',
            'Write three things you\'re grateful for',
            'Get some sunlight or fresh air'
        ]
    elif any(word in user_lower for word in ['stress', 'overwhelmed', 'pressure']):
        return [
            'Break tasks into smaller steps',
            'Practice 5-minute meditation',
            'Take three conscious breaths',
            'Prioritize your most important tasks'
        ]
    else:
        return [
            'Take a moment for deep breathing',
            'Check in with your emotions',
            'Practice self-compassion',
            'Do something nurturing for yourself'
        ]

def load_models():
    """Load training data and set up Gemini"""
    global knowledge_corpus, config
    
    print("🚀 Starting MAI - Your training data + Gemini fallback...")
    
    # Load training data
    try:
        artifacts_dir = "./mai_artifacts"
        print(f"📁 Looking for trained models in: {artifacts_dir}")
        
        with open(os.path.join(artifacts_dir, "knowledge_corpus.pkl"), "rb") as f:
            knowledge_corpus = pickle.load(f)
        print(f"✅ Loaded {len(knowledge_corpus)} mental health knowledge entries")
        
        if knowledge_corpus:
            print("📖 Sample training data:")
            for i, entry in enumerate(knowledge_corpus[:2]):
                print(f"   {i+1}. {entry[:50]}...")
        
        try:
            with open(os.path.join(artifacts_dir, "config.json"), "r") as f:
                config = json.load(f)
            print("✅ Loaded training config")
        except:
            print("⚠️  No config file found")
            
    except Exception as e:
        print(f"❌ Could not load training data: {e}")
        knowledge_corpus = []
    
    # Set up Gemini
    gemini_success = setup_gemini()
    
    print("\n🎯 AI System Ready!")
    print(f"   📚 Training entries: {len(knowledge_corpus)}")
    print(f"   🤖 Gemini fallback: {'✅ Available' if gemini_success else '❌ Not available'}")
    print(f"   🔄 Strategy: Training data first, then Gemini")

# Initialize on startup
load_models()

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "training_entries": len(knowledge_corpus),
        "gemini_available": gemini_model is not None,
        "gemini_library_installed": GEMINI_AVAILABLE,
        "api_key_found": bool(os.getenv('GEMINI_API_KEY')),
        "strategy": "Training data first, Gemini fallback"
    }

@app.post("/generate", response_model=ChatResponse)
async def generate_response_endpoint(request: ChatRequest):
    try:
        user_input = request.user_input.strip()
        print(f"\n💬 REQUEST: {user_input}")
        
        if not user_input:
            return ChatResponse(response="I'm here to listen. What's on your mind?")
        
        # Check if it's a physical health question
        physical_terms = ['knee', 'back', 'shoulder', 'hurt', 'pain', 'sick', 'fever']
        if any(term in user_input.lower() for term in physical_terms):
            print("🏥 Physical health question detected")
            return ChatResponse(
                response="I focus on mental health support. For physical health concerns, I'd recommend speaking with a healthcare professional. Is there anything about how this situation is affecting you emotionally that I can help with?",
                suggestions=["How is this affecting your mood?", "Any stress about the situation?", "Want to talk about your feelings?", "Need emotional support?"]
            )
        
        # First, try training data
        training_match = find_training_data_match(user_input)
        
        if training_match:
            # Use training data + topic-based suggestions
            suggestions = generate_suggestions_for_topic(user_input)
            print(f"✅ Response: Training data + topic suggestions")
            return ChatResponse(
                response=training_match,
                suggestions=suggestions
            )
        
        # No training data match - use Gemini
        print("🤖 No training match - using Gemini...")
        result = call_gemini(user_input, request.quiz_summary)
        
        print(f"✅ Response: Gemini AI")
        return ChatResponse(
            response=result['response'],
            suggestions=result['suggestions']
        )
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return ChatResponse(
            response="I'm here to support you. Can you tell me more about what you're experiencing?",
            suggestions=["Take a deep breath", "Ground yourself", "Be kind to yourself", "You're not alone"]
        )

@app.get("/test-gemini/{query}")
async def test_gemini_directly(query: str):
    """Test Gemini directly"""
    result = call_gemini(query)
    return {
        "query": query,
        "gemini_available": gemini_model is not None,
        "response": result['response'],
        "suggestions": result['suggestions']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)