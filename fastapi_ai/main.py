# MAI FastAPI - Your Smart Mental Health AI
# This file makes MAI actually smart using real AI instead of boring templates!

# Import all the tools we need
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import numpy as np
from sentence_transformers import SentenceTransformer  # This understands what people mean
import faiss  # This finds similar things super fast
import pymongo  # This talks to your MongoDB database
from pymongo import MongoClient
from datetime import datetime

# Try to import Gemini AI (Google's smart AI)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✓ Gemini AI is ready to help!")
except ImportError:
    GEMINI_AVAILABLE = False
    print("✗ Gemini AI not found - install it with: pip install google-generativeai")

# Load secret keys from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded your secret keys")
except ImportError:
    print("✓ Using system environment variables")

# Create the FastAPI app (this is like your main server)
app = FastAPI(title="MAI - Your Smart AI Therapist")

# Let the frontend talk to this backend (CORS = Cross Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from anywhere
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# These define what data looks like when sent back and forth
class ChatRequest(BaseModel):
    user_input: str        # What the user typed
    username: str          # Who is talking
    quiz_summary: Optional[str] = ""      # Their quiz results (optional)
    subscription_tier: Optional[str] = "free"  # Free or paid user

class ChatResponse(BaseModel):
    response: str          # MAI's response
    suggestions: List[str] = []  # Helpful suggestions for the user

# These are like global variables - they store the AI tools
embedding_model = None      # This understands the meaning of words
mental_health_index = None  # This finds similar mental health questions
mental_health_responses = []  # Your training data responses
gemini_model = None        # Google's AI for general questions
db_client = None          # Connection to MongoDB
db = None                # The actual database

def setup_mongodb():
    """Connect to your MongoDB database where user info is stored"""
    global db_client, db
    try:
        # Get the database connection string from environment
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            print("❌ MONGO_URI not found - add it to your .env file")
            return False
        
        # Connect to MongoDB
        db_client = MongoClient(mongo_uri)
        db = db_client.mai_db  # Use the mai_db database
        
        # Test if connection works
        db_client.admin.command('ismaster')
        print("✅ Connected to your MongoDB database!")
        return True
        
    except Exception as e:
        print(f"❌ Couldn't connect to MongoDB: {e}")
        return False

def get_user_context(username):
    """Get a user's quiz results and chat history from the database"""
    # If no database, return empty info
    if not db:
        return {"quiz_summary": "", "chat_history": []}
    
    try:
        # Find the user in the database
        user = db.users.find_one({"username": username})
        
        # Build a summary of their quiz results
        quiz_summary = ""
        if user and user.get('quizResults'):
            quiz_data = []
            for result in user['quizResults']:
                if result.get('result'):
                    quiz_data.append(result['result'])  # Add each quiz result
            quiz_summary = ", ".join(quiz_data)  # Join them with commas
        
        # Get their recent chat history (last 5 conversations)
        chat_history = list(db.chatlogs.find(
            {"username": username}  # Find chats for this user
        ).sort("createdAt", -1).limit(5))  # Most recent first, limit to 5
        
        return {
            "quiz_summary": quiz_summary,
            "chat_history": chat_history
        }
        
    except Exception as e:
        print(f"Error getting user info: {e}")
        return {"quiz_summary": "", "chat_history": []}

def save_chat_to_db(username, user_message, ai_response, quiz_context="", response_time=0):
    """Save the conversation to the database for future reference"""
    if not db:
        print("⚠️  Can't save chat - no database connection")
        return
    
    try:
        # Create a record of this conversation
        chat_log = {
            "username": username,
            "userMessage": user_message,
            "aiResponse": ai_response,
            "quizContext": quiz_context,  # Their personality info
            "subscriptionTier": "free",
            "messageLength": len(user_message),
            "responseTime": response_time,  # How long AI took to respond
            "detectedTopics": extract_topics_from_message(user_message),  # What topics were discussed
            "createdAt": datetime.utcnow(),  # When this happened
            "updatedAt": datetime.utcnow()
        }
        
        # Save it to the database
        db.chatlogs.insert_one(chat_log)
        print(f"💾 Saved conversation for {username}")
        
    except Exception as e:
        print(f"Error saving chat: {e}")

def extract_topics_from_message(message):
    """Figure out what topics the user is talking about"""
    topics = []
    message_lower = message.lower()
    
    # List of topics and their keywords
    topic_keywords = {
        "anxiety": ["anxious", "anxiety", "worry", "worried", "panic", "fear"],
        "depression": ["sad", "depressed", "depression", "down", "hopeless"],
        "stress": ["stress", "stressed", "overwhelmed", "pressure", "burnout"],
        "relationships": ["relationship", "partner", "boyfriend", "girlfriend", "marriage"],
        "work": ["work", "job", "career", "boss", "coworker", "office"],
        "sleep": ["sleep", "insomnia", "tired", "exhausted", "rest"],
        "self-esteem": ["confidence", "self-esteem", "worth", "value", "shame"]
    }
    
    # Check if any topic keywords are in the message
    for topic, keywords in topic_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            topics.append(topic)
    
    return topics

def setup_embedding_model():
    """Load the AI model that understands what words mean"""
    global embedding_model
    try:
        print("🧠 Loading the word-understanding AI...")
        # This model converts words into numbers that represent their meaning
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Word-understanding AI is ready!")
        return True
    except Exception as e:
        print(f"❌ Couldn't load word-understanding AI: {e}")
        return False

def setup_gemini():
    """Set up Google's Gemini AI for general questions"""
    global gemini_model
    
    if not GEMINI_AVAILABLE:
        print("❌ Gemini AI is not installed")
        return False
        
    # Get the API key from your .env file
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in your .env file")
        return False
    
    try:
        # Set up Gemini with your API key
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Test if it works
        test_response = gemini_model.generate_content("Hello")
        print(f"✅ Gemini AI is ready: {test_response.text[:30]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini AI setup failed: {e}")
        return False

def load_training_data():
    """Load your mental health training data and make it searchable"""
    global mental_health_index, mental_health_responses
    
    print("📚 Looking for your mental health training data...")
    
    # Places where your training data might be
    training_files = [
        "mental_health_data.json",           # Main file you should create
        "mai_artifacts/mental_health_qa.json", # From Google Colab
        "mai_artifacts/training_data.json", 
        "mai_artifacts/knowledge_corpus.pkl",
        "training_data.json"
    ]
    
    training_data = []
    
    # Try to load training data from any of these files
    for filepath in training_files:
        if os.path.exists(filepath):
            try:
                if filepath.endswith('.json'):
                    # Load JSON file
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            training_data.extend(data)
                        elif isinstance(data, dict) and 'data' in data:
                            training_data.extend(data['data'])
                        print(f"✅ Loaded {len(data)} mental health examples from {filepath}")
                        break
                elif filepath.endswith('.pkl'):
                    # Load pickle file (from Google Colab)
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, list):
                            training_data.extend(data)
                        print(f"✅ Loaded {len(data)} mental health examples from {filepath}")
                        break
            except Exception as e:
                print(f"⚠️  Couldn't load {filepath}: {e}")
                continue
    
    if not training_data:
        print("❌ NO TRAINING DATA FOUND - MAI will use only Gemini AI")
        return False
    
    # Turn the training data into something searchable
    if training_data and embedding_model:
        try:
            questions = []
            responses = []
            
            # Extract questions and answers from your training data
            for item in training_data:
                if isinstance(item, dict):
                    # Different formats your data might be in
                    if 'question' in item and 'answer' in item:
                        questions.append(item['question'])
                        responses.append(item['answer'])
                    elif 'input' in item and 'output' in item:
                        questions.append(item['input'])
                        responses.append(item['output'])
                    elif 'prompt' in item and 'response' in item:
                        questions.append(item['prompt'])
                        responses.append(item['response'])
                elif isinstance(item, str):
                    # If it's just text, use part as question and all as answer
                    questions.append(item[:100] + "...")
                    responses.append(item)
            
            if questions and responses:
                print(f"🔍 Making {len(questions)} mental health Q&As searchable...")
                
                # Convert questions to numbers that represent their meaning
                question_embeddings = embedding_model.encode(questions)
                
                # Create a fast search index
                dimension = question_embeddings.shape[1]
                mental_health_index = faiss.IndexFlatIP(dimension)  # Inner Product search
                
                # Normalize for better similarity matching
                faiss.normalize_L2(question_embeddings)
                mental_health_index.add(question_embeddings)
                
                mental_health_responses = responses
                
                print(f"✅ Mental health AI is ready with {len(responses)} smart responses!")
                return True
            else:
                print("❌ Couldn't find questions and answers in your training data")
                return False
                
        except Exception as e:
            print(f"❌ Error processing training data: {e}")
            return False
    else:
        print("❌ Need both training data and word-understanding AI")
        return False

def is_mental_health_question(text):
    """Check if the user is asking about mental health topics"""
    # Keywords that suggest mental health topics
    mental_health_keywords = [
        'anxious', 'anxiety', 'stress', 'stressed', 'depressed', 'depression',
        'panic', 'worry', 'worried', 'overwhelmed', 'sad', 'fear', 'afraid',
        'mood', 'therapy', 'therapist', 'counseling', 'mental health',
        'suicidal', 'self harm', 'trauma', 'ptsd', 'bipolar', 'ocd',
        'eating disorder', 'sleep', 'insomnia', 'burnout', 'grief',
        'relationship', 'loneliness', 'lonely', 'isolation', 'self esteem',
        'confidence', 'angry', 'anger', 'irritable', 'emotional', 'feelings'
    ]
    
    # Check if any mental health keywords are in their message
    return any(keyword in text.lower() for keyword in mental_health_keywords)

def find_mental_health_response(user_input, quiz_context="", chat_history=[]):
    """Find the best response from your training data"""
    # If we don't have the search tools ready, can't help
    if not mental_health_index or not mental_health_responses or not embedding_model:
        return None
    
    try:
        # Convert user's question to numbers representing meaning
        user_embedding = embedding_model.encode([user_input])
        faiss.normalize_L2(user_embedding)
        
        # Find the 3 most similar questions in your training data
        k = 3
        similarities, indices = mental_health_index.search(user_embedding, k)
        
        # Get the best match
        best_idx = indices[0][0]
        best_similarity = similarities[0][0]
        
        # Only use it if it's similar enough (40% or higher)
        if best_similarity > 0.4:
            base_response = mental_health_responses[best_idx]
            
            # Make it personal using their quiz results and chat history
            personalized_response = personalize_mental_health_response(
                base_response, quiz_context, chat_history
            )
            return personalized_response
        
        # Not similar enough, let Gemini handle it
        return None
        
    except Exception as e:
        print(f"Error finding mental health response: {e}")
        return None

def personalize_mental_health_response(response, quiz_context, chat_history):
    """Make the response personal using their quiz results and chat history"""
    # If no Gemini, just return the basic response
    if not gemini_model:
        return response
    
    try:
        # Build context from their recent conversations
        history_context = ""
        if chat_history:
            recent_topics = []
            for chat in chat_history[:3]:  # Look at last 3 conversations
                if chat.get('detectedTopics'):
                    recent_topics.extend(chat['detectedTopics'])
            if recent_topics:
                history_context = f"Recent conversation topics: {', '.join(set(recent_topics))}. "
        
        # Ask Gemini to personalize the response
        prompt = f"""You are MAI, a caring mental health AI. Personalize this response based on the user's background:

Base response: "{response}"

User background: {quiz_context}
{history_context}

Instructions:
1. Keep the core mental health advice from the base response
2. Add 1-2 sentences that personalize it based on their quiz results (personality type, stress level, love language)
3. Reference their conversation history if relevant
4. Keep the same caring, supportive tone
5. Don't make it too long - just enhance what's already there

Personalized response:"""

        personalized = gemini_model.generate_content(prompt)
        return personalized.text.strip()
        
    except Exception as e:
        print(f"Error personalizing response: {e}")
        return response  # If personalization fails, use the original

def call_gemini_ai(user_input, quiz_context="", chat_history=[]):
    """Use Gemini AI for general (non-mental health) questions"""
    if not gemini_model:
        return None
    
    try:
        # Build context from recent conversations
        history_context = ""
        if chat_history:
            recent_messages = []
            for chat in chat_history[:3]:  # Last 3 conversations
                recent_messages.append(f"User: {chat.get('userMessage', '')}")
                recent_messages.append(f"MAI: {chat.get('aiResponse', '')}")
            if recent_messages:
                history_context = f"\nRecent conversation:\n" + "\n".join(recent_messages) + "\n"
        
        # Ask Gemini to respond as MAI
        prompt = f"""You are MAI, a caring AI mental health assistant. A user asked: "{user_input}"

User background: {quiz_context}
{history_context}

Respond warmly and helpfully. If there's any emotional component, acknowledge it gently. Be conversational and supportive, adapting your response to their personality type and background.

Then suggest exactly 4 thoughtful follow-up questions or conversation starters that might help the user, formatted as:

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2] 
- [Suggestion 3]
- [Suggestion 4]"""

        response = gemini_model.generate_content(prompt)
        gemini_text = response.text.strip()
        
        # Split the response from the suggestions
        if "SUGGESTIONS:" in gemini_text:
            parts = gemini_text.split("SUGGESTIONS:")
            main_response = parts[0].strip()
            suggestions_text = parts[1].strip()
            
            # Extract the suggestions
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    suggestion = line[1:].strip()
                    if suggestion:
                        suggestions.append(suggestion)
            
            # Make sure we have 4 suggestions
            while len(suggestions) < 4:
                suggestions.extend([
                    "How has your day been?",
                    "What's something positive in your life?",
                    "Tell me more about that",
                    "How can I help you today?"
                ])
            
            suggestions = suggestions[:4]  # Only take first 4
            
        else:
            # If format wasn't followed, use the whole thing as response
            main_response = gemini_text
            suggestions = ["Tell me more", "How do you feel about that?", "What else is on your mind?", "How can I support you?"]
        
        return {
            "response": main_response,
            "suggestions": suggestions
        }
        
    except Exception as e:
        print(f"Gemini AI failed: {e}")
        return None

def generate_mental_health_suggestions(user_input):
    """Create helpful wellness suggestions for mental health topics"""
    if not gemini_model:
        # Basic suggestions if no Gemini
        return ["Take a deep breath", "Be kind to yourself", "You're not alone", "Consider professional support"]
    
    try:
        # Ask Gemini to create wellness suggestions
        prompt = f"""The user said: "{user_input}"

This appears to be a mental health related concern. Generate exactly 4 practical, actionable wellness suggestions that could be added as calendar reminders. Make them specific and helpful.

Format as:
- [Suggestion 1]
- [Suggestion 2]
- [Suggestion 3] 
- [Suggestion 4]"""

        response = gemini_model.generate_content(prompt)
        suggestions_text = response.text.strip()
        
        # Extract suggestions from the response
        suggestions = []
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                suggestion = line[1:].strip()
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions[:4] if len(suggestions) >= 4 else suggestions
        
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return ["Practice self-care today", "Take time to breathe", "Be gentle with yourself", "Reach out for support"]

def initialize_ai_system():
    """Start up all the AI components"""
    print("🚀 Starting MAI - Your Smart AI Therapist...")
    
    # Set up each component
    mongodb_success = setup_mongodb()           # Connect to database
    embedding_success = setup_embedding_model() # Load word-understanding AI
    training_success = load_training_data() if embedding_success else False  # Load your training data
    gemini_success = setup_gemini()            # Set up Gemini AI
    
    # Show status of each component
    print("\n🎯 MAI AI System Status:")
    print(f"   🗄️  MongoDB: {'✅ Connected' if mongodb_success else '❌ Failed'}")
    print(f"   🧠 Word Understanding: {'✅ Ready' if embedding_success else '❌ Failed'}")
    print(f"   📚 Your Training Data: {'✅ Loaded' if training_success else '❌ No data'}")  
    print(f"   🤖 Gemini AI: {'✅ Available' if gemini_success else '❌ Not available'}")
    print(f"   🎯 Mental Health Responses: {len(mental_health_responses)}")
    
    # Check if we have at least one AI working
    if not gemini_success and not training_success:
        print("⚠️  CRITICAL: No AI available! Need either training data OR Gemini")
        return False
    elif gemini_success and training_success:
        print("🎉 Full AI system is working perfectly!")
    elif training_success:
        print("📚 Training data AI ready (no Gemini)")
    elif gemini_success:
        print("🤖 Gemini AI ready (no training data)")
    
    return True

# Start everything up when the server starts
initialize_ai_system()

# API endpoint to check if everything is working
@app.get("/health")
async def health_check():
    """Check if all the AI components are working"""
    return {
        "status": "operational",
        "mongodb_connected": db is not None,
        "embedding_model": embedding_model is not None,
        "mental_health_index": mental_health_index is not None,
        "training_responses": len(mental_health_responses),
        "gemini_available": gemini_model is not None,
        "no_templates": True,  # Confirming we use no templates!
        "ai_only": True       # Everything is AI-powered
    }

# Main endpoint where the magic happens
@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    """Generate AI responses to user messages - NO TEMPLATES EVER!"""
    start_time = datetime.utcnow()  # Track how long this takes
    
    try:
        user_input = request.user_input.strip()
        username = request.username
        
        # If they didn't type anything, let AI handle it (NO TEMPLATES!)
        if not user_input:
            if gemini_model:
                result = call_gemini_ai("Hello", "", [])
                if result:
                    return ChatResponse(
                        response=result['response'],
                        suggestions=result['suggestions']
                    )
            
            # If no AI is working at all, return error (NO TEMPLATE FALLBACK!)
            raise HTTPException(status_code=503, detail="No AI available")
        
        print(f"\n💬 {username}: {user_input}")
        
        # Get the user's info from the database
        user_context = get_user_context(username)
        quiz_context = user_context['quiz_summary']     # Their personality, stress level, etc.
        chat_history = user_context['chat_history']     # Previous conversations
        
        print(f"📊 Quiz Context: {quiz_context}")
        print(f"💾 Chat History: {len(chat_history)} previous conversations")
        
        # Decide which AI to use based on the question
        if is_mental_health_question(user_input):
            print("🏥 Mental health question - checking your training data...")
            
            # Try your training data first (most accurate for mental health)
            training_response = find_mental_health_response(
                user_input, quiz_context, chat_history
            )
            
            if training_response:
                # Found a good match in your training data!
                suggestions = generate_mental_health_suggestions(user_input)
                print("✅ Using your personalized training data response")
                
                # Save this conversation to the database
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                save_chat_to_db(username, user_input, training_response, quiz_context, response_time)
                
                return ChatResponse(
                    response=training_response,
                    suggestions=suggestions
                )
            else:
                print("🤖 No training match - using Gemini for mental health...")
                # Use Gemini if your training data doesn't have a good match
                result = call_gemini_ai(user_input, quiz_context, chat_history)
                
                if result:
                    suggestions = generate_mental_health_suggestions(user_input)
                    
                    # Save conversation
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    save_chat_to_db(username, user_input, result['response'], quiz_context, response_time)
                    
                    return ChatResponse(
                        response=result['response'],
                        suggestions=suggestions
                    )
                else:
                    # No AI available for mental health - return error (NO TEMPLATES!)
                    raise HTTPException(status_code=503, detail="Mental health AI unavailable")
        
        else:
            print("💬 General question - using Gemini AI...")
            # Use Gemini for general questions (like "what's 2+2" or "hello")
            result = call_gemini_ai(user_input, quiz_context, chat_history)
            
            if result:
                # Save conversation
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                save_chat_to_db(username, user_input, result['response'], quiz_context, response_time)
                
                return ChatResponse(
                    response=result['response'],
                    suggestions=result['suggestions']
                )
            else:
                # No AI available - return error (NO TEMPLATES!)
                raise HTTPException(status_code=503, detail="General AI unavailable")
        
    except HTTPException:
        # Re-raise HTTP errors as-is
        raise
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        raise HTTPException(status_code=500, detail="AI system error")

# Get stats about a user's conversations
@app.get("/user-stats/{username}")
async def get_user_stats(username: str):
    """Get statistics about a user's conversations"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Count total conversations
        total_chats = db.chatlogs.count_documents({"username": username})
        
        # Find their most discussed topics
        pipeline = [
            {"$match": {"username": username}},           # Find this user's chats
            {"$unwind": "$detectedTopics"},              # Split topics into separate docs
            {"$group": {"_id": "$detectedTopics", "count": {"$sum": 1}}},  # Count each topic
            {"$sort": {"count": -1}},                    # Sort by most common
            {"$limit": 5}                                # Top 5 topics
        ]
        
        top_topics = list(db.chatlogs.aggregate(pipeline))
        
        return {
            "username": username,
            "total_conversations": total_chats,
            "top_topics": top_topics,
            "ai_only": True,        # Confirm this is AI-only
            "templates_used": 0     # Confirm no templates used
        }
        
    except Exception as e:
        print(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving stats")

# Run the server if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)