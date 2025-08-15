# mai fastapi - handles the ai brain stuff
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import numpy as np
from dotenv import load_dotenv
import random

# ai stuff
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import faiss

load_dotenv()

app = FastAPI(title="MAI AI Service", version="1.0.0")

# let frontend talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models get loaded once when server starts up
retriever = None
generator = None
knowledge_corpus = []
search_index = None

class ChatRequest(BaseModel):
    user_input: str
    quiz_summary: Optional[str] = ""

@app.on_event("startup")
async def load_models():
    """load the ai brain when server boots up"""
    global retriever, generator, knowledge_corpus, search_index
    
    print("starting mai...")
    
    artifacts_path = "./mai_artifacts"
    
    if os.path.exists(f"{artifacts_path}/knowledge_corpus.pkl"):
        print("loading trained stuff...")
        
        # grab the knowledge base
        with open(f"{artifacts_path}/knowledge_corpus.pkl", 'rb') as f:
            knowledge_corpus = pickle.load(f)
        
        # load ai models  
        retriever = SentenceTransformer(f"{artifacts_path}/retriever_model")
        
        # simpler generator setup to avoid issues
        try:
            generator = pipeline("text2text-generation", 
                               model=f"{artifacts_path}/generator_model",
                               tokenizer=f"{artifacts_path}/generator_tokenizer",
                               max_length=50,
                               do_sample=True,
                               temperature=0.7)
        except:
            print("using fallback generator...")
            generator = None
        
        # load search index
        search_index = faiss.read_index(f"{artifacts_path}/faiss_index.bin")
        print(f"loaded {len(knowledge_corpus)} knowledge entries")
        
    else:
        print("no trained models, using demo...")
        await setup_demo()

async def setup_demo():
    """backup plan if no trained models"""
    global retriever, generator, knowledge_corpus, search_index
    
    retriever = SentenceTransformer('all-MiniLM-L6-v2')
    generator = None  # we'll use template responses
    
    # basic mental health responses
    knowledge_corpus = [
        "anxiety is tough. try 5-4-3-2-1 grounding: 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste",
        "feeling overwhelmed? break big tasks into tiny steps. progress beats perfection",
        "your feelings are valid. it's ok to not be ok sometimes",
        "stress happens. deep breaths help: in for 4, hold for 4, out for 6",
        "relationships need boundaries. saying no protects your mental health",
        "introverts recharge alone, extroverts with people. both are normal",
        "sleep affects everything - mood, focus, energy. prioritize rest",
        "small wins count. celebrate every step forward, no matter how tiny"
    ]
    
    # make it searchable
    embeddings = retriever.encode(knowledge_corpus)
    search_index = faiss.IndexFlatL2(embeddings.shape[1])
    search_index.add(embeddings.astype('float32'))
    print("demo ready")

def create_smart_response(user_input, quiz_summary, relevant_docs):
    """create good responses using templates and knowledge"""
    
    # extract personality info
    personality_type = ""
    stress_level = ""
    love_language = ""
    
    if quiz_summary:
        parts = quiz_summary.split(", ")
        for part in parts:
            if len(part) == 4 and part.isupper():  # MBTI type
                personality_type = part
            elif "stress" in part.lower():
                stress_level = part
            elif any(word in part.lower() for word in ["time", "words", "touch", "acts", "gifts"]):
                love_language = part
    
    # get relevant knowledge
    context = relevant_docs[0] if relevant_docs else "be kind to yourself"
    
    # create personalized response based on input
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["hello", "hi", "hey"]):
        responses = [
            f"hey there! i'm mai, your mental health buddy. how are you feeling today?",
            f"hi! i'm here to chat and support you. what's on your mind?",
            f"hello! nice to meet you. i'm mai and i care about your wellbeing. how can i help?"
        ]
        
    elif any(word in user_lower for word in ["anxious", "anxiety", "nervous", "worried"]):
        responses = [
            f"anxiety can feel overwhelming. try the 5-4-3-2-1 technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
            f"when anxiety hits, remember to breathe slowly. in for 4 counts, hold for 4, out for 6. you've got this.",
            f"anxiety is tough but temporary. ground yourself by focusing on what's real right now around you."
        ]
        
    elif any(word in user_lower for word in ["stressed", "stress", "overwhelmed", "pressure"]):
        responses = [
            f"feeling stressed is totally normal. try breaking whatever's overwhelming you into smaller, manageable pieces.",
            f"stress can feel heavy. remember that you don't have to carry it all at once - one step at a time is enough.",
            f"when everything feels urgent, pick just one thing to focus on first. progress beats perfection."
        ]
        
    elif any(word in user_lower for word in ["sad", "down", "depressed", "low"]):
        responses = [
            f"it's okay to feel down sometimes. your feelings are valid and this difficult time will pass.",
            f"low energy days are real. even small acts of self-care like drinking water count as wins.",
            f"feeling sad doesn't define you. be gentle with yourself as you work through these emotions."
        ]
        
    elif any(word in user_lower for word in ["don't know", "unsure", "confused", "lost"]):
        responses = [
            f"feeling uncertain is part of being human. you don't need to have all the answers right now.",
            f"it's okay not to know what to do next. sometimes the best step is just being present with yourself.",
            f"confusion can actually be a sign you're growing. take things one moment at a time."
        ]
        
    elif any(word in user_lower for word in ["tired", "exhausted", "drained"]):
        responses = [
            f"exhaustion is your body's way of asking for rest. it's okay to slow down and recharge.",
            f"being tired isn't a weakness - it's information. listen to what your body needs right now.",
            f"rest isn't earned, it's needed. give yourself permission to take breaks without guilt."
        ]
        
    else:
        # general supportive responses
        responses = [
            f"i hear you. whatever you're going through, your feelings matter and you're not alone in this.",
            f"thank you for sharing with me. it takes courage to talk about what's bothering you.",
            f"you're doing better than you think. be patient with yourself as you navigate whatever you're facing."
        ]
    
    # add personality-specific touch
    base_response = random.choice(responses)
    
    if personality_type:
        if "I" in personality_type:  # introvert
            base_response += " take some quiet time for yourself if you need it."
        elif "E" in personality_type:  # extrovert
            base_response += " talking with someone you trust might help too."
            
    if stress_level and "high" in stress_level.lower():
        base_response += " given your stress levels, be extra gentle with yourself right now."
        
    return base_response

@app.post("/generate")
async def generate_response(request: ChatRequest):
    """make a helpful response using ai"""
    
    try:
        # find similar knowledge entries
        query_embedding = retriever.encode([request.user_input])
        distances, indices = search_index.search(query_embedding.astype('float32'), k=3)
        
        # grab relevant stuff
        relevant_docs = [knowledge_corpus[i] for i in indices[0]]
        
        # use our smart template system instead of buggy generator
        response_text = create_smart_response(request.user_input, request.quiz_summary, relevant_docs)
        
        return {
            "response": response_text,
            "retrieved_docs": relevant_docs[:2]  # send back what knowledge we used
        }
        
    except Exception as e:
        print(f"oops: {e}")
        # fallback if ai breaks
        return {
            "response": "i'm here to listen. can you tell me more about what's bothering you?",
            "retrieved_docs": ["fallback"]
        }

@app.get("/health")
async def health_check():
    """check if everything is working"""
    return {
        "status": "good to go",
        "models_loaded": retriever is not None,
        "knowledge_count": len(knowledge_corpus)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)