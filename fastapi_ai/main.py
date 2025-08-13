# mai fastapi - handles the ai brain stuff
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, pickle
import numpy as np
from dotenv import load_dotenv

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
        generator = pipeline("text2text-generation", 
                           model=f"{artifacts_path}/generator_model",
                           tokenizer=f"{artifacts_path}/generator_tokenizer")
        
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
    generator = pipeline("text2text-generation", model="facebook/bart-base")
    
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

@app.post("/generate")
async def generate_response(request: ChatRequest):
    """make a helpful response using ai"""
    
    try:
        # find similar knowledge entries
        query_embedding = retriever.encode([request.user_input])
        distances, indices = search_index.search(query_embedding.astype('float32'), k=3)
        
        # grab relevant stuff
        relevant_docs = [knowledge_corpus[i] for i in indices[0]]
        context = " ".join(relevant_docs)
        
        # build prompt with personality info
        personality_hint = f"User info: {request.quiz_summary}. " if request.quiz_summary else ""
        
        prompt = f"""{personality_hint}You are MAI, a caring mental health buddy.

Context: {context}

User: {request.user_input}

Respond with empathy and practical tips:"""
        
        # generate the response
        result = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
        response_text = result[0]['generated_text']
        
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
