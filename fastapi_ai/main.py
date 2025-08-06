from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

app = FastAPI()

# retriever
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

# generator
generator = pipeline("text2text-generation", model="facebook/bart-large")

texts = ["Feeling happy today", "Overwhelmed with stress", "Need some rest"]
index = faiss.IndexFlatL2(384)
embeddings = retriever_model.encode(texts)
index.add(np.array(embeddings))

class ChatRequest(BaseModel):
    user_input:str
    quiz_summary:str

@app.post("/generate")
def generate(req: ChatRequest):
    vec = retriever_model.encode([req.user_input])
    D,I = index.search(np.array(vec),3)
    docs = [texts[i] for i in I[0]]
    prompt = f"Context: {docs}. Traits: {req.quiz_summary}. User: {req.user_input}"
    response = generator(prompt,max_length=120,do_sample=False)
    return {"response":response[0]['generated_text'],"retrieved_docs":docs}
