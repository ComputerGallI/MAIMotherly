from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="facebook/bart-large")

texts = ["feeling anxious","i am happy today","work stress is high"]
index = faiss.IndexFlatL2(384)
embeddings = retriever_model.encode(texts)
index.add(np.array(embeddings))

class ChatRequest(BaseModel):
    user_input:str
    quiz_summary:str

@app.post("/generate")
def generate_response(req: ChatRequest):
    vec = retriever_model.encode([req.user_input])
    D,I = index.search(np.array(vec),3)
    docs = [texts[i] for i in I[0]]
    context = " ".join(docs)
    prompt = f"Context: {context}. Traits: {req.quiz_summary}. Question: {req.user_input}"
    response = generator(prompt,max_length=120,do_sample=False)
    return {"response":response[0]['generated_text'],"retrieved_docs":docs}
