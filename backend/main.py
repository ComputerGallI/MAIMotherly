# backend/main.py
import os
from dotenv import load_dotenv

# Load .env BEFORE importing anything that reads env
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import auth as auth_routes
from api import calendar as calendar_routes
# from api import chat as chat_routes  # keep commented for now

app = FastAPI(title="MAI Backend")

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)
app.include_router(calendar_routes.router)
# app.include_router(chat_routes.router)

@app.get("/")
def root():
    return {"ok": True, "service": "mai-backend"}
