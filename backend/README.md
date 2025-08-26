# MAI Backend (FastAPI)
## Summary
This is the FastAPI backend for **MAI**.  
It supports:
- Chat generation using a local knowledge corpus
- Gemini API integration
- Google Login (OAuth) and Google Calendar integration


## Folder Structure

backend/
- api/
  - auth.py → Google Login / OAuth endpoints (uses services/google_auth.py)
  - calendar.py → Calendar endpoints (uses services/google_calendar.py)
  - chat.py → Chat endpoints: /health, /generate, /debug/*
- artifacts/ 
  - knowledge_corpus.pkl → REQUIRED. Knowledge base of guidance/advice entries.
  - faiss_index.bin → OPTIONAL but recommended. FAISS vector index for semantic retrieval.
  - config.json → OPTIONAL. Metadata (date, corpus size, categories, embedding dimension, etc.).
  - retriever_model/ → OPTIONAL. Saved SentenceTransformer directory (enables local embedding on server if present).
  - generator_model/ → OPTIONAL. Saved seq2seq model (e.g., BART) for local text generation.
  - generator_tokenizer/ → OPTIONAL. Tokenizer directory for the generator model.
- modelsAI/
  - model_loader.py → Loads the knowledge corpus
  - search_engine.py → Search and response logic
- pydantic/
  - schema.py → Pydantic request/response models
- services/
  - gemini.py → Gemini API client
  - google_auth.py → Google OAuth helper
  - google_calendar.py → Google Calendar helper
- main.py → FastAPI entry point: sets up CORS and mounts routers
- requirements.txt → Python dependencies



## How Things Connect

- **main.py** starts the FastAPI app, enables CORS, and includes routers from `api/`.
- **api/chat.py**  
  - `/health` → service status and model info  
  - `/generate` → chat endpoint (uses `modelsAI/model_loader.py`, `modelsAI/search_engine.py`, and `pydantic/schema.py`)  
  - `/debug/corpus` and `/debug/search/{query}` → developer helpers
- **api/auth.py** → routes for Google Login (uses `services/google_auth.py`)  
- **api/calendar.py** → routes for Google Calendar (uses `services/google_calendar.py`)  
- **services/gemini.py** → stub for Gemini API integration



## Setup

1. Install dependencies:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Artifacts**
    - TODO: Place the trained corpus here:
     ```
     backend/artifacts/knowledge_corpus.pkl
     ```
    - Path at Enviroment Variables
     ```bash
     export ARTIFACTS_PATH=./backend/artifacts
     ```

3. **Environment variables**
- **Artifacts**
  - `ARTIFACTS_PATH=`

- **Gemini**
  - `GEMINI_API_KEY`

- **Google OAuth**
  - `GOOGLE_CLIENT_ID`
  - `GOOGLE_CLIENT_SECRET`
  - `GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback`

- **Google Calendar**
  - `GOOGLE_CALENDAR_SCOPES=https://www.googleapis.com/auth/calendar.events`

  
4. **Run the server**\
  On windows PowerShell:
    ```bash
    python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
    ```
    Other:
      ```bash
      uvicorn main:app --reload --host 0.0.0.0 --port 8000
      ```
5. **Run the tests**\
Make sure that the server is running before in another terminal.\
  On windows PowerShell:
    ```bash
    python -m pytest -q
    ```
    Other
    ```bash
    pytest -q
    ```
## Current Endpoints

- `GET /health` → Returns service status and model info

- `POST /generate` → Generates a chat reply  
  - **Request body:**
    ```json
    {
      "user_input": "I'm anxious about a presentation",
      "quiz_summary": "",
      "subscription_tier": "free"
    }
    ```
  - **Response body:**
    ```json
    {
      "response": "I understand that nervousness before important events. Try taking deep breaths...",
      "suggestions": ["Try deep breathing", "Practice positive visualization"]
    }
    ```

- `GET /debug/corpus` → Returns a sample of the loaded corpus  

- `GET /debug/search/{query}` → Returns top matches for the given query  

## Notes

- The frontend can call FastAPI directly (`http://localhost:8000`)   

- If the knowledge corpus is missing, the chat falls back to a generic supportive response.

- `Template.ENV` contains a template to add the respective credentials. Note its name must be update to `.ENV` so it works.

