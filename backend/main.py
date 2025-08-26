from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import chat  # eventually also: auth, calendar
from dotenv import load_dotenv  
load_dotenv()
                 
app = FastAPI(title="MAI AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(chat.router, prefix="")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
