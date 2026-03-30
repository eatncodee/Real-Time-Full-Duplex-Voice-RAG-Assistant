from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, chat,voice

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API using OpenAI and ChromaDB",
    version="3.1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(voice.router)



@app.get("/")
async def root():
    return {
        "message": "RAG API is running",
        "endpoints": {
            "upload": "POST /documents/upload",
            "ask": "POST /chat/ask",
            "list": "GET /documents/list",
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}