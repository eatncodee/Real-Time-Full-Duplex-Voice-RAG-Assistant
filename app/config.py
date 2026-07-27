from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("key")
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    CHAT_MODEL = "gemini-3.1-flash-lite"
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "docs"
    # Shared secret required in the X-API-Key header for /documents routes
    # (upload/list/delete/clear). Set DOCS_API_KEY in your .env — without it,
    # anyone with the URL can wipe or read your whole knowledge base.
    DOCS_API_KEY = os.getenv("DOCS_API_KEY")

settings = Settings()