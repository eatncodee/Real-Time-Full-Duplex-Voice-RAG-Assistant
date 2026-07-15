from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("key")
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    CHAT_MODEL = "gemini-3.1-flash-lite"
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "docs"

settings = Settings()