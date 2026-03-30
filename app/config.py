from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OpenAI_key")
    EMBEDDING_MODEL = "text-embedding-3-large"
    CHAT_MODEL = "gpt-4o-mini"
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "docs"

settings = Settings()