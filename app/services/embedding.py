from google import genai
import chromadb
from app.config import settings

client = genai.Client(api_key=settings.OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)


def create_embedding(text: str) -> list:
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text]
    )
    return result.embeddings[0].values

def create_embeddings_batch(texts: list[str]) -> list:
    return [create_embedding(t) for t in texts]