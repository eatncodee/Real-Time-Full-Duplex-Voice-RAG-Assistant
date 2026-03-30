from openai import OpenAI
import chromadb
from app.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=settings.COLLECTION_NAME)

def create_embedding(text: str) -> list:
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def create_embeddings_batch(texts: list[str]) -> list:
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]