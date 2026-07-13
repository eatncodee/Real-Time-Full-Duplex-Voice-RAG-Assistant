from fastembed import TextEmbedding
import chromadb
from app.config import settings

# Loaded once at import time, reused for every call — this is why it's fast (~5-15ms)
_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)


def create_embedding(text: str) -> list:
    embedding = list(_model.embed([text]))[0]
    return embedding.tolist()


def create_embeddings_batch(texts: list[str]) -> list:
    embeddings = list(_model.embed(texts))
    return [e.tolist() for e in embeddings]