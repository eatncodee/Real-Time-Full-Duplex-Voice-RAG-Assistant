from fastapi import APIRouter, HTTPException, UploadFile, File, Header, Depends
from pydantic import BaseModel
from typing import List
from app.config import settings
from app.database import get_collection, reset_collection
from app.services.embedding import create_embedding, create_embeddings_batch
from app.services.file_process import process_file, chunk_text
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB


async def require_api_key(x_api_key: str = Header(...)):
    if not settings.DOCS_API_KEY or x_api_key != settings.DOCS_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# Auth is off while this stays local/private. Before deploying anywhere
# reachable, set DOCS_API_KEY in .env and add
# `dependencies=[Depends(require_api_key)]` back to the line below.
router = APIRouter(prefix="/documents", tags=["documents"])


class Document(BaseModel):
    text: str


class DocumentBatch(BaseModel):
    documents: List[str]


@router.post("/upload")
async def upload_document(doc: Document):
    collection = get_collection()
    try:
        embedding = await asyncio.to_thread(create_embedding, doc.text)
        if embedding:
            doc_id = f"doc_{uuid.uuid4().hex}"
            collection.add(
                documents=[doc.text],
                embeddings=[embedding],
                ids=[doc_id]
            )
            return {
                "message": "Document uploaded successfully",
                "id": doc_id,
                "text": doc.text
            }
    except Exception:
        logger.exception("upload_document failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload-batch")
async def upload_batch(batch: DocumentBatch):
    collection = get_collection()
    try:
        embeddings = await asyncio.to_thread(create_embeddings_batch, batch.documents)
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
        ids = [f"doc_{uuid.uuid4().hex}" for _ in batch.documents]

        collection.add(
            documents=batch.documents,
            embeddings=embeddings,
            ids=ids
        )
        return {
            "message": f"Successfully uploaded {len(batch.documents)} documents",
            "count": len(batch.documents)
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("upload_batch failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    collection = get_collection()
    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large")

        if content and file.filename:

            # Each of these is CPU-bound (PDF parsing, chunking, local
            # embedding) — offloaded to a thread so it doesn't stall every
            # other active connection (including live voice sessions) on
            # this process's event loop while it runs.
            text = await asyncio.to_thread(process_file, file.filename, content)

            # chunk_text returns a list of dicts:
            # {"text": "[HEADER]\n...", "section": "HEADER", "chunk_no": i}
            chunk_dicts = await asyncio.to_thread(chunk_text, text, 1000, 200)

            chunk_texts = [c["text"] for c in chunk_dicts]
            embeddings = await asyncio.to_thread(create_embeddings_batch, chunk_texts)

            doc_total_chars = len(text)
            metadata = []
            for c in chunk_dicts:
                metadata.append({
                    "source": file.filename,
                    "section": c["section"],
                    "chunk_no": c["chunk_no"],
                    # Precomputed so rag.py can decide whole-doc vs
                    # section-only retrieval from metadata alone, without
                    # fetching every chunk first just to measure it.
                    "doc_total_chars": doc_total_chars,
                })

            # Re-uploading the same filename after a chunking change can
            # produce fewer chunks than last time — without this delete,
            # the old chunks past the new count never get overwritten and
            # silently pollute retrieval forever.
            collection.delete(where={"source": file.filename})

            collection.add(
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=metadata,
                ids=[f"{file.filename}_{i}" for i in range(len(chunk_texts))]
            )

            return {
                "message": f"File uploaded successfully",
                "filename": file.filename,
                "chunks_created": len(chunk_texts),
                "total_characters": len(text)
            }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("upload_file failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/list")
async def list_documents():
    collection = get_collection()
    try:
        results = collection.get()
        documents = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                documents.append({
                    'id': results['ids'][i],
                    'text': doc
                })
        return {
            "count": len(documents),
            "documents": documents
        }
    except Exception:
        logger.exception("list_documents failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/clear")
async def clear_documents():
    try:
        reset_collection()
        return {"message": "All documents cleared successfully"}
    except Exception:
        logger.exception("clear_documents failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/count")
async def count_documents():
    collection = get_collection()
    return {"count": collection.count()}


@router.delete("/del/{target}")
async def delete_doc(target: str, mode: str = "file"):
    collection = get_collection()

    if not target or "/" in target or "\\" in target or ".." in target:
        raise HTTPException(status_code=400, detail="Invalid target")

    try:
        if mode == "file":
            collection.delete(where={"source": target})
            msg = f"Deleted all chunks for file: {target}"
        else:
            collection.delete(ids=[target])
            msg = f"Deleted specific chunk ID: {target}"
        return {"message": msg}
    except Exception:
        logger.exception("delete_doc failed")
        raise HTTPException(status_code=500, detail="Internal server error")