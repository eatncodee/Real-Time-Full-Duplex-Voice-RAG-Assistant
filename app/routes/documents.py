from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from app.database import get_collection, reset_collection
from app.services.embedding import create_embedding, create_embeddings_batch
from app.services.file_process import process_file, chunk_text

router = APIRouter(prefix="/documents", tags=["documents"])


class Document(BaseModel):
    text: str


class DocumentBatch(BaseModel):
    documents: List[str]


@router.post("/upload")
async def upload_document(doc: Document):
    collection = get_collection()
    try:
        embedding = create_embedding(doc.text)
        if embedding:
            count = collection.count()
            collection.add(
                documents=[doc.text],
                embeddings=[embedding],
                ids=[f"doc_{count}"]
            )
            return {
                "message": "Document uploaded successfully",
                "id": f"doc_{count}",
                "text": doc.text
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-batch")
async def upload_batch(batch: DocumentBatch):
    collection = get_collection()
    try:
        count = collection.count()
        embeddings = create_embeddings_batch(batch.documents)
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
        ids = [f"doc_{count+i}" for i in range(len(batch.documents))]

        collection.add(
            documents=batch.documents,
            embeddings=embeddings,
            ids=ids
        )
        return {
            "message": f"Successfully uploaded {len(batch.documents)} documents",
            "count": len(batch.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    collection = get_collection()
    try:
        content = await file.read()
        if content and file.filename:

            text = process_file(file.filename, content)

            # chunk_text now returns a list of dicts:
            # {"text": "[HEADER]\n...", "section": "HEADER", "chunk_no": i}
            chunk_dicts = chunk_text(text, chunk_size=1000, overlap=200)

            chunk_texts = [c["text"] for c in chunk_dicts]
            embeddings = create_embeddings_batch(chunk_texts)

            metadata = []
            for c in chunk_dicts:
                metadata.append({
                    "source": file.filename,
                    "section": c["section"],
                    "chunk_no": c["chunk_no"],
                })

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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_documents():
    try:
        reset_collection()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


router.get("/count")
async def count_documents():
    collection = get_collection()
    return {"count": collection.count()}


@router.delete("/del/{target}")
async def delete_doc(target: str, mode: str = "file"):
    collection = get_collection()

    try:
        if mode == "file":
            collection.delete(where={"source": target})
            msg = f"Deleted all chunks for file: {target}"
        else:
            collection.delete(ids=[target])
            msg = f"Deleted specific chunk ID: {target}"
        return {"message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))