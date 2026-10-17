"""Document management routes — upload, delete, stats, list, clear."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Request, UploadFile, File, HTTPException

from api.schemas import DocumentDeleteResponse, DocumentUploadResponse, StatsResponse

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    allowed = {".txt", ".pdf", ".md", ".pptx"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed)}",
        )

    # Save to temp file, ingest, then clean up
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / (file.filename or "upload" + suffix)
    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        pipeline = request.app.state.pipeline
        result = pipeline.ingest_document(tmp_path)
        return DocumentUploadResponse(**result)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/documents")
async def list_documents(request: Request) -> List[Dict[str, Any]]:
    """Return all unique documents stored in ChromaDB."""
    pipeline = request.app.state.pipeline
    coll = pipeline.vector_store.collection
    results = coll.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    docs_map: Dict[str, Dict[str, Any]] = {}
    for meta in metadatas:
        if not meta:
            continue
        doc_id = meta.get("document_id") or meta.get("source") or meta.get("document_name")
        doc_name = meta.get("document_name", "Unknown Document")
        if not doc_id:
            continue
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "id": str(doc_id),
                "name": doc_name,
                "chunks": 0,
                "ingested_at": meta.get("ingested_at", ""),
            }
        docs_map[doc_id]["chunks"] += 1

    return list(docs_map.values())


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str, request: Request):
    pipeline = request.app.state.pipeline
    n = pipeline.delete_document(doc_id)
    # If not deleted by id directly, try deleting by document_name
    if n == 0:
        results = pipeline.vector_store.collection.get(
            where={"document_name": {"$eq": doc_id}},
            include=[],
        )
        ids = results.get("ids", [])
        if ids:
            pipeline.vector_store.collection.delete(ids=ids)
            n = len(ids)
    return DocumentDeleteResponse(deleted=n)


@router.post("/documents/clear")
async def clear_documents(request: Request):
    """Wipe all indexed chunks from ChromaDB."""
    pipeline = request.app.state.pipeline
    pipeline.vector_store.reset_collection()
    return {"status": "cleared", "total_chunks": 0}


@router.get("/documents/stats", response_model=StatsResponse)
async def document_stats(request: Request):
    pipeline = request.app.state.pipeline
    stats = pipeline.get_stats()
    return StatsResponse(**stats)
