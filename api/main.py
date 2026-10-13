import logging
from contextlib import asynccontextmanager
from uuid import UUID

import psycopg
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException

from api.schemas import (ChatRequest, ChatResponse, CompareRequest, CompareResponse, DocumentOut, Passage,
                         SearchRequest, SearchResponse, TaskOut, UploadResponse)
from config.settings import settings
from db import repository as repo
from ingestion import pipeline
from ingestion.uploads import UploadError, save_upload
from rag import search as rag_search
from rag import vector_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        n = repo.fail_interrupted()
        if n:
            logger.warning("%d traitement(s) interrompu(s) marque(s) en erreur", n)
    except Exception as e:  # base indisponible au demarrage : /health reste utilisable
        logger.warning("Nettoyage au demarrage impossible : %s", e)
    yield


app = FastAPI(title="Assistant multimodal", version="0.2.0", lifespan=lifespan)
NOT_YET = HTTPException(status_code=501, detail="Pas encore implemente")


@app.exception_handler(psycopg.OperationalError)
async def database_down(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Base de donnees indisponible"})


@app.exception_handler(ResponseHandlingException)
async def vector_db_down(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Base vectorielle indisponible"})


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Documents (Module 1) ----------------------------------------------------
@app.post("/api/documents/upload", response_model=UploadResponse, status_code=202)
def upload_document(background: BackgroundTasks, response: Response, file: UploadFile = File(...)):
    try:
        saved = save_upload(file.file, file.filename or "", settings.upload_dir,
                            settings.max_file_size_mb * 1024 * 1024, settings.extensions)
    except UploadError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    common = dict(document_id=saved.document_id, filename=saved.safe_name, source_type=saved.source_type,
                  size_bytes=saved.size_bytes, sha256=saved.sha256)
    if repo.create_document(saved) is None:                       # contenu deja importe
        existing = repo.get_document(saved.document_id)
        last = repo.latest_task(saved.document_id)
        response.status_code = 200
        return UploadResponse(**common, duplicate=True, status=existing["status"],
                              task_id=last["id"] if last else None)

    task = repo.create_task(saved.document_id)
    background.add_task(pipeline.process_document, saved.document_id, task["id"])
    return UploadResponse(**common, duplicate=False, status="pending", task_id=task["id"])


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents():
    return repo.list_documents()


@app.get("/api/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: UUID):
    doc = repo.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document introuvable")
    return doc


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: UUID):
    doc = repo.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document introuvable")
    vector_store.get_store().delete_document(doc_id)     # d'abord les vecteurs : si Qdrant est en panne, rien n'est supprime
    repo.delete_document(doc_id)                         # puis la base (cascade : morceaux et taches)
    pipeline.stored_path(settings.upload_dir, doc["sha256"], doc["filename"]).unlink(missing_ok=True)
    return Response(status_code=204)


@app.get("/api/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: UUID):
    task = repo.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Tache introuvable")
    return task


# --- Recherche / chat / comparaison : semaines 3 a 5 -------------------------
@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Recherche semantique : les passages les plus proches de la question, avec leur source."""
    source_types = [s.value for s in req.source_types] if req.source_types else None
    hits = rag_search.search(req.query, req.top_k, req.document_ids, source_types)
    return SearchResponse(passages=[
        Passage(document_id=h["document_id"], filename=h["filename"], source_type=h["source_type"],
                content=h["content"], score=h["score"], page_number=h["page_number"],
                start_time=h["start_time"], end_time=h["end_time"]) for h in hits])


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    raise NOT_YET


@app.post("/api/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    raise NOT_YET
