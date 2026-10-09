from uuid import UUID

from fastapi import FastAPI, HTTPException

from api.schemas import (ChatRequest, ChatResponse, CompareRequest, CompareResponse,
                         DocumentOut, SearchRequest, SearchResponse, TaskOut, UploadResponse)

app = FastAPI(title="Assistant multimodal", version="0.1.0")

NOT_YET = HTTPException(status_code=501, detail="Pas encore implemente")


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Documents (Module 1) : implementes en semaine 2-3 -----------------------
@app.post("/api/documents/upload", response_model=UploadResponse, status_code=202)
def upload_document():
    raise NOT_YET


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents():
    raise NOT_YET


@app.get("/api/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: UUID):
    raise NOT_YET


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: UUID):
    raise NOT_YET


@app.get("/api/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: UUID):
    raise NOT_YET


# --- Recherche / chat / comparaison : semaines 3 a 5 -------------------------
@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    raise NOT_YET


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    raise NOT_YET


@app.post("/api/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    raise NOT_YET
