"""Contrats de l'API (Module 8). Les memes modeles servent a valider et a documenter (Swagger)."""
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class SourceType(str, Enum):
    pdf = "pdf"
    text = "text"
    video = "video"


class Status(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    error = "error"


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    source_type: SourceType
    size_bytes: int
    status: Status
    error_message: str | None = None
    page_count: int | None = None
    duration_sec: float | None = None
    created_at: datetime


class UploadResponse(BaseModel):
    document_id: UUID
    task_id: UUID
    status: Status


class TaskOut(BaseModel):
    id: UUID
    document_id: UUID
    status: Status
    progress: int = Field(ge=0, le=100)
    error_message: str | None = None


class Passage(BaseModel):
    """Un passage source : page pour un PDF, horodatage pour une video."""
    document_id: UUID
    filename: str
    source_type: SourceType
    content: str
    score: float | None = None
    page_number: int | None = None
    start_time: float | None = None
    end_time: float | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[UUID] | None = None
    source_types: list[SourceType] | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    passages: list[Passage]


class ChatMode(str, Enum):
    rag = "rag"
    agent = "agent"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[UUID] | None = None
    mode: ChatMode = ChatMode.rag
    conversation_id: UUID | None = None


class AgentStep(BaseModel):
    tool: str
    input: str
    summary: str


class ChatResponse(BaseModel):
    answer: str
    sufficient_sources: bool          # False -> "informations insuffisantes" (T07)
    sources: list[Passage]
    verified: bool | None = None      # renseigne par verify_sources (semaine 5)
    steps: list[AgentStep] = []       # trace de l'agent (mode agent)
    conversation_id: UUID | None = None


class CompareRequest(BaseModel):
    document_ids: list[UUID] = Field(min_length=2)
    question: str = Field(min_length=1, max_length=4000)


class CompareResponse(BaseModel):
    common_points: list[str]
    differences: list[str]
    contradictions: list[str]
    sources: list[Passage]


class ErrorOut(BaseModel):
    detail: str
