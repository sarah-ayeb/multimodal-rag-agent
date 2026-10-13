"""Traitement d'un document en arriere-plan : extraction -> segments -> base, avec suivi de statut."""
import logging
from pathlib import Path
from uuid import UUID

from config.settings import settings
from db import repository as repo
from ingestion.chunker import chunk_segments
from ingestion.models import Segment
from ingestion.pdf_extractor import PdfError, extract_pdf
from ingestion.video_transcriber import VideoError, load_model, transcribe_video
from rag import embeddings, vector_store

logger = logging.getLogger(__name__)
_whisper = None


class ProcessingError(Exception):
    """Echec previsible du traitement (message destine a l'utilisateur)."""


def stored_path(upload_dir: str | Path, sha256: str, filename: str) -> Path:
    return Path(upload_dir) / f"{sha256.strip()}{Path(filename).suffix.lower()}"


def get_whisper_model():
    """Modele Whisper charge une seule fois (le chargement prend quelques secondes)."""
    global _whisper
    if _whisper is None:
        _whisper = load_model(settings.whisper_model or "small")
    return _whisper


def _extract(doc: dict, path: Path) -> tuple[list[Segment], dict]:
    kind = doc["source_type"]
    if kind == "pdf":
        r = extract_pdf(path)
        if r.needs_ocr:
            raise ProcessingError("PDF scanne : OCR requis (non pris en charge pour l'instant)")
        if not r.segments:
            raise ProcessingError("Aucun texte exploitable dans le PDF")
        return r.segments, {"page_count": r.page_count}
    if kind == "text":
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            raise ProcessingError("Fichier texte illisible : encodage UTF-8 attendu")
        if not text:
            raise ProcessingError("Le fichier texte est vide")
        return [Segment(content=text, source_type="text")], {}
    if kind == "video":
        res = transcribe_video(path, model=get_whisper_model())
        return res.segments, {"duration_sec": res.duration_sec}
    raise ProcessingError(f"Type de source inconnu : {kind}")


def _fail(document_id: UUID, task_id: UUID, message: str) -> None:
    """Enregistre l'echec et nettoie : un document en erreur ne laisse ni morceaux ni vecteurs."""
    for cleanup in (lambda: repo.replace_chunks(document_id, []),
                    lambda: vector_store.get_store().delete_document(document_id)):
        try:
            cleanup()
        except Exception:
            logger.exception("Nettoyage impossible apres echec")
    try:
        repo.set_document_status(document_id, "error", error=message)
        repo.update_task(task_id, status="error", error=message)
    except Exception:
        logger.exception("Impossible d'enregistrer l'echec du traitement")


def process_document(document_id: UUID, task_id: UUID) -> None:
    """Ne leve jamais d'exception : toute erreur devient un statut 'error' avec un message."""
    try:
        doc = repo.get_document(document_id)
        if doc is None:
            return
        repo.set_document_status(document_id, "processing")
        repo.update_task(task_id, status="processing", progress=5)

        path = stored_path(settings.upload_dir, doc["sha256"], doc["filename"])
        if not path.is_file():
            raise ProcessingError("Fichier source introuvable sur le disque")

        segments, meta = _extract(doc, path)                                   # 1. extraction
        repo.update_task(task_id, progress=40)

        chunks = chunk_segments(segments, settings.chunk_size, settings.chunk_overlap)   # 2. chunking
        if not chunks:
            raise ProcessingError("Aucun contenu a indexer")
        chunk_ids = repo.replace_chunks(document_id, chunks)
        repo.update_task(task_id, progress=60)

        embedder = embeddings.get_embedder()                                   # 3. embeddings
        vectors = embedder.embed_passages([c.content for c in chunks])
        repo.update_task(task_id, progress=85)

        vector_store.get_store().index_document(                               # 4. Qdrant
            document_id, doc["filename"], doc["source_type"], chunk_ids, chunks, vectors, embedder.model_name)

        repo.set_document_status(document_id, "done", **meta)
        repo.update_task(task_id, status="done", progress=100)
    except (PdfError, VideoError, ProcessingError) as e:
        _fail(document_id, task_id, str(e))
    except Exception as e:
        logger.exception("Erreur inattendue pendant le traitement de %s", document_id)
        _fail(document_id, task_id, f"Erreur inattendue : {e}")
