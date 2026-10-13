from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from api.main import app
from config.settings import settings
from db import repository as repo
from ingestion import pipeline
from rag import search as rag_search, vector_store
from tests.test_pdf_extractor import LOREM, make_pdf
from tests.fakes import FailingEmbedder
from tests.test_video_transcriber import FakeModel, FakeSeg, ffmpeg, needs_ffmpeg


@pytest.fixture
def client(db, index):
    return TestClient(app)


def upload(c, data: bytes, name: str):
    return c.post("/api/documents/upload", files={"file": (name, data, "application/octet-stream")})


def pdf_bytes(root: Path, pages) -> bytes:
    src = root / "src"
    src.mkdir(exist_ok=True)
    f = src / "t.pdf"
    make_pdf(f, pages)
    return f.read_bytes()


def stored(root: Path):
    d = root / "uploads"
    return [p for p in d.iterdir() if p.is_file()] if d.exists() else []


# --- traitement reussi (T01, T02) ------------------------------------------
def test_pdf_is_processed_with_status_and_page_numbers(client, db):
    r = upload(client, pdf_bytes(db, [[LOREM], [LOREM + " Seconde page."]]), "Mon cours.pdf")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending" and body["duplicate"] is False

    task = client.get(f"/api/tasks/{body['task_id']}").json()
    assert task["status"] == "done" and task["progress"] == 100
    doc = client.get(f"/api/documents/{body['document_id']}").json()
    assert doc["status"] == "done" and doc["page_count"] == 2 and doc["filename"] == "Mon_cours.pdf"
    chunks = repo.get_chunks(UUID(body["document_id"]))
    assert [c["page_number"] for c in chunks] == [1, 2]


def test_text_file_is_processed(client, db):
    body = upload(client, "Bonjour, voici un cours.".encode(), "notes.txt").json()
    doc = client.get(f"/api/documents/{body['document_id']}").json()
    assert doc["status"] == "done"
    assert repo.get_chunks(UUID(body["document_id"]))[0]["content"] == "Bonjour, voici un cours."


@needs_ffmpeg
def test_video_is_transcribed_with_timestamps(client, db, monkeypatch):
    monkeypatch.setattr(pipeline, "get_whisper_model",
                        lambda: FakeModel([FakeSeg(0.0, 2.0, " Bonjour "), FakeSeg(2.0, 4.5, "Voici le cours.")]))
    src = db / "src"
    src.mkdir()
    f = src / "v.mp4"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-f", "lavfi",
           "-i", "color=c=black:s=64x64:d=3", "-shortest", "-pix_fmt", "yuv420p", str(f))
    body = upload(client, f.read_bytes(), "cours.mp4").json()
    doc = client.get(f"/api/documents/{body['document_id']}").json()
    assert doc["status"] == "done" and doc["duration_sec"] > 2
    chunks = repo.get_chunks(UUID(body["document_id"]))
    assert [(c["start_time"], c["end_time"]) for c in chunks] == [(0.0, 4.5)]       # phrases regroupees
    assert chunks[0]["content"] == "Bonjour Voici le cours."


# --- erreurs gerees sans casser le systeme (T10) ----------------------------
def test_scanned_pdf_ends_in_error_with_ocr_message(client, db):
    body = upload(client, pdf_bytes(db, [[], [], [LOREM]]), "scan.pdf").json()
    doc = client.get(f"/api/documents/{body['document_id']}").json()
    task = client.get(f"/api/tasks/{body['task_id']}").json()
    assert doc["status"] == "error" and "OCR" in doc["error_message"]
    assert task["status"] == "error"
    assert client.get("/health").status_code == 200            # le serveur continue de repondre


def test_corrupted_pdf_ends_in_error(client, db):
    body = upload(client, b"%PDF-1.4\n" + b"garbage " * 100, "casse.pdf").json()
    assert client.get(f"/api/documents/{body['document_id']}").json()["status"] == "error"


def test_unexpected_exception_becomes_error_status(client, db, monkeypatch):
    def boom(doc, path):
        raise RuntimeError("boom")
    monkeypatch.setattr(pipeline, "_extract", boom)
    body = upload(client, "du texte".encode(), "a.txt").json()
    doc = client.get(f"/api/documents/{body['document_id']}").json()
    assert doc["status"] == "error" and "Erreur inattendue" in doc["error_message"]


def test_interrupted_processing_is_marked_as_error(client, db, monkeypatch):
    monkeypatch.setattr(pipeline, "process_document", lambda *a: None)    # le traitement n'a jamais abouti
    body = upload(client, "du texte".encode(), "a.txt").json()
    assert client.get(f"/api/documents/{body['document_id']}").json()["status"] == "pending"
    assert repo.fail_interrupted() == 1
    doc = client.get(f"/api/documents/{body['document_id']}").json()
    assert doc["status"] == "error" and "interrompu" in doc["error_message"]
    assert client.get(f"/api/tasks/{body['task_id']}").json()["status"] == "error"


# --- doublons, liste, suppression (F06, F05, T08) ---------------------------
def test_duplicate_upload_is_not_reprocessed(client, db):
    data = pdf_bytes(db, [[LOREM]])
    first = upload(client, data, "a.pdf").json()
    second = upload(client, data, "copie.pdf")
    assert second.status_code == 200
    assert second.json()["duplicate"] is True and second.json()["document_id"] == first["document_id"]
    assert second.json()["task_id"] == first["task_id"] and second.json()["status"] == "done"
    assert len(client.get("/api/documents").json()) == 1 and len(stored(db)) == 1


def test_list_documents_newest_first(client, db):
    upload(client, "premier".encode(), "a.txt")
    upload(client, "second".encode(), "b.txt")
    names = [d["filename"] for d in client.get("/api/documents").json()]
    assert names == ["b.txt", "a.txt"]


def test_delete_removes_file_chunks_and_metadata(client, db, index):
    body = upload(client, pdf_bytes(db, [[LOREM]]), "a.pdf").json()
    doc_id = body["document_id"]
    assert len(stored(db)) == 1 and repo.get_chunks(UUID(doc_id)) and index.count(UUID(doc_id)) > 0

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert stored(db) == [] and repo.get_chunks(UUID(doc_id)) == [] and index.count(UUID(doc_id)) == 0
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.get(f"/api/tasks/{body['task_id']}").status_code == 404
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_unknown_ids_return_404(client):
    unknown = "3f2b8c1e-0000-4000-8000-000000000001"
    assert client.get(f"/api/documents/{unknown}").status_code == 404
    assert client.get(f"/api/tasks/{unknown}").status_code == 404


# --- upload : rejets et securite (T03) --------------------------------------
def test_unsupported_format_message(client):
    r = upload(client, b"MZ", "setup.exe")
    assert r.status_code == 415 and "Formats acceptes" in r.json()["detail"]


def test_malicious_filename_stays_in_upload_dir(client, db):
    r = upload(client, "contenu".encode(), "../../evil.txt")
    assert r.status_code == 202
    assert all(p.parent == db / "uploads" for p in stored(db))
    assert not (db.parent / "evil.txt").exists()


# --- base indisponible ------------------------------------------------------
def test_database_down_returns_503(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql://a:b@localhost:1/c")
    r = TestClient(app).get("/api/documents")
    assert r.status_code == 503 and "indisponible" in r.json()["detail"]


# --- indexation : chunking + embeddings + Qdrant -----------------------------
def test_long_text_is_chunked_and_every_chunk_is_indexed(client, db, index):
    text = " ".join(f"mot{i}" for i in range(600))                          # ~3500 caracteres
    body = upload(client, text.encode(), "long.txt").json()
    chunks = repo.get_chunks(UUID(body["document_id"]))
    assert len(chunks) > 3 and all(len(c["content"]) <= 800 for c in chunks)
    assert index.count(UUID(body["document_id"])) == len(chunks)            # PostgreSQL et Qdrant coherents


def test_indexed_pdf_can_be_searched_with_page_reference(client, db, index):
    pages = [["L'angine est une infection de la gorge avec fievre et douleur."],
             ["Le traitement repose sur les antibiotiques lorsque la cause est bacterienne."]]
    upload(client, pdf_bytes(db, pages), "cours.pdf")
    hit = rag_search.search("traitement antibiotiques bacterienne")[0]
    assert hit["page_number"] == 2 and hit["filename"] == "cours.pdf"


def test_duplicate_upload_does_not_duplicate_vectors(client, db, index):
    data = pdf_bytes(db, [[LOREM]])
    body = upload(client, data, "a.pdf").json()
    n = index.count(UUID(body["document_id"]))
    upload(client, data, "copie.pdf")
    assert index.count(UUID(body["document_id"])) == n


def test_embedding_failure_leaves_no_chunks_and_no_vectors(client, db, index, monkeypatch):
    monkeypatch.setattr(pipeline.embeddings, "get_embedder", lambda: FailingEmbedder())
    body = upload(client, "du texte a indexer".encode(), "a.txt").json()
    doc = client.get(f"/api/documents/{body['document_id']}").json()
    assert doc["status"] == "error" and "modele indisponible" in doc["error_message"]
    assert repo.get_chunks(UUID(body["document_id"])) == [] and index.count(UUID(body["document_id"])) == 0


def test_delete_returns_503_and_keeps_everything_when_qdrant_is_down(client, db, index, monkeypatch):
    from qdrant_client import QdrantClient
    body = upload(client, "du texte".encode(), "a.txt").json()
    down = vector_store.VectorStore(QdrantClient(url="http://localhost:1", timeout=2))
    monkeypatch.setattr(vector_store, "get_store", lambda: down)
    r = client.delete(f"/api/documents/{body['document_id']}")
    assert r.status_code == 503 and "vectorielle" in r.json()["detail"]
    assert client.get(f"/api/documents/{body['document_id']}").status_code == 200       # rien n'a ete supprime
