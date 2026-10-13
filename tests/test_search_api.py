from fastapi.testclient import TestClient

from api.main import app
from tests.test_pdf_extractor import make_pdf
from tests.test_pipeline_api import upload

ANGINE = "L'angine est une infection de la gorge avec fievre et douleur a la deglutition."
TRAITEMENT = "Le traitement repose sur les antibiotiques lorsque la cause est bacterienne."


def pdf(root, pages, name="t.pdf"):
    src = root / "src"
    src.mkdir(exist_ok=True)
    make_pdf(src / name, pages)
    return (src / name).read_bytes()


def search(client, **body):
    return client.post("/api/search", json=body)


def test_search_returns_best_passage_with_page_and_filename(db, index):
    client = TestClient(app)
    upload(client, pdf(db, [[ANGINE], [TRAITEMENT]]), "cours.pdf")
    r = search(client, query="traitement antibiotiques bacterienne", top_k=2)
    assert r.status_code == 200
    top = r.json()["passages"][0]
    assert top["page_number"] == 2 and top["filename"] == "cours.pdf" and top["source_type"] == "pdf"
    assert top["start_time"] is None and 0 < top["score"] <= 1.0001
    assert len(r.json()["passages"]) == 2


def test_search_respects_top_k(db, index):
    client = TestClient(app)
    upload(client, pdf(db, [[ANGINE], [TRAITEMENT]]), "cours.pdf")
    assert len(search(client, query="angine", top_k=1).json()["passages"]) == 1


def test_search_filters_by_document(db, index):
    client = TestClient(app)
    a = upload(client, pdf(db, [[ANGINE]], "a.pdf"), "a.pdf").json()
    b = upload(client, pdf(db, [[TRAITEMENT]], "b.pdf"), "b.pdf").json()
    r = search(client, query="angine", document_ids=[b["document_id"]])
    assert {p["filename"] for p in r.json()["passages"]} == {"b.pdf"}
    r = search(client, query="angine", document_ids=[a["document_id"], b["document_id"]], top_k=5)
    assert {p["filename"] for p in r.json()["passages"]} == {"a.pdf", "b.pdf"}


def test_search_filters_by_source_type(db, index):
    client = TestClient(app)
    upload(client, pdf(db, [[ANGINE]]), "cours.pdf")
    upload(client, ANGINE.encode(), "notes.txt")
    r = search(client, query="angine", source_types=["text"], top_k=5)
    assert {p["filename"] for p in r.json()["passages"]} == {"notes.txt"}
    assert all(p["page_number"] is None for p in r.json()["passages"])


def test_search_with_nothing_indexed_returns_empty_list(db, index):
    r = search(TestClient(app), query="n'importe quoi")
    assert r.status_code == 200 and r.json() == {"passages": []}


def test_deleted_document_is_no_longer_found(db, index):
    client = TestClient(app)
    body = upload(client, pdf(db, [[ANGINE]]), "cours.pdf").json()
    assert search(client, query="angine").json()["passages"]
    client.delete(f"/api/documents/{body['document_id']}")
    assert search(client, query="angine").json() == {"passages": []}
