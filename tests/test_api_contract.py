from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_search_rejects_empty_query():
    assert client.post("/api/search", json={"query": ""}).status_code == 422


def test_search_rejects_bad_top_k():
    assert client.post("/api/search", json={"query": "x", "top_k": 99}).status_code == 422


def test_compare_needs_two_documents():
    body = {"document_ids": ["3f2b8c1e-0000-4000-8000-000000000001"], "question": "q"}
    assert client.post("/api/compare", json=body).status_code == 422


def test_invalid_uuid_rejected():
    assert client.get("/api/documents/pas-un-uuid").status_code == 422


def test_unimplemented_returns_501():
    assert client.get("/api/documents").status_code == 501
