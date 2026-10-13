import sys
import types
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from ingestion.models import Segment
from rag import embeddings
from rag.vector_store import VectorStore, VectorStoreError
from tests.fakes import FakeEmbedder


@pytest.fixture
def store():
    return VectorStore(QdrantClient(":memory:"))


def index(store, emb, doc_id, filename, source_type, texts, pages=None):
    chunks = [Segment(content=t, source_type=source_type, page_number=(pages[i] if pages else None),
                      start_time=None if pages else float(i), end_time=None if pages else i + 1.0)
              for i, t in enumerate(texts)]
    store.index_document(doc_id, filename, source_type, [uuid4() for _ in chunks], chunks,
                         emb.embed_passages(texts), emb.model_name)


ANGINE = "L'angine est une infection de la gorge avec fièvre et douleur à la déglutition"
TRAITEMENT = "Le traitement repose sur les antibiotiques lorsque la cause est bactérienne"
ORGA = "Le module se termine par un test de pré-requis obligatoire"


def test_search_returns_closest_chunk_with_its_source(store):
    emb, doc = FakeEmbedder(), uuid4()
    index(store, emb, doc, "cours.pdf", "pdf", [ANGINE, TRAITEMENT, ORGA], pages=[1, 2, 3])
    hits = store.search(emb.embed_query("Quels sont les symptômes de l'angine, fièvre et gorge ?"), 3)
    assert hits[0]["page_number"] == 1 and hits[0]["filename"] == "cours.pdf"
    assert hits[0]["content"] == ANGINE and hits[0]["score"] > hits[1]["score"]
    assert hits[0]["embedding_model"] == "fake-embedder"


def test_video_chunks_keep_timestamps(store):
    emb, doc = FakeEmbedder(), uuid4()
    index(store, emb, doc, "cours.mp4", "video", [ANGINE, TRAITEMENT])
    hit = store.search(emb.embed_query("antibiotiques bactérienne traitement"), 1)[0]
    assert hit["source_type"] == "video" and (hit["start_time"], hit["end_time"]) == (1.0, 2.0)
    assert hit["page_number"] is None


def test_filters_by_document_and_source_type(store):
    emb, d1, d2 = FakeEmbedder(), uuid4(), uuid4()
    index(store, emb, d1, "a.pdf", "pdf", [ANGINE], pages=[1])
    index(store, emb, d2, "b.mp4", "video", [ANGINE])
    q = emb.embed_query("angine")
    assert {h["filename"] for h in store.search(q, 5)} == {"a.pdf", "b.mp4"}
    assert [h["filename"] for h in store.search(q, 5, document_ids=[d2])] == ["b.mp4"]
    assert [h["filename"] for h in store.search(q, 5, source_types=["pdf"])] == ["a.pdf"]


def test_delete_removes_only_that_document(store):
    emb, d1, d2 = FakeEmbedder(), uuid4(), uuid4()
    index(store, emb, d1, "a.pdf", "pdf", [ANGINE, ORGA], pages=[1, 2])
    index(store, emb, d2, "b.pdf", "pdf", [TRAITEMENT], pages=[1])
    assert (store.count(d1), store.count(d2)) == (2, 1)
    store.delete_document(d1)
    assert (store.count(d1), store.count(d2)) == (0, 1)


def test_reindexing_does_not_duplicate(store):
    emb, doc = FakeEmbedder(), uuid4()
    index(store, emb, doc, "a.pdf", "pdf", [ANGINE, ORGA], pages=[1, 2])
    index(store, emb, doc, "a.pdf", "pdf", [ANGINE, ORGA], pages=[1, 2])
    assert store.count(doc) == 2


def test_empty_store_is_handled(store):
    assert store.search(FakeEmbedder().embed_query("x"), 3) == []
    assert store.count(uuid4()) == 0
    store.delete_document(uuid4())                      # ne leve pas d'erreur


def test_changed_embedding_dimension_is_detected(store):
    emb, doc = FakeEmbedder(), uuid4()
    index(store, emb, doc, "a.pdf", "pdf", [ANGINE], pages=[1])
    with pytest.raises(VectorStoreError, match="reindexez"):
        store.ensure_collection(384)


# --- prefixes des modeles e5 ------------------------------------------------
class RecordingST:
    seen: list = []

    def __init__(self, name):
        pass

    def encode(self, texts, **kw):
        RecordingST.seen = list(texts)
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=RecordingST))


def test_e5_models_get_query_and_passage_prefixes(fake_sentence_transformers):
    e = embeddings.SentenceTransformerEmbedder("intfloat/multilingual-e5-base")
    assert e.dim == 2
    e.embed_passages(["un morceau"])
    assert RecordingST.seen == ["passage: un morceau"]
    e.embed_query("une question")
    assert RecordingST.seen == ["query: une question"]


def test_other_models_get_no_prefix(fake_sentence_transformers):
    e = embeddings.SentenceTransformerEmbedder("BAAI/bge-m3")
    e.embed_passages(["un morceau"])
    assert RecordingST.seen == ["un morceau"]
