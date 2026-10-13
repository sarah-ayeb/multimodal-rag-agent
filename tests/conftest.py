"""Fixtures communes. Les tests qui touchent la base utilisent une base SEPAREE `assistant_test`
(creee automatiquement) : vos donnees de developpement ne sont jamais effacees.
Si PostgreSQL n'est pas joignable, ces tests sont ignores (skipped)."""
import urllib.parse
from pathlib import Path

import psycopg
import pytest

from config.settings import settings

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@pytest.fixture(scope="session")
def test_db_url():
    parsed = urllib.parse.urlparse(settings.database_url)
    test_url = parsed._replace(path="/assistant_test").geturl()
    admin_url = parsed._replace(path="/postgres").geturl()
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as c:
            if not c.execute("SELECT 1 FROM pg_database WHERE datname = 'assistant_test'").fetchone():
                c.execute("CREATE DATABASE assistant_test")
        with psycopg.connect(test_url, autocommit=True, connect_timeout=3) as c:
            c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            c.execute(SCHEMA.read_text(encoding="utf-8"))
    except psycopg.OperationalError as e:
        pytest.skip(f"PostgreSQL indisponible : {e}")
    return test_url


@pytest.fixture
def db(test_db_url, monkeypatch, tmp_path):
    """Base de test videe + dossier d'upload temporaire. Retourne le dossier temporaire racine."""
    monkeypatch.setattr(settings, "database_url", test_db_url)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    with psycopg.connect(test_db_url, autocommit=True) as c:
        c.execute("TRUNCATE documents, chunks, tasks, conversations, messages CASCADE")
    return tmp_path


@pytest.fixture
def index(monkeypatch):
    """Qdrant en memoire + faux modele d'embeddings (aucun service ni telechargement necessaire)."""
    from qdrant_client import QdrantClient

    from rag import embeddings, vector_store
    from tests.fakes import FakeEmbedder
    store = vector_store.VectorStore(QdrantClient(":memory:"))
    monkeypatch.setattr(vector_store, "get_store", lambda: store)
    monkeypatch.setattr(embeddings, "get_embedder", lambda: FakeEmbedder())
    return store
