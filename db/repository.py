"""Acces a PostgreSQL : documents, taches, segments (chunks)."""
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from db.database import get_conn

INTERRUPTED_MSG = "Traitement interrompu (arret du serveur)"


def create_document(saved) -> dict | None:
    """Insere le document. Retourne None si ce contenu existe deja (meme id ou meme SHA-256)."""
    with get_conn() as c:
        return c.execute(
            """INSERT INTO documents (id, filename, source_type, size_bytes, sha256)
               VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING *""",
            (saved.document_id, saved.safe_name, saved.source_type, saved.size_bytes, saved.sha256),
        ).fetchone()


def get_document(document_id: UUID) -> dict | None:
    with get_conn() as c:
        return c.execute("SELECT * FROM documents WHERE id = %s", (document_id,)).fetchone()


def list_documents() -> list[dict]:
    with get_conn() as c:
        return c.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()


def delete_document(document_id: UUID) -> dict | None:
    """Supprime le document ; la cascade SQL supprime ses segments et ses taches."""
    with get_conn() as c:
        return c.execute("DELETE FROM documents WHERE id = %s RETURNING *", (document_id,)).fetchone()


def set_document_status(document_id: UUID, status: str, error: str | None = None,
                        page_count: int | None = None, duration_sec: float | None = None) -> None:
    with get_conn() as c:
        c.execute(
            """UPDATE documents SET status = %s, error_message = %s,
                      page_count = COALESCE(%s, page_count), duration_sec = COALESCE(%s, duration_sec),
                      updated_at = now() WHERE id = %s""",
            (status, error, page_count, duration_sec, document_id))


def create_task(document_id: UUID, kind: str = "ingest") -> dict:
    with get_conn() as c:
        return c.execute("INSERT INTO tasks (document_id, kind) VALUES (%s, %s) RETURNING *",
                         (document_id, kind)).fetchone()


def get_task(task_id: UUID) -> dict | None:
    with get_conn() as c:
        return c.execute("SELECT * FROM tasks WHERE id = %s", (task_id,)).fetchone()


def latest_task(document_id: UUID) -> dict | None:
    with get_conn() as c:
        return c.execute("SELECT * FROM tasks WHERE document_id = %s ORDER BY created_at DESC LIMIT 1",
                         (document_id,)).fetchone()


def update_task(task_id: UUID, status: str | None = None, progress: int | None = None,
                error: str | None = None) -> None:
    sets, vals = ["updated_at = now()"], []
    if status is not None:
        sets.append("status = %s"); vals.append(status)
    if progress is not None:
        sets.append("progress = %s"); vals.append(progress)
    if error is not None:
        sets.append("error_message = %s"); vals.append(error)
    with get_conn() as c:
        c.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = %s", vals + [task_id])


def replace_chunks(document_id: UUID, segments: list) -> list[UUID]:
    """Remplace tous les morceaux d'un document. Retourne leurs ids (reutilises comme ids Qdrant)."""
    ids = [uuid4() for _ in segments]
    with get_conn() as c:
        c.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        with c.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (id, document_id, chunk_index, content, page_number, start_time, end_time, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [(ids[i], document_id, i, s.content, s.page_number, s.start_time, s.end_time, Jsonb(s.metadata))
                 for i, s in enumerate(segments)])
    return ids


def get_chunks(document_id: UUID) -> list[dict]:
    with get_conn() as c:
        return c.execute("SELECT * FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                         (document_id,)).fetchall()


def fail_interrupted() -> int:
    """Au demarrage : les traitements restes 'en attente' ou 'en cours' ont ete tues avec le serveur."""
    with get_conn() as c:
        c.execute("UPDATE tasks SET status = 'error', error_message = %s, updated_at = now() "
                  "WHERE status IN ('pending', 'processing')", (INTERRUPTED_MSG,))
        cur = c.execute("UPDATE documents SET status = 'error', error_message = %s, updated_at = now() "
                        "WHERE status IN ('pending', 'processing')", (INTERRUPTED_MSG,))
        return cur.rowcount
