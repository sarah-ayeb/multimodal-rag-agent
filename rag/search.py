"""Recherche semantique (version minimale pour verifier l'indexation).
Essai : python -m rag.search "ma question" [--top 5]"""
import sys
from uuid import UUID

from rag import embeddings, vector_store


def search(question: str, top_k: int = 5, document_ids: list[UUID] | None = None,
           source_types: list[str] | None = None) -> list[dict]:
    vec = embeddings.get_embedder().embed_query(question)
    return vector_store.get_store().search(vec, top_k, document_ids, source_types)


if __name__ == "__main__":
    args = sys.argv[1:]
    top = int(args[args.index("--top") + 1]) if "--top" in args else 5
    for n, hit in enumerate(search(args[0], top), 1):
        where = f"page {hit['page_number']}" if hit["page_number"] else (
            f"{hit['start_time']}s -> {hit['end_time']}s" if hit["start_time"] is not None else "texte")
        print(f"\n#{n} score {hit['score']:.3f} | {hit['filename']} | {where}\n   {hit['content'][:250]}")
