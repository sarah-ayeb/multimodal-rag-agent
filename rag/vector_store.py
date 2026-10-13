"""Base vectorielle Qdrant : un point par morceau (chunk). Vecteurs normalises -> similarite cosinus."""
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector, MatchAny, MatchValue,
                                  PayloadSchemaType, PointStruct, VectorParams)

from config.settings import settings

COLLECTION = "chunks"
BATCH = 128


class VectorStoreError(Exception):
    pass


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION):
        self.client = client
        self.collection = collection

    def _exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, dim: int) -> None:
        if not self._exists():
            self.client.create_collection(self.collection,
                                          vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            for field in ("document_id", "source_type"):
                try:
                    self.client.create_payload_index(self.collection, field, PayloadSchemaType.KEYWORD)
                except Exception:
                    pass                                   # sans effet en mode local (tests)
            return
        size = self.client.get_collection(self.collection).config.params.vectors.size
        if size != dim:
            raise VectorStoreError(f"Le modele d'embeddings a change (dimension {size} -> {dim}) : "
                                   "supprimez la collection Qdrant et reindexez les documents")

    def index_document(self, document_id: UUID, filename: str, source_type: str, chunk_ids: list,
                       chunks: list, vectors: list[list[float]], embedding_model: str) -> None:
        self.ensure_collection(len(vectors[0]))
        self.delete_document(document_id)                 # reindexation propre : pas de doublons
        points = [PointStruct(id=str(cid), vector=vec, payload={
            "chunk_id": str(cid), "document_id": str(document_id), "filename": filename,
            "source_type": source_type, "page_number": c.page_number, "start_time": c.start_time,
            "end_time": c.end_time, "content": c.content, "embedding_model": embedding_model})
            for cid, c, vec in zip(chunk_ids, chunks, vectors)]
        for i in range(0, len(points), BATCH):
            self.client.upsert(self.collection, points=points[i:i + BATCH])

    def search(self, vector: list[float], limit: int = 5, document_ids: list | None = None,
               source_types: list[str] | None = None) -> list[dict]:
        if not self._exists():
            return []
        must = []
        if document_ids:
            must.append(FieldCondition(key="document_id", match=MatchAny(any=[str(d) for d in document_ids])))
        if source_types:
            must.append(FieldCondition(key="source_type", match=MatchAny(any=[str(s) for s in source_types])))
        res = self.client.query_points(self.collection, query=vector, limit=limit,
                                       query_filter=Filter(must=must) if must else None, with_payload=True)
        return [{"score": p.score, **p.payload} for p in res.points]

    def _doc_filter(self, document_id: UUID) -> Filter:
        return Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))])

    def count(self, document_id: UUID) -> int:
        if not self._exists():
            return 0
        return self.client.count(self.collection, count_filter=self._doc_filter(document_id), exact=True).count

    def delete_document(self, document_id: UUID) -> None:
        if self._exists():
            self.client.delete(self.collection, points_selector=FilterSelector(filter=self._doc_filter(document_id)))


_client = None


def get_store() -> VectorStore:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, timeout=10)
    return VectorStore(_client)
