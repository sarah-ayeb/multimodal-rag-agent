"""Modele d'embeddings (texte -> vecteur). Le modele se change dans .env (EMBEDDING_MODEL)."""
from config.settings import settings

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


class SentenceTransformerEmbedder:
    """Les modeles e5 demandent un prefixe : 'query: ' pour les questions, 'passage: ' pour les morceaux."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer     # import tardif : lourd (PyTorch)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = len(self.model.encode(["test"])[0])
        e5 = "e5" in model_name.lower()
        self._query_prefix, self._passage_prefix = ("query: ", "passage: ") if e5 else ("", "")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]


_embedder = None


def get_embedder():
    """Modele charge une seule fois (le chargement prend quelques secondes)."""
    global _embedder
    name = settings.embedding_model or DEFAULT_EMBEDDING_MODEL
    if _embedder is None or _embedder.model_name != name:
        _embedder = SentenceTransformerEmbedder(name)
    return _embedder
