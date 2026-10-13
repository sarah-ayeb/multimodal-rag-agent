import math
import re
import zlib


class FakeEmbedder:
    """Faux modele d'embeddings : sac de mots hache en 64 dimensions, normalise.
    Deux textes qui partagent des mots ont des vecteurs proches : suffisant pour tester la recherche."""
    model_name = "fake-embedder"
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FailingEmbedder(FakeEmbedder):
    def embed_passages(self, texts):
        raise RuntimeError("modele indisponible")
