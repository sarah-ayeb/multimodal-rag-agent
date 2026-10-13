from dataclasses import dataclass, field


@dataclass
class Segment:
    """Un morceau de contenu extrait (schema du cahier des charges, section 2.3).
    document_id est ajoute plus tard, au moment de l'enregistrement en base."""
    content: str
    source_type: str                      # "pdf" | "text" | "video"
    page_number: int | None = None        # PDF (1 = premiere page)
    start_time: float | None = None       # video, en secondes
    end_time: float | None = None
    metadata: dict = field(default_factory=dict)
