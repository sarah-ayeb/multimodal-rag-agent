"""Extraction du texte d'un PDF avec PyMuPDF : un segment par page, numero de page conserve."""
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz
from ingestion.models import Segment

MIN_CHARS_PER_PAGE = 20      # en dessous, la page est consideree sans texte exploitable
OCR_PAGE_RATIO = 0.5         # si plus de 50 % des pages sont vides -> probablement un scan


class PdfError(Exception):
    """PDF illisible, corrompu ou protege par mot de passe."""


@dataclass
class PdfExtraction:
    segments: list[Segment] = field(default_factory=list)
    page_count: int = 0
    empty_pages: list[int] = field(default_factory=list)   # numeros de pages (1-based)
    needs_ocr: bool = False


def _normalize(text: str) -> str:
    """Nettoyage d'un bloc de texte."""
    text = text.replace("\x00", "").replace("\u00ad", "")           # caractere nul, tiret conditionnel
    text = re.sub(r"([a-zà-öø-ÿ])-\n([a-zà-öø-ÿ])", r"\1\2", text)  # mots coupes en fin de ligne
    text = re.sub(r"\s*\n\s*", " ", text)                            # lignes d'un meme paragraphe
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return text.strip()


def _skeleton(text: str) -> str:
    """Forme comparable d'un bloc : chiffres masques, pour reperer 'Page 3' / 'Page 4'."""
    return re.sub(r"\d+", "#", text.lower())


def _page_blocks(page: "fitz.Page") -> list[str]:
    blocks = page.get_text("blocks", sort=True)
    return [b[4] for b in blocks if b[6] == 0 and b[4].strip()]    # b[6] == 0 : bloc de texte


def extract_pdf(path: str | Path) -> PdfExtraction:
    path = Path(path)
    try:
        doc = fitz.open(path)
    except Exception as e:  # fichier corrompu ou pas un PDF
        raise PdfError(f"PDF illisible : {e}") from e
    with doc:
        if doc.needs_pass:
            raise PdfError("PDF protege par mot de passe")
        pages = [[_normalize(b) for b in _page_blocks(p)] for p in doc]

    # Retrait des en-tetes / pieds de page : blocs courts repetes sur la moitie des pages
    n = len(pages)
    repeated: set[str] = set()
    if n >= 4:
        counts = Counter(_skeleton(b) for blocks in pages for b in set(blocks) if len(b) < 100)
        repeated = {s for s, c in counts.items() if c >= max(3, n * 0.5)}

    result = PdfExtraction(page_count=n)
    for i, blocks in enumerate(pages, start=1):
        kept = [b for b in blocks
                if b and not re.fullmatch(r"\d+", b)        # numero de page seul
                and _skeleton(b) not in repeated]
        text = "\n\n".join(kept)
        if len(text) < MIN_CHARS_PER_PAGE:
            result.empty_pages.append(i)
            continue
        result.segments.append(Segment(content=text, source_type="pdf", page_number=i))

    result.needs_ocr = n > 0 and len(result.empty_pages) / n > OCR_PAGE_RATIO
    return result
if __name__ == "__main__":
    # Essai rapide : python -m ingestion.pdf_extractor mon_fichier.pdf [--full]
    r = extract_pdf(sys.argv[1])
    full = "--full" in sys.argv
    print(f"{r.page_count} pages | {len(r.segments)} segments | pages vides: {r.empty_pages} | OCR requis: {r.needs_ocr}")
    for s in (r.segments if full else r.segments[:3]):
        text = s.content if full else s.content[:400]
        print(f"\n--- page {s.page_number} ({len(s.content)} caracteres) ---\n{text}")