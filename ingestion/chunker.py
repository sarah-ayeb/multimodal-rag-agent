"""Decoupage en morceaux (chunking) : PDF/texte -> morceaux de taille reguliere, video -> fenetres de transcription.

Chaque morceau garde sa provenance : numero de page (PDF) ou horodatage de debut et de fin (video).
"""
import sys
from itertools import groupby

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.models import Segment

DEFAULT_SIZE = 800
DEFAULT_OVERLAP = 100


def _chunk_text_segment(seg: Segment, splitter: RecursiveCharacterTextSplitter) -> list[Segment]:
    return [Segment(content=piece.strip(), source_type=seg.source_type, page_number=seg.page_number,
                    start_time=seg.start_time, end_time=seg.end_time, metadata=dict(seg.metadata))
            for piece in splitter.split_text(seg.content) if piece.strip()]


def _merge(group: list[Segment]) -> Segment:
    return Segment(content=" ".join(s.content.strip() for s in group), source_type="video",
                   start_time=group[0].start_time, end_time=group[-1].end_time,
                   metadata=dict(group[0].metadata))


def _chunk_video(segs: list[Segment], size: int, overlap: int) -> list[Segment]:
    """Regroupe les phrases de la transcription en fenetres d'environ `size` caracteres,
    avec un recouvrement d'environ `overlap` caracteres (en phrases entieres)."""
    out: list[Segment] = []
    i, n = 0, len(segs)
    while i < n:
        j, length = i, 0
        while j < n:
            add = len(segs[j].content) + (1 if j > i else 0)
            if j > i and length + add > size:
                break
            length += add
            j += 1
        out.append(_merge(segs[i:j]))
        if j >= n:
            break
        k, ov = j, 0                                   # recule pour creer le recouvrement
        while k - 1 > i and ov + len(segs[k - 1].content) + 1 <= overlap:
            k -= 1
            ov += len(segs[k].content) + 1
        i = k                                          # toujours > i : la boucle progresse
    return out


def chunk_segments(segments: list[Segment], chunk_size: int = DEFAULT_SIZE,
                   overlap: int = DEFAULT_OVERLAP) -> list[Segment]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Il faut 0 <= overlap < chunk_size")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""],
                                              keep_separator="end")      # le point reste a la fin de sa phrase
    segments = [s for s in segments if s.content and s.content.strip()]
    out: list[Segment] = []
    for source_type, run in groupby(segments, key=lambda s: s.source_type):
        run = list(run)
        if source_type == "video":
            out.extend(_chunk_video(run, chunk_size, overlap))
        else:
            for seg in run:
                out.extend(_chunk_text_segment(seg, splitter))
    return out


if __name__ == "__main__":
    # Essai : python -m ingestion.chunker fichier.pdf|.txt|.mp4 [--size 800] [--overlap 100] [--full]
    args = sys.argv[1:]
    opt = lambda k, d: int(args[args.index(k) + 1]) if k in args else d
    path = args[0]
    if path.lower().endswith(".pdf"):
        from ingestion.pdf_extractor import extract_pdf
        raw = extract_pdf(path).segments
    elif path.lower().endswith(".txt"):
        raw = [Segment(content=open(path, encoding="utf-8").read(), source_type="text")]
    else:
        from ingestion.video_transcriber import load_model, transcribe_video
        raw = transcribe_video(path, model=load_model("small")).segments
    chunks = chunk_segments(raw, opt("--size", DEFAULT_SIZE), opt("--overlap", DEFAULT_OVERLAP))
    sizes = [len(c.content) for c in chunks]
    print(f"{len(raw)} segments -> {len(chunks)} morceaux | taille min/moy/max : "
          f"{min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)} caracteres")
    for n, c in enumerate(chunks if "--full" in args else chunks[:3], 1):
        where = f"page {c.page_number}" if c.page_number else f"{c.start_time}s -> {c.end_time}s" if c.start_time is not None else "texte"
        print(f"\n--- morceau {n} ({where}, {len(c.content)} car.) ---\n{c.content}")
