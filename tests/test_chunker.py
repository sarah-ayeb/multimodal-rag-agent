import pytest

from ingestion.chunker import chunk_segments
from ingestion.models import Segment

WORDS = " ".join(f"mot{i}" for i in range(400))          # ~2300 caracteres, coupable aux espaces


def pdf_page(text, page):
    return Segment(content=text, source_type="pdf", page_number=page)


def video_seg(text, start, end):
    return Segment(content=text, source_type="video", start_time=start, end_time=end)


def test_short_page_stays_one_chunk_with_its_page_number():
    out = chunk_segments([pdf_page("Un court paragraphe.", 3)], 800, 100)
    assert len(out) == 1 and out[0].page_number == 3 and out[0].source_type == "pdf"


def test_long_page_is_split_and_keeps_page_number():
    out = chunk_segments([pdf_page(WORDS, 7)], 500, 50)
    assert len(out) > 3
    assert all(len(c.content) <= 500 for c in out)
    assert all(c.page_number == 7 for c in out)


def test_no_text_is_lost():
    out = chunk_segments([pdf_page(WORDS, 1)], 500, 50)
    seen = set(" ".join(c.content for c in out).split())
    assert seen == set(WORDS.split())


def test_consecutive_chunks_overlap():
    out = chunk_segments([pdf_page(WORDS, 1)], 500, 80)
    for a, b in zip(out, out[1:]):
        assert b.content.split()[0] in a.content.split()[-15:]          # le debut du suivant est la fin du precedent


def test_chunks_do_not_cross_pages():
    out = chunk_segments([pdf_page(WORDS, 1), pdf_page(WORDS, 2)], 500, 50)
    assert [c.page_number for c in out] == sorted(c.page_number for c in out)
    assert {c.page_number for c in out} == {1, 2}


def test_plain_text_has_no_page_or_time():
    out = chunk_segments([Segment(content=WORDS, source_type="text")], 500, 50)
    assert all(c.page_number is None and c.start_time is None for c in out)


def test_video_sentences_are_merged_with_timestamps():
    segs = [video_seg("Bonjour a tous.", 0.0, 2.0), video_seg("Voici le cours.", 2.0, 4.5),
            video_seg("Aujourd'hui, les infections.", 4.5, 8.0)]
    out = chunk_segments(segs, 800, 100)
    assert len(out) == 1
    assert out[0].start_time == 0.0 and out[0].end_time == 8.0
    assert out[0].content == "Bonjour a tous. Voici le cours. Aujourd'hui, les infections."


def test_video_windows_split_with_sentence_overlap():
    segs = [video_seg(f"Phrase numero {i} de la transcription.", i * 3.0, i * 3.0 + 3.0) for i in range(30)]
    out = chunk_segments(segs, 200, 80)
    assert len(out) > 3 and all(len(c.content) <= 200 for c in out)
    assert out[0].start_time == 0.0 and out[-1].end_time == 90.0
    for a, b in zip(out, out[1:]):
        assert b.start_time < a.end_time                 # recouvrement
        assert b.start_time > a.start_time               # mais la fenetre avance toujours


def test_video_sentence_longer_than_size_is_kept_whole():
    out = chunk_segments([video_seg("x" * 300, 0.0, 10.0)], 200, 50)
    assert len(out) == 1 and out[0].start_time == 0.0


def test_empty_segments_are_dropped_and_input_not_modified():
    segs = [pdf_page("   ", 1), pdf_page("Contenu utile.", 2)]
    out = chunk_segments(segs, 800, 100)
    assert [c.page_number for c in out] == [2] and segs[0].content == "   "


def test_metadata_is_preserved():
    seg = Segment(content="Texte.", source_type="pdf", page_number=1, metadata={"auteur": "A"})
    assert chunk_segments([seg], 800, 100)[0].metadata == {"auteur": "A"}


@pytest.mark.parametrize("size,overlap", [(100, 100), (100, 150), (0, 0), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_segments([pdf_page("x", 1)], size, overlap)


def test_chunks_do_not_start_with_punctuation():
    text = "L'apprentissage automatique ameliore ses performances. " * 40
    out = chunk_segments([pdf_page(text, 1)], 300, 50)
    assert len(out) > 3
    assert all(c.content[0].isalnum() for c in out)
    assert all(c.content.rstrip().endswith(".") for c in out[:-1])      # les phrases gardent leur point
