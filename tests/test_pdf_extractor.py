import textwrap

import pymupdf as fitz
import pytest

from ingestion.pdf_extractor import PdfError, extract_pdf

LOREM = ("L'intelligence artificielle generative transforme la recherche documentaire "
         "et permet de repondre a des questions en langage naturel.")


def make_pdf(path, pages, header=None):
    doc = fitz.open()
    for i, lines in enumerate(pages, start=1):
        page = doc.new_page()
        if header:
            page.insert_text((72, 40), header)
        page.insert_text((72, 760), str(i))                       # numero de page
        y = 100
        for entry in lines:                                       # paragraphe : lignes serrees, comme un vrai PDF
            for line in textwrap.wrap(entry, 70):
                page.insert_text((72, y), line)
                y += 13
    doc.save(path)
    doc.close()


def test_page_numbers_and_content(tmp_path):
    f = tmp_path / "a.pdf"
    make_pdf(f, [[LOREM], [LOREM + " Deuxieme page."]])
    r = extract_pdf(f)
    assert r.page_count == 2
    assert [s.page_number for s in r.segments] == [1, 2]
    assert all(s.source_type == "pdf" for s in r.segments)
    assert "Deuxieme page" in r.segments[1].content


def test_page_number_alone_is_removed(tmp_path):
    f = tmp_path / "b.pdf"
    make_pdf(f, [[LOREM]])
    assert extract_pdf(f).segments[0].content.strip() == LOREM


def test_repeated_header_is_removed(tmp_path):
    f = tmp_path / "c.pdf"
    make_pdf(f, [[LOREM + f" Page {i}."] for i in range(1, 6)], header="Cours IA - Universite")
    r = extract_pdf(f)
    assert all("Cours IA" not in s.content for s in r.segments)


def test_hyphenated_word_is_joined(tmp_path):
    f = tmp_path / "d.pdf"
    make_pdf(f, [["La methode est tres perfor-", "mante pour la recherche documentaire."]])
    assert "performante" in extract_pdf(f).segments[0].content


def test_empty_page_is_flagged_not_failed(tmp_path):
    f = tmp_path / "e.pdf"
    make_pdf(f, [[LOREM], [], [LOREM]])
    r = extract_pdf(f)
    assert r.empty_pages == [2] and len(r.segments) == 2 and not r.needs_ocr


def test_scanned_pdf_needs_ocr(tmp_path):
    f = tmp_path / "f.pdf"
    make_pdf(f, [[], [], [LOREM]])
    assert extract_pdf(f).needs_ocr


def test_corrupted_file_raises(tmp_path):
    f = tmp_path / "g.pdf"
    f.write_bytes(b"ceci n'est pas un pdf")
    with pytest.raises(PdfError):
        extract_pdf(f)
