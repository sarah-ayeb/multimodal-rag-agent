import io
from pathlib import Path

import pytest

from ingestion.uploads import UploadError, sanitize_filename, save_upload

ALLOWED = {"pdf", "txt", "mp4", "mkv", "avi", "mov", "mp3", "wav"}
PDF = b"%PDF-1.4\n" + b"contenu de test " * 50
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100


def save(tmp_path, data, name, max_bytes=10_000_000):
    return save_upload(io.BytesIO(data), name, tmp_path, max_bytes, ALLOWED)


def stored_files(tmp_path):
    return [p for p in Path(tmp_path).iterdir() if p.is_file()]


# --- nettoyage du nom -------------------------------------------------------
def test_sanitize_whatsapp_name():
    assert sanitize_filename("WhatsApp Video 2026-09-03 at 13.34.24.mp4") == "WhatsApp_Video_2026-09-03_at_13.34.24.mp4"


def test_sanitize_accents_and_paths():
    assert sanitize_filename("C:\\docs\\cours été.PDF") == "cours_ete.pdf"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("") == "fichier"


# --- fichiers valides (T01, T02) --------------------------------------------
def test_valid_pdf_is_stored_by_hash(tmp_path):
    r = save(tmp_path, PDF, "Mon Cours.pdf")
    assert r.source_type == "pdf" and r.size_bytes == len(PDF) and not r.duplicate
    assert r.path.name == f"{r.sha256}.pdf" and r.path.read_bytes() == PDF
    assert r.safe_name == "Mon_Cours.pdf"


def test_valid_video_and_text(tmp_path):
    assert save(tmp_path, MP4, "v.mp4").source_type == "video"
    assert save(tmp_path, "Bonjour, ceci est un cours.".encode("utf-8"), "n.txt").source_type == "text"


# --- rejets (T03) -----------------------------------------------------------
def test_unsupported_format_rejected(tmp_path):
    with pytest.raises(UploadError) as e:
        save(tmp_path, b"MZ....", "virus.exe")
    assert e.value.status_code == 415 and "pdf" in str(e.value)
    assert stored_files(tmp_path) == []


def test_fake_pdf_rejected(tmp_path):
    with pytest.raises(UploadError) as e:
        save(tmp_path, b"ceci est du texte, pas un pdf", "faux.pdf")
    assert e.value.status_code == 400


def test_empty_file_rejected(tmp_path):
    with pytest.raises(UploadError) as e:
        save(tmp_path, b"", "vide.pdf")
    assert e.value.status_code == 400


def test_binary_text_file_rejected(tmp_path):
    with pytest.raises(UploadError):
        save(tmp_path, b"abc\x00\x01def", "binaire.txt")


def test_too_large_rejected_and_tmp_cleaned(tmp_path):
    with pytest.raises(UploadError) as e:
        save(tmp_path, PDF, "gros.pdf", max_bytes=100)
    assert e.value.status_code == 413
    assert stored_files(tmp_path) == [] and list((tmp_path / ".tmp").iterdir()) == []


# --- doublons (F06) ---------------------------------------------------------
def test_duplicate_detected_and_not_stored_twice(tmp_path):
    a = save(tmp_path, PDF, "a.pdf")
    b = save(tmp_path, PDF, "copie_de_a.pdf")
    assert not a.duplicate and b.duplicate
    assert a.document_id == b.document_id and len(stored_files(tmp_path)) == 1
