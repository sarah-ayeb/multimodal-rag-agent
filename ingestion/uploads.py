"""Validation et stockage des fichiers importes (Module 1 : F01, F02, F03, F06).

Les fichiers sont stockes sous leur empreinte SHA-256 (`<sha256>.<ext>`) :
- le nom donne par l'utilisateur n'est jamais utilise comme chemin (pas de path traversal) ;
- deux fichiers identiques donnent le meme chemin : les doublons sont detectes sans base de donnees.
"""
import hashlib
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

CHUNK_SIZE = 1024 * 1024
HEAD_BYTES = 8192
# Les fichiers audio sont traites comme des videos (meme pipeline : FFmpeg puis Whisper).
SOURCE_TYPES = {"pdf": "pdf", "txt": "text", "mp4": "video", "mkv": "video",
                "avi": "video", "mov": "video", "mp3": "video", "wav": "video"}
ID_NAMESPACE = uuid.UUID("6f1d3a52-8c1e-4b7a-9d0e-2b5f4c8a1e37")


class UploadError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class SavedFile:
    document_id: uuid.UUID
    sha256: str
    path: Path
    size_bytes: int
    safe_name: str
    source_type: str
    duplicate: bool


def sanitize_filename(name: str, max_len: int = 100) -> str:
    """Nom sur pour l'affichage : sans chemin, sans accents, sans espaces ni caracteres speciaux."""
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "fichier"
    ext = re.sub(r"[^A-Za-z0-9]", "", ext).lower()
    stem = stem[: max(1, max_len - len(ext) - 1)]
    return f"{stem}.{ext}" if ext else stem


def _content_matches(ext: str, head: bytes) -> bool:
    """Controle de coherence entre l'extension et les premiers octets (integrite de base)."""
    if ext == "pdf":
        return head.startswith(b"%PDF-")
    if ext == "txt":
        return b"\x00" not in head
    if ext == "mp4":
        return head[4:8] == b"ftyp"
    if ext == "mov":
        return head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip")
    if ext == "mkv":
        return head.startswith(b"\x1a\x45\xdf\xa3")
    if ext == "avi":
        return head[:4] == b"RIFF" and head[8:12] == b"AVI "
    if ext == "wav":
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if ext == "mp3":
        return head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)
    return False


def save_upload(stream: BinaryIO, filename: str, upload_dir: str | Path, max_bytes: int,
                allowed_extensions: set[str]) -> SavedFile:
    ext = Path((filename or "").replace("\\", "/")).suffix.lower().lstrip(".")
    if ext not in allowed_extensions or ext not in SOURCE_TYPES:
        accepted = ", ".join(sorted(e for e in allowed_extensions if e in SOURCE_TYPES))
        raise UploadError(415, f"Format non pris en charge : « .{ext} ». Formats acceptes : {accepted}")

    upload_dir = Path(upload_dir)
    tmp_dir = upload_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4().hex}.part"

    digest = hashlib.sha256()
    size = 0
    head = b""
    try:
        with open(tmp, "wb") as out:                       # ecriture par blocs : pas de video entiere en memoire
            while chunk := stream.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadError(413, f"Fichier trop volumineux (maximum {max_bytes // (1024 * 1024)} Mo)")
                if len(head) < HEAD_BYTES:
                    head += chunk[: HEAD_BYTES - len(head)]
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise UploadError(400, "Le fichier est vide")
        if not _content_matches(ext, head):
            raise UploadError(400, f"Le contenu ne correspond pas au format .{ext} "
                                   "(fichier corrompu, renomme, ou texte non UTF-8)")
        sha = digest.hexdigest()
        final = upload_dir / f"{sha}.{ext}"
        duplicate = final.exists()
        if not duplicate:
            os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)

    return SavedFile(document_id=uuid.uuid5(ID_NAMESPACE, sha), sha256=sha, path=final, size_bytes=size,
                     safe_name=sanitize_filename(filename), source_type=SOURCE_TYPES[ext], duplicate=duplicate)
