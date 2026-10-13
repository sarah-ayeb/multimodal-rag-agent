"""Video -> audio (FFmpeg) -> transcription (faster-whisper) -> segments horodates."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.models import Segment

FFMPEG_TIMEOUT_SEC = 1800
FALLBACK_PATHS = [r"C:\ffmpeg\bin\ffmpeg.exe"]   # emplacements essayes si FFmpeg n'est pas dans le PATH


class VideoError(Exception):
    """Video illisible, sans audio, sans parole, ou outil manquant."""


@dataclass
class TranscriptionResult:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration_sec: float = 0.0           # duree de l'audio
    extract_sec: float = 0.0            # temps d'extraction audio (FFmpeg)
    transcribe_sec: float = 0.0         # temps de transcription (hors chargement du modele)
    realtime_factor: float = 0.0        # transcribe_sec / duration_sec : 1.0 = aussi long que la video


def find_ffmpeg() -> str | None:
    """Cherche FFmpeg : variable FFMPEG_PATH, puis PATH, puis emplacements habituels (sans toucher au PATH)."""
    env = os.environ.get("FFMPEG_PATH")
    if env:
        p = Path(env)
        if p.is_dir():
            p = p / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if p.is_file():
            return str(p)
    found = shutil.which("ffmpeg")
    if found:
        return found
    for c in FALLBACK_PATHS:
        if Path(c).is_file():
            return c
    return None


def extract_audio(video_path: str | Path, wav_path: str | Path) -> float:
    """Extrait la piste audio en WAV mono 16 kHz (format attendu par Whisper). Retourne la duree en secondes."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise VideoError(f"Fichier introuvable : {video_path}")
    ffmpeg_bin = find_ffmpeg()
    if ffmpeg_bin is None:
        raise VideoError("FFmpeg introuvable : installez-le dans C:\\ffmpeg\\bin ou definissez FFMPEG_PATH")
    cmd = [ffmpeg_bin, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as e:
        raise VideoError("Extraction audio trop longue (timeout)") from e
    out = Path(wav_path)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size < 100:
        reason = "aucune piste audio" if "does not contain any stream" in proc.stderr or "Output file #0 does not contain" in proc.stderr \
            else "fichier video invalide ou corrompu"
        raise VideoError(f"Extraction audio impossible ({reason})")
    with wave.open(str(out), "rb") as w:
        return w.getnframes() / w.getframerate()


def load_model(name: str = "small", device: str = "cpu", compute_type: str = "int8"):
    """Charge faster-whisper (telecharge le modele la premiere fois)."""
    from faster_whisper import WhisperModel
    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe_video(video_path: str | Path, model=None, model_name: str = "small",
                     language: str | None = None) -> TranscriptionResult:
    """Transcrit une video. `model` peut etre fourni (reutilisation, tests) ; sinon il est charge."""
    res = TranscriptionResult()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        t0 = time.perf_counter()
        res.duration_sec = extract_audio(video_path, wav)
        res.extract_sec = time.perf_counter() - t0

        if model is None:
            model = load_model(model_name)

        t1 = time.perf_counter()
        raw_segments, info = model.transcribe(str(wav), language=language, vad_filter=True, beam_size=5)
        for s in raw_segments:                          # generateur : la transcription se fait ici
            text = s.text.strip()
            if text:
                res.segments.append(Segment(content=text, source_type="video",
                                            start_time=round(s.start, 2), end_time=round(s.end, 2)))
        res.transcribe_sec = time.perf_counter() - t1
        res.language = getattr(info, "language", None)

    if not res.segments:
        raise VideoError("Aucune parole detectee dans la video")
    res.realtime_factor = res.transcribe_sec / res.duration_sec if res.duration_sec else 0.0
    return res


def _mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


if __name__ == "__main__":
    # Essai : python -m ingestion.video_transcriber video.mp4 [--model small] [--lang fr] [--full]
    args = sys.argv[1:]
    opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
    name = opt("--model", "small")
    t0 = time.perf_counter()
    model = load_model(name)
    load_sec = time.perf_counter() - t0
    r = transcribe_video(args[0], model=model, language=opt("--lang"))
    print(f"Modele : {name} | chargement : {load_sec:.1f}s | langue detectee : {r.language}")
    print(f"Duree audio : {_mmss(r.duration_sec)} | extraction FFmpeg : {r.extract_sec:.1f}s | transcription : {r.transcribe_sec:.1f}s")
    print(f"Facteur temps reel : {r.realtime_factor:.2f}  (1.00 = transcription aussi longue que la video)")
    print(f"{len(r.segments)} segments")
    for s in (r.segments if "--full" in args else r.segments[:5]):
        print(f"[{_mmss(s.start_time)} -> {_mmss(s.end_time)}] {s.content}")
