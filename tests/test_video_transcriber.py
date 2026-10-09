import os
import subprocess
from dataclasses import dataclass

import pytest

from ingestion import video_transcriber as vt
from ingestion.video_transcriber import VideoError, transcribe_video

needs_ffmpeg = pytest.mark.skipif(vt.find_ffmpeg() is None, reason="FFmpeg non installe")


@dataclass
class FakeSeg:
    start: float
    end: float
    text: str


@dataclass
class FakeInfo:
    language: str = "fr"


class FakeModel:
    def __init__(self, segs):
        self.segs = segs

    def transcribe(self, path, **kwargs):
        return iter(self.segs), FakeInfo()


def ffmpeg(*args):
    subprocess.run([vt.find_ffmpeg(), "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture
def video_with_audio(tmp_path):
    f = tmp_path / "v.mp4"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-f", "lavfi",
           "-i", "color=c=black:s=64x64:d=3", "-shortest", "-pix_fmt", "yuv420p", str(f))
    return f


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoError, match="introuvable"):
        transcribe_video(tmp_path / "nope.mp4", model=FakeModel([]))


def test_missing_ffmpeg_raises(tmp_path, monkeypatch):
    f = tmp_path / "x.mp4"
    f.write_bytes(b"x")
    monkeypatch.setattr(vt, "find_ffmpeg", lambda: None)
    with pytest.raises(VideoError, match="FFmpeg"):
        transcribe_video(f, model=FakeModel([]))


@needs_ffmpeg
def test_segments_have_timestamps_and_skip_empty(video_with_audio):
    model = FakeModel([FakeSeg(0.0, 2.5, " Bonjour a tous "), FakeSeg(2.5, 3.0, "  "), FakeSeg(3.0, 5.2, "Voici le cours.")])
    r = transcribe_video(video_with_audio, model=model)
    assert [s.content for s in r.segments] == ["Bonjour a tous", "Voici le cours."]
    assert r.segments[0].start_time == 0.0 and r.segments[1].end_time == 5.2
    assert all(s.source_type == "video" and s.page_number is None for s in r.segments)
    assert r.language == "fr" and 2.5 < r.duration_sec < 3.5


@needs_ffmpeg
def test_no_speech_raises(video_with_audio):
    with pytest.raises(VideoError, match="parole"):
        transcribe_video(video_with_audio, model=FakeModel([]))


@needs_ffmpeg
def test_video_without_audio_track_raises(tmp_path):
    f = tmp_path / "muet.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x64:d=2", "-pix_fmt", "yuv420p", str(f))
    with pytest.raises(VideoError, match="audio"):
        transcribe_video(f, model=FakeModel([]))


@needs_ffmpeg
def test_corrupted_video_raises(tmp_path):
    f = tmp_path / "corrompu.mp4"
    f.write_bytes(b"pas une video du tout")
    with pytest.raises(VideoError):
        transcribe_video(f, model=FakeModel([]))


def test_find_ffmpeg_uses_env_variable(tmp_path, monkeypatch):
    fake = tmp_path / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    fake.write_bytes(b"x")
    monkeypatch.setenv("FFMPEG_PATH", str(tmp_path))          # dossier
    assert vt.find_ffmpeg() == str(fake)
    monkeypatch.setenv("FFMPEG_PATH", str(fake))              # fichier
    assert vt.find_ffmpeg() == str(fake)


def test_find_ffmpeg_fallback_path(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"x")
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr(vt.shutil, "which", lambda name: None)
    monkeypatch.setattr(vt, "FALLBACK_PATHS", [str(fake)])
    assert vt.find_ffmpeg() == str(fake)
