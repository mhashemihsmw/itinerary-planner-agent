import os
import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

import main


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeWhisper:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, *_args, **_kwargs):
        return self._segments, {}


class FakeTempDir:
    def __init__(self, path: str):
        self._path = path

    def __enter__(self):
        return self._path

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeYoutubeDL:
    def __init__(self, opts, info):
        self.opts = opts
        self._info = info

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def download(self, _urls):
        return None

    def extract_info(self, _url, download=False):
        return self._info


def _patch_tempdir(monkeypatch, tmp_path: Path):
    fake = FakeTempDir(str(tmp_path))
    monkeypatch.setattr(main.tempfile, "TemporaryDirectory", lambda: fake)


def _patch_whisper(monkeypatch, segments):
    monkeypatch.setattr(main, "get_whisper_model", lambda: FakeWhisper(segments))


def _patch_yt(monkeypatch, metadata):
    monkeypatch.setattr(main, "YoutubeDL", lambda opts: FakeYoutubeDL(opts, metadata))


@pytest.mark.asyncio
async def test_fetch_caption_text_combines_transcript_and_metadata(monkeypatch, tmp_path):
    # Arrange: make a dummy audio file so glob finds it
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_text("dummy")
    _patch_tempdir(monkeypatch, tmp_path)
    _patch_whisper(monkeypatch, [FakeSegment("hello world"), FakeSegment("from test")])
    metadata = {"title": "Sample Title", "description": "Sample Description"}
    _patch_yt(monkeypatch, metadata)

    # Act
    combined = await main.fetch_caption_text("https://www.youtube.com/watch?v=dummy")

    # Assert
    assert "hello world" in combined
    assert "Sample Title" in combined
    assert len(combined) >= 10


@pytest.mark.asyncio
async def test_fetch_caption_text_raises_for_short_combined(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_text("dummy")
    _patch_tempdir(monkeypatch, tmp_path)
    _patch_whisper(monkeypatch, [FakeSegment("short")])
    metadata = {"title": "", "description": ""}
    _patch_yt(monkeypatch, metadata)

    with pytest.raises(HTTPException) as exc:
        await main.fetch_caption_text("https://www.tiktok.com/@user/video/123")

    assert exc.value.status_code == 400
    assert "too short" in exc.value.detail


@pytest.mark.asyncio
async def test_fetch_caption_text_metadata_failure(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_text("dummy")
    _patch_tempdir(monkeypatch, tmp_path)
    _patch_whisper(monkeypatch, [FakeSegment("valid transcript text")])

    class FailingYoutubeDL(FakeYoutubeDL):
        def extract_info(self, _url, download=False):
            raise RuntimeError("metadata error")

    monkeypatch.setattr(main, "YoutubeDL", lambda opts: FailingYoutubeDL(opts, {}))

    combined = await main.fetch_caption_text("https://www.instagram.com/reel/abc")

    assert "valid transcript text" in combined
    assert len(combined) >= 10
