from types import SimpleNamespace

from audio import AudioChunk
from transcribe import (
    Segment,
    format_timestamp,
    format_transcript,
    transcribe_all,
    transcribe_chunk,
)


def fake_asr(payload):
    return SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=lambda **kw: payload))
    )


def chunk_at(tmp_path, offset_sec):
    path = tmp_path / "a.wav"
    path.write_bytes(b"")
    return AudioChunk(path, offset_sec)


def test_chunk_offset_applied(tmp_path):
    client = fake_asr(
        SimpleNamespace(
            segments=[
                SimpleNamespace(start=1.0, text="你好"),
                SimpleNamespace(start=601.0, text="世界"),
            ]
        )
    )
    segs = transcribe_chunk(client, chunk_at(tmp_path, 1200.0), "asr")
    assert segs == [Segment(1201.0, "你好"), Segment(1801.0, "世界")]


def test_fallback_when_no_segments(tmp_path):
    client = fake_asr(SimpleNamespace(text="只有文本"))
    assert transcribe_chunk(client, chunk_at(tmp_path, 0.0), "asr") == [
        Segment(0.0, "只有文本")
    ]


def test_empty_text_yields_no_segments(tmp_path):
    client = fake_asr(SimpleNamespace(text="  "))
    assert transcribe_chunk(client, chunk_at(tmp_path, 0.0), "asr") == []


def test_transcribe_all_filters_and_reports_progress(tmp_path):
    chunks = [chunk_at(tmp_path, 0.0), chunk_at(tmp_path, 1200.0)]
    seen = []

    def one(chunk):
        return [Segment(chunk.offset_sec, "x")] if chunk.offset_sec == 0 else []

    result = transcribe_all(chunks, one, progress=lambda done, total: seen.append((done, total)))
    assert result == [Segment(0.0, "x")]
    assert seen == [(1, 2), (2, 2)]


def test_timestamp_over_hour():
    assert format_timestamp(3725) == "1:02:05"


def test_transcript_line_format():
    assert format_transcript([Segment(5, "你好")]) == "[00:05] 你好"
