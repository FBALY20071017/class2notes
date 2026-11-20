from dataclasses import dataclass
from typing import Callable

from audio import AudioChunk


@dataclass(frozen=True)
class Segment:
    start: float
    text: str


def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def transcribe_chunk(client, chunk: AudioChunk, model: str) -> list[Segment]:
    with open(chunk.path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
        )

    segments = []
    for item in _field(response, "segments") or []:
        text = str(_field(item, "text", "")).strip()
        if text:
            start = chunk.offset_sec + float(_field(item, "start", 0.0))
            segments.append(Segment(start, text))
    if segments:
        return segments

    text = str(_field(response, "text", "")).strip()
    return [Segment(chunk.offset_sec, text)] if text else []


def transcribe_all(
    chunks: list[AudioChunk],
    transcribe_one: Callable[[AudioChunk], list[Segment]],
    progress: Callable[[int, int], None] | None = None,
) -> list[Segment]:
    total = len(chunks)
    segments: list[Segment] = []
    for done, chunk in enumerate(chunks, start=1):
        segments.extend(transcribe_one(chunk))
        if progress:
            progress(done, total)
    return segments


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_transcript(segments: list[Segment]) -> str:
    return "\n".join(f"[{format_timestamp(seg.start)}] {seg.text}" for seg in segments)


def merge_segments(segments: list[Segment], max_chars: int = 240) -> list[Segment]:
    merged: list[Segment] = []
    buffer = ""
    start = None
    for segment in segments:
        if start is None:
            start = segment.start
        buffer += segment.text
        if len(buffer) >= max_chars or buffer.endswith(("。", "！", "？", ".", "!", "?")):
            merged.append(Segment(start, buffer))
            buffer = ""
            start = None
    if buffer:
        merged.append(Segment(start, buffer))
    return merged
