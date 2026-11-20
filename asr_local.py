from audio import AudioChunk
from transcribe import Segment

LOCAL_ASR_MODEL = "mlx-community/whisper-large-v3-turbo"


def segments_from_result(result: dict) -> list[Segment]:
    segments = []
    for item in result.get("segments", []) or []:
        text = str(item.get("text", "")).strip()
        if text:
            segments.append(Segment(float(item.get("start", 0.0)), text))
    return segments


def transcribe_chunk_local(chunk: AudioChunk, model: str = LOCAL_ASR_MODEL) -> list[Segment]:
    import mlx_whisper

    result = mlx_whisper.transcribe(str(chunk.path), path_or_hf_repo=model)
    return [
        Segment(segment.start + chunk.offset_sec, segment.text)
        for segment in segments_from_result(result)
    ]
