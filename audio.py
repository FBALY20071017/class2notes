import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset_sec: float


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def compute_chunk_ranges(duration: float, chunk_seconds: int) -> list[tuple[float, float]]:
    ranges = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


def extract_audio(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        check=True,
    )
    return out


def split_audio(wav: Path, chunk_seconds: int, out_dir: Path) -> list[AudioChunk]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(wav)
    chunks = []
    for index, (start, end) in enumerate(compute_chunk_ranges(duration, chunk_seconds)):
        path = out_dir / f"chunk_{index:03d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{end - start:.3f}",
                "-i",
                str(wav),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(path),
            ],
            check=True,
        )
        chunks.append(AudioChunk(path, start))
    return chunks


def prepare_audio(src: Path, work_dir: Path, chunk_seconds: int = 1200) -> list[AudioChunk]:
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = extract_audio(src, work_dir)
    return split_audio(wav, chunk_seconds, work_dir)
