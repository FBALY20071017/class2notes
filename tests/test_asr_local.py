from asr_local import segments_from_result
from transcribe import Segment


def test_maps_whisper_segments():
    result = {"segments": [{"start": 1.5, "text": " 你好 "}, {"start": 3.0, "text": "  "}]}
    assert segments_from_result(result) == [Segment(1.5, "你好")]


def test_empty_result():
    assert segments_from_result({"segments": []}) == []
