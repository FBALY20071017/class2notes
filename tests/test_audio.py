from audio import compute_chunk_ranges


def test_ranges_cover_duration_in_steps():
    assert compute_chunk_ranges(3900, 1200) == [(0, 1200), (1200, 2400), (2400, 3600), (3600, 3900)]


def test_short_audio_single_chunk():
    assert compute_chunk_ranges(90, 1200) == [(0, 90)]


def test_exact_multiple_has_no_empty_tail():
    assert compute_chunk_ranges(2400, 1200) == [(0, 1200), (1200, 2400)]
