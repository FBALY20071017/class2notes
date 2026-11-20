from types import SimpleNamespace

import pytest

from notes import assemble_markdown, build_notes_messages, generate_notes, split_transcript


class FakeLLM:
    def __init__(self):
        self.calls = 0

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="## 一句话总结\n- 测试"))]
        )


def test_split_preserves_content():
    text = "\n".join(f"[00:{i:02d}] " + "字" * 30 for i in range(10))
    parts = split_transcript(text, 100)
    assert "\n".join(parts) == text
    assert all(parts)


def test_messages_include_target_language():
    msgs = build_notes_messages("内容", "English")
    assert msgs[0]["role"] == "system"
    assert "English" in msgs[0]["content"] + msgs[1]["content"]


def test_single_call_when_short():
    client = FakeLLM()
    out = generate_notes(client, "短文本", "llm", "中文", max_chars_per_call=12000)
    assert client.calls == 1
    assert "一句话总结" in out


def test_map_reduce_when_long():
    client = FakeLLM()
    text = "\n".join("字" * 200 for _ in range(200))
    generate_notes(client, text, "llm", "中文", max_chars_per_call=12000)
    assert client.calls == 4 + 1


def test_empty_transcript_rejected():
    with pytest.raises(ValueError):
        generate_notes(FakeLLM(), "  \n", "llm", "中文")


def test_assemble_wraps_transcript():
    md = assemble_markdown("标题", "lecture.mp4", "## 笔记", "[00:05] 你好")
    assert md.startswith("# 标题")
    assert "<details>" in md
    assert "[00:05] 你好" in md
