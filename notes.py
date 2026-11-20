from prompts import MERGE_SYSTEM, NOTES_SYSTEM, NOTES_USER


def split_transcript(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        line_size = len(line) + 1
        if current and size + line_size > max_chars:
            parts.append("\n".join(current))
            current = [line]
            size = line_size
        else:
            current.append(line)
            size += line_size
    if current:
        parts.append("\n".join(current))
    return parts


def build_notes_messages(transcript: str, target_lang: str) -> list[dict]:
    return [
        {"role": "system", "content": NOTES_SYSTEM.format(target_lang=target_lang)},
        {"role": "user", "content": NOTES_USER.format(transcript=transcript)},
    ]


def _clean_notes(text: str) -> str:
    index = text.find("## ")
    return text[index:] if index > 0 else text.strip()


def _complete(client, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return _clean_notes(response.choices[0].message.content.strip())


def generate_notes(
    client,
    transcript: str,
    model: str,
    target_lang: str,
    max_chars_per_call: int = 3000,
) -> str:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("转录为空")

    parts = split_transcript(transcript, max_chars_per_call)
    if len(parts) == 1:
        return _complete(client, model, build_notes_messages(parts[0], target_lang))

    partials = [_complete(client, model, build_notes_messages(part, target_lang)) for part in parts]
    merge_messages = [
        {"role": "system", "content": MERGE_SYSTEM.format(target_lang=target_lang)},
        {"role": "user", "content": "\n\n---\n\n".join(partials)},
    ]
    return _complete(client, model, merge_messages)


def assemble_markdown(title: str, source: str, notes_md: str, transcript_md: str) -> str:
    return f"""# {title}

> 来源：{source}

{notes_md.strip()}

## 完整转写

<details>
<summary>展开查看全文</summary>

{transcript_md.strip()}

</details>
"""
