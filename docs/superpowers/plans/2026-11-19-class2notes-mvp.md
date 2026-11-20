# class2notes v0.1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Streamlit 单页应用，把音/视频文件转成四段式 Markdown 学习笔记，默认全程本地运行。

**Architecture:** 流水线式：`audio.prepare_audio` 用 ffmpeg 抽音轨并切片 → 逐片转写（`asr_local.transcribe_chunk_local` 或 `transcribe.transcribe_chunk`），合并时间戳偏移 → `notes.generate_notes` 单次或 map-reduce 生成笔记 → `notes.assemble_markdown` 组装为可下载 Markdown。UI 只做编排、后端切换和展示。

**Tech Stack:** Python 3.12、Streamlit、openai SDK、Ollama、mlx-whisper、ffmpeg/ffprobe、pytest。

**Spec:** `docs/superpowers/specs/2026-11-19-class2notes-design.md`

## Global Constraints

- Python ≥ 3.10（本机已有 Homebrew Python 3.11，用 `/opt/homebrew/bin/python3.11 -m venv .venv`；系统自带 3.9 不要用）
- `ffmpeg` 与 `ffprobe` 必须在 PATH 上
- 默认后端全本地：Ollama `qwen3.8:latest` + `mlx-community/whisper-large-v3-turbo`；云端仅作为可选后端
- Ollama 上下文必须为 16384（`OLLAMA_CONTEXT_LENGTH=16384` 后重启 Ollama）
- 本地模式单次输入上限 3000 字符，云端 12000 字符
- 任何情况下仓库中不得出现真实密钥；云端 key 只从环境变量或侧边栏读取
- 全部 UI 文案为中文；音频处理目标格式 16kHz 单声道 wav
- 切片长度默认 1200 秒
- 不实现实时、插件、说话人分离、登录、部署

## Review Focus

1. 本地 Ollama 上下文不足导致长转写被截断 → 本地默认 3000 字符/次，README 写明调高上下文的方法
2. 纯静音 / 无人声文件 → 明确提示"未识别到语音"，不生成空笔记
3. Ollama 未启动或模型未下载 → 页面显示可操作的中文错误，不白屏
4. 超过 20 分钟的音频 → 切片与 map-reduce 正常，末尾内容不丢
5. 全英文或中英混说音频 → 转写与笔记语言正确（target_lang 生效）

## File Structure

- `app.py` — Streamlit 界面、后端切换与流程编排
- `audio.py` — ffprobe 时长、切片区间计算、ffmpeg 抽取与切分
- `transcribe.py` — 云端 ASR 调用、时间戳偏移合并、全文格式化
- `asr_local.py` — mlx-whisper 本地转写
- `notes.py` — prompt 组装、单次/map-reduce 笔记生成、Markdown 组装
- `prompts.py` — 提示词常量
- `tests/test_audio.py`、`tests/test_transcribe.py`、`tests/test_asr_local.py`、`tests/test_notes.py`
- `requirements.txt`、`.env.example`、`.gitignore`、`LICENSE`、`README.md`

---

### Task 1: 环境与仓库骨架

**Files:**
- Create: `.gitignore`、`requirements.txt`、`.env.example`、`LICENSE`、`README.md`（占位）

**Interfaces:**
- Produces: 可用的 venv 与依赖；后续任务直接 import `streamlit`、`openai`、`mlx_whisper`、`pytest`

- [ ] **Step 1: 安装系统依赖**

Run: `brew install ffmpeg`
Expected: `ffmpeg -version` 有输出（Python 3.11 已装，无需安装）

- [ ] **Step 2: 初始化仓库与虚拟环境**

```bash
cd /Users/hku/python_work/class2notes
git init
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
```

- [ ] **Step 3: 写 `.gitignore`**

内容：`.venv/`、`__pycache__/`、`.env`、`.streamlit/secrets.toml`、`work/`、`*.pyc`、`.DS_Store`

- [ ] **Step 4: 写 `requirements.txt`**

```
streamlit>=1.37
openai>=1.40
mlx-whisper>=0.4
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 5: 写 `.env.example`**

```
# 默认：本地 Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3.8:latest

# 可选：云端（留空则使用本地）
OPENAI_BASE_URL=
OPENAI_API_KEY=
ASR_MODEL=

# 本地转写模型
LOCAL_ASR_MODEL=mlx-community/whisper-large-v3-turbo
```

- [ ] **Step 6: 写 MIT `LICENSE`，`README.md` 先放项目名与一行简介**

- [ ] **Step 7: 安装并验证**

Run: `pip install -r requirements.txt && python -c "import streamlit, openai, mlx_whisper, pytest; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 8: 配置 Ollama 上下文并验证**

```bash
launchctl setenv OLLAMA_CONTEXT_LENGTH 16384
# 退出菜单栏 Ollama 后重新打开
open -a Ollama
ollama run qwen3.8:latest "你好" --verbose >/dev/null
ollama ps
```
Expected: `CONTEXT` 列为 `16384`

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt .env.example LICENSE README.md docs
git commit -m "chore: scaffold project"
```

---

### Task 2: 音频抽取与切片

**Files:**
- Create: `audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces:
  - `AudioChunk`（frozen dataclass，字段 `path: Path`、`offset_sec: float`）
  - `compute_chunk_ranges(duration: float, chunk_seconds: int) -> list[tuple[float, float]]`
  - `probe_duration(path: Path) -> float`
  - `extract_audio(src: Path, out_dir: Path) -> Path`（输出 `out_dir/audio.wav`，16kHz 单声道）
  - `split_audio(wav: Path, chunk_seconds: int, out_dir: Path) -> list[AudioChunk]`
  - `prepare_audio(src: Path, work_dir: Path, chunk_seconds: int = 1200) -> list[AudioChunk]`

- [ ] **Step 1: 写失败测试 `tests/test_audio.py`**

```python
from audio import compute_chunk_ranges

def test_ranges_cover_duration_in_steps():
    assert compute_chunk_ranges(3900, 1200) == [(0, 1200), (1200, 2400), (2400, 3600), (3600, 3900)]

def test_short_audio_single_chunk():
    assert compute_chunk_ranges(90, 1200) == [(0, 90)]

def test_exact_multiple_has_no_empty_tail():
    assert compute_chunk_ranges(2400, 1200) == [(0, 1200), (1200, 2400)]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_audio.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'audio'`

- [ ] **Step 3: 实现 `audio.py`**

`compute_chunk_ranges` 按步长生成 `(start, min(start+chunk, duration))`，跳过零长度尾段。`probe_duration` 用 `ffprobe -v error -show_entries format=duration -of csv=p=0`。`extract_audio` 用 `ffmpeg -y -i src -vn -ac 1 -ar 16000 -c:a pcm_s16le out.wav`。`split_audio` 对每个区间执行 `ffmpeg -y -ss <start> -t <len> -i wav chunk_N.wav`，返回带 `offset_sec=start` 的 `AudioChunk`。`prepare_audio` 组合三者并把中间文件放进 `work_dir`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_audio.py -v`
Expected: 3 passed

- [ ] **Step 5: 端到端验证 ffmpeg 链路（无需网络）**

```bash
mkdir -p work
ffmpeg -y -f lavfi -i sine=frequency=440:duration=90 work/tone.wav
python -c "from pathlib import Path; from audio import prepare_audio; print(prepare_audio(Path('work/tone.wav'), Path('work'), 1200))"
```
Expected: 打印一个 `AudioChunk`，offset 0

- [ ] **Step 6: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "feat: extract audio and split into chunks"
```

---

### Task 3: 云端转写与通用合并

**Files:**
- Create: `transcribe.py`
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: `AudioChunk`（Task 2）
- Produces:
  - `Segment`（frozen dataclass，字段 `start: float`、`text: str`）
  - `transcribe_chunk(client, chunk: AudioChunk, model: str) -> list[Segment]`（`start` 已加 `chunk.offset_sec`）
  - `transcribe_all(chunks: list[AudioChunk], transcribe_one, progress=None) -> list[Segment]`（`transcribe_one: Callable[[AudioChunk], list[Segment]]`；空文本片段丢弃；`progress(done, total)`）
  - `format_timestamp(seconds: float) -> str`（`mm:ss`，超 1 小时为 `h:mm:ss`）
  - `format_transcript(segments: list[Segment]) -> str`（每行 `[mm:ss] 文本`）

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path
from types import SimpleNamespace
from audio import AudioChunk
from transcribe import Segment, transcribe_chunk, transcribe_all, format_timestamp, format_transcript

def fake_asr(payload):
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=lambda **kw: payload)))

def test_chunk_offset_applied():
    client = fake_asr(SimpleNamespace(segments=[SimpleNamespace(start=1.0, text="你好"), SimpleNamespace(start=601.0, text="世界")]))
    segs = transcribe_chunk(client, AudioChunk(Path("a.wav"), 1200.0), "asr")
    assert segs == [Segment(1200.0, "你好"), Segment(1801.0, "世界")]

def test_fallback_when_no_segments():
    client = fake_asr(SimpleNamespace(text="只有文本"))
    assert transcribe_chunk(client, AudioChunk(Path("a.wav"), 0.0), "asr") == [Segment(0.0, "只有文本")]

def test_empty_text_yields_no_segments():
    client = fake_asr(SimpleNamespace(text="  "))
    assert transcribe_chunk(client, AudioChunk(Path("a.wav"), 0.0), "asr") == []

def test_transcribe_all_filters_and_reports_progress():
    chunks = [AudioChunk(Path("a.wav"), 0.0), AudioChunk(Path("b.wav"), 1200.0)]
    seen = []
    def one(c):
        return [Segment(c.offset_sec, "x")] if c.offset_sec == 0 else []
    assert transcribe_all(chunks, one, progress=lambda d, t: seen.append((d, t))) == [Segment(0.0, "x")]
    assert seen == [(1, 2), (2, 2)]

def test_timestamp_over_hour():
    assert format_timestamp(3725) == "1:02:05"

def test_transcript_line_format():
    assert format_transcript([Segment(5, "你好")]) == "[00:05] 你好"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_transcribe.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'transcribe'`

- [ ] **Step 3: 实现 `transcribe.py`**

调用 `client.audio.transcriptions.create(model=model, file=open(path, "rb"), response_format="verbose_json")`；优先读 `resp.segments`（每项 `.start`、`.text`）并加偏移；没有 `segments` 时用 `resp.text` 生成单段；`text.strip()` 为空则不产出。用 `try/except AttributeError` 同时兼容 dict 与对象两种返回。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_transcribe.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add transcribe.py tests/test_transcribe.py
git commit -m "feat: cloud transcription with timestamp offsets"
```

---

### Task 4: 本地转写（mlx-whisper）

**Files:**
- Create: `asr_local.py`
- Test: `tests/test_asr_local.py`

**Interfaces:**
- Consumes: `AudioChunk`（Task 2）、`Segment`（Task 3）
- Produces:
  - `segments_from_result(result: dict) -> list[Segment]`（相对时间，未加偏移；空文本丢弃）
  - `transcribe_chunk_local(chunk: AudioChunk, model: str = "mlx-community/whisper-large-v3-turbo") -> list[Segment]`（`start` 已加 `chunk.offset_sec`）

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path
from audio import AudioChunk
from asr_local import segments_from_result
from transcribe import Segment

def test_maps_whisper_segments():
    result = {"segments": [{"start": 1.5, "text": " 你好 "}, {"start": 3.0, "text": "  "}]}
    assert segments_from_result(result) == [Segment(1.5, "你好")]

def test_empty_result():
    assert segments_from_result({"segments": []}) == []
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_asr_local.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'asr_local'`

- [ ] **Step 3: 实现 `asr_local.py`**

`segments_from_result` 遍历 `result.get("segments", [])`，取 `start` 与 `text.strip()`，非空才保留。`transcribe_chunk_local` 调用 `mlx_whisper.transcribe(str(chunk.path), path_or_hf_repo=model)`，把每个 `Segment.start` 加上 `chunk.offset_sec` 后返回。首次调用会自动下载模型（约 1.6GB）。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_asr_local.py -v`
Expected: 2 passed

- [ ] **Step 5: 用真实短音频验证（首次会下载模型）**

```bash
python -c "
from pathlib import Path
from audio import AudioChunk
from asr_local import transcribe_chunk_local
print(transcribe_chunk_local(AudioChunk(Path('work/tone.wav'), 0.0)))
"
```
Expected: 打印 `[]`（正弦音无人声）且无异常

- [ ] **Step 6: Commit**

```bash
git add asr_local.py tests/test_asr_local.py
git commit -m "feat: local whisper transcription via mlx"
```

---

### Task 5: 提示词与笔记生成

**Files:**
- Create: `prompts.py`、`notes.py`
- Test: `tests/test_notes.py`

**Interfaces:**
- Consumes: `Segment`、`format_transcript`（Task 3）
- Produces:
  - `split_transcript(text: str, max_chars: int) -> list[str]`（按行边界切，拼回去等于原文）
  - `build_notes_messages(transcript: str, target_lang: str) -> list[dict]`
  - `generate_notes(client, transcript: str, model: str, target_lang: str, max_chars_per_call: int = 3000) -> str`
  - `assemble_markdown(title: str, source: str, notes_md: str, transcript_md: str) -> str`

- [ ] **Step 1: 写失败测试**

```python
from types import SimpleNamespace
import pytest
from notes import split_transcript, build_notes_messages, generate_notes, assemble_markdown

class FakeLLM:
    def __init__(self):
        self.calls = 0
    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))
    def _create(self, **kw):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="## 一句话总结\n- 测试"))])

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
    c = FakeLLM()
    out = generate_notes(c, "短文本", "llm", "中文", max_chars_per_call=12000)
    assert c.calls == 1 and "一句话总结" in out

def test_map_reduce_when_long():
    c = FakeLLM()
    text = "\n".join("字" * 200 for _ in range(200))
    generate_notes(c, text, "llm", "中文", max_chars_per_call=12000)
    assert c.calls == 4 + 1

def test_empty_transcript_rejected():
    with pytest.raises(ValueError):
        generate_notes(FakeLLM(), "  \n", "llm", "中文")

def test_assemble_wraps_transcript():
    md = assemble_markdown("标题", "lecture.mp4", "## 笔记", "[00:05] 你好")
    assert md.startswith("# 标题")
    assert "<details>" in md and "[00:05] 你好" in md
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_notes.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'notes'`

- [ ] **Step 3: 写 `prompts.py`**

三个常量：`NOTES_SYSTEM`（规定四段结构与输出语言规则）、`NOTES_USER`（`{transcript}` 占位）、`MERGE_SYSTEM`（把多份分片要点合并去重成同一四段结构）。四段标题固定为 `## 一句话总结`、`## 核心要点`、`## 术语表`、`## 完整转写`（最后一段由代码而非模型生成）。

- [ ] **Step 4: 实现 `notes.py`**

`split_transcript` 逐行累加，超过上限就开新段。`generate_notes` 先 `strip` 后为空则 `raise ValueError("转录为空")`；不超过上限时一次调用；否则分段各生成要点，再以 `MERGE_SYSTEM` 合并，返回合并结果。`assemble_markdown` 输出：

```markdown
# {title}

> 来源：{source}

{notes_md}

## 完整转写

<details>
<summary>展开查看全文</summary>

{transcript_md}

</details>
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_notes.py -v`
Expected: 6 passed

- [ ] **Step 6: 全量测试并 Commit**

Run: `pytest -v`
Expected: 全部通过

```bash
git add prompts.py notes.py tests/test_notes.py
git commit -m "feat: generate structured notes with map-reduce"
```

---

### Task 6: Streamlit 界面

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `prepare_audio`（T2）、`transcribe_chunk`/`transcribe_all`/`format_transcript`（T3）、`transcribe_chunk_local`（T4）、`generate_notes`/`assemble_markdown`（T5）

- [ ] **Step 1: 实现 `app.py`**

侧边栏：
- 转写后端：`本地 Whisper`（默认）/ `云端 API`；选云端时显示 `base_url`、`api_key`、ASR 模型输入框
- 笔记后端：`本地 Ollama`（默认，`http://localhost:11434/v1`，key 固定 `ollama`）/ `云端 API`
- 模型下拉：`qwen3.8:latest`（默认）、`deepseek:latest`、`gpt-oss-20b`、自定义
- 切片分钟数（默认 20）、笔记语言（中文 / English）

主区：`st.file_uploader`（mp3/m4a/wav/mp4，单文件）、`st.button("生成笔记")`。点击后写入临时目录，`st.status` 显示三步（切片 / 转写 / 生成笔记），转写进度用 `st.progress`；完成后 `st.tabs(["笔记", "全文"])` 展示，`st.download_button("下载 Markdown")`。

本地模式 `max_chars_per_call` 传 3000，云端传 12000。错误分别 `st.error` 并停止：Ollama 连接失败（提示"请先启动 Ollama 并执行 OLLAMA_CONTEXT_LENGTH 设置"）、ffmpeg 缺失、转写为空（"未识别到语音，请换一段有说话的音频"）、云端 API 异常（展示错误前 300 字符）。

- [ ] **Step 2: 启动并检查页面**

Run: `streamlit run app.py --server.headless true`
Expected: 终端出现 `You can now view your Streamlit app`，页面渲染正常，两个后端默认均为本地

- [ ] **Step 3: 验证错误路径（关闭 Ollama）**

退出 Ollama 后上传 `work/tone.wav` 并生成。
Expected: 页面显示可操作的中文提示，不白屏

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit UI with local and cloud backends"
```

---

### Task 7: 真实音频端到端与提示词调优

**Files:**
- Modify: `prompts.py`（按实际输出调整）

**Interfaces:**
- Consumes: 全部前置任务；用户自备 10 分钟真实录音

- [ ] **Step 1: 本地全链路跑通，记录耗时**

Expected: 10 分钟音频 5 分钟内出笔记；四段齐全；`ollama ps` 显示 `CONTEXT 16384`

- [ ] **Step 2: 检查质量并调整 `prompts.py`**

重点看：要点是否按主题归类而非流水账；术语表是否只收专业词；对照原文抽查 3 条是否编造。改动后重跑同一音频确认改善。

- [ ] **Step 3: 验证英文音频与 target_lang**

Expected: 选 English 时笔记为英文；选中文时术语表有中英对照

- [ ] **Step 4: 全量测试并 Commit**

Run: `pytest -v`

```bash
git add prompts.py
git commit -m "fix: tune note prompts on real lecture audio"
```

---

### Task 8: README 与演示物料

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 写 README**

结构：一句话定位 → 演示 GIF（顶部）→ 它做什么 → 快速开始（**无需 API key**：装 ffmpeg → venv 装依赖 → 设 `OLLAMA_CONTEXT_LENGTH=16384` 并重启 Ollama → `streamlit run app.py`）→ 本地模型说明（默认 qwen3.8，备选 deepseek / gpt-oss，质量对比说明）→ 可选云端配置（base_url/key，成本表）→ 工作原理（文字版架构图）→ 设计取舍（为什么本地优先、为什么切片 20 分钟、为什么四段固定）→ 隐私（音频与文本不出本机）→ Roadmap（v0.2 实时字幕 Chrome 扩展）。

- [ ] **Step 2: 录 60–90 秒屏幕演示**

录：上传 → 进度 → 笔记出现 → 下载。转成 GIF 或 MP4，放进 `assets/` 并在 README 引用。

- [ ] **Step 3: 按 README 从零复跑一遍**

Run: 新开终端，严格照 Quickstart 执行（不提供任何口头补充）
Expected: 无需 API key 即可跑通；发现缺步骤就补进 README

- [ ] **Step 4: Commit**

```bash
git add README.md assets
git commit -m "docs: add demo, architecture and design rationale"
```

---

### Task 9: 发布到 GitHub

**Files:** 无代码改动

- [ ] **Step 1: 检查提交历史**

Run: `git log --oneline`
Expected: ≥7 条有意义的 commit，无密钥文件（`git log --all -- .env` 为空）

- [ ] **Step 2: 创建公开仓库并推送（推送前需用户确认）**

```bash
gh repo create class2notes --public --source=. --push
```

- [ ] **Step 3: 在新目录 clone 并按 README 复跑**

Run: `cd /tmp && git clone <repo-url> c2n-check && cd c2n-check`
Expected: 不填任何 key，按 Quickstart 能启动；不通过则修 README 或代码后补 commit

---

## Self-Review

- 规格覆盖：输入格式（T2/T6）、切片（T2）、本地转写（T4）、云端与降级（T3）、语言（T5/T7）、四段笔记（T5）、下载（T6）、README 与物料（T8）、发布（T9）
- 类型一致：`AudioChunk.path/offset_sec`、`Segment.start/text`、`transcribe_one` 回调签名、`max_chars_per_call` 默认值在各任务间一致
- Review Focus 归属：1→T1 Step 8 + T5 默认 3000 + T7 Step 1；2→T3 空文本 + T4 空结果 + T5 空转录；3→T6 Step 3；4→T5 map-reduce 测试 + T7；5→T7 Step 3
- 比例：计划约为代码量 1.5 倍，测试均为短断言，无整函数替写
