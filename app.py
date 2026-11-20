import os
import tempfile
from functools import partial
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from asr_local import LOCAL_ASR_MODEL, transcribe_chunk_local
from audio import has_speech, prepare_audio
from notes import assemble_markdown, generate_notes
from transcribe import format_transcript, merge_segments, transcribe_all, transcribe_chunk

load_dotenv()

OLLAMA_DEFAULT = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODELS = [
    "modelscope.cn/unsloth/gpt-oss-20b-GGUF:Q4_K_M",
    "deepseek:latest",
    "qwen3.8:latest",
]

st.set_page_config(page_title="class2notes", page_icon="📝", layout="wide")


def unload_ollama(base_url: str, model: str) -> None:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        httpx.post(
            f"{root}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
    except httpx.HTTPError:
        pass


with st.sidebar:
    st.header("设置")
    asr_mode = st.radio("转写方式", ["本地 Whisper（免费）", "云端 API"], index=0)
    llm_mode = st.radio("笔记生成", ["本地 Ollama（免费）", "云端 API"], index=0)

    cloud_base = ""
    cloud_key = ""
    asr_model = ""
    if "云端" in asr_mode or "云端" in llm_mode:
        cloud_base = st.text_input(
            "云端 Base URL", os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        )
        cloud_key = st.text_input("云端 API Key", os.getenv("OPENAI_API_KEY", ""), type="password")
        asr_model = st.text_input("云端 ASR 模型", os.getenv("ASR_MODEL", "FunAudioLLM/SenseVoiceSmall"))

    if "本地" in llm_mode:
        llm_base = st.text_input("Ollama 地址", OLLAMA_DEFAULT)
        preset = st.selectbox("笔记模型", OLLAMA_MODELS + ["自定义"])
        llm_model = (
            st.text_input("自定义模型名", "modelscope.cn/unsloth/gpt-oss-20b-GGUF:Q4_K_M")
            if preset == "自定义"
            else preset
        )
    else:
        llm_base = cloud_base
        llm_model = st.text_input("笔记模型", os.getenv("LLM_MODEL", ""))

    target_lang = st.selectbox("笔记语言", ["中文", "English"])
    chunk_minutes = st.number_input("切片长度（分钟）", min_value=5, max_value=60, value=20)
    st.caption(
        "本地模式无需任何 API key。默认 gpt-oss-20b（11GB）；"
        "qwen3.8 在 24GB 内存下处理长文本可能显存不足。"
    )

st.title("class2notes")
st.caption("上传网课或会议录音，生成一页结构化 Markdown 笔记。")

uploaded = st.file_uploader("音频或视频文件", type=["mp3", "m4a", "wav", "mp4", "mov"])

if st.button("生成笔记", type="primary", disabled=uploaded is None):
    local_asr = "本地" in asr_mode
    local_llm = "本地" in llm_mode

    if not local_asr and not cloud_key:
        st.error("使用云端转写需要在侧边栏填写 API Key")
        st.stop()
    if not local_llm and not (cloud_base and cloud_key):
        st.error("使用云端生成需要在侧边栏填写 Base URL 和 API Key")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / uploaded.name
        src.write_bytes(uploaded.getvalue())

        with st.status("准备音频…", expanded=True) as status:
            try:
                chunks = prepare_audio(src, Path(tmp), int(chunk_minutes) * 60)
                status.write(f"已切成 {len(chunks)} 段")

                audible = [chunk for chunk in chunks if has_speech(chunk.path)]
                if not audible:
                    raise RuntimeError("未识别到语音，请换一段有说话的音频")

                if local_asr and local_llm:
                    unload_ollama(llm_base, llm_model)

                status.write("转写中…")
                if local_asr:
                    transcribe_one = partial(
                        transcribe_chunk_local,
                        model=os.getenv("LOCAL_ASR_MODEL", LOCAL_ASR_MODEL),
                    )
                else:
                    asr_client = OpenAI(base_url=cloud_base, api_key=cloud_key)
                    transcribe_one = lambda chunk: transcribe_chunk(asr_client, chunk, asr_model)

                progress = st.progress(0.0)
                segments = transcribe_all(
                    audible,
                    transcribe_one,
                    progress=lambda done, total: progress.progress(done / total),
                )
                if not segments:
                    raise RuntimeError("未识别到语音，请换一段有说话的音频")

                segments = merge_segments(segments)
                status.write(f"转写完成，共 {len(segments)} 段；生成笔记中…")
                transcript_md = format_transcript(segments)

                llm_client = OpenAI(
                    base_url=llm_base,
                    api_key="ollama" if local_llm else cloud_key,
                )
                notes_md = generate_notes(
                    llm_client,
                    transcript_md,
                    llm_model,
                    target_lang,
                    max_chars_per_call=2000 if local_llm else 12000,
                )

                st.session_state["md"] = assemble_markdown(
                    Path(uploaded.name).stem, uploaded.name, notes_md, transcript_md
                )
                st.session_state["notes"] = notes_md
                st.session_state["transcript"] = transcript_md
                st.session_state["file"] = uploaded.name
                status.update(label="完成", state="complete")
            except Exception as error:
                status.update(label="失败", state="error")
                message = str(error)
                if "Insufficient Memory" in message or "OutOfMemory" in message:
                    message = "本地模型显存不足：请换用 gpt-oss-20b 或 deepseek，或缩短切片长度。"
                st.error(message[:300])

if "md" in st.session_state:
    st.subheader(st.session_state["file"])
    tab_notes, tab_transcript = st.tabs(["笔记", "全文"])
    with tab_notes:
        st.markdown(st.session_state["notes"])
    with tab_transcript:
        st.text(st.session_state["transcript"])
    st.download_button(
        "下载 Markdown",
        data=st.session_state["md"],
        file_name="notes.md",
        mime="text/markdown",
    )
