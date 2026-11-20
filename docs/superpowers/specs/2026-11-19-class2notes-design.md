# class2notes 设计（v0.1，一天版）

## 意图与原因

用户需要一个**可被验证的 AI 能力证据**用于职规赛（岗位叙事：商业分析师）。项目定位为"商业分析师的 AI 会议纪要助手"：上传一段网课/会议录音，得到一页结构化 Markdown 笔记。目标是在一天内上线一个他人能 clone 并跑起来的 v0.1，并作为第一个 GitHub 开源项目。

## 目标

- 输入 mp3 / m4a / wav / mp4 单文件 → 输出四段式笔记 + 带时间戳全文 → 下载 `.md`
- **默认全本地运行**：Ollama 生成笔记 + mlx-whisper 转写，零 API 费用、无需任何 key、音频不出本机
- 云端 OpenAI 兼容服务作为可选后端（`base_url` 可配置）
- 别人能按 README 三步跑起来

## 非目标

- 实时字幕、浏览器插件、说话人分离、账号系统、在线部署、移动端

## 反目标（失败的定义）

- 只能在作者本机跑通的脚本
- 需要作者自己的密钥才能用的工具
- 只会套壳、README 讲不出设计取舍

## 约束

- 一天内完成；单机运行；本机为 arm64 Mac / 24GB 内存
- 本地模型：Ollama `qwen3.8:latest`（27.8B，nvfp4，约 26 tok/s）、`deepseek:latest`、gpt-oss-20b 备选
- Ollama 默认上下文 4096，必须调至 16384 才能处理长转写
- 中文用户为主

## 方案比较

- **A. Streamlit 单页 + 本地优先（选定）**：最快出可演示成果，零成本，离线可讲
- B. Chrome 扩展 + 实时字幕：区分度高，但一天不可能，列为 v0.2 Roadmap
- C. CLI 脚本：最快，但不可演示，不适合作为竞赛证据

## 已定决策

- 流水线：ffmpeg 抽音轨（16k 单声道 wav）→ 按 20 分钟切片 → 逐片转写（本地 mlx-whisper 或云端 ASR）→ 合并带偏移时间戳 → LLM 生成笔记（本地 Ollama 或云端）
- 本地 ASR：`mlx-community/whisper-large-v3-turbo`，按切片调用并累加偏移
- 本地 LLM：默认 `qwen3.8:latest`；`OLLAMA_CONTEXT_LENGTH=16384`，README 写明设置方法
- 笔记固定四段：一句话总结 / 核心要点 / 术语表 / 完整转写（`<details>` 折叠）
- 长文：转写超过单次上限时 map-reduce；本地模式默认每次 3000 字符，云端 12000
- 云端 ASR 不返回 `segments` 时降级为整片一段（无时间戳），不报错
- 代码结构：`app.py`（UI）、`audio.py`、`transcribe.py`（云端 + 合并）、`asr_local.py`（本地）、`notes.py`、`prompts.py`
- 验收物料：README（架构图、设计取舍、成本耗时表）、60–90 秒录屏、≥3 个有意义的 commit

## 留给实现者

prompt 措辞、页面样式、错误文案细节、备选模型的推荐排序。
