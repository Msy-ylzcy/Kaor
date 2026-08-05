# Kaor 中文使用手册

本文对应 Windows x64 的 CPU、AMD 和 NVIDIA 三套解压即用发行包，也保留源码开发命令。页面面向桌面端，不设计移动端运行 AI 工作流。

## 1. 启动

CPU/AMD 用户从 Release 下载对应 ZIP 并完整解压。NVIDIA 用户把
`Kaor-Windows-x64-NVIDIA-Setup.exe`、`Kaor-Windows-x64-NVIDIA.parts.json`
和从 `.zip.001` 开始的全部分片下载到同一目录，双击 Setup 完成校验与解压。
随后在解压目录双击：

```text
Kaor.exe
```

无需安装 Python、Node.js、FFmpeg、PaddleOCR、PyTorch、UVR5 或 CUDA Toolkit，也不要为发行包运行 `pip install`、创建虚拟环境或设置开发环境变量。程序默认打开 `http://127.0.0.1:8765/`；端口被占用时会顺延并打开实际地址。

源码开发环境已经安装依赖时，在项目根目录运行：

```powershell
npm.cmd run build --prefix apps/web
$env:KAOR_DATA_DIR = "$PWD\data"
$env:KAOR_NO_BROWSER = "1"
.\.venv-nvidia-cu126\Scripts\python.exe -m backend.main
```

打开 `http://127.0.0.1:8765/`。页面仅针对桌面端工作流设计。关闭后端进程会停止本地服务。

常用环境变量：

| 变量 | 作用 |
| --- | --- |
| `KAOR_DATA_DIR` | 项目、CSV、缓存和导出目录 |
| `KAOR_PORT` | 首选端口，默认 `8765` |
| `KAOR_NO_BROWSER=1` | 禁止启动时自动打开浏览器 |

## 2. 工作流总览

1. 导入视频并等待项目创建及本地音轨准备完成。
2. 在视频上用 8 个控制点调整原字幕区和译文区。
3. 选择 `字幕`、`音频` 或 `混合` 模式。
4. 运行本地识别并检查证据 CSV。
5. 在字幕工作区人工新增、删除或修订 Cue。
6. 混合模式可先运行 AI 融合，得到最终 `source.csv`。
7. 单独运行 AI 翻译，得到 `translated.csv`。
8. 调整人物/单条颜色、字幕位置和排版，先预览再导出。

识别与翻译是两个独立步骤。只想提取字幕时，完成第 5 或第 6 步后直接下载 `source.csv`。

### 2.1 导入视频时的音轨准备

导入不是只把视频复制到项目目录。后端注册片源后会立即探测时长、分辨率、帧率和流信息，并用 FFmpeg 提取：

```text
data\projects\<project_id>\cache\audio\mix.wav
```

成功提示为“视频与本地音轨已准备完成”。提取音轨失败时，视频仍会保留并显示单独警告；修复 FFmpeg 或源音轨问题后，可运行 UVR5 阶段触发缺失音轨重提取，或使用工作区重置重新准备。

### 2.2 重置当前工作区

“重置工作区”保留项目的原视频、`manifest.json` 和项目设置，删除 Kaor 为该项目生成的内容：

- 整个 `cache/`，包括旧 `mix.wav`、`vocals.wav`、16 kHz 音频、切片、内嵌字幕缓存和 worker 日志。
- 整个 `exports/`，包括预览、ASS 和已渲染视频。
- `ocr.csv`、`speech.csv`、`translated.csv` 以及旧 `source.csv` 字幕行。
- 当前进程中该项目的历史任务记录。

重置完成后会创建空表头 `source.csv`，重新探测保留的片源，并重新提取新的 `mix.wav`。存在运行中或排队任务时，先取消或等待任务结束；“删除项目”才会同时删除片源。

## 3. 框选字幕区和译文区

选择对应区域后，拖动四个顶点或四条边中点。顶点改变两个方向，边中点只改变一条边，适合精确贴合透视或非标准字幕区域。

- 原字幕区应覆盖完整描边、第二行和可能叠加的第二条字幕。
- 译文区应避开原字幕、人物脸部、UI 和重要画面内容。
- 两人同时说话时，译文区需要足够高度容纳多个 `layer`。

## 4. 选择识别模式

### 4.1 字幕模式

字幕模式优先提取内嵌字幕流；没有可用流时使用 PaddleOCR。画面 OCR 会：

- 只处理所选 ROI。
- 使用变化检测复用静止字幕。
- 对渐入字幕等待更清晰的候选帧。
- 对长驻字幕定期复查，而不是每次采样都生成新 Cue。
- 使用多图像变体和多帧共识。
- 过滤低质量短 ASCII、孤立数字和切换碎片。
- 按文本、位置和时间合并重复采样。

运行时视频字幕区显示最新 OCR 文字，底部表格持续接收任务快照。任务卡的 `采样 A/B` 表示已处理采样帧/预计总采样帧，旁边还会显示设备、批次、真正调用 OCR 的帧数、复用帧数和处理速度。最终文件为 `ocr.csv`，并同步成为当前 `source.csv`。

OCR 帧批处理保持“自动”时，会按 PaddlePaddle 识别到的设备总显存选择外层帧批次和文本识别批次。常见 16 GB 显卡属于 14 GiB 以上档位，自动使用 `40/80`；若当前内容仍触发 GPU OOM，Kaor 会释放缓存并把失败批次递归二分，直至该批次成功或单帧仍报错。任务指标中的 `ocr_batch_backoffs` 和 `effective_ocr_batch_size` 可确认是否发生回退。

### 4.2 音频模式

音频模式适合无字幕、画面字幕很差或需要语音转写的场景。导入阶段已经准备 `mix.wav`；识别面板随后提供三个可独立运行的阶段：

1. `UVR5 人声分离`：读取 `mix.wav`，用 BS-RoFormer 写出 `vocals.wav`。
2. `静音切分`：把人声转为 16 kHz 单声道 `speech-16k-mono.wav`，按静音边界写出 `slices/slice-*.wav` 和时间偏移清单 `slices/slices.json`。
3. `ASR 打标`：按切片或切片批次转写，把相对时间恢复为视频绝对时间，再执行可选的时间边界精修与说话人分段，写出 `speech.csv` 和当前 `source.csv`。

需要调整切分参数时，可以只重跑第 2、3 阶段；更换 ASR 模型或批大小时，只重跑第 3 阶段即可。独立运行的前置关系为 `mix.wav -> vocals.wav -> speech-16k-mono.wav + slices.json -> speech.csv`。“一键完成音频三阶段”会用面板中的同一组参数顺序完成全部阶段；混合模式的“仅运行音频三阶段”行为相同。

任务卡实时显示阶段消息。静音切分会依次显示读取音频、检测边界和“正在写入第 N/总数个切片”；NeMo ASR 显示当前切片批次范围/总数，FunASR 显示当前切片序号/总数。一键流程的进度区间为 UVR5 `0%-55%`、16 kHz 准备约 `57%`、切分 `59%-68%`、ASR `68%-100%`；单独运行时每个阶段使用自己的 `0%-100%` 进度。

解压版固定读取程序目录内的 UVR5 资源：

```text
models\uvr\model_bs_roformer_ep_317_sdr_12.9755.ckpt
models\uvr\model_bs_roformer_ep_317_sdr_12.9755.yaml
```

Kaor 不读取外部 UVR5 安装目录。YAML 与 checkpoint 在第一次启动 UVR 阶段时从各自固定上游地址下载到上述目录；YAML 核对 `2273` 字节和固定 SHA-256，checkpoint 支持断点续传并核对 `639331213` 字节和固定 SHA-256。下载成功后可离线复用，推理由独立的 `KaorAudioWorker.exe` 执行。

选择源语言后，模型列表只显示该语言的专用 ASR。模型标记为“已下载”时直接从 `models/asr/<model-id>/` 加载；否则首次任务自动下载。当前列表覆盖日语、英语、中文、韩语、西班牙语、法语、德语和俄语。

输出为 `speech.csv`，并同步成为当前 `source.csv`。

### 4.3 混合模式

混合模式依次生成：

```text
ocr.csv
speech.csv
```

两份文件不会互相覆盖。OCR 适合确认画面实际写法、颜色、叠层和非语音文字；ASR 适合消除持续字幕的重复采样，并为同音字、漏字、时间和说话人提供第二份证据。

点击“融合校正”后，AI 会收到明确标记的 `OCR_CSV` 和 `SPEECH_CSV`。融合提示词要求它：

- 区分两种证据的典型错误，不盲从任一份。
- 合并重复 OCR，但保留真正同时出现的不同字幕。
- 修正渐入浅色帧、同音替换、标点、VAD 边界和时间漂移。
- 保留两个说话人、重叠组、轨道和层。
- 只返回源语言 `source.csv`，不翻译。

融合完成并校对后，再进入翻译步骤。

## 5. 音频识别选项

### 5.1 静音切分与 ASR 批量参数

静音切分的 RMS 和边界判断采用 GPT-SoVITS `slicer2.py` 风格，所有当前参数与默认值如下：

| 界面/API 参数 | 默认值 | 当前范围 | 作用 |
| --- | ---: | ---: | --- |
| `slicer_threshold_db` | `-34 dB` | `-100..0` | RMS 低于该值视为静音；数值越接近 `0`，静音判定越积极 |
| `slicer_min_length_ms` | `4000 ms` | API `100..600000` | 到达该片段长度后，才会在合格的中间静音处分段 |
| `slicer_min_interval_ms` | `200 ms` | `10..60000` | 触发中间切分所需的最短连续静音 |
| `slicer_hop_size_ms` | `10 ms` | `1..1000` | RMS 检测和边界搜索的时间步长 |
| `slicer_max_sil_kept_ms` | `500 ms` | `1..60000` | 每个切片边界最多保留的静音 |
| `slicer_max_length_ms` | `30000 ms` | `1000..600000` | 超过该长度的片段会在目标附近的低能量点再次切开，可在三阶段参数中调整 |
| `asr_batch_size` | `4` | `1..64` | NeMo ASR 每次处理的切片数；FunASR 当前逐片处理；不参与静音边界判断 |

参数关系必须满足：最短片段不小于最短静音，最短静音不小于检测步长，保留静音不小于检测步长，最长片段不小于最短片段。最长片段默认 `30000 ms`，可在 `1000-600000 ms` 范围调整。点击“恢复默认参数”会同时还原全部切分参数和 ASR 批大小。

NeMo/Parakeet 的 ASR 置信度使用解码器原生 token 最大概率，并在词内和 Cue 内取均值；不会把不可比较的负序列分数当成 `0..1` 置信度。结果写入兼容字段 `ocr_confidence`，低于 `0.85` 的 Cue 默认标记为待复核，供人工校对和 AI 融合重点检查。

### 5.2 计算设备

`自动` 会在 PyTorch 能访问 CUDA 时使用 NVIDIA GPU。OCR 的 PaddlePaddle 与音频的 PyTorch/NeMo 在不同子进程运行，以降低 Windows CUDA DLL 冲突风险。

### 5.3 时间边界精修

此选项不是再次 OCR，也不是凭空生成时间。它先使用 ASR 原生词/段时间戳，再读取 16 kHz 单声道人声 WAV 的短时能量，寻找语音开始和结束边缘，在有限窗口内微调 Cue。相邻非重叠字幕不会被精修到互相穿越；本来就重叠的说话保持重叠。

### 5.4 说话人分段

若 ASR 自带说话人标签，Kaor 优先保留。否则本地 VAD 和说话人嵌入模型会把语音切成发言段并聚类，再按时间重叠把 `speaker_id` 分配给 Cue。

说话人聚类表示“声音相似的一组”，不是自动获得角色真名。短句、音乐残留、多人抢话和音色接近时可能需要人工调整。OCR 模式还会把字幕颜色作为辅助线索。

## 6. CSV 文件和字段

项目目录通常包含：

```text
data\projects\<project_id>\ocr.csv
data\projects\<project_id>\speech.csv
data\projects\<project_id>\source.csv
data\projects\<project_id>\translated.csv
```

| 文件 | 何时出现 | 是否作为翻译输入 |
| --- | --- | --- |
| `ocr.csv` | 字幕/混合模式 OCR 后 | 否，作为证据 |
| `speech.csv` | 音频/混合模式 ASR 后 | 否，作为证据 |
| `source.csv` | 单源识别或融合校正后 | 是 |
| `translated.csv` | 翻译成功后 | 否，作为译文副本 |

主要字段：

| 字段 | 含义 |
| --- | --- |
| `cue_id` | 稳定字幕 ID |
| `start_ms` / `end_ms` | 毫秒时间边界 |
| `group_id` | 同时出现字幕的重叠组 |
| `layer` | 同组显示层 |
| `track_id` | 时间跟踪轨道 |
| `speaker_id` / `speaker_name` | 机器说话人标识和人工角色名 |
| `speaker_color` | `#RRGGBB` 字幕颜色 |
| `source_kind` | `ocr`、`speech`、`manual` 或 `imported` |
| `source_text` | 源语言文本 |
| `ocr_confidence` | 识别置信度；音频 Cue 也可用此兼容字段携带 ASR 置信度 |
| `target_text` | 译文 |
| `review_status` | 校对状态 |

文件使用 UTF-8 with BOM。不要手工改变表头或复用已有 `cue_id`。

## 7. 人工校对

字幕表顶部可切换“最终 / OCR / ASR”三种视图。最终视图可编辑，OCR 与 ASR 视图保留原始识别结果并只读；点击证据行会跳转视频时间，便于对照最终表。选中最终字幕后点击“框选帧 OCR”，在当前帧拖出临时矩形，松开后等待单帧识别，在确认窗口中可先修改候选文本，再确认覆盖该 cue 的 `source_text` 与置信度。

播放器左右方向键、上一帧/下一帧按钮按项目 FPS 移动到相邻帧；按住 `Shift` 仍移动约一秒。框选单帧 OCR 不会修改持久化的源字幕 ROI。

底部字幕工作区可拖动上边缘调高，也可展开；表格独立纵向滚动并固定表头。

建议按以下顺序检查：

1. 时间和重叠关系。
2. 重复 Cue、孤立 `0`/`1` 和无意义短文本。
3. 源文本与置信度。
4. 说话人和人物名。
5. 颜色、译文和状态。

### 新增字幕

把播放头定位到开始位置，点击“新增字幕”。新 Cue 默认持续约 2 秒，并继承当前选择的层、人物和颜色；随后直接编辑时间与文本。

### 删除字幕

选中错误 Cue，点击“删除字幕”并确认。若 `translated.csv` 已存在，新增、编辑和删除会同步维护译文表结构，避免两份表 ID 错位。

### 两人同时说话和后叠字幕

真正并行的两条内容应保留两个 Cue，并置于同一 `group_id` 的不同 `layer`。若只是同一句逐字补全或渐入，保留最终完整 Cue，删除短暂碎片。

## 8. 人物颜色和吸管

选中 Cue 后，可在右侧颜色编辑器输入十六进制色值或使用颜色选择器：

- `仅当前字幕`：只修改当前 `cue_id`。
- `该人物全部`：按 `speaker_id` 优先、人物名次之，修改当前项目中该人物所有 Cue。

吸管会跳到所选 Cue 对应帧，从视频画面取色。吸管结果默认只应用于当前字幕；确认人物身份后再点击“该人物全部”。这能避免颜色相同但人物不同、同一人物换色或特效字幕造成误改。

## 9. AI 接口、模型和思考强度

翻译设置支持：

- Base URL，例如 `https://host.example/v1`
- API key
- API path，通常为 `/chat/completions`
- 手动模型 ID
- 从上游 `/models` 获取模型列表并下拉选择
- 思考强度 `reasoning_effort`

融合和翻译任务右下角会出现 AI 实时浮窗。它显示任务阶段、批次、重试/拆批、已完成行数、上游 SSE 返回的 `reasoning_content`（若有）以及输出预览。部分中转站只返回最终 JSON，此时浮窗会显示阶段和输出状态，不代表模型提供了独立思考字段。

每个请求都会附带完整参考表：翻译请求使用完整 `source.csv`，融合请求同时使用完整 `ocr.csv` 与 `speech.csv`；当前批次仅限制输出行。遇到 Cloudflare `HTTP 524` 或读取超时，Kaor 会自动把输出批次二分并重试，避免一次请求超过中转站的 120 秒代理上限。
- 自定义请求头 JSON
- 超时、温度、批大小和上下文条数

中转站的 key 与官方服务 key 使用同一输入，但 Base URL、路径和请求头必须按中转站要求填写。先“获取模型”或“测试连接”，再保存配置。

### 9.1 本地翻译模型

翻译步骤切换到“本地模型”后有三种接入方式：

1. `一键部署`：自动检测硬件，选择 Qwen3 GGUF 与 llama.cpp CPU/Vulkan/CUDA 运行时，断点下载后核对固定字节数、SHA-256 和 GGUF 文件头，启动成功才切换翻译配置。
2. `本地 GGUF`：选择已有 `llama-server.exe` 和 `.gguf` 文件，由 Kaor 管理进程。
3. `已有服务`：接入正在运行的 Ollama、LM Studio 或其他 OpenAI 兼容本地地址。

推荐不是强制选择。上下文越大、模型越大，占用的内存或显存越多。默认 8K 上下文优先保证稳定；显存不足时选择更小模型。AMD 包的本地翻译通过 Vulkan 使用 AMD GPU，但 OCR、UVR5、ASR 和说话人处理固定使用 CPU，这是当前 Windows AMD 发行版的能力限制。NVIDIA 包的 OCR 与音频可使用 CUDA，CPU 包全程可在无独显机器运行。

本地模型只监听 `127.0.0.1`。一键部署需要联网下载 llama.cpp 运行时和所选模型；完成后融合与翻译请求只在本机回环地址传递。llama.cpp 使用 MIT License，其许可证文本随 Kaor 保留；官方 Qwen3 GGUF 使用 Apache-2.0，一键部署会从所选模型的同一固定 revision 下载 `LICENSE`，保存到本地模型的 `models/licenses/` 目录并写入模型 manifest。许可证下载失败时部署不会被标记为完成。

<a id="runtime-log-repair"></a>

### 9.2 运行日志和维修指南

顶栏终端图标切换到运行日志页。日志每 1.5 秒刷新，可按日志源、`CRITICAL/ERROR/WARNING/INFO/DEBUG` 和关键词过滤。点击一行可展开完整正文并复制；命中已知错误时，右侧只列出相关维修条目。`导出诊断包` 会生成日志、系统摘要和维修规则 ZIP，并遮蔽常见 API Key、Bearer Token 与访问令牌。

每个维修条目都可打开包内的[完整故障排查文档](TROUBLESHOOTING.zh-CN.md#repair-guide-index)并定位到对应错误；故障文档也会链接回本节说明。

## 10. 融合与翻译分离

AI 融合只消费 `ocr.csv` 和 `speech.csv`，输出源语言 `source.csv`。AI 翻译只消费校对后的 `source.csv`，输出 `translated.csv`。二者可以使用同一套接口配置，但任务、提示词和结果文件不同。

融合按钮会先区分两类证据问题：

- `missing=['ocr.csv']` 或 `missing=['speech.csv']`：文件本身尚未生成，表示相应 OCR/音频识别分支未完成，或工作区重置已经移除旧结果。
- `empty=['ocr.csv']` 或 `empty=['speech.csv']`：文件已经存在，但固定表头后没有任何字幕行。此时检查 OCR ROI、源语言、音频内容、静音切分和相应任务结果，再重跑该分支。

只有两份 CSV 都存在且各自至少有一行 Cue 时，融合任务才会提交给 AI。

任务复用与断点文件位于项目 `cache/`：`task-state.json` 保存源文件与参数指纹，`ocr-checkpoint.json` 保存 OCR 最近快照，`audio/asr-checkpoint.json` 保存已完成音频切片，`ai/*-checkpoint.json` 保存 AI 已完成行。重新提交相同参数时，完整 CSV/WAV/切片会直接复用；缺失或损坏时从首个不完整阶段继续。修改字幕参数后可使用“重置工作区”清理旧状态。

翻译上下文可以包括文件标题、源文件名、故事简介、语气、人物资料和术语表。置信度会随每条 Cue 发送，模型被要求重点检查低置信度行。

翻译提示词禁止以下内容出现在 `target_text`：

```text
/N
\N
字面量 \n
真实换行
<br> / <br/>
```

即使上游仍返回这些标记，Kaor 也会在解析时替换为空格并压缩多余空白。换行由本地排版器根据译文区、字号和最大行数处理。

## 11. 预览和导出

先生成短预览，检查：

- 译文没有覆盖原字幕或重要画面。
- 两人同时说话时不同 `layer` 没有互相压住。
- 人物颜色正确。
- 字号、描边、最大行数和每行字符数适合译文区。

导出使用 ASS 和 FFmpeg，不修改原视频。项目导出文件位于：

```text
data\projects\<project_id>\exports\
```

## 12. 模型和网络

本地运行不等于永不联网：第一次运行 UVR5 会从固定上游地址下载匹配 YAML 和约 610 MiB 的 BS-Roformer checkpoint；所选语言的 ASR 模型首次缺失时会自动下载；发行包未携带 NeMo 说话人模型时，首次需要说话人聚类也会下载；一键部署本地翻译时还会下载固定 llama.cpp 运行时和用户选择的 GGUF。checkpoint 下载支持断点续传，完整性检查失败的文件不会被启动。

从源码创建 NVIDIA 环境时，PyTorch 必须显式走官方 CUDA 12.6 索引；若 `pip` 默认使用不含 GPU wheel 的镜像，请先执行：

```powershell
.\.venv-nvidia-cu126\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0+cu126 torchaudio==2.11.0+cu126 torchvision==0.26.0+cu126
.\.venv-nvidia-cu126\Scripts\python.exe -m pip install -r requirements-nvidia-cu126.txt
```

会发送给 AI 的只有融合/翻译所需的 CSV 和用户选择的项目上下文，不会上传视频、音频或图像帧。使用中转站时，应单独确认其日志和数据保留策略。

## 13. 三套发行包的边界

| 包 | 内置推理设备 | 说明 |
| --- | --- | --- |
| CPU | OCR CPU、音频 CPU、本地翻译 CPU | 兼容性优先 |
| AMD | OCR CPU、音频 CPU、本地翻译 Vulkan | Windows PaddleOCR 与当前 PyTorch 音频链不宣称 AMD GPU 加速 |
| NVIDIA | OCR CUDA 12.6、音频 CUDA 12.6、本地翻译 CUDA | 需要兼容驱动，不要求安装 CUDA Toolkit |

三套包均包含程序启动与固定流水线所需的 Python 运行时、库和二进制。BS-Roformer YAML 与 checkpoint、语言专用 ASR 与可选本地翻译模型按需下载，不属于“额外配置环境”；下载完成后由 Kaor 自行管理。用户不需要执行任何 Python 或包管理命令。每套发行资产提供外部 SHA-256，包内另有逐文件清单。

## 14. 开源许可证

Kaor 源代码采用 MIT License。静音切分器的 RMS 与静音边界判断改编自 GPT-SoVITS `tools/slicer2.py`，Copyright (c) 2024 RVC-Boss，按 MIT License 使用；Kaor 的 WAV I/O 与 `slices.json` 清单处理为本地实现。保留的许可证文本位于 `licenses/GPT-SoVITS-MIT.txt`。llama.cpp、Qwen3 和 BS-Roformer 的许可与来源边界见 `THIRD_PARTY_NOTICES.md` 及 `licenses/`；Kaor 的 MIT 不会重新许可第三方模型与运行时。
