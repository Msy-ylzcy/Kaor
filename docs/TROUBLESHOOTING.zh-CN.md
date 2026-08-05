# Kaor 中文故障排查

本文适用于 Windows x64 CPU、AMD 和 NVIDIA 三套解压即用包及源码环境。发行包已经包含 Python 与运行库，普通用户不需要执行 `pip install` 或配置虚拟环境。先在 WebUI 顶栏打开“运行输出”，按报错行关联的维修指南处理；不要只凭任务管理器利用率判断实际设备。

日志页的筛选、复制和诊断包操作见[使用手册：运行日志和维修指南](USER_GUIDE.zh-CN.md#runtime-log-repair)。

## 1. 先收集状态

```powershell
$Base = "http://127.0.0.1:8765"
Invoke-RestMethod "$Base/api/health" | ConvertTo-Json -Depth 8
Invoke-RestMethod "$Base/api/ocr/capabilities" | ConvertTo-Json -Depth 12
Invoke-RestMethod "$Base/api/audio/capabilities" | ConvertTo-Json -Depth 12
$Workspace = Invoke-RestMethod "$Base/api/workspace"
$ProjectId = $Workspace.project.id
Invoke-RestMethod "$Base/api/jobs?project_id=$ProjectId" | ConvertTo-Json -Depth 16
```

异步任务接口先返回任务对象。真正错误位于 `/api/jobs/<job-id>` 的 `status`、`message` 和 `error.detail`。排查时保留完整错误，不只截取页面第一行。

<a id="repair-guide-index"></a>

WebUI 日志页可按来源、级别和关键词筛选，点击日志行展开完整正文，右侧显示匹配的处理步骤。“导出诊断包”会收集应用日志、音频 Worker 日志、llama-server 日志和系统摘要，并遮蔽常见 API Key/Token。每个维修卡片的“打开完整排障文档”会定位到下列对应章节。主要精确匹配如下：

| 日志包含 | 维修条目 |
| --- | --- |
| `PaddleOCR is not installed` / `No module named 'paddle'` | OCR 运行时缺失，重新完整解压对应版本并检查安全软件隔离区 |
| `CUDA was requested` / `no kernel image` / `cudnn` | 驱动或 CUDA 发行包不匹配 |
| `out of memory` / `failed to allocate` | 降低 OCR/ASR 批量、本地模型或上下文 |
| `bundled UVR model` / `UVR model size mismatch` | 包内 BS-Roformer 缺失或损坏 |
| `KaorAudioWorker.exe` / `worker exited without a result` | 音频 Worker 缺失、被拦截或崩溃 |
| `PyTorch is required` / `No module named 'torch'` / `audio-separator is required` / `nemo` / `funasr` | 音频推理模块缺失或发行包目录被混用 |
| `ffmpeg was not found` / `Invalid data found` | FFmpeg 或片源异常 |
| `HTTP 401` / `HTTP 403` | API Key、权限或认证头错误 |
| `HTTP 429` | 上游限流或余额限制 |
| `HTTP 524` / `origin_response_timeout` | 中转站等待上游超时，程序会缩小输出批次续跑 |
| `Expecting value: line 1 column 1` / `response was not valid JSON` | 空响应、HTML 错误页或模型未返回 JSON |
| `fusion requires completed OCR and speech CSV` | `ocr.csv` 或 `speech.csv` 未完成 |
| `llama-server` / `GGUF` / `Vulkan` / `local model port` | 本地翻译运行时、模型、驱动或端口问题 |
| `SHA-256 mismatch` / `invalid GGUF header` | 下载被截断或内容被代理替换 |
| `NameResolutionError` / `ConnectError` / `connection reset` / `model download failed` / `huggingface` | DNS、代理或模型仓库连接失败；保留 `.part` 后恢复网络重试 |
| `No space left on device` | 磁盘空间不足 |
| `PermissionError` / `Access is denied` | 目录权限、文件占用或安全软件拦截 |

<a id="permission-denied"></a>

## 2. 本地服务或页面打不开

解压版双击 `Kaor.exe`。源码环境从项目根目录启动：

```powershell
$env:KAOR_DATA_DIR = "$PWD\data"
$env:KAOR_NO_BROWSER = "1"
.\.venv-nvidia-cu126\Scripts\python.exe -m backend.main
```

然后访问终端显示的地址。常见原因：

- `8765` 被占用：查看启动日志中的实际顺延端口，或设置 `KAOR_PORT`。
- 只打开前端 `index.html`：必须让后端提供 `/api`，不要直接打开静态文件。
- 页面还是旧版本：运行 `npm.cmd run build --prefix apps/web` 后重启后端并强制刷新浏览器。
- 双击后没有窗口：查看程序目录的 `startup-trace.log` 和 `data/logs/kaor.log`；确认文件已完整解压且 `Kaor.exe`、`KaorAudioWorker.exe`、`_internal/` 均未被安全软件隔离。
- 出现 `Unable to configure formatter 'default'` 或 `NoneType ... isatty`：正在使用历史包。当前启动器关闭 Uvicorn 控制台 formatter，重新下载最新 Release 并解压到新目录。

### 2.1 NVIDIA 分片 Setup 失败

NVIDIA 版必须把 Setup、`parts.json` 和从 `.zip.001` 开始的全部分片放在同一目录。
无界面校验或解压失败时，查看 Setup 旁边的
`Kaor-Windows-x64-NVIDIA-Setup.error.log`：

- `Release part is missing`：补下载错误中点名的分片。
- `size mismatch` / `SHA-256 mismatch`：删除被点名的损坏分片，只从同一个 Release 重新下载。
- `Assembled archive SHA-256 mismatch`：清单与分片混用了不同版本，重新下载整套资产。
- `Not enough free disk space`：把整套分片移到空间更大的磁盘后重试。
- `The destination already exists`：先重命名现有 `Kaor-Windows-x64-NVIDIA/`；Setup 不覆盖已有目录。

Setup 不依赖 Python、CUDA Toolkit 或第三方压缩软件。完整文件清单、命令行校验方法和
全部 Setup 报错见 [Windows 发行说明](RELEASE_BUILD.zh-CN.md#nvidia-分片使用方法)。

<a id="ocr-runtime-missing"></a>

## 3. OCR 未安装、卡在 2% 或使用 CPU

检查：

```powershell
.\.venv-nvidia-cu126\Scripts\python.exe -m pip show paddlepaddle-gpu paddleocr paddlex
Invoke-RestMethod "$Base/api/ocr/capabilities" | ConvertTo-Json -Depth 12
```

`PaddleOCR is not installed` 表示当前运行时缺少 OCR 包。解压版出现该错误通常是压缩包未完整解压、文件被安全软件隔离或发行资产损坏，先核对 Release SHA-256，再重新完整解压；不要在解压版目录里用 `pip` 补包。源码环境才需要检查是否误用了系统 Python，而不是项目虚拟环境。

AMD 发行包显示 OCR `CPU` 属于设计结果，不是回退故障。Windows PaddleOCR 当前发行链没有通用 AMD GPU 后端；AMD 版的 OCR、UVR、ASR 和说话人处理固定走 CPU，AMD GPU 只用于本地翻译的 Vulkan 后端。CPU 包显示 CPU 也属于预期。NVIDIA 包才应在 OCR 能力接口中显示 CUDA。

卡在约 2% 通常处于运行时和模型加载阶段。模型缺失时可能联网下载；模型已存在时检查任务 `error.detail`、磁盘、杀毒软件和模型目录。

GPU 利用率不持续 100% 不等于走 CPU。视频解码、ROI 裁剪、变化检测和 CSV 合并会占用 CPU；静止字幕还会复用结果。任务管理器查看 `CUDA`/`Compute` 图表，并结合任务快照中的 `device`、`batch_size`、`ocr_frames`、`reused_frames`、`decode_seconds` 和 `ocr_seconds`。

<a id="gpu-out-of-memory"></a>

### 3.1 自动批次、16 GB 显卡与 OOM 回退

OCR 的“自动”批处理会读取 PaddlePaddle 报告的设备总显存，选择“外层视频帧批次/文本识别批次”：

| 设备或总显存 | 自动批次 |
| --- | ---: |
| CPU | `2/4` |
| GPU 少于 6 GiB | `6/24` |
| GPU 6 至不足 8 GiB | `12/32` |
| GPU 8 至不足 10 GiB | `24/48` |
| GPU 10 至不足 14 GiB | `32/64` |
| GPU 14 至不足 20 GiB | `40/80` |
| GPU 20 GiB 及以上 | `48/96` |

厂商标称 16 GB 的显卡通常会落入 14 GiB 以上档位，因此自动外层帧批次为 `40`。任务卡里的“批 40”表示初始批次；画面复杂度、ROI 尺寸和高精度共识仍会影响峰值显存。

识别批次命中 `out of memory`、`CUDA_ERROR_OUT_OF_MEMORY`、`resource exhausted` 或显存分配失败时，Kaor 会清理 GPU 缓存，并把当前失败批次递归二分后继续。该回退只拆分当前批次，不会永久改写用户配置；快照中的 `ocr_batch_backoffs` 是拆分次数，`effective_ocr_batch_size` 是实际成功的最小有效批次。若单帧仍报显存错误，缩小 ROI、关闭高精度共识或手动降低批次后重跑。

### 3.2 OCR 帧进度长时间不变

当前 OCR 会先计算预计总采样帧数，每完成一个批次就按 `processed_frames / sampled_frames_total` 更新，并在任务卡显示 `采样 A/B`；最终还会发布 `OCR complete` 快照。排查时以 `A/B`、`updated_at`、`ocr_frames`、`reused_frames` 和 `ocr_seconds` 是否继续变化为准。`A/B` 持续变化表示任务正在运行；数值和任务更新时间都长期不变时，再查看 `error.detail`、显存和 worker 状态。

## 4. OCR 重复、出现 0/1、渐入误识别

按顺序处理：

1. 缩紧原字幕 ROI，排除弹幕、台标和播放器 UI。
2. 保持高精度共识和噪声过滤开启。
3. 对短字幕提高采样率，对长驻字幕不要盲目提高采样率。
4. 检查实时快照中的 `reused_frames` 是否增长。
5. 在校对表删除残留的孤立数字或无意义短 Cue。

当前流水线会延后判断渐入字幕、定期复查长驻字幕、按文本/位置/时间合并轨道，并过滤低置信度短 ASCII。特效、闪烁或多个字幕区域同时存在时仍可能需要人工校对或改用混合模式。

<a id="uvr-model-damaged"></a>

## 5. UVR5 不可用或找不到模型

查看：

```powershell
$Caps = Invoke-RestMethod "$Base/api/audio/capabilities"
$Caps.uvr_model | ConvertTo-Json -Depth 8
```

解压版固定需要：

```text
models\uvr\model_bs_roformer_ep_317_sdr_12.9755.ckpt
models\uvr\model_bs_roformer_ep_317_sdr_12.9755.yaml
```

程序不再读取外部 UVR5 路径。checkpoint 必须为 `639331213` 字节，SHA-256 为 `5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa`；文件不符时重新完整解压发行包。

常见错误：

- `UVR model not found`：checkpoint 路径不对。
- `UVR model size mismatch`：文件下载不完整或不是预期模型。
- `UVR model config not found`：缺少对应 YAML。
- `audio-separator is required`：解压版的内部 Python 依赖缺失或被隔离；核对发行 SHA-256 并重新完整解压，不要在包内运行 `pip install`。源码环境则重新安装对应 requirements。
- `vocal separation produced an invalid WAV file`：检查 FFmpeg、磁盘空间和源音轨。

Kaor 直接使用 UVR5 推理核心和本地 checkpoint，不调用 UVR 图形界面。

<a id="audio-worker-missing"></a>

### 5.1 导入音轨和三个独立阶段

视频导入后应立即存在：

```text
data\projects\<project_id>\cache\audio\mix.wav
```

导入提示音轨提取失败时，先检查 FFmpeg stderr、视频是否含音频流、磁盘空间和源文件可读性。视频会继续保留；启动 UVR5 时若 `mix.wav` 缺失或小于有效 WAV 头，Kaor 会从原视频重新提取一次。

三个独立按钮有明确的前置产物：

- UVR5：读取 `mix.wav`，生成 `vocals.wav`。
- 静音切分：要求 `vocals.wav`；缺失时返回 `vocals.wav is missing; run UVR5 vocal separation first`。
- ASR 打标：要求 `speech-16k-mono.wav` 和 `slices/slices.json`；缺失时返回 `audio slices are missing; run silence slicing first`。

“一键完成音频三阶段”会顺序补齐全部文件。单独重跑 UVR5 后，旧切片仍在磁盘上，但内容已经与新人声文件失配，应继续重跑静音切分和 ASR；单独重跑切分后应继续重跑 ASR，才能更新 `speech.csv`。

任务消息会持续给出当前阶段和片段进度：切分阶段显示写入切片 `N/总数`，NeMo ASR 显示切片批次范围/总数，FunASR 显示切片 `N/总数`。worker 进度文件约每 250 毫秒读取一次，WebUI 约每 400 毫秒刷新任务卡。

### 5.2 旧混合任务停在 85% 或完成后没有 `speech.csv`

任务卡显示 `Transcribing separated speech with native confidence` 且停在约 `85%`，说明 OCR 和 UVR5 已经结束，旧流程正在等待 ASR。已确认的失败样本把约 536 秒人声作为一个完整 WAV 一次交给 NeMo；单次 `transcribe` 返回前没有中间进度，最终还可能返回 0 条 Cue，使旧任务错误显示完成或只留下空表头 CSV。

当前流程会先按静音边界切片，并按可调的 `max_length_ms` 再次拆分过长片段，再按 `asr_batch_size` 批量交给 NeMo。该失败样本现在得到 105 个短片，最长约 8.64 秒；任务消息会持续显示当前批次范围。ASR 返回 0 条 Cue 会把任务标记为失败，且不会用空结果覆盖原有 `speech.csv`。升级后已经存在的空表不会自动补内容，需要重新运行“静音切分”和“ASR 打标”，或重新运行一键音频三阶段。

### 5.3 静音切分参数报错或切得不合适

默认值为 `-34 dB`、最短片段 `4000 ms`、最短静音 `200 ms`、检测步长 `10 ms`、保留静音 `500 ms`，最长片段默认 `30000 ms`，并可在界面调整。参数需满足：

```text
min_length_ms >= min_interval_ms >= hop_size_ms
max_sil_kept_ms >= hop_size_ms
max_length_ms >= min_length_ms
```

- 切片过碎：降低阈值，例如从 `-34` 调到 `-40`，或增大最短片段/最短静音。
- 多句话粘成一片：提高阈值，例如从 `-34` 调到 `-28`，或减小最短片段/最短静音。
- 词头词尾被切：增大保留静音。
- 边界太粗：减小检测步长，但会增加 RMS 计算量。
- `max_length_ms must be at least min_length_ms`：最长片段小于最短片段；把最长片段调到不小于最短片段（可用 `1000-600000 ms`）。

切分完成后先确认任务结果的 `slice_count` 大于 0，或 `slices.json` 的 `slices` 数组非空，再运行 ASR。`ASR worker completed with 0 speech cues` 表示切片存在但没有得到有效文本，应试听 `vocals.wav` 和若干 `slice-*.wav`，再核对语言/模型与阈值。

<a id="cuda-unavailable"></a>
<a id="pytorch-audio-runtime"></a>

## 6. PyTorch、CUDA 或音频任务使用 CPU

先确认发行包边界：

| 包 | OCR | UVR / ASR | 本地翻译 |
| --- | --- | --- | --- |
| CPU | CPU | CPU | CPU |
| AMD | CPU | CPU | AMD Vulkan |
| NVIDIA | CUDA 12.6 | CUDA 12.6 | CUDA |

AMD 的 PyTorch 音频链使用 CPU wheel；这是当前 Windows AMD 发行版的明确限制，也保证普通 AMD 机器解压可用。把 `torch.cuda.is_available()` 为 `False` 当作 AMD 包故障是不正确的。WebUI 本地模型面板分别显示 OCR、音频和本地翻译的实际设备。

### 6.1 三阶段参数长期显示“等待本地模型目录”

语言和 ASR 模型列表本身只需要扫描本地 `models/asr/`，不需要加载推理模型。旧版首屏却把该列表与 PaddleOCR、PyTorch/CUDA 两个完整能力探测放在同一个等待组中；任一运行时冷启动较慢，音频下拉框也会一直空着。

当前前端会先独立请求轻量的 `/api/audio/models`，随后分别更新 OCR 与音频运行时状态。正常情况下模型列表应立即出现；若仍为空，直接检查：

```powershell
Invoke-RestMethod "http://127.0.0.1:8765/api/audio/models" | ConvertTo-Json -Depth 6
```

该接口能返回列表而 `/api/audio/capabilities` 较慢时，只代表 Torch/CUDA 正在冷启动，不影响选择语言和模型。两个接口都失败时，检查后端是否已重启到当前版本以及浏览器控制台中的 HTTP 状态。

检查音频子进程能力：

```powershell
Invoke-RestMethod "$Base/api/audio/capabilities" | ConvertTo-Json -Depth 12
.\.venv-nvidia-cu126\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

关键字段应显示 `torch_available: true`、`cuda_available: true` 和正确显卡名称。若使用了 CPU 版 torch，重新从 PyTorch 官方 CUDA 索引安装匹配版本；国内通用 PyPI 镜像通常没有所需 GPU wheel。

音频任务不会始终跑满 GPU：FFmpeg 解码、WAV 读写、UVR 分块拼接、ASR 前后处理、时间精修和 RTTM/Cue 合并均包含 CPU 或 I/O 阶段。判断路径以能力接口和任务消息为准，不以单一利用率百分比为准。

### 6.2 `pip check` 报 cuDNN 9.5/9.9 冲突

当前 Paddle CUDA 12.6 wheel 的包元数据声明 `nvidia-cudnn-cu12==9.5.1.17`，但实际加载的 Paddle 二进制会报告它由 cuDNN 9.9 编译。Kaor 的 NVIDIA 开发环境因此固定 `9.9.0.52`，以真实二进制兼容性为准；`pip check` 会显示这条已知上游元数据冲突。

不要为消除该提示单独降级到 9.5。实测降级会让 PaddleOCR 间接探测 Torch 时出现 `WinError 127` 或 cuDNN 版本警告。Kaor 在导入 PaddleOCR 时会屏蔽 ModelScope 非必要的 Torch 分布式探测，并把真正的 UVR5/ASR/说话人模型放到独立音频子进程，避免两套 DLL 在同一进程加载。

核验实际运行路径：

```powershell
.\.venv-nvidia-cu126\Scripts\python.exe -m backend.audio_worker probe
Invoke-RestMethod "$Base/api/ocr/capabilities" | ConvertTo-Json -Depth 12
```

<a id="network-download"></a>

## 7. 语言 ASR 模型下载失败

选择音频/混合模式后，Kaor 根据语言和模型 ID 检查 `models/asr/<model-id>/`。未安装时自动从模型仓库下载。

常见错误：

- `no language-specific ASR model is configured`：选择了列表外语言。
- `ASR model ... is for ..., not ...`：语言与模型不匹配。
- `huggingface_hub is required`：环境缺少下载器。
- `ASR model download failed`：网络、代理、磁盘或模型仓库访问失败。
- `downloaded NeMo model has no .nemo checkpoint`：下载目录不完整。

处理步骤：

1. 保持 VPN/代理可用并重试。
2. 检查磁盘空间和 `models/asr/` 写入权限。
3. 删除不完整的单个模型目录后重新开始该模型下载，不要清空其他已完成模型。
4. 下载后确认目录存在 `.kaor-model.json` 和模型权重。

## 8. 时间精修没有变化或边界不准

时间边界精修依赖可读的 16 kHz 单声道人声 WAV 和 ASR 初始时间戳。任务提示 `Using native ASR timestamps (audio refinement unavailable)` 时，表示能量精修未取得可用结果，系统保留 ASR 原生时间。

可能原因：

- 人声分离结果接近静音。
- 背景音乐残留使能量边界模糊。
- Cue 太短或相邻语句间隔太小。
- 语音重叠，单一能量包络无法区分两人。

先听取分离后人声文件，再人工修正少数时间边界。重叠 Cue 会被保留，不会强制串行化。

## 9. 说话人没有区分或区分错误

说话人来源按优先级包括 ASR 原生标签、本地 VAD/说话人嵌入聚类和 OCR 字幕颜色。它们得到的是稳定分组，不是角色真名。

常见失败场景：短促语气词、两人音色接近、多人同时说话、UVR 残留音乐、变声和极短视频。处理方式：

- 保留 `diarization` 开启并确保本地说话人模型存在。
- 音频很差时使用混合模式，让 OCR 颜色辅助判断。
- 在表格修改 `speaker_name`。
- 选中一个确认无误的 Cue，用“该人物全部”统一颜色。
- 个别例外使用“仅当前字幕”，不要批量覆盖。

如果说话人处理失败，Kaor 应保留空身份或标记待复核，而不是伪造角色名。

<a id="fusion-input-missing"></a>

## 10. 混合模式或 AI 融合失败

先确认两份证据存在且都有字幕行：

```powershell
Get-Item ".\data\projects\$ProjectId\ocr.csv"
Get-Item ".\data\projects\$ProjectId\speech.csv"
@(Import-Csv ".\data\projects\$ProjectId\ocr.csv").Count
@(Import-Csv ".\data\projects\$ProjectId\speech.csv").Count
```

后端会把缺文件与空 CSV 分开报告：

- `fusion requires completed OCR and speech CSV files; missing=[...]`：对应文件未生成。完成缺少的 OCR 或音频分支；工作区刚重置时也会出现此错误。
- `fusion evidence CSV files contain no subtitle rows; empty=[...]`：文件存在但只有表头。检查 OCR ROI、字幕语言、`vocals.wav`、切分结果和 ASR 模型后重跑相应分支；新建同名空文件不会满足融合条件。

两份文件均有 Cue 后，融合失败的其他常见原因：

- 只完成了一个分支。
- AI 接口未配置或模型不支持要求的 JSON 输出。
- 上游截断了包含两份 CSV 的响应。
- 返回了译文而不是源语言修正表。
- 输出 Cue 缺字段、ID 重复或时间无效。

融合与翻译是两项任务。融合成功只更新 `source.csv`；看不到译文属于正常流程，随后还要单独点击翻译。

<a id="translation-auth"></a>
<a id="translation-rate-limit"></a>
<a id="translation-invalid-json"></a>

## 11. API 连接失败和空响应

错误 `translation request failed: Expecting value: line 1 column 1 (char 0)` 通常表示上游返回空正文、HTML 错误页、网关文本或流式事件，而不是预期 JSON。

当前诊断应包含 HTTP 状态、响应 `Content-Type` 和截断后的响应预览。重点检查：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 401 | key 错误、过期或认证头被覆盖 | 重填 key，检查自定义 `Authorization` |
| 403 | 模型权限、余额、地区或中转站策略 | 在上游控制台核对权限 |
| 404 | Base URL、API path 或模型 ID 错 | 拆分 Base URL 与 `/chat/completions` |
| 429 | 额度或速率限制 | 等待、充值或减小批次 |
| 524 | Cloudflare/中转站代理在约 120 秒内没有收到上游完整响应 | Kaor 会自动缩小当前输出批次并重试；确认 API 端点、模型响应速度和网络代理均正常 |
| 5xx | 中转站/上游故障 | 查看响应预览并重试 |
| HTML/空正文 | 反向代理、WAF、登录页或端点错误 | 用最小请求直接测试端点 |
| 超时 | 批次过大或网络慢 | 增加超时、减小 batch |

<a id="translation-timeout"></a>

### 11.1 HTTP 524 的具体处理

`HTTP 524` 表示请求已经连到代理，但代理等待上游响应超过约 120 秒；它通常是单次 AI 请求过大或模型生成太慢，不是 OCR/ASR 文件损坏。Kaor 会保留完整参考表，只把“本次要求返回的字幕行”递归二分，例如 `80 -> 40 -> 20`，每个成功子批次立即写入项目 `cache/ai/` 下的 checkpoint。重新提交同一任务时会跳过已完成行。

建议顺序：

1. 先等待自动拆批完成，并在 AI 实时浮窗中观察重试/拆批事件。
2. 仍失败时把 AI 批大小调小、关闭不必要的长背景说明，并用最小 `chat/completions` 请求验证中转站返回的是 JSON。
3. 如果拆到单行仍收到 524，问题在上游模型或代理的响应时限；更换模型/端点后重新提交即可复用已有 checkpoint。

最小测试：

```powershell
$ApiKey = [System.Net.NetworkCredential]::new("", (Read-Host "API Key" -AsSecureString)).Password
$Endpoint = "https://host.example/v1/chat/completions"
$Headers = @{ Authorization = "Bearer $ApiKey"; "Content-Type" = "application/json" }
$Body = @{ model = "MODEL_ID"; messages = @(@{ role = "user"; content = "Return JSON: {`"status`":`"ok`"}" }) } | ConvertTo-Json -Depth 8
Invoke-WebRequest $Endpoint -Method Post -Headers $Headers -Body $Body | Select-Object StatusCode, Headers, Content
$ApiKey = $null
```

## 12. 从上游获取模型列表失败

模型下拉框使用兼容 `/models` 端点。常见错误包括端点不存在、返回结构不是 `{data:[...]}`/`{models:[...]}`、需要不同认证头，或中转站禁止列举模型。

获取失败时仍可手动填写模型 ID。不要把模型展示名当作模型 ID。思考强度只在上游支持 `reasoning_effort` 时填写；不支持时留空。

## 13. 翻译成功但译文没有显示

依次检查：

```powershell
Get-Item ".\data\projects\$ProjectId\translated.csv"
Import-Csv ".\data\projects\$ProjectId\translated.csv" | Select-Object -First 5 cue_id,source_text,target_text,review_status
Invoke-RestMethod "$Base/api/workspace" | ConvertTo-Json -Depth 12
```

可能原因：

- 页面仍显示源字幕层或旧工作区快照。
- 上游返回的 `target_text` 为空。
- 返回 `cue_id` 与源表不匹配，任务被拒绝写入。
- 翻译后又运行识别/融合，当前 `source.csv` 结构发生变化。

刷新工作区前先确认任务状态为 `completed`，并检查 `translated.csv` 的 `target_text`。

## 14. `/N`、`\\N` 或换行没有生效

Kaor 的译文不是由 AI 决定手工换行。提示词明确禁止 `/N`、`\\N`、字面 `\\n`、真实换行和 HTML `<br>`；解析器会把残留标记替换为空格。

若旧 `translated.csv` 已经包含这些内容，需要重新翻译，或在校对表手工改一次使后端保存清洗后的单行文本。最终 ASS 的换行由本地排版参数和译文 ROI 负责。

## 15. 新增、删除或改色没有同步

新增/删除 Cue 会同步维护 `source.csv` 与已存在的 `translated.csv`。若操作失败：

- 检查 `cue already exists`：手工新增 ID 与现有 ID 冲突。
- 检查 `cue not found`：页面选中的是旧快照中的 Cue。
- 检查 `translated cues must match source cue IDs`：两份表此前已被外部工具改乱。
- 检查 `speaker_id or speaker_name is required`：使用“该人物全部”前还没有人物标识或姓名。
- 检查颜色格式是否严格为 `#RRGGBB`。

“仅当前字幕”只更新一个 `cue_id`；“该人物全部”按 `speaker_id` 优先匹配，否则按人物名匹配。吸管默认只改当前 Cue。

## 16. CSV 缺失、404 或表头错误

检查：

```powershell
Get-ChildItem ".\data\projects\$ProjectId" -Filter *.csv | Select-Object Name,Length,LastWriteTime
(Get-Content ".\data\projects\$ProjectId\source.csv" -Encoding UTF8 -TotalCount 1)
```

固定表头：

```text
cue_id,start_ms,end_ms,group_id,layer,track_id,speaker_id,speaker_name,speaker_color,source_kind,source_text,ocr_confidence,target_text,review_status
```

- `ocr.csv` 只在 OCR 分支完成后存在。
- `speech.csv` 只在音频分支完成后存在。
- `source.csv` 是当前最终源表。
- `translated.csv` 只在翻译成功后存在。

不要用 Excel 改列名、改变列顺序或把毫秒时间保存成科学计数法。

## 17. 字幕框太小或无法滚动

拖动底部工作区上边缘调高，点击展开按钮获得更大校对空间。表格本身可以纵向滚动并固定表头。若仍显示旧的小面板，重新构建前端、重启服务并强制刷新浏览器缓存。

<a id="ffmpeg-runtime"></a>
<a id="disk-space"></a>

## 18. FFmpeg、预览和导出失败

```powershell
& .\bin\ffmpeg.exe -version
& .\bin\ffmpeg.exe -hide_banner -filters | Select-String -Pattern '\bass\b'
& .\bin\ffmpeg.exe -hide_banner -i "D:\Videos\input.mp4" 2>&1 | Select-Object -First 80
```

常见错误：

- `ffmpeg was not found`：把 `ffmpeg.exe` 放入 `bin/` 或加入 `PATH`。
- `project video is missing`：项目媒体副本被移动或数据目录变更。
- `project has no subtitles`：`source.csv` 没有 Cue。
- `ffmpeg render failed`：查看任务 `error.detail` 中完整 stderr。
- MP4 音频 copy 失败：先把源音频转为 AAC。

译文遮挡画面时，调整译文 ROI；两条字幕挤在一起时，扩大 ROI 高度并检查 `group_id` 与 `layer`。

## 19. 任务排队、取消和 404

```powershell
Invoke-RestMethod "$Base/api/jobs?project_id=$ProjectId" | Format-Table id,kind,status,progress,message
Invoke-RestMethod "$Base/api/jobs/JOB_ID/cancel" -Method Post -ContentType "application/json" -Body "{}"
```

OCR、音频、混合、融合、翻译和导出都可能排队。进程重启后，旧内存任务 ID 可能返回 `404 job not found`；项目 CSV 和已完成输出仍保留在磁盘。

### 19.1 工作区重置返回 409 或音轨警告

重置会保留原视频和项目设置，删除 `cache/`、`exports/`、四份 CSV 的内容/文件以及项目任务记录，然后重新探测片源并提取 `mix.wav`。常见结果：

- `cancel or wait for active jobs before resetting the workspace`：仍有运行中或排队任务；先取消或等待结束。
- `project video is missing`：项目清单还在，但注册片源已经被移动或删除；恢复 `manifest.json` 指向的文件后再重置。
- “工作区已清空，但音轨重新提取失败”：字幕、缓存和导出已经清理，片源仍保留；修复 FFmpeg/音轨问题后再次重置或运行 UVR5。

重置成功后看到空 `source.csv` 属于预期结果，它只含固定表头；需要重新运行识别。若希望连片源和整个项目目录一起删除，使用删除项目操作。

### 19.2 任务重复执行、跳过或断点续跑

每个 OCR、UVR、切片、ASR、融合和翻译任务都会用源文件指纹及参数签名写入 `cache/task-state.json`。再次提交相同签名时，后端按阶段检查：

- 完整 CSV、WAV、切片清单或导出文件直接复用；
- 文件缺失、大小为零、CSV 只有表头或清单不匹配时，从第一个不完整阶段继续；
- AI 已完成的 cue 会从 `cache/ai/*-checkpoint.json` 读取，剩余 cue 继续请求；
- OCR 和 ASR 分别从 `cache/ocr-checkpoint.json`、`cache/audio/asr-checkpoint.json` 的最近帧/切片继续，并保留少量重叠用于去重。

要主动重新识别，使用“重置工作区”；它清除缓存和生成表格但保留原视频，然后重新准备本地音轨。

### 19.3 单帧框选 OCR 失败

在最终字幕表选中一行后，播放器先定位到该行起始帧，再点击“框选帧 OCR”。拖框坐标必须落在视频画面内；识别结果会先进入确认窗口，确认后才覆盖该行。若候选为空，检查 OCR 运行时、当前帧是否已加载以及框选区域是否包含足够文字；原字幕 ROI 不会被这次临时框选改写。

<a id="local-model-runtime"></a>
<a id="download-integrity"></a>

## 20. 本地翻译模型部署或启动失败

一键部署按“检测硬件 -> 下载固定 llama.cpp -> 校验并解压 -> 下载 GGUF -> 校验大小/SHA-256/GGUF 头 -> 启动 -> `/health` 探活 -> 切换翻译配置”执行。任何阶段失败都不会把未探活的手动 GGUF 设为默认翻译源。

- AMD 报 `Vulkan`：更新显卡驱动，确认系统能识别 Vulkan；模型本身可先切 CPU 后端验证。
- NVIDIA 报 CUDA DLL 或 `no kernel image`：更新驱动并确认下载的是 NVIDIA 包；不需要另装 CUDA Toolkit。
- `local model port 18080 is already in use`：关闭占用进程，或在手动配置中改端口。
- `invalid GGUF header`、`size mismatch`、`SHA-256 mismatch`：删除对应损坏文件或 `.part` 后重试；不要关闭完整性校验。
- `did not become ready within 180 seconds`：查看日志源 `local-models/logs/llama-server.log`，减小模型或上下文，并确认可用内存/显存。
- Hugging Face/GitHub 连接中断：保留 `.part`，恢复网络后再次点击即可续传。
- `Downloading model license` 失败：一键部署会从 Qwen3 模型相同的固定 revision 下载 Apache-2.0 `LICENSE`；恢复网络后重试。许可证文件未成功保存时任务不会完成，不要手工伪造或跳过。
- 外部 Ollama/LM Studio 显示不可达：确认服务已启动、Base URL 含正确 `/v1`，并能访问 `/v1/models`。

初次没有保存过在线 API 配置时，界面不会显示“恢复在线 API”；可到“接口”标签填写新的在线配置。

### 20.1 发行构建中的可选依赖提示

下列信息在 `build-portable.ps1` 最终成功、且包内
`AUDIO-RUNTIME-PROBE.json` 的 `error` 为空时属于可选组件提示：

- `SoX could not be found`：NeMo 探测到可选 SoX 工具；Kaor 的音频读取、切分和转码使用包内 FFmpeg、TorchAudio 与本地 WAV 实现。
- `No ccache found`：只影响维护者重复编译扩展的速度，不影响已打包推理。
- `Megatron ... not found, using Apex version`：来自 NeMo 的训练组件探测；Kaor 的 ASR/说话人推理不启动 Megatron 训练。
- `OneLogger ... disabled` / `No exporters were provided`：NeMo 遥测导出器未配置；Kaor 不依赖该遥测。
- `Failed to collect ... paddle.tensorrt` / `serving plugin is not available`：Paddle 的可选 TensorRT/Serving 子模块未启用；Kaor 使用本地 PaddleOCR 推理路径。
- `pydub ... defaulting to ffmpeg`：构建虚拟环境未把 FFmpeg 放进系统 PATH；最终包必须存在 `bin/ffmpeg.exe`，打包后启动器会把它加入子进程 PATH。

若构建最终退出非零、严格 Worker probe 有任一模块为 `false`、`error` 非空，或包内
`bin/ffmpeg.exe` 缺失，则按真实失败处理，不能只依据上面的提示忽略。

## 21. 生成诊断摘要

优先点击 WebUI 日志页的“导出诊断包”。需要命令行收集时：

```powershell
$Report = Join-Path $PWD "diagnostics"
New-Item -ItemType Directory -Path $Report -Force | Out-Null
Invoke-RestMethod "$Base/api/health" | ConvertTo-Json -Depth 8 | Set-Content "$Report\health.json" -Encoding UTF8
Invoke-RestMethod "$Base/api/ocr/capabilities" | ConvertTo-Json -Depth 12 | Set-Content "$Report\ocr.json" -Encoding UTF8
Invoke-RestMethod "$Base/api/audio/capabilities" | ConvertTo-Json -Depth 12 | Set-Content "$Report\audio.json" -Encoding UTF8
Invoke-RestMethod "$Base/api/jobs?project_id=$ProjectId" | ConvertTo-Json -Depth 16 | Set-Content "$Report\jobs.json" -Encoding UTF8
```

分享前检查文件中是否含本地绝对路径、上游 URL 或敏感错误正文。不要附带 API key、视频、音频、CSV、`manifest.json` 或凭据文件。

## 许可证提示

Kaor 代码采用 MIT License。静音切分器的 RMS 与静音边界判断改编自 GPT-SoVITS `tools/slicer2.py`，Copyright (c) 2024 RVC-Boss，按 MIT License 使用；许可证文本随仓库保存在 `licenses/GPT-SoVITS-MIT.txt`，详细归属见 `THIRD_PARTY_NOTICES.md`。排查模型和二进制文件时仍需遵守各自许可证；不要把本地下载的第三方模型直接视为 MIT 内容发布。
