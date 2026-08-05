# Windows 免环境发行构建

本文面向项目维护者，说明 CPU、AMD 和 NVIDIA 三套 Windows x64 portable
包的构建边界、固定依赖、完整性校验和 GitHub Release 流程。

三套最终发行物都必须是免环境版本：完整解压后直接运行 `Kaor.exe`，用户不安装
Python、Paddle、PyTorch、FFmpeg、Node.js、CUDA Toolkit 或 UVR5，也不执行 `pip`。

## 发行矩阵

| Profile | OCR | UVR / ASR / 说话人 | 本地翻译 | 目标机器 |
| --- | --- | --- | --- | --- |
| `cpu` | Paddle CPU | PyTorch CPU | CPU 后端 | 无独显或兼容性优先 |
| `amd` | Paddle CPU | PyTorch CPU | Vulkan AMD GPU | AMD 显卡 Windows 机器 |
| `nvidia-cu126` | Paddle CUDA 12.6 | PyTorch CUDA 12.6 | CUDA 或 Vulkan | 支持 CUDA 12.6 wheel 的 NVIDIA 显卡 |

Windows 下 AMD profile 不把 CUDA wheel 伪装成 AMD wheel。OCR、UVR、ASR 和
说话人聚类固定使用 CPU 内核，这是该发行档位的明确限制；由“一键部署本地模型”
管理的 llama.cpp Vulkan 推理服务使用 AMD GPU。这样同一发行包可在 AMD 驱动版本
不同的机器上启动，并在 Vulkan 不可用时允许本地翻译切到 CPU。

## 每个 ZIP 包含什么

- `Kaor.exe`：无控制台 WebUI 主程序。
- `KaorAudioWorker.exe`：隔离 UVR、ASR 和说话人任务的控制台子进程。两个 EXE
  共用一个 PyInstaller `COLLECT` 目录中的 DLL 和 Python 模块。
- `bin/ffmpeg.exe`：固定的 FFmpeg 可执行文件。
- `models/paddlex/`：离线 OCR 检测和识别模型。
- `models/uvr/`：运行时按需创建；BS-Roformer YAML 与 checkpoint 都不进入公开发行包。
- `models/diarization/`：默认不进入发行包；仅在两个完整 NeMo 说话人模型均存在、已核对再分发条款，并显式传入 `-BundleDiarizationModels` 时才携带。否则首次需要聚类时由程序管理下载。
- `fonts/`、`docs/`、许可证和第三方通知。
- `RELEASE.json`、`AUDIO-RUNTIME-PROBE.json`、`DEPENDENCIES-PYTHON.txt` 和
  `SHA256SUMS.txt`。

BS-Roformer YAML 与 checkpoint、语言专用 ASR checkpoint、本地翻译 GGUF，以及构建输入未
提供时的 NeMo 说话人模型由程序按需下载，不把授权边界不清晰或非必选的数 GB
模型重复塞进三个发行包。下载后仍在本机推理，不需要安装 Python。

## 固定的 OCR 输入

仓库的 `models/` 目录不进入 Git，因此 GitHub Actions 的干净 checkout 不带 OCR
权重。Release workflow 会直接从 PaddlePaddle 官方 Hugging Face 仓库下载下面两套
PP-OCRv6 medium 文件，写入 `models/paddlex/official_models/` 后才调用打包脚本：

| 模型 | 固定 revision |
| --- | --- |
| `PaddlePaddle/PP-OCRv6_medium_det` | `e42c8690e385ae9639912cad7b65e1de8075314d` |
| `PaddlePaddle/PP-OCRv6_medium_rec` | `e5a92bcbc5cc1b494628e458d267778f0704fd7c` |

下载清单如下；不下载 `.gitattributes`，因为它不是运行时文件：

| 模型 | 文件 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| det | `inference.json` | 312150 | `0f1a7ec35da36173529c7a60238b7f7919e3831929c3f700ad90ad4896adecd5` |
| det | `inference.pdiparams` | 61960476 | `85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960` |
| det | `inference.yml` | 886 | `7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871` |
| det | `README.md` | 6366 | `c727acae3889adc1f9caff2db94ec32fa3e85c31da943241aceb0ee01ef8f808` |
| rec | `inference.json` | 221814 | `0b2e25e990bd072f1bf77d59d67d508bce6c4bd44af6624e0fb27d6da2cd00e8` |
| rec | `inference.pdiparams` | 76465087 | `1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319` |
| rec | `inference.yml` | 150580 | `991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129` |
| rec | `README.md` | 23474 | `75ff9c4853ed171f36224127805062f9bcd1cfbbcacae3d577344866df74f6d3` |

上述表格由 `licenses/PP-OCRV6-MODEL-MANIFEST.json` 提供给 Workflow 和本地打包
脚本共同读取。任一下载或本地文件的长度、SHA-256 不符都会立即中止。最终
portable 目录同时携带该机器清单、`licenses/APACHE-2.0.txt`、模型原始
`README.md` 和 `licenses/PP-OCRV6-MODEL-NOTICE.txt`。本地维护者构建使用相同目录
布局；缺少 detector 或 recognizer 文件时不会临时联网补齐。

## UVR 模型资产首次下载

公开 portable 包不打包下面两项上游模型资产，并在归档前显式检查它们不存在：

```text
models\uvr\model_bs_roformer_ep_317_sdr_12.9755.yaml
models\uvr\model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

用户第一次启动 UVR 阶段时，Kaor 下载匹配 YAML 与 checkpoint 到 `models/uvr/`。
YAML 固定到 `TRvlvr/application_data` 提交
`22b79fc01ada8f3b9e3526ad0ed645af414a7cde`，核对 `2273` 字节及 SHA-256
`2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f`。
checkpoint 从 `TRvlvr/model_repo` 的 `all_public_uvr_models` Release 获取，以
`.ckpt.part` 临时文件断点续传，完成后核对 `639331213` 字节及 SHA-256
`5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa` 再原子改名。
运行时只读取程序目录的 `models/uvr/`，不扫描目标机器上的外部 UVR5 路径。

## 本机构建命令

环境前提仅针对维护者的构建机：Windows x64、Python 3.12、Node.js 20 和 npm。
最终用户不需要这些环境。

```powershell
.\scripts\build-portable.ps1 -RuntimeProfile cpu -Version 0.2.0
.\scripts\build-portable.ps1 -RuntimeProfile amd -Version 0.2.0
.\scripts\build-portable.ps1 -RuntimeProfile nvidia-cu126 -Version 0.2.0
```

Torch 系列索引始终由脚本显式指定：CPU/AMD 使用清华 PyPI 镜像安装无 CUDA
后缀的 Torch、TorchAudio 和 TorchVision，NVIDIA 使用 PyTorch 官方
`/whl/cu126` 安装三者的匹配版本。随后运行时检查 `torch.version.cuda`：CPU/AMD
必须为空，NVIDIA 必须存在，并实际加载 TorchVision 原生算子，避免镜像解析错包。

常用开关：

| 参数 | 作用 |
| --- | --- |
| `-SkipInstall` | 复用已完成安装的 `.venv-<profile>` |
| `-SkipFrontend` | 复用已生成的 `apps/web/dist` |
| `-SkipTests` | 跳过源码测试，仅用于本地迭代 |
| `-SkipArchive` | 只生成 portable 目录 |
| `-RequireAccelerator` | 在有对应显卡的构建机上增加硬件探测 |
| `-BundleDiarizationModels` | 已确认权重再分发条款后，把两份完整 NeMo 说话人模型写入发行包 |
| `-LocalInferenceRuntimeDirectory PATH` | 把已准备的本地翻译 runtime 放进 `bin/local-inference` |

`-RequireAccelerator` 默认关闭，因为 GitHub 托管的 Windows runner 没有 NVIDIA
GPU。即使没有显卡，构建仍校验 Paddle 是否为 CUDA build、Torch 是否来自
`cu126`，并由打包后的 `KaorAudioWorker.exe probe` 验证所有动态音频模块确实被
收集。probe 的 `error` 必须为空，`torch_version` 和 `torchvision_version` 必须为
固定匹配版本；CPU/AMD 的 `torch_cuda_version` 必须为空，NVIDIA 的该字段必须
存在。只找到模块但实际导入或原生算子加载失败也会中止构建。

## 构建输出和校验

CPU 和 AMD 各自产生一个普通 ZIP：

```text
artifacts/releases/Kaor-Windows-x64-CPU/
artifacts/releases/Kaor-Windows-x64-CPU.zip
artifacts/releases/Kaor-Windows-x64-CPU.zip.sha256
artifacts/releases/Kaor-Windows-x64-CPU.zip.release.json

artifacts/releases/Kaor-Windows-x64-AMD/
artifacts/releases/Kaor-Windows-x64-AMD.zip
artifacts/releases/Kaor-Windows-x64-AMD.zip.sha256
artifacts/releases/Kaor-Windows-x64-AMD.zip.release.json
```

完整 NVIDIA ZIP 始终只作为 `.build/portable/nvidia-cu126/` 下的构建中间物，
Release 目录统一输出 1900 MiB 分片和本地组装器：

```text
artifacts/releases/Kaor-Windows-x64-NVIDIA/
artifacts/releases/Kaor-Windows-x64-NVIDIA.zip.001
artifacts/releases/Kaor-Windows-x64-NVIDIA.zip.002
artifacts/releases/Kaor-Windows-x64-NVIDIA.zip.<NNN>
artifacts/releases/Kaor-Windows-x64-NVIDIA.parts.json
artifacts/releases/Kaor-Windows-x64-NVIDIA.parts.sha256
artifacts/releases/Kaor-Windows-x64-NVIDIA-Setup.exe
artifacts/releases/Kaor-Windows-x64-NVIDIA-Setup.cs
```

`parts.json` 记录完整 ZIP、每片、Setup、Setup 源码的长度和 SHA-256；
`parts.sha256` 是便于人工核查的文本副本。完整 ZIP 的哈希也会写入这两个文件，
但完整 ZIP 本身不是 GitHub asset。Setup 源码由仓库内
`scripts/portable-split-assembler/Program.cs` 编译，构建机缺少 Windows .NET
Framework C# 编译器时直接失败，不下载或使用不可审计的第三方解包程序。

脚本会依次检查 Python 版本、Torch/Paddle 运行时、前端产物、后端测试、两个
PyInstaller EXE、OCR 模型、UVR YAML/checkpoint 排除策略、严格音频 worker probe、包内逐文件 SHA-256 以及
完整 ZIP SHA-256。CPU/AMD 单 ZIP 超过 GitHub 单个 asset 的 2 GiB 时仍默认失败；
NVIDIA 无论整包是否刚好低于该上限都进入上述分片流程，并强制至少两片、每片
小于 2 GiB，从而让 Release 和安装方式保持唯一且稳定。

## NVIDIA 分片使用方法

用户从同一个 GitHub Release 下载以下文件并放进同一目录：

1. `Kaor-Windows-x64-NVIDIA-Setup.exe`；
2. `Kaor-Windows-x64-NVIDIA.parts.json`；
3. 从 `.zip.001` 到最后一片的全部 `.zip.<NNN>` 文件。

双击 `Kaor-Windows-x64-NVIDIA-Setup.exe`。程序会按清单逐片读取并核对 SHA-256，
流式重建完整 ZIP，再次核对整包 SHA-256，然后用 Windows 自带的
`System.IO.Compression` 解压到同目录的 `Kaor-Windows-x64-NVIDIA/`。完成后可直接
启动 `Kaor.exe`，用户不安装 Python、CUDA Toolkit 或压缩软件，也不会在首启时
下载运行库。支持的 Windows 10/11 本身提供该小工具使用的 .NET Framework 4.x。

Setup 不覆盖已经存在的目标目录，拒绝 ZIP 路径穿越和符号链接，并在失败或取消后
清理它自己创建的临时整包和临时解压目录。安装时需要的可用空间约为“完整压缩包
大小 + 解压后大小 + 512 MiB”；只校验时约为“完整压缩包大小 + 256 MiB”。

维护者或自动化可运行：

```powershell
& .\Kaor-Windows-x64-NVIDIA-Setup.exe `
  --verify-only .\Kaor-Windows-x64-NVIDIA.parts.json

& .\Kaor-Windows-x64-NVIDIA-Setup.exe `
  --headless .\Kaor-Windows-x64-NVIDIA.parts.json D:\Kaor
```

`--headless` 的第三个参数是目标目录的父目录。命令行或双击模式失败时，Setup
退出码为 `1`，并在自身旁边写入 `Kaor-Windows-x64-NVIDIA-Setup.error.log`；
下次运行会先清理旧日志，若仍失败则写入本次完整异常。

常见错误及处理：

| 错误 | 原因与处理 |
| --- | --- |
| `Release manifest was not found` | `parts.json` 不在 Setup 同目录；重新下载并保持原文件名。 |
| `Release part is missing` | 缺少错误中点名的 `.zip.<NNN>`；从同一个 Release 补下载。 |
| `Release part size mismatch` | 分片未下载完整或被浏览器改名；删除该片后重新下载。 |
| `Release part SHA-256 mismatch` | 分片损坏或混入其他版本；只重下被点名的分片。 |
| `Assembled archive SHA-256 mismatch` | 分片与清单不是同一 Release；重新下载整套清单与分片。 |
| `Not enough free disk space` | 在错误所示磁盘释放空间，或把整套文件移动到空间更大的磁盘。 |
| `The destination already exists` | Setup 从不覆盖已有目录；先重命名现有 `Kaor-Windows-x64-NVIDIA/`。 |
| `Unsafe archive path` / `Symbolic links are not accepted` | 发行物结构不符合门禁；保留 error log 并停止发布该资产集。 |

修改分片器后必须跑本地完整往返测试。测试使用小型随机夹具，不需要 CUDA 或
Python，会验证正常重组/解压、逐文件一致性、篡改单片后拒绝以及恢复后再次通过：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\test-portable-split.ps1
```

## GitHub Release

`.github/workflows/release.yml` 支持推送 `vX.Y.Z` tag 或手动输入版本。三个 profile
在独立 Windows runner 上并行构建。workflow 会从官方固定 revision 下载两套
PP-OCRv6 运行文件与模型卡，并从上游下载 BS-Roformer YAML；所有文件先核对
固定大小和哈希，再调用同一构建脚本。workflow 不下载 checkpoint。只有三个
profile 的完整发行资产集
及其 sidecar 都完成且复核 SHA-256 后，publish job 才进入发布阶段。工作流会先
创建草稿 Release，或将同标签的可变 Release 暂时转成草稿；完整上传、删除同标签
残留的旧 Kaor 资产并再次确认资产精确白名单后，才切换为公开状态。任一步失败时
Release 保持草稿，避免公开半套或混合版本资产。已经启用 immutable 的 Release
不会被覆盖，必须提高版本号重新发布。NVIDIA 所有分片、清单、Setup 和可审计源码
都属于同一资产集，缺一项或多出一项即不得发布。publish job 还会实际执行 Setup 的
`--verify-only`，流式重建完整 ZIP 并复核整包哈希。

checkpoint 不进入 Kaor 的 GitHub Release，因此公开发行不再依赖权重再分发许可
变量。其来源、固定长度、哈希和未解决的许可边界仍保留在
`licenses/BS-ROFORMER-MODEL-NOTICE.txt`，首次使用由用户设备直接从上游获取。
发布前仍应在真实 CPU、AMD、NVIDIA 机器各完成一次：启动、首次模型下载、OCR、
UVR、切分、
ASR、AI 校对/翻译、渲染和重置项目的端到端验收。CI 的无 GPU runner 只能证明
依赖和打包结构完整，不能代替真实显卡驱动兼容测试。

## NVIDIA 体积预算

Windows CUDA 12.6 的 Torch wheel 自带完整 CUDA DLL。当前固定版本中，仅
`torch/lib/*.dll` 就是 `3,989,069,456` 字节；逐文件使用与 ZIP 相同的 Deflate
算法后仍约 `2,565,588,700` 字节，已经超过 GitHub 单个 Release asset 的 2 GiB
上限。Paddle CUDA 和 OCR 模型尚未计入，因此删除 Python 示例、
训练模块或静态 `.lib` 文件并不能让完整 NVIDIA CUDA 包变成单一 ZIP。

维护者可复测原始大小和 Deflate 下限：

```powershell
.\.venv-build\Scripts\python.exe scripts\audit-portable-size.py `
  --pattern *.dll `
  .venv-nvidia-cu126\Lib\site-packages\torch\lib
```

CPU 和 AMD profile 可继续使用单 ZIP。NVIDIA profile 若保留 GPU OCR 与 GPU
音频，完整 CUDA 运行库也必须作为同一 Release 的发行资产交付，不允许要求用户
安装 Python/CUDA 或在首启时临时补环境。构建脚本现在把超限的完整 ZIP 流式切成
1900 MiB 分片，并生成带逐片/整包 SHA-256 的 Setup；解包后的目录仍是完整
portable 目录。不要通过删除 cuBLAS、cuDNN、cuFFT、
cuSPARSE 或 cuSOLVER 强行缩包：`torch_cuda.dll` 直接依赖这些 DLL，会让 Torch
导入或实际推理在目标机上失败。分片解决的是 GitHub 单资产上限，不改变 NVIDIA
完整运行时内容；首次使用 UVR 时只新增受校验的 checkpoint，不补装 Python、Torch
或 CUDA 运行库。公开 Release 前仍需在干净 NVIDIA 机器完成实际推理验收。
