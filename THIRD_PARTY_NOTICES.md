# Third-Party Notices

Kaor source code is licensed under the MIT License. Kaor also depends on third-party software, model files, and optional binaries that remain subject to their own licenses. This document is an overview, not a replacement for the license files shipped by those projects.

## Major runtime dependencies

| Component | Purpose | License commonly published by upstream |
| --- | --- | --- |
| FastAPI | Local HTTP API | MIT |
| Uvicorn | Local ASGI server | BSD-3-Clause |
| HTTPX | Translation API client | BSD-3-Clause |
| Pydantic | Data validation | MIT |
| keyring | Operating-system credential storage | MIT |
| NumPy | Numerical operations | BSD-3-Clause |
| OpenCV | Video frames and image processing | Apache-2.0 |
| PaddlePaddle | CPU/NVIDIA inference runtime | Apache-2.0 |
| PaddleOCR | OCR pipeline | Apache-2.0 |
| PaddleX | PaddleOCR pipeline support | Apache-2.0 |
| Noto Sans SC | Portable CJK subtitle rendering | SIL Open Font License 1.1 |

## GPT-SoVITS audio slicer

Kaor's silence-based speech slicer is adapted from `tools/slicer2.py` in
GPT-SoVITS, copyright (c) 2024 RVC-Boss, under the MIT License. The retained
license text is shipped as `licenses/GPT-SoVITS-MIT.txt`. Kaor exposes the
upstream slicer's threshold, minimum length, minimum silence interval, analysis
hop size, and retained-silence parameters, then uses each slice's sample offset
to restore absolute subtitle timestamps after ASR.

## Web dependencies

| Component | Purpose | License commonly published by upstream |
| --- | --- | --- |
| React / React DOM | WebUI | MIT |
| Vite | Frontend build tool | MIT |
| TypeScript | Type checking and compilation | Apache-2.0 |
| Lucide | Interface icons | ISC |

## Build tooling

PyInstaller is distributed under GPL-2.0-or-later with an exception that permits distributing bundled applications under the application's chosen license, subject to PyInstaller's complete license terms. pytest and other development-only packages retain their own licenses.

## OCR models

Portable releases bundle the official `PaddlePaddle/PP-OCRv6_medium_det` and
`PaddlePaddle/PP-OCRv6_medium_rec` inference files from Hugging Face. The
detection model is pinned to revision
`e42c8690e385ae9639912cad7b65e1de8075314d`; the recognition model is pinned to
revision `e5a92bcbc5cc1b494628e458d267778f0704fd7c`. Both model cards identify
the models as Apache License 2.0.

The release workflow downloads only the required runtime files and original
model cards, then verifies every byte length and SHA-256 before packaging.
Kaor retains the complete Apache-2.0 text as `licenses/APACHE-2.0.txt`, the
original `README.md` beside each model, and the fixed source revisions and
human-readable file notice as `licenses/PP-OCRV6-MODEL-NOTICE.txt`. The shared
machine-readable manifest is `licenses/PP-OCRV6-MODEL-MANIFEST.json`. These model files remain
under their upstream terms; the Kaor MIT License does not relicense them.

## BS-Roformer vocal-separation model

The three Windows portable profiles bundle the BS-Roformer inference runtime
and `model_bs_roformer_ep_317_sdr_12.9755.yaml`, but do not redistribute the
matching checkpoint (attributed as a viperx model by the upstream catalog).
When the user first starts the UVR stage, Kaor downloads the checkpoint directly
from the fixed upstream location, verifies its size and SHA-256, and stores it
under the portable package's `models/uvr/` directory. Kaor does not scan an
existing UVR5 installation or an environment-variable path.

The checkpoint is downloaded by Kaor's first-use downloader from the public
`TRvlvr/model_repo` GitHub Release and is validated as 639,331,213 bytes with
SHA-256
`5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa`.
The matching YAML comes from
`ZFTurbo/Music-Source-Separation-Training`, whose repository publishes the MIT
License retained as `licenses/ZFTURBO-MSS-MIT.txt`.

**Redistribution status:** as checked on 2026-08-05, the
`TRvlvr/model_repo` repository has no `LICENSE` file and the
`all_public_uvr_models` Release page does not state a license for this
checkpoint. Public download availability is not, by itself, an explicit grant
to redistribute the weight inside another product. The ZFTurbo repository's
MIT License is the license published by the repository that contains the YAML,
and no file-specific exception for that YAML was found; it must not be
presented as the checkpoint's license. Kaor therefore excludes this weight from
its public archives. Before publishing any future Kaor archive containing the
weight, the release maintainer must obtain and retain an explicit redistribution
grant from the model rightsholder/distributor or replace the checkpoint with a
model whose terms expressly permit the intended redistribution. Kaor's MIT
License cannot override that boundary.

Full provenance, hashes, source URLs, and this unresolved license boundary are
retained in `licenses/BS-ROFORMER-MODEL-NOTICE.txt`.

## NeMo diarization models

Kaor includes the NeMo runtime but does not include the
`vad_multilingual_marblenet.nemo` or `titanet_large.nemo` weights in default
portable releases. When speaker clustering first needs them, NeMo retrieves
the model archives from NVIDIA's upstream service into Kaor's managed model
directory. The weights remain subject to their upstream terms. A maintainer
build may embed both files only with the explicit
`-BundleDiarizationModels` switch after verifying the applicable
redistribution terms.

## Local translation runtime and models

Kaor's one-click local translation feature downloads third-party assets only
after the user starts deployment:

| Component | Use | Upstream license and retained notice |
| --- | --- | --- |
| llama.cpp | CPU, Vulkan, or CUDA `llama-server` runtime | MIT; Kaor retains the upstream text in `licenses/LLAMA_CPP-MIT.txt` |
| Official Qwen3 GGUF models | Local fusion and translation weights | Apache-2.0 in the official `Qwen/Qwen3-*-GGUF` repositories; see `licenses/QWEN3-NOTICE.txt` |

The llama.cpp runtime is downloaded from a fixed upstream GitHub Release and is
not relicensed by Kaor. For Qwen3, the model download and its license download
are one deployment transaction: Kaor fetches `LICENSE` from the same pinned
repository revision as the selected GGUF, stores it under the local model
`models/licenses/` directory, and records the file in the model manifest. A
deployment whose license file was not saved is not complete. Each upstream
license continues to apply to its own files.

## FFmpeg and codecs

FFmpeg is not covered by the Kaor MIT License. FFmpeg can be built under LGPL or GPL terms depending on its configuration and enabled codec libraries. In particular, an FFmpeg build that includes GPL components requires the distributor to follow the applicable GPL obligations.

Before publishing a portable archive:

1. Run `bin\ffmpeg.exe -version` and record the configuration.
2. Identify whether the selected build is LGPL or GPL and which external codec libraries are present.
3. Include the matching FFmpeg license, copyright notices, and any source-code offer or source location required by that build.
4. Keep FFmpeg as a clearly identified third-party executable in `bin/`.

The default Windows build script extracts FFmpeg 7.1 from the
`imageio-ffmpeg==0.6.0` Windows x64 wheel. That executable identifies itself as the
Gyan essentials build and reports `--enable-gpl`, `--enable-version3`, `libx264`,
`libx265`, and `libass`. It is therefore a GPLv3 FFmpeg build, distributed separately
under `bin/ffmpeg.exe`; it is not covered by Kaor's MIT License. Upstream build
and source information is available from https://www.gyan.dev/ffmpeg/builds/ and
https://ffmpeg.org/download.html. Release maintainers must include the corresponding
GPL text and source information with public binary releases.

## Complete dependency data

Python and npm resolve additional transitive dependencies. The authoritative list for a given release is the installed distribution metadata and the generated npm lock file at build time. Release maintainers should archive a dependency inventory and license scan with each release, and include every required license file in the distributed package.

Project names and trademarks belong to their respective owners. Their inclusion here does not imply endorsement.
