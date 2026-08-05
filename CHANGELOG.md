# Changelog

## 0.2.0 - 2026-08-05

Kaor 0.2.0 is the first public Windows desktop release.

### Included

- Desktop WebUI with video import, frame-accurate seeking, two editable 8-point regions, live task progress, logs, repair guides, and workspace reset.
- Subtitle, audio, and hybrid recognition modes with PaddleOCR, UVR5-compatible BS-Roformer inference, silence slicing, language-specific ASR, timing refinement, and speaker cues.
- Separate OCR, ASR, fusion, translation, review, ASS, and burn-in stages with resumable artifacts and CSV export.
- OpenAI-compatible APIs, relay endpoints, upstream model discovery, reasoning controls, and managed local llama.cpp/Ollama/LM Studio workflows.
- Manual cue creation/deletion, single-frame OCR replacement, OCR/ASR/final views, per-cue and per-speaker colors, and video-frame eyedropper.
- Self-contained Windows x64 CPU, AMD, and NVIDIA CUDA 12.6 packages. Python, Node.js, FFmpeg, CUDA Toolkit, and UVR5 do not need to be installed separately.

### Runtime downloads

- Public archives exclude the BS-Roformer YAML and checkpoint because their upstream pages do not state redistribution terms. The first UVR task downloads both from pinned upstream locations, verifies size and SHA-256, supports checkpoint resume, and then reuses the local copies.
- Language-specific ASR, optional NeMo diarization, and local translation models are downloaded only when selected.

### Hardware profiles

- `CPU`: OCR, audio, and local translation use CPU execution.
- `AMD`: OCR and audio use CPU; managed local translation can use Vulkan on the AMD GPU.
- `NVIDIA`: PaddleOCR and PyTorch audio use CUDA 12.6; managed local translation can use CUDA or Vulkan.
