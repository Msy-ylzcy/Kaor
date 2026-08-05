# 参与贡献

感谢参与 Kaor。项目的目标是提供可审阅、可复现、本地优先的视频字幕工作流。功能数量不是唯一标准，识别结果的可追踪性、数据不被静默覆盖以及普通硬件上的可运行性同样重要。

## 开始之前

1. 搜索已有 Issue 和 Pull Request，避免重复工作。
2. 较大的架构变化、模型替换、数据格式变化或新平台打包，请先建立 Issue 说明动机、兼容性和迁移方案。
3. 不要提交视频样本、API key、用户项目目录、OCR 模型权重、FFmpeg 二进制或构建产物。
4. 测试素材必须具有明确的再分发许可，或使用程序生成的最小样本。

## 本地环境

推荐使用 Python 3.12 x64 和 Node.js 20+。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install pytest==8.4.1
npm.cmd ci --prefix apps/web
```

将 `ffmpeg` 和 `ffprobe` 放入 `bin/` 或加入 `PATH`。需要实际 OCR 时，将模型放在 `models/`；普通单元测试不应依赖在线模型下载。

## 开发约定

- 保持视频、OCR、时间轴、CSV、ASS 和渲染流程默认在本地执行。
- 新增网络访问必须有明确的 UI 操作、可见目标地址和文档说明；后台静默上传视频或项目数据属于阻断合并的问题。
- 保持 `source.csv` 与 `translated.csv` 分离，任何自动修正都必须可审阅。
- 不要让翻译模型改变 `cue_id`、时间、说话人或轨道字段。
- 同时说话和后叠字幕应保留独立 cue 与 layer，不要为了实现简单而合并文本。
- 新的硬件加速后端必须提供 CPU 回退路径，并在对应真实硬件或 CI runner 上验证。
- 依赖升级应说明原因，更新对应 requirements 文件，并检查 portable 包体积和许可证变化。
- Python 和 TypeScript 修改应尽量保持类型清晰，复杂识别逻辑要配最小回归测试。

## 验证

提交前至少运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/backend -q
.\.venv\Scripts\python.exe -m compileall -q backend kaor.py
npm.cmd run build --prefix apps/web
```

涉及 UI 的修改应检查 1024×768、1280×720 和 1920×1080 等桌面窗口，确认没有文字溢出、控件遮挡或页面级水平滚动。Kaor 不把移动端作为运行或验收目标。涉及 OCR、重叠跟踪或字幕排版的修改，应增加覆盖失败场景的测试。

## Pull Request

Pull Request 描述应包含：

- 问题和用户可见行为。
- 实现方式及主要取舍。
- 已运行的测试命令与结果。
- 数据格式、隐私边界、依赖、许可证和发布包是否受到影响。
- UI 变化的截图，或识别/渲染变化的最小输入与输出说明。

保持提交范围集中，不要夹带无关格式化、生成文件或依赖锁文件重写。维护者可能要求拆分过大的 Pull Request。

## 许可证

提交贡献即表示你有权提交该内容，并同意该贡献按项目的 MIT License 发布。第三方代码或资源必须保留原许可证与归属信息，并在 Pull Request 中明确指出。
