# 动作识别 + RTSP 推流系统

基于 YOLOv8 姿态检测、阿里云百炼 Qwen2.5-Omni-7B 多模态大模型和 FFmpeg 的实时动作识别与 RTSP 推流系统。

## 功能特性

- **实时骨骼检测**：使用 YOLOv8n-pose 模型进行人体姿态检测
- **AI 动作识别**：调用阿里云百炼 Qwen2.5-Omni-7B 多模态大模型识别动作
- **RTSP 推流**：通过 FFmpeg 将处理后的视频流推送到 RTSP 服务器
- **多线程架构**：
  - 主线程：视频读取和 YOLO 检测
  - AI 识别线程：异步调用大模型进行动作识别
  - 推流线程：异步写入 FFmpeg 避免阻塞
- **中文显示支持**：在视频上叠加中文字幕显示识别结果

## 环境要求

- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- LangChain OpenAI
- Pillow
- NumPy
- FFmpeg (系统环境)

## 安装依赖

```bash
pip install opencv-python ultralytics langchain-openai pillow numpy
```

## 配置说明

### 1. 阿里云百炼 API 配置

在代码中配置你的 API Key：

```python
chat_model = ChatOpenAI(
    api_key="your-api-key-here",  # 替换为你的阿里云百炼 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen2.5-omni-7b",
    temperature=0.3,
    max_tokens=256,
    timeout=8,
)
```

### 2. 视频源配置

```python
VIDEO_PATH = "1.mp4"  # 输入视频文件路径
RTSP_URL = "rtsp://127.0.0.1:554/stream"  # RTSP 推流地址
```

### 3. YOLO 模型

默认使用 `yolov8n-pose.pt` 模型，首次运行会自动下载。

## 使用方法

1. 确保 FFmpeg 已安装并添加到系统 PATH
2. 配置好阿里云百炼 API Key
3. 准备输入视频文件（默认 `1.mp4`）
4. 运行程序：

```bash
python 3qwen.py
```

5. 按 `Q` 键退出程序

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  视频输入    │────▶│  YOLO检测   │────▶│  骨骼标注   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌─────────────────┐
              │   动作识别队列   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Qwen2.5-Omni-7B │
              │   动作识别      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐     ┌─────────────┐
              │   叠加中文文字   │────▶│  RTSP推流   │
              └─────────────────┘     └─────────────┘
```

## 性能参数

- 视频分辨率：720x1280
- 帧率：25 FPS
- 推流码率：1500 kbps
- AI 识别频率：每 15 帧识别一次
- 推流队列缓冲：60 帧（约 2 秒）

## 文件说明

| 文件 | 说明 |
|------|------|
| `3qwen.py` | 主程序文件 |
| `yolov8n-pose.pt` | YOLO 姿态检测模型（自动下载） |
| `1.mp4` | 默认输入视频文件 |

## 注意事项

1. **API Key 安全**：请勿将 API Key 提交到公共仓库
2. **网络要求**：需要稳定的网络连接以调用阿里云百炼 API
3. **字体支持**：程序会自动尝试加载 Windows 系统字体，确保中文字符正确显示
4. **FFmpeg 配置**：确保 FFmpeg 支持 libx264 编码器

## 许可证

MIT License
