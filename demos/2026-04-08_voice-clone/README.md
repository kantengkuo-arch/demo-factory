# 🎙️ AI 语音克隆 Demo

上传一段参考语音，输入文字，AI 用克隆的声音朗读出来。

## 技术栈

- **后端**：Python + FastAPI
- **语音克隆**：OpenVoice v2（完整克隆模式）
- **回退方案**：edge-tts（微软云语音，无需 GPU）
- **音频处理**：soundfile, pydub, ffmpeg
- **前端**：HTML + JavaScript（暗色主题）

## 快速启动

```bash
cd backend
pip install -r requirements.txt
python3 app.py
```

然后浏览器打开 `frontend/index.html`。

> 默认使用 edge-tts 引擎（无需下载模型，开箱即用）。
> 如需完整语音克隆，请额外安装 OpenVoice 并下载模型检查点。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回当前引擎信息 |
| GET | `/voices` | 获取可用的预设音色列表 |
| POST | `/upload_reference` | 上传参考音频（5-10 秒） |
| POST | `/clone_voice` | 语音克隆/合成 |
| GET | `/download/{filename}` | 下载生成的音频文件 |

## 双引擎模式

1. **OpenVoice 模式**：上传参考音频 → 提取音色特征 → 克隆语音。需要安装 OpenVoice + 下载模型。
2. **edge-tts 模式**（默认回退）：选择预设音色 → 微软云合成。无需 GPU，开箱即用。

## 项目结构

```
voice-clone/
├── backend/
│   ├── app.py              # FastAPI 后端
│   └── requirements.txt    # Python 依赖
├── frontend/
│   └── index.html          # 前端页面
├── demo_meta.json          # Demo 元信息
└── README.md               # 本文件
```
