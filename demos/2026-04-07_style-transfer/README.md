# 🎨 图片风格迁移 Demo

使用 PyTorch + VGG19 神经网络，将照片转换为油画、素描、水彩、印象派等艺术风格。

## 技术栈

- **核心算法**：PyTorch + torchvision VGG19 预训练模型
- **后端**：FastAPI
- **前端**：HTML + CSS + JavaScript（暗色主题）
- **图像处理**：Pillow, NumPy

## 快速启动

```bash
cd backend
pip install -r requirements.txt
python3 app.py
```

然后用浏览器打开 `frontend/index.html`，后端运行在 http://localhost:8000。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回设备信息 |
| GET | `/styles` | 获取所有可用风格列表 |
| POST | `/upload` | 上传原始图片，返回 file_id |
| POST | `/style_transfer` | 执行风格迁移（参数：file_id, style） |
| GET | `/download/{filename}` | 下载风格化后的图片 |

## 支持风格

| 风格 | Key | 说明 |
|------|-----|------|
| 🖼️ 油画 | `oil_painting` | 浓郁色彩的古典油画风格 |
| ✏️ 素描 | `sketch` | 黑白线条的铅笔素描效果 |
| 🎨 水彩 | `watercolor` | 柔和透明的水彩画风格 |
| 🌅 印象派 | `impressionist` | 光影交错的印象派画风 |

## 项目结构

```
2026-04-07_style-transfer/
├── backend/
│   ├── app.py              # FastAPI 后端 + 风格迁移算法
│   └── requirements.txt    # Python 依赖
├── frontend/
│   └── index.html          # 前端页面（暗色主题）
├── demo_meta.json          # Demo 元数据
└── README.md
```

## 说明

- 自动检测 GPU/CPU，GPU 模式下使用 512px 分辨率，CPU 模式使用 256px
- 风格参考图为程序化生成，无需额外下载风格图片
- 单张图片处理时间：GPU 约 10-30 秒，CPU 约 30-120 秒
