# 📄 AI 文档智能问答（RAG）

上传 PDF/TXT/DOCX 文档，AI 实时问答并引用原文来源，本地运行零隐私风险。

## 技术栈

- **核心算法**：RAG（检索增强生成）— 向量检索 + 上下文摘要
- **后端**：FastAPI + LangChain + ChromaDB + Sentence-Transformers
- **前端**：HTML + CSS + JavaScript（暗色主题）

## 快速启动

```bash
cd backend
pip install -r requirements.txt
python3 app.py
```

然后用浏览器打开 `frontend/index.html`，后端运行在 http://localhost:8000。

> ⚠️ 首次启动会自动下载嵌入模型（all-MiniLM-L6-v2，约 90MB），请确保网络畅通。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /api/upload | 上传文档并向量化处理 |
| POST | /api/chat | 基于文档智能问答 |
| GET | /api/documents | 获取已上传文档列表 |
| DELETE | /api/documents/{document_id} | 删除指定文档 |

## 功能特点

- 🔍 **智能检索**：基于向量相似度匹配，精准定位文档相关内容
- 📎 **来源引用**：每个回答附带原文引用和页码，可溯源验证
- 📁 **多文档管理**：支持上传多份文档，按集合管理
- 🔒 **本地运行**：数据不出本机，适合敏感文档场景
