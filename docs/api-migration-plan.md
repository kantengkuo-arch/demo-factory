# 🔧 Demo Factory — 统一 API 配置方案

## 1. 统一配置文件

所有 demo 共用一个 API 配置文件：`platform/api_config.json`

```json
{
  "version": 1,
  "llm": {
    "base_url": "https://your-api.com/v1",
    "api_key": "sk-xxx",
    "model": "gpt-4o",
    "comment": "Chat/Completion — 用于文本生成、问答、摘要"
  },
  "embedding": {
    "base_url": "https://your-api.com/v1",
    "api_key": "sk-xxx",
    "model": "text-embedding-3-small",
    "comment": "文本向量化 — 用于 RAG 检索"
  },
  "image": {
    "base_url": "https://your-api.com/v1",
    "api_key": "sk-xxx",
    "model": "dall-e-3",
    "comment": "图像生成/编辑 — 用于风格迁移等"
  },
  "tts": {
    "base_url": "https://your-api.com/v1",
    "api_key": "sk-xxx",
    "model": "tts-1",
    "comment": "语音合成 — 用于语音克隆/TTS"
  }
}
```

## 2. 统一 API 客户端

创建 `platform/api_client.py` — 所有 demo import 这一个文件：

```python
from platform.api_client import DemoAPIClient

client = DemoAPIClient()  # 自动读取 api_config.json
# 然后直接调用
embedding = client.embed("一段文本")
answer = client.chat("请总结这段话...")
image = client.image_edit(content_img, style_img)
audio = client.tts("要说的话", voice="alloy")
```

## 3. 各 Demo 改造计划

### ✅ RAG 文档问答 (issue #19)
- embedding: `sentence-transformers` → `client.embed()`
- 回答生成: 拼接 context → `client.chat()`
- **可删依赖**: sentence-transformers, torch, langchain (大部分)
- **保留**: chromadb (轻量向量库), pypdf, docx2txt
- **预计从 5GB → 200MB**

### ✅ 图片风格迁移 ×2 (issue #7, #11)
- 风格迁移: `PyTorch + VGG19` → `client.image_edit()`
- **可删依赖**: torch, torchvision
- **预计从 3GB → 50MB**
- ⚠️ 需要图像生成 API 支持 style transfer，否则改用 img2img

### ✅ 语音克隆 (issue #8)
- TTS: `OpenVoice` → `client.tts()`
- **可删依赖**: torch, openvoice, melo-tts
- **预计从 4GB → 50MB**
- ⚠️ 声音克隆功能取决于 API 是否支持

### ⚠️ 时序预测 (issue #14)
- 预测: `TimesFM` → 较难替换，没有标准 API
- **方案 A**: 用 `client.chat()` 让 LLM 分析趋势（效果可能不如专业模型）
- **方案 B**: 改用 Prophet/statsmodels（纯统计，依赖小）
- **方案 C**: 保留本地 TimesFM，但单独部署

## 4. Scout 模板更新

更新 scout agent 的 AGENTS.md，要求：
- 新 demo **必须使用 `DemoAPIClient`**，禁止引入 torch 等重型依赖
- API 契约中标注使用哪些 API 能力（llm/embedding/image/tts）
- requirements.txt 目标：< 10 个包，安装 < 200MB

## 5. 执行步骤

Phase 1: 基础设施
1. 创建 `platform/api_config.json`（你填 API 地址和 key）
2. 创建 `platform/api_client.py`（统一客户端）
3. 更新 scout/coder/frontend AGENTS.md

Phase 2: 改造现有 demo（按优先级）
1. RAG 文档问答 — 最简单，只需替换 embedding + chat
2. 风格迁移 — 取决于图像 API 能力
3. 语音克隆 — 取决于 TTS API 能力
4. 时序预测 — 评估后决定方案
