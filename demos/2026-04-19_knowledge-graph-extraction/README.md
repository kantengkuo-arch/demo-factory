# AI 知识图谱抽取

从非结构化文本中自动抽取实体和关系，生成交互式知识图谱可视化。

## 功能说明

1. **文本输入** — 粘贴任意文本（新闻、百科、技术文档等）
2. **智能抽取** — LLM 驱动的实体识别（NER）+ 关系抽取（RE）
3. **图谱可视化** — D3.js 力导向图，节点颜色区分实体类型，边表示关系
4. **交互探索** — 拖拽、缩放、悬停高亮、点击查看详情
5. **筛选过滤** — 按实体类型、置信度阈值筛选
6. **数据导出** — 支持 JSON / CSV / GraphML 格式

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | 原生 HTML/CSS/JS + D3.js v7（力导向图） |
| 后端 | Python FastAPI + networkx（图分析） |
| AI | LLM API（GPT-4o-mini）— NER + 关系推理 |

## 启动方式

### 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 前端

直接用浏览器打开 `frontend/index.html`，或用任意静态服务器：

```bash
cd frontend
python -m http.server 3000
```

访问 http://localhost:3000

## API 接口概要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/extract-knowledge-graph` | 文本 → 实体 + 关系抽取 |
| POST | `/api/analyze-graph-metrics` | 图谱拓扑指标分析 |
| GET | `/api/entity-types` | 获取支持的实体类型 |
| POST | `/api/export-graph` | 导出图谱数据 |

## 效果说明

输入一段包含人物、组织、地点等信息的文本，系统自动：
- 识别实体并分类（👤人物 🏢组织 📍地点 📅事件 💡概念）
- 推理实体间关系（任职于、位于、创办了等）
- 生成交互式力导向图，节点大小反映重要度，边粗细反映置信度
