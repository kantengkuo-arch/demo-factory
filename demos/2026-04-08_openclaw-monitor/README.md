# 🐾 OpenClaw 多 Agent 协作监控面板

实时监控 OpenClaw 多 Agent 协作状态、系统资源、任务流转和日志的 Web 面板。

## 技术栈

- **后端**：FastAPI + WebSocket + psutil
- **前端**：HTML + CSS + JavaScript + Chart.js
- **通信**：WebSocket 实时推送（每 5 秒刷新）

## 功能模块

| 模块 | 说明 |
|------|------|
| 🤖 Agent 拓扑图 | Boss → Scout → Coder → Reviewer 流转可视化 |
| 🌐 Gateway 状态 | PID、运行时长、会话数 |
| 📋 任务流转 | GitHub Issues 实时状态和处理阶段 |
| 📊 系统资源 | CPU / 内存 / 磁盘使用率 + 实时曲线 |
| 🚨 告警系统 | 内存超限、磁盘不足、心跳超时、Gateway 崩溃 |
| 📋 日志 | Gateway 日志实时滚动，支持 Agent 过滤 |

## 快速启动

```bash
cd backend
pip install -r requirements.txt
python3 app.py
```

浏览器打开 `http://localhost:8001`

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/status` | 获取完整状态快照 |
| GET | `/api/agents` | 获取 Agent 状态 |
| GET | `/api/system` | 获取系统资源 |
| GET | `/api/logs?lines=50&agent=coder` | 获取日志（支持过滤） |
| GET | `/` | 前端页面 |
| WS | `/ws` | WebSocket 实时数据推送 |

## 项目结构

```
2026-04-08_openclaw-monitor/
├── backend/
│   ├── app.py              # FastAPI 后端服务
│   └── requirements.txt    # Python 依赖
├── frontend/
│   └── index.html          # 暗色主题前端面板
├── demo_meta.json          # Demo 元数据
└── README.md               # 项目说明
```

## 端口

默认使用 **8001** 端口（避免与其他 Demo 的 8000 冲突）。
