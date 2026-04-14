# 🏭 Demo Factory Portfolio

> Demo Factory 的管理面板——展示工厂产出、按需启动 Demo、查看自进化曲线和 Agent 协作时间线。

## 快速启动

### 1. 启动后端（端口 7000）

```bash
cd ~/projects/demo-factory/platform/backend
pip3 install -r requirements.txt
nohup python3 app.py > /tmp/portfolio.log 2>&1 &
```

### 2. 启动前端（端口 7001）

```bash
cd ~/projects/demo-factory/platform/frontend
nohup python3 -m http.server 7001 > /tmp/portfolio-fe.log 2>&1 &
```

### 3. 在你的电脑上访问

在你的 Windows 终端运行（不是 ECS 上）：

```bash
ssh -L 7000:localhost:7000 -L 7001:localhost:7001 root@你的ECS公网IP
```

浏览器打开 `http://localhost:7001`

## 功能

### Demo 列表页
- 统计卡片：总 Demo 数、平均评分、运行中数量、热门技术栈
- Demo 网格：每个 Demo 一张卡片，显示名称、描述、评分、运行状态
- 按需启动/停止：点击按钮启动 Demo，启动后可一键打开体验
- 评分趋势图：折线图展示工厂自进化曲线

### 协作时间线页
- 从 GitHub Issue Timeline API 自动抓取 label 变更事件
- 水平条形图展示每个 Demo 从选题到完成的各阶段耗时

### 竞技场页
- 同方向 Demo 自动对比评分
- VS 对战卡片展示，高分标绿

## 架构
platform/
├── backend/
│   ├── app.py              # FastAPI 主入口（端口 7000）
│   ├── timeline.py          # 时间线数据
│   ├── arena.py             # 竞技场数据
│   └── requirements.txt
└── frontend/
└── index.html           # 单页应用（Chart.js + 原生 JS）

## API 列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| GET | /api/demos | 所有 Demo 列表 + 运行状态 |
| GET | /api/demos/{slug} | 单个 Demo 详情 + README |
| POST | /api/demos/{slug}/start | 按需启动（返回端口号） |
| POST | /api/demos/{slug}/stop | 停止运行 |
| GET | /api/stats | 统计数据 + 评分趋势 |
| GET | /api/timeline | Agent 协作时间线 |
| GET | /api/arena | 竞技场对比数据 |

## 按需启动原理

- 每个 Demo 启动在独立端口（从 9001 递增）
- 优先用 Docker（如果有 Dockerfile），否则用 venv + uvicorn
- 运行状态存在内存中，Portfolio 重启后需重新启动 Demo

## 常用运维命令

```bash
# 查看后端日志
tail -f /tmp/portfolio.log

# 重启后端
pkill -f "python3 app.py" && cd ~/projects/demo-factory/platform/backend && nohup python3 app.py > /tmp/portfolio.log 2>&1 &

# 重启前端
pkill -f "http.server 7001" && cd ~/projects/demo-factory/platform/frontend && nohup python3 -m http.server 7001 > /tmp/portfolio-fe.log 2>&1 &
```
