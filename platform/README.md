# 🏭 Demo Factory — Portfolio Management Panel

Demo Factory 的管理面板，用于浏览、启动/停止和评估所有已产出的 AI Demo。

## 架构

```
platform/
├── backend/       # FastAPI 后端 (port 7000)
│   ├── app.py
│   └── requirements.txt
└── frontend/      # 单页前端 (纯 HTML + Chart.js)
    └── index.html
```

- **Backend**：提供 Demo 列表、详情、启停控制、统计等 REST API
- **Frontend**：暗色主题单页面，展示 Demo 卡片、评分趋势图、运行状态

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/demos` | 获取所有 Demo 列表（含运行状态） |
| GET | `/api/demos/{slug}` | 获取单个 Demo 详情 + README |
| POST | `/api/demos/{slug}/start` | 启动 Demo（自动分配端口） |
| POST | `/api/demos/{slug}/stop` | 停止 Demo |
| GET | `/api/stats` | 统计数据（总数、平均分、趋势） |

## 快速启动

```bash
# 安装依赖
cd platform/backend
pip install -r requirements.txt

# 启动后端 (默认 port 7000)
python app.py

# 前端直接用浏览器打开
open platform/frontend/index.html
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FACTORY_ROOT` | `/root/projects/demo-factory` | 工厂根目录，用于定位 `demos/` 下的注册表和项目文件 |

## 工作原理

1. 后端从 `demos/_registry.json` 读取已注册的 Demo 列表
2. 启动 Demo 时自动创建 venv、安装依赖、分配端口、用 uvicorn 拉起
3. 前端通过 REST API 获取数据，展示卡片和评分趋势图
