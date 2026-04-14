# 🔭 AI 时间序列预测 — 用 Google TimesFM 预测趋势

上传时序 CSV 数据，AI 自动预测未来趋势。支持 Google TimesFM 预训练模型，零样本预测股票、销量、气温等时序数据。

## 技术栈

- **核心算法**：Google TimesFM 2.0 预训练时间序列基础模型（可选），统计回退方法（线性趋势 + 季节性分解）
- **后端**：FastAPI + pandas + numpy
- **前端**：HTML + CSS + JavaScript（暗色主题）

## 快速启动

```bash
cd backend
pip install -r requirements.txt
python3 app.py
```

然后用浏览器打开 `frontend/index.html`，后端运行在 http://localhost:8000。

### 可选：启用 Google TimesFM 模型

```bash
pip install timesfm[torch]
```

安装后重启服务即可自动使用 TimesFM 进行预测。未安装时使用统计回退方法，同样可正常运行。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /api/upload | 上传 CSV 时序数据文件 |
| POST | /api/predict | 执行时序预测 |
| GET | /api/demo-data/{dataset_name} | 获取内置示例数据（stock/temperature/sales） |

## 内置示例数据

- **stock** — 模拟科技股票价格（180天日线）
- **temperature** — 模拟城市气温（365天日均温度）
- **sales** — 模拟电商日销量（90天）
