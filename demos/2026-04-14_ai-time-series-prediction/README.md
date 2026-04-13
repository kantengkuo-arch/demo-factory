# 📈 AI 时间序列预测

上传历史时间序列数据（如股价、气温、销量等），通过 AI 模型预测未来趋势并交互式可视化展示。

## 技术栈

- **核心算法**：Prophet（自动季节性分析）、LSTM（深度学习非线性拟合）、线性回归（快速基线）
- **后端**：FastAPI + pandas + PyTorch + Prophet + scikit-learn
- **前端**：HTML + CSS + JavaScript（暗色主题）

## 快速启动

```bash
cd backend
pip install -r requirements.txt
python3 app.py
```

然后用浏览器打开 `frontend/index.html`，后端运行在 http://localhost:8000。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /api/upload | 上传 CSV 文件并自动解析 |
| POST | /api/predict | 执行时间序列预测 |
| GET | /api/models | 获取可用预测模型列表 |
