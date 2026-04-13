"""
AI 时间序列预测 — 后端服务

提供 CSV 上传、自动解析、多模型预测（Prophet / LSTM / 线性回归）功能。
基于 FastAPI 构建，严格遵循 API 契约。
"""

import uuid
import io
import tempfile
import os
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# 应用初始化
# ============================================================

app = FastAPI(title="AI 时间序列预测", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存中缓存已上传的文件数据，key 为 file_id
_file_store: Dict[str, pd.DataFrame] = {}


# ============================================================
# 请求/响应模型
# ============================================================

class PredictRequest(BaseModel):
    """预测请求体"""
    file_id: str
    date_column: str
    value_column: str
    model_type: str  # prophet / lstm / linear
    forecast_steps: int


# ============================================================
# 工具函数
# ============================================================

def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    自动检测 DataFrame 中的日期列。
    优先按列名匹配，其次尝试解析内容。
    """
    # 常见日期列名
    date_keywords = ["date", "time", "datetime", "timestamp", "日期", "时间"]
    for col in df.columns:
        if col.lower().strip() in date_keywords:
            return col

    # 尝试解析每一列
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].head(10))
                return col
            except Exception:
                continue
    return None


def _detect_value_columns(df: pd.DataFrame, date_col: Optional[str]) -> List[str]:
    """返回所有数值列（排除日期列）"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if date_col and date_col in numeric_cols:
        numeric_cols.remove(date_col)
    return numeric_cols


def _prepare_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    准备用于预测的干净时间序列 DataFrame，包含 ds / y 两列。
    会排序、去重、去 NaN。
    """
    series = df[[date_col, value_col]].copy()
    series.columns = ["ds", "y"]
    series["ds"] = pd.to_datetime(series["ds"])
    series = series.dropna().drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    return series


# ============================================================
# 预测引擎
# ============================================================

def _predict_prophet(series: pd.DataFrame, steps: int) -> dict:
    """使用 Facebook Prophet 进行预测"""
    from prophet import Prophet

    m = Prophet(yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False)
    m.fit(series)

    # 推断频率
    freq = pd.infer_freq(series["ds"])
    if freq is None:
        freq = "D"

    future = m.make_future_dataframe(periods=steps, freq=freq)
    forecast = m.predict(future)

    # 历史数据
    hist = series[["ds", "y"]].values.tolist()
    historical_data = [[r[0].strftime("%Y-%m-%d"), float(r[1])] for r in hist]

    # 预测数据（仅未来部分）
    future_part = forecast.iloc[len(series):]
    predictions = []
    for _, row in future_part.iterrows():
        predictions.append([
            row["ds"].strftime("%Y-%m-%d"),
            round(float(row["yhat"]), 4),
            round(float(row["yhat_lower"]), 4),
            round(float(row["yhat_upper"]), 4),
        ])

    # 模型评估：在训练集上计算 MAE / MAPE
    train_pred = forecast.iloc[:len(series)]
    y_true = series["y"].values
    y_pred = train_pred["yhat"].values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))))

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "model_metrics": {"mae": round(mae, 4), "mape": round(mape, 4)},
        "status": "success",
    }


def _predict_lstm(series: pd.DataFrame, steps: int) -> dict:
    """使用简易 LSTM 网络进行预测"""
    import torch
    import torch.nn as nn

    values = series["y"].values.astype(np.float64)
    dates = series["ds"].values

    # 归一化
    v_min, v_max = values.min(), values.max()
    v_range = v_max - v_min if v_max != v_min else 1.0
    normed = (values - v_min) / v_range

    # 构造滑动窗口
    lookback = min(30, len(normed) - 1)
    if lookback < 3:
        raise HTTPException(status_code=400, detail="数据点太少，LSTM 至少需要 4 个数据点")

    X, Y = [], []
    for i in range(len(normed) - lookback):
        X.append(normed[i: i + lookback])
        Y.append(normed[i + lookback])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, lookback, 1)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    # 简易 LSTM 模型
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = SimpleLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 训练
    epochs = 100
    model.train()
    for _ in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 多步预测
    model.eval()
    current_window = normed[-lookback:].tolist()
    preds_normed = []
    with torch.no_grad():
        for _ in range(steps):
            inp = torch.tensor([current_window[-lookback:]], dtype=torch.float32).unsqueeze(-1)
            p = model(inp).item()
            preds_normed.append(p)
            current_window.append(p)

    preds_raw = np.array(preds_normed) * v_range + v_min

    # 推断频率并生成未来日期
    freq = pd.infer_freq(series["ds"])
    if freq is None:
        freq = "D"
    last_date = pd.Timestamp(dates[-1])
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]

    # 简单置信区间（用训练集残差标准差）
    with torch.no_grad():
        train_preds = model(X).numpy() * v_range + v_min
    train_true = values[lookback:]
    residual_std = float(np.std(train_true - train_preds))

    historical_data = [
        [pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)] for d, v in zip(dates, values)
    ]
    predictions = []
    for i, d in enumerate(future_dates):
        yhat = round(float(preds_raw[i]), 4)
        lower = round(yhat - 1.96 * residual_std, 4)
        upper = round(yhat + 1.96 * residual_std, 4)
        predictions.append([d.strftime("%Y-%m-%d"), yhat, lower, upper])

    mae = float(np.mean(np.abs(train_true - train_preds)))
    mape = float(np.mean(np.abs((train_true - train_preds) / (train_true + 1e-9))))

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "model_metrics": {"mae": round(mae, 4), "mape": round(mape, 4)},
        "status": "success",
    }


def _predict_linear(series: pd.DataFrame, steps: int) -> dict:
    """使用线性回归进行预测"""
    from sklearn.linear_model import LinearRegression

    values = series["y"].values.astype(np.float64)
    dates = series["ds"].values
    n = len(values)

    # 特征：简单递增序号
    X_train = np.arange(n).reshape(-1, 1)
    y_train = values

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 训练集预测 & 指标
    y_pred_train = model.predict(X_train)
    mae = float(np.mean(np.abs(y_train - y_pred_train)))
    mape = float(np.mean(np.abs((y_train - y_pred_train) / (y_train + 1e-9))))
    residual_std = float(np.std(y_train - y_pred_train))

    # 推断频率
    freq = pd.infer_freq(series["ds"])
    if freq is None:
        freq = "D"
    last_date = pd.Timestamp(dates[-1])
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]

    X_future = np.arange(n, n + steps).reshape(-1, 1)
    y_future = model.predict(X_future)

    historical_data = [
        [pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)] for d, v in zip(dates, values)
    ]
    predictions = []
    for i, d in enumerate(future_dates):
        yhat = round(float(y_future[i]), 4)
        lower = round(yhat - 1.96 * residual_std, 4)
        upper = round(yhat + 1.96 * residual_std, 4)
        predictions.append([d.strftime("%Y-%m-%d"), yhat, lower, upper])

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "model_metrics": {"mae": round(mae, 4), "mape": round(mape, 4)},
        "status": "success",
    }


# ============================================================
# 路由
# ============================================================

@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    上传 CSV 文件并返回解析结果。
    自动检测日期列和数值列，返回预览数据和 file_id。
    """
    try:
        content = await file.read()
        # 尝试多种编码
        for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise HTTPException(status_code=400, detail="无法识别文件编码，请使用 UTF-8 编码的 CSV")

        if df.empty or len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="CSV 文件至少需要 2 列（日期列 + 数值列）")

        date_col = _detect_date_column(df)
        value_cols = _detect_value_columns(df, date_col)

        if not value_cols:
            raise HTTPException(status_code=400, detail="未检测到数值列，请确认 CSV 格式正确")

        file_id = str(uuid.uuid4())
        _file_store[file_id] = df

        # 预览：最多前 5 行
        preview = df.head(5).values.tolist()

        return {
            "columns": df.columns.tolist(),
            "rows_count": len(df),
            "date_column": date_col,
            "value_columns": value_cols,
            "preview": preview,
            "file_id": file_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析 CSV 失败：{str(e)}")


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    执行时间序列预测。
    支持 prophet / lstm / linear 三种模型。
    """
    # 校验 file_id
    if req.file_id not in _file_store:
        raise HTTPException(status_code=404, detail="文件未找到，请重新上传")

    df = _file_store[req.file_id]

    # 校验列名
    if req.date_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"日期列 '{req.date_column}' 不存在")
    if req.value_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"数值列 '{req.value_column}' 不存在")
    if req.forecast_steps < 1 or req.forecast_steps > 365:
        raise HTTPException(status_code=400, detail="预测步数应在 1~365 之间")

    try:
        series = _prepare_series(df, req.date_column, req.value_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"数据预处理失败：{str(e)}")

    if len(series) < 3:
        raise HTTPException(status_code=400, detail="有效数据点不足 3 条，无法预测")

    # 路由到对应模型
    try:
        if req.model_type == "prophet":
            result = _predict_prophet(series, req.forecast_steps)
        elif req.model_type == "lstm":
            result = _predict_lstm(series, req.forecast_steps)
        elif req.model_type == "linear":
            result = _predict_linear(series, req.forecast_steps)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型：{req.model_type}，可选 prophet/lstm/linear")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")


@app.get("/api/models")
async def list_models():
    """获取可用预测模型列表"""
    return {
        "models": [
            {
                "name": "prophet",
                "display_name": "Prophet (自动)",
                "description": "适合有季节性的数据",
            },
            {
                "name": "lstm",
                "display_name": "LSTM (深度学习)",
                "description": "适合复杂非线性模式",
            },
            {
                "name": "linear",
                "display_name": "线性回归",
                "description": "快速基线预测",
            },
        ]
    }


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
