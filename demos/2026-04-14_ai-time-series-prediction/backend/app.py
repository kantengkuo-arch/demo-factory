"""
AI 时间序列预测 — 用 Google TimesFM 预测趋势
后端服务：接收时序 CSV 数据，使用 TimesFM / 统计回退模型进行预测
"""

import os
import uuid
import io
import math
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ========== 应用初始化 ==========

app = FastAPI(title="AI 时间序列预测", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 文件上传临时存储（内存中，key: file_id -> DataFrame）
uploaded_data: dict[str, dict] = {}

# ========== TimesFM 模型加载 ==========
# 尝试加载 Google TimesFM，如果不可用则回退到统计方法

timesfm_model = None
TIMESFM_AVAILABLE = False

try:
    import timesfm

    # 延迟加载模型，在首次预测时初始化
    _timesfm_initialized = False

    def _ensure_timesfm_loaded():
        """按需加载 TimesFM 模型，避免启动时阻塞"""
        global timesfm_model, _timesfm_initialized, TIMESFM_AVAILABLE
        if _timesfm_initialized:
            return
        try:
            tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    per_core_batch_size=32,
                    horizon_len=128,
                    backend="cpu",  # 默认 CPU，避免 GPU 依赖问题
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-200m-pytorch",
                ),
            )
            timesfm_model = tfm
            TIMESFM_AVAILABLE = True
            print("✅ TimesFM 模型加载成功")
        except Exception as e:
            print(f"⚠️ TimesFM 模型加载失败，将使用统计回退方法: {e}")
            TIMESFM_AVAILABLE = False
        _timesfm_initialized = True

    print("📦 timesfm 包已安装，模型将在首次预测时加载")
except ImportError:
    print("⚠️ timesfm 未安装，将使用统计回退方法（numpy 线性回归 + 趋势分解）")

    def _ensure_timesfm_loaded():
        pass


# ========== 统计回退预测方法 ==========

def _fallback_predict(values: np.ndarray, horizon: int) -> dict:
    """
    统计回退预测方法：使用线性回归趋势 + 季节性分解
    当 TimesFM 不可用时使用此方法
    """
    n = len(values)
    x = np.arange(n)

    # 线性回归拟合趋势
    coeffs = np.polyfit(x, values, deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    trend = slope * np.arange(n, n + horizon) + intercept

    # 尝试提取简单季节性（如果数据足够长）
    seasonal = np.zeros(horizon)
    residuals = values - (slope * x + intercept)

    # 尝试检测周期（7天、30天等常见周期）
    for period in [7, 14, 30, 12]:
        if n >= period * 2:
            # 计算周期内的平均残差
            period_means = np.array([
                np.mean(residuals[i::period]) for i in range(period)
            ])
            seasonal = np.tile(period_means, (horizon // period) + 1)[:horizon]
            break

    predictions = trend + seasonal

    # 计算置信区间（基于残差标准差）
    residual_std = np.std(residuals) if len(residuals) > 1 else 0.0
    confidence = []
    for i in range(horizon):
        # 置信区间随预测步数增大
        width = residual_std * (1.0 + 0.1 * i)
        confidence.append({
            "lower": float(predictions[i] - 1.96 * width),
            "upper": float(predictions[i] + 1.96 * width),
        })

    return {
        "predictions": [float(v) for v in predictions],
        "confidence_intervals": confidence,
    }


def _timesfm_predict(values: np.ndarray, horizon: int, freq_str: str) -> dict:
    """
    使用 TimesFM 模型进行预测
    """
    _ensure_timesfm_loaded()

    if not TIMESFM_AVAILABLE or timesfm_model is None:
        return _fallback_predict(values, horizon)

    # TimesFM 频率映射
    freq_map = {
        "D": 0,  # daily
        "W": 1,  # weekly
        "M": 2,  # monthly
        "H": 3,  # hourly
    }
    freq_id = freq_map.get(freq_str, 0)

    try:
        # TimesFM 2.0 API: forecast 方法
        forecast_input = [values.astype(np.float32)]
        point_forecast, experimental_quantile_forecast = timesfm_model.forecast(
            forecast_input,
            freq=[freq_id],
        )

        preds = point_forecast[0][:horizon].tolist()

        # 构建置信区间
        confidence = []
        if experimental_quantile_forecast is not None and len(experimental_quantile_forecast) > 0:
            quantiles = experimental_quantile_forecast[0]
            for i in range(min(horizon, len(preds))):
                confidence.append({
                    "lower": float(quantiles[i][0]) if quantiles.shape[1] > 0 else preds[i] * 0.9,
                    "upper": float(quantiles[i][-1]) if quantiles.shape[1] > 0 else preds[i] * 1.1,
                })
        else:
            # 没有分位数输出时，用简单估计
            std = np.std(values[-min(30, len(values)):])
            for i in range(horizon):
                width = std * (1.0 + 0.05 * i)
                confidence.append({
                    "lower": float(preds[i] - 1.96 * width),
                    "upper": float(preds[i] + 1.96 * width),
                })

        return {
            "predictions": [float(v) for v in preds],
            "confidence_intervals": confidence,
        }
    except Exception as e:
        print(f"⚠️ TimesFM 预测失败，回退到统计方法: {e}")
        return _fallback_predict(values, horizon)


# ========== 内置示例数据生成 ==========

def _generate_stock_data() -> pd.DataFrame:
    """生成模拟股票价格数据（180天）"""
    np.random.seed(42)
    dates = pd.date_range(start="2025-10-01", periods=180, freq="D")
    # 模拟股票：起始价格100，带趋势和波动
    price = 100.0
    prices = []
    for i in range(180):
        # 轻微上升趋势 + 随机波动
        change = np.random.normal(0.05, 1.5)
        price = max(price + change, 10)  # 不跌破10
        prices.append(round(price, 2))
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "price": prices})


def _generate_temperature_data() -> pd.DataFrame:
    """生成模拟气温数据（365天）"""
    np.random.seed(123)
    dates = pd.date_range(start="2025-04-01", periods=365, freq="D")
    # 正弦波模拟季节性 + 随机噪声
    day_of_year = np.arange(365)
    base_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    noise = np.random.normal(0, 2, 365)
    temps = np.round(base_temp + noise, 1)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "temperature": temps})


def _generate_sales_data() -> pd.DataFrame:
    """生成模拟电商销量数据（90天）"""
    np.random.seed(456)
    dates = pd.date_range(start="2026-01-01", periods=90, freq="D")
    # 基础销量 + 周末高峰 + 增长趋势 + 随机波动
    base = 200
    sales = []
    for i, d in enumerate(dates):
        trend = i * 0.8  # 每天增长 0.8 单
        weekend_boost = 50 if d.dayofweek >= 5 else 0  # 周末加成
        noise = np.random.normal(0, 20)
        val = max(int(base + trend + weekend_boost + noise), 50)
        sales.append(val)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": sales})


# 缓存示例数据
DEMO_DATASETS = {
    "stock": {
        "generator": _generate_stock_data,
        "description": "模拟科技股票价格数据（2025年10月 - 2026年3月，180天日线）",
        "recommended_horizon": 30,
    },
    "temperature": {
        "generator": _generate_temperature_data,
        "description": "模拟城市气温数据（2025年4月 - 2026年3月，365天日均温度）",
        "recommended_horizon": 30,
    },
    "sales": {
        "generator": _generate_sales_data,
        "description": "模拟电商店铺日销量数据（2026年1-3月，90天）",
        "recommended_horizon": 14,
    },
}


# ========== 工具函数 ==========

def _detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """自动检测 DataFrame 中的日期时间列"""
    # 优先检查列名包含常见日期关键词的列
    date_keywords = ["date", "time", "timestamp", "日期", "时间", "dt"]
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                pd.to_datetime(df[col])
                return col
            except (ValueError, TypeError):
                continue

    # 尝试每一列是否可以解析为日期
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].head(10))
                return col
            except (ValueError, TypeError):
                continue

    return None


def _detect_frequency(dates: pd.DatetimeIndex) -> str:
    """自动检测时序数据的频率"""
    if len(dates) < 2:
        return "D"
    # 计算中位数间隔
    diffs = dates[1:] - dates[:-1]
    median_diff = pd.Timedelta(np.median([d.total_seconds() for d in diffs]), unit="s")

    if median_diff <= pd.Timedelta(hours=2):
        return "H"
    elif median_diff <= pd.Timedelta(days=2):
        return "D"
    elif median_diff <= pd.Timedelta(days=10):
        return "W"
    else:
        return "M"


# ========== API 路由 ==========

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传时序数据文件（CSV 格式）
    解析 CSV 内容，返回列信息和预览数据
    """
    try:
        # 验证文件类型
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="仅支持 CSV 格式文件")

        # 读取文件内容
        content = await file.read()

        # 尝试多种编码解析
        df = None
        for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        if df is None:
            raise HTTPException(status_code=400, detail="无法解析 CSV 文件，请检查文件编码和格式")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV 文件为空")

        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="CSV 至少需要两列（时间列和数值列）")

        # 生成唯一文件 ID
        file_id = str(uuid.uuid4())[:8]

        # 存储解析后的数据
        uploaded_data[file_id] = {
            "df": df,
            "filename": file.filename,
            "uploaded_at": datetime.datetime.now().isoformat(),
        }

        # 生成预览数据（前 5 行）
        preview = df.head(5).to_dict(orient="records")

        return {
            "file_id": file_id,
            "columns": list(df.columns),
            "rows_count": len(df),
            "preview": preview,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


class PredictRequest(BaseModel):
    """预测请求参数"""
    file_id: str
    time_column: str
    value_column: str
    horizon: int
    frequency: Optional[str] = None


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    执行时序预测
    根据上传的数据和配置参数，使用 TimesFM 或统计回退方法进行预测
    """
    try:
        # 验证 file_id
        if req.file_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="未找到对应的上传文件，请重新上传")

        df = uploaded_data[req.file_id]["df"]

        # 验证列名
        if req.time_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"时间列 '{req.time_column}' 不存在，可用列: {list(df.columns)}"
            )
        if req.value_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"数值列 '{req.value_column}' 不存在，可用列: {list(df.columns)}"
            )

        # 验证预测步数
        if req.horizon < 1 or req.horizon > 365:
            raise HTTPException(status_code=400, detail="预测步数 horizon 必须在 1-365 之间")

        # 解析时间列
        try:
            dates = pd.to_datetime(df[req.time_column])
        except Exception:
            raise HTTPException(status_code=400, detail=f"无法将 '{req.time_column}' 列解析为日期格式")

        # 解析数值列
        try:
            values = pd.to_numeric(df[req.value_column], errors="coerce").dropna().values
        except Exception:
            raise HTTPException(status_code=400, detail=f"无法将 '{req.value_column}' 列解析为数值")

        if len(values) < 10:
            raise HTTPException(status_code=400, detail="有效数据点不足 10 个，无法进行有意义的预测")

        # 检测或使用指定频率
        freq_str = req.frequency if req.frequency else _detect_frequency(pd.DatetimeIndex(dates))

        # 执行预测
        result = _timesfm_predict(values, req.horizon, freq_str)

        # 生成预测时间戳
        freq_map = {"D": "D", "W": "W", "M": "MS", "H": "h"}
        pd_freq = freq_map.get(freq_str, "D")
        last_date = dates.max()
        future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(pd_freq),
                                     periods=req.horizon, freq=pd_freq)
        timestamps = [d.strftime("%Y-%m-%d") for d in future_dates]

        # 模型信息
        model_info = {
            "engine": "timesfm-2.0" if TIMESFM_AVAILABLE else "statistical-fallback",
            "description": "Google TimesFM 预训练时间序列基础模型" if TIMESFM_AVAILABLE
                           else "统计回退方法（线性趋势 + 季节性分解）",
            "data_points_used": int(len(values)),
            "frequency_detected": freq_str,
        }

        return {
            "predictions": result["predictions"],
            "timestamps": timestamps,
            "confidence_intervals": result["confidence_intervals"],
            "model_info": model_info,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get("/api/demo-data/{dataset_name}")
async def get_demo_data(dataset_name: str):
    """
    获取内置示例数据集
    支持的数据集：stock（股票）、temperature（气温）、sales（销量）
    """
    try:
        if dataset_name not in DEMO_DATASETS:
            raise HTTPException(
                status_code=404,
                detail=f"未找到数据集 '{dataset_name}'，可用数据集: {list(DEMO_DATASETS.keys())}"
            )

        dataset_info = DEMO_DATASETS[dataset_name]
        df = dataset_info["generator"]()

        # 同时将 demo 数据存入 uploaded_data，方便后续直接用于预测
        file_id = f"demo-{dataset_name}"
        uploaded_data[file_id] = {
            "df": df,
            "filename": f"{dataset_name}_demo.csv",
            "uploaded_at": datetime.datetime.now().isoformat(),
        }

        return {
            "data": df.to_dict(orient="records"),
            "description": dataset_info["description"],
            "recommended_horizon": dataset_info["recommended_horizon"],
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取示例数据失败: {str(e)}")


# ========== 启动入口 ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
