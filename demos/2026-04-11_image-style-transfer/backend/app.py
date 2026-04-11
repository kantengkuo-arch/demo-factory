"""
图片风格转换工具 — 后端服务

使用经典 Neural Style Transfer (VGG19) 实现图片风格转换。
支持自定义风格图片上传和预设风格模板（梵高、毕加索、中国水墨等）。
异步任务处理，支持进度查询。
"""

import os
import uuid
import time
import threading
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# ==================== 配置 ====================

# 上传文件和结果图片的存储目录
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
STATIC_DIR = Path("static")
PRESET_DIR = STATIC_DIR / "presets"
PREVIEW_DIR = STATIC_DIR / "previews"

# 确保目录存在
for d in [UPLOAD_DIR, RESULT_DIR, STATIC_DIR, PRESET_DIR, PREVIEW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 图片处理尺寸（CPU 上用 256 保证速度，GPU 可提到 512）
IMAGE_SIZE = 256 if not torch.cuda.is_available() else 512

# 经典 NST 默认迭代次数（CPU 上用 50 次，GPU 可提到 300）
NST_ITERATIONS = 50 if not torch.cuda.is_available() else 300

# 预设风格定义
PRESET_STYLES = {
    "vangogh": {
        "id": "vangogh",
        "name": "梵高星夜",
        "preview_url": "/static/previews/vangogh.jpg",
        "description": "梵高《星夜》的漩涡笔触风格",
    },
    "picasso": {
        "id": "picasso",
        "name": "毕加索立体派",
        "preview_url": "/static/previews/picasso.jpg",
        "description": "毕加索立体主义的几何分割风格",
    },
    "chinese_ink": {
        "id": "chinese_ink",
        "name": "中国水墨",
        "preview_url": "/static/previews/chinese_ink.jpg",
        "description": "传统中国水墨画的写意风格",
    },
    "monet": {
        "id": "monet",
        "name": "莫奈印象派",
        "preview_url": "/static/previews/monet.jpg",
        "description": "莫奈印象派的光影色彩风格",
    },
    "ukiyoe": {
        "id": "ukiyoe",
        "name": "浮世绘",
        "preview_url": "/static/previews/ukiyoe.jpg",
        "description": "日本浮世绘的平面装饰风格",
    },
}

# ==================== 任务存储 ====================
# 内存中的任务状态（生产环境应使用 Redis 等）
tasks: dict = {}

# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="图片风格转换工具",
    description="基于 VGG19 的 Neural Style Transfer API",
    version="1.0.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录，用于访问预设风格预览图和结果图片
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")


# ==================== 工具函数 ====================


def get_device() -> torch.device:
    """获取可用的计算设备（GPU 优先）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(image_path: str, size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    加载图片并转换为 [0,1] 范围的 tensor（不做 ImageNet 标准化，
    标准化由模型内部的 Normalization 层统一处理）。

    Args:
        image_path: 图片文件路径
        size: 目标尺寸（短边）

    Returns:
        形状为 (1, 3, H, W) 的 tensor，值域 [0, 1]
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        # 不在这里做 Normalize，由模型内部的 Normalization 层处理
    ])
    # 增加 batch 维度
    return transform(image).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    将 [0,1] 范围的 tensor 转换回 PIL Image。

    Args:
        tensor: 形状为 (1, 3, H, W) 的 tensor，值域 [0, 1]

    Returns:
        PIL Image 对象
    """
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image.clamp(0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(image)


# ==================== VGG19 特征提取 ====================


class ContentLoss(nn.Module):
    """内容损失层：衡量生成图片与内容图片在特征空间的距离"""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    """风格损失层：使用 Gram 矩阵衡量风格特征的匹配程度"""

    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = self._gram_matrix(target_feature).detach()

    @staticmethod
    def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
        """计算 Gram 矩阵，用于捕获风格特征的相关性"""
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gram = self._gram_matrix(x)
        self.loss = nn.functional.mse_loss(gram, self.target)
        return x


class Normalization(nn.Module):
    """对输入进行 ImageNet 标准化（在模型内部处理）"""

    def __init__(self, device: torch.device):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def build_style_transfer_model(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    device: torch.device,
):
    """
    构建风格转换模型：在 VGG19 的特定层插入内容损失和风格损失。

    Args:
        content_img: 内容图片 tensor
        style_img: 风格图片 tensor
        device: 计算设备

    Returns:
        (model, style_losses, content_losses) 三元组
    """
    # 使用预训练 VGG19 的特征提取部分
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    # 内容和风格的目标层
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    normalization = Normalization(device).to(device)
    content_losses = []
    style_losses = []

    # 逐层构建模型，在目标层后插入损失层
    model = nn.Sequential(normalization)
    i = 0  # 卷积层计数器

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            name = f"layer_{i}"

        model.add_module(name, layer)

        # 在内容目标层后插入 ContentLoss
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        # 在风格目标层后插入 StyleLoss
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # 裁剪掉最后一个损失层之后的多余层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[: i + 1]

    return model, style_losses, content_losses


# ==================== 风格转换核心逻辑 ====================


def run_style_transfer(
    content_path: str,
    style_path: str,
    output_path: str,
    task_id: str,
    strength: float = 0.8,
    num_iterations: int = NST_ITERATIONS,
):
    """
    执行经典 Neural Style Transfer。

    在后台线程中运行，通过 tasks 字典更新进度。

    Args:
        content_path: 内容图片路径
        style_path: 风格图片路径
        output_path: 输出图片路径
        task_id: 任务 ID，用于进度更新
        strength: 风格强度 (0.1 - 1.0)
        num_iterations: 优化迭代次数
    """
    try:
        device = get_device()
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 5

        # 加载图片
        content_img = load_image(content_path).to(device)
        style_img = load_image(style_path).to(device)
        tasks[task_id]["progress"] = 10

        # 用内容图片初始化生成图片
        input_img = content_img.clone().requires_grad_(True)

        # 构建模型
        model, style_losses, content_losses = build_style_transfer_model(
            content_img, style_img, device
        )
        tasks[task_id]["progress"] = 15

        # 风格权重根据 strength 参数调整
        # strength 越大，风格损失权重越高
        style_weight = int(1e5 * (strength * 2))  # 基础值 * 强度因子
        content_weight = 1

        # 使用 L-BFGS 优化器（适合风格转换任务）
        optimizer = optim.LBFGS([input_img])

        iteration = [0]  # 用列表包装以在闭包中修改

        while iteration[0] < num_iterations:

            def closure():
                """L-BFGS 优化器需要的闭包函数"""
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)

                style_score = sum(sl.loss for sl in style_losses) * style_weight
                content_score = sum(cl.loss for cl in content_losses) * content_weight
                loss = style_score + content_score
                loss.backward()

                iteration[0] += 1

                # 更新进度（15% ~ 95%）
                progress = 15 + int(80 * iteration[0] / num_iterations)
                tasks[task_id]["progress"] = min(progress, 95)

                return loss

            optimizer.step(closure)

        # 确保最终像素值在合法范围内
        input_img.data.clamp_(0, 1)

        # 保存结果
        result_image = tensor_to_image(input_img)
        result_image.save(output_path, quality=95)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        # 生成可访问的 URL
        result_filename = Path(output_path).name
        tasks[task_id]["result_url"] = f"/results/{result_filename}"

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error_message"] = str(e)


# ==================== API 路由 ====================


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_images(
    content_image: UploadFile = File(...),
    style_image: Optional[UploadFile] = File(None),
):
    """
    上传图片文件。

    - content_image: 必需，内容图片
    - style_image: 可选，风格图片（使用预设风格时可不传）

    返回上传后的图片 ID。
    """
    # 验证内容图片类型
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if content_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式: {content_image.content_type}，支持: {', '.join(allowed_types)}",
        )

    # 保存内容图片
    content_id = str(uuid.uuid4())
    content_ext = Path(content_image.filename or "image.jpg").suffix or ".jpg"
    content_path = UPLOAD_DIR / f"{content_id}{content_ext}"
    with open(content_path, "wb") as f:
        content = await content_image.read()
        f.write(content)

    result = {
        "content_id": content_id,
        "style_id": None,
        "status": "success",
    }

    # 保存风格图片（如果有）
    if style_image is not None:
        if style_image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的风格图片格式: {style_image.content_type}",
            )
        style_id = str(uuid.uuid4())
        style_ext = Path(style_image.filename or "image.jpg").suffix or ".jpg"
        style_path = UPLOAD_DIR / f"{style_id}{style_ext}"
        with open(style_path, "wb") as f:
            style_content = await style_image.read()
            f.write(style_content)
        result["style_id"] = style_id

    return result


@app.post("/api/style-transfer")
async def style_transfer(request: dict):
    """
    执行风格转换。

    请求体参数：
    - content_id: 内容图片 ID（由 /api/upload 返回）
    - style_source: 风格来源，"upload" 或 "preset"
    - style_id: 风格图片 ID（style_source 为 "upload" 时必需）
    - preset_style: 预设风格名（style_source 为 "preset" 时必需）
    - algorithm: 算法选择，"classic_nst" 或 "stable_diffusion"
    - strength: 风格强度 0.1-1.0，默认 0.8

    返回任务 ID 用于查询进度。
    """
    content_id = request.get("content_id")
    style_source = request.get("style_source")
    style_id = request.get("style_id")
    preset_style = request.get("preset_style")
    algorithm = request.get("algorithm", "classic_nst")
    strength = request.get("strength", 0.8)

    # 参数校验
    if not content_id:
        raise HTTPException(status_code=400, detail="缺少 content_id 参数")
    if not style_source or style_source not in ("upload", "preset"):
        raise HTTPException(status_code=400, detail="style_source 必须为 'upload' 或 'preset'")
    if strength < 0.1 or strength > 1.0:
        raise HTTPException(status_code=400, detail="strength 必须在 0.1 到 1.0 之间")

    # 查找内容图片
    content_path = _find_uploaded_file(content_id)
    if not content_path:
        raise HTTPException(status_code=404, detail=f"未找到内容图片: {content_id}")

    # 确定风格图片路径
    if style_source == "upload":
        if not style_id:
            raise HTTPException(status_code=400, detail="style_source 为 'upload' 时必须提供 style_id")
        style_path = _find_uploaded_file(style_id)
        if not style_path:
            raise HTTPException(status_code=404, detail=f"未找到风格图片: {style_id}")
    elif style_source == "preset":
        if not preset_style or preset_style not in PRESET_STYLES:
            available = ", ".join(PRESET_STYLES.keys())
            raise HTTPException(
                status_code=400,
                detail=f"无效的预设风格: {preset_style}，可选: {available}",
            )
        style_path = str(PRESET_DIR / f"{preset_style}.jpg")
        if not Path(style_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"预设风格图片文件不存在: {preset_style}（请确保 static/presets/ 下有对应图片）",
            )

    # 目前仅支持经典 NST 算法
    if algorithm == "stable_diffusion":
        raise HTTPException(
            status_code=501,
            detail="Stable Diffusion 风格转换尚未实现，请使用 classic_nst 算法",
        )

    # 创建异步任务
    task_id = str(uuid.uuid4())
    output_path = str(RESULT_DIR / f"{task_id}.jpg")

    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "result_url": None,
        "error_message": None,
        "created_at": time.time(),
    }

    # 在后台线程中执行风格转换（避免阻塞主线程）
    thread = threading.Thread(
        target=run_style_transfer,
        args=(str(content_path), str(style_path), output_path, task_id, strength),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "status": "success"}


@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    查询风格转换任务的进度。

    路径参数：
    - task_id: 任务 ID（由 /api/style-transfer 返回）

    返回任务状态、进度百分比、结果 URL（完成时）或错误信息（失败时）。
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"未找到任务: {task_id}")

    task = tasks[task_id]
    return {
        "status": task["status"],
        "progress": task["progress"],
        "result_url": task.get("result_url"),
        "error_message": task.get("error_message"),
    }


@app.get("/api/presets")
async def get_presets():
    """
    获取预设风格列表。

    返回所有可用的预设风格及其预览图 URL。
    """
    presets = [
        {
            "id": style["id"],
            "name": style["name"],
            "preview_url": style["preview_url"],
        }
        for style in PRESET_STYLES.values()
    ]
    return {"presets": presets, "status": "success"}


# ==================== 辅助函数 ====================


def _find_uploaded_file(file_id: str) -> Optional[Path]:
    """
    根据文件 ID 在上传目录中查找文件（支持多种扩展名）。

    Args:
        file_id: 上传时生成的 UUID

    Returns:
        文件路径，未找到返回 None
    """
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        path = UPLOAD_DIR / f"{file_id}{ext}"
        if path.exists():
            return path
    return None


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
