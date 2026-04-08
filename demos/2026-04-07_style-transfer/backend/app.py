"""
图片风格迁移 Demo — FastAPI 后端
使用 PyTorch + VGG19 实现神经风格迁移
支持 SSE 实时推送处理步骤状态
"""

import io
import os
import json
import uuid
import time
import base64
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn

app = FastAPI(title="🎨 图片风格迁移 Demo", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 输出目录
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备检测：优先用 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片尺寸限制（CPU 模式下用较小尺寸以加快处理）
IMG_SIZE = 512 if torch.cuda.is_available() else 256

# 预定义风格参数（不同风格使用不同的权重配比）
STYLE_PRESETS = {
    "oil_painting": {
        "name": "油画",
        "emoji": "🖼️",
        "description": "浓郁色彩的古典油画风格",
        "style_weight": 1e6,
        "content_weight": 1,
        "num_steps": 200,
        "style_layers_weights": {"conv_1": 1.0, "conv_2": 0.8, "conv_3": 0.6, "conv_4": 0.4, "conv_5": 0.2},
    },
    "sketch": {
        "name": "素描",
        "emoji": "✏️",
        "description": "黑白线条的铅笔素描效果",
        "style_weight": 1e5,
        "content_weight": 5,
        "num_steps": 150,
        "style_layers_weights": {"conv_1": 1.0, "conv_2": 1.0, "conv_3": 0.5, "conv_4": 0.2, "conv_5": 0.1},
    },
    "watercolor": {
        "name": "水彩",
        "emoji": "🎨",
        "description": "柔和透明的水彩画风格",
        "style_weight": 5e5,
        "content_weight": 2,
        "num_steps": 180,
        "style_layers_weights": {"conv_1": 0.5, "conv_2": 1.0, "conv_3": 1.0, "conv_4": 0.5, "conv_5": 0.2},
    },
    "impressionist": {
        "name": "印象派",
        "emoji": "🌅",
        "description": "光影交错的印象派画风",
        "style_weight": 8e5,
        "content_weight": 1,
        "num_steps": 200,
        "style_layers_weights": {"conv_1": 0.8, "conv_2": 0.8, "conv_3": 1.0, "conv_4": 0.8, "conv_5": 0.5},
    },
}

# ==================== 风格迁移核心算法 ====================

# 内容损失
class ContentLoss(nn.Module):
    """计算内容损失：生成图片与内容图片在特征层的 MSE"""
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


# Gram 矩阵：用于捕获风格纹理
def gram_matrix(x):
    """计算 Gram 矩阵，用于衡量特征图之间的相关性"""
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


# 风格损失
class StyleLoss(nn.Module):
    """计算风格损失：生成图片与风格图片在 Gram 矩阵上的 MSE"""
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x


# 图像归一化
class Normalization(nn.Module):
    """ImageNet 标准归一化"""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def load_image(image_bytes: bytes, max_size: int = IMG_SIZE) -> torch.Tensor:
    """加载图片并转换为 Tensor"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # 等比缩放
    ratio = max_size / max(image.size)
    new_size = tuple(int(d * ratio) for d in image.size)
    image = image.resize(new_size, Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """将 Tensor 转回 PIL Image"""
    image = tensor.cpu().clone().squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    return image


def generate_style_image(style_key: str, size: tuple) -> torch.Tensor:
    """
    根据风格类型生成程序化的风格参考图
    这样就不需要外部风格图片文件
    """
    w, h = size[3], size[2]
    np.random.seed(42)

    if style_key == "oil_painting":
        # 油画风格：浓郁色块 + 纹理
        img = np.zeros((h, w, 3), dtype=np.float32)
        for _ in range(200):
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            r = np.random.randint(5, 30)
            color = np.random.rand(3) * 0.8 + 0.2
            y_coords, x_coords = np.ogrid[-cy:h-cy, -cx:w-cx]
            mask = x_coords**2 + y_coords**2 <= r**2
            img[mask] = color
        # 添加画布纹理
        noise = np.random.rand(h, w, 3).astype(np.float32) * 0.1
        img = np.clip(img + noise, 0, 1)

    elif style_key == "sketch":
        # 素描风格：线条 + 阴影
        img = np.ones((h, w, 3), dtype=np.float32) * 0.95
        for _ in range(500):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            angle = np.random.rand() * np.pi
            length = np.random.randint(10, 60)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            x2 = np.clip(x2, 0, w-1)
            y2 = np.clip(y2, 0, h-1)
            # 画线
            num_points = max(abs(x2-x1), abs(y2-y1), 1)
            xs = np.linspace(x1, x2, num_points, dtype=int)
            ys = np.linspace(y1, y2, num_points, dtype=int)
            xs = np.clip(xs, 0, w-1)
            ys = np.clip(ys, 0, h-1)
            gray = np.random.rand() * 0.3
            img[ys, xs] = gray
        # 添加铅笔纹理
        noise = np.random.rand(h, w, 3).astype(np.float32) * 0.05
        img = np.clip(img - noise, 0, 1)

    elif style_key == "watercolor":
        # 水彩风格：渐变色块 + 晕染
        img = np.ones((h, w, 3), dtype=np.float32) * 0.95
        for _ in range(100):
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            r = np.random.randint(20, 80)
            color = np.random.rand(3) * 0.5 + 0.3
            y_coords, x_coords = np.ogrid[-cy:h-cy, -cx:w-cx]
            dist = np.sqrt(x_coords**2 + y_coords**2).astype(np.float32)
            mask = dist <= r
            alpha = np.clip(1.0 - dist / r, 0, 1)[..., np.newaxis] * 0.3
            blend = np.where(mask[..., np.newaxis], img * (1 - alpha) + color * alpha, img)
            img = blend.astype(np.float32)

    elif style_key == "impressionist":
        # 印象派风格：点彩 + 鲜艳色彩
        img = np.zeros((h, w, 3), dtype=np.float32)
        for _ in range(3000):
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            r = np.random.randint(2, 8)
            color = np.array([
                np.random.choice([0.9, 0.7, 0.3, 0.1]),
                np.random.choice([0.8, 0.5, 0.3, 0.2]),
                np.random.choice([0.9, 0.6, 0.4, 0.2]),
            ])
            y_coords, x_coords = np.ogrid[-cy:h-cy, -cx:w-cx]
            mask = x_coords**2 + y_coords**2 <= r**2
            img[mask] = color
    else:
        img = np.random.rand(h, w, 3).astype(np.float32)

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor


def build_model_and_losses(content_img, style_img, style_layers_weights):
    """
    构建风格迁移模型：在 VGG19 的特定层插入内容和风格损失模块
    """
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()

    # ImageNet 归一化参数
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    # 内容层
    content_layers = ["conv_4"]
    # 风格层
    style_layers = list(style_layers_weights.keys())

    i = 0  # conv 层计数
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
            continue

        model.add_module(name, layer)

        # 插入内容损失
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        # 插入风格损失
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append((style_loss, style_layers_weights[name]))

    # 裁剪最后一个损失层之后的网络
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i + 1]

    return model, content_losses, style_losses


def run_style_transfer(content_img, style_key, preset, step_callback=None):
    """
    执行风格迁移主流程
    step_callback: 可选回调函数 (step_id: str, message: str, details: dict|None) -> None
    返回风格化后的图片 Tensor
    """
    # 生成程序化风格参考图
    style_img = generate_style_image(style_key, content_img.size())
    if step_callback:
        step_callback("style_generated", "风格参考图已生成")

    # 输入图片初始化为内容图片的副本
    input_img = content_img.clone()

    if step_callback:
        step_callback("building_model", "正在构建 VGG19 模型并插入损失层...")

    model, content_losses, style_losses = build_model_and_losses(
        content_img, style_img, preset["style_layers_weights"]
    )

    if step_callback:
        step_callback("model_ready", f"模型构建完成，使用 {len(content_losses)} 个内容层 + {len(style_losses)} 个风格层")

    # 优化器直接优化像素值
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])

    num_steps = preset["num_steps"]
    style_weight = preset["style_weight"]
    content_weight = preset["content_weight"]

    step = [0]
    last_reported = [0]

    while step[0] < num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)

            # 计算内容损失
            c_loss = sum(cl.loss for cl in content_losses) * content_weight

            # 计算风格损失（带权重）
            s_loss = sum(sl.loss * w for sl, w in style_losses) * style_weight

            total_loss = c_loss + s_loss
            total_loss.backward()

            step[0] += 1

            # 每 10% 进度回调一次
            if step_callback:
                pct = int(step[0] / num_steps * 100)
                if pct >= last_reported[0] + 10 or step[0] == num_steps:
                    last_reported[0] = pct
                    step_callback("optimizing", f"迭代优化中 {step[0]}/{num_steps} ({pct}%)", {
                        "step": step[0],
                        "total_steps": num_steps,
                        "progress": pct,
                        "content_loss": round(c_loss.item(), 4),
                        "style_loss": round(s_loss.item(), 4),
                        "total_loss": round(total_loss.item(), 4),
                    })

            return total_loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    if step_callback:
        step_callback("optimization_done", "迭代优化完成")

    return input_img


# ==================== API 路由 ====================

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "img_size": IMG_SIZE,
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/styles")
async def get_styles():
    """获取所有可用的风格列表"""
    styles = []
    for key, preset in STYLE_PRESETS.items():
        styles.append({
            "key": key,
            "name": preset["name"],
            "emoji": preset["emoji"],
            "description": preset["description"],
        })
    return {"styles": styles}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传原始图片，返回文件 ID"""
    # 验证文件类型
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB 限制
        raise HTTPException(status_code=400, detail="图片大小不能超过 10MB")

    # 保存上传文件
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "image.png")[1] or ".png"
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    with open(filepath, "wb") as f:
        f.write(content)

    # 获取图片信息
    img = Image.open(io.BytesIO(content))
    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "width": img.width,
        "height": img.height,
        "ext": ext,
    }


@app.post("/style_transfer")
async def style_transfer(file_id: str = Form(...), style: str = Form(...)):
    """
    执行风格迁移
    参数：
    - file_id: 上传图片返回的文件 ID
    - style: 风格 key（oil_painting / sketch / watercolor / impressionist）
    """
    # 验证风格参数
    if style not in STYLE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的风格：{style}，可选：{list(STYLE_PRESETS.keys())}"
        )

    # 查找上传的文件
    upload_file = None
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(file_id):
            upload_file = os.path.join(UPLOAD_DIR, fname)
            break

    if not upload_file or not os.path.exists(upload_file):
        raise HTTPException(status_code=404, detail="图片未找到，请重新上传")

    # 读取图片
    with open(upload_file, "rb") as f:
        image_bytes = f.read()

    preset = STYLE_PRESETS[style]

    # 执行风格迁移
    start_time = time.time()
    content_img = load_image(image_bytes)
    output_tensor = run_style_transfer(content_img, style, preset)
    elapsed = round(time.time() - start_time, 2)

    # 保存结果
    output_image = tensor_to_image(output_tensor)
    output_id = str(uuid.uuid4())
    output_path = os.path.join(OUTPUT_DIR, f"{output_id}.png")
    output_image.save(output_path, "PNG")

    return {
        "output_id": output_id,
        "style": style,
        "style_name": preset["name"],
        "processing_time": elapsed,
        "device": str(DEVICE),
        "download_url": f"/download/{output_id}.png",
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """下载处理后的风格化图片"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(filepath, media_type="image/png", filename=filename)


# ==================== SSE 步骤定义 ====================

# 风格迁移处理流水线的 6 个步骤
TRANSFER_STEPS = [
    {"step": 1, "total": 6, "name": "接收图片", "emoji": "📤", "description": "读取上传的内容图和风格图"},
    {"step": 2, "total": 6, "name": "图像预处理", "emoji": "🔧", "description": "缩放至统一尺寸并归一化"},
    {"step": 3, "total": 6, "name": "VGG19 特征提取", "emoji": "🧠", "description": "提取内容特征与风格特征"},
    {"step": 4, "total": 6, "name": "Gram 矩阵计算", "emoji": "📐", "description": "计算风格纹理的 Gram 矩阵"},
    {"step": 5, "total": 6, "name": "L-BFGS 迭代优化", "emoji": "🔄", "description": "迭代生成风格化图像"},
    {"step": 6, "total": 6, "name": "输出结果", "emoji": "🎨", "description": "返回最终风格迁移图片"},
]


def _sse_event(data: dict) -> str:
    """将字典序列化为 SSE 格式的 data 行"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _emit_step(step_info: dict, status: str) -> str:
    """生成一条步骤事件"""
    return _sse_event({**step_info, "status": status})


def run_style_transfer_stream(content_img, style_key, preset):
    """
    执行风格迁移主流程（生成器版本），每个步骤产出 SSE 事件。
    最终 yield 包含 base64 结果图片的事件。
    """
    # ---- Step 1: 接收图片 ----
    yield _emit_step(TRANSFER_STEPS[0], "running")
    style_img = generate_style_image(style_key, content_img.size())
    input_img = content_img.clone()
    yield _emit_step(TRANSFER_STEPS[0], "done")

    # ---- Step 2: 图像预处理（已在 load_image 完成，这里确认归一化参数） ----
    yield _emit_step(TRANSFER_STEPS[1], "running")
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
    yield _emit_step(TRANSFER_STEPS[1], "done")

    # ---- Step 3: VGG19 特征提取 ----
    yield _emit_step(TRANSFER_STEPS[2], "running")
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
    normalization_layer = Normalization(normalization_mean, normalization_std)
    yield _emit_step(TRANSFER_STEPS[2], "done")

    # ---- Step 4: Gram 矩阵计算（构建模型并插入损失层） ----
    yield _emit_step(TRANSFER_STEPS[3], "running")
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization_layer)
    content_layers = ["conv_4"]
    style_layers = list(preset["style_layers_weights"].keys())

    i = 0
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
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append((style_loss, preset["style_layers_weights"][name]))

    # 裁剪模型
    for idx in range(len(model) - 1, -1, -1):
        if isinstance(model[idx], (ContentLoss, StyleLoss)):
            break
    model = model[:idx + 1]
    yield _emit_step(TRANSFER_STEPS[3], "done")

    # ---- Step 5: L-BFGS 迭代优化 ----
    yield _emit_step(TRANSFER_STEPS[4], "running")
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])

    num_steps = preset["num_steps"]
    style_weight = preset["style_weight"]
    content_weight = preset["content_weight"]

    step_counter = [0]
    while step_counter[0] < num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            c_loss = sum(cl.loss for cl in content_losses) * content_weight
            s_loss = sum(sl.loss * w for sl, w in style_losses) * style_weight
            total_loss = c_loss + s_loss
            total_loss.backward()
            step_counter[0] += 1
            return total_loss
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    yield _emit_step(TRANSFER_STEPS[4], "done")

    # ---- Step 6: 输出结果 ----
    yield _emit_step(TRANSFER_STEPS[5], "running")
    output_image = tensor_to_image(input_img)
    output_id = str(uuid.uuid4())
    output_path = os.path.join(OUTPUT_DIR, f"{output_id}.png")
    output_image.save(output_path, "PNG")

    # 将结果图片编码为 base64
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    yield _emit_step(TRANSFER_STEPS[5], "done")

    # 发送最终结果
    yield _sse_event({
        "type": "result",
        "output_id": output_id,
        "download_url": f"/download/{output_id}.png",
        "image_base64": img_base64,
    })


@app.post("/transfer-stream")
async def transfer_stream(file_id: str = Form(...), style: str = Form(...)):
    """
    SSE 版本的风格迁移接口：实时推送每个处理步骤的状态
    参数：
    - file_id: 上传图片返回的文件 ID
    - style: 风格 key（oil_painting / sketch / watercolor / impressionist）
    返回 text/event-stream，逐步推送 6 个步骤的 running/done 状态，最后推送结果
    """
    # 验证风格参数
    if style not in STYLE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的风格：{style}，可选：{list(STYLE_PRESETS.keys())}"
        )

    # 查找上传的文件
    upload_file = None
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(file_id):
            upload_file = os.path.join(UPLOAD_DIR, fname)
            break

    if not upload_file or not os.path.exists(upload_file):
        raise HTTPException(status_code=404, detail="图片未找到，请重新上传")

    # 读取图片
    with open(upload_file, "rb") as f:
        image_bytes = f.read()

    preset = STYLE_PRESETS[style]
    content_img = load_image(image_bytes)

    def event_generator():
        """SSE 事件生成器"""
        yield from run_style_transfer_stream(content_img, style, preset)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
