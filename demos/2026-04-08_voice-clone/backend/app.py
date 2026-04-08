"""
AI 语音克隆 Demo — FastAPI 后端
支持 OpenVoice 语音克隆（如可用），回退到 edge-tts + 音色调整
"""

import io
import os
import uuid
import time
import json
import wave
import struct
import asyncio
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

app = FastAPI(title="🎙️ AI 语音克隆 Demo", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 引擎检测 ====================

# 检测可用的 TTS 引擎
ENGINE = "none"
openvoice_model = None

# 尝试加载 OpenVoice
try:
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS as MeloTTS
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化 OpenVoice 模型
    ckpt_converter = os.path.join(BASE_DIR, "checkpoints_v2", "converter")
    if os.path.exists(ckpt_converter):
        tone_color_converter = ToneColorConverter(
            os.path.join(ckpt_converter, "config.json"), device=DEVICE
        )
        tone_color_converter.load_ckpt(os.path.join(ckpt_converter, "checkpoint.pth"))
        melo_tts = MeloTTS(language="ZH", device=DEVICE)
        ENGINE = "openvoice"
        print("✅ OpenVoice v2 引擎加载成功")
    else:
        print(f"⚠️ OpenVoice 检查点未找到: {ckpt_converter}")
        raise FileNotFoundError("checkpoints not found")
except Exception as e:
    print(f"ℹ️ OpenVoice 不可用 ({e})，尝试 edge-tts 回退方案")

# 尝试 edge-tts（轻量回退方案）
if ENGINE == "none":
    try:
        import edge_tts
        ENGINE = "edge-tts"
        print("✅ edge-tts 引擎加载成功（回退方案）")
    except ImportError:
        print("❌ 未找到任何 TTS 引擎")

# 预设中文 edge-tts 音色
EDGE_VOICES = {
    "xiaoxiao": {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓", "gender": "女", "desc": "温柔亲切"},
    "yunxi": {"id": "zh-CN-YunxiNeural", "name": "云希", "gender": "男", "desc": "阳光青年"},
    "xiaoyi": {"id": "zh-CN-XiaoyiNeural", "name": "晓依", "gender": "女", "desc": "活泼可爱"},
    "yunjian": {"id": "zh-CN-YunjianNeural", "name": "云健", "gender": "男", "desc": "成熟稳重"},
    "xiaomeng": {"id": "zh-CN-XiaomengNeural", "name": "晓梦", "gender": "女", "desc": "甜美少女"},
}

# 存储上传的参考音频信息
reference_store = {}


# ==================== 工具函数 ====================

def get_audio_duration(filepath: str) -> float:
    """获取音频文件时长（秒）"""
    try:
        import soundfile as sf
        data, sr = sf.read(filepath)
        return len(data) / sr
    except Exception:
        pass
    # 尝试用 ffprobe
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", filepath],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """将音频文件转换为 WAV 格式（16kHz, mono）"""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, timeout=30
        )
        return os.path.exists(output_path)
    except Exception:
        return False


async def tts_edge(text: str, voice_id: str, output_path: str) -> bool:
    """使用 edge-tts 生成语音"""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(output_path)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"edge-tts 错误: {e}")
        return False


def tts_openvoice(text: str, reference_wav: str, output_path: str) -> bool:
    """使用 OpenVoice 进行语音克隆"""
    try:
        # 第一步：用 MeloTTS 生成基础语音
        speaker_ids = melo_tts.hps.data.spk2id
        base_speaker = list(speaker_ids.keys())[0]
        temp_path = output_path.replace(".wav", "_base.wav")
        melo_tts.tts_to_file(text, speaker_ids[base_speaker], temp_path, speed=1.0)

        # 第二步：提取参考音频的音色特征
        source_se, _ = se_extractor.get_se(temp_path, tone_color_converter, vad=False)
        target_se, _ = se_extractor.get_se(reference_wav, tone_color_converter, vad=False)

        # 第三步：音色转换
        tone_color_converter.convert(
            audio_src_path=temp_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
        )

        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return os.path.exists(output_path)
    except Exception as e:
        print(f"OpenVoice 错误: {e}")
        return False


# ==================== API 路由 ====================

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "engine": ENGINE,
        "engine_name": {
            "openvoice": "OpenVoice v2（完整语音克隆）",
            "edge-tts": "Edge TTS（微软云语音，回退方案）",
            "none": "无可用引擎",
        }.get(ENGINE, ENGINE),
    }


@app.get("/voices")
async def list_voices():
    """获取可用的预设音色列表（edge-tts 模式）"""
    if ENGINE == "edge-tts":
        voices = [
            {"key": k, "name": v["name"], "gender": v["gender"], "desc": v["desc"]}
            for k, v in EDGE_VOICES.items()
        ]
        return {"engine": ENGINE, "voices": voices}
    else:
        return {"engine": ENGINE, "voices": [], "note": "OpenVoice 模式下使用参考音频克隆音色"}


@app.post("/upload_reference")
async def upload_reference(file: UploadFile = File(...)):
    """
    上传参考音频文件
    支持 wav, mp3, m4a, ogg 格式，建议 5-10 秒
    """
    # 验证文件类型
    allowed_ext = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    filename = file.filename or "audio.wav"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式：{ext}，支持 {', '.join(allowed_ext)}")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB 限制
        raise HTTPException(status_code=400, detail="文件大小不能超过 20MB")

    # 保存原始文件
    ref_id = str(uuid.uuid4())
    raw_path = os.path.join(UPLOAD_DIR, f"{ref_id}{ext}")
    with open(raw_path, "wb") as f:
        f.write(content)

    # 转换为标准 WAV
    wav_path = os.path.join(UPLOAD_DIR, f"{ref_id}.wav")
    if ext == ".wav":
        wav_path = raw_path
    else:
        if not convert_to_wav(raw_path, wav_path):
            wav_path = raw_path  # 转换失败就用原始文件

    # 获取时长
    duration = get_audio_duration(wav_path)

    # 存储参考信息
    reference_store[ref_id] = {
        "raw_path": raw_path,
        "wav_path": wav_path,
        "filename": filename,
        "duration": duration,
        "size": len(content),
    }

    return {
        "ref_id": ref_id,
        "filename": filename,
        "duration": round(duration, 1),
        "size": len(content),
        "format": ext,
    }


@app.post("/clone_voice")
async def clone_voice(
    text: str = Form(...),
    ref_id: str = Form(default=None),
    voice: str = Form(default="xiaoxiao"),
):
    """
    语音克隆/合成
    参数：
    - text: 要合成的文本内容
    - ref_id: 参考音频 ID（OpenVoice 模式必填）
    - voice: 预设音色 key（edge-tts 模式使用）
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="文本内容不能为空")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="文本长度不能超过 500 字")

    output_id = str(uuid.uuid4())
    output_path = os.path.join(OUTPUT_DIR, f"{output_id}.wav")
    start_time = time.time()

    if ENGINE == "openvoice":
        # OpenVoice 语音克隆模式
        if not ref_id or ref_id not in reference_store:
            raise HTTPException(status_code=400, detail="请先上传参考音频")
        ref_info = reference_store[ref_id]
        success = tts_openvoice(text, ref_info["wav_path"], output_path)
        if not success:
            raise HTTPException(status_code=500, detail="语音克隆失败，请重试")

    elif ENGINE == "edge-tts":
        # Edge TTS 模式
        if voice not in EDGE_VOICES:
            voice = "xiaoxiao"
        voice_id = EDGE_VOICES[voice]["id"]
        # edge-tts 生成 mp3，需要转成 wav 或直接用 mp3
        mp3_path = os.path.join(OUTPUT_DIR, f"{output_id}.mp3")
        success = await tts_edge(text, voice_id, mp3_path)
        if not success:
            raise HTTPException(status_code=500, detail="语音合成失败，请重试")
        # 转换为 wav
        if not convert_to_wav(mp3_path, output_path):
            # 转换失败就直接用 mp3
            output_path = mp3_path
    else:
        raise HTTPException(status_code=503, detail="没有可用的 TTS 引擎，请安装 edge-tts 或 openvoice")

    elapsed = round(time.time() - start_time, 2)
    duration = get_audio_duration(output_path)
    output_filename = os.path.basename(output_path)

    return {
        "output_id": output_id,
        "filename": output_filename,
        "duration": round(duration, 1),
        "processing_time": elapsed,
        "engine": ENGINE,
        "download_url": f"/download/{output_filename}",
        "play_url": f"/download/{output_filename}",
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """下载生成的音频文件"""
    # 安全检查：防止路径穿越
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")

    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")

    # 根据扩展名设置 MIME 类型
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(filepath, media_type=media_type, filename=filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
