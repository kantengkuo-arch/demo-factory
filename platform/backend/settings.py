import json
from pathlib import Path

CONFIG_PATH = Path.home() / "projects/demo-factory/platform/config/model_api.json"

MODEL_TYPES = {
    "chat": "对话模型",
    "embedding": "Embedding 模型",
    "image": "图像模型",
    "speech": "语音模型",
}

def mask_key(key: str) -> str:
    if not key or len(key) <= 8:
        return "未设置"
    return key[:4] + "****" + key[-4:]

def load_config():
    if not CONFIG_PATH.exists():
        return {k: {"api_url": "", "api_key_masked": "未设置", "model": "", "provider": ""} for k in MODEL_TYPES}
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    result = {}
    for k in MODEL_TYPES:
        c = config.get(k, {})
        result[k] = {
            "api_url": c.get("api_url", ""),
            "api_key_masked": mask_key(c.get("api_key", "")),
            "model": c.get("model", ""),
            "provider": c.get("provider", ""),
            "label": MODEL_TYPES[k],
        }
    return result

def save_config(model_type: str, data: dict):
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {}
    if model_type not in config:
        config[model_type] = {}
    if data.get("api_url"):
        config[model_type]["api_url"] = data["api_url"]
    if data.get("api_key"):
        config[model_type]["api_key"] = data["api_key"]
    if data.get("model"):
        config[model_type]["model"] = data["model"]
    if data.get("provider"):
        config[model_type]["provider"] = data["provider"]
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return load_config()


def test_api(model_type: str) -> dict:
    """测试指定类型的模型 API 是否可用"""
    import httpx

    if not CONFIG_PATH.exists():
        return {"success": False, "error": "配置文件不存在"}

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    cfg = config.get(model_type, {})
    api_url = cfg.get("api_url", "")
    api_key = cfg.get("api_key", "")
    model = cfg.get("model", "")

    if not api_url or not api_key:
        return {"success": False, "error": "API URL 或 Key 未设置"}

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with httpx.Client(timeout=15) as client:
            if model_type == "chat":
                resp = client.post(
                    f"{api_url}/chat/completions",
                    headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
                )
            elif model_type == "embedding":
                resp = client.post(
                    f"{api_url}/embeddings",
                    headers=headers,
                    json={"model": model, "input": "test"}
                )
            elif model_type == "image":
                # 图像模型只验证认证，不真正生成
                resp = client.post(
                    f"{api_url}/images/generations",
                    headers=headers,
                    json={"model": model, "prompt": "a white dot", "size": "256x256", "n": 1}
                )
            elif model_type == "speech":
                resp = client.post(
                    f"{api_url}/audio/speech",
                    headers=headers,
                    json={"model": model, "input": "test", "voice": "alloy"},
                )
            else:
                return {"success": False, "error": f"未知模型类型: {model_type}"}

            if resp.status_code == 200:
                return {"success": True, "message": f"✅ 连接成功，模型 {model} 可用"}
            elif resp.status_code == 401:
                return {"success": False, "error": "API Key 无效（401）"}
            elif resp.status_code == 404:
                return {"success": False, "error": f"模型 {model} 不存在（404）"}
            elif resp.status_code == 429:
                return {"success": True, "message": "连接成功，但触发了速率限制（429），说明 Key 有效"}
            else:
                body = resp.text[:200]
                return {"success": False, "error": f"HTTP {resp.status_code}: {body}"}

    except httpx.ConnectError:
        return {"success": False, "error": f"无法连接 {api_url}，请检查 URL"}
    except httpx.TimeoutException:
        return {"success": False, "error": "请求超时（15秒）"}
    except Exception as e:
        return {"success": False, "error": str(e)}
