"""
Demo Factory Portfolio — 后端 API
管理所有 demo 的展示、启动、停止
"""
import json
import subprocess
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(title="Demo Factory Portfolio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 配置 ===
FACTORY_ROOT = Path(os.environ.get("FACTORY_ROOT", "/root/projects/demo-factory"))
DEMOS_DIR = FACTORY_ROOT / "demos"
REGISTRY_FILE = DEMOS_DIR / "_registry.json"

# 运行中的 demo 容器记录（内存中，重启后清空）
# 格式：{ "demo-slug": { "port": 9001, "container_id": "abc123", "started_at": "..." } }
running_demos = {}

# 端口分配：从 9001 开始递增
NEXT_PORT = 9001


def load_registry():
    """读取 demo 注册表"""
    if not REGISTRY_FILE.exists():
        return []
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def find_available_port():
    """找一个可用端口"""
    global NEXT_PORT
    port = NEXT_PORT
    NEXT_PORT += 1
    return port


@app.get("/health")
def health():
    return {"status": "healthy", "service": "portfolio"}


@app.get("/api/demos")
def list_demos():
    """列出所有 demo 及其运行状态"""
    registry = load_registry()
    for demo in registry:
        slug = demo.get("slug", "")
        if slug in running_demos:
            demo["running"] = True
            demo["port"] = running_demos[slug]["port"]
            demo["url"] = f"http://localhost:{running_demos[slug]['port']}"
        else:
            demo["running"] = False
            demo["port"] = None
            demo["url"] = None
    return {"demos": registry, "total": len(registry), "running_count": len(running_demos)}


@app.get("/api/demos/{slug}")
def get_demo(slug: str):
    """获取单个 demo 详情"""
    registry = load_registry()
    demo = next((d for d in registry if d.get("slug") == slug), None)
    if not demo:
        raise HTTPException(status_code=404, detail=f"Demo '{slug}' not found")

    # 读取 README
    folder = demo.get("folder", "")
    readme_path = DEMOS_DIR / folder / "README.md"
    demo["readme"] = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    # 运行状态
    if slug in running_demos:
        demo["running"] = True
        demo["port"] = running_demos[slug]["port"]
        demo["url"] = f"http://localhost:{running_demos[slug]['port']}"
    else:
        demo["running"] = False

    return demo


@app.post("/api/demos/{slug}/start")
def start_demo(slug: str):
    """按需启动一个 demo（用 Docker 或直接 python 进程）"""
    if slug in running_demos:
        return {"message": "已经在运行", "port": running_demos[slug]["port"]}

    registry = load_registry()
    demo = next((d for d in registry if d.get("slug") == slug), None)
    if not demo:
        raise HTTPException(status_code=404, detail=f"Demo '{slug}' not found")

    folder = demo.get("folder", "")
    demo_dir = DEMOS_DIR / folder
    backend_dir = demo_dir / "backend"

    if not (backend_dir / "app.py").exists():
        raise HTTPException(status_code=400, detail="Demo 没有 backend/app.py")

    port = find_available_port()
    dockerfile = demo_dir / "Dockerfile"

    try:
        if dockerfile.exists():
            # 有 Dockerfile，用 Docker 启动
            container_name = f"demo-{slug}"
            # 构建镜像
            subprocess.run(
                ["docker", "build", "-t", container_name, "."],
                cwd=str(demo_dir), check=True, capture_output=True, timeout=120
            )
            # 启动容器
            result = subprocess.run(
                ["docker", "run", "-d", "--name", container_name,
                 "-p", f"{port}:8000", container_name],
                capture_output=True, text=True, check=True, timeout=30
            )
            container_id = result.stdout.strip()
        else:
            # 没有 Dockerfile，用 venv + python 启动
            venv_dir = f"/tmp/demo_venv_{slug}"
            setup_cmds = f"""
                python3 -m venv {venv_dir} && \
                {venv_dir}/bin/pip install -r {backend_dir}/requirements.txt -q && \
                cd {backend_dir} && \
                PORT={port} {venv_dir}/bin/python3 -c "
import uvicorn
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('app', 'app.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
uvicorn.run(mod.app, host='0.0.0.0', port={port})
" &
            """
            result = subprocess.Popen(
                ["bash", "-c", setup_cmds],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            container_id = str(result.pid)

        running_demos[slug] = {
            "port": port,
            "container_id": container_id,
            "started_at": datetime.now().isoformat(),
            "method": "docker" if dockerfile.exists() else "process",
        }
        return {"message": "启动成功", "port": port, "url": f"http://localhost:{port}"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"启动失败: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@app.post("/api/demos/{slug}/stop")
def stop_demo(slug: str):
    """停止一个 demo"""
    if slug not in running_demos:
        return {"message": "没有在运行"}

    info = running_demos[slug]
    try:
        if info["method"] == "docker":
            container_name = f"demo-{slug}"
            subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=15)
            subprocess.run(["docker", "rm", container_name], capture_output=True, timeout=15)
        else:
            # 杀 python 进程树
            pid = info["container_id"]
            subprocess.run(["kill", "-9", pid], capture_output=True)
            # 清理 venv
            venv_dir = f"/tmp/demo_venv_{slug}"
            subprocess.run(["rm", "-rf", venv_dir], capture_output=True)
    except Exception:
        pass

    del running_demos[slug]
    return {"message": "已停止"}


@app.get("/api/stats")
def get_stats():
    """工厂统计数据"""
    registry = load_registry()
    total = len(registry)
    scored = [d for d in registry if d.get("score") is not None]
    avg_score = round(sum(d["score"] for d in scored) / len(scored), 1) if scored else 0

    # 按时间排序的评分趋势（用于自进化曲线）
    score_trend = []
    for d in sorted(scored, key=lambda x: x.get("created_at", "")):
        score_trend.append({
            "name": d["name"],
            "score": d["score"],
            "date": d.get("created_at", "")[:10],
        })

    # 技术栈统计
    tech_count = {}
    for d in registry:
        for t in d.get("tech_stack", []):
            tech_count[t] = tech_count.get(t, 0) + 1

    return {
        "total_demos": total,
        "avg_score": avg_score,
        "running_count": len(running_demos),
        "score_trend": score_trend,
        "top_tech": sorted(tech_count.items(), key=lambda x: -x[1])[:10],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)


# === 集成时间线和竞技场 API ===

@app.get("/api/timeline")
def get_timeline():
    """Agent 协作时间线（甘特图数据）"""
    from timeline import get_all_timeline_data
    return get_all_timeline_data()


@app.get("/api/arena")
def get_arena():
    """竞技场对比数据"""
    from arena import get_arena_data
    return get_arena_data()
