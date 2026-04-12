"""
🏭 Demo Factory — Portfolio Management Panel
Backend API server (port 7000)
"""

import json
import os
import signal
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Demo Factory Portfolio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FACTORY_ROOT = Path(os.environ.get("FACTORY_ROOT", "/root/projects/demo-factory"))
DEMOS_DIR = FACTORY_ROOT / "demos"
REGISTRY_FILE = DEMOS_DIR / "_registry.json"

# In-memory running state: slug -> {pid, port, process}
running_demos: Dict[str, dict] = {}

# Port allocation
_next_port = 9001


def _alloc_port() -> int:
    global _next_port
    port = _next_port
    _next_port += 1
    return port


def _load_registry() -> list:
    if not REGISTRY_FILE.exists():
        return []
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_demo_status(slug: str) -> str:
    info = running_demos.get(slug)
    if info is None:
        return "stopped"
    # Check if process is still alive
    proc: subprocess.Popen = info.get("process")
    if proc and proc.poll() is None:
        return "running"
    # Process died — clean up
    running_demos.pop(slug, None)
    return "stopped"


def _enrich(demo: dict) -> dict:
    """Attach runtime status and port to a demo record."""
    slug = demo.get("slug", "")
    status = _get_demo_status(slug)
    demo["run_status"] = status
    demo["run_port"] = running_demos.get(slug, {}).get("port")
    return demo


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── GET /api/demos ────────────────────────────────────────────────────────────
@app.get("/api/demos")
def list_demos():
    demos = _load_registry()
    return [_enrich(d) for d in demos]


# ── GET /api/demos/{slug} ────────────────────────────────────────────────────
@app.get("/api/demos/{slug}")
def get_demo(slug: str):
    demos = _load_registry()
    demo = next((d for d in demos if d.get("slug") == slug), None)
    if demo is None:
        raise HTTPException(status_code=404, detail="Demo not found")

    # Try to read README.md
    folder = demo.get("folder", "")
    readme_path = DEMOS_DIR / folder / "README.md"
    readme = ""
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
    demo["readme"] = readme

    return _enrich(demo)


# ── POST /api/demos/{slug}/start ─────────────────────────────────────────────
@app.post("/api/demos/{slug}/start")
async def start_demo(slug: str):
    if _get_demo_status(slug) == "running":
        info = running_demos[slug]
        return {"status": "already_running", "port": info["port"]}

    demos = _load_registry()
    demo = next((d for d in demos if d.get("slug") == slug), None)
    if demo is None:
        raise HTTPException(status_code=404, detail="Demo not found")

    folder = demo.get("folder", "")
    backend_dir = DEMOS_DIR / folder / "backend"
    app_file = backend_dir / "app.py"
    req_file = backend_dir / "requirements.txt"

    if not app_file.exists():
        raise HTTPException(status_code=400, detail="No backend/app.py found")

    port = _alloc_port()
    venv_dir = f"/tmp/demo_venv_{slug}"

    # Create venv + install deps
    try:
        subprocess.run(
            ["python3", "-m", "venv", venv_dir],
            check=True, capture_output=True, timeout=30,
        )
        if req_file.exists():
            subprocess.run(
                [f"{venv_dir}/bin/pip", "install", "-q", "-r", str(req_file)],
                check=True, capture_output=True, timeout=120,
            )
        # Make sure uvicorn is available
        subprocess.run(
            [f"{venv_dir}/bin/pip", "install", "-q", "uvicorn", "fastapi"],
            check=True, capture_output=True, timeout=60,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Setup failed: {e.stderr.decode()[:500]}")

    # Launch uvicorn
    proc = subprocess.Popen(
        [
            f"{venv_dir}/bin/uvicorn",
            "app:app",
            "--host", "0.0.0.0",
            "--port", str(port),
        ],
        cwd=str(backend_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    # Give it a moment to bind
    await asyncio.sleep(1)

    if proc.poll() is not None:
        raise HTTPException(status_code=500, detail="Demo process exited immediately")

    running_demos[slug] = {"pid": proc.pid, "port": port, "process": proc, "venv": venv_dir}
    return {"status": "started", "port": port, "pid": proc.pid}


# ── POST /api/demos/{slug}/stop ──────────────────────────────────────────────
@app.post("/api/demos/{slug}/stop")
def stop_demo(slug: str):
    info = running_demos.pop(slug, None)
    if info is None:
        return {"status": "not_running"}

    proc: subprocess.Popen = info.get("process")
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    # Clean up venv
    venv = info.get("venv", "")
    if venv and os.path.isdir(venv):
        subprocess.run(["rm", "-rf", venv], capture_output=True)

    return {"status": "stopped"}


# ── GET /api/stats ────────────────────────────────────────────────────────────
@app.get("/api/stats")
def stats():
    demos = _load_registry()
    total = len(demos)

    scores = []
    trend = []
    for d in sorted(demos, key=lambda x: x.get("created_at", "")):
        s = d.get("score")
        name = d.get("name", d.get("slug", "?"))
        if s is not None:
            scores.append(s)
            trend.append({"name": name, "score": s})
        else:
            trend.append({"name": name, "score": None})

    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    running_count = sum(1 for slug in running_demos if _get_demo_status(slug) == "running")

    return {
        "total": total,
        "avg_score": avg_score,
        "running": running_count,
        "trend": trend,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
