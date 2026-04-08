"""
OpenClaw 多 Agent 协作监控面板 - 后端服务
实时监控 OpenClaw Agent 状态、系统资源、任务流转和日志
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="OpenClaw Monitor", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 常量配置 ============
OPENCLAW_HOME = Path.home() / ".openclaw"
DEMO_FACTORY_DIR = Path("/root/projects/demo-factory")
AGENT_NAMES = ["boss", "coder", "reviewer", "scout"]
# Agent 流转关系：Boss → Scout → Coder → Reviewer
AGENT_FLOW = {
    "boss": ["scout"],
    "scout": ["coder"],
    "coder": ["reviewer"],
    "reviewer": ["boss"],
}
# 标签到 Agent 阶段的映射
LABEL_TO_AGENT = {
    "scout": "scout",
    "coder": "coder",
    "reviewer": "reviewer",
    "done": "done",
}
# 心跳超时阈值（秒）
HEARTBEAT_TIMEOUT = 600  # 10 分钟（2 个 5 分钟周期）
# 资源告警阈值
MEMORY_ALERT_THRESHOLD = 80
DISK_ALERT_THRESHOLD = 85
# 系统监控历史数据点数
MAX_HISTORY_POINTS = 60

# ============ 全局状态 ============
# WebSocket 连接管理
connected_clients: List[WebSocket] = []
# 系统监控历史数据
system_history: Dict[str, List] = {
    "cpu": [],
    "memory": [],
    "disk": [],
    "timestamps": [],
}
# 告警列表
alerts: List[Dict] = []
# 日志缓冲
log_buffer: List[str] = []
MAX_LOG_LINES = 200


def run_command(cmd: str, timeout: int = 10) -> str:
    """执行命令并返回输出"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"Error: {e}"


def get_gateway_info() -> Dict[str, Any]:
    """获取 Gateway 运行状态"""
    info = {
        "running": False,
        "pid": None,
        "uptime": None,
        "start_time": None,
        "sessions": 0,
    }
    try:
        # 从 systemd 获取 Gateway 状态
        pid_output = run_command(
            "systemctl --user show openclaw-gateway --property=MainPID --value"
        )
        if pid_output and pid_output.isdigit() and int(pid_output) > 0:
            pid = int(pid_output)
            info["pid"] = pid
            info["running"] = True
            try:
                proc = psutil.Process(pid)
                create_time = proc.create_time()
                info["start_time"] = datetime.fromtimestamp(
                    create_time, tz=timezone(timedelta(hours=8))
                ).strftime("%Y-%m-%d %H:%M:%S")
                uptime_seconds = time.time() - create_time
                hours = int(uptime_seconds // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                info["uptime"] = f"{hours}h {minutes}m"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass

    # 获取会话数
    try:
        status_output = run_command("openclaw status 2>/dev/null | grep Sessions", timeout=15)
        if "active" in status_output:
            parts = status_output.split("│")
            for part in parts:
                if "active" in part:
                    num = "".join(filter(str.isdigit, part.split("active")[0].strip().split()[-1]))
                    if num:
                        info["sessions"] = int(num)
                    break
    except Exception:
        pass

    return info


def get_agent_status() -> List[Dict[str, Any]]:
    """获取所有 Agent 的状态"""
    agents = []
    tz_cst = timezone(timedelta(hours=8))
    now = datetime.now(tz_cst)

    for name in AGENT_NAMES:
        agent_info = {
            "name": name,
            "role": _get_agent_role(name),
            "status": "idle",  # idle / working / error
            "last_heartbeat": None,
            "heartbeat_ago": None,
        }

        # 检查 Agent workspace
        workspace_dir = OPENCLAW_HOME / f"workspace-{name}"
        if not workspace_dir.exists():
            agent_info["status"] = "error"
            agents.append(agent_info)
            continue

        # 检查心跳：通过 session store 的时间戳推断
        try:
            session_dir = OPENCLAW_HOME / "sessions"
            if session_dir.exists():
                agent_sessions = []
                for f in session_dir.iterdir():
                    if f.name.startswith(f"agent-{name}") and f.suffix == ".json":
                        agent_sessions.append(f)
                if agent_sessions:
                    latest = max(agent_sessions, key=lambda x: x.stat().st_mtime)
                    mtime = latest.stat().st_mtime
                    last_active = datetime.fromtimestamp(mtime, tz=tz_cst)
                    agent_info["last_heartbeat"] = last_active.strftime("%H:%M:%S")
                    diff = (now - last_active).total_seconds()
                    if diff < 60:
                        agent_info["heartbeat_ago"] = "刚刚"
                    elif diff < 3600:
                        agent_info["heartbeat_ago"] = f"{int(diff // 60)}分钟前"
                    else:
                        agent_info["heartbeat_ago"] = f"{int(diff // 3600)}小时前"

                    if diff > HEARTBEAT_TIMEOUT:
                        agent_info["status"] = "error"
                    elif diff < 120:
                        agent_info["status"] = "working"
                    else:
                        agent_info["status"] = "idle"
        except Exception:
            pass

        agents.append(agent_info)

    return agents


def _get_agent_role(name: str) -> str:
    """获取 Agent 职责描述"""
    roles = {
        "boss": "任务调度 & 用户交互",
        "scout": "技术调研 & 方案设计",
        "coder": "代码编写 & 实现",
        "reviewer": "代码审查 & 质量控制",
    }
    return roles.get(name, "未知")


def get_issue_tasks() -> List[Dict[str, Any]]:
    """获取 GitHub Issues 任务列表"""
    tasks = []
    try:
        output = run_command(
            f"cd {DEMO_FACTORY_DIR} && gh issue list --state open --json number,title,labels,createdAt --limit 10",
            timeout=15,
        )
        if output:
            issues = json.loads(output)
            for issue in issues:
                labels = [l["name"] for l in issue.get("labels", [])]
                stage = "unknown"
                for label in labels:
                    if label in LABEL_TO_AGENT:
                        stage = LABEL_TO_AGENT[label]
                        break

                tasks.append({
                    "number": issue["number"],
                    "title": issue["title"],
                    "labels": labels,
                    "stage": stage,
                    "created_at": issue.get("createdAt", ""),
                })
    except Exception:
        pass

    return tasks


def get_system_metrics() -> Dict[str, Any]:
    """获取系统资源使用情况"""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    metrics = {
        "cpu": {
            "percent": cpu_percent,
            "cores": psutil.cpu_count(),
        },
        "memory": {
            "percent": memory.percent,
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
        },
        "disk": {
            "percent": disk.percent,
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
        },
    }
    return metrics


def check_alerts(metrics: Dict, gateway_info: Dict, agents: List[Dict]) -> List[Dict]:
    """检查各种告警条件"""
    current_alerts = []
    now_str = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")

    # Gateway 崩溃检测
    if not gateway_info.get("running"):
        current_alerts.append({
            "level": "critical",
            "message": "🚨 Gateway 未运行！",
            "time": now_str,
        })

    # 内存超过 80%
    mem_pct = metrics["memory"]["percent"]
    if mem_pct > MEMORY_ALERT_THRESHOLD:
        current_alerts.append({
            "level": "warning",
            "message": f"⚠️ 内存使用 {mem_pct}%，超过 {MEMORY_ALERT_THRESHOLD}% 阈值",
            "time": now_str,
        })

    # 磁盘超过 85%
    disk_pct = metrics["disk"]["percent"]
    if disk_pct > DISK_ALERT_THRESHOLD:
        current_alerts.append({
            "level": "warning",
            "message": f"⚠️ 磁盘使用 {disk_pct}%，超过 {DISK_ALERT_THRESHOLD}% 阈值",
            "time": now_str,
        })

    # Agent 心跳超时
    for agent in agents:
        if agent["status"] == "error":
            current_alerts.append({
                "level": "warning",
                "message": f"⚠️ Agent [{agent['name']}] 心跳超时",
                "time": now_str,
            })

    return current_alerts


def get_recent_logs(lines: int = 50, agent_filter: str = None) -> List[str]:
    """获取 Gateway 最近的日志"""
    try:
        cmd = f"journalctl --user -u openclaw-gateway --no-pager -n {lines} --output=short-iso 2>/dev/null"
        output = run_command(cmd, timeout=10)
        if not output:
            cmd = f"journalctl -u openclaw-gateway --no-pager -n {lines} --output=short-iso 2>/dev/null"
            output = run_command(cmd, timeout=10)
        if output:
            log_lines = output.split("\n")
            if agent_filter:
                log_lines = [
                    line for line in log_lines
                    if agent_filter.lower() in line.lower()
                ]
            return log_lines[-lines:]
    except Exception:
        pass
    return []


def collect_full_snapshot() -> Dict[str, Any]:
    """收集一次完整的监控数据快照"""
    metrics = get_system_metrics()
    gateway = get_gateway_info()
    agents = get_agent_status()
    tasks = get_issue_tasks()
    current_alerts = check_alerts(metrics, gateway, agents)

    # 更新历史数据
    now_str = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
    system_history["cpu"].append(metrics["cpu"]["percent"])
    system_history["memory"].append(metrics["memory"]["percent"])
    system_history["disk"].append(metrics["disk"]["percent"])
    system_history["timestamps"].append(now_str)

    for key in system_history:
        if len(system_history[key]) > MAX_HISTORY_POINTS:
            system_history[key] = system_history[key][-MAX_HISTORY_POINTS:]

    return {
        "type": "snapshot",
        "timestamp": now_str,
        "gateway": gateway,
        "agents": agents,
        "tasks": tasks,
        "metrics": metrics,
        "alerts": current_alerts,
        "history": {
            "cpu": system_history["cpu"],
            "memory": system_history["memory"],
            "disk": system_history["disk"],
            "timestamps": system_history["timestamps"],
        },
    }


# ============ 路由 ============

@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "ok", "service": "openclaw-monitor", "timestamp": time.time()}


@app.get("/api/status")
async def get_status():
    """获取当前整体状态（REST API）"""
    return collect_full_snapshot()


@app.get("/api/agents")
async def get_agents():
    """获取 Agent 状态"""
    return {"agents": get_agent_status()}


@app.get("/api/system")
async def get_system():
    """获取系统资源"""
    return get_system_metrics()


@app.get("/api/logs")
async def get_logs(lines: int = 50, agent: str = None):
    """获取 Gateway 日志"""
    return {"logs": get_recent_logs(lines=lines, agent_filter=agent)}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """返回前端页面"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点 - 实时推送监控数据"""
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        # 首次连接立即发送完整数据
        snapshot = collect_full_snapshot()
        await websocket.send_json(snapshot)

        # 持续推送数据（每 5 秒）
        while True:
            await asyncio.sleep(5)
            try:
                snapshot = collect_full_snapshot()
                await websocket.send_json(snapshot)
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
