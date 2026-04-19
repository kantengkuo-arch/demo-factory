"""
监控 API — agent 状态、cron 状态、token 消耗
通过调用 openclaw CLI 获取数据
"""
import json
import subprocess


def run_cmd(cmd, timeout=15):
    """执行命令并返回输出"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception as e:
        return f"error: {str(e)}"


def get_cron_status():
    """获取所有 cron 任务状态"""
    try:
        raw = run_cmd("openclaw cron list --json 2>/dev/null")
        if not raw or raw.startswith("error"):
            # fallback: 解析非 JSON 输出
            raw = run_cmd("openclaw cron list 2>/dev/null")
            lines = [l.strip() for l in raw.split('\n') if l.strip() and not l.startswith('ID') and not l.startswith('─') and not l.startswith('🦞')]
            jobs = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 6:
                    jobs.append({
                        "id": parts[0],
                        "name": parts[1] if len(parts) > 1 else "",
                        "schedule": " ".join(parts[2:5]) if len(parts) > 4 else "",
                        "status": next((p for p in parts if p in ["ok", "error", "disabled"]), "unknown"),
                        "agent": next((p for p in parts if p in ["scout", "coder", "frontend", "reviewer", "evaluator", "trend", "boss"]), ""),
                    })
            return {"jobs": jobs}
        return json.loads(raw)
    except Exception as e:
        return {"jobs": [], "error": str(e)}


def get_agent_sessions():
    """获取所有 agent 的 session 和 token 消耗"""
    try:
        raw = run_cmd("openclaw sessions --all-agents --json 2>/dev/null", timeout=20)
        if not raw or raw.startswith("error"):
            return {"agents": {}, "total_tokens": 0}
        
        data = json.loads(raw)
        sessions = data.get("sessions", [])
        
        # 按 agent 汇总 token
        agent_stats = {}
        for s in sessions:
            agent_id = s.get("agentId", "unknown")
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "agent_id": agent_id,
                    "session_count": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "model": s.get("model", ""),
                    "last_active_ms": 0,
                }
            
            stats = agent_stats[agent_id]
            # 只统计 cron 主 session，避免重复计数（跳过 :run: 的子session）
            if ":run:" not in s.get("key", ""):
                stats["session_count"] += 1
                stats["total_input_tokens"] += s.get("inputTokens", 0)
                stats["total_output_tokens"] += s.get("outputTokens", 0)
                stats["total_tokens"] += s.get("totalTokens", 0)
                age_ms = s.get("ageMs", 0)
                if stats["last_active_ms"] == 0 or age_ms < stats["last_active_ms"]:
                    stats["last_active_ms"] = age_ms
        
        # 计算总消耗
        total_tokens = sum(a["total_tokens"] for a in agent_stats.values())
        
        return {
            "agents": agent_stats,
            "total_tokens": total_tokens,
            "total_sessions": data.get("count", 0),
        }
    except Exception as e:
        return {"agents": {}, "total_tokens": 0, "error": str(e)}


def get_gateway_health():
    """获取 gateway 健康状态"""
    try:
        raw = run_cmd("openclaw health --json 2>/dev/null")
        if raw and not raw.startswith("error"):
            return json.loads(raw)
        return {"status": "unknown"}
    except Exception:
        return {"status": "unknown"}


def get_monitor_data():
    """汇总所有监控数据"""
    cron = get_cron_status()
    sessions = get_agent_sessions()
    
    return {
        "cron": cron,
        "token_usage": sessions,
        "agent_count": len(sessions.get("agents", {})),
    }
