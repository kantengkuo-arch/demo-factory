"""
时间线 API — 从 GitHub Issue 事件中提取 agent 协作甘特图数据
原理：每次 agent 改 label 时，GitHub 会记录一个 timeline event，
      我们用 gh CLI 读出来，计算每个阶段耗时。
"""
import json
import subprocess
from datetime import datetime


LABEL_ORDER = [
    "needs-scout", "needs-coding",
    "needs-review", "needs-merge"
]


def get_issue_timeline(issue_number: int) -> list:
    """用 gh CLI 获取 issue 的 timeline 事件"""
    try:
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_number),
             "--json", "title,labels,createdAt,closedAt,comments",
             "--repo", "kantengkuo-arch/demo-factory"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return []
        return json.loads(result.stdout)
    except Exception:
        return []


def get_all_timeline_data() -> dict:
    """获取所有有 done label 的 issue 的时间线"""
    try:
        # 获取所有 done 的 issue
        result = subprocess.run(
            ["gh", "issue", "list",
             "--label", "done",
             "--state", "all",
             "--json", "number,title,createdAt,closedAt,labels",
             "--limit", "50",
             "--repo", "kantengkuo-arch/demo-factory"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return {"issues": []}

        issues = json.loads(result.stdout)
        timeline_data = []

        for issue in issues:
            # 用 timeline events API 获取 label 变更历史
            events_result = subprocess.run(
                ["gh", "api",
                 f"/repos/kantengkuo-arch/demo-factory/issues/{issue['number']}/timeline",
                 "--paginate"],
                capture_output=True, text=True, timeout=15
            )
            if events_result.returncode != 0:
                continue

            events = json.loads(events_result.stdout) if events_result.stdout.strip() else []

            # 提取 label 事件，计算每个阶段耗时
            label_events = []
            for e in events:
                if e.get("event") == "labeled":
                    label_name = e.get("label", {}).get("name", "")
                    if label_name in LABEL_ORDER:
                        label_events.append({
                            "label": label_name,
                            "time": e.get("created_at", ""),
                        })

            # 计算阶段耗时
            phases = []
            for i, le in enumerate(label_events):
                start = datetime.fromisoformat(le["time"].replace("Z", "+00:00"))
                if i + 1 < len(label_events):
                    end = datetime.fromisoformat(label_events[i+1]["time"].replace("Z", "+00:00"))
                else:
                    end = datetime.now(start.tzinfo)
                duration_min = round((end - start).total_seconds() / 60, 1)
                phases.append({
                    "label": le["label"],
                    "start": le["time"],
                    "duration_min": duration_min,
                })

            if phases:
                timeline_data.append({
                    "number": issue["number"],
                    "title": issue["title"],
                    "phases": phases,
                })

        return {"issues": timeline_data}

    except Exception as e:
        return {"issues": [], "error": str(e)}
