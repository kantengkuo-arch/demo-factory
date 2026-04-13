"""
竞技场 API — 同方向 demo 自动对比
"""
import json
from pathlib import Path
import os

FACTORY_ROOT = Path(os.environ.get("FACTORY_ROOT", "/root/projects/demo-factory"))
REGISTRY_FILE = FACTORY_ROOT / "demos" / "_registry.json"


def get_arena_data() -> dict:
    """找出同方向的 demo，两两对比"""
    if not REGISTRY_FILE.exists():
        return {"matchups": []}

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # 按方向分组
    by_direction = {}
    for d in registry:
        direction = d.get("direction", "").strip()
        if not direction or d.get("score") is None:
            continue
        by_direction.setdefault(direction, []).append(d)

    matchups = []
    for direction, demos in by_direction.items():
        if len(demos) < 2:
            continue
        # 按评分排序
        demos.sort(key=lambda x: x.get("score", 0), reverse=True)
        # 两两对比（取最好的两个）
        a, b = demos[0], demos[1]
        diff = a["score"] - b["score"]
        if diff > 15:
            verdict = f"🏆 {a['name']} 大幅领先（+{diff}分）"
        elif diff > 5:
            verdict = f"👑 {a['name']} 略胜一筹（+{diff}分）"
        else:
            verdict = "⚖️ 势均力敌"
        matchups.append({
            "direction": direction,
            "demos": [
                {"name": a["name"], "score": a["score"], "slug": a.get("slug", "")},
                {"name": b["name"], "score": b["score"], "slug": b.get("slug", "")},
            ],
            "verdict": verdict,
        })

    return {"matchups": matchups}
