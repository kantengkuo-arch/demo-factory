"""
根据 demos/_registry.json 自动更新根目录 README.md 中的 Demo 列表
"""
import json
from pathlib import Path
from datetime import datetime, timedelta


def main():
    root = Path(__file__).parent.parent
    registry_file = root / "demos" / "_registry.json"
    readme_file = root / "README.md"

    with open(registry_file, "r", encoding="utf-8") as f:
        demos = json.load(f)

    total = len(demos)
    one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    this_week = [d for d in demos if d.get("created_at", "") >= one_week_ago]
    latest = demos[-1]["name"] if demos else "-"

    if demos:
        table_lines = ["| # | 名称 | 方向 | 技术栈 | 日期 |",
                       "|---|------|------|--------|------|"]
        for i, d in enumerate(demos, 1):
            tech = ", ".join(d.get("tech_stack", [])[:3])
            date = d.get("created_at", "")[:10]
            folder = d.get("folder", "")
            table_lines.append(
                f"| {i} | [{d['name']}](demos/{folder}/) | {d.get('direction','')} | {tech} | {date} |"
            )
        demo_table = "\n".join(table_lines)
    else:
        demo_table = "_暂无 Demo，即将由 AI 自动生成..._"

    readme_content = f"""# 🏭 Demo Factory

> 一个由 AI Agent 驱动的 Demo 自动生产线。自动发现热门 AI 方向，自动调研、编码、审查、部署。

## 📊 Demo 统计

| 总数 | 本周新增 | 最新 Demo |
|------|----------|-----------|
| {total}    | {len(this_week)}        | {latest}         |

## 📁 Demo 列表

{demo_table}

## 🏗️ 项目结构

- `platform/` - 监控平台（前后端）
- `demos/` - AI 自动生成的 Demo 项目
- `openclaw/` - AI Agent 配置

## 🚀 快速开始

详见 [docs/setup.md](docs/setup.md)
"""

    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"✅ README 已更新: {total} 个 Demo")


if __name__ == "__main__":
    main()
