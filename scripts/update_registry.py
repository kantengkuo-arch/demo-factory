"""
扫描 demos/ 目录，读取每个 demo 的 demo_meta.json，
汇总生成 demos/_registry.json
"""
import json
from pathlib import Path


def main():
    demos_dir = Path(__file__).parent.parent / "demos"
    registry = []

    for item in sorted(demos_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("_"):
            continue

        meta_file = item / "demo_meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                meta["folder"] = item.name
                registry.append(meta)

    output = demos_dir / "_registry.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print(f"✅ 注册表已更新: {len(registry)} 个 Demo")


if __name__ == "__main__":
    main()
