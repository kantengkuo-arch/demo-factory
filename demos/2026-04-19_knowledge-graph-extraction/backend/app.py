"""
AI 知识图谱抽取系统 — 后端服务

通过 LLM API 从文本中抽取实体（NER）和关系（RE），
使用 networkx 构建和分析图结构，支持多种格式导出。

所有模型调用均通过外部 API，不依赖任何本地 NLP 模型。
"""

import json
import os
import re
import time
import uuid
import csv
import io
from pathlib import Path
from typing import Optional

import httpx
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ==================== 配置 ====================

# 模型 API 配置文件路径
MODEL_CONFIG_PATH = Path.home() / "projects" / "demo-factory" / "platform" / "config" / "model_api.json"

# 导出文件存储目录
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# 支持的实体类型定义
ENTITY_TYPES = [
    {
        "type": "PERSON",
        "description": "人物、姓名",
        "examples": ["张三", "乔布斯", "马云"],
    },
    {
        "type": "ORGANIZATION",
        "description": "组织、公司、机构",
        "examples": ["苹果公司", "清华大学", "联合国"],
    },
    {
        "type": "LOCATION",
        "description": "地点、地名、国家",
        "examples": ["北京", "硅谷", "长江"],
    },
    {
        "type": "EVENT",
        "description": "事件、活动",
        "examples": ["世界大战", "双十一", "奥运会"],
    },
    {
        "type": "CONCEPT",
        "description": "概念、技术、理论",
        "examples": ["人工智能", "量子计算", "相对论"],
    },
    {
        "type": "PRODUCT",
        "description": "产品、作品",
        "examples": ["iPhone", "微信", "哈利波特"],
    },
    {
        "type": "DATE",
        "description": "日期、时间",
        "examples": ["2024年", "明朝", "21世纪"],
    },
]

DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT"]


def load_model_config() -> dict:
    """
    从配置文件加载模型 API 信息。

    Returns:
        包含 chat 配置的字典
    """
    if not MODEL_CONFIG_PATH.exists():
        raise RuntimeError(f"模型配置文件不存在: {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================== LLM API 调用 ====================


async def call_llm_api(messages: list[dict], temperature: float = 0.3, max_tokens: int = 4000) -> str:
    """
    调用外部 LLM Chat API。

    Args:
        messages: 对话消息列表
        temperature: 生成温度（知识抽取用低温度保证确定性）
        max_tokens: 最大生成 token 数

    Returns:
        模型生成的文本

    Raises:
        HTTPException: API 调用失败时
    """
    config = load_model_config()
    chat_config = config["chat"]

    api_url = chat_config["api_url"].rstrip("/") + "/chat/completions"
    api_key = chat_config["api_key"]
    model = chat_config["model"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"LLM API 调用失败: {e.response.status_code} - {e.response.text}",
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM API 调用异常: {str(e)}")


# ==================== 知识抽取核心逻辑 ====================


def build_ner_prompt(text: str, entity_types: list[str], max_entities: int) -> list[dict]:
    """
    构造实体抽取（NER）的 prompt。

    使用 few-shot 示例和严格的 JSON 格式约束。

    Args:
        text: 待分析文本
        entity_types: 目标实体类型列表
        max_entities: 最大实体数量

    Returns:
        LLM 消息列表
    """
    types_desc = ", ".join(entity_types)

    system_prompt = f"""你是一个专业的命名实体识别（NER）专家。你的任务是从给定文本中抽取实体。

## 规则
1. 只抽取以下类型的实体: {types_desc}
2. 最多抽取 {max_entities} 个最重要的实体
3. 统计每个实体在文本中出现的次数
4. 为每个实体给出一个置信度分数（0-1），表示你对该识别结果的确信程度
5. 为每个实体生成简短描述（基于文本上下文）
6. 输出必须是严格的 JSON 格式，不要包含任何其他文字

## 输出格式
```json
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "实体类型",
      "mentions": 出现次数,
      "confidence": 置信度,
      "description": "基于文本的实体简短描述"
    }}
  ]
}}
```

## 示例

输入文本: "马云于1999年在杭州创立了阿里巴巴集团，后来阿里巴巴成为全球最大的电子商务公司之一。"

输出:
```json
{{
  "entities": [
    {{"name": "马云", "type": "PERSON", "mentions": 1, "confidence": 0.98, "description": "阿里巴巴集团创始人"}},
    {{"name": "阿里巴巴集团", "type": "ORGANIZATION", "mentions": 2, "confidence": 0.97, "description": "全球最大的电子商务公司之一"}},
    {{"name": "杭州", "type": "LOCATION", "mentions": 1, "confidence": 0.95, "description": "阿里巴巴集团的创立地"}}
  ]
}}
```"""

    user_prompt = f"请从以下文本中抽取实体，严格按照 JSON 格式输出：\n\n{text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_re_prompt(text: str, entities: list[dict]) -> list[dict]:
    """
    构造关系抽取（RE）的 prompt。

    基于已识别的实体，推理实体间关系。

    Args:
        text: 原始文本
        entities: 已识别的实体列表

    Returns:
        LLM 消息列表
    """
    # 构造实体列表描述
    entity_list = "\n".join([f"- {e['name']} ({e['type']})" for e in entities])

    system_prompt = f"""你是一个专业的关系抽取专家。你的任务是基于给定文本和已识别的实体，推理实体间的关系。

## 已识别的实体
{entity_list}

## 规则
1. 只从文本中能明确推断的关系才输出，不要臆造
2. 关系类型使用英文小写 + 下划线命名（如 works_at, located_in, founded_by, part_of, married_to 等）
3. 为每个关系给出置信度分数（0-1）
4. 提供支持该关系的原文片段作为证据
5. source 和 target 必须使用实体列表中的准确名称
6. 输出必须是严格的 JSON 格式

## 常用关系类型参考
- works_at: 在某组织工作/任职
- located_in: 位于某地
- founded_by: 由某人创立
- part_of: 属于/隶属于
- married_to: 与某人结婚
- born_in: 出生于
- created: 创造/发明
- participated_in: 参与了
- cooperates_with: 合作关系
- competes_with: 竞争关系
- belongs_to: 归属于
- happened_at: 发生在某地/某时

## 输出格式
```json
{{
  "relations": [
    {{
      "source": "源实体名称",
      "target": "目标实体名称",
      "relation_type": "关系类型",
      "confidence": 置信度,
      "evidence_text": "支持该关系的原文片段"
    }}
  ]
}}
```

## 示例

文本: "马云于1999年在杭州创立了阿里巴巴集团"
实体: 马云(PERSON), 阿里巴巴集团(ORGANIZATION), 杭州(LOCATION)

输出:
```json
{{
  "relations": [
    {{"source": "马云", "target": "阿里巴巴集团", "relation_type": "founded_by", "confidence": 0.98, "evidence_text": "马云于1999年在杭州创立了阿里巴巴集团"}},
    {{"source": "阿里巴巴集团", "target": "杭州", "relation_type": "located_in", "confidence": 0.90, "evidence_text": "于1999年在杭州创立了阿里巴巴集团"}}
  ]
}}
```"""

    user_prompt = f"请基于以下文本和已识别的实体，抽取实体间的关系，严格按照 JSON 格式输出：\n\n{text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_json_from_llm(response_text: str) -> dict:
    """
    从 LLM 响应中提取 JSON 内容。

    处理常见情况：直接 JSON、markdown 代码块包裹、前后有文字等。

    Args:
        response_text: LLM 原始输出文本

    Returns:
        解析后的字典

    Raises:
        ValueError: 无法解析 JSON 时
    """
    text = response_text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 { 和最后一个 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 输出中解析 JSON: {text[:200]}...")


# ==================== networkx 图分析 ====================


def build_graph(entities: list[dict], relations: list[dict]) -> nx.DiGraph:
    """
    从实体和关系数据构建 networkx 有向图。

    Args:
        entities: 实体列表
        relations: 关系列表

    Returns:
        networkx 有向图对象
    """
    G = nx.DiGraph()

    # 添加节点（实体）
    for entity in entities:
        G.add_node(
            entity["id"],
            name=entity["name"],
            type=entity["type"],
            mentions=entity.get("mentions", 1),
            confidence=entity.get("confidence", 0.8),
        )

    # 添加边（关系）
    for relation in relations:
        G.add_edge(
            relation["source_entity_id"],
            relation["target_entity_id"],
            relation_type=relation["relation_type"],
            confidence=relation.get("confidence", 0.8),
            evidence_text=relation.get("evidence_text", ""),
        )

    return G


def compute_graph_metrics(G: nx.DiGraph) -> dict:
    """
    计算知识图谱的拓扑结构指标。

    Args:
        G: networkx 有向图

    Returns:
        包含密度、平均度、连通分量数、中心节点等指标的字典
    """
    if G.number_of_nodes() == 0:
        return {
            "density": 0.0,
            "average_degree": 0.0,
            "connected_components": 0,
            "central_entities": [],
        }

    # 图密度
    density = nx.density(G)

    # 平均度数（无向视角）
    undirected = G.to_undirected()
    degrees = [d for _, d in undirected.degree()]
    average_degree = sum(degrees) / len(degrees) if degrees else 0.0

    # 弱连通分量数
    connected_components = nx.number_weakly_connected_components(G)

    # 度中心性（识别最重要的节点）
    centrality = nx.degree_centrality(G)

    # 取 top 5 中心节点
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    central_entities = []
    for node_id, score in sorted_nodes:
        node_data = G.nodes[node_id]
        central_entities.append(
            {
                "entity_id": node_id,
                "entity_name": node_data.get("name", node_id),
                "centrality_score": round(score, 4),
            }
        )

    return {
        "density": round(density, 4),
        "average_degree": round(average_degree, 2),
        "connected_components": connected_components,
        "central_entities": central_entities,
    }


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="AI 知识图谱抽取系统",
    description="基于 LLM 的实体识别和关系抽取，自动生成知识图谱",
    version="1.0.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载导出文件目录
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")


# ==================== API 路由 ====================


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/extract-knowledge-graph")
async def extract_knowledge_graph(request: dict):
    """
    从文本中抽取实体和关系，构建知识图谱。

    完整流程：
    1. 调用 LLM API 进行命名实体识别（NER）
    2. 调用 LLM API 进行关系抽取（RE）
    3. 使用 networkx 构建图结构
    4. 过滤低置信度结果
    5. 返回图谱数据 + 统计指标

    请求体参数：
    - text: 待分析文本
    - entity_types: 目标实体类型列表（可选）
    - max_entities: 最大实体数量（可选，默认 50）
    - confidence_threshold: 置信度阈值（可选，默认 0.7）
    """
    start_time = time.time()

    text = request.get("text", "").strip()
    entity_types = request.get("entity_types", DEFAULT_ENTITY_TYPES)
    max_entities = request.get("max_entities", 50)
    confidence_threshold = request.get("confidence_threshold", 0.7)

    # 参数校验
    if not text:
        raise HTTPException(status_code=400, detail="文本内容不能为空")
    if len(text) > 50000:
        raise HTTPException(status_code=400, detail="文本过长，最大支持 50000 字符")
    if max_entities < 1 or max_entities > 200:
        raise HTTPException(status_code=400, detail="max_entities 应在 1-200 之间")
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise HTTPException(status_code=400, detail="confidence_threshold 应在 0-1 之间")

    # 验证实体类型
    valid_types = {et["type"] for et in ENTITY_TYPES}
    for t in entity_types:
        if t not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的实体类型: {t}，可选: {', '.join(valid_types)}",
            )

    # 如果文本过长，截取前 10000 字符（LLM 上下文限制）
    analysis_text = text[:10000] if len(text) > 10000 else text

    # ========== 步骤 1：实体抽取（NER） ==========
    ner_messages = build_ner_prompt(analysis_text, entity_types, max_entities)
    ner_response = await call_llm_api(ner_messages, temperature=0.2)

    try:
        ner_result = parse_json_from_llm(ner_response)
        raw_entities = ner_result.get("entities", [])
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"实体抽取结果解析失败: {str(e)}")

    if not raw_entities:
        return {
            "entities": [],
            "relations": [],
            "graph_stats": {
                "total_entities": 0,
                "total_relations": 0,
                "processing_time": round(time.time() - start_time, 2),
            },
            "status": "success",
        }

    # 为每个实体分配 ID，并按置信度过滤
    entities = []
    entity_name_to_id = {}  # 名称到 ID 的映射（关系抽取时需要）

    for raw_entity in raw_entities:
        confidence = raw_entity.get("confidence", 0.8)
        if confidence < confidence_threshold:
            continue

        entity_id = str(uuid.uuid4())[:8]
        entity = {
            "id": entity_id,
            "name": raw_entity.get("name", ""),
            "type": raw_entity.get("type", "CONCEPT"),
            "mentions": raw_entity.get("mentions", 1),
            "confidence": round(confidence, 2),
            "description": raw_entity.get("description", ""),
        }
        entities.append(entity)
        entity_name_to_id[entity["name"]] = entity_id

    # 限制最大实体数
    entities = entities[:max_entities]
    entity_name_to_id = {e["name"]: e["id"] for e in entities}

    # ========== 步骤 2：关系抽取（RE） ==========
    relations = []
    if len(entities) >= 2:
        re_messages = build_re_prompt(analysis_text, entities)
        re_response = await call_llm_api(re_messages, temperature=0.2)

        try:
            re_result = parse_json_from_llm(re_response)
            raw_relations = re_result.get("relations", [])
        except (ValueError, KeyError):
            # 关系抽取失败不影响整体流程，返回空关系
            raw_relations = []

        # 处理关系数据，映射实体名称到 ID
        for raw_rel in raw_relations:
            source_name = raw_rel.get("source", "")
            target_name = raw_rel.get("target", "")
            confidence = raw_rel.get("confidence", 0.8)

            # 跳过低置信度关系
            if confidence < confidence_threshold:
                continue

            # 确保源和目标实体都存在
            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)

            if not source_id or not target_id:
                continue
            if source_id == target_id:
                continue  # 跳过自环

            relation = {
                "id": str(uuid.uuid4())[:8],
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "relation_type": raw_rel.get("relation_type", "related_to"),
                "confidence": round(confidence, 2),
                "evidence_text": raw_rel.get("evidence_text", ""),
            }
            relations.append(relation)

    # 计算处理时间
    processing_time = round(time.time() - start_time, 2)

    return {
        "entities": entities,
        "relations": relations,
        "graph_stats": {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "processing_time": processing_time,
        },
        "status": "success",
    }


@app.post("/api/analyze-graph-metrics")
async def analyze_graph_metrics(request: dict):
    """
    分析知识图谱的拓扑结构指标。

    使用 networkx 计算：
    - 图密度
    - 平均度数
    - 连通分量数
    - 中心性排名 top 5 实体

    请求体参数：
    - entities: 实体数组
    - relations: 关系数组
    """
    entities = request.get("entities", [])
    relations = request.get("relations", [])

    if not entities:
        raise HTTPException(status_code=400, detail="实体列表不能为空")

    # 构建 networkx 图
    G = build_graph(entities, relations)

    # 计算指标
    metrics = compute_graph_metrics(G)

    return {"metrics": metrics, "status": "success"}


@app.get("/api/entity-types")
async def get_entity_types():
    """
    获取支持的实体类型列表。

    返回所有可用的实体类型、描述和示例。
    """
    return {"entity_types": ENTITY_TYPES, "status": "success"}


@app.post("/api/export-graph")
async def export_graph(request: dict):
    """
    导出知识图谱数据。

    支持格式：
    - json: 标准 JSON 格式（包含实体和关系）
    - csv: CSV 格式（实体表 + 关系表打包）
    - graphml: GraphML 格式（可导入 Gephi 等工具）

    请求体参数：
    - entities: 实体数组
    - relations: 关系数组
    - format: 导出格式 ("json" | "csv" | "graphml")
    """
    entities = request.get("entities", [])
    relations = request.get("relations", [])
    export_format = request.get("format", "json")

    if not entities:
        raise HTTPException(status_code=400, detail="实体列表不能为空")
    if export_format not in ("json", "csv", "graphml"):
        raise HTTPException(status_code=400, detail="format 必须为 'json'、'csv' 或 'graphml'")

    file_id = str(uuid.uuid4())[:8]

    if export_format == "json":
        # JSON 导出
        filename = f"knowledge_graph_{file_id}.json"
        filepath = EXPORT_DIR / filename
        export_data = {
            "entities": entities,
            "relations": relations,
            "metadata": {
                "total_entities": len(entities),
                "total_relations": len(relations),
                "export_format": "json",
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    elif export_format == "csv":
        # CSV 导出（实体和关系分两个文件，打包为一个 CSV，用空行分隔）
        filename = f"knowledge_graph_{file_id}.csv"
        filepath = EXPORT_DIR / filename

        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)

            # 实体表
            writer.writerow(["## Entities"])
            writer.writerow(["id", "name", "type", "mentions", "confidence", "description"])
            for entity in entities:
                writer.writerow([
                    entity.get("id", ""),
                    entity.get("name", ""),
                    entity.get("type", ""),
                    entity.get("mentions", 1),
                    entity.get("confidence", 0),
                    entity.get("description", ""),
                ])

            writer.writerow([])  # 空行分隔

            # 关系表
            writer.writerow(["## Relations"])
            writer.writerow(["id", "source_entity_id", "target_entity_id", "relation_type", "confidence", "evidence_text"])
            for relation in relations:
                writer.writerow([
                    relation.get("id", ""),
                    relation.get("source_entity_id", ""),
                    relation.get("target_entity_id", ""),
                    relation.get("relation_type", ""),
                    relation.get("confidence", 0),
                    relation.get("evidence_text", ""),
                ])

    elif export_format == "graphml":
        # GraphML 导出（可用 Gephi / yEd 等工具打开）
        filename = f"knowledge_graph_{file_id}.graphml"
        filepath = EXPORT_DIR / filename

        G = build_graph(entities, relations)
        # 将节点属性中的 description 也加入
        for entity in entities:
            if entity["id"] in G.nodes:
                G.nodes[entity["id"]]["description"] = entity.get("description", "")

        nx.write_graphml(G, str(filepath))

    # 获取文件大小
    file_size = filepath.stat().st_size

    return {
        "download_url": f"/exports/{filename}",
        "filename": filename,
        "file_size": file_size,
        "status": "success",
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
