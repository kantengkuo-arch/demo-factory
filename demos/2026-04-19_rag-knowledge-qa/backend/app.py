"""
RAG 智能知识库问答系统 — 后端服务

完整 RAG 流程：文档上传 → 文本解析 → 智能分块 → Embedding API 向量化
→ ChromaDB 存储 → 相似度检索 → LLM API 生成回答

所有模型调用均通过外部 API，不依赖任何本地模型。
"""

import json
import os
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ==================== 配置 ====================

# 模型 API 配置文件路径
MODEL_CONFIG_PATH = Path.home() / "projects" / "demo-factory" / "platform" / "config" / "model_api.json"

# 数据存储目录
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chromadb"

for d in [UPLOAD_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 文档元信息存储（内存 + JSON 持久化）
DOC_META_PATH = DATA_DIR / "documents_meta.json"
CHAT_HISTORY_PATH = DATA_DIR / "chat_history.json"


def load_model_config() -> dict:
    """
    从配置文件加载模型 API 信息。

    Returns:
        包含 chat 和 embedding 配置的字典
    """
    if not MODEL_CONFIG_PATH.exists():
        raise RuntimeError(f"模型配置文件不存在: {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_store(path: Path, default=None):
    """从 JSON 文件加载数据，不存在则返回默认值"""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


def save_json_store(path: Path, data):
    """将数据保存到 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==================== 文档解析 ====================


def parse_pdf(file_path: str) -> str:
    """
    解析 PDF 文件，提取全部文本内容。

    Args:
        file_path: PDF 文件路径

    Returns:
        提取的文本字符串
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def parse_docx(file_path: str) -> str:
    """
    解析 DOCX 文件，提取全部段落文本。

    Args:
        file_path: DOCX 文件路径

    Returns:
        提取的文本字符串
    """
    from docx import Document

    doc = Document(file_path)
    texts = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(texts)


def parse_text(file_path: str) -> str:
    """
    读取纯文本文件（TXT / Markdown）。

    Args:
        file_path: 文件路径

    Returns:
        文件文本内容
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_document(file_path: str, filename: str) -> str:
    """
    根据文件扩展名自动选择解析器。

    Args:
        file_path: 文件路径
        filename: 原始文件名（用于判断扩展名）

    Returns:
        提取的文本内容

    Raises:
        ValueError: 不支持的文件格式
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in (".txt", ".md", ".markdown"):
        return parse_text(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，支持 PDF/TXT/MD/DOCX")


# ==================== 文本分块 ====================


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    将长文本按固定长度切分为带重叠的 chunks。

    使用段落边界作为优先切分点，保持语义完整性。
    当找不到合适的段落边界时，回退到句号/换行符切分。

    Args:
        text: 原始文本
        chunk_size: 每块最大字符数
        overlap: 相邻块之间的重叠字符数

    Returns:
        文本块列表
    """
    if not text or not text.strip():
        return []

    # 如果文本很短，直接返回
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # 如果还没到结尾，尝试在自然断点处切分
        if end < len(text):
            # 优先在段落边界切分
            split_pos = text.rfind("\n\n", start, end)
            if split_pos == -1 or split_pos <= start:
                # 其次在换行处切分
                split_pos = text.rfind("\n", start, end)
            if split_pos == -1 or split_pos <= start:
                # 再次在句号处切分（中英文句号）
                for sep in ["。", ".", "！", "!", "？", "?"]:
                    split_pos = text.rfind(sep, start, end)
                    if split_pos > start:
                        split_pos += 1  # 包含句号
                        break
            if split_pos > start:
                end = split_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 下一块的起始位置（带重叠）
        start = end - overlap if end < len(text) else end

    return chunks


# ==================== Embedding API 调用 ====================


async def call_embedding_api(texts: list[str]) -> list[list[float]]:
    """
    调用外部 Embedding API 将文本列表转换为向量。

    使用配置文件中的 embedding 配置，兼容 OpenAI 格式的 API。

    Args:
        texts: 待向量化的文本列表

    Returns:
        向量列表，每个向量是一个 float 列表

    Raises:
        HTTPException: API 调用失败时
    """
    config = load_model_config()
    embedding_config = config["embedding"]

    api_url = embedding_config["api_url"].rstrip("/") + "/embeddings"
    api_key = embedding_config["api_key"]
    model = embedding_config["model"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": texts,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            # 按 index 排序，确保顺序一致
            embeddings = sorted(result["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings]
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Embedding API 调用失败: {e.response.status_code} - {e.response.text}",
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Embedding API 调用异常: {str(e)}")


# ==================== LLM Chat API 调用 ====================


async def call_chat_api(messages: list[dict], temperature: float = 0.7) -> str:
    """
    调用外部 LLM Chat API 生成回答。

    使用配置文件中的 chat 配置，兼容 OpenAI Chat Completions 格式。

    Args:
        messages: 对话消息列表，格式 [{"role": "system"|"user"|"assistant", "content": "..."}]
        temperature: 生成温度，0-1

    Returns:
        模型生成的回答文本

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
                detail=f"Chat API 调用失败: {e.response.status_code} - {e.response.text}",
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Chat API 调用异常: {str(e)}")


# ==================== ChromaDB 向量存储 ====================

# 初始化 ChromaDB 持久化客户端
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# 使用一个 collection 存储所有文档的向量
COLLECTION_NAME = "rag_knowledge_base"


def get_collection():
    """获取或创建 ChromaDB collection"""
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="RAG 智能知识库问答系统",
    description="上传文档，智能对话问答，支持来源追溯",
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


# ==================== API 路由 ====================


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=1000),
    overlap: int = Form(default=200),
):
    """
    上传文档并自动完成全流程处理：解析 → 分块 → 向量化 → 存储。

    - file: 文档文件（PDF/TXT/MD/DOCX）
    - chunk_size: 分块大小，默认 1000 字符
    - overlap: 重叠长度，默认 200 字符

    返回文档 ID 和分块数量。
    """
    # 验证文件格式
    filename = file.filename or "unknown.txt"
    ext = Path(filename).suffix.lower()
    allowed_exts = {".pdf", ".txt", ".md", ".markdown", ".docx"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持: {', '.join(allowed_exts)}",
        )

    # 参数校验
    if chunk_size < 100 or chunk_size > 10000:
        raise HTTPException(status_code=400, detail="chunk_size 应在 100-10000 之间")
    if overlap < 0 or overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="overlap 应在 0 到 chunk_size 之间")

    # 生成文档 ID 并保存文件
    document_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{document_id}{ext}"

    file_content = await file.read()
    file_size = len(file_content)
    with open(save_path, "wb") as f:
        f.write(file_content)

    try:
        # 步骤 1：解析文档
        text = parse_document(str(save_path), filename)
        if not text.strip():
            raise HTTPException(status_code=400, detail="文档内容为空，无法处理")

        # 步骤 2：文本分块
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档分块后无有效内容")

        # 步骤 3：调用 Embedding API 向量化
        # 分批处理，避免单次请求过大（每批最多 20 个 chunks）
        BATCH_SIZE = 20
        all_embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            batch_embeddings = await call_embedding_api(batch)
            all_embeddings.extend(batch_embeddings)

        # 步骤 4：存储到 ChromaDB
        collection = get_collection()
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=chunk_ids,
            embeddings=all_embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        # 保存文档元信息
        docs_meta = load_json_store(DOC_META_PATH, default={})
        docs_meta[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "upload_time": datetime.now(timezone.utc).isoformat(),
            "total_chunks": len(chunks),
            "file_size": file_size,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        save_json_store(DOC_META_PATH, docs_meta)

        return {
            "document_id": document_id,
            "filename": filename,
            "total_chunks": len(chunks),
            "processing_status": "completed",
            "status": "success",
        }

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        # 清理已保存的文件
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """
    获取已上传的文档列表。

    返回所有文档的基本信息（ID、文件名、上传时间、分块数、文件大小）。
    """
    docs_meta = load_json_store(DOC_META_PATH, default={})

    documents = [
        {
            "document_id": meta["document_id"],
            "filename": meta["filename"],
            "upload_time": meta["upload_time"],
            "total_chunks": meta["total_chunks"],
            "file_size": meta["file_size"],
        }
        for meta in docs_meta.values()
    ]

    # 按上传时间倒序排列
    documents.sort(key=lambda x: x["upload_time"], reverse=True)

    return {"documents": documents, "status": "success"}


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """
    删除指定文档及其所有向量数据。

    路径参数：
    - document_id: 文档 ID

    同时清理：上传文件、ChromaDB 中的向量、元信息记录。
    """
    docs_meta = load_json_store(DOC_META_PATH, default={})

    if document_id not in docs_meta:
        raise HTTPException(status_code=404, detail=f"未找到文档: {document_id}")

    meta = docs_meta[document_id]

    # 从 ChromaDB 删除该文档的所有 chunks
    try:
        collection = get_collection()
        # 查询该文档的所有 chunk IDs
        results = collection.get(
            where={"document_id": document_id},
            include=[],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception as e:
        # 即使向量删除失败也继续清理其他数据
        pass

    # 删除上传的文件
    for ext in [".pdf", ".txt", ".md", ".markdown", ".docx"]:
        file_path = UPLOAD_DIR / f"{document_id}{ext}"
        if file_path.exists():
            file_path.unlink()
            break

    # 删除元信息
    del docs_meta[document_id]
    save_json_store(DOC_META_PATH, docs_meta)

    return {"message": f"文档 '{meta['filename']}' 已成功删除", "status": "success"}


@app.post("/api/chat")
async def chat(request: dict):
    """
    基于已上传文档进行智能问答。

    RAG 流程：
    1. 将用户问题向量化
    2. 在 ChromaDB 中检索最相关的文档片段
    3. 拼接上下文 + 问题构造 prompt
    4. 调用 LLM API 生成回答

    请求体参数：
    - question: 用户问题
    - document_ids: 限定搜索的文档 ID 列表（空数组搜索全部）
    - top_k: 检索的相关片段数量，默认 5
    - temperature: 生成温度 0-1，默认 0.7
    """
    question = request.get("question", "").strip()
    document_ids = request.get("document_ids", [])
    top_k = request.get("top_k", 5)
    temperature = request.get("temperature", 0.7)

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k 应在 1-20 之间")
    if temperature < 0 or temperature > 1:
        raise HTTPException(status_code=400, detail="temperature 应在 0-1 之间")

    collection = get_collection()

    # 检查是否有文档
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="尚未上传任何文档，请先上传文档后再提问")

    # 步骤 1：将问题向量化
    question_embedding = (await call_embedding_api([question]))[0]

    # 步骤 2：在 ChromaDB 中检索相关片段
    query_params = {
        "query_embeddings": [question_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    # 如果指定了文档 ID，添加过滤条件
    if document_ids:
        if len(document_ids) == 1:
            query_params["where"] = {"document_id": document_ids[0]}
        else:
            query_params["where"] = {"document_id": {"$in": document_ids}}

    results = collection.query(**query_params)

    # 解析检索结果
    sources = []
    context_parts = []

    if results["documents"] and results["documents"][0]:
        for i, (doc, meta, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # ChromaDB cosine distance → similarity score (1 - distance)
            similarity = max(0.0, 1.0 - distance)
            sources.append(
                {
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "chunk_text": doc,
                    "similarity_score": round(similarity, 4),
                }
            )
            context_parts.append(f"[来源 {i + 1}: {meta['filename']}]\n{doc}")

    # 步骤 3：构造 RAG prompt
    context_text = "\n\n---\n\n".join(context_parts) if context_parts else "（未检索到相关内容）"

    system_prompt = (
        "你是一个智能知识库问答助手。请根据以下检索到的文档内容回答用户的问题。\n\n"
        "回答要求：\n"
        "1. 仅基于提供的文档内容作答，如果文档中没有相关信息，请明确说明\n"
        "2. 回答要准确、简洁、有条理\n"
        "3. 如果涉及多个来源，请综合整理后回答\n"
        "4. 使用中文回答\n\n"
        f"以下是检索到的相关文档内容：\n\n{context_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # 步骤 4：调用 LLM API 生成回答
    answer = await call_chat_api(messages, temperature=temperature)

    # 保存问答历史
    conversation_id = str(uuid.uuid4())
    history = load_json_store(CHAT_HISTORY_PATH, default=[])
    history.append(
        {
            "id": conversation_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_count": len(sources),
            "document_ids": document_ids,
        }
    )
    save_json_store(CHAT_HISTORY_PATH, history)

    return {
        "answer": answer,
        "sources": sources,
        "question": question,
        "status": "success",
    }


@app.get("/api/chat-history")
async def get_chat_history(limit: int = 50):
    """
    获取问答历史记录。

    查询参数：
    - limit: 返回的最大记录数，默认 50

    返回按时间倒序排列的对话历史。
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit 应在 1-500 之间")

    history = load_json_store(CHAT_HISTORY_PATH, default=[])

    # 按时间倒序，取最近 limit 条
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    recent = history[:limit]

    conversations = [
        {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "timestamp": item["timestamp"],
            "document_count": item.get("document_count", 0),
        }
        for item in recent
    ]

    return {"conversations": conversations, "status": "success"}


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
