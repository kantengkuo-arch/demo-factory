"""
AI 文档智能问答（RAG）- 后端服务

上传 PDF/TXT/DOCX 文档，通过向量检索 + LLM 生成实现智能问答，
支持答案来源引用和多文档管理。

技术栈：FastAPI + LangChain + ChromaDB + Sentence-Transformers
"""

import os
import uuid
import shutil
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ────────────────────────────── 配置 ──────────────────────────────

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# 嵌入模型（首次运行会自动下载，约 90MB）
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 文本分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ────────────────────────────── 初始化 ──────────────────────────────

app = FastAPI(title="RAG 文档智能问答", version="1.0.0")

# CORS 配置 - 允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化嵌入模型（全局单例）
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 初始化 ChromaDB 持久化客户端
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# 内存中的文档元数据注册表（生产环境应用数据库替代）
# 格式: { document_id: { "filename": str, "upload_time": str, "collection_name": str, "pages": int } }
_document_registry: dict[str, dict] = {}

# ────────────────────────────── 数据模型 ──────────────────────────────


class ChatRequest(BaseModel):
    """问答请求体"""
    question: str
    document_id: Optional[str] = None
    collection_name: Optional[str] = "default"


class SourceItem(BaseModel):
    """引用来源条目"""
    content: str
    page: int
    score: float


class ChatResponse(BaseModel):
    """问答响应体"""
    answer: str
    sources: list[SourceItem]
    status: str = "success"


# ────────────────────────────── 工具函数 ──────────────────────────────


def _get_or_create_collection(name: str):
    """获取或创建 ChromaDB 集合"""
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """使用 Sentence-Transformers 将文本转换为向量"""
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def _load_document(file_path: str) -> list:
    """
    根据文件扩展名选择合适的 loader 加载文档。
    返回 LangChain Document 列表。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 PDF/TXT/DOCX")
    return loader.load()


def _generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    基于检索到的上下文生成回答。
    当前采用简单的上下文拼接摘要策略（无需外部 LLM API）。
    如需接入 LLM，替换此函数即可。
    """
    if not context_chunks:
        return "抱歉，我在已上传的文档中没有找到与您问题相关的内容。请尝试换个问法，或上传包含相关信息的文档。"

    # 构建基于上下文的摘要式回答
    combined = "\n\n---\n\n".join(context_chunks)
    answer = (
        f"根据文档内容，以下是与「{question}」相关的信息：\n\n"
        f"{combined}\n\n"
        f"以上内容直接引用自您上传的文档。如需更深入的分析，请提出更具体的问题。"
    )
    return answer


# ────────────────────────────── 路由 ──────────────────────────────


@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
):
    """
    上传文档并处理（向量化存储）。

    接受 PDF / TXT / DOCX 文件，解析后切块、嵌入并存入 ChromaDB。
    """
    # 校验文件类型
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，仅支持 {', '.join(allowed_extensions)}",
        )

    # 保存上传文件到磁盘
    document_id = str(uuid.uuid4())
    save_dir = os.path.join(UPLOAD_DIR, document_id)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename or "document")

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 加载并解析文档
        documents = _load_document(file_path)
        pages = len(documents)

        if pages == 0:
            raise HTTPException(status_code=400, detail="文档内容为空，无法处理")

        # 文本分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(status_code=400, detail="文档切块后内容为空")

        # 提取文本和元数据
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "page": chunk.metadata.get("page", 0),
                "source": file.filename or "unknown",
            }
            for chunk in chunks
        ]
        ids = [f"{document_id}_{i}" for i in range(len(texts))]

        # 向量化并存入 ChromaDB
        embeddings = _embed_texts(texts)
        collection = _get_or_create_collection(collection_name)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        # 注册文档元数据
        _document_registry[document_id] = {
            "filename": file.filename or "unknown",
            "upload_time": datetime.now(timezone.utc).isoformat(),
            "collection_name": collection_name,
            "pages": pages,
        }

        return {
            "document_id": document_id,
            "filename": file.filename,
            "pages": pages,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        # 清理失败的上传
        shutil.rmtree(save_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_documents(req: ChatRequest):
    """
    基于已上传文档进行智能问答。

    通过向量相似度检索相关文档片段，然后生成回答并返回引用来源。
    """
    try:
        collection_name = req.collection_name or "default"

        # 获取集合
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"集合 '{collection_name}' 不存在，请先上传文档",
            )

        # 构建查询过滤条件（可选按 document_id 过滤）
        where_filter = None
        if req.document_id:
            where_filter = {"document_id": req.document_id}

        # 向量检索
        query_embedding = _embed_texts([req.question])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            where=where_filter,
        )

        # 解析检索结果
        sources: list[SourceItem] = []
        context_chunks: list[str] = []

        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            for doc_text, meta, distance in zip(docs, metas, distances):
                # ChromaDB cosine distance → 相似度分数（1 - distance）
                similarity_score = round(1.0 - distance, 4)
                sources.append(SourceItem(
                    content=doc_text,
                    page=meta.get("page", 0),
                    score=similarity_score,
                ))
                context_chunks.append(doc_text)

        # 生成回答
        answer = _generate_answer(req.question, context_chunks)

        return ChatResponse(answer=answer, sources=sources, status="success")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@app.get("/api/documents")
def list_documents():
    """获取已上传的文档列表"""
    try:
        documents = [
            {
                "id": doc_id,
                "filename": meta["filename"],
                "upload_time": meta["upload_time"],
            }
            for doc_id, meta in _document_registry.items()
        ]
        return {"documents": documents, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    """
    删除指定文档。

    从 ChromaDB 中移除该文档的所有向量，并清理本地文件。
    """
    try:
        # 检查文档是否存在
        if document_id not in _document_registry:
            raise HTTPException(status_code=404, detail=f"文档 {document_id} 不存在")

        meta = _document_registry[document_id]
        collection_name = meta["collection_name"]

        # 从 ChromaDB 中删除该文档的所有 chunk
        try:
            collection = chroma_client.get_collection(name=collection_name)
            # 查询该文档的所有 chunk ID
            all_ids = collection.get(
                where={"document_id": document_id},
                include=[],
            )
            if all_ids and all_ids["ids"]:
                collection.delete(ids=all_ids["ids"])
        except Exception:
            pass  # 集合可能已被删除，忽略

        # 清理本地文件
        upload_dir = os.path.join(UPLOAD_DIR, document_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir, ignore_errors=True)

        # 从注册表移除
        del _document_registry[document_id]

        return {"status": "success"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


# ────────────────────────────── 启动 ──────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
