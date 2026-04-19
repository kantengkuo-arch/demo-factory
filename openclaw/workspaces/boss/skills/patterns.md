
## 图片风格转换工具 — 2026-04-11
**技术点**：VGG19 Neural Style Transfer、Gram 矩阵风格损失、L-BFGS 优化器、FastAPI 异步任务+线程模型、前端拖拽上传+进度轮询
**可复用**：① 异步任务架构（POST 创建 → task_id → GET 轮询）适用于所有耗时 AI 推理场景 ② 前端 Mock 模式自动检测（fetch /health 超时降级）③ toast 通知组件 ④ 暗色主题 CSS 变量体系
**踩坑**：① load_image 里做了 ImageNet Normalize，模型内部又有 Normalization 层——导致双重标准化，标准做法是只在模型内部做一次 ② 预设风格定义了但没放实际图片文件，前端降级到 emoji 体验差 ③ Stable Diffusion 选项后端未实现但前端可选，应禁用或加提示
**评分**：92/100

## AI 时间序列预测 — 2026-04-14
**技术点**：FastAPI + Plotly.js 交互图表 + Google TimesFM 预训练模型（统计回退兜底）+ 多编码 CSV 解析 + 内置示例数据集
**可复用**：统计回退预测方法（线性趋势+季节性分解）可用于任何时序类 Demo；多编码 CSV 解析逻辑；Plotly 暗色主题图表配置；拖拽上传+骨架屏加载的前端模式
**踩坑**：前后端 API 参数名不一致（time_column vs date_column, horizon vs forecast_steps）导致联调失败——这个问题在 Mock 模式下完全隐藏，审查流程未做真实联调测试。教训：**审查必须关闭 Mock 模式做端到端测试**
**评分**：71/100

## RAG 智能知识库问答 — 2026-04-19
**技术点**：RAG 全流程（文档解析→智能分块→Embedding API→ChromaDB→LLM 生成）、零本地模型设计（SiliconFlow Embedding+Chat API）、JSON 持久化元信息、Pydantic 数据验证
**可复用**：① 外部 API 调用模式（从 config JSON 读取 api_url/api_key/model，httpx 异步请求，兼容 OpenAI 格式）② 智能分块函数（段落→换行→句号边界优先切分，带重叠）③ ChromaDB 持久化+余弦相似度检索 ④ ChatGPT 风格对话 UI + Markdown 渲染 + 来源引用折叠卡片 ⑤ JSON 文件持久化避免重启丢数据
**踩坑**：① 前端 `<link rel="stylesheet">` 误引 JS 文件，应删除或改为 `<script>` ② `/api/chat` 用 dict 而非 Pydantic model 接收参数，与其他接口风格不统一 ③ 之前 #19 的 RAG Demo 用了本地 sentence-transformers 模型导致依赖巨大+启动慢，这次改为全 API 调用是更好的方案
**评分**：95/100
