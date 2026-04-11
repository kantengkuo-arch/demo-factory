
## 图片风格转换工具 — 2026-04-11
**技术点**：VGG19 Neural Style Transfer、Gram 矩阵风格损失、L-BFGS 优化器、FastAPI 异步任务+线程模型、前端拖拽上传+进度轮询
**可复用**：① 异步任务架构（POST 创建 → task_id → GET 轮询）适用于所有耗时 AI 推理场景 ② 前端 Mock 模式自动检测（fetch /health 超时降级）③ toast 通知组件 ④ 暗色主题 CSS 变量体系
**踩坑**：① load_image 里做了 ImageNet Normalize，模型内部又有 Normalization 层——导致双重标准化，标准做法是只在模型内部做一次 ② 预设风格定义了但没放实际图片文件，前端降级到 emoji 体验差 ③ Stable Diffusion 选项后端未实现但前端可选，应禁用或加提示
**评分**：92/100
