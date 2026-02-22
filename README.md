# ScaffoldSence: RAG-based Scaffolding Engineering QA System
# 基于RAG的脚手架工程问答系统

This project is a Retrieval-Augmented Generation (RAG) question-answering system built upon the current scaffolding engineering regulations and standards of the People's Republic of China. It integrates ChromaDB for vector retrieval and the DeepSeek API for generative responses, enabling accurate regulation-level querying and context-aware answers.
本项目基于中华人民共和国现行脚手架工程相关规范与标准文件，构建了一个面向工程实践场景的 RAG（Retrieval-Augmented Generation）问答系统。系统通过 ChromaDB 向量检索结合 DeepSeek 大模型生成能力，实现对规范条文的精准检索与语义级回答。



这个项目现在包含一个可直接运行的问答系统：
- 后端：纯 Python HTTP 服务（无额外 Web 框架依赖），读取本地 `chroma_db`，调用 DeepSeek API 生成答案
- 前端：ChatGPT 风格单页界面，包含聊天输入区和设置面板
- 检索：优先向量检索；若本机未缓存 embedding 模型会自动降级为关键词检索

## 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 启动服务

```bash
source .venv/bin/activate
python app/main.py --host 0.0.0.0 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 3. 设置说明

在右上角 `设置` 中填写：
- `DeepSeek API Key`（必须）
- `Base URL`（默认 `https://api.deepseek.com`）
- `Model`（默认 `deepseek-chat`，可改 `deepseek-reasoner`）
- `Collection`（默认 `regulation_chunks_v1`）
- `Top K`（默认 5）
- `Temperature`（默认 0.2）

设置会保存在浏览器 `localStorage`。

## 4. 返回内容

每次回答都会返回并展示引用片段，包含：
- `doc_id`
- `doc_name`
- `section_path`
- `page_start/page_end`
- `前后文（context_before/context_after）`

## 5. 可选环境变量

- `DEEPSEEK_API_KEY`：后端默认 API Key（前端不填时可用）
- `CHROMA_PERSIST_DIR`：默认 `./chroma_db`
- `CHROMA_COLLECTION`：默认 `regulation_chunks_v1`
- `EMBED_MODEL`：默认 `BAAI/bge-large-zh-v1.5`
- `EMBED_DEVICE`：默认 `cpu`
- `DISABLE_EMBED_MODEL`：设为 `1` 时不加载 embedding 模型，直接使用关键词检索（离线环境推荐）
- `DEEPSEEK_BASE_URL`：默认 `https://api.deepseek.com`
- `DEEPSEEK_MODEL`：默认 `deepseek-chat`
