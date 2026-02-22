#!/usr/bin/env python3
"""RAG QA server (stdlib HTTP) for local ChromaDB + DeepSeek API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, parse, request

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

DEFAULT_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "regulation_chunks_v1")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
DEFAULT_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))


@dataclass
class ReferenceItem:
    rank: int
    chunk_id: str
    score: float
    doc_id: str
    doc_name: str
    section_path: str
    page_start: Optional[int]
    page_end: Optional[int]
    matched_text: str
    context_before: str
    context_after: str


class RAGService:
    def __init__(self, chroma_dir: str, embed_model: str) -> None:
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.embedder = None
        self.embed_model_name = embed_model
        disable_embed = os.getenv("DISABLE_EMBED_MODEL", "0") == "1"
        if disable_embed:
            print("Embedding model disabled by DISABLE_EMBED_MODEL=1; using keyword retrieval.")
            return
        try:
            self.embedder = SentenceTransformer(
                embed_model, device=os.getenv("EMBED_DEVICE", "cpu")
            )
            print(f"Embedding model loaded: {embed_model}")
        except Exception as exc:
            print(
                "Warning: embedding model load failed, will fallback to keyword retrieval. "
                f"detail={exc}"
            )

    def list_collections(self) -> List[str]:
        return [col.name for col in self.client.list_collections()]

    def _get_collection(self, collection_name: str):
        names = self.list_collections()
        if collection_name not in names:
            raise ValueError(
                f"Collection '{collection_name}' 不存在，可用 collections: {names}"
            )
        return self.client.get_collection(name=collection_name)

    @staticmethod
    def _normalize_page(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _doc_chunks(collection: Any, doc_id: str, cache: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if doc_id in cache:
            return cache[doc_id]
        if not doc_id:
            cache[doc_id] = []
            return cache[doc_id]

        data = collection.get(where={"doc_id": doc_id}, include=["metadatas", "documents"])
        rows: List[Dict[str, Any]] = []
        for idx, chunk_id in enumerate(data.get("ids", [])):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document": (data.get("documents") or [""])[idx] or "",
                    "metadata": (data.get("metadatas") or [{}])[idx] or {},
                }
            )

        def source_row_key(item: Dict[str, Any]) -> int:
            raw = (item.get("metadata") or {}).get("source_row", 10**9)
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 10**9

        rows.sort(key=source_row_key)
        cache[doc_id] = rows
        return rows

    def _neighbor_context(
        self,
        collection: Any,
        doc_id: str,
        chunk_id: str,
        cache: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        chunks = self._doc_chunks(collection=collection, doc_id=doc_id, cache=cache)
        if not chunks:
            return {"before": "", "after": ""}

        for i, item in enumerate(chunks):
            if item["chunk_id"] != chunk_id:
                continue
            before = chunks[i - 1]["document"] if i > 0 else ""
            after = chunks[i + 1]["document"] if i < len(chunks) - 1 else ""
            return {"before": before, "after": after}
        return {"before": "", "after": ""}

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        text = text.strip().lower()
        if not text:
            return []
        # Keep Chinese words and alnum terms.
        terms = [t for t in re.split(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", text) if t]
        return terms

    @staticmethod
    def _char_bigrams(text: str) -> List[str]:
        compact = re.sub(r"\s+", "", text.lower())
        if len(compact) < 2:
            return [compact] if compact else []
        return [compact[i : i + 2] for i in range(len(compact) - 1)]

    def _keyword_search(self, question: str, collection_name: str, top_k: int) -> List[ReferenceItem]:
        collection = self._get_collection(collection_name=collection_name)
        raw = collection.get(include=["metadatas", "documents"])

        ids = raw.get("ids", [])
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])
        if not ids:
            return []

        terms = self._extract_terms(question)
        bigrams = self._char_bigrams(question)

        scored: List[Dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            doc_text = self._safe_text(docs[idx])
            if not doc_text:
                continue

            lowered = doc_text.lower()
            term_hit = sum(lowered.count(t) for t in terms if len(t) >= 2)
            bigram_hit = sum(1 for bg in bigrams if bg and bg in lowered)
            score = term_hit + 0.1 * bigram_hit
            if score <= 0:
                continue

            scored.append(
                {
                    "chunk_id": chunk_id,
                    "document": doc_text,
                    "metadata": metas[idx] or {},
                    "score": float(score),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        picked = scored[:top_k]

        refs: List[ReferenceItem] = []
        doc_cache: Dict[str, List[Dict[str, Any]]] = {}
        for rank, item in enumerate(picked, start=1):
            metadata = item["metadata"]
            doc_id = self._safe_text(metadata.get("doc_id"))
            ctx = self._neighbor_context(
                collection=collection,
                doc_id=doc_id,
                chunk_id=item["chunk_id"],
                cache=doc_cache,
            )
            refs.append(
                ReferenceItem(
                    rank=rank,
                    chunk_id=item["chunk_id"],
                    score=round(item["score"], 6),
                    doc_id=doc_id,
                    doc_name=self._safe_text(metadata.get("doc_name")),
                    section_path=self._safe_text(metadata.get("section_path")),
                    page_start=self._normalize_page(metadata.get("page_start")),
                    page_end=self._normalize_page(metadata.get("page_end")),
                    matched_text=self._safe_text(item["document"]),
                    context_before=self._safe_text(ctx["before"]),
                    context_after=self._safe_text(ctx["after"]),
                )
            )
        return refs

    def search(self, question: str, collection_name: str, top_k: int) -> List[ReferenceItem]:
        if self.embedder is None:
            return self._keyword_search(
                question=question,
                collection_name=collection_name,
                top_k=top_k,
            )

        collection = self._get_collection(collection_name=collection_name)
        query_vector = self.embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        result = collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        refs: List[ReferenceItem] = []
        doc_cache: Dict[str, List[Dict[str, Any]]] = {}
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] or {}
            document = documents[idx] or ""
            distance = float(distances[idx]) if idx < len(distances) else 1.0

            doc_id = self._safe_text(metadata.get("doc_id"))
            ctx = self._neighbor_context(
                collection=collection,
                doc_id=doc_id,
                chunk_id=chunk_id,
                cache=doc_cache,
            )

            refs.append(
                ReferenceItem(
                    rank=idx + 1,
                    chunk_id=chunk_id,
                    score=round(1.0 - distance, 6),
                    doc_id=doc_id,
                    doc_name=self._safe_text(metadata.get("doc_name")),
                    section_path=self._safe_text(metadata.get("section_path")),
                    page_start=self._normalize_page(metadata.get("page_start")),
                    page_end=self._normalize_page(metadata.get("page_end")),
                    matched_text=self._safe_text(document),
                    context_before=self._safe_text(ctx["before"]),
                    context_after=self._safe_text(ctx["after"]),
                )
            )
        return refs


def build_prompt(question: str, references: List[ReferenceItem]) -> str:
    if not references:
        return f"用户问题：{question}\n\n未检索到有效知识片段，请明确告知无法基于知识库回答。"

    context_chunks: List[str] = []
    for ref in references:
        context_chunks.append(
            "\n".join(
                [
                    f"[R{ref.rank}]",
                    f"doc_id: {ref.doc_id}",
                    f"doc_name: {ref.doc_name}",
                    f"section_path: {ref.section_path}",
                    f"page_start/page_end: {ref.page_start}/{ref.page_end}",
                    f"匹配片段: {ref.matched_text}",
                    f"上文: {ref.context_before}",
                    f"下文: {ref.context_after}",
                ]
            )
        )

    return "\n\n".join(
        [
            f"用户问题：{question}",
            "以下是检索片段：",
            "\n\n".join(context_chunks),
            "请仅基于以上内容回答，并在关键结论后标注引用编号，如 [R1][R2]。",
        ]
    )


def call_deepseek(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    req = request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"DeepSeek HTTPError {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
        return (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception as exc:
        raise RuntimeError(f"DeepSeek response parse failed: {exc}, body={body}") from exc


class QARequestHandler(BaseHTTPRequestHandler):
    rag_service: RAGService = None  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._json_response(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})
            return

        try:
            file_path.resolve().relative_to(FRONTEND_DIR.resolve())
        except ValueError:
            self._json_response(HTTPStatus.FORBIDDEN, {"detail": "Forbidden"})
            return

        content = file_path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript",
            "application/json",
        ):
            content_type = f"{content_type}; charset=utf-8"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = parse.urlparse(self.path).path
        if path == "/api/health":
            self._json_response(HTTPStatus.OK, {"status": "ok"})
            return

        if path == "/api/config":
            self._json_response(
                HTTPStatus.OK,
                {
                    "default_base_url": DEFAULT_DEEPSEEK_BASE_URL,
                    "default_model": DEFAULT_DEEPSEEK_MODEL,
                    "default_collection": DEFAULT_COLLECTION,
                    "available_collections": self.rag_service.list_collections(),
                },
            )
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            self._send_file(FRONTEND_DIR / rel)
            return

        if path in ("/", "/index.html"):
            self._send_file(FRONTEND_DIR / "index.html")
            return

        self._json_response(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})

    def do_POST(self) -> None:
        path = parse.urlparse(self.path).path
        if path != "/api/chat":
            self._json_response(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            payload = json.loads(raw_body.decode("utf-8") if raw_body else "{}")
        except json.JSONDecodeError:
            self._json_response(HTTPStatus.BAD_REQUEST, {"detail": "请求体不是合法 JSON。"})
            return
        if not isinstance(payload, dict):
            self._json_response(HTTPStatus.BAD_REQUEST, {"detail": "请求体必须是 JSON 对象。"})
            return

        question = str(payload.get("question", "")).strip()
        if not question:
            self._json_response(HTTPStatus.BAD_REQUEST, {"detail": "问题不能为空。"})
            return

        api_key = str(payload.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            self._json_response(
                HTTPStatus.BAD_REQUEST, {"detail": "缺少 DeepSeek API Key，请在设置中填写。"}
            )
            return

        base_url = str(payload.get("base_url") or DEFAULT_DEEPSEEK_BASE_URL).strip()
        model = str(payload.get("model") or DEFAULT_DEEPSEEK_MODEL).strip()
        collection_name = str(payload.get("collection_name") or DEFAULT_COLLECTION).strip()

        try:
            top_k = int(payload.get("top_k", DEFAULT_TOP_K))
        except (TypeError, ValueError):
            top_k = DEFAULT_TOP_K
        top_k = max(1, min(20, top_k))

        try:
            temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = DEFAULT_TEMPERATURE
        temperature = max(0.0, min(1.5, temperature))

        try:
            references = self.rag_service.search(
                question=question,
                collection_name=collection_name,
                top_k=top_k,
            )
        except Exception as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"detail": f"检索失败: {exc}"})
            return

        system_prompt = (
            "你是一个严谨的法规问答助手。"
            "只能依据给定检索片段回答，不要编造条款或页码；"
            "如果依据不足，需要明确说明依据不足。"
        )
        user_prompt = build_prompt(question=question, references=references)

        try:
            answer = call_deepseek(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
        except Exception as exc:
            self._json_response(HTTPStatus.BAD_GATEWAY, {"detail": str(exc)})
            return

        self._json_response(
            HTTPStatus.OK,
            {
                "answer": answer,
                "used_model": model,
                "references": [asdict(ref) for ref in references],
            },
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local RAG QA server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    if not FRONTEND_DIR.exists():
        raise SystemExit(f"Frontend directory not found: {FRONTEND_DIR}")

    rag_service = RAGService(chroma_dir=DEFAULT_CHROMA_DIR, embed_model=DEFAULT_EMBED_MODEL)
    QARequestHandler.rag_service = rag_service

    server = ThreadingHTTPServer((args.host, args.port), QARequestHandler)
    print(f"Server started on http://{args.host}:{args.port}")
    print(f"Using collection: {DEFAULT_COLLECTION}")
    print(f"Using chroma dir: {DEFAULT_CHROMA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
