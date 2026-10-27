#!/usr/bin/env python3
"""book.py 的独立示例：书籍解析、章节分块、千问向量化和 Chroma 检索。

无需启动 RAGFlow，也不依赖旁边的 naive_demo.py 或 qa_demo.py。
支持 TXT、Markdown、HTML、文本型 PDF 和 DOCX；不包含 DeepDOC OCR、
版面识别、图片描述、表格结构识别或 RAGFlow 检索分词字段。

核心流程与 book.py 一致：清理目录 -> 识别章节编号 -> hierarchical_merge；
找不到章节编号时才使用 naive_merge。层级模式不受 --chunk-size 限制，
保留原函数 depth=5 和短单段合并阈值 218；Token 数使用轻量估算。
为便于复现，章节类型使用全文统计，不使用原版的随机抽样。
目录清理保守处理，冒号标题仅识别短行；Markdown 为本 demo 扩展支持。

运行示例：
    python book_demo.py --chunks-only
    python book_demo.py book_test.md --chunks-only
    export DASHSCOPE_API_KEY='your-api-key'
    python book_demo.py book_test.md --query '如何提高检索准确率？'
    python book_demo.py book.pdf --from-page 2 --to-page 10

Chroma 默认保存到脚本同级 book_chroma_db/，集合名为 book-demo。
--chunks-only 只解析和打印分块，不需要任何第三方依赖（PDF/DOCX 除外），
不会请求向量接口，也不会写入数据库。
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import sys
import time
from bisect import bisect_left
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://ws-leyp0n1kyjsllxss.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "text-embedding-v2"
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent / "book_chroma_db"
DEFAULT_COLLECTION_NAME = "book-demo"
DEFAULT_DELIMITER = "\n!?。；！？"
DEFAULT_SAMPLE = """第一章 知识库基础
第一节 文档处理
书籍先被解析成有序文本段，再根据章节标题组织检索分块。
有章节编号时，每段正文会附带其上级标题，避免检索结果失去上下文。
第二节 向量检索
文档块和用户问题使用同一个向量模型编码，再按语义相似度检索。
第二章 检索优化
第一节 数据质量
清理重复内容、去掉目录，并保留完整章节语义，可以减少无关内容。
第二节 召回策略
合理设置召回数量，并增加重排序步骤，有助于提高检索准确率。
"""

# 这五组规则来自 rag/nlp/__init__.py 的 BULLET_PATTERN，组内顺序表示层级。
BULLET_PATTERN = [
    [
        r"第[零一二三四五六七八九十百0-9]+(分?编|部分)",
        r"第[零一二三四五六七八九十百0-9]+章",
        r"第[零一二三四五六七八九十百0-9]+节",
        r"第[零一二三四五六七八九十百0-9]+条",
        r"[\(（][零一二三四五六七八九十百]+[\)）]",
    ],
    [
        r"第[0-9]+章",
        r"第[0-9]+节",
        r"[0-9]{,2}[\. 、]",
        r"[0-9]{,2}\.[0-9]{,2}[^a-zA-Z/%~-]",
        r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
        r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
    ],
    [
        r"第[零一二三四五六七八九十百0-9]+章",
        r"第[零一二三四五六七八九十百0-9]+节",
        r"[零一二三四五六七八九十百]+[ 、]",
        r"[\(（][零一二三四五六七八九十百]+[\)）]",
        r"[\(（][0-9]{,2}[\)）]",
    ],
    [
        r"PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
        r"Chapter (I+V?|VI*|XI|IX|X)",
        r"Section [0-9]+",
        r"Article [0-9]+",
    ],
    [r"^#[^#]", r"^##[^#]", r"^###.*", r"^####.*", r"^#####.*", r"^######.*"],
]

# 每个 section 是 (文本, 布局类型)，顺序与原始文档一致。
Section = tuple[str, str]


class DemoError(RuntimeError):
    """解析、向量接口或数据库操作失败。"""


def token_count(text: str) -> int:
    """中文逐字、英文按单词、其他符号单独计数，不等同于模型真实 Token。"""
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", text))


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class _BookHTMLParser(HTMLParser):
    """保留段落边界，将 HTML 标题转换为可识别的 Markdown 层级。"""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.skip_depth += 1
        elif not self.skip_depth:
            if re.fullmatch(r"h[1-6]", tag):
                self.parts.append("\n" + "#" * int(tag[1]) + " ")
            elif tag in {"p", "div", "section", "article", "br", "li", "tr"}:
                self.parts.append("\n")
            elif tag in {"td", "th"}:
                self.parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def read_sections(path: Path | None, from_page: int = 1, to_page: int | None = None) -> tuple[str, list[Section]]:
    """按行/段落读取书籍；PDF 页码使用从 1 开始的闭区间。"""
    if from_page < 1 or (to_page is not None and to_page < from_page):
        raise DemoError("页码必须从 1 开始，且结束页不得小于起始页")
    if path is None:
        return "built-in-book.txt", [(line.strip(), "") for line in DEFAULT_SAMPLE.splitlines() if line.strip()]
    path = path.expanduser()
    if not path.is_file():
        raise DemoError(f"文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DemoError("PDF 解析需要安装 pypdf；扫描件需先进行 OCR") from exc
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise DemoError("请先解密 PDF 再运行 demo")
        if from_page > len(reader.pages):
            raise DemoError("起始页超过 PDF 总页数")
        last_page = min(to_page or len(reader.pages), len(reader.pages))
        text = "\n".join(reader.pages[index].extract_text() or "" for index in range(from_page - 1, last_page))
    elif suffix == ".docx":
        try:
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise DemoError("DOCX 解析需要安装 python-docx") from exc
        document = Document(path)
        sections: list[Section] = []
        # 按 XML 正文顺序读取，避免把表格统一移到文末。
        for element in document.element.body:
            if element.tag.endswith("}p"):
                paragraph = Paragraph(element, document)
                text = paragraph.text.strip()
                if text:
                    style = paragraph.style.name if paragraph.style else ""
                    heading = re.match(r"(?:Heading|标题)\s*([1-6])", style, re.IGNORECASE)
                    sections.append(("#" * int(heading[1]) + " " + text if heading else text, "title" if heading else ""))
            elif element.tag.endswith("}tbl"):
                for row in Table(element, document).rows:
                    text = "\t".join(cell.text.strip() for cell in row.cells).strip()
                    if text:
                        sections.append((text, ""))
        return path.name, sections
    elif suffix in {".txt", ".md", ".markdown", ".html", ".htm"}:
        text = _decode(path.read_bytes())
        if suffix in {".html", ".htm"}:
            parser = _BookHTMLParser()
            parser.feed(text)
            parser.close()
            text = "".join(parser.parts)
        breakpoint()
    else:
        raise DemoError(f"不支持 {suffix or '无扩展名'}，请使用 TXT、Markdown、HTML、PDF 或 DOCX；旧版 DOC 请先转为 DOCX")
    return path.name, [(line.strip(), "") for line in text.splitlines() if line.strip()]


def remove_contents_table(sections: list[Section]) -> list[Section]:
    """遇到目录标记，寻找首个目录项在正文中的重复标题，最多向后查 128 行。

    若找不到正文起点，只删除带点线/页码的明确目录项，避免吞掉正文。
    与原版直接删除首项相比，这里保守保留无法确认是目录的内容。
    """
    result = list(sections)
    index = 0
    while index < len(result):
        label = re.sub(r"\s+", "", re.sub(r"^#{1,6}\s*", "", result[index][0])).lower()
        if label not in {"目录", "目次", "contents", "tableofcontents", "致谢", "acknowledge"}:
            index += 1
            continue
        result.pop(index)
        if index == len(result):
            break
        first_title = re.sub(r"(?:[.．·…]{2,}\s*|\s+)\d+\s*$", "", result[index][0]).strip()
        body_start = next((j for j in range(index + 1, min(index + 129, len(result))) if result[j][0] == first_title), None)
        if body_start is not None:
            del result[index:body_start]
        else:
            while index < len(result) and re.search(r"[.．·…]{2,}\s*\d+\s*$", result[index][0]):
                result.pop(index)
            index += 1
    return result


def make_colon_as_title(sections: list[Section]) -> list[Section]:
    """把冒号结尾且较短的独立行标为标题，不复制原文、不把长正文当标题。"""
    return [(text, "title" if text.endswith((":", "：")) and len(text) < 32 else layout) for text, layout in sections]


def not_bullet(line: str) -> bool:
    """排除页码、连续小数等容易误识别为章节编号的文本。"""
    return any(re.match(pattern, line) for pattern in (r"0", r"[0-9]+ +[0-9~个只-]", r"[0-9]+\.{2,}", r"[0-9]+(\.[0-9]+){2,}[的中]"))


def bullets_category(sections: list[Section]) -> int:
    """统计每组章节编号的命中数，选择命中最多的一组；无命中返回 -1。"""
    hits = [sum(any(re.match(pattern, text.strip()) for pattern in group) and not not_bullet(text.strip()) for text, _ in sections) for group in BULLET_PATTERN]
    return max(range(len(hits)), key=hits.__getitem__) if any(hits) else -1


def _not_title(text: str) -> bool:
    if re.match(r"第[零一二三四五六七八九十百0-9]+条", text):
        return False
    return len(text.split()) > 12 or (" " not in text and len(text) >= 32) or bool(re.search(r"[,;，。；！!]", text))


def hierarchical_merge(bull: int, sections: list[Section], depth: int = 5) -> list[str]:
    """按原 hierarchical_merge 的索引回溯规则，为正文补齐祖先标题。

    从正文/低级标题向高级标题回溯；若遇到更新的高级标题，移除已经失效的
    低级标题，避免上一章的小节名串入下一章。短单段按原版 218 阈值合并。
    """
    if not sections or bull < 0:
        return []
    sections = [(text, layout) for text, layout in sections if len(text.split("@")[0].strip()) > 1 and not re.fullmatch(r"[0-9]+", text.split("@")[0].strip())]
    levels: list[list[int]] = [[] for _ in range(len(BULLET_PATTERN[bull]) + 2)]
    for index, (text, layout) in enumerate(sections):
        level = next((j for j, pattern in enumerate(BULLET_PATTERN[bull]) if re.match(pattern, text.strip())), None)
        if level is None:
            level = len(levels) - (2 if re.search(r"title|head", layout) and not _not_title(text) else 1)
        levels[level].append(index)
    levels.reverse()
    visited: set[int] = set()
    paths: list[list[int]] = []
    for offset, indices in enumerate(levels[:depth]):
        for index in indices:
            if index in visited:
                continue
            path = [index]
            visited.add(index)
            if offset + 1 != len(levels) - 1:
                for parents in levels[offset + 1 :]:
                    # bisect_left - 1 找到当前文本之前最后出现的该级标题。
                    position = bisect_left(parents, index) - 1
                    if position >= 0:
                        parent = parents[position]
                        if parent > path[-1]:
                            path.pop()
                        path.append(parent)
                visited.update(path)
            paths.append(path)

    merged: list[list[str]] = [[]]
    estimated_tokens = [0]
    for path in paths:
        texts = [sections[index][0] for index in reversed(path)]
        if len(texts) == 1:
            size = token_count(re.sub(r"@@[0-9]+.*", "", texts[0]))
            if size + estimated_tokens[-1] < 218:
                merged[-1].append(texts[0])
                estimated_tokens[-1] += size
                continue
        else:
            size = 218
        merged.append(texts)
        estimated_tokens.append(size)
    return ["\n".join(texts) for texts in merged if texts]


def naive_merge(sections: list[Section], chunk_size: int = 512, delimiter: str = DEFAULT_DELIMITER) -> list[str]:
    """无章节结构时按估算 Token 合并；chunk_size 是目标阈值而非硬上限。"""
    if chunk_size < 1:
        return ["\n".join(text for text, _ in sections)] if sections else []
    custom = sorted(set(re.findall(r"`([^`]+)`", delimiter)), key=len, reverse=True)
    if custom:
        pattern = "|".join(re.escape(item) for item in custom)
        return [part.strip() for text, _ in sections for part in re.split(pattern, text) if part.strip()]
    pattern = "|".join(re.escape(char) for char in dict.fromkeys(delimiter))
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for text, _ in sections:
        units = re.split(f"({pattern})", text) if pattern and token_count(text) >= chunk_size else [text]
        for unit in units:
            if not unit or (pattern and re.fullmatch(pattern, unit)):
                continue
            if current and size > chunk_size:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(unit)
            size += token_count(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_book(sections: list[Section], chunk_size: int = 512, delimiter: str = DEFAULT_DELIMITER) -> tuple[list[str], int, int]:
    """返回分块、章节规则编号和清理后的段落数量。"""
    cleaned = make_colon_as_title(remove_contents_table(sections))
    bull = bullets_category(cleaned)
    chunks = hierarchical_merge(bull, cleaned, 5) if bull >= 0 else naive_merge(cleaned, chunk_size, delimiter)
    return chunks, bull, len(cleaned)


class QwenEmbeddingClient:
    """直接通过标准库请求千问兼容接口，无需安装 OpenAI SDK。"""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, batch_size: int = 4, timeout: float = 60) -> None:
        if not api_key.strip() or batch_size < 1 or timeout <= 0:
            raise DemoError("API Key 不能为空，batch-size 和 timeout 必须大于 0")
        url = base_url.rstrip("/")
        self.url = url if url.endswith("/embeddings") else url + "/embeddings"
        self.api_key, self.model = api_key, model
        self.batch_size, self.timeout = batch_size, timeout

    def _request(self, texts: list[str]) -> tuple[list[list[float]], int]:
        request = Request(
            self.url,
            data=json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(4):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                items = sorted(body["data"], key=lambda item: item["index"])
                if [item["index"] for item in items] != list(range(len(texts))):
                    raise DemoError("向量接口返回的数量或 index 与输入不一致")
                vectors = [[float(value) for value in item["embedding"]] for item in items]
                if not vectors or not vectors[0] or any(len(vector) != len(vectors[0]) or not all(math.isfinite(v) for v in vector) for vector in vectors):
                    raise DemoError("向量接口返回空向量、不同维度或非有限值")
                return vectors, int((body.get("usage") or {}).get("total_tokens") or 0)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if attempt == 3 or (exc.code not in {408, 429} and exc.code < 500):
                    raise DemoError(f"向量请求失败 HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt == 3:
                    raise DemoError(f"无法连接向量接口：{exc}") from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise DemoError(f"向量响应格式错误：{exc}") from exc
            time.sleep(min(2**attempt, 8))
        raise DemoError("向量请求失败")

    def encode(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """批量编码并累计接口实际 Token 用量，校验跨批次维度一致。"""
        vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), self.batch_size):
            batch, usage = self._request(texts[start : start + self.batch_size])
            if vectors and len(batch[0]) != len(vectors[0]):
                raise DemoError("不同批次的向量维度不一致")
            vectors.extend(batch)
            total_tokens += usage
        return vectors, total_tokens


class BookChromaStore:
    """持久化外部生成的向量，不使用 Chroma 默认模型或下载任何模型。"""

    def __init__(self, path: Path, collection_name: str, model: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise DemoError("请先安装 chromadb：pip install -r requirements-book-demo.txt") from exc
        self.path = path.expanduser().resolve()
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=None, metadata={"hnsw:space": "cosine", "embedding_model": model})
        self.model = model
        if (self.collection.metadata or {}).get("embedding_model") != model:
            raise DemoError("collection 的向量模型不匹配，请换用新的 collection 名称")

    def save(self, source: str, chunks: list[str], vectors: list[list[float]]) -> list[str]:
        """同一 source 重跑时替换旧块，不影响其他文档。"""
        if not chunks or len(chunks) != len(vectors):
            raise DemoError("分块不能为空，且数量必须与向量一致")
        dimension = len(vectors[0])
        if not dimension or any(len(vector) != dimension or not all(math.isfinite(v) for v in vector) for vector in vectors):
            raise DemoError("向量必须非空、维度一致且全部为有限数值")
        existing = self.collection.get(limit=1, include=["embeddings"])
        old_vectors = existing.get("embeddings")
        if old_vectors is not None and len(old_vectors) and len(old_vectors[0]) != dimension:
            raise DemoError("collection 的向量维度不匹配，请换用新的 collection 名称")
        ids = [hashlib.sha256(f"{source}\0{index}\0{text}".encode("utf-8")).hexdigest() for index, text in enumerate(chunks, 1)]
        metadata = [{"source": source, "chunk_index": index, "estimated_tokens": token_count(text), "embedding_model": self.model, "parser": "book"} for index, text in enumerate(chunks, 1)]
        self.collection.delete(where={"source": source})
        # 大书籍按 Chroma 当前允许的批次写入，避免超过底层数据库批量上限。
        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self.collection.upsert(ids=ids[start:end], documents=chunks[start:end], embeddings=vectors[start:end], metadatas=metadata[start:end])
        return ids

    def query(self, vector: list[float], top_k: int) -> list[dict]:
        """在整个 collection 中进行余弦检索，分数为 1 - distance。"""
        count = min(max(0, top_k), self.collection.count())
        if not count:
            return []
        result = self.collection.query(query_embeddings=[vector], n_results=count, include=["documents", "metadatas", "distances"])
        return [{"document": text, "metadata": metadata, "score": 1 - float(distance)} for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("document", nargs="?", type=Path, help="书籍路径；不传则读取内置示例")
    parser.add_argument("--chunks-only", action="store_true", help="只打印分块，不调用 API 或写入 Chroma")
    parser.add_argument("--chunk-size", type=int, default=512, help="仅无章节结构时生效；0 表示整篇合并")
    parser.add_argument("--delimiter", default=DEFAULT_DELIMITER, help="无章节结构时使用的分隔符，支持反引号包裹的自定义分隔符")
    parser.add_argument("--from-page", type=int, default=1, help="PDF 起始页，从 1 开始，包含该页")
    parser.add_argument("--to-page", type=int, help="PDF 结束页，包含该页，默认到末页")
    parser.add_argument("--query", default="如何提高检索准确率？")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"), help="建议用 DASHSCOPE_API_KEY 或 QWEN_API_KEY 环境变量")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    args = parser.parse_args(argv)
    if args.chunk_size < 0 or args.top_k < 1 or args.batch_size < 1 or args.timeout <= 0:
        parser.error("chunk-size 不得小于 0；top-k、batch-size、timeout 必须大于 0")
    if args.from_page < 1 or (args.to_page is not None and args.to_page < args.from_page):
        parser.error("PDF 页码必须从 1 开始，结束页不得小于起始页")
    if (args.from_page != 1 or args.to_page is not None) and (args.document is None or args.document.suffix.lower() != ".pdf"):
        parser.error("页码参数只适用于 PDF")
    return args


def main(argv: list[str] | None = None) -> int:
    """完整入口：解析 -> 分块 -> 千问编码 -> 持久化 -> 相似度检索。"""
    args = parse_args(argv)
    try:
        name, sections = read_sections(args.document, args.from_page, args.to_page)
        chunks, category, cleaned_count = chunk_book(sections, args.chunk_size, args.delimiter)
        if not chunks:
            raise DemoError("未解析到可用正文；如果是扫描 PDF，请先 OCR 后再试")
    except Exception as exc:
        print(f"Error: 书籍解析失败：{exc}", file=sys.stderr)
        return 1
    print(f"document: {name}")
    print(f"sections: {len(sections)}, cleaned sections: {cleaned_count}, chunks: {len(chunks)}")
    print(f"merge mode: {'hierarchical_merge, bullet category=' + str(category) if category >= 0 else 'naive_merge'}")
    for index, text in enumerate(chunks, 1):
        print(f"\n--- chunk {index} ({token_count(text)} estimated tokens) ---\n{text}")
    if args.chunks_only:
        return 0
    # api_key = args.api_key
    api_key = "sk-ws-H.PPYPYHY.mejn.MEQCICtT0AUBQ5D-G7PWYTCiKuh61Bycb-SSLxq-Wpn1T6v-AiBmwk6Wr5EXpaC_PPdtohngD0uhaP15cxdpLQ4cw1mA1A"
    if not api_key and sys.stdin.isatty():
        api_key = getpass.getpass("Qwen API key: ")
    if not api_key or not api_key.strip():
        print("Error: 请设置 DASHSCOPE_API_KEY 或 QWEN_API_KEY；离线测试可用 --chunks-only", file=sys.stderr)
        return 2
    try:
        client = QwenEmbeddingClient(api_key, args.base_url, args.model, args.batch_size, args.timeout)
        # 先检查数据库模型，再请求向量；所有编码成功后才替换原文档的数据。
        store = BookChromaStore(args.chroma_path, args.collection_name, args.model)
        vectors, document_usage = client.encode(chunks)
        query_vectors, query_usage = client.encode([args.query])
        if len(query_vectors[0]) != len(vectors[0]):
            raise DemoError("文档向量与查询向量维度不一致")
        source = str(args.document.expanduser().resolve()) if args.document else name
        ids = store.save(source, chunks, vectors)
        ranked = store.query(query_vectors[0], args.top_k)
    except Exception as exc:
        print(f"Error: 向量化或 Chroma 操作失败：{exc}", file=sys.stderr)
        return 1
    print("\n=== vector retrieval ===")
    print(f"model: {args.model}, dimension: {len(query_vectors[0])}")
    print(f"provider token usage: document={document_usage}, query={query_usage}")
    print(f"chroma path: {store.path}\ncollection: {args.collection_name}")
    print(f"stored chunks: {len(ids)}, collection total: {store.collection.count()}\nquery: {args.query}")
    for rank, row in enumerate(ranked, 1):
        preview = re.sub(r"\s+", " ", row["document"])
        metadata = row["metadata"]
        print(f"{rank}. score={row['score']:.6f}, source={metadata.get('source')}, chunk={metadata.get('chunk_index')}: {preview[:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
