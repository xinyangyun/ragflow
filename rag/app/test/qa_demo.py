#!/usr/bin/env python3
"""RAGFlow QA 解析器、千问向量化和 Chroma 检索的独立运行示例。

QA 模式与 naive 模式不同：它不按照 chunk_size 合并长文本，而是先从文档
中识别问题和答案，每一组 Q&A 直接作为一个检索 chunk。

支持格式：
    CSV/TXT   每条记录两列，第一列是问题，第二列是答案
    XLS/XLSX  每行前两个非空单元格分别作为问题和答案
    Markdown  标题作为问题，标题下正文作为答案，保留标题层级
    DOCX      Heading 样式作为问题，标题下正文作为答案
    PDF       文本型 PDF，识别 Q:/A:、问题:/回答: 和编号问题

运行示例：
    python qa_demo.py qa.csv --pairs-only
    export DASHSCOPE_API_KEY='your-api-key'
    python qa_demo.py qa.xlsx --query '如何重置密码？'
    python qa_demo.py faq.md --chroma-path ./qa_chroma_db
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = (
    "https://ws-leyp0n1kyjsllxss.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_MODEL = "text-embedding-v2"
DEFAULT_DOCUMENT = Path(__file__).resolve().parent / "qa_test_20.md"
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent / "qa_chroma_db"
DEFAULT_COLLECTION_NAME = "qa-demo"
DEFAULT_BATCH_SIZE = 4
DEFAULT_TIMEOUT = 60.0

DEFAULT_PAIRS = [
    ("如何提高知识库检索准确率？", "可以优化问答质量、向量模型、召回数量和重排序策略。"),
    ("忘记密码后怎么办？", "在登录页面点击忘记密码，并通过绑定邮箱完成重置。"),
    ("QA 解析器如何切分文档？", "每一组问题和答案直接作为一个独立的检索块。"),
]


class DemoError(RuntimeError):
    """文档解析、向量调用或 Chroma 操作失败时抛出的统一异常。"""


@dataclass(frozen=True)
class QAPair:
    """从文档中解析出的一组问题和答案。"""

    question: str
    answer: str
    position: int

    def document(self, english: bool = False) -> str:
        """转换成与 qa.py content_with_weight 相近的可检索文本。"""
        question_prefix = "Question: " if english else "问题："
        answer_prefix = "Answer: " if english else "回答："
        return (
            f"{question_prefix}{remove_qa_prefix(self.question)}\n"
            f"{answer_prefix}{remove_qa_prefix(self.answer)}"
        )


def remove_qa_prefix(text: str) -> str:
    """移除问题、答案、Q、A 等常见前缀，与 qa.py 的 rmPrefix 对齐。"""
    return re.sub(
        r"^(问题|答案|回答|user|assistant|Q|A|Question|Answer|问|答)[\t:： ]+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def _decode(data: bytes) -> str:
    """按照常见中英文编码顺序解码文本文件。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _valid_pair(question: str, answer: str) -> bool:
    return bool(question.strip() and answer.strip())


def parse_txt(path: Path) -> list[QAPair]:
    """按 qa.py 规则解析 TXT，并允许答案占用后续多行。"""
    lines = _decode(path.read_bytes()).splitlines()
    comma_count = sum(1 for line in lines if len(line.split(",")) == 2)
    tab_count = sum(1 for line in lines if len(line.split("\t")) == 2)
    delimiter = "\t" if tab_count >= comma_count else ","

    pairs: list[QAPair] = []
    question = ""
    answer_lines: list[str] = []
    start_line = 0

    def flush() -> None:
        answer = "\n".join(answer_lines).strip()
        if _valid_pair(question, answer):
            pairs.append(QAPair(question.strip(), answer, start_line))

    for line_number, line in enumerate(lines, start=1):
        columns = line.split(delimiter)
        if len(columns) == 2:
            flush()
            question = columns[0].strip()
            answer_lines = [columns[1].strip()]
            start_line = line_number
        elif question:
            # 无法拆成两列的行视为上一条答案的续行。
            answer_lines.append(line)
    flush()
    return pairs


def parse_csv(path: Path) -> list[QAPair]:
    """解析 CSV/TSV；严格要求每条有效记录恰好包含两列。"""
    text = _decode(path.read_bytes())
    delimiter = "\t" if any("\t" in line for line in text.splitlines()) else ","
    pairs: list[QAPair] = []
    for row_number, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter), start=1):
        if len(row) != 2 or not _valid_pair(row[0], row[1]):
            continue
        pairs.append(QAPair(row[0].strip(), row[1].strip(), row_number))
    return pairs


def parse_excel(path: Path) -> list[QAPair]:
    """将每行前两个非空单元格解析成 Q&A，支持多个工作表。"""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise DemoError(
            "解析 Excel 需要 openpyxl，请执行：pip install openpyxl"
        ) from exc

    workbook = load_workbook(path, read_only=True, data_only=True)
    pairs: list[QAPair] = []
    global_position = 0
    try:
        for worksheet in workbook.worksheets:
            for row in worksheet.iter_rows(values_only=True):
                global_position += 1
                values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                if len(values) >= 2:
                    pairs.append(QAPair(values[0], values[1], global_position))
    finally:
        workbook.close()
    return pairs


def _flush_heading_answer(
    pairs: list[QAPair],
    question_stack: list[str],
    answer_lines: list[str],
    position: int,
) -> None:
    """将标题路径和其下正文整理成一组 Q&A。"""
    question = "\n".join(item for item in question_stack if item.strip()).strip()
    answer = "\n".join(answer_lines).strip()
    if _valid_pair(question, answer):
        pairs.append(QAPair(question, answer, position))


def parse_markdown(path: Path) -> list[QAPair]:
    """按 qa.py 的 Markdown 规则，用标题路径作为问题、标题下正文作为答案。"""
    lines = _decode(path.read_bytes()).splitlines()
    pairs: list[QAPair] = []
    question_stack: list[str] = []
    level_stack: list[int] = []
    answer_lines: list[str] = []
    answer_position = 0
    in_code_block = False

    for line_number, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        # 与 qa.py 的 mdQuestionLevel 一致：统计行首 # 数量并去掉标题标记。
        heading_match = re.match(r"#*", line) if not in_code_block else None
        level = len(heading_match.group(0)) if heading_match else 0
        question = line.lstrip("#").lstrip() if level else ""
        if not level or level > 6:
            if question_stack:
                answer_lines.append(line)
            continue

        _flush_heading_answer(pairs, question_stack, answer_lines, answer_position)
        answer_lines = []
        answer_position = line_number

        while level_stack and level <= level_stack[-1]:
            level_stack.pop()
            question_stack.pop()
        level_stack.append(level)
        question_stack.append(question)

    _flush_heading_answer(pairs, question_stack, answer_lines, answer_position)
    return pairs


def _docx_heading_level(paragraph) -> int:
    """读取 DOCX Heading 样式中的标题级别。"""
    style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
    if not style_name.startswith("Heading"):
        return 0
    match = re.search(r"\d+", style_name)
    return int(match.group()) if match else 1


def parse_docx(path: Path) -> list[QAPair]:
    """按照 DOCX 标题层级生成问题路径，并把标题下段落作为答案。"""
    try:
        from docx import Document
    except ImportError as exc:
        raise DemoError(
            "解析 DOCX 需要 python-docx，请执行：pip install python-docx"
        ) from exc

    document = Document(path)
    pairs: list[QAPair] = []
    question_stack: list[str] = []
    level_stack: list[int] = []
    answer_lines: list[str] = []
    answer_position = 0

    for position, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.replace("\u3000", " ").strip()
        if not text:
            continue
        level = _docx_heading_level(paragraph)
        if not level:
            if question_stack:
                answer_lines.append(text)
            continue

        _flush_heading_answer(pairs, question_stack, answer_lines, answer_position)
        answer_lines = []
        answer_position = position
        while level_stack and level <= level_stack[-1]:
            level_stack.pop()
            question_stack.pop()
        level_stack.append(level)
        question_stack.append(text)

    _flush_heading_answer(pairs, question_stack, answer_lines, answer_position)

    # 两列表格在独立 demo 中也按 Q&A 读取，便于处理常见 FAQ DOCX。
    for table in document.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if len(values) >= 2:
                pairs.append(QAPair(values[0], values[1], len(pairs) + 1))
    return pairs


QUESTION_PREFIX = re.compile(
    r"^(?:Q(?:uestion)?|问题|问)\s*[:：]\s*(.+)$", re.IGNORECASE
)
ANSWER_PREFIX = re.compile(
    r"^(?:A(?:nswer)?|答案|回答|答)\s*[:：]\s*(.*)$", re.IGNORECASE
)
NUMBERED_QUESTION = re.compile(
    r"^(?:第[零一二三四五六七八九十百0-9]+问|[（(]?[0-9]{1,3}[）).、])\s*(.+)$"
)


def parse_question_answer_lines(lines: Iterable[str]) -> list[QAPair]:
    """从普通文本行中识别显式 Q/A 前缀或带问号的编号问题。"""
    pairs: list[QAPair] = []
    question = ""
    answer_lines: list[str] = []
    start_line = 0

    def flush() -> None:
        answer = "\n".join(answer_lines).strip()
        if _valid_pair(question, answer):
            pairs.append(QAPair(question.strip(), answer, start_line))

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            if question and answer_lines:
                answer_lines.append("")
            continue

        question_match = QUESTION_PREFIX.match(line)
        numbered_match = NUMBERED_QUESTION.match(line)
        if question_match or (
            numbered_match and numbered_match.group(1).rstrip().endswith(("?", "？"))
        ):
            flush()
            question = (
                question_match.group(1) if question_match else numbered_match.group(1)
            ).strip()
            answer_lines = []
            start_line = line_number
            continue

        answer_match = ANSWER_PREFIX.match(line)
        if answer_match and question:
            answer_lines.append(answer_match.group(1))
        elif question:
            answer_lines.append(line)

    flush()
    return pairs


def parse_pdf(path: Path) -> list[QAPair]:
    """从文本型 PDF 提取文本，再识别 Q&A；扫描件需要先进行 OCR。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DemoError("解析 PDF 需要 pypdf，请执行：pip install pypdf") from exc

    lines: list[str] = []
    for page in PdfReader(path).pages:
        lines.extend((page.extract_text() or "").splitlines())
    return parse_question_answer_lines(lines)


def read_qa_pairs(path: Path | None) -> tuple[str, str, list[QAPair]]:
    """根据文件扩展名选择对应解析器并返回 Q&A 列表。"""
    if path is None:
        return (
            "built-in-sample.csv",
            "sample",
            [QAPair(question, answer, index) for index, (question, answer) in enumerate(DEFAULT_PAIRS, start=1)],
        )
    if not path.is_file():
        raise DemoError(f"文档不存在：{path}")

    suffix = path.suffix.lower()
    parsers = {
        ".txt": parse_txt,
        ".csv": parse_csv,
        ".xls": parse_excel,
        ".xlsx": parse_excel,
        ".md": parse_markdown,
        ".markdown": parse_markdown,
        ".mdx": parse_markdown,
        ".docx": parse_docx,
        ".pdf": parse_pdf,
    }
    parser = parsers.get(suffix)
    if parser is None:
        raise DemoError("仅支持 TXT、CSV、XLS/XLSX、Markdown、DOCX 和 PDF")
    return path.name, suffix.lstrip("."), parser(path)


def embeddings_url(base_url: str) -> str:
    """补齐 OpenAI 兼容接口的 /embeddings 路径。"""
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/embeddings") else f"{normalized}/embeddings"


class QwenEmbeddingClient:
    """面向千问 text-embedding-v2 的轻量 OpenAI 兼容客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        batch_size: int = 4,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key.strip():
            raise ValueError("API key 不能为空")
        self.api_key = api_key
        self.url = embeddings_url(base_url)
        self.model = model
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    def _request(self, texts: list[str]) -> tuple[list[list[float]], int]:
        payload = json.dumps(
            {"model": self.model, "input": texts, "encoding_format": "float"}
        ).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retryable = False
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                items = body.get("data")
                if not isinstance(items, list):
                    raise DemoError(f"向量响应格式不正确：{str(body)[:500]}")
                items.sort(key=lambda item: item.get("index", 0))
                vectors = [[float(value) for value in item["embedding"]] for item in items]
                if len(vectors) != len(texts):
                    raise DemoError(
                        f"服务返回 {len(vectors)} 个向量，但输入有 {len(texts)} 条"
                    )
                usage = body.get("usage") or {}
                return vectors, int(usage.get("total_tokens") or 0)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = DemoError(f"HTTP {exc.code}: {detail[:500]}")
                retryable = exc.code in {408, 429} or exc.code >= 500
            except (URLError, TimeoutError) as exc:
                last_error = DemoError(str(exc))
                retryable = True
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, DemoError) as exc:
                last_error = exc if isinstance(exc, DemoError) else DemoError(str(exc))

            if not retryable or attempt == self.max_retries:
                break
            time.sleep(min(2**attempt, 8))
        raise DemoError(f"向量请求失败：{last_error}") from last_error

    def encode(self, texts: Iterable[str]) -> tuple[list[list[float]], int]:
        """批量编码文本并累计服务端 Token 用量。"""
        values = list(texts)
        vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(values), self.batch_size):
            batch_vectors, batch_tokens = self._request(
                values[start : start + self.batch_size]
            )
            vectors.extend(batch_vectors)
            total_tokens += batch_tokens
        return vectors, total_tokens

    def encode_query(self, text: str) -> tuple[list[float], int]:
        vectors, total_tokens = self.encode([text])
        return vectors[0], total_tokens


class QAChromaStore:
    """将问答文本、向量和问答元数据持久化到 Chroma。"""

    def __init__(self, path: Path, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise DemoError(
                "缺少 chromadb，请执行：pip install -r "
                "rag/app/test/requirements-qa-demo.txt"
            ) from exc

        self.path = path.expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _record_id(source: str, pair: QAPair) -> str:
        value = (
            f"{source}\0{pair.position}\0{pair.question}\0{pair.answer}"
        ).encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def save(
        self,
        source: str,
        source_format: str,
        pairs: list[QAPair],
        documents: list[str],
        vectors: list[list[float]],
        model: str,
    ) -> list[str]:
        """替换同一来源的旧记录，并写入最新 Q&A 数据。"""
        if not (len(pairs) == len(documents) == len(vectors)):
            raise DemoError("Q&A、文档文本和向量数量不一致")

        self.collection.delete(where={"source": source})
        ids = [self._record_id(source, pair) for pair in pairs]
        metadatas = [
            {
                "source": source,
                "source_format": source_format,
                "position": pair.position,
                "question": pair.question,
                "answer": pair.answer,
                "embedding_model": model,
            }
            for pair in pairs
        ]
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )
        return ids

    def query(self, query_vector: list[float], top_k: int) -> list[dict]:
        """使用查询向量检索 collection 中最相近的问答。"""
        result_count = min(max(0, top_k), self.collection.count())
        if result_count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {
                "id": record_id,
                "document": document or "",
                "metadata": metadata or {},
                "distance": float(distance),
                "score": 1.0 - float(distance),
            }
            for record_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "document",
        nargs="?",
        type=Path,
        default=DEFAULT_DOCUMENT,
        help=f"待解析的 Markdown 文件，默认：{DEFAULT_DOCUMENT.name}",
    )
    parser.add_argument("--query", default="如何提高知识库检索准确率？")
    parser.add_argument(
        "--pairs-only",
        action="store_true",
        help="只解析并显示问答，不调用向量接口，也不写入 Chroma",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"),
        help="默认读取 DASHSCOPE_API_KEY 或 QWEN_API_KEY",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document_name, source_format, pairs = read_qa_pairs(args.document)
    except (DemoError, OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"document: {document_name}")
    print(f"format: {source_format}, qa pairs: {len(pairs)}")
    for index, pair in enumerate(pairs, start=1):
        print(f"\n--- Q&A {index}, position={pair.position} ---")
        print(f"Q: {pair.question}")
        print(f"A: {pair.answer}")

    if args.pairs_only:
        return 0
    if not pairs:
        print("Error: 没有识别到有效的问答对", file=sys.stderr)
        return 1

    api_key = args.api_key
    if not api_key and sys.stdin.isatty():
        api_key = getpass.getpass("Qwen API key: ")
    if not api_key:
        print(
            "Error: 请设置 DASHSCOPE_API_KEY、传入 --api-key，或使用 --pairs-only",
            file=sys.stderr,
        )
        return 2

    documents = [pair.document() for pair in pairs]
    client = QwenEmbeddingClient(
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
        model=DEFAULT_MODEL,
        batch_size=DEFAULT_BATCH_SIZE,
        timeout=DEFAULT_TIMEOUT,
    )

    try:
        vectors, document_tokens = client.encode(documents)
        query_vector, query_tokens = client.encode_query(args.query)
        source = str(args.document.expanduser().resolve())
        store = QAChromaStore(DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME)
        stored_ids = store.save(
            source,
            source_format,
            pairs,
            documents,
            vectors,
            DEFAULT_MODEL,
        )
        ranked = store.query(query_vector, top_k=3)
    except (DemoError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\n=== Chroma QA retrieval ===")
    print(f"model: {DEFAULT_MODEL}")
    print(f"endpoint: {client.url}")
    print(f"embedding dimension: {len(query_vector)}")
    print(f"provider token usage: document={document_tokens}, query={query_tokens}")
    print(f"chroma path: {store.path}")
    print(f"collection: {store.collection_name}")
    print(f"stored pairs: {len(stored_ids)}, collection total: {store.collection.count()}")
    print(f"query: {args.query}")
    for rank, row in enumerate(ranked, start=1):
        metadata = row["metadata"]
        print(f"\n{rank}. score={row['score']:.6f}")
        print(f"   Q: {metadata.get('question', '')}")
        print(f"   A: {metadata.get('answer', '')}")
        print(f"   source: {metadata.get('source', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
