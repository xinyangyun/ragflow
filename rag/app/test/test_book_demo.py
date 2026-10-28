"""book_demo 的离线测试；从本目录直接运行，不加载 RAGFlow 服务配置。"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import book_demo as demo


class BookChunkTests(unittest.TestCase):
    def test_chinese_chapter_paths(self):
        sections = [
            (text, "")
            for text in (
                "第一章 基础",
                "第一节 数据",
                "这是第一章第一节的正文。",
                "第二节 模型",
                "这是第一章第二节的正文。",
                "第二章 实践",
                "这里没有小节，不应串入上一章第二节。",
            )
        ]
        chunks, category, _ = demo.chunk_book(sections)
        self.assertGreaterEqual(category, 0)
        self.assertIn("第一章 基础\n第一节 数据\n这是第一章第一节的正文。", chunks)
        last = next(chunk for chunk in chunks if "这里没有小节" in chunk)
        self.assertTrue(last.startswith("第二章 实践\n"))
        self.assertNotIn("第二节 模型", last)

    def test_matches_upstream_hierarchy(self):
        # 仅加载上游纯函数 AST，避免导入 rag.nlp 引发模型和服务初始化。
        upstream = Path(__file__).resolve().parents[2] / "nlp" / "__init__.py"
        tree = ast.parse(upstream.read_text(encoding="utf-8"))
        names = {"hierarchical_merge", "not_title"}
        functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        scope = {"re": demo.re, "logging": logging, "BULLET_PATTERN": demo.BULLET_PATTERN, "num_tokens_from_string": demo.token_count}
        exec(compile(functions, str(upstream), "exec"), scope)
        default_source = demo.DEFAULT_DOCUMENT.read_text(encoding="utf-8")
        for source in (default_source, "# 手册\n## 基础\n正文内容。\n## 实践\n实践内容。", "开篇说明\n第一章 基础\n（一）定义\n术语正文\n第二章 实践\n更多正文"):
            sections = [(line, "") for line in source.splitlines() if line]
            category = demo.bullets_category(sections)
            expected = ["\n".join(chunk) for chunk in scope["hierarchical_merge"](category, sections, 5) if chunk]
            self.assertEqual(demo.hierarchical_merge(category, sections), expected)

    def test_hierarchy_ignores_chunk_size(self):
        _, sections = demo.read_sections(None)
        self.assertEqual(demo.chunk_book(sections, 1)[0], demo.chunk_book(sections, 10000)[0])

    def test_no_headings_uses_naive_merge(self):
        sections = [("这是没有编号的普通段落。" * 10, ""), ("第二段普通正文。" * 10, "")]
        chunks, category, _ = demo.chunk_book(sections, 20)
        self.assertEqual(category, -1)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(demo.chunk_book(sections, 10000)[0]), 1)

    def test_contents_removed_without_mutating_input(self):
        source = [(line, "") for line in ("目录", "第一章 基础 .... 1", "第二章 实践 .... 3", "第一章 基础", "正文内容")]
        self.assertEqual(demo.remove_contents_table(source), source[3:])
        self.assertEqual(len(source), 5)
        self.assertEqual(demo.remove_contents_table([("目录", "")]), [])
        self.assertEqual(demo.remove_contents_table([("目录", ""), ("保留这段不能确认是目录的文字", "")]), [("保留这段不能确认是目录的文字", "")])

    def test_colon_title_is_not_duplicated(self):
        source = [("注意事项：", ""), ("第一条提醒内容。", "")]
        result = demo.make_colon_as_title(source)
        self.assertEqual(len(result), len(source))
        self.assertEqual(result[0], ("注意事项：", "title"))

    def test_markdown_heading_hierarchy(self):
        chunks, category, _ = demo.chunk_book([("# 手册", ""), ("## 小节", ""), ("小节正文内容。", "")])
        self.assertEqual(category, 4)
        self.assertEqual(chunks, ["# 手册\n## 小节\n小节正文内容。"])

    def test_html_and_text_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            html = Path(folder) / "book.html"
            html.write_text("<h1>手册</h1><h2>检索</h2><p>测试正文</p><script>ignore()</script>", encoding="utf-8")
            _, sections = demo.read_sections(html)
            self.assertEqual(sections, [("# 手册", ""), ("## 检索", ""), ("测试正文", "")])
            text = Path(folder) / "book.txt"
            text.write_bytes("第一章 基础\n正文测试".encode("gb18030"))
            self.assertEqual(demo.read_sections(text)[1][0][0], "第一章 基础")

    def test_invalid_page_range(self):
        with self.assertRaises(demo.DemoError):
            demo.read_sections(None, from_page=0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            demo.parse_args(["--to-page", "3", "--chunks-only"])

    def test_empty_input_is_error(self):
        with tempfile.TemporaryDirectory() as folder:
            document = Path(folder) / "empty.txt"
            document.write_text("\n\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(demo.main([str(document), "--chunks-only"]), 1)

    def test_standalone_copy_runs_without_other_demos(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "book_demo.py"
            shutil.copyfile(demo.__file__, script)
            shutil.copyfile(demo.DEFAULT_DOCUMENT, Path(folder) / demo.DEFAULT_DOCUMENT.name)
            result = subprocess.run([sys.executable, "-I", str(script), "--chunks-only"], cwd=folder, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("hierarchical_merge", result.stdout)
            self.assertFalse((Path(folder) / "book_chroma_db").exists())


class EmbeddingTests(unittest.TestCase):
    @staticmethod
    def response(items):
        return io.BytesIO(json.dumps({"data": items, "usage": {"total_tokens": 7}}).encode())

    def test_response_order_and_payload(self):
        items = [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]
        with patch.object(demo, "urlopen", return_value=self.response(items)) as request:
            vectors, usage = demo.QwenEmbeddingClient("test-key").encode(["甲", "乙"])
        self.assertEqual(vectors, [[1, 0], [0, 1]])
        self.assertEqual(usage, 7)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["model"], "text-embedding-v2")
        self.assertEqual(payload["input"], ["甲", "乙"])

    def test_invalid_embedding_response(self):
        for items in ([], [{"index": 1, "embedding": [1]}], [{"index": 0, "embedding": []}], [{"index": 0, "embedding": [float("nan")]}]):
            with self.subTest(items=items), patch.object(demo, "urlopen", return_value=self.response(items)), self.assertRaises(demo.DemoError):
                demo.QwenEmbeddingClient("test-key").encode(["甲"])


class ChromaTests(unittest.TestCase):
    def test_persistence_replacement_and_query(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            self.skipTest("未安装 chromadb")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "db"
            store = demo.BookChromaStore(path, "book-unit-test", demo.DEFAULT_MODEL)
            store.save("book-a", ["甲", "乙"], [[1, 0], [0, 1]])
            store.save("book-b", ["丙"], [[0.5, 0.5]])
            store.save("book-a", ["甲新版"], [[1, 0]])
            self.assertEqual(store.collection.count(), 2)
            reopened = demo.BookChromaStore(path, "book-unit-test", demo.DEFAULT_MODEL)
            self.assertEqual(reopened.query([1, 0], 1)[0]["document"], "甲新版")
            self.assertTrue((path / "chroma.sqlite3").is_file())
            with self.assertRaises(demo.DemoError):
                reopened.save("book-a", ["维度错误"], [[1, 0, 0]])
            self.assertEqual(reopened.collection.count(), 2)
            with self.assertRaises(demo.DemoError):
                demo.BookChromaStore(path, "book-unit-test", "different-model")

    def test_main_with_mock_embeddings_and_real_chroma(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            self.skipTest("未安装 chromadb")
        with tempfile.TemporaryDirectory() as folder:

            def encode(client, texts):
                return [[1.0, index / 10 + 0.1] for index, _ in enumerate(texts)], len(texts)

            with patch.object(demo.QwenEmbeddingClient, "encode", encode), contextlib.redirect_stdout(io.StringIO()) as output:
                status = demo.main(["--api-key", "test-key", "--chroma-path", folder])
            self.assertEqual(status, 0)
            self.assertIn("=== vector retrieval ===", output.getvalue())
            self.assertIn("stored chunks: 13", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
