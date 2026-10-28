"""table_demo 的离线测试，不调用真实向量接口。"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import table_demo as demo


class TableParserTests(unittest.TestCase):
    def test_default_csv_one_row_one_chunk(self):
        table = demo.read_table()
        self.assertEqual(len(table.records), 8)
        self.assertEqual(len([demo.record_document(record) for record in table.records]), 8)
        self.assertIn("- 商品名称: 无线键盘", demo.record_document(table.records[0]))

    def test_column_types_and_leading_zero_text(self):
        table = demo.read_table()
        self.assertEqual(table.column_types["价格"], "float")
        self.assertEqual(table.column_types["库存"], "int")
        self.assertEqual(table.column_types["是否上架"], "bool")
        self.assertEqual(table.column_types["更新时间"], "datetime")
        self.assertEqual(table.column_types["row_id"], "text")

    def test_csv_quoted_comma_is_preserved(self):
        table = demo.read_table()
        self.assertEqual(table.records[0].values["描述"], "静音按键，适合日常办公")
        self.assertIn("蓝牙、2.4G", table.records[1].values["描述"])

    def test_tsv_and_malformed_row(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "table.tsv"
            path.write_text("名称\t库存\n键盘\t3\n错误行\n鼠标\t2\n", encoding="utf-8")
            table = demo.read_table(path)
            self.assertEqual(len(table.records), 2)
            self.assertEqual(table.failed_rows, [3])

    def test_duplicate_headers_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.csv"
            path.write_text("名称,名称\n键盘,鼠标\n", encoding="utf-8")
            with self.assertRaises(demo.DemoError):
                demo.read_table(path)

    def test_column_roles(self):
        record = demo.read_table().records[0]
        with patch.dict(demo.COLUMN_ROLES, {"价格": "metadata", "描述": "indexing"}, clear=True):
            document = demo.record_document(record)
            metadata = demo.record_metadata(record, "test.csv")
        self.assertNotIn("- 价格:", document)
        self.assertIn("价格", metadata)
        self.assertIn("- 描述:", document)
        self.assertNotIn("描述", metadata)

    def test_rows_only_main(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = demo.main(["--rows-only"])
        self.assertEqual(status, 0)
        self.assertIn("rows: 8, chunks: 8", output.getvalue())


class EmbeddingTests(unittest.TestCase):
    def test_response_is_sorted_by_index(self):
        body = {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}], "usage": {"total_tokens": 5}}
        response = io.BytesIO(json.dumps(body).encode())
        with patch.object(demo, "urlopen", return_value=response):
            vectors, usage = demo.QwenEmbeddingClient("test-key").encode(["甲", "乙"])
        self.assertEqual(vectors, [[1, 0], [0, 1]])
        self.assertEqual(usage, 5)


class ChromaTests(unittest.TestCase):
    def test_save_replace_and_query(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            self.skipTest("未安装 chromadb")
        table = demo.read_table()
        records = table.records[:2]
        documents = [demo.record_document(record) for record in records]
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.object(demo, "DEFAULT_CHROMA_PATH", Path(folder) / "db"),
                patch.object(demo, "DEFAULT_COLLECTION_NAME", "table-unit-test"),
            ):
                store = demo.TableChromaStore()
                store.save("test.csv", records, documents, [[1, 0], [0, 1]])
                self.assertEqual(store.collection.count(), 2)
                store.save("test.csv", records[:1], documents[:1], [[1, 0]])
                self.assertEqual(store.collection.count(), 1)
                result = store.query([1, 0])
                self.assertEqual(result[0]["metadata"]["商品名称"], "无线键盘")
                self.assertTrue((Path(folder) / "db" / "chroma.sqlite3").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
