from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from retrieval_arena.hashing import directory_inventory, sha256_directory, sha256_file, sha256_json, sha256_jsonl


class HashingTests(unittest.TestCase):
    @contextmanager
    def workspace_tempdir(self) -> Iterator[Path]:
        root = Path(".tmp_unit_tests")
        root.mkdir(exist_ok=True)
        path = root / f"tmp_{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_sha256_file_is_stable_for_same_bytes(self):
        with self.workspace_tempdir() as tmp:
            first = tmp / "first.txt"
            second = tmp / "second.txt"
            first.write_text("same bytes\n", encoding="utf-8")
            second.write_text("same bytes\n", encoding="utf-8")

            self.assertEqual(sha256_file(first), sha256_file(second))

            second.write_text("changed bytes\n", encoding="utf-8")
            self.assertNotEqual(sha256_file(first), sha256_file(second))

    def test_directory_hash_uses_sorted_relative_posix_paths(self):
        with self.workspace_tempdir() as tmp:
            (tmp / "b").mkdir()
            (tmp / "a").mkdir()
            (tmp / "b" / "two.txt").write_text("two\n", encoding="utf-8")
            (tmp / "a" / "one.txt").write_text("one\n", encoding="utf-8")

            inventory = directory_inventory(tmp)
            self.assertEqual([item["path"] for item in inventory], ["a/one.txt", "b/two.txt"])
            first_hash = sha256_directory(tmp)
            second_hash = sha256_directory(tmp)
            self.assertEqual(first_hash, second_hash)

    def test_json_hash_is_independent_of_key_order(self):
        self.assertEqual(
            sha256_json({"b": 2, "a": {"z": 3, "y": 4}}),
            sha256_json({"a": {"y": 4, "z": 3}, "b": 2}),
        )

    def test_jsonl_hash_canonicalizes_object_key_order(self):
        with self.workspace_tempdir() as tmp:
            first = tmp / "first.jsonl"
            second = tmp / "second.jsonl"
            first.write_text('{"b":2,"a":1}\n{"x":"y"}\n', encoding="utf-8")
            second.write_text('{"a":1,"b":2}\n\n{"x":"y"}\n', encoding="utf-8")

            self.assertEqual(sha256_jsonl(first), sha256_jsonl(second))


if __name__ == "__main__":
    unittest.main()
