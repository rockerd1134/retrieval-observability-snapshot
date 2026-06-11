from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from retrieval_arena.errors import ValidationError
from retrieval_arena.manifests import compute_manifest_hash, finalize_manifest, read_manifest, validate_manifest, write_manifest


class ManifestTests(unittest.TestCase):
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

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": "retrieval_arena.manifest.v1",
            "created_at": "2026-05-25T00:00:00+00:00",
            "kind": "toy_manifest",
            "nested": {"b": 2, "a": 1},
        }

    def test_finalize_manifest_excludes_manifest_hash_field(self):
        payload = self.manifest_payload()
        payload["manifest_hash"] = "0" * 64

        finalized = finalize_manifest(payload)
        expected_hash = compute_manifest_hash({key: value for key, value in payload.items() if key != "manifest_hash"})

        self.assertEqual(finalized["manifest_hash"], expected_hash)
        self.assertNotEqual(finalized["manifest_hash"], "0" * 64)

    def test_write_manifest_uses_deterministic_formatting(self):
        with self.workspace_tempdir() as tmp:
            path = tmp / "manifest.json"
            written = write_manifest(path, self.manifest_payload())
            text = path.read_text(encoding="utf-8")

            self.assertTrue(text.endswith("\n"))
            self.assertIn('  "created_at": "2026-05-25T00:00:00+00:00",\n', text)
            self.assertEqual(read_manifest(path), written)

    def test_validate_manifest_accepts_finalized_manifest(self):
        finalized = finalize_manifest(self.manifest_payload())
        self.assertEqual(validate_manifest(finalized, verify_hash=True), finalized)

    def test_validate_manifest_rejects_missing_schema_version(self):
        manifest = finalize_manifest(self.manifest_payload())
        del manifest["schema_version"]
        manifest = finalize_manifest(manifest)

        with self.assertRaisesRegex(ValidationError, "schema_version"):
            validate_manifest(manifest)

    def test_validate_manifest_rejects_missing_manifest_hash(self):
        manifest = self.manifest_payload()

        with self.assertRaisesRegex(ValidationError, "manifest_hash"):
            validate_manifest(manifest)

    def test_validate_manifest_rejects_malformed_manifest_hash(self):
        manifest = self.manifest_payload()
        manifest["manifest_hash"] = "not-a-sha"

        with self.assertRaisesRegex(ValidationError, "64-character"):
            validate_manifest(manifest)

    def test_validate_manifest_rejects_hash_mismatch_when_requested(self):
        manifest = finalize_manifest(self.manifest_payload())
        manifest["kind"] = "changed"

        with self.assertRaisesRegex(ValidationError, "does not match"):
            validate_manifest(manifest, verify_hash=True)


if __name__ == "__main__":
    unittest.main()
