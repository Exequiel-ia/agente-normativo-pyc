import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion_assistant import _curated_page_numbers
from rag_pipeline import build_knowledge_base, save_metadata


class CurationTests(unittest.TestCase):
    def test_page_selection_and_exclusion(self):
        sections = [
            {"pages": "1-3", "include": True, "classification": "Conocimiento principal"},
            {"pages": "4-5", "include": False, "classification": "Excluir"},
            {"pages": "6", "include": True, "classification": "Anexo técnico"},
        ]
        self.assertEqual(_curated_page_numbers(sections), {1, 2, 3, 6})

    def test_index_only_contains_approved_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ejemplo.pdf").write_bytes(b"test-double")
            metadata_path = root / "knowledge_metadata.json"
            save_metadata(metadata_path, {"ejemplo.pdf": {"indexed_pages": [1]}})
            pages = [
                MagicMock(extract_text=lambda: "Contenido permitido de normativa " * 8),
                MagicMock(extract_text=lambda: "Contenido excluido de normativa " * 8),
            ]
            reader = MagicMock()
            reader.pages = pages
            with patch("rag_pipeline.PdfReader", return_value=reader):
                kb = build_knowledge_base(root, metadata_path)
            self.assertTrue(kb.chunks)
            self.assertTrue(all(chunk.page == 1 for chunk in kb.chunks))
            self.assertTrue(all("excluido" not in chunk.text for chunk in kb.chunks))


if __name__ == "__main__":
    unittest.main()
