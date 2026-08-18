"""Tests for document parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from kgcontext import Document, UnsupportedFormatError, parse_file


class TestParseFile:
    def test_parse_txt(self, tmp_txt: Path) -> None:
        doc = parse_file(str(tmp_txt))
        assert isinstance(doc, Document)
        assert "concurrent users" in doc.content
        assert doc.doc_type == "txt"
        assert doc.source_path == str(tmp_txt.resolve())
        assert doc.metadata["filename"] == "test.txt"

    def test_parse_md(self, tmp_md: Path) -> None:
        doc = parse_file(str(tmp_md))
        assert isinstance(doc, Document)
        assert "Requirements" in doc.content
        assert "JSON" in doc.content
        assert doc.doc_type == "md"

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            parse_file(str(tmp_path / "nonexistent.txt"))

    def test_parse_unsupported_format(self, tmp_path: Path) -> None:
        p = tmp_path / "test.xyz"
        p.write_text("data")
        with pytest.raises(UnsupportedFormatError, match="Unsupported format"):
            parse_file(str(p))

    def test_parse_override_doc_type(self, tmp_path: Path) -> None:
        p = tmp_path / "test.xyz"
        p.write_text("plain text content")
        doc = parse_file(str(p), doc_type="txt")
        assert doc.doc_type == "txt"
        assert doc.content == "plain text content"

    def test_parse_pdf(self, tmp_path: Path) -> None:
        """Create a minimal PDF and parse it."""
        import pymupdf

        p = tmp_path / "test.pdf"
        doc_writer = pymupdf.open()  # new empty document
        page = doc_writer.new_page()
        page.insert_text((50, 72), "PDF test content for parsing")
        doc_writer.save(str(p))
        doc_writer.close()

        doc = parse_file(str(p))
        assert isinstance(doc, Document)
        assert "PDF test content" in doc.content
        assert doc.doc_type == "pdf"