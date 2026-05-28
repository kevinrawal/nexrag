from unittest.mock import MagicMock, patch

import pytest

from nexrag.exceptions import LoaderError
from nexrag.loaders.pdf import PDFLoader


def _make_mock_reader(pages_text: list[str], is_encrypted: bool = False):
    """Build a minimal PdfReader mock."""
    pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    reader = MagicMock()
    reader.is_encrypted = is_encrypted
    reader.pages = pages
    return reader


class TestPDFLoader:
    def test_load_path_returns_one_document(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_reader = _make_mock_reader(["Page one text.", "Page two text."])

        with patch("nexrag.loaders.pdf.PDFLoader._open", return_value=(mock_reader, str(pdf_file))):
            loader = PDFLoader()
            docs = loader._extract(mock_reader, str(pdf_file))

        assert len(docs) == 1
        assert "Page one text." in docs[0].content
        assert "Page two text." in docs[0].content

    def test_metadata_source_is_path(self, tmp_path):
        pdf_file = tmp_path / "resume.pdf"
        pdf_file.write_bytes(b"%PDF fake")
        mock_reader = _make_mock_reader(["Resume content."])

        loader = PDFLoader()
        source = str(pdf_file.resolve())
        docs = loader._extract(mock_reader, source)

        assert docs[0].metadata["source"] == source

    def test_metadata_page_count(self):
        mock_reader = _make_mock_reader(["p1", "p2", "p3"])
        loader = PDFLoader()
        docs = loader._extract(mock_reader, "test.pdf")
        assert docs[0].metadata["page_count"] == 3

    def test_empty_pdf_raises_loader_error(self):
        mock_reader = _make_mock_reader(["", "   "])
        loader = PDFLoader()
        with pytest.raises(LoaderError, match="no extractable text"):
            loader._extract(mock_reader, "empty.pdf")

    def test_encrypted_pdf_raises_loader_error(self, tmp_path):
        pdf_file = tmp_path / "enc.pdf"
        pdf_file.write_bytes(b"%PDF fake encrypted")

        mock_reader = _make_mock_reader([])
        mock_reader.is_encrypted = True

        loader = PDFLoader()
        with pytest.raises(LoaderError, match="encrypted"):
            loader._open(str(pdf_file), lambda _: mock_reader)

    def test_nonexistent_file_raises_loader_error(self):
        loader = PDFLoader()
        with pytest.raises(LoaderError, match="not found"):
            loader._open("/does/not/exist.pdf", MagicMock())

    def test_wrong_type_raises_loader_error(self):
        loader = PDFLoader()
        with pytest.raises(LoaderError):
            loader._open(12345, MagicMock())  # type: ignore

    def test_source_override(self):
        mock_reader = _make_mock_reader(["content"])
        loader = PDFLoader(source_override="s3://bucket/key.pdf")
        docs = loader._extract(mock_reader, "s3://bucket/key.pdf")
        assert docs[0].metadata["source"] == "s3://bucket/key.pdf"

    def test_pages_joined_by_double_newline(self):
        mock_reader = _make_mock_reader(["First page.", "Second page."])
        loader = PDFLoader()
        docs = loader._extract(mock_reader, "x.pdf")
        assert docs[0].content == "First page.\n\nSecond page."

    def test_missing_pypdf_raises_loader_error(self):
        loader = PDFLoader()
        with patch.dict("sys.modules", {"pypdf": None}):
            with pytest.raises((LoaderError, ImportError)):
                loader.load(b"fake bytes")
