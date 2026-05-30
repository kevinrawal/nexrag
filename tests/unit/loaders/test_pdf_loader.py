from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from nexrag.exceptions import LoaderError
from nexrag.loaders.pdf import PDFLoader, _parse_pdf_date


def _make_mock_reader(
    pages_text: list[str], is_encrypted: bool = False, metadata: dict | None = None
):
    """Build a minimal PdfReader mock with optional metadata."""
    pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    reader = MagicMock()
    reader.is_encrypted = is_encrypted
    reader.pages = pages

    if metadata is not None:
        meta = MagicMock()
        # Explicitly default all known PDF metadata attributes to None so that
        # MagicMock doesn't auto-create truthy objects for unset fields.
        for field in [
            "author",
            "title",
            "subject",
            "creator",
            "producer",
            "creation_date",
            "modification_date",
        ]:
            setattr(meta, field, None)
        for attr, val in metadata.items():
            setattr(meta, attr, val)
        meta.get = lambda key, default=None: metadata.get(key, default)
        reader.metadata = meta
    else:
        reader.metadata = None

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


class TestParsePdfDate:
    def test_full_date_string(self):
        assert _parse_pdf_date("D:20240115143022") == "2024-01-15T14:30:22"

    def test_date_without_prefix(self):
        assert _parse_pdf_date("20240115143022") == "2024-01-15T14:30:22"

    def test_date_with_timezone(self):
        assert _parse_pdf_date("D:20240115143022+05'30'") == "2024-01-15T14:30:22"

    def test_date_only_eight_digits(self):
        assert _parse_pdf_date("D:20240115") == "2024-01-15T00:00:00"

    def test_empty_string_returns_none(self):
        assert _parse_pdf_date("") is None

    def test_too_short_returns_none(self):
        assert _parse_pdf_date("D:2024") is None

    def test_malformed_returns_none(self):
        assert _parse_pdf_date("not-a-date") is None


class TestPDFLoaderMetadata:
    def test_default_extracts_page_count(self):
        reader = _make_mock_reader(["content"])
        loader = PDFLoader()
        docs = loader._extract(reader, "test.pdf")
        assert docs[0].metadata["page_count"] == 1

    def test_all_string_fields_extracted(self):
        reader = _make_mock_reader(
            ["content"],
            metadata={
                "author": "Alice",
                "title": "My PDF",
                "subject": "Testing",
                "creator": "Word",
                "producer": "Adobe",
            },
        )
        loader = PDFLoader()
        docs = loader._extract(reader, "test.pdf")
        meta = docs[0].metadata
        assert meta["author"] == "Alice"
        assert meta["title"] == "My PDF"
        assert meta["subject"] == "Testing"
        assert meta["creator"] == "Word"
        assert meta["producer"] == "Adobe"

    def test_datetime_object_serialised_to_iso(self):
        dt = datetime(2024, 1, 15, 14, 30, 22)
        reader = _make_mock_reader(
            ["content"],
            metadata={"creation_date": dt, "modification_date": dt},
        )
        loader = PDFLoader()
        docs = loader._extract(reader, "test.pdf")
        assert docs[0].metadata["created_at"] == dt.isoformat()
        assert docs[0].metadata["modified_at"] == dt.isoformat()

    def test_raw_date_string_parsed(self):
        reader = _make_mock_reader(
            ["content"],
            metadata={"creation_date": None, "/CreationDate": "D:20240115143022"},
        )
        loader = PDFLoader()
        docs = loader._extract(reader, "test.pdf")
        assert docs[0].metadata["created_at"] == "2024-01-15T14:30:22"

    def test_missing_pdf_properties_omitted(self):
        reader = _make_mock_reader(["content"], metadata={})
        loader = PDFLoader()
        docs = loader._extract(reader, "test.pdf")
        meta = docs[0].metadata
        assert "author" not in meta
        assert "title" not in meta
        assert "created_at" not in meta
        assert "page_count" in meta

    def test_metadata_fields_whitelist(self):
        reader = _make_mock_reader(
            ["content"],
            metadata={"author": "Alice", "title": "My PDF"},
        )
        loader = PDFLoader(metadata_fields=["title", "page_count"])
        docs = loader._extract(reader, "test.pdf")
        meta = docs[0].metadata
        assert "title" in meta
        assert "page_count" in meta
        assert "author" not in meta

    def test_include_metadata_false_suppresses_all(self):
        reader = _make_mock_reader(
            ["content"],
            metadata={"author": "Alice", "title": "My PDF"},
        )
        loader = PDFLoader(include_metadata=False)
        docs = loader._extract(reader, "test.pdf")
        meta = docs[0].metadata
        assert "author" not in meta
        assert "title" not in meta
        assert "page_count" not in meta
        assert meta["source"] == "test.pdf"

    def test_no_reader_metadata_still_returns_page_count(self):
        reader = _make_mock_reader(["p1", "p2"])
        reader.metadata = None
        loader = PDFLoader()
        docs = loader._extract(reader, "test.pdf")
        assert docs[0].metadata["page_count"] == 2

    def test_malformed_date_omitted_not_raised(self):
        reader = _make_mock_reader(
            ["content"],
            metadata={"creation_date": None, "/CreationDate": "not-a-date"},
        )
        loader = PDFLoader()
        docs = loader._extract(reader, "test.pdf")
        assert "created_at" not in docs[0].metadata

    def test_source_always_present_when_include_metadata_false(self):
        reader = _make_mock_reader(["content"])
        loader = PDFLoader(include_metadata=False)
        docs = loader._extract(reader, "s3://bucket/key.pdf")
        assert docs[0].metadata["source"] == "s3://bucket/key.pdf"
