from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nexrag.core.models.document import Document
from nexrag.exceptions import LoaderError
from nexrag.loaders.auto import AutoLoader

_PDF_BYTES = b"%PDF-1.4 fake content"
_JPEG_BYTES = b"\xff\xd8\xff fake jpeg"


def _make_doc(content: str = "text") -> Document:
    doc = MagicMock(spec=Document)
    doc.content = content
    return doc


class TestAutoLoader:
    def setup_method(self):
        self.loader = AutoLoader()

    def test_pdf_bytes_dispatches_to_pdf_loader(self):
        fake_docs = [_make_doc("pdf content")]
        with patch("nexrag.loaders.pdf.PDFLoader.load", return_value=fake_docs) as mock_load:
            result = self.loader.load(_PDF_BYTES)
        mock_load.assert_called_once_with(_PDF_BYTES)
        assert result == fake_docs

    def test_str_dispatches_to_raw_loader(self):
        fake_docs = [_make_doc("text content")]
        with patch("nexrag.loaders.raw.RawTextLoader.load", return_value=fake_docs) as mock_load:
            result = self.loader.load("some plain text")
        mock_load.assert_called_once_with("some plain text")
        assert result == fake_docs

    def test_unknown_bytes_raises_loader_error(self):
        with pytest.raises(LoaderError, match="AutoLoader could not identify"):
            self.loader.load(_JPEG_BYTES)

    def test_unknown_bytes_error_mentions_supported_formats(self):
        with pytest.raises(LoaderError, match="%PDF"):
            self.loader.load(b"\x00\x00\x00\x00 unknown format")

    def test_path_input_raises_loader_error(self):
        with pytest.raises(LoaderError, match="expects bytes or str"):
            self.loader.load(Path("file.pdf"))  # type: ignore

    def test_int_input_raises_loader_error(self):
        with pytest.raises(LoaderError, match="expects bytes or str"):
            self.loader.load(42)  # type: ignore

    def test_source_override_propagated_to_pdf_loader(self):
        loader = AutoLoader(source_override="s3://bucket/file.pdf")
        fake_docs = [_make_doc("pdf")]
        with patch("nexrag.loaders.pdf.PDFLoader.load", return_value=fake_docs):
            with patch("nexrag.loaders.pdf.PDFLoader.__init__", return_value=None) as mock_init:
                loader.load(_PDF_BYTES)
        mock_init.assert_called_once_with(source_override="s3://bucket/file.pdf")

    def test_source_override_propagated_to_raw_loader(self):
        loader = AutoLoader(source_override="my-doc-id")
        fake_docs = [_make_doc("text")]
        with patch("nexrag.loaders.raw.RawTextLoader.load", return_value=fake_docs):
            with patch("nexrag.loaders.raw.RawTextLoader.__init__", return_value=None) as mock_init:
                loader.load("some text")
        mock_init.assert_called_once_with(source="my-doc-id")

    def test_empty_bytes_raises_loader_error(self):
        with pytest.raises(LoaderError):
            self.loader.load(b"")

    def test_pdf_magic_case_sensitive(self):
        # Magic byte detection is exact — lowercase %pdf does not match
        with pytest.raises(LoaderError):
            self.loader.load(b"%pdf-1.4 lowercase")

    def test_no_source_override_raw_loader_has_no_source_in_metadata(self):
        # When source_override is unset, text documents must not get a synthetic
        # source — that would make every text look like a duplicate of the first.
        loader = AutoLoader()
        docs = loader.load("hello world")
        assert "source" not in docs[0].metadata

    def test_no_source_override_pdf_loader_has_no_source_in_metadata(self):
        from unittest.mock import MagicMock, patch

        from nexrag.core.models.document import Document

        loader = AutoLoader()
        fake_doc = MagicMock(spec=Document)
        fake_doc.metadata = {}
        with patch("nexrag.loaders.pdf.PDFLoader.load", return_value=[fake_doc]):
            docs = loader.load(b"%PDF-1.4 content")
        assert "source" not in docs[0].metadata
