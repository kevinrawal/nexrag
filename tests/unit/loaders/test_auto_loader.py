from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nexrag.core.models.document import Document
from nexrag.exceptions import LoaderError
from nexrag.loaders.auto import AutoLoader


def _make_doc(content: str = "text") -> Document:
    doc = MagicMock(spec=Document)
    doc.content = content
    return doc


class TestAutoLoader:
    def setup_method(self):
        self.loader = AutoLoader()

    def test_pdf_extension_dispatches_to_pdf_loader(self):
        fake_docs = [_make_doc("pdf content")]
        with patch("nexrag.loaders.pdf.PDFLoader.load", return_value=fake_docs) as mock_load:
            result = self.loader.load("report.pdf")
        mock_load.assert_called_once_with("report.pdf")
        assert result == fake_docs

    def test_txt_extension_dispatches_to_raw_loader(self):
        fake_docs = [_make_doc("txt content")]
        with patch("nexrag.loaders.raw.RawTextLoader.load", return_value=fake_docs) as mock_load:
            result = self.loader.load("notes.txt")
        mock_load.assert_called_once_with("notes.txt")
        assert result == fake_docs

    def test_md_extension_dispatches_to_raw_loader(self):
        fake_docs = [_make_doc("md content")]
        with patch("nexrag.loaders.raw.RawTextLoader.load", return_value=fake_docs) as mock_load:
            result = self.loader.load("readme.md")
        mock_load.assert_called_once_with("readme.md")
        assert result == fake_docs

    def test_path_object_works(self):
        fake_docs = [_make_doc("content")]
        with patch("nexrag.loaders.pdf.PDFLoader.load", return_value=fake_docs):
            result = self.loader.load(Path("doc.pdf"))
        assert result == fake_docs

    def test_unknown_extension_raises_loader_error(self):
        with pytest.raises(LoaderError, match="auto-detect"):
            self.loader.load("data.xlsx")

    def test_no_extension_raises_loader_error(self):
        with pytest.raises(LoaderError, match="auto-detect"):
            self.loader.load("noextension")

    def test_error_message_lists_supported_extensions(self):
        with pytest.raises(LoaderError, match=r"\.pdf"):
            self.loader.load("file.docx")

    def test_uppercase_extension_works(self):
        fake_docs = [_make_doc("content")]
        with patch("nexrag.loaders.pdf.PDFLoader.load", return_value=fake_docs):
            result = self.loader.load("DOC.PDF")
        assert result == fake_docs
