from pathlib import Path

import pytest

from nexrag.exceptions import LoaderError
from nexrag.loaders.raw import RawTextLoader


class TestRawTextLoader:
    def setup_method(self):
        self.loader = RawTextLoader()

    def test_load_string_returns_one_document(self):
        docs = self.loader.load("Hello world. This is some text.")
        assert len(docs) == 1

    def test_document_content_matches_input(self):
        text = "Resume content goes here."
        docs = self.loader.load(text)
        assert docs[0].content == text

    def test_no_source_when_not_provided(self):
        docs = self.loader.load("Some text")
        assert "source" not in docs[0].metadata

    def test_custom_default_source(self):
        loader = RawTextLoader(source="my-doc-id")
        docs = loader.load("content")
        assert docs[0].metadata["source"] == "my-doc-id"

    def test_tuple_input_overrides_source(self):
        docs = self.loader.load(("content here", "custom-source"))
        assert docs[0].content == "content here"
        assert docs[0].metadata["source"] == "custom-source"

    def test_empty_string_raises_loader_error(self):
        with pytest.raises(LoaderError):
            self.loader.load("")

    def test_whitespace_only_raises_loader_error(self):
        with pytest.raises(LoaderError):
            self.loader.load("   \n\t  ")

    def test_wrong_type_raises_loader_error(self):
        with pytest.raises(LoaderError):
            self.loader.load(42)  # type: ignore

    def test_bad_tuple_raises_loader_error(self):
        with pytest.raises(LoaderError):
            self.loader.load(("only one element",))  # type: ignore

    def test_doc_id_is_set(self):
        docs = self.loader.load("text")
        assert docs[0].doc_id
        assert isinstance(docs[0].doc_id, str)

    def test_path_input_raises_loader_error(self):
        with pytest.raises(LoaderError, match="expects str"):
            self.loader.load(Path("file.txt"))  # type: ignore

    def test_bytes_input_raises_loader_error(self):
        with pytest.raises(LoaderError, match="expects str"):
            self.loader.load(b"some bytes")  # type: ignore
