"""Tests for _stabilise_doc_ids — stable parent_doc_id across re-ingestions."""

import hashlib

from nexrag.core.models.document import Document
from nexrag.core.pipeline.ingestion import _stabilise_doc_ids


class TestStabiliseDocIds:
    def test_source_present_produces_deterministic_id(self):
        doc = Document(content="text", metadata={"source": "s3://bucket/file.pdf"})
        [stabilised] = _stabilise_doc_ids([doc])
        expected = hashlib.sha256(b"s3://bucket/file.pdf").hexdigest()[:32]
        assert stabilised.doc_id == expected

    def test_same_source_always_gives_same_doc_id(self):
        doc1 = Document(content="text", metadata={"source": "my-doc"})
        doc2 = Document(content="text", metadata={"source": "my-doc"})
        # doc1 and doc2 have different random UUIDs before stabilisation
        assert doc1.doc_id != doc2.doc_id
        [s1] = _stabilise_doc_ids([doc1])
        [s2] = _stabilise_doc_ids([doc2])
        assert s1.doc_id == s2.doc_id

    def test_no_source_keeps_original_doc_id(self):
        doc = Document(content="text", metadata={})
        original_id = doc.doc_id
        [stabilised] = _stabilise_doc_ids([doc])
        assert stabilised.doc_id == original_id

    def test_content_and_metadata_unchanged(self):
        doc = Document(content="hello", metadata={"source": "x", "tenant": "acme"})
        [stabilised] = _stabilise_doc_ids([doc])
        assert stabilised.content == "hello"
        assert stabilised.metadata["tenant"] == "acme"
        assert stabilised.metadata["source"] == "x"

    def test_multiple_docs_processed_independently(self):
        doc_with = Document(content="a", metadata={"source": "src-a"})
        doc_without = Document(content="b", metadata={})
        original_without_id = doc_without.doc_id

        results = _stabilise_doc_ids([doc_with, doc_without])
        assert len(results) == 2

        expected = hashlib.sha256(b"src-a").hexdigest()[:32]
        assert results[0].doc_id == expected
        assert results[1].doc_id == original_without_id

    def test_already_stable_doc_returned_as_is(self):
        source = "stable-source"
        stable_id = hashlib.sha256(source.encode()).hexdigest()[:32]
        doc = Document(content="text", metadata={"source": source}, doc_id=stable_id)
        [result] = _stabilise_doc_ids([doc])
        assert result is doc  # same object, not reconstructed
