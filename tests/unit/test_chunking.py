"""
Unit tests – text splitter / chunking service.
"""

from langchain_core.documents import Document
from app.services.chunking.text_splitter import split_documents


def test_chunks_created():
    docs = [Document(page_content="word " * 500, metadata={"source": "test.pdf", "page": 1})]
    chunks = split_documents(docs)
    assert len(chunks) > 1, "Expected multiple chunks for long document"
    print(f"  ✓ {len(chunks)} chunks created from 1 document")


def test_metadata_preserved():
    docs = [Document(page_content="hello world " * 100, metadata={"source": "foo.pdf", "page": 2})]
    chunks = split_documents(docs)
    for chunk in chunks:
        assert chunk.metadata["source"] == "foo.pdf"
        assert chunk.metadata["page"] == 2
        assert "chunk_index" in chunk.metadata
    print("  ✓ Metadata preserved and chunk_index assigned")


def test_empty_input():
    chunks = split_documents([])
    assert chunks == []
    print("  ✓ Empty input returns empty list")


if __name__ == "__main__":
    test_chunks_created()
    test_metadata_preserved()
    test_empty_input()
    print("\n✅ All chunking tests passed.")
