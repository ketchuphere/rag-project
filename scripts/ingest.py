"""
CLI helper – ingest PDFs into the local vector store on disk.
Useful for pre-indexing a large corpus before starting the Streamlit app.

Usage:
    python scripts/ingest.py path/to/doc1.pdf path/to/doc2.pdf

Future improvements:
  - Add --output-dir flag to specify a custom persistent ChromaDB path.
  - Support glob patterns: python scripts/ingest.py docs/*.pdf
  - Print per-file ingestion statistics and total chunk count on completion.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ingestion.pdf_loader import load_pdfs
from app.services.chunking.text_splitter import split_documents


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <pdf_file> [<pdf_file> ...]")
        sys.exit(1)

    paths = [open(p, "rb") for p in sys.argv[1:]]  # noqa: WPS515

    class _NamedFile:
        def __init__(self, f, name):
            self._f = f
            self.name = name
        def read(self, *a, **kw):
            return self._f.read(*a, **kw)
        def seek(self, *a, **kw):
            return self._f.seek(*a, **kw)

    named = [_NamedFile(open(p, "rb"), p) for p in sys.argv[1:]]
    documents, stats = load_pdfs(named)
    chunks = split_documents(documents)

    for s in stats:
        print(f"  {s['name']}: {s['extracted_pages']}/{s['pages']} pages → {len(chunks)} total chunks")

    print(f"\n✅ Ingested {len(documents)} pages into {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
