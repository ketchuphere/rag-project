"""
Ingestion service – streaming PDF loader.

Improvements implemented:
   Streaming ingestion: pages yielded one-by-one via a generator,
     so large PDFs never load fully into memory.
   Multimodal extraction: embedded images extracted per page and stored
     as base64 blobs in Document metadata for vision-capable pipelines.
"""

from __future__ import annotations

import base64
import io
from typing import Generator

from PyPDF2 import PdfReader
from langchain_core.documents import Document



def stream_pdf_pages(pdf_file) -> Generator[Document, None, None]:
    """
    Yield one Document per extractable page without loading the full PDF.

    Args:
        pdf_file: File-like object with a .name attribute (e.g. Streamlit UploadedFile).

    Yields:
        LangChain Document with page_content and metadata
        (source, page, images list of base64 strings).
    """
    reader = PdfReader(pdf_file)
    file_name = getattr(pdf_file, "name", "unknown.pdf")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        images: list[str] = []
        if hasattr(page, "images"):
            for img_obj in page.images:
                try:
                    b64 = base64.b64encode(img_obj.data).decode("utf-8")
                    images.append(b64)
                except Exception:
                    pass

        yield Document(
            page_content=text,
            metadata={
                "source": file_name,
                "page": page_number,
                "images": images,          # list[str] base64 – may be empty
                "has_images": len(images) > 0,
            },
        )


def load_pdfs(pdf_files: list) -> tuple[list[Document], list[dict]]:
    """
    Eagerly load all PDFs using the streaming generator under the hood.

    Returns:
        documents  – all extracted page Documents
        file_stats – per-file metadata (name, total pages, extracted, image count)
    """
    documents: list[Document] = []
    file_stats: list[dict] = []

    for pdf in pdf_files:
        reader = PdfReader(pdf)
        total_pages = len(reader.pages)
        file_name = getattr(pdf, "name", "unknown.pdf")

        # Rewind for streaming pass
        if hasattr(pdf, "seek"):
            pdf.seek(0)

        page_docs = list(stream_pdf_pages(pdf))
        documents.extend(page_docs)

        file_stats.append({
            "name": file_name,
            "pages": total_pages,
            "extracted_pages": len(page_docs),
            "image_count": sum(len(d.metadata.get("images", [])) for d in page_docs),
        })

    return documents, file_stats
