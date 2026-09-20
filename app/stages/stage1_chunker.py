"""Stage 1: Deterministic In-Memory PDF Slicer / Chunker.

Slices arbitrary length PDFs into in-memory 5-page PDF streams using PyMuPDF.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import pymupdf

from app.config import PAGE_CHUNK_SIZE


@dataclass
class PDFChunk:
    """Represents an isolated, in-memory PDF slice."""
    chunk_index: int
    start_page: int
    end_page: int
    page_count: int
    pdf_bytes: bytes


def get_total_page_count(pdf_source: str | Path | bytes) -> int:
    """Returns total number of pages in the given PDF."""
    if isinstance(pdf_source, (str, Path)):
        doc = pymupdf.open(str(pdf_source))
    else:
        doc = pymupdf.open(stream=pdf_source, filetype="pdf")
    if doc.is_encrypted and not doc.authenticate(""):
        doc.close()
        raise PermissionError(f"PDF is password-protected and cannot be decrypted: {pdf_source}")
    count = len(doc)
    doc.close()
    return count


def slice_pdf_chunks(
    pdf_source: str | Path | bytes,
    chunk_size: int = PAGE_CHUNK_SIZE,
) -> Generator[PDFChunk, None, None]:
    """Slices a source PDF into sequential in-memory PDF byte chunks.

    Args:
        pdf_source: File path, Path object, or raw PDF bytes.
        chunk_size: Number of consecutive pages per chunk (default: 5).

    Yields:
        PDFChunk instances containing standalone PDF bytes.
    """
    if isinstance(pdf_source, (str, Path)):
        doc = pymupdf.open(str(pdf_source))
    else:
        doc = pymupdf.open(stream=pdf_source, filetype="pdf")

    if doc.is_encrypted and not doc.authenticate(""):
        doc.close()
        raise PermissionError(f"PDF is password-protected and cannot be decrypted: {pdf_source}")

    total_pages = len(doc)

    try:
        chunk_idx = 0
        for start_idx in range(0, total_pages, chunk_size):
            end_idx = min(start_idx + chunk_size, total_pages)
            chunk_doc = pymupdf.open()
            chunk_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx - 1)
            chunk_bytes = chunk_doc.tobytes()
            chunk_doc.close()

            yield PDFChunk(
                chunk_index=chunk_idx,
                start_page=start_idx + 1,
                end_page=end_idx,
                page_count=end_idx - start_idx,
                pdf_bytes=chunk_bytes,
            )
            chunk_idx += 1
    finally:
        doc.close()
