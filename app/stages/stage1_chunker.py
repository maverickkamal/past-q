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


def get_optimal_chunk_size(
    total_pages: int,
    max_single_chunk_pages: int = 25,
    large_booklet_chunk_size: int = 20,
) -> int:
    """Calculates optimal chunk size balancing API call efficiency, memory, and LLM attention.

    - Booklets with <= 25 pages are processed in a single chunk (1 API call).
    - Massive compilations (> 25 pages, up to 400+ pages) are split into 20-page chunks
      (preserving visual fidelity and LLM attention without memory spikes).
    """
    if total_pages <= max_single_chunk_pages:
        return total_pages
    return large_booklet_chunk_size


def slice_pdf_chunks(
    pdf_source: str | Path | bytes,
    chunk_size: int | None = None,
) -> Generator[PDFChunk, None, None]:
    """Slices a source PDF into sequential in-memory PDF byte chunks.

    Args:
        pdf_source: File path, Path object, or raw PDF bytes.
        chunk_size: Number of consecutive pages per chunk. If None, computes adaptive size.

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
    effective_chunk_size = chunk_size if chunk_size is not None else get_optimal_chunk_size(total_pages)

    try:
        chunk_idx = 0
        for start_idx in range(0, total_pages, effective_chunk_size):
            end_idx = min(start_idx + effective_chunk_size, total_pages)
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
