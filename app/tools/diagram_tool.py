"""Diagram, specimen, and table cropping engine using native pymupdf."""

from __future__ import annotations

import io
from pathlib import Path
from google.adk.tools import ToolContext
from google.genai import types
from PIL import Image
import pymupdf

from app.config import ASSETS_DIR, RENDER_DPI


def crop_diagram_from_bytes(
    pdf_bytes: bytes,
    page_in_chunk: int,
    box_2d: list[int],
    label: str,
    output_dir: str | Path = ASSETS_DIR,
) -> tuple[str, bytes]:
    """Crops a normalized bounding box from an in-memory PDF page and returns WebP bytes.

    Args:
        pdf_bytes: Raw binary bytes of the 5-page PDF slice.
        page_in_chunk: 1-based page index relative to the chunk (1 to 5).
        box_2d: Normalized coordinates [ymin, xmin, ymax, xmax] on a 0-1000 scale.
        label: Semantic identifier slug (e.g., 'table-q1-electrolytes', 'brachial-plexus').
        output_dir: Target directory path for WebP files.

    Returns:
        A tuple containing the relative disk file path string and the raw WebP bytes.
    """
    if not pdf_bytes:
        raise ValueError("Active PDF chunk bytes are empty or not provided.")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    zero_based_index = max(0, page_in_chunk - 1)
    if zero_based_index >= len(doc):
        doc.close()
        raise IndexError(
            f"Requested page_in_chunk {page_in_chunk} exceeds chunk page count {len(doc)}."
        )

    page = doc[zero_based_index]
    page_rect = page.rect
    width: float = page_rect.width
    height: float = page_rect.height

    ymin, xmin, ymax, xmax = box_2d
    crop_rect = pymupdf.Rect(
        (xmin / 1000.0) * width,
        (ymin / 1000.0) * height,
        (xmax / 1000.0) * width,
        (ymax / 1000.0) * height,
    )

    # Render at 200 DPI (72 DPI standard base)
    zoom = float(RENDER_DPI) / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, clip=crop_rect)
    doc.close()

    image_mode = "RGBA" if pix.alpha else "RGB"
    image = Image.frombytes(image_mode, (pix.width, pix.height), pix.samples)

    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=85)
    webp_bytes = buffer.getvalue()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    clean_label = label.strip().replace(" ", "_").replace("/", "_")
    if not clean_label.endswith(".webp"):
        clean_label = f"{clean_label}.webp"

    target_file = out_path / clean_label
    target_file.write_bytes(webp_bytes)

    rel_path = f"assets/{clean_label}"
    return rel_path, webp_bytes


async def crop_diagram_tool(
    tool_context: ToolContext,
    page_in_chunk: int,
    box_2d: list[int],
    label: str,
) -> str:
    """Crops an anatomical diagram, spotter, graph, or data table into a WebP asset.

    Args:
        tool_context: Google ADK ToolContext for artifact persistence.
        page_in_chunk: 1-based page number within the 5-page chunk (1 to 5).
        box_2d: Normalized bounding box [ymin, xmin, ymax, xmax] scaled 0 to 1000.
        label: Descriptive slug for the diagram or table.

    Returns:
        Markdown image link referencing the persisted asset.
    """
    artifact_part = await tool_context.load_artifact("current_chunk.pdf")
    if not artifact_part or not artifact_part.inline_data or not artifact_part.inline_data.data:
        raise ValueError(
            "Session artifact 'current_chunk.pdf' is missing or empty in active ToolContext."
        )

    pdf_bytes: bytes = artifact_part.inline_data.data

    file_path, webp_bytes = crop_diagram_from_bytes(
        pdf_bytes=pdf_bytes,
        page_in_chunk=page_in_chunk,
        box_2d=box_2d,
        label=label,
        output_dir=ASSETS_DIR,
    )

    image_part = types.Part.from_bytes(data=webp_bytes, mime_type="image/webp")
    await tool_context.save_artifact(f"{label}.webp", image_part)

    return f"![{label}]({file_path})"
