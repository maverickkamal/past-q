"""Stage 2: Master Assembly & Pointer-Only Manifest Generator.

This stage takes the verbatim Markdown chunks produced by Stage 1, joins them
into a master booklet document, uses Gemini Flash with structured output to detect
exam boundaries (emitting integer page pointers only), and deterministically slices
the master document into isolated exam paper strings in Python.
"""

from __future__ import annotations

import os
import re
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.schemas import ExamManifest, ExamPointer

load_dotenv()

STAGE2_MODEL = os.getenv("STAGE2_MODEL", "gemini-flash-latest")

STAGE2_SYSTEM_INSTRUCTION = """\
You are an expert preclinical medical exam curator.
You will receive a master Markdown document transcribed from a preclinical past question booklet.
The document contains page break delimiters in the exact format:
<!-- PAGE BREAK [N] -->
where [N] is the 1-based booklet page number.

Your responsibility:
1. Scan all page headings, departmental titles, course codes, exam titles, academic sessions, and instructions.
2. Identify every distinct examination paper, continuous assessment test, spotter mock, or quiz present in the booklet.
3. For each detected examination, emit an ExamPointer specifying:
   - exam_id: A deterministic slug combining discipline, session, and paper (e.g. 'ANA_2012_CAT_P1', 'PHY_2013_PROF_P2').
   - discipline: Exactly one of 'Anatomy', 'Physiology', or 'Biochemistry'.
   - level: '200L' if 200 Level (e.g., 200 series course codes ANA 201/202, 200L C.A., 2nd Year), '300L' if 300 Level (e.g., 300 series course codes ANA 301, 3rd Year, Part 1 MBBS), or 'UNKNOWN'.
   - category: One of 'CA_QUIZ' (continuous assessments/in-course tests/quizzes), 'STEEPLECHASE' (practical/spotter/station exams), 'ESSAY' (essay/SEQ/SAQ papers), 'OBJECTIVE' (pure MCQ papers).
   - paper_title: The official title (e.g. '200L First Semester Continuous Assessment Test', 'Paper I (Gross Anatomy)').
   - session: The academic session (e.g. '2012/2013') or 'UNKNOWN'.
   - examiner: The primary lecturer/examiner if indicated, otherwise 'UNKNOWN'.
   - start_page: The 1-based integer page where this exam starts.
   - end_page: The 1-based integer page where this exam ends (inclusive).
4. STRICT INVARIANT:
   - NEVER output, summarize, or reproduce question stems, MCQ options, or exam content.
   - Output ONLY the structured ExamManifest containing the list of ExamPointer objects.
"""


def create_stage2_agent(model_name: str | None = None) -> Agent:
    """Initializes the Stage 2 manifest generator agent with structured schema."""
    return Agent(
        name="stage2_manifest_generator",
        model=model_name or STAGE2_MODEL,
        output_schema=ExamManifest,
        instruction=STAGE2_SYSTEM_INSTRUCTION,
    )



root_agent = create_stage2_agent()


def assemble_master_markdown(chunk_transcriptions: list[str]) -> str:
    """Concatenates chunk markdown texts into a single master document.

    Args:
        chunk_transcriptions: Ordered list of Markdown strings from Stage 1 chunks.

    Returns:
        Unified master Markdown document.
    """
    cleaned_chunks = [c.strip() for c in chunk_transcriptions if c and c.strip()]
    return "\n\n".join(cleaned_chunks)


def slice_markdown_pages(master_markdown: str, start_page: int, end_page: int) -> str:
    """Extracts markdown text corresponding to booklet pages start_page through end_page.

    Args:
        master_markdown: Complete transcribed booklet markdown containing <!-- PAGE BREAK [N] --> tags.
        start_page: 1-based starting booklet page index.
        end_page: 1-based ending booklet page index (inclusive).

    Returns:
        The sliced markdown segment corresponding to the requested page span.
    """
    matches = list(
        re.finditer(
            r"<!--\s*PAGE\s+BREAK\s*\[(\d+)\]\s*-->", master_markdown, re.IGNORECASE
        )
    )
    if not matches:
        return master_markdown.strip()

    # Map page number to character start offset
    page_map: dict[int, int] = {int(m.group(1)): m.start() for m in matches}
    sorted_pages = sorted(page_map.keys())

    # Determine start character offset
    if start_page <= sorted_pages[0]:
        start_pos = 0
    else:
        applicable_starts = [p for p in sorted_pages if p <= start_page]
        start_pos = page_map[applicable_starts[-1]] if applicable_starts else 0

    # Determine end character offset (start of first page > end_page, or end of text)
    subsequent_pages = [p for p in sorted_pages if p > end_page]
    if subsequent_pages:
        end_pos = page_map[subsequent_pages[0]]
    else:
        end_pos = len(master_markdown)

    return master_markdown[start_pos:end_pos].strip()


def slice_exam_papers(
    master_markdown: str, manifest: ExamManifest
) -> dict[str, tuple[ExamPointer, str]]:
    """Deterministically partitions master markdown into isolated exam paper strings.

    Args:
        master_markdown: Full transcribed booklet text.
        manifest: Validated ExamManifest containing page pointers.

    Returns:
        A dictionary mapping exam_id to a tuple of (ExamPointer, isolated_markdown_text).
    """
    sliced_papers: dict[str, tuple[ExamPointer, str]] = {}
    for pointer in manifest.exams:
        isolated_text = slice_markdown_pages(
            master_markdown=master_markdown,
            start_page=pointer.start_page,
            end_page=pointer.end_page,
        )
        sliced_papers[pointer.exam_id] = (pointer, isolated_text)

    return sliced_papers


async def generate_manifest(
    master_markdown: str,
    agent: Agent | None = None,
    app_name: str = "medical_exam_pipeline",
) -> ExamManifest:
    """Executes the Stage 2 manifest agent turn and returns a validated ExamManifest.

    Args:
        master_markdown: Full transcribed booklet markdown text.
        agent: Optional custom Agent instance.
        app_name: ADK application name.

    Returns:
        Pydantic ExamManifest object with all detected exam pointers.
    """
    manifest_agent = agent or create_stage2_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=manifest_agent,
        session_service=session_service,
        app_name=app_name,
    )

    user_id = "exam_pipeline_worker"
    session_id = "stage2_manifest_session"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    prompt = (
        "Analyze the following transcribed past questions booklet markdown.\n"
        "Identify all examination boundaries, discipline, session, examiner, and page ranges.\n"
        "Emit the ExamManifest JSON object.\n\n"
        f"{master_markdown}"
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    collected_json_text: list[str] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.content and event.content.role == "model":
            for part in event.content.parts:
                if part.text:
                    collected_json_text.append(part.text)

    raw_json = "".join(collected_json_text).strip()
    if not raw_json:
        raise RuntimeError("Stage 2 agent returned an empty response.")

    return ExamManifest.model_validate_json(raw_json)
