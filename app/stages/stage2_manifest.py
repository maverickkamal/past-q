"""Stage 2: Master Assembly & Pointer-Only Manifest Generator.

This stage takes the verbatim Markdown chunks produced by Stage 1, joins them
into a master booklet document, uses Gemini Flash with structured output to detect
exam boundaries (emitting integer page pointers only), and deterministically slices
the master document into isolated exam paper strings in Python.
"""

import asyncio
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.schemas import ExamManifest, ExamPointer, PreScanMetadata
from app.tools.retry_handler import (
    calculate_backoff,
    configure_clean_logging,
    get_retry_reason,
    is_resource_exhausted_error,
)

load_dotenv()
configure_clean_logging()

STAGE2_MODEL = os.getenv("STAGE2_MODEL", "gemini-3.5-flash-lite")

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
4. COMPILATIONS, QUESTION BANKS, AND TEXTBOOKS:
   - If the booklet is a subject question bank, MCQ compilation, or revision textbook (e.g. Martin Caon, MedHub compilation) rather than a single university exam:
   - Identify EACH Chapter, System, or Major Topic Section as a distinct exam paper.
   - For example:
     - paper_title: 'Chapter 1: Organisation of the Body' (Pages 1-6) -> exam_id: 'ANA_CH01_ORGANISATION'
     - paper_title: 'Chapter 2: Cells and Tissues' (Pages 7-34) -> exam_id: 'ANA_CH02_CELLS'
   - Set category='OBJECTIVE' for MCQ banks, or 'CA_QUIZ'/'ESSAY' as appropriate.
   - NEVER return an empty exams list (`[]`) if the document contains medical questions or MCQs. Every page containing questions must belong to an exam entry.
5. STRICT INVARIANT:
   - NEVER output, summarize, or reproduce question stems, MCQ options, or exam content.
   - Output ONLY the structured ExamManifest containing the list of ExamPointer objects.
"""


def create_stage2_agent(model_name: str | None = None) -> Agent:
    """Initializes the Stage 2 manifest generator agent with structured schema and HTTP retry options."""
    return Agent(
        name="stage2_manifest_generator",
        model=model_name or STAGE2_MODEL,
        output_schema=ExamManifest,
        instruction=STAGE2_SYSTEM_INSTRUCTION,
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(initial_delay=2.0, attempts=3),
            )
        ),
    )



root_agent = create_stage2_agent()


def assemble_master_markdown(chunk_transcriptions: list[str] | str) -> str:
    """Concatenates chunk markdown texts into a single master document.

    Args:
        chunk_transcriptions: Ordered list of Markdown strings from Stage 1 chunks or a single unified Markdown string.

    Returns:
        Unified master Markdown document.
    """
    if isinstance(chunk_transcriptions, str):
        return chunk_transcriptions.strip()
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

    page_map: dict[int, int] = {int(m.group(1)): m.start() for m in matches}
    sorted_pages = sorted(page_map.keys())

    if start_page <= sorted_pages[0]:
        start_pos = 0
    else:
        applicable_starts = [p for p in sorted_pages if p <= start_page]
        start_pos = page_map[applicable_starts[-1]] if applicable_starts else 0

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
    pre_scan_meta: PreScanMetadata | None = None,
) -> ExamManifest:
    """Executes the Stage 2 manifest agent turn and returns a validated ExamManifest.

    Args:
        master_markdown: Full transcribed booklet markdown text.
        agent: Optional custom Agent instance.
        app_name: ADK application name.
        pre_scan_meta: Optional pre-scan metadata for context hints.

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

    hints = ""
    if pre_scan_meta:
        hints = (
            "Pre-scan detected booklet metadata:\n"
            f"- Discipline hint: {pre_scan_meta.discipline}\n"
            f"- Session hint: {pre_scan_meta.session}\n"
            f"- Examiner hint: {pre_scan_meta.examiner}\n"
            f"- Institution hint: {pre_scan_meta.institution}\n\n"
        )

    prompt = (
        "Analyze the following transcribed past questions booklet markdown.\n"
        f"{hints}"
        "Identify all examination boundaries, discipline, session, examiner, and page ranges.\n"
        "Emit the ExamManifest JSON object.\n\n"
        f"{master_markdown}"
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    max_retries = 6
    raw_json = ""
    for attempt in range(1, max_retries + 1):
        try:
            session_id = f"stage2_manifest_session_{attempt}"
            await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
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
            break
        except Exception as exc:
            if is_resource_exhausted_error(exc) and attempt < max_retries:
                backoff = calculate_backoff(attempt=attempt, base=6.0)
                reason = get_retry_reason(exc)
                print(
                    f"\n  {reason} during Stage 2 manifest generation. "
                    f"Retrying in {int(backoff)}s (attempt {attempt}/{max_retries})...",
                    flush=True,
                )
                await asyncio.sleep(backoff)
            else:
                raise

    manifest = ExamManifest.model_validate_json(raw_json)
    if pre_scan_meta:
        for ep in manifest.exams:
            if (ep.institution == "UNKNOWN" or not ep.institution) and pre_scan_meta.institution != "UNKNOWN":
                ep.institution = pre_scan_meta.institution

    return manifest


def synthesize_fallback_manifest(
    master_markdown: str,
    pre_scan_meta: PreScanMetadata | None = None,
    source_name: str = "booklet",
) -> ExamManifest:
    """Deterministically synthesizes an ExamManifest when LLM manifest generation produces 0 exams.

    Ensures that compilations, textbooks, and past-question collections are never dropped:
    1. Attempts to detect Chapter or Section headings and their corresponding page break locations.
    2. If chapters are found, partitions the document into chapter-wise ExamPointers.
    3. If no chapters are found, partitions the document into consecutive ~20-page blocks.

    Args:
        master_markdown: Full transcribed markdown containing page break markers.
        pre_scan_meta: Optional pre-scan metadata for discipline/level context.
        source_name: Name of the PDF or source booklet file.

    Returns:
        ExamManifest containing partitioned ExamPointers.
    """
    page_matches = list(
        re.finditer(
            r"<!--\s*PAGE\s+BREAK\s*\[(\d+)\]\s*-->", master_markdown, re.IGNORECASE
        )
    )
    if not page_matches:
        page_numbers = [1]
        page_offsets = {1: 0}
    else:
        page_offsets = {int(m.group(1)): m.start() for m in page_matches}
        page_numbers = sorted(page_offsets.keys())

    min_page = page_numbers[0]
    max_page = page_numbers[-1]

    discipline = "Anatomy"
    if pre_scan_meta and pre_scan_meta.discipline in ("Anatomy", "Physiology", "Biochemistry"):
        discipline = pre_scan_meta.discipline
    elif "physio" in source_name.lower():
        discipline = "Physiology"
    elif "bioch" in source_name.lower():
        discipline = "Biochemistry"

    disc_slug = discipline[:3].upper()
    level = "200L"
    if pre_scan_meta and getattr(pre_scan_meta, "level", None) in ("200L", "300L"):
        level = getattr(pre_scan_meta, "level")
    session = pre_scan_meta.session if (pre_scan_meta and pre_scan_meta.session != "UNKNOWN") else "COMPILATION"
    institution = pre_scan_meta.institution if (pre_scan_meta and pre_scan_meta.institution != "UNKNOWN") else "UNKNOWN"

    chapter_regex = re.compile(
        r"(?:^|\n)\s*(?:#+\s*)?(?:Chapter|CHAPTER|Section|SECTION)\s+([0-9IVXLCDM]+)[\s:\-–—]+([^\n\r]+)",
        re.IGNORECASE,
    )
    chapter_matches = list(chapter_regex.finditer(master_markdown))

    exams: list[ExamPointer] = []
    clean_stem = re.sub(r"[^A-Za-z0-9]", "_", Path(source_name).stem).strip("_")

    if len(chapter_matches) >= 2:
        chapter_pages: list[tuple[int, str, str]] = []  # (start_page, chapter_num, title)
        for m in chapter_matches:
            ch_num = m.group(1).strip()
            ch_title = m.group(2).strip()
            ch_title = re.sub(r"[\*#_]+", "", ch_title).strip()
            m_offset = m.start()
            prior_pages = [p for p in page_numbers if page_offsets.get(p, 0) <= m_offset]
            start_p = prior_pages[-1] if prior_pages else min_page
            chapter_pages.append((start_p, ch_num, ch_title))

        for idx, (start_p, ch_num, ch_title) in enumerate(chapter_pages):
            if idx + 1 < len(chapter_pages):
                next_start = chapter_pages[idx + 1][0]
                end_p = max(start_p, next_start - 1)
            else:
                end_p = max_page

            title_slug = re.sub(r"[^A-Za-z0-9]", "_", ch_title)[:12].strip("_")
            exam_id = f"{disc_slug}_CH{ch_num}_{title_slug}".rstrip("_")
            exams.append(
                ExamPointer(
                    exam_id=exam_id,
                    discipline=discipline,
                    level=level,
                    category="OBJECTIVE",
                    paper_title=f"Chapter {ch_num}: {ch_title}",
                    session=session,
                    examiner="UNKNOWN",
                    institution=institution,
                    start_page=start_p,
                    end_page=end_p,
                )
            )

    if len(exams) < 2:
        exams = []
        chunk_size = 20
        part_idx = 1
        for start_p in range(min_page, max_page + 1, chunk_size):
            end_p = min(start_p + chunk_size - 1, max_page)
            exams.append(
                ExamPointer(
                    exam_id=f"{disc_slug}_{clean_stem[:14]}_P{part_idx:02d}",
                    discipline=discipline,
                    level=level,
                    category="OBJECTIVE",
                    paper_title=f"{clean_stem.replace('_', ' ').title()} - Part {part_idx} (Pages {start_p}-{end_p})",
                    session=session,
                    examiner="UNKNOWN",
                    institution=institution,
                    start_page=start_p,
                    end_page=end_p,
                )
            )
            part_idx += 1

    return ExamManifest(exams=exams)
