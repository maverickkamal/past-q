"""NotebookLM Markdown Study Set Exporter.

Queries the medical past questions database and generates high-signal,
pedagogically structured Markdown documents optimized for NotebookLM source grounding,
flashcard generation, and student revision.

Usage:
    python -m app.exporters.export_study_set --discipline Anatomy --level 200L
    python -m app.exporters.export_study_set --region "Lower Limb" --category STEEPLECHASE
    python -m app.exporters.export_study_set --topic "Brachial Plexus" --output exports/brachial_plexus.md
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, DATABASE_PATH
from app.curriculum import resolve_course_code
from app.database import get_connection, get_questions_by_filter


def generate_notebooklm_markdown(
    questions: list[dict[str, Any]],
    title: str | None = None,
    filter_desc: str | None = None,
) -> str:
    """Formats an array of structured questions into high-signal NotebookLM Markdown."""
    if not questions:
        return "# Preclinical Medical Study Set\n\nNo questions matched the specified criteria.\n"

    disciplines = sorted({q.get("discipline") for q in questions if q.get("discipline")})
    levels = sorted({q.get("level") for q in questions if q.get("level") and q.get("level") != "UNKNOWN"})
    categories = sorted({q.get("category") for q in questions if q.get("category")})
    regions = sorted({q.get("system_region") for q in questions if q.get("system_region")})
    
    doc_title = title or f"Preclinical Medical Study Set: {', '.join(disciplines)} ({', '.join(levels) or 'All Levels'})"

    lines: list[str] = [
        f"# {doc_title}",
        "",
        "> **NotebookLM Grounding & Study Guide Document**  ",
        f"> Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> Questions Included: **{len(questions)}**  ",
        f"> Disciplines: **{', '.join(disciplines)}** | Levels: **{', '.join(levels) or 'General'}**  ",
        f"> Exam Categories: **{', '.join(categories)}**  ",
        "",
        "---",
        "",
        "## Table of Contents",
    ]

    grouped_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        reg = q.get("system_region") or "General Topics"
        grouped_by_region[reg].append(q)
    for reg, q_list in grouped_by_region.items():
        anchor = reg.lower().replace(" ", "-").replace(",", "").replace("/", "")
        lines.append(f"- [{reg}](#{anchor}) ({len(q_list)} questions)")

    lines.append("")
    lines.append("---")
    lines.append("")

    for reg, q_list in grouped_by_region.items():
        lines.append(f"## {reg}")
        lines.append("")

        for i, q in enumerate(q_list, 1):
            q_num = q.get("question_number", str(i))
            topic = q.get("topic", "General Topic")
            cat = q.get("category", "EXAM")
            sub_type = q.get("sub_type", "")
            exam_id = q.get("exam_id", "")
            course = q.get("course_code") or resolve_course_code(reg, q.get("discipline")) or "N/A"
            marks = f"{q.get('total_marks')} marks" if q.get("total_marks") is not None else "Unstated marks"
            recurrence = q.get("recurrence_count", 1)
            repeat_badge = f" 🔥 **High-Yield Repeat ({recurrence}x)** |" if recurrence > 1 else ""

            lines.append(f"### Q{q_num}: {topic}")
            lines.append(
                f"*Exam: `{exam_id}` | Course: `{course}` | Format: `{cat} ({sub_type})` | {repeat_badge} Marks: `{marks}`*"
            )
            lines.append("")

            stem = q.get("stem_text", "").strip()
            if stem:
                lines.append(f"> {stem}")
                lines.append("")
            items = q.get("items", [])
            if items:
                is_mcq = cat == "OBJECTIVE" or sub_type in ("SBA", "MULTIPLE_TRUE_FALSE")
                if is_mcq:
                    lines.append("**Options:**")
                    for it in items:
                        lbl = it.get("label", "")
                        txt = it.get("text", "")
                        lines.append(f"- **({lbl})** {txt}")
                else:
                    lines.append("**Sub-questions / Mark Allocations:**")
                    for it in items:
                        lbl = it.get("label", "")
                        txt = it.get("text", "")
                        it_marks = f" `[{it.get('marks')} marks]`" if it.get("marks") is not None else ""
                        lines.append(f"- **({lbl})** {txt}{it_marks}")
                lines.append("")

            has_diag = q.get("has_diagram")
            diag_path = q.get("diagram_path")
            if has_diag and diag_path:
                lines.append(f"![Specimen / Visual Diagram for Q{q_num}]({diag_path})")
                lines.append(f"*Figure: Visual specimen asset linked at `{diag_path}`*")
                lines.append("")

            lines.append("---")
            lines.append("")

    lines.append("## Study Instructions for NotebookLM")
    lines.append("1. **Generate Audio Overview**: NotebookLM can generate a medical discussion podcast from this grounding document.")
    lines.append("2. **Create Practice Flashcards**: Ask NotebookLM: *'Generate 10 rapid-recall flashcards covering the subparts in this guide.'*")
    lines.append("3. **Identify High-Yield Topics**: Prompt: *'Summarize the key anatomical and physiological concepts tested across these past questions.'*")
    lines.append("")

    return "\n".join(lines)


def export_study_set(
    discipline: str | None = None,
    level: str | None = None,
    category: str | None = None,
    system_region: str | None = None,
    course_code: str | None = None,
    topic: str | None = None,
    examiner: str | None = None,
    institution: str | None = None,
    min_recurrence: int = 1,
    output_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Queries questions by filter criteria, writes a NotebookLM Markdown document, and returns content."""
    questions = get_questions_by_filter(
        discipline=discipline,
        level=level,
        category=category,
        system_region=system_region,
        course_code=course_code,
        topic=topic,
        examiner=examiner,
        institution=institution,
        db_path=db_path,
    )

    if min_recurrence > 1:
        questions = [q for q in questions if (q.get("recurrence_count") or 1) >= min_recurrence]

    filter_tokens = [t for t in (discipline, level, category, system_region, topic, institution) if t]
    title_suffix = " - ".join(filter_tokens) if filter_tokens else "All Questions"
    title = f"Medical Past Questions Study Guide: {title_suffix}"

    md_content = generate_notebooklm_markdown(questions, title=title)

    if output_path:
        out_file = Path(output_path)
    else:
        exports_dir = BASE_DIR / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "_".join(filter_tokens or ["preclinical_study_set"]).replace(" ", "_").replace("/", "_")
        out_file = exports_dir / f"{safe_name}_NotebookLM.md"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md_content, encoding="utf-8")
    print(f"Exported {len(questions)} questions to: {out_file}")

    return md_content


def main():
    parser = argparse.ArgumentParser(description="NotebookLM Study Set Exporter")
    parser.add_argument("--discipline", type=str, choices=["Anatomy", "Physiology", "Biochemistry"], help="Filter by discipline")
    parser.add_argument("--level", type=str, choices=["200L", "300L"], help="Filter by academic level")
    parser.add_argument("--category", type=str, choices=["ESSAY", "OBJECTIVE", "STEEPLECHASE", "CA_QUIZ"], help="Filter by exam category")
    parser.add_argument("--region", type=str, help="Filter by canonical system region")
    parser.add_argument("--course", type=str, help="Filter by course code, e.g. 'ANA 201a'")
    parser.add_argument("--topic", type=str, help="Filter by topic keyword")
    parser.add_argument("--examiner", type=str, help="Filter by lecturer/examiner")
    parser.add_argument("--institution", type=str, help="Filter by university/college institution (e.g. BUK, ABU, UNILAG)")
    parser.add_argument("--min-recurrence", type=int, default=1, help="Filter for questions repeated at least N times")
    parser.add_argument("--output", type=str, default=None, help="Target markdown output path")
    parser.add_argument("--db", type=str, default=None, help="Custom database path")

    args = parser.parse_args()

    export_study_set(
        discipline=args.discipline,
        level=args.level,
        category=args.category,
        system_region=args.region,
        course_code=args.course,
        topic=args.topic,
        examiner=args.examiner,
        institution=args.institution,
        min_recurrence=args.min_recurrence,
        output_path=args.output,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
