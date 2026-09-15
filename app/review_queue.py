"""Human-in-the-Loop (HITL) Review Queue CLI.

Provides a lightweight CLI for medical exam curators and lecturers to inspect,
diagnose, and approve questions flagged with 'NEEDS_REVIEW' during ingestion.

Usage:
    python src/review_queue.py --list [--discipline Anatomy] [--exam ANA_2022_STEEPLECHASE]
    python src/review_queue.py --show ANA_2022_STEEPLECHASE_MOCK_QStation_1
    python src/review_queue.py --approve ANA_2022_STEEPLECHASE_MOCK_QStation_1
    python src/review_queue.py --stats
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import (
    approve_question,
    get_database_stats,
    get_question_by_id,
    get_review_queue,
)


def list_review_queue(discipline: str | None = None, exam_id: str | None = None) -> None:
    """Lists all questions currently flagged for human review."""
    items = get_review_queue(discipline=discipline, exam_id=exam_id)
    if not items:
        print("\nAll clear! No questions currently flagged for review.")
        return

    print(f"\n=== PENDING REVIEW QUEUE ({len(items)} items) ===")
    print(f"{'ID':<38} | {'Num':<8} | {'Discipline':<12} | {'Category':<14} | {'Flags'}")
    print("-" * 105)

    for it in items:
        q_id = it.get("id", "UNKNOWN")
        q_num = it.get("question_number", "?")
        disc = f"{it.get('discipline', '?')} ({it.get('level', '?')})"
        cat = it.get("category", "?")
        flags = ", ".join(it.get("flag_reasons_list", []))
        print(f"{q_id:<38} | {q_num:<8} | {disc:<12} | {cat:<14} | {flags}")
    print("-" * 105)
    print("Use: python app/review_queue.py --show <ID> to inspect details\n")


def show_question_detail(question_id: str) -> None:
    """Displays comprehensive inspection details for a single question."""
    q = get_question_by_id(question_id)
    if not q:
        print(f"Error: Question '{question_id}' not found in database.", file=sys.stderr)
        return

    print(f"\n=======================================================")
    print(f"QUESTION ID: {q.get('id')}")
    print(f"=======================================================")
    print(f"Exam ID:          {q.get('exam_id')}")
    print(f"Question Number:  {q.get('question_number')}")
    print(f"Review Status:    {q.get('review_status')}")
    print(f"Discipline/Level: {q.get('discipline')} ({q.get('level', 'UNKNOWN')})")
    print(f"Course Code:      {q.get('course_code', 'N/A')}")
    print(f"System Region:    {q.get('system_region')}")
    print(f"Topic:            {q.get('topic')}")
    print(f"Category/Subtype: {q.get('category')} / {q.get('sub_type')}")
    print(f"Curriculum Style: {q.get('curriculum_style')}")
    print(f"Total Marks:      {q.get('total_marks')}")
    print(f"Diagram Linked:   {bool(q.get('has_diagram'))} ({q.get('diagram_path')})")

    flags = q.get("flag_reasons_list", [])
    if flags:
        print(f"\nDIAGNOSTIC FLAGS ({len(flags)}):")
        for f in flags:
            print(f"  - {f}")

    print(f"\nSTEM TEXT:")
    print(f"  {q.get('stem_text')}")

    items = q.get("items", [])
    if items:
        print(f"\nITEMS / SUBPARTS ({len(items)}):")
        for it in items:
            marks_str = f" [{it.get('marks')} marks]" if it.get("marks") is not None else ""
            print(f"  ({it.get('label')}) {it.get('text')}{marks_str}")

    print(f"=======================================================")
    print(f"To approve: python src/review_queue.py --approve {q.get('id')}\n")


def approve_flagged_question(question_id: str) -> None:
    """Marks a flagged question as APPROVED in the database."""
    success = approve_question(question_id)
    if success:
        print(f"Successfully approved question: {question_id}")
    else:
        print(f"Failed to approve question '{question_id}'. Ensure ID exists.", file=sys.stderr)


def show_stats() -> None:
    """Prints overall ingestion and review queue summary statistics."""
    stats = get_database_stats()
    print("\n=== MEDICAL PAST QUESTIONS DATABASE STATS ===")
    print(f"Total Exam Papers Ingested:  {stats['total_exams']}")
    print(f"Total Questions Ingested:    {stats['total_questions']}")
    print(f"Approved Questions:          {stats['approved_questions']}")
    print(f"Pending Review Questions:    {stats['needs_review_questions']}")
    print("=============================================\n")


def main():
    parser = argparse.ArgumentParser(description="Medical Past Questions Review Queue CLI")
    parser.add_argument("--list", action="store_true", help="List all questions flagged for review")
    parser.add_argument("--show", type=str, metavar="QUESTION_ID", help="Inspect a flagged question in detail")
    parser.add_argument("--approve", type=str, metavar="QUESTION_ID", help="Approve a question and clear flags")
    parser.add_argument("--stats", action="store_true", help="Display overall database and review queue stats")
    parser.add_argument("--discipline", type=str, default=None, help="Filter list by discipline")
    parser.add_argument("--exam", type=str, default=None, help="Filter list by exam_id")

    args = parser.parse_args()

    if args.list:
        list_review_queue(discipline=args.discipline, exam_id=args.exam)
    elif args.show:
        show_question_detail(args.show)
    elif args.approve:
        approve_flagged_question(args.approve)
    elif args.stats:
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
