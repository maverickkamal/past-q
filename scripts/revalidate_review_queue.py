"""Re-validates all questions in NEEDS_REVIEW against updated Stage 4 rules.

Upgrades false-positive questions (textual steeplechase stations without visual cues,
and questions where stems can be synthesized from subparts) to APPROVED.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATABASE_PATH
from app.schemas import QuestionItem, StructuredQuestion
from app.stages.stage4_validator import validate_question


def revalidate_review_queue(db_path: str | Path | None = None) -> None:
    target_db = str(db_path or DATABASE_PATH)
    conn = sqlite3.connect(target_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, exam_id, question_number, category, sub_type, discipline, level,
               course_code, system_region, topic, examiner, curriculum_style, stem_text,
               items_json, total_marks, has_diagram, diagram_path, flag_reasons
        FROM questions
        WHERE review_status = 'NEEDS_REVIEW'
    """)
    rows = cursor.fetchall()
    print(f"Found {len(rows)} question(s) currently marked 'NEEDS_REVIEW' in {target_db}.")

    upgraded_count = 0
    retained_count = 0

    for r in rows:
        qid = r["id"]
        raw_items = json.loads(r["items_json"] or "[]")
        items = [QuestionItem(**it) for it in raw_items]

        stem = r["stem_text"] or ""
        # Auto-promote or synthesize stem if missing
        if len(stem.strip()) < 3 and items:
            if len(items) == 1 and len(items[0].text.strip()) >= 3:
                stem = items[0].text.strip()
                items = []
            else:
                cat = r["category"]
                if cat == "STEEPLECHASE":
                    stem = "Answer the following practical/station questions:"
                elif cat == "CA_QUIZ":
                    stem = "Answer the following test questions:"
                elif cat == "ESSAY":
                    stem = "Answer the following questions:"
                else:
                    stem = "Address the following question items:"

        # Re-construct StructuredQuestion
        sq = StructuredQuestion(
            question_number=r["question_number"],
            category=r["category"],
            sub_type=r["sub_type"],
            discipline=r["discipline"],
            level=r["level"],
            course_code=r["course_code"],
            system_region=r["system_region"],
            topic=r["topic"],
            examiner=r["examiner"] or "UNKNOWN",
            curriculum_style=r["curriculum_style"],
            stem_text=stem,
            items=items,
            total_marks=r["total_marks"],
            has_diagram=bool(r["has_diagram"]),
            diagram_path=r["diagram_path"],
        )

        res = validate_question(sq, check_asset_exists=True)

        if res.review_status == "APPROVED":
            new_items_json = json.dumps([it.model_dump() for it in items])
            cursor.execute(
                """
                UPDATE questions
                SET review_status = 'APPROVED',
                    stem_text = ?,
                    items_json = ?,
                    flag_reasons = '[]'
                WHERE id = ?
                """,
                (stem, new_items_json, qid),
            )
            upgraded_count += 1
            print(f"  [UPGRADED -> APPROVED] {qid}: {stem[:55]}...")
        else:
            new_flags_json = json.dumps(res.flag_reasons)
            cursor.execute(
                """
                UPDATE questions
                SET stem_text = ?,
                    flag_reasons = ?
                WHERE id = ?
                """,
                (stem, new_flags_json, qid),
            )
            retained_count += 1
            print(f"  [RETAINED REVIEW] {qid}: {res.flag_reasons}")

    conn.commit()
    conn.close()

    print("\n" + "=" * 55)
    print(f"RE-VALIDATION RUN COMPLETE")
    print(f"Total Reviewed:  {len(rows)}")
    print(f"Upgraded to APPROVED: {upgraded_count}")
    print(f"Retained in REVIEW:   {retained_count}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    revalidate_review_queue()
