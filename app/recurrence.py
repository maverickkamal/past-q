"""Cross-Year Question Recurrence Matcher & High-Yield Analysis Engine.

Analyzes questions in the SQLite database to identify recurring questions across
different academic sessions, exams, and years. Assigns recurrence clusters,
updates database recurrence counts, and generates a High-Yield Recurrence Report.

Usage:
    python -m app.recurrence --run
    python -m app.recurrence --report
    python -m app.recurrence --threshold 0.75
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, DATABASE_PATH
from app.database import get_connection, update_recurrence_info
from app.tools.typesafe_matcher import TypeSafeRecurrenceJudge


def normalize_question_text(stem: str, items: list[dict[str, Any]] | None = None) -> str:
    """Cleans and standardizes question text for robust semantic comparison."""
    combined = stem or ""
    if items:
        item_texts = [it.get("text", "") for it in items if it.get("text")]
        combined += " " + " ".join(item_texts)

    text = combined.lower()

    boilerplate = [
        r"\b(write\s+(a\s+)?(brief\s+)?(short\s+)?notes?\s+(on|about)?)\b",
        r"\b(discuss\s+(the\s+)?(briefly\s+)?)\b",
        r"\b(describe\s+(the\s+)?(briefly\s+)?)\b",
        r"\b(explain\s+(the\s+)?(briefly\s+)?)\b",
        r"\b(list\s+(out\s+)?(the\s+)?)\b",
        r"\b(name\s+the)\b",
        r"\b(outline\s+the)\b",
        r"\b(what\s+is\s+the|what\s+are\s+the)\b",
        r"\b(with\s+the\s+aid\s+of\s+(a\s+)?(labeled\s+)?diagram)\b",
        r"\b(add\s+a\s+note\s+on)\b",
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    stopwords = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "with", "for", "by", "at", "about"}
    tokens = [t for t in text.split() if len(t) > 2 and t not in stopwords]
    return " ".join(tokens)


def compute_similarity(text_a: str, text_b: str) -> float:
    """Computes hybrid similarity between two normalized question texts."""
    if not text_a or not text_b:
        return 0.0

    if text_a == text_b:
        return 1.0

    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    seq_ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()

    return (0.6 * seq_ratio) + (0.4 * jaccard)


class DisjointSet:
    """Disjoint-set / Union-Find structure for clustering repeating questions."""
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def run_recurrence_matcher(
    similarity_threshold: float = 0.72,
    db_path: str | Path | None = None,
    use_typesafe: bool = True,
    typesafe_threshold: float = 0.80,
) -> dict[str, list[dict[str, Any]]]:
    """Scans all questions in the database, clusters duplicates, and updates recurrence counts.

    Uses a hybrid two-tier approach:
    1. Lexical fast-path (SequenceMatcher + Jaccard) for obvious duplicates (sim >= threshold).
    2. Semantic judgment via TypeSafe AI's Jev model for candidate pairs in the semantic
       band (0.15 <= sim < threshold, or shared topic/system_region), bridging BMAS didactic
       and CCMAS clinical vignette questions.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = """
    SELECT q.id, q.exam_id, q.question_number, q.discipline, q.level, 
           q.course_code, q.system_region, q.topic, q.category, q.curriculum_style,
           q.stem_text, q.items_json, e.academic_year, e.paper_title
    FROM questions q
    JOIN exams e ON q.exam_id = e.id
    WHERE q.review_status = 'APPROVED'
    """
    rows = [dict(r) for r in cursor.execute(query).fetchall()]
    conn.close()

    print(f"Loaded {len(rows)} approved questions from database for recurrence analysis.")

    judge = TypeSafeRecurrenceJudge(probability_threshold=typesafe_threshold) if use_typesafe else None
    if judge and judge.is_configured:
        print(f"TypeSafe AI Jev System One engine ACTIVE (model: {judge.model}, threshold: {typesafe_threshold}).")
    elif use_typesafe:
        print("TypeSafe AI engine unconfigured (TYPESAFE_API_KEY missing) - running in lexical mode.")

    by_discipline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_discipline[r["discipline"]].append(r)

    uf = DisjointSet()
    match_pairs: list[tuple[str, str, float]] = []

    for discipline, q_list in by_discipline.items():
        print(f"Comparing {len(q_list)} {discipline} questions...")
        for i in range(len(q_list)):
            q1 = q_list[i]
            items1 = q1.get("items_json")
            import json
            parsed_items1 = json.loads(items1) if items1 else []
            norm1 = normalize_question_text(q1.get("stem_text", ""), parsed_items1)

            for j in range(i + 1, len(q_list)):
                q2 = q_list[j]
                if q1["exam_id"] == q2["exam_id"]:
                    continue

                items2 = q2.get("items_json")
                parsed_items2 = json.loads(items2) if items2 else []
                norm2 = normalize_question_text(q2.get("stem_text", ""), parsed_items2)

                sim = compute_similarity(norm1, norm2)
                is_match = False
                match_score = sim

                if sim >= similarity_threshold:
                    is_match = True
                elif judge and judge.is_configured:
                    is_same_topic = bool(
                        q1.get("topic")
                        and q1.get("topic") == q2.get("topic")
                        and q1.get("topic") not in ("General", "Unknown", "None", "")
                    )
                    if sim >= 0.25 or is_same_topic:
                        decision = judge.evaluate_pair(q1, q2)
                        if decision.is_recurrence:
                            is_match = True
                            match_score = decision.probability

                if is_match:
                    uf.union(q1["id"], q2["id"])
                    match_pairs.append((q1["id"], q2["id"], round(match_score, 3)))

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    question_map = {r["id"]: r for r in rows}

    for r in rows:
        q_id = r["id"]
        root = uf.find(q_id)
        clusters[root].append(r)

    repeating_clusters = {root: members for root, members in clusters.items() if len(members) >= 2}

    print(f"\nFound {len(match_pairs)} matching question pairs forming {len(repeating_clusters)} distinct recurrence clusters!")
    cluster_idx = 1
    for root, members in repeating_clusters.items():
        count = len(members)
        first_member = members[0]
        disc = first_member.get("discipline", "MED")[:3].upper()
        slug = re.sub(r"[^a-zA-Z0-9]", "_", first_member.get("topic", "TOPIC")[:15]).upper()
        cluster_id = f"REC_{disc}_{slug}_{cluster_idx:03d}"

        for m in members:
            update_recurrence_info(
                question_id=m["id"],
                cluster_id=cluster_id,
                count=count,
                db_path=db_path,
            )
        cluster_idx += 1

    return repeating_clusters


def generate_recurrence_report(
    clusters: dict[str, list[dict[str, Any]]],
    output_path: str | Path | None = None,
) -> str:
    """Generates an executive High-Yield Recurrence Report Markdown document."""
    lines: list[str] = [
        "# Preclinical Medical Past Questions: High-Yield Recurrence Report",
        "",
        "> **Exam Intelligence & Recurrence Analysis**  ",
        f"> Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> Recurrence Clusters Identified: **{len(clusters)}**  ",
        "> Criteria: Cross-session semantic similarity $\\ge 0.72$  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "Questions in this report have appeared repeatedly across multiple preclinical examinations and academic sessions. "
        "They represent the core, non-negotiable medical concepts most heavily favored by university examiners.",
        "",
        "---",
        "",
    ]

    sorted_clusters = sorted(clusters.values(), key=lambda c: len(c), reverse=True)

    cluster_num = 1
    for members in sorted_clusters:
        count = len(members)
        m0 = members[0]
        topic = m0.get("topic", "General Topic")
        disc = m0.get("discipline", "Discipline")
        region = m0.get("system_region", "Region")
        sessions = sorted({m.get("academic_year") for m in members if m.get("academic_year") and m.get("academic_year") != "UNKNOWN"})
        session_str = ", ".join(sessions) if sessions else "Multiple Sessions"

        lines.append(f"## High-Yield Pattern #{cluster_num}: {topic} ({count}x Repeats)")
        lines.append(f"**Discipline:** `{disc}` | **System/Region:** `{region}` | **Academic Sessions:** `{session_str}`")
        lines.append("")
        lines.append("### Occurrences Across Past Papers:")

        for m in members:
            exam_title = m.get("paper_title", m.get("exam_id"))
            year = m.get("academic_year", "N/A")
            q_num = m.get("question_number", "?")
            stem = m.get("stem_text", "").strip()
            lines.append(f"- **[{year}] {exam_title} (Q{q_num})**")
            if stem:
                lines.append(f"  > *\"{stem}\"*")
            lines.append("")

        lines.append("---")
        lines.append("")
        cluster_num += 1

    report_content = "\n".join(lines)

    target_file = Path(output_path) if output_path else BASE_DIR / "exports" / "high_yield_recurrence_report.md"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(report_content, encoding="utf-8")
    print(f"Recurrence report saved to: {target_file}")

    return report_content


def main():
    parser = argparse.ArgumentParser(description="Cross-Year Past Question Recurrence Matcher")
    parser.add_argument("--run", action="store_true", help="Run recurrence matching and update database counts")
    parser.add_argument("--report", action="store_true", help="Generate Markdown recurrence report")
    parser.add_argument("--threshold", type=float, default=0.72, help="Similarity threshold ratio (0.0 to 1.0)")
    parser.add_argument("--use-jev", action="store_true", default=True, help="Enable TypeSafe Jev System One semantic matching")
    parser.add_argument("--no-jev", action="store_false", dest="use_jev", help="Disable TypeSafe Jev (lexical only)")
    parser.add_argument("--jev-threshold", type=float, default=0.80, help="Probability threshold for TypeSafe Jev recurrence (0.0 to 1.0)")
    parser.add_argument("--db", type=str, default=None, help="Custom database path")
    parser.add_argument("--output", type=str, default=None, help="Target markdown report path")

    args = parser.parse_args()

    do_run = args.run or (not args.run and not args.report)
    do_report = args.report or (not args.run and not args.report)

    clusters = {}
    if do_run:
        clusters = run_recurrence_matcher(
            similarity_threshold=args.threshold,
            db_path=args.db,
            use_typesafe=args.use_jev,
            typesafe_threshold=args.jev_threshold,
        )

    if do_report:
        if not clusters:
            clusters = run_recurrence_matcher(
                similarity_threshold=args.threshold,
                db_path=args.db,
                use_typesafe=args.use_jev,
                typesafe_threshold=args.jev_threshold,
            )
        generate_recurrence_report(clusters, output_path=args.output)


if __name__ == "__main__":
    main()
