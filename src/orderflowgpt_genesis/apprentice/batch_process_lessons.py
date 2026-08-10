#!/usr/bin/env python3
"""
batch_process_lessons.py

Process ALL Fabio lessons in one go with incremental ontology learning.

Usage:
    python batch_process_lessons.py \
        --input assets/fabio/videos \
        --output assets/fabio/output \
        --ontology src/orderflowgpt_genesis/apprentice/fabio_ontology.json \
        --max-frames 50

Directory structure expected:
    assets/fabio/videos/
        Lesson01.mp4
        Lesson02.mp4
        ...
    assets/fabio/transcripts/
        Lesson01.txt
        Lesson02.txt
        ...

Or per-lesson folders:
    assets/fabio/lessons/
        Lesson01/
            video.mp4
            transcript.txt
        Lesson02/
            ...
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Import core processing function from process_real_video.py
# This assumes process_real_video.py is in the same directory
sys.path.insert(0, str(Path(__file__).parent))
try:
    from process_real_video import process_video_folder, load_ontology, extract_concepts_from_text
except ImportError as e:
    print(f"ERROR: Cannot import from process_real_video.py: {e}")
    print("Ensure process_real_video.py (v5+) is in the same directory.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Lesson discovery
# ---------------------------------------------------------------------------

def discover_lessons(input_dir: Path) -> List[Dict]:
    """Find all lesson video+transcript pairs."""
    lessons = []

    # Strategy 1: Paired files in separate folders
    video_dir = input_dir / "videos" if (input_dir / "videos").exists() else input_dir
    transcript_dir = input_dir / "transcripts" if (input_dir / "transcripts").exists() else input_dir

    video_extensions = {".mp4", ".avi", ".mkv", ".mov"}
    transcript_extensions = {".txt", ".srt", ".json", ".vtt"}

    videos = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions])

    for video in videos:
        stem = video.stem
        # Look for matching transcript
        transcript = None
        for ext in transcript_extensions:
            candidate = transcript_dir / f"{stem}{ext}"
            if candidate.exists():
                transcript = candidate
                break
            # Try lowercase
            candidate = transcript_dir / f"{stem.lower()}{ext}"
            if candidate.exists():
                transcript = candidate
                break

        if transcript:
            lessons.append({
                "name": stem,
                "video": str(video),
                "transcript": str(transcript),
            })
        else:
            print(f"[WARN] No transcript found for {video.name}")

    # Strategy 2: Per-lesson subfolders
    if not lessons:
        for subdir in sorted(input_dir.iterdir()):
            if subdir.is_dir():
                videos = [f for f in subdir.iterdir() if f.suffix.lower() in video_extensions]
                transcripts = [f for f in subdir.iterdir() if f.suffix.lower() in transcript_extensions]
                if videos and transcripts:
                    lessons.append({
                        "name": subdir.name,
                        "video": str(videos[0]),
                        "transcript": str(transcripts[0]),
                    })

    return lessons


# ---------------------------------------------------------------------------
# Audit & evolution (embedded to avoid import dependency)
# ---------------------------------------------------------------------------

def audit_lesson(output_dir: Path, ontology: dict) -> Tuple[dict, dict]:
    """Run P1+P2 audit on lesson output. Returns (p1_results, p2_results)."""
    concepts_path = output_dir / "concepts.json"
    coverage_path = output_dir / "coverage_report.json"

    # P1: Concept audit
    p1 = {"total_extracted": 0, "invented_concepts": [], "duplicates_by_name": {}}
    if concepts_path.exists():
        with open(concepts_path, "r", encoding="utf-8") as f:
            concepts = json.load(f)
        ontology_names = {c["name"] for c in ontology.get("concepts", [])}
        name_counts = {}
        for c in concepts:
            name = c.get("name", "UNKNOWN")
            name_counts[name] = name_counts.get(name, 0) + 1
            if name not in ontology_names:
                p1["invented_concepts"].append({
                    "name": name,
                    "confidence_score": c.get("confidence_score", 0),
                    "definition": c.get("definition", "")[:200],
                })
        p1["total_extracted"] = len(concepts)
        p1["duplicates_by_name"] = {k: v for k, v in name_counts.items() if v > 1}
        p1["invented_count"] = len(p1["invented_concepts"])

    # P2: Coverage analysis
    p2 = {"uncovered_count": 0, "candidate_keywords": {}}
    if coverage_path.exists():
        with open(coverage_path, "r", encoding="utf-8") as f:
            coverage = json.load(f)
        p2["uncovered_count"] = coverage.get("uncovered_segments", 0)
        p2["total_segments"] = coverage.get("total_segments", 0)
        p2["coverage_percent"] = coverage.get("coverage_percent", 0)
        p2["uncovered_examples"] = coverage.get("uncovered_examples", [])[:5]

    return p1, p2


def evolve_ontology(ontology: dict, p1: dict, all_transcript_texts: List[str], discovered_rels: Optional[List[dict]] = None) -> dict:
    """Evolve ontology based on audit results and cumulative transcript analysis."""
    import re
    evolved = json.loads(json.dumps(ontology))
    existing_names = {c["name"] for c in evolved["concepts"]}
    existing_ids = {c["concept_id"] for c in evolved["concepts"]}
    next_num = max([int(re.search(r'\d+', cid).group()) for cid in existing_ids if re.search(r'\d+', cid)] + [0]) + 1
    changes = []

    # Add invented concepts
    for inv in p1.get("invented_concepts", []):
        name = inv["name"]
        if name in existing_names or len(name) < 3:
            continue
        new_id = f"auto_{next_num:03d}"
        while new_id in existing_ids:
            next_num += 1
            new_id = f"auto_{next_num:03d}"

        evolved["concepts"].append({
            "concept_id": new_id,
            "name": name,
            "category": "Auto_Discovered",
            "definition": inv.get("definition", f"Auto-discovered from lesson extraction."),
            "keywords": [name.lower(), name.lower().replace(" ", "_")],
            "visual_markers": [],
            "parent_concepts": [],
            "related_concepts": [],
            "opposite_concepts": [],
            "confidence_weight": min(1.0, inv.get("confidence_score", 0.5)),
        })
        existing_names.add(name)
        existing_ids.add(new_id)
        next_num += 1
        changes.append(f"ADDED '{name}' ({new_id})")

    # Extract new keywords from all cumulative transcript text
    # Find frequently occurring bigrams/trigrams not already in any concept's keywords
    all_existing_kws = set()
    for c in evolved["concepts"]:
        all_existing_kws.update(kw.lower() for kw in c.get("keywords", []))

    word_freq = {}
    for text in all_transcript_texts:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

    # Find high-frequency words not in keywords — suggest for relevant concepts
    for word, count in sorted(word_freq.items(), key=lambda x: -x[1])[:50]:
        if word in all_existing_kws or count < 3:
            continue
        # Find best matching concept by name similarity
        best_concept = None
        best_score = 0
        for c in evolved["concepts"]:
            cname = c["name"].lower()
            score = 0
            if word in cname:
                score = 10
            elif any(word in kw for kw in c.get("keywords", [])):
                score = 5
            if score > best_score:
                best_score = score
                best_concept = c
        if best_concept and best_score >= 5:
            if word not in [k.lower() for k in best_concept.get("keywords", [])]:
                best_concept.setdefault("keywords", []).append(word)
                changes.append(f"KEYWORD '{word}' -> '{best_concept['name']}' (freq={count})")

    evolved["evolution_log"] = evolved.get("evolution_log", []) + changes
    evolved["evolution_version"] = evolved.get("evolution_version", "1.0.0")
    parts = evolved["evolution_version"].split(".")
    if len(parts) == 3:
        parts[2] = str(int(parts[2]) + 1)
        evolved["evolution_version"] = ".".join(parts)

    return evolved


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_all_lessons(
    input_dir: str,
    output_dir: str,
    ontology_path: str,
    max_frames: int = 50,
    mode: str = "transcript-only",
) -> dict:
    """Process all lessons sequentially with incremental learning."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load initial ontology
    ontology = load_ontology(ontology_path)
    if not ontology:
        print("ERROR: Could not load ontology. Exiting.")
        sys.exit(1)

    print(f"[Batch] Initial ontology: {len(ontology.get('concepts', []))} concepts")

    # Discover lessons
    lessons = discover_lessons(input_path)
    if not lessons:
        print(f"ERROR: No lessons found in {input_dir}")
        print("Expected: video files with matching transcript files.")
        sys.exit(1)

    print(f"[Batch] Discovered {len(lessons)} lessons:")
    for ls in lessons:
        print(f"  • {ls['name']}")

    # Cumulative state
    cumulative = {
        "lessons": [],
        "total_concepts_extracted": 0,
        "total_concepts_unique": set(),
        "total_relationships": 0,
        "total_frames": 0,
        "total_segments": 0,
        "total_covered_segments": 0,
        "evolution_history": [],
    }
    all_transcript_texts = []

    # Process each lesson
    for idx, lesson in enumerate(lessons, 1):
        print(f"\n{'='*70}")
        print(f"LESSON {idx}/{len(lessons)}: {lesson['name']}")
        print(f"{'='*70}")

        lesson_output = output_path / lesson["name"]
        lesson_output.mkdir(parents=True, exist_ok=True)

        # Save current evolved ontology for this lesson to use
        evolved_ontology_path = output_path / "evolved_ontology.json"
        with open(evolved_ontology_path, "w", encoding="utf-8") as f:
            json.dump(ontology, f, indent=2, ensure_ascii=False)

        # Process lesson
        try:
            result = process_video_folder(
                video_path=lesson["video"],
                transcript_path=lesson["transcript"],
                output_dir=str(lesson_output),
                lesson_name=lesson["name"],
                ontology_path=str(evolved_ontology_path),
                max_frames=max_frames,
                mode=mode,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {lesson['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Collect transcript text for cumulative analysis
        try:
            with open(lesson["transcript"], "r", encoding="utf-8") as f:
                all_transcript_texts.append(f.read())
        except:
            pass

        # Audit P1 + P2
        p1, p2 = audit_lesson(lesson_output, ontology)

        # Load discovered relationships from this lesson
        discovered_rels = []
        discovered_path = lesson_output / "discovered_relationships.json"
        if discovered_path.exists():
            with open(discovered_path, "r", encoding="utf-8") as f:
                discovered_rels = json.load(f)

        print(f"\n[Audit P1] {p1['total_extracted']} concepts, {p1.get('invented_count', 0)} invented, "
              f"{len(p1.get('duplicates_by_name', {}))} duplicates")
        print(f"[Audit P2] {p2.get('coverage_percent', 0):.1f}% coverage, "
              f"{p2.get('uncovered_count', 0)} uncovered segments")
        print(f"[Audit] Discovered relationships: {len(discovered_rels)}")

        # Evolve ontology for next lesson
        print(f"\n[Evolve] Updating ontology...")
        old_concept_count = len(ontology.get("concepts", []))
        old_rel_count = len(ontology.get("relationships", []))
        ontology = evolve_ontology(ontology, p1, all_transcript_texts, discovered_rels)
        new_concept_count = len(ontology.get("concepts", []))
        new_rel_count = len(ontology.get("relationships", []))
        print(f"[Evolve] Ontology grew: {old_concept_count} -> {new_concept_count} concepts (+{new_concept_count - old_concept_count}), "
              f"{old_rel_count} -> {new_rel_count} relationships (+{new_rel_count - old_rel_count})")

        # Update cumulative
        report = result["report"]
        graph = result["graph"]
        coverage = result["coverage"]

        cumulative["lessons"].append({
            "name": lesson["name"],
            "concepts_extracted": report.total_concepts_learned,
            "frames_processed": report.total_frames_processed,
            "confidence": result["confidence"],
            "coverage_percent": coverage.get("coverage_percent", 0),
            "graph_concepts": graph.get("total_concepts", 0),
            "graph_relationships": graph.get("total_relationships", 0),
            "graph_predefined_rels": graph.get("predefined_relationships", 0),
            "graph_discovered_rels": graph.get("discovered_relationships", 0),
            "invented_concepts": p1.get("invented_count", 0),
            "discovered_relationships": len(discovered_rels),
        })
        cumulative["total_concepts_extracted"] += report.total_concepts_learned
        cumulative["total_frames"] += report.total_frames_processed
        cumulative["total_segments"] += coverage.get("total_segments", 0)
        cumulative["total_covered_segments"] += coverage.get("covered_segments", 0)
        cumulative["total_relationships"] = graph.get("total_relationships", 0)
        # Track unique concept names from graph
        for cname in graph.get("concepts", []):
            cumulative["total_concepts_unique"].add(cname)

        # Save intermediate evolved ontology
        with open(evolved_ontology_path, "w", encoding="utf-8") as f:
            json.dump(ontology, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    initial_ontology = load_ontology(ontology_path)
    final_concepts = len(ontology.get("concepts", []))
    final_rels = len(ontology.get("relationships", []))
    initial_concepts = len(initial_ontology.get("concepts", []))
    initial_rels = len(initial_ontology.get("relationships", []))

    print(f"Lessons processed: {len(cumulative['lessons'])}")
    print(f"Total concepts extracted (all lessons): {cumulative['total_concepts_extracted']}")
    print(f"Unique concepts in final graph: {len(cumulative['total_concepts_unique'])}")
    print(f"Total frames processed: {cumulative['total_frames']}")
    print(f"Transcript coverage: {cumulative['total_covered_segments']}/{cumulative['total_segments']} "
          f"({100*cumulative['total_covered_segments']/max(1,cumulative['total_segments']):.1f}%)")
    print(f"Final ontology: {final_concepts} concepts (+{final_concepts - initial_concepts}), "
          f"{final_rels} relationships (+{final_rels - initial_rels})")

    total_discovered_rels = sum(l.get("discovered_relationships", 0) for l in cumulative["lessons"])
    print(f"Total discovered relationships across all lessons: {total_discovered_rels}")

    # Save cumulative report
    cumulative_path = output_path / "batch_report.json"
    cumulative["total_concepts_unique"] = sorted(cumulative["total_concepts_unique"])
    with open(cumulative_path, "w", encoding="utf-8") as f:
        json.dump(cumulative, f, indent=2, ensure_ascii=False)
    print(f"\nSaved cumulative report: {cumulative_path}")

    # Save final evolved ontology
    final_ontology_path = output_path / "fabio_ontology_final.json"
    with open(final_ontology_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2, ensure_ascii=False)
    print(f"Saved final evolved ontology: {final_ontology_path}")

    return cumulative


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch process all Fabio lessons with incremental ontology learning"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input directory containing videos/transcripts or lesson subfolders")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for all lesson results")
    parser.add_argument("--ontology", required=True,
                        help="Path to initial fabio_ontology.json")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Max frames per lesson")
    parser.add_argument("--mode", choices=["transcript-only", "vision"], default="transcript-only",
                        help="Processing mode")

    args = parser.parse_args()

    process_all_lessons(
        input_dir=args.input,
        output_dir=args.output,
        ontology_path=args.ontology,
        max_frames=args.max_frames,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()