#!/usr/bin/env python3
"""
analyze_lesson.py

P1 + P2 analysis for any apprentice output folder.

Usage:
    python analyze_lesson.py --output assets/fabio/output/Lesson01 --ontology fabio_ontology.json
    python analyze_lesson.py --output assets/fabio/output/Lesson01 --fix --ontology-out fabio_ontology_evolved.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def p1_concept_audit(output_dir: Path, ontology: dict) -> dict:
    """P1: Detect duplicates, invented concepts, and naming inconsistencies."""
    concepts_path = output_dir / "concepts.json"
    if not concepts_path.exists():
        return {"error": "concepts.json not found"}

    concepts = load_json(concepts_path)
    ontology_names = {c["name"] for c in ontology.get("concepts", [])}
    ontology_ids = {c["concept_id"] for c in ontology.get("concepts", [])}

    # Check for duplicates by name
    name_counts = Counter(c.get("name", "UNKNOWN") for c in concepts)
    duplicates = {name: count for name, count in name_counts.items() if count > 1}

    # Check for invented concepts (not in ontology)
    invented = []
    for c in concepts:
        name = c.get("name", "")
        if name and name not in ontology_names:
            invented.append({
                "name": name,
                "concept_id": c.get("concept_id", "UNKNOWN"),
                "definition": c.get("definition", "")[:200],
                "confidence_score": c.get("confidence_score", 0),
            })

    # Check for ID collisions
    extracted_ids = {c.get("concept_id", "") for c in concepts}
    id_collisions = extracted_ids & ontology_ids

    # Check naming inconsistencies (similar names, different IDs)
    name_to_ids = defaultdict(list)
    for c in concepts:
        name_to_ids[c.get("name", "")].append(c.get("concept_id", "UNKNOWN"))
    inconsistent = {name: ids for name, ids in name_to_ids.items() if len(ids) > 1}

    return {
        "total_extracted": len(concepts),
        "total_ontology": len(ontology_names),
        "duplicates_by_name": duplicates,
        "invented_concepts": invented,
        "invented_count": len(invented),
        "id_collisions_with_ontology": list(id_collisions),
        "inconsistent_naming": inconsistent,
        "coverage_ratio": round(len(name_counts) / max(1, len(ontology_names)) * 100, 1),
    }


def p2_coverage_analysis(output_dir: Path, ontology: dict) -> dict:
    """P2: Analyze uncovered segments and suggest keyword expansions."""
    coverage_path = output_dir / "coverage_report.json"
    if not coverage_path.exists():
        return {"error": "coverage_report.json not found"}

    coverage = load_json(coverage_path)
    uncovered = coverage.get("uncovered_examples", [])

    # Extract n-grams from uncovered segments that might be new keywords
    candidate_keywords = Counter()
    for seg in uncovered:
        text = seg.get("text_preview", "").lower()
        # Extract capitalized phrases (likely trading terms)
        phrases = re.findall(r'\b[a-z]+(?:\s+[a-z]+){1,3}\b', text)
        for ph in phrases:
            if len(ph) > 8 and ph not in {"the market", "this is", "look at", "going to"}:
                candidate_keywords[ph] += 1

    # Find uncovered segments that contain ontology keywords but weren't matched
    # (indicates matching logic failure)
    missed_matches = []
    all_keywords = set()
    for c in ontology.get("concepts", []):
        all_keywords.update(kw.lower() for kw in c.get("keywords", []))

    for seg in uncovered:
        text = seg.get("text_preview", "").lower()
        matched_kws = [kw for kw in all_keywords if kw in text]
        if matched_kws:
            missed_matches.append({
                "index": seg.get("index"),
                "text_preview": text[:120],
                "matched_keywords": matched_kws,
            })

    return {
        "total_segments": coverage.get("total_segments", 0),
        "uncovered_count": coverage.get("uncovered_segments", 0),
        "uncovered_percent": coverage.get("coverage_percent", 0),
        "uncovered_examples": uncovered[:10],
        "candidate_new_keywords": dict(candidate_keywords.most_common(20)),
        "missed_matches": missed_matches[:10],
        "missed_match_count": len(missed_matches),
    }


def suggest_ontology_fixes(p1: dict, p2: dict, ontology: dict) -> dict:
    """Generate concrete suggestions for ontology evolution."""
    suggestions = []

    # From P1: Add invented concepts
    for inv in p1.get("invented_concepts", [])[:10]:
        suggestions.append({
            "type": "ADD_CONCEPT",
            "name": inv["name"],
            "reason": f"Extracted from lesson with confidence {inv['confidence_score']:.2f} but not in ontology",
            "proposed_definition": inv["definition"],
        })

    # From P2: Expand keywords
    for kw, count in p2.get("candidate_new_keywords", {}).items():
        # Find which concept this keyword might belong to
        best_match = None
        best_score = 0
        for c in ontology.get("concepts", []):
            name_words = set(c["name"].lower().split())
            kw_words = set(kw.split())
            score = len(name_words & kw_words)
            if score > best_score:
                best_score = score
                best_match = c["name"]
        if best_match:
            suggestions.append({
                "type": "EXPAND_KEYWORDS",
                "concept": best_match,
                "keyword": kw,
                "frequency": count,
                "reason": f"Found {count} times in uncovered segments, semantically related to {best_match}",
            })
        else:
            suggestions.append({
                "type": "NEW_KEYWORD_CONCEPT",
                "keyword": kw,
                "frequency": count,
                "reason": f"Found {count} times in uncovered segments, no clear concept match — may need new concept",
            })

    # From missed matches
    for mm in p2.get("missed_matches", [])[:5]:
        suggestions.append({
            "type": "FIX_MATCHING",
            "segment_index": mm["index"],
            "matched_keywords": mm["matched_keywords"],
            "reason": "Segment contains ontology keywords but was not matched — check keyword matching logic",
        })

    return {
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
    }


def evolve_ontology(ontology: dict, p1: dict, p2: dict) -> dict:
    """Apply non-destructive evolution to ontology based on audit results."""
    evolved = json.loads(json.dumps(ontology))
    existing_names = {c["name"] for c in evolved["concepts"]}
    existing_ids = {c["concept_id"] for c in evolved["concepts"]}
    next_id = max([int(re.search(r'\d+', cid).group()) for cid in existing_ids if re.search(r'\d+', cid)] + [0]) + 1

    changes_log = []

    # Add invented concepts
    for inv in p1.get("invented_concepts", []):
        name = inv["name"]
        if name in existing_names:
            continue
        # Generate ID
        prefix = "auto_"
        new_id = f"{prefix}{next_id:03d}"
        while new_id in existing_ids:
            next_id += 1
            new_id = f"{prefix}{next_id:03d}"

        new_concept = {
            "concept_id": new_id,
            "name": name,
            "category": "Auto_Discovered",
            "definition": inv.get("definition", f"Auto-discovered concept from lesson extraction."),
            "keywords": [name.lower(), name.lower().replace(" ", "_")],
            "visual_markers": [],
            "parent_concepts": [],
            "related_concepts": [],
            "opposite_concepts": [],
            "confidence_weight": min(1.0, inv.get("confidence_score", 0.5)),
        }
        evolved["concepts"].append(new_concept)
        existing_names.add(name)
        existing_ids.add(new_id)
        next_id += 1
        changes_log.append(f"ADDED concept '{name}' ({new_id})")

    # Expand keywords
    keyword_additions = defaultdict(list)
    for sugg in suggest_ontology_fixes(p1, p2, ontology).get("suggestions", []):
        if sugg["type"] == "EXPAND_KEYWORDS":
            keyword_additions[sugg["concept"]].append(sugg["keyword"])

    for c in evolved["concepts"]:
        name = c["name"]
        if name in keyword_additions:
            existing_kws = set(k.lower() for k in c.get("keywords", []))
            new_kws = [kw for kw in keyword_additions[name] if kw.lower() not in existing_kws]
            if new_kws:
                c["keywords"] = c.get("keywords", []) + new_kws
                changes_log.append(f"EXPANDED '{name}' keywords: {new_kws}")

    evolved["evolution_log"] = evolved.get("evolution_log", []) + changes_log
    evolved["evolution_version"] = evolved.get("evolution_version", "1.0.0")
    # Bump patch version
    parts = evolved["evolution_version"].split(".")
    if len(parts) == 3:
        parts[2] = str(int(parts[2]) + 1)
        evolved["evolution_version"] = ".".join(parts)

    return evolved


def main():
    parser = argparse.ArgumentParser(description="P1+P2 Audit for Apprentice Lesson Output")
    parser.add_argument("--output", "-o", required=True, help="Lesson output directory")
    parser.add_argument("--ontology", required=True, help="Path to ontology JSON")
    parser.add_argument("--fix", action="store_true", help="Generate evolved ontology")
    parser.add_argument("--ontology-out", help="Path to save evolved ontology (default: overwrite input)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    ontology = load_json(Path(args.ontology))

    print("=" * 70)
    print("P1: CONCEPT AUDIT")
    print("=" * 70)
    p1 = p1_concept_audit(output_dir, ontology)
    print(f"Total extracted concepts: {p1['total_extracted']}")
    print(f"Ontology concepts: {p1['total_ontology']}")
    print(f"Coverage ratio: {p1['coverage_ratio']}%")
    print(f"Duplicate names: {len(p1['duplicates_by_name'])}")
    if p1['duplicates_by_name']:
        for name, count in p1['duplicates_by_name'].items():
            print(f"  [DUP] '{name}' appears {count} times")
    print(f"Invented concepts (not in ontology): {p1['invented_count']}")
    for inv in p1['invented_concepts'][:5]:
        print(f"  [NEW] '{inv['name']}' (conf={inv['confidence_score']:.2f}): {inv['definition'][:80]}")
    if p1['invented_count'] > 5:
        print(f"  ... and {p1['invented_count'] - 5} more")

    print()
    print("=" * 70)
    print("P2: COVERAGE ANALYSIS")
    print("=" * 70)
    p2 = p2_coverage_analysis(output_dir, ontology)
    print(f"Total segments: {p2['total_segments']}")
    print(f"Uncovered: {p2['uncovered_count']} ({100 - p2['uncovered_percent']:.1f}% covered)")
    print(f"Missed matches (has keywords but not detected): {p2['missed_match_count']}")
    if p2['missed_matches']:
        print("  Examples:")
        for mm in p2['missed_matches'][:3]:
            print(f"    [{mm['segment_index']}] {mm['text_preview'][:60]}...")
            print(f"      Keywords found: {mm['matched_keywords']}")
    print(f"Candidate new keywords from uncovered text: {len(p2['candidate_new_keywords'])}")
    for kw, count in list(p2['candidate_new_keywords'].items())[:10]:
        print(f"  '{kw}': {count}")

    print()
    print("=" * 70)
    print("SUGGESTIONS")
    print("=" * 70)
    sugg = suggest_ontology_fixes(p1, p2, ontology)
    print(f"Total suggestions: {sugg['suggestion_count']}")
    for s in sugg['suggestions'][:10]:
        print(f"  [{s['type']}] {s.get('name', s.get('concept', s.get('keyword', '')))}")
        print(f"    {s['reason']}")

    if args.fix:
        print()
        print("=" * 70)
        print("EVOLVING ONTOLOGY")
        print("=" * 70)
        evolved = evolve_ontology(ontology, p1, p2)
        out_path = Path(args.ontology_out or args.ontology)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evolved, f, indent=2, ensure_ascii=False)
        print(f"Saved evolved ontology: {out_path}")
        print(f"Version: {evolved.get('evolution_version', 'unknown')}")
        print(f"Changes: {len(evolved.get('evolution_log', []))}")
        for log in evolved.get("evolution_log", [])[-10:]:
            print(f"  {log}")


if __name__ == "__main__":
    main()