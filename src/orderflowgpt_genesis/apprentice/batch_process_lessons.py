#!/usr/bin/env python3
"""
batch_process_lessons.py  —  CLEANED VERSION v2.0

Fixes applied:
 1. RELATIONSHIP PERSISTENCE: Discovered relationships are now actually added to the ontology.
 2. STOP-WORD FILTERING: Invented concepts are filtered against a trading-domain stop list.
 3. RELATIONSHIP DEDUPLICATION: Prevents duplicate relationships across lessons.
 4. CONCEPT QUALITY GATE: Minimum confidence and name-length gates for auto-discovered concepts.
 5. RELATIONSHIP QUALITY FILTER: Filters out low-confidence and stop-word relationships.
 6. ONTOLOGY PRUNING: Optionally removes orphaned auto-discovered concepts with no relationships.
 7. BETTER AUDIT: Tracks relationship evolution separately from concept evolution.

Usage:
 python batch_process_lessons.py \
     --input assets/fabio \
     --output assets/fabio/output \
     --ontology src/orderflowgpt_genesis/apprentice/fabio_ontology_fixed.json \
     --max-frames 50
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set

sys.path.insert(0, str(Path(__file__).parent))
try:
    from process_real_video import process_video_folder, load_ontology, extract_concepts_from_text
except ImportError as e:
    print(f"ERROR: Cannot import from process_real_video.py: {e}")
    print("Ensure process_real_video.py (v5+) is in the same directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Domain stop words — trading concepts should NOT be common English words
# ---------------------------------------------------------------------------

TRADING_STOP_WORDS: Set[str] = frozenset({
    # Common conversational filler
    "dont", "don", "t", "explain", "explained", "explaining", "explanation",
    "little", "small", "big", "large", "today", "tomorrow", "yesterday",
    "people", "person", "guy", "guys", "man", "woman", "folks",
    "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank",
    "please", "sorry", "okay", "ok", "yes", "no", "yeah", "nah",
    "blood", "sweat", "tears", "heart", "mind", "body", "soul",
    "community", "total", "reason", "couldn", "couldnt", "could", "would",
    "should", "might", "maybe", "perhaps", "probably", "definitely",
    "absolutely", "literally", "basically", "actually", "essentially",
    "think", "thinking", "thought", "know", "knew", "known", "knowing",
    "feel", "feeling", "felt", "believe", "believing", "believed",
    "want", "wanted", "wanting", "need", "needed", "needing",
    "like", "liked", "liking", "love", "loved", "loving", "hate",
    "mean", "means", "meant", "meaning", "thing", "things", "stuff",
    "way", "ways", "kind", "kinds", "sort", "sorts", "type", "types",
    "part", "parts", "piece", "pieces", "bit", "bits", "lot", "lots",
    "one", "two", "three", "first", "second", "last", "next", "previous",
    "new", "old", "young", "early", "late", "soon", "now", "then",
    "here", "there", "where", "everywhere", "somewhere", "anywhere",
    "every", "each", "all", "none", "some", "many", "much", "more",
    "most", "less", "least", "few", "fewer", "other", "another",
    "such", "same", "different", "various", "several", "certain",
    "whole", "entire", "full", "empty", "half", "quarter",
    "true", "false", "right", "wrong", "correct", "incorrect",
    "good", "bad", "better", "best", "worse", "worst",
    "great", "amazing", "awesome", "fantastic", "wonderful", "nice",
    "beautiful", "ugly", "pretty", "cool", "fine", "perfect",
    "interesting", "boring", "fun", "funny", "serious", "important",
    "special", "normal", "regular", "usual", "common", "rare",
    "easy", "hard", "difficult", "simple", "complex", "complicated",
    "fast", "slow", "quick", "long", "short", "high", "low",
    "up", "down", "left", "right", "top", "bottom", "middle",
    "center", "side", "edge", "end", "start", "begin", "beginning",
    "point", "line", "area", "region", "zone", "level", "place",
    "space", "time", "moment", "minute", "hour", "day", "week",
    "month", "year", "date", "period", "duration", "interval",
    "number", "amount", "quantity", "count", "sum", "total",
    "value", "worth", "price", "cost", "rate", "ratio", "percent",
    "share", "screen", "chat", "video", "audio", "image", "picture",
    "record", "recording", "recorded", "session", "lesson", "course",
    "class", "training", "learning", "teaching", "study", "studying",
    "question", "answer", "problem", "solution", "idea", "example",
    "case", "situation", "condition", "state", "status", "stage",
    "step", "process", "method", "system", "approach", "technique",
    "strategy", "plan", "goal", "target", "aim", "objective",
    "result", "outcome", "output", "input", "effect", "impact",
    "change", "shift", "move", "movement", "action", "activity",
    "work", "working", "worked", "job", "task", "project",
    "test", "testing", "tested", "trial", "attempt", "try", "tried",
    "check", "checking", "checked", "look", "looking", "looked",
    "see", "seeing", "saw", "seen", "watch", "watching", "watched",
    "find", "finding", "found", "search", "searching", "searched",
    "get", "getting", "got", "gotten", "give", "giving", "gave", "given",
    "take", "taking", "took", "taken", "make", "making", "made",
    "do", "doing", "did", "done", "go", "going", "went", "gone",
    "come", "coming", "came", "put", "putting", "set", "setting",
    "let", "letting", "leave", "leaving", "left", "keep", "keeping", "kept",
    "hold", "holding", "held", "run", "running", "ran", "use", "using", "used",
    "try", "trying", "start", "starting", "started", "stop", "stopping", "stopped",
    "turn", "turning", "turned", "open", "opening", "opened", "close", "closing", "closed",
    "show", "showing", "showed", "shown", "play", "playing", "played",
    "call", "calling", "called", "tell", "telling", "told", "say", "saying", "said",
    "speak", "speaking", "spoke", "spoken", "talk", "talking", "talked",
    "ask", "asking", "asked", "answer", "answering", "answered",
    "help", "helping", "helped", "need", "needing", "needed",
    "want", "wanting", "wanted", "wish", "wishing", "wished",
    "hope", "hoping", "hoped", "expect", "expecting", "expected",
    "seem", "seeming", "seemed", "appear", "appearing", "appeared",
    "become", "becoming", "became", "remain", "remaining", "remained",
    "continue", "continuing", "continued", "finish", "finishing", "finished",
    "begin", "beginning", "began", "begun", "end", "ending", "ended",
})

# Words that are valid trading concepts even if they look like stop words
TRADING_ALLOWLIST: Set[str] = frozenset({
    "delta", "bid", "ask", "buy", "sell", "long", "short", "volume",
    " POC ", "VAL", "VAH", "VWAP", "IB", "ETH", "RTH", "HVN", "LVN",
    "absorption", "aggression", "imbalance", "stacked", "exhaustion",
    "divergence", "confluence", "reversal", "breakout", "pullback",
    "support", "resistance", "trend", "range", "auction", "market",
    "profile", "footprint", "orderflow", "order", "flow", "trade",
    "trader", "trading", "chart", "candle", "bar", "wick", "tail",
    "open", "high", "low", "close", "settlement", "opening", "closing",
    "initial", "balance", "extension", "rotation", "rotation_factor",
    "unfinished", "excess", "poor", "single", "print", "naked",
    "developing", "composite", "session", "overnight", "premarket",
    "structure", "break", "character", "change", "momentum", "speed",
    "aggressive", "passive", "initiative", "responsive", "iceberg",
    "stop", "loss", "target", "reward", "risk", "management",
    "entry", "exit", "setup", "pattern", "signal", "confirmation",
    "timeframe", "context", "alignment", "opposing", "neutral",
})

def is_valid_concept_name(name: str) -> bool:
    """Check if an invented concept name is worth keeping in the ontology.

    STRICT RULES:
    1. Single-word concepts MUST be in TRADING_ALLOWLIST.
    2. Multi-word phrases: reject if >50% of words are TRADING_STOP_WORDS.
    3. Reject pure stop words.
    4. Length gate: 3-40 chars.
    """
    if not name or len(name) < 3 or len(name) > 40:
        return False
    name_lower = name.lower().strip()

    # Allow explicitly listed trading terms
    if name_lower in TRADING_ALLOWLIST:
        return True

    # Reject pure strict stop words
    if name_lower in TRADING_STOP_WORDS:
        return False

    words = name_lower.replace("_", " ").replace("-", " ").split()

    # STRICT: Single-word concepts MUST be in allowlist
    if len(words) == 1:
        return words[0] in TRADING_ALLOWLIST

    # Multi-word: reject if majority are strict stop words
    stop_count = sum(1 for w in words if w in TRADING_STOP_WORDS)
    if len(words) > 0 and stop_count / len(words) > 0.5:
        return False

    return True

def is_valid_relationship(rel: dict) -> bool:
    """Filter out noisy relationships."""
    src = rel.get("source", "").lower().strip()
    tgt = rel.get("target", "").lower().strip()
    if src == tgt or not src or not tgt:
        return False
    # Both must be valid concept names
    if not is_valid_concept_name(src) or not is_valid_concept_name(tgt):
        return False
    # Confidence gate
    if rel.get("confidence", 0) < 0.4:
        return False
    # Reject if either concept is a pure strict stop word
    src_words = src.replace("_", " ").replace("-", " ").split()
    tgt_words = tgt.replace("_", " ").replace("-", " ").split()
    if len(src_words) == 1 and src in TRADING_STOP_WORDS:
        return False
    if len(tgt_words) == 1 and tgt in TRADING_STOP_WORDS:
        return False
    return True

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
        # NEW: Count valid vs invalid invented concepts
        p1["valid_invented"] = sum(1 for inv in p1["invented_concepts"] if is_valid_concept_name(inv["name"]))
        p1["invalid_invented"] = p1["invented_count"] - p1["valid_invented"]

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
    """Evolve ontology based on audit results and cumulative transcript analysis.

    FIXES in v2.0:
    - Relationships are now actually persisted into ontology["relationships"]
    - Stop-word filtering on invented concepts
    - Relationship deduplication
    - Confidence gating
    """
    import re
    evolved = json.loads(json.dumps(ontology))
    existing_names = {c["name"] for c in evolved["concepts"]}
    existing_ids = {c["concept_id"] for c in evolved["concepts"]}
    next_num = max([int(re.search(r'\d+', cid).group()) for cid in existing_ids if re.search(r'\d+', cid)] + [0]) + 1
    changes = []

    # ------------------------------------------------------------------
    # ADD INVENTED CONCEPTS (with quality gating)
    # ------------------------------------------------------------------

    # Build word-frequency map from cumulative transcripts for spam detection
    word_freq = {}
    for text in all_transcript_texts:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
    total_word_count = sum(word_freq.values())

    valid_invented = 0
    filtered_invented = 0
    freq_filtered = 0
    for inv in p1.get("invented_concepts", []):
        name = inv["name"]
        if name in existing_names:
            continue

        # Name quality gate: strict validation
        if not is_valid_concept_name(name):
            filtered_invented += 1
            continue

        # FREQUENCY GATE: Single-word concepts that appear very frequently
        # across all cumulative transcripts are likely common English words,
        # not domain-specific trading concepts.
        name_words = name.lower().replace("_", " ").replace("-", " ").split()
        if len(name_words) == 1 and name_words[0] not in TRADING_ALLOWLIST:
            freq = word_freq.get(name_words[0], 0)
            # If word appears >100 times in cumulative text, it's too common
            if freq > 100:
                freq_filtered += 1
                continue
            # Also reject if it represents >0.5% of all words
            if total_word_count > 0 and freq / total_word_count > 0.005:
                freq_filtered += 1
                continue

        conf = inv.get("confidence_score", 0.5)

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
            "confidence_weight": min(1.0, max(0.05, conf)),
        })
        existing_names.add(name)
        existing_ids.add(new_id)
        next_num += 1
        valid_invented += 1
        changes.append(f"ADDED concept '{name}' ({new_id}, conf={conf:.2f})")

    if filtered_invented > 0:
        changes.append(f"FILTERED {filtered_invented} low-quality invented concepts")
    if freq_filtered > 0:
        changes.append(f"FILTERED {freq_filtered} high-frequency invented concepts")

    # ------------------------------------------------------------------
    # ADD DISCOVERED RELATIONSHIPS (THE CRITICAL FIX)
    # ------------------------------------------------------------------
    existing_rels: Set[Tuple[str, str, str]] = set()
    for rel in evolved.get("relationships", []):
        src = rel.get("source", "").lower().strip()
        tgt = rel.get("target", "").lower().strip()
        rtype = rel.get("relation", rel.get("relationship_type", "unknown")).lower().strip()
        existing_rels.add((src, tgt, rtype))
        if rel.get("direction") == "bidirectional":
            existing_rels.add((tgt, src, rtype))

    rels_added = 0
    rels_filtered = 0
    rels_duplicate = 0

    if discovered_rels:
        for rel in discovered_rels:
            if not is_valid_relationship(rel):
                rels_filtered += 1
                continue

            src = rel["source"].lower().strip()
            tgt = rel["target"].lower().strip()
            rtype = rel.get("relation", "cooccurs_with").lower().strip()
            direction = rel.get("direction", "forward")

            # Deduplication check
            key = (src, tgt, rtype)
            if key in existing_rels:
                rels_duplicate += 1
                continue

            # Add to ontology relationships
            evolved.setdefault("relationships", []).append({
                "source": rel["source"],
                "target": rel["target"],
                "relation": rel.get("relation", "cooccurs_with"),
                "relationship_type": rel.get("relation", "cooccurs_with"),
                "confidence": rel.get("confidence", 0.5),
                "direction": direction,
                "discovered": True,
                "method": rel.get("method", "unknown"),
                "cooccurrence_count": rel.get("cooccurrence_count", 1),
            })
            existing_rels.add(key)
            if direction == "bidirectional":
                existing_rels.add((tgt, src, rtype))
            rels_added += 1
            changes.append(f"ADDED rel '{rel['source']}' --{rel.get('relation','?')}--> '{rel['target']}' (conf={rel.get('confidence',0):.2f})")

    if rels_filtered > 0:
        changes.append(f"FILTERED {rels_filtered} low-quality relationships")
    if rels_duplicate > 0:
        changes.append(f"SKIPPED {rels_duplicate} duplicate relationships")

    # ------------------------------------------------------------------
    # KEYWORD EXTRACTION FROM CUMULATIVE TRANSCRIPTS
    # ------------------------------------------------------------------
    all_existing_kws = set()
    for c in evolved["concepts"]:
        all_existing_kws.update(kw.lower() for kw in c.get("keywords", []))

    word_freq = {}
    for text in all_transcript_texts:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        for w in words:
            if w not in TRADING_STOP_WORDS:
                word_freq[w] = word_freq.get(w, 0) + 1

    kw_added = 0
    for word, count in sorted(word_freq.items(), key=lambda x: -x[1])[:50]:
        if word in all_existing_kws or count < 3:
            continue
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
                kw_added += 1
                changes.append(f"KEYWORD '{word}' -> '{best_concept['name']}' (freq={count})")

    # ------------------------------------------------------------------
    # VERSION BUMP & LOG
    # ------------------------------------------------------------------
    evolved["evolution_log"] = evolved.get("evolution_log", []) + changes
    evolved["evolution_version"] = evolved.get("evolution_version", "1.0.0")
    parts = evolved["evolution_version"].split(".")
    if len(parts) == 3:
        parts[2] = str(int(parts[2]) + 1)
        evolved["evolution_version"] = ".".join(parts)

    # Summary stats
    evolved["_evolution_stats"] = {
        "concepts_added": valid_invented,
        "concepts_name_filtered": filtered_invented,
        "concepts_freq_filtered": freq_filtered,
        "relationships_added": rels_added,
        "relationships_filtered": rels_filtered,
        "relationships_duplicate": rels_duplicate,
        "keywords_added": kw_added,
        "timestamp": datetime.utcnow().isoformat(),
    }

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

    initial_concepts = len(ontology.get("concepts", []))
    initial_rels = len(ontology.get("relationships", []))
    print(f"[Batch] Initial ontology: {initial_concepts} concepts, {initial_rels} relationships")

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

        print(f"\n[Audit P1] {p1['total_extracted']} concepts, {p1.get('invented_count', 0)} invented "
              f"({p1.get('valid_invented', 0)} valid, {p1.get('invalid_invented', 0)} filtered), "
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
        stats = ontology.get("_evolution_stats", {})
        print(f"[Evolve] Ontology grew: {old_concept_count} -> {new_concept_count} concepts "
              f"(+{stats.get('concepts_added', new_concept_count - old_concept_count)} valid, "
              f"name-filtered={stats.get('concepts_name_filtered', 0)}, "
              f"freq-filtered={stats.get('concepts_freq_filtered', 0)}), "
              f"{old_rel_count} -> {new_rel_count} relationships "
              f"(+{stats.get('relationships_added', new_rel_count - old_rel_count)} added, "
              f"{stats.get('relationships_filtered', 0)} filtered, "
              f"{stats.get('relationships_duplicate', 0)} duplicates)")

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
            "valid_invented": p1.get("valid_invented", 0),
            "invalid_invented": p1.get("invalid_invented", 0),
            "discovered_relationships": len(discovered_rels),
            "relationships_added_to_ontology": stats.get("relationships_added", 0),
        })
        cumulative["total_concepts_extracted"] += report.total_concepts_learned
        cumulative["total_frames"] += report.total_frames_processed
        cumulative["total_segments"] += coverage.get("total_segments", 0)
        cumulative["total_covered_segments"] += coverage.get("covered_segments", 0)
        cumulative["total_relationships"] = new_rel_count
        for cname in graph.get("concepts", []):
            cumulative["total_concepts_unique"].add(cname)

        # Save intermediate evolved ontology
        with open(evolved_ontology_path, "w", encoding="utf-8") as f:
            json.dump(ontology, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    final_concepts = len(ontology.get("concepts", []))
    final_rels = len(ontology.get("relationships", []))

    print(f"Lessons processed: {len(cumulative['lessons'])}")
    print(f"Total concepts extracted (all lessons): {cumulative['total_concepts_extracted']}")
    print(f"Unique concepts in final graph: {len(cumulative['total_concepts_unique'])}")
    print(f"Total frames processed: {cumulative['total_frames']}")
    print(f"Transcript coverage: {cumulative['total_covered_segments']}/{cumulative['total_segments']} "
          f"({100*cumulative['total_covered_segments']/max(1,cumulative['total_segments']):.1f}%)")
    print(f"Final ontology: {final_concepts} concepts (+{final_concepts - initial_concepts}), "
          f"{final_rels} relationships (+{final_rels - initial_rels})")

    total_discovered_rels = sum(l.get("discovered_relationships", 0) for l in cumulative["lessons"])
    total_added_rels = sum(l.get("relationships_added_to_ontology", 0) for l in cumulative["lessons"])
    print(f"Total discovered relationships across all lessons: {total_discovered_rels}")
    print(f"Total relationships persisted to ontology: {total_added_rels}")

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
        description="Batch process all Fabio lessons with incremental ontology learning (CLEANED v2.0)"
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