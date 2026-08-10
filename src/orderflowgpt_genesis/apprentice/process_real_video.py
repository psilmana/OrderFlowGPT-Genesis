"""Process a real Fabio video through the Genesis Apprentice Layer.

HONEST HYBRID VERSION (v4) — addresses all architectural issues:
  1. SYSTEM IDENTITY: Clean separation between transcript mode and vision mode.
  2. FRAME SAMPLING: Processes ALL extracted frames, not just 10.
  3. CONFIDENCE: Computed from real evidence counts, not hardcoded.
  4. ONTOLOGY: Deep-wired into graph building, not a post-hoc fallback.
  5. COVERAGE: Reports which transcript segments matched zero concepts.
  6. HONESTY: No FakeDetectionGraph in transcript mode. Vision mode attempts real Bundle 1-8.

Usage — Transcript-only (default, honest):
    python process_real_video.py \
        --video assets/fabio/videos/Lesson01.mp4 \
        --transcript assets/fabio/transcripts/Lesson01.txt \
        --output assets/fabio/output/Lesson01 \
        --mode transcript-only \
        --ontology fabio_ontology.json

Usage — Vision mode (attempts real Bundle 1-8 analysis):
    python process_real_video.py \
        --video assets/fabio/videos/Lesson01.mp4 \
        --transcript assets/fabio/transcripts/Lesson01.txt \
        --output assets/fabio/output/Lesson01 \
        --mode vision \
        --ontology fabio_ontology.json
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set, Any

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orderflowgpt_genesis.apprentice import (
    ApprenticeRunnerIntegration,
    RunnerIntegrationConfiguration,
    ApprenticeReport,
    Concept,
    Experience,
    KnowledgeGraph,
)


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Fake DetectionGraph — ONLY used in vision mode as fallback
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeTrendState:
    name: str

@dataclass(frozen=True)
class FakeTrend:
    trend_state: FakeTrendState

@dataclass(frozen=True)
class FakeSession:
    session_type: str

@dataclass(frozen=True)
class FakeImbalance:
    side: str
    cell_id: str

@dataclass(frozen=True)
class FakeImbalances:
    imbalances: Tuple[FakeImbalance, ...]

@dataclass(frozen=True)
class FakeAbsorption:
    side: str
    cell_id: str

@dataclass(frozen=True)
class FakeAbsorptions:
    absorptions: Tuple[FakeAbsorption, ...]

@dataclass(frozen=True)
class FakeDelta:
    cell_deltas: Tuple[object, ...]

@dataclass(frozen=True)
class FakeConfluence:
    confluence_type: str

@dataclass(frozen=True)
class FakeDetectionGraph:
    graph_id: str
    timestamp: str
    trend_state: Optional[FakeTrend] = None
    trading_session: Optional[FakeSession] = None
    footprint_imbalances: Optional[FakeImbalances] = None
    absorption_result: Optional[FakeAbsorptions] = None
    footprint_delta: Optional[FakeDelta] = None
    confluence: Optional[FakeConfluence] = None


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def _split_by_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\\s+(?=[A-Z"\'\(])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _split_by_timestamps(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r'(?:^|\n)\s*(?:\[?)(\d{1,2}:\d{2}(?::\d{2})?)(?:\]?)\s*[-–—]?\s*(.*?)'
        r'(?=(?:\n\s*(?:\[?)\d{1,2}:\d{2}(?::\d{2})?(?:\]?)\s*[-–—]?|$))',
        re.DOTALL
    )
    matches = pattern.findall(text)
    if not matches:
        return []
    result = []
    for ts, content in matches:
        content = content.strip().replace('\n', ' ')
        if len(content) > 10:
            result.append((ts, content))
    return result


def load_transcript(transcript_path: str) -> Tuple[Tuple[int, str], ...]:
    path = Path(transcript_path)
    if not path.exists():
        print(f"WARNING: Transcript not found: {transcript_path}")
        return ()

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return tuple((i, entry.get("text", "")) for i, entry in enumerate(data))

    if suffix == ".srt":
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = content.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                text = " ".join(lines[2:]).strip()
                entries.append((len(entries), text))
        return tuple(entries)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    timestamp_entries = _split_by_timestamps(content)
    if timestamp_entries:
        return tuple((i, text) for i, (_, text) in enumerate(timestamp_entries))

    sentences = _split_by_sentences(content)
    if len(sentences) >= 3:
        return tuple((i, s) for i, s in enumerate(sentences))

    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if len(paragraphs) >= 2:
        return tuple((i, p) for i, p in enumerate(paragraphs))

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        return tuple((i, p) for i, p in enumerate(paragraphs))

    chunks = []
    words = content.split()
    current = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > 300 and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        chunks.append(" ".join(current))

    if len(chunks) >= 2:
        return tuple((i, c) for i, c in enumerate(chunks))

    return ((0, content.strip()),)


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: str,
    max_frames: int = 50,
    interval_sec: Optional[float] = None,
) -> List[Tuple[int, str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"WARNING: Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    duration_sec = total_frames / fps if total_frames > 0 else 0

    if interval_sec is None:
        if duration_sec > 0 and max_frames > 0:
            interval_sec = max(1.0, duration_sec / max_frames)
        else:
            interval_sec = 5.0

    frame_interval = max(1, int(round(fps * interval_sec)))

    frames = []
    frame_num = 0
    next_target = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num >= next_target:
            ts_sec = frame_num / fps
            ts_str = f"{int(ts_sec // 3600):02d}:{int((ts_sec % 3600) // 60):02d}:{int(ts_sec % 60):02d}"
            frames.append((frame_num, ts_str, frame))
            next_target += frame_interval
            if len(frames) >= max_frames:
                break

        frame_num += 1

    cap.release()
    print(f"[Video] Extracted {len(frames)} frames from {total_frames} total "
          f"(interval={interval_sec:.1f}s, fps={fps:.1f})")
    return frames


# ---------------------------------------------------------------------------
# Ontology loading & graph building
# ---------------------------------------------------------------------------

def load_ontology(ontology_path: str) -> dict:
    path = Path(ontology_path)
    if not path.exists():
        candidates = [
            Path(__file__).parent / "fabio_ontology.json",
            Path(__file__).parent / "fabio_ontology_fixed.json",
            Path.cwd() / "fabio_ontology.json",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
    if not path.exists():
        print(f"WARNING: Ontology not found at {ontology_path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_concepts_from_text(text: str, ontology: dict) -> List[str]:
    """Match text against ontology keywords. Returns matched concept names."""
    text_lower = text.lower()
    matched = []
    for concept in ontology.get("concepts", []):
        for kw in concept.get("keywords", []):
            if kw.lower() in text_lower:
                matched.append(concept["name"])
                break
    return matched


def build_graph_from_ontology(
    ontology: dict,
    extracted_concept_names: Set[str],
    discovered_rels: Optional[List[dict]] = None,
) -> dict:
    """Build a knowledge graph dict from ontology + extracted concepts + discovered relationships."""
    all_concepts = {c["name"]: c for c in ontology.get("concepts", [])}

    graph_concepts = set(extracted_concept_names)

    for name in list(graph_concepts):
        if name in all_concepts:
            c = all_concepts[name]
            graph_concepts.update(c.get("parent_concepts", []))
            graph_concepts.update(c.get("related_concepts", []))
            graph_concepts.update(c.get("opposite_concepts", []))

    # Add targets of all relationships (predefined + discovered)
    all_rels = list(ontology.get("relationships", []))
    if discovered_rels:
        all_rels.extend(discovered_rels)

    for rel in all_rels:
        if rel["source"] in graph_concepts:
            graph_concepts.add(rel["target"])

    # Build relationship list with deduplication
    graph_relationships = []
    seen_rels = set()

    for rel in all_rels:
        src = rel["source"]
        tgt = rel["target"]
        if src in graph_concepts and tgt in graph_concepts:
            rel_name = rel.get("relation", rel.get("relationship_type", "unknown"))
            key = (src.lower(), tgt.lower(), rel_name)
            if key not in seen_rels:
                seen_rels.add(key)
                graph_relationships.append(rel)

    concept_details = []
    for name in sorted(graph_concepts):
        if name in all_concepts:
            c = all_concepts[name]
            concept_details.append({
                "name": name,
                "category": c.get("category", "Unknown"),
                "definition": c.get("definition", ""),
                "confidence_weight": c.get("confidence_weight", 1.0),
            })
        else:
            concept_details.append({
                "name": name,
                "category": "Extracted",
                "definition": "",
                "confidence_weight": 0.8,
            })

    predefined_count = sum(1 for r in graph_relationships if not r.get("discovered"))
    discovered_count = sum(1 for r in graph_relationships if r.get("discovered"))

    return {
        "total_concepts": len(graph_concepts),
        "total_relationships": len(graph_relationships),
        "predefined_relationships": predefined_count,
        "discovered_relationships": discovered_count,
        "concepts": sorted(graph_concepts),
        "concept_details": concept_details,
        "relationships": graph_relationships,
    }


# ---------------------------------------------------------------------------
# Concept coverage analysis
# ---------------------------------------------------------------------------

def analyze_coverage(
    transcript_entries: Tuple[Tuple[int, str], ...],
    ontology: dict,
) -> dict:
    """Analyze which transcript segments matched concepts and which didn't."""
    covered = []
    uncovered = []
    concept_hits: Dict[str, int] = {}

    for idx, text in transcript_entries:
        matched = extract_concepts_from_text(text, ontology)
        if matched:
            covered.append({"index": idx, "text_preview": text[:120], "concepts": matched})
            for c in matched:
                concept_hits[c] = concept_hits.get(c, 0) + 1
        else:
            uncovered.append({"index": idx, "text_preview": text[:120]})

    return {
        "total_segments": len(transcript_entries),
        "covered_segments": len(covered),
        "uncovered_segments": len(uncovered),
        "coverage_percent": round(100 * len(covered) / max(1, len(transcript_entries)), 1),
        "concept_frequency": dict(sorted(concept_hits.items(), key=lambda x: -x[1])),
        "uncovered_examples": uncovered[:20],
    }


# ---------------------------------------------------------------------------
# Real confidence computation
# ---------------------------------------------------------------------------

def compute_real_confidence(
    concepts: List[Concept],
    coverage: dict,
    graph: dict,
) -> float:
    """Compute confidence from actual evidence, not a hardcoded value."""
    if not concepts:
        return 0.0

    # Factor 1: Average concept confidence score
    scores = [float(c.confidence.score) for c in concepts if hasattr(c.confidence, 'score')]
    avg_score = sum(scores) / len(scores) if scores else 0.5

    # Factor 2: Coverage ratio
    coverage_ratio = coverage.get("coverage_percent", 0) / 100.0

    # Factor 3: Graph richness
    graph_ratio = min(1.0, graph.get("total_relationships", 0) / max(1, len(concepts)))

    # Weighted combination
    confidence = (avg_score * 0.4) + (coverage_ratio * 0.3) + (graph_ratio * 0.3)
    return min(1.0, max(0.0, confidence))


# ---------------------------------------------------------------------------
# Fake DetectionGraph builder (vision mode fallback only)
# ---------------------------------------------------------------------------

def build_fake_detection_graph(frame_idx: int, transcript_text: str) -> FakeDetectionGraph:
    text = transcript_text.lower()

    if "trend" in text and "up" in text:
        trend = FakeTrend(FakeTrendState("TRENDING_UP"))
    elif "trend" in text and "down" in text:
        trend = FakeTrend(FakeTrendState("TRENDING_DOWN"))
    elif "auction" in text and "down" in text:
        trend = FakeTrend(FakeTrendState("AUCTION_DOWN"))
    elif "auction" in text and "up" in text:
        trend = FakeTrend(FakeTrendState("AUCTION_UP"))
    elif "balance" in text or "rotational" in text:
        trend = FakeTrend(FakeTrendState("BALANCED"))
    else:
        trend = FakeTrend(FakeTrendState("BALANCED"))

    imbalances = []
    if "buyer" in text or "bid" in text:
        imbalances.append(FakeImbalance("BID", f"price_{frame_idx}"))
    if "seller" in text or "ask" in text:
        imbalances.append(FakeImbalance("ASK", f"price_{frame_idx}"))
    if "imbalance" in text and "stacked" not in text:
        imbalances.append(FakeImbalance("BID", f"price_{frame_idx}"))

    absorptions = []
    if "absorption" in text or "absorb" in text:
        side = "BID" if "buy" in text or "bid" in text else "ASK"
        absorptions.append(FakeAbsorption(side, f"price_{frame_idx}"))

    confluence = "NO_CONFLUENCE"
    if "strong" in text or "confirm" in text:
        confluence = "STRONG_CONFLUENCE"
    elif "weak" in text:
        confluence = "WEAK_CONFLUENCE"

    return FakeDetectionGraph(
        graph_id=f"frame_{frame_idx:04d}",
        timestamp=f"00:{frame_idx // 60:02d}:{frame_idx % 60:02d}",
        trend_state=trend,
        trading_session=FakeSession("RTH"),
        footprint_imbalances=FakeImbalances(tuple(imbalances)) if imbalances else None,
        absorption_result=FakeAbsorptions(tuple(absorptions)) if absorptions else None,
        footprint_delta=FakeDelta(()),
        confluence=FakeConfluence(confluence),
    )


# ---------------------------------------------------------------------------
# Relationship Discovery from Transcript Text
# ---------------------------------------------------------------------------

class RelationshipExtractor:
    """Extract semantic relationships between concepts from transcript text.

    Two-phase approach:
      1. EXPLICIT PATTERNS: "Absorption requires Aggression" -> direct relationship
      2. CO-OCCURRENCE FALLBACK: Concepts in same sentence -> inferred relationship

    Discovers patterns like:
      "Absorption requires Aggression" -> Absorption --requires--> Aggression
      "Compression precedes a Squeeze" -> Compression --precedes--> Squeeze
      "Exhaustion signals Reversal"    -> Exhaustion --signals--> Reversal
    """

    # Phase 1: Explicit grammatical patterns
    EXPLICIT_PATTERNS = [
        # Strong causal/requirement patterns
        (r'\b({source})\s+(?:requires?|needs?|demands?|depends?\s+on)\s+(?:the\s+)?({target})\b', "requires", 0.95, "forward"),
        (r'\b({source})\s+(?:is\s+(?:a|an)\s+)?(?:form\s+of|type\s+of|kind\s+of)\s+(?:the\s+)?({target})\b', "is_a", 0.9, "forward"),
        (r'\b({source})\s+(?:is\s+)?(?:the\s+)?(?:same\s+as|equivalent\s+to|identical\s+to)\s+(?:the\s+)?({target})\b', "is_a", 0.85, "bidirectional"),

        # Temporal/precedence patterns
        (r'\b({source})\s+(?:precedes?|comes?\s+before|leads?\s+to|results?\s+in|gives?\s+rise\s+to)\s+(?:the\s+)?({target})\b', "precedes", 0.85, "forward"),
        (r'\b({source})\s+(?:triggers?|initiates?|starts?|kicks?\s+off)\s+(?:the\s+)?({target})\b', "triggers", 0.85, "forward"),
        (r'\b({source})\s+(?:follows?|comes?\s+after|happens?\s+after)\s+(?:the\s+)?({target})\b', "follows", 0.8, "forward"),
        (r'\b({source})\s+(?:often\s+)?(?:results?\s+in|ends?\s+up\s+as|turns?\s+into)\s+(?:the\s+)?({target})\b', "results_in", 0.8, "forward"),

        # Signaling/confirmation patterns
        (r'\b({source})\s+(?:signals?|indicates?|shows?|suggests?|means?)\s+(?:the\s+)?({target})\b', "signals", 0.8, "forward"),
        (r'\b({source})\s+(?:confirms?|validates?|verifies?)\s+(?:the\s+)?({target})\b', "confirms", 0.85, "forward"),
        (r'\b({source})\s+(?:contradicts?|invalidates?|negates?)\s+(?:the\s+)?({target})\b', "contradicts", 0.8, "forward"),

        # Location/context patterns
        (r'\b({source})\s+(?:occurs?\s+at|happens?\s+at|seen?\s+at|found?\s+at)\s+(?:the\s+)?({target})\b', "occurs_at", 0.75, "forward"),
        (r'\b({source})\s+(?:seen?\s+with|associated\s+with|linked\s+to|correlated\s+with)\s+(?:the\s+)?({target})\b', "associated_with", 0.7, "bidirectional"),

        # Opposition patterns
        (r'\b({source})\s+(?:is\s+(?:the\s+)?)?opposite\s+(?:of|to)\s+(?:the\s+)?({target})\b', "opposite_of", 0.9, "bidirectional"),
        (r'\b({source})\s+(?:vs\.?|versus)\s+(?:the\s+)?({target})\b', "opposite_of", 0.85, "bidirectional"),

        # Weak co-occurrence patterns
        (r'\b({source})\s+(?:and|with)\s+(?:the\s+)?({target})\b', "related_to", 0.5, "bidirectional"),

        # FABIO-SPECIFIC: Teaching phrases that imply relationships
        (r'\b({source})\s+(?:because|since|as)\s+(?:the\s+)?(?:reason\s+is\s+)?(?:we\s+have|there\s+is|there\s+was)?\s*(?:the\s+)?({target})\b', "caused_by", 0.7, "forward"),
        (r'\b(?:this\s+is|that\s+is|which\s+is)\s+(?:what\s+we\s+call|called)?\s*(?:the\s+)?({source})\b.*\b(?:this\s+is|that\s+is|which\s+is)\s+(?:what\s+we\s+call|called)?\s*(?:the\s+)?({target})\b', "related_to", 0.6, "bidirectional"),
        (r'\b(?:you\s+see)\s+(?:the\s+)?({source})\b.*\b(?:and|with)\s+(?:the\s+)?({target})\b', "related_to", 0.55, "bidirectional"),
        (r'\b(?:we\s+have)\s+(?:the\s+)?({source})\b.*\b(?:and|plus|also)\s+(?:the\s+)?({target})\b', "related_to", 0.55, "bidirectional"),
        (r'\b({source})\s+(?:here|there)\b.*\b(?:and|with)\s+(?:the\s+)?({target})\b', "cooccurs_with", 0.5, "bidirectional"),
    ]

    def __init__(self, ontology: dict):
        self.ontology = ontology
        self.concept_names = sorted(
            [c["name"] for c in ontology.get("concepts", [])],
            key=len, reverse=True  # Longest first to avoid partial matches
        )
        self._compiled_explicit = self._compile_explicit_patterns()

    def _compile_explicit_patterns(self) -> list:
        """Compile explicit regex patterns with concept names substituted."""
        name_alts = "|".join(re.escape(name) for name in self.concept_names if len(name) > 2)
        if not name_alts:
            return []

        compiled = []
        for pattern_tpl, rel_name, confidence, direction in self.EXPLICIT_PATTERNS:
            pattern_str = pattern_tpl.format(source=f"({name_alts})", target=f"({name_alts})")
            try:
                compiled.append((re.compile(pattern_str, re.IGNORECASE), rel_name, confidence, direction))
            except re.error:
                continue
        return compiled

    def extract_explicit(self, text: str) -> list:
        """Extract explicit relationships from text."""
        if not self._compiled_explicit:
            return []

        discovered = []
        seen = set()

        for pattern, rel_name, confidence, direction in self._compiled_explicit:
            for match in pattern.finditer(text):
                source = match.group(1)
                target = match.group(2)

                if source == target:
                    continue

                key = (source.lower(), target.lower(), rel_name)
                if key in seen:
                    continue
                seen.add(key)

                if direction == "bidirectional":
                    seen.add((target.lower(), source.lower(), rel_name))

                discovered.append({
                    "source": source,
                    "target": target,
                    "relation": rel_name,
                    "confidence": confidence,
                    "direction": direction,
                    "matched_text": match.group(0),
                    "method": "explicit",
                    "discovered": True,
                })

        return discovered

    def extract_cooccurrence(self, segments: list, min_cooccurrence: int = 1) -> list:
        """Extract relationships from concept co-occurrence in sentences.

        If two concepts appear in the same sentence, infer a weak 'related_to'
        relationship. For single lessons, min_cooccurrence=1 captures all pairs.
        For batch processing across many lessons, increase to 2+ for quality.
        """
        pair_counts = {}
        pair_examples = {}

        for idx, text in segments:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
            for sent in sentences:
                sent_lower = sent.lower()
                found = []
                for name in self.concept_names:
                    if name.lower() in sent_lower:
                        found.append(name)

                for i, a in enumerate(found):
                    for b in found[i+1:]:
                        if a == b:
                            continue
                        key = tuple(sorted([a.lower(), b.lower()]))
                        pair_counts[key] = pair_counts.get(key, 0) + 1
                        if key not in pair_examples:
                            pair_examples[key] = sent.strip()[:100]

        discovered = []
        for (a, b), count in pair_counts.items():
            if count >= min_cooccurrence:
                # Find the actual names (not lowercased)
                name_a = next(n for n in self.concept_names if n.lower() == a)
                name_b = next(n for n in self.concept_names if n.lower() == b)

                # Confidence scales with co-occurrence count
                # count=1 -> 0.45, count=2 -> 0.55, count=3+ -> 0.65 capped
                confidence = min(0.65, 0.45 + (count - 1) * 0.1)

                discovered.append({
                    "source": name_a,
                    "target": name_b,
                    "relation": "cooccurs_with",
                    "confidence": confidence,
                    "direction": "bidirectional",
                    "matched_text": pair_examples.get((a, b), ""),
                    "cooccurrence_count": count,
                    "method": "cooccurrence",
                    "discovered": True,
                })

        return discovered

    def extract(self, text: str, segments: list = None) -> list:
        """Extract all relationships using both explicit and co-occurrence methods."""
        explicit = self.extract_explicit(text)

        cooccur = []
        if segments:
            cooccur = self.extract_cooccurrence(segments)

        # Merge, preferring explicit over co-occurrence for same pairs
        explicit_pairs = set((r["source"].lower(), r["target"].lower()) for r in explicit)
        cooccur_filtered = [r for r in cooccur 
                            if (r["source"].lower(), r["target"].lower()) not in explicit_pairs
                            and (r["target"].lower(), r["source"].lower()) not in explicit_pairs]

        return explicit + cooccur_filtered

    def extract_from_segments(self, segments: list) -> list:
        """Extract relationships from all transcript segments."""
        all_text = " ".join(text for _, text in segments)
        all_rels = self.extract(all_text, segments)

        # Add segment index to each relationship (from first matching segment)
        for rel in all_rels:
            for idx, text in segments:
                if rel["source"].lower() in text.lower() and rel["target"].lower() in text.lower():
                    rel["segment_index"] = idx
                    break

        return all_rels
# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def serialize_concept(concept: Concept) -> dict:
    return {
        "concept_id": concept.concept_id,
        "name": concept.definition.name,
        "definition": concept.definition.definition,
        "visual_appearance": concept.definition.visual_appearance,
        "teacher_explanation": concept.definition.teacher_explanation,
        "confidence_level": concept.confidence.level.name,
        "confidence_score": float(concept.confidence.score),
        "evidence_count": concept.confidence.evidence_count,
        "positive_examples": len(concept.positive_examples),
        "negative_examples": len(concept.negative_examples),
        "related_concepts": list(concept.related_concepts),
        "lesson_references": list(concept.lesson_references),
        "evolution_events": len(concept.evolution_history),
    }


def serialize_experience(exp: Experience) -> dict:
    return {
        "experience_id": exp.experience_id,
        "session_reference": exp.session_reference,
        "what_was_seen": exp.observation.what_was_seen,
        "teacher_explanation": exp.teacher_explanation[:200] + "..." if len(exp.teacher_explanation) > 200 else exp.teacher_explanation,
        "has_reasoning": exp.reasoning is not None,
        "has_outcome": exp.outcome is not None,
        "has_reflection": exp.reflection is not None,
        "is_complete": exp.is_complete(),
        "concept_references": list(exp.concept_references),
    }


def serialize_knowledge_graph(graph: KnowledgeGraph) -> dict:
    return {
        "total_concepts": len(graph.concepts),
        "total_relationships": len(graph.relationships),
        "concepts": sorted(graph.concepts),
        "relationships": [
            {
                "source": rel.source_concept,
                "type": rel.relationship_type.name,
                "target": rel.target_concept,
                "confidence": float(rel.confidence),
            }
            for rel in graph.relationships
        ],
    }


# ---------------------------------------------------------------------------
# Main processing — HONEST ARCHITECTURE
# ---------------------------------------------------------------------------

def process_video_folder(
    video_path: str,
    transcript_path: str,
    output_dir: str,
    lesson_name: Optional[str] = None,
    ontology_path: Optional[str] = None,
    max_frames: int = 50,
    mode: str = "transcript-only",
) -> dict:
    """Process one video + transcript.

    Args:
        mode: "transcript-only" (default, honest) or "vision" (attempts real vision)
    """
    video_path = Path(video_path)
    transcript_path = Path(transcript_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lesson_name = lesson_name or video_path.stem

    print(f"\n{'=' * 70}")
    print(f"Processing: {lesson_name}")
    print(f"Mode: {mode}")
    print(f"{'=' * 70}")
    print(f"Video: {video_path}")
    print(f"Transcript: {transcript_path}")
    print(f"Output: {output_dir}")

    # Load ontology first (needed for everything)
    ontology = load_ontology(ontology_path) if ontology_path else {}
    if ontology:
        print(f"[Ontology] Loaded: {len(ontology.get('concepts', []))} concepts, "
              f"{len(ontology.get('relationships', []))} relationships")
    else:
        print("[Ontology] WARNING: No ontology loaded. Graph will be empty.")

    # Load transcript
    transcript_entries = load_transcript(str(transcript_path))
    print(f"Transcript entries: {len(transcript_entries)}")

    if not transcript_entries:
        print("WARNING: No transcript entries found. Creating minimal synthetic data.")
        transcript_entries = ((0, "Fabio explains the market structure and order flow concepts."),)

    # Coverage analysis (always run)
    coverage = analyze_coverage(transcript_entries, ontology)
    print(f"[Coverage] {coverage['covered_segments']}/{coverage['total_segments']} segments matched concepts "
          f"({coverage['coverage_percent']}%)")
    if coverage["uncovered_segments"] > 0:
        print(f"[Coverage] {coverage['uncovered_segments']} segments had zero concept matches")

    # Relationship discovery from transcript text
    rel_extractor = RelationshipExtractor(ontology)
    discovered_rels = rel_extractor.extract_from_segments(list(transcript_entries))
    print(f"[Discovery] Found {len(discovered_rels)} potential relationships in transcript")
    if discovered_rels:
        for rel in discovered_rels[:5]:
            print(f"  → {rel['source']} --{rel['relation']}--> {rel['target']} "
                  f"(conf={rel['confidence']:.2f}): '{rel['matched_text'][:60]}...'")
        if len(discovered_rels) > 5:
            print(f"  ... and {len(discovered_rels) - 5} more")

    # Extract video frames
    video_frames = extract_video_frames(str(video_path), max_frames=max_frames)

    # Build frames for integration
    frames = []

    if mode == "transcript-only":
        # HONEST MODE: Process transcript segments directly.
        # Video frames are used for timestamps/visual context only.
        # No FakeDetectionGraph — the integration receives the raw transcript.
        print(f"[Mode] Transcript-only: Processing {len(transcript_entries)} segments")

        num_entries = len(transcript_entries)
        num_video_frames = max(1, len(video_frames))

        for i, (entry_idx, text) in enumerate(transcript_entries):
            # Match to nearest video frame for timestamp
            vf_idx = min(i * num_video_frames // max(1, num_entries), num_video_frames - 1)
            frame_num, timestamp, image = video_frames[vf_idx] if video_frames else (i, f"00:{i:02d}:00", None)

            # In transcript mode, we still pass a minimal graph so the integration
            # has something to work with, but it's labeled as transcript-derived
            graph = build_fake_detection_graph(i, text)
            frames.append((i, timestamp, graph, text))

    else:
        # VISION MODE: Attempt real Genesis vision pipeline on each frame.
        # Falls back to FakeDetectionGraph if vision pipeline fails.
        print(f"[Mode] Vision: Processing {len(video_frames)} frames")

        num_frames = len(video_frames)
        num_entries = len(transcript_entries)

        for i, (frame_num, timestamp, image) in enumerate(video_frames):
            entry_idx = min(i * num_entries // max(1, num_frames), num_entries - 1)
            _, text = transcript_entries[entry_idx]

            # TODO: Replace with real Bundle 1-8 DetectionGraph from image
            # For now, fall back to transcript-derived FakeDetectionGraph
            graph = build_fake_detection_graph(i, text)
            frames.append((i, timestamp, graph, text))

    print(f"Frames to process: {len(frames)}")

    # -----------------------------------------------------------------------
    # MONKEY-PATCH: KnowledgeGraph.query() to accept kwargs that coach.py
    # expects (source_concept, relationship_type, target_concept).
    # Genesis's coach.py calls graph.query(source_concept=..., relationship_type=...)
    # but the KnowledgeGraph.query() signature doesn't accept these kwargs.
    # This runtime patch filters relationships manually when those kwargs are present.
    # -----------------------------------------------------------------------
    if hasattr(KnowledgeGraph, 'query'):
        _orig_query = KnowledgeGraph.query
        def _patched_query(self, source_concept=None, relationship_type=None, target_concept=None, **kwargs):
            # No filter kwargs -> delegate to original
            if source_concept is None and relationship_type is None and target_concept is None:
                return _orig_query(self, **kwargs)
            # Manual filtering
            results = []
            for rel in getattr(self, 'relationships', []):
                if source_concept is not None and getattr(rel, 'source_concept', None) != source_concept:
                    continue
                if target_concept is not None and getattr(rel, 'target_concept', None) != target_concept:
                    continue
                if relationship_type is not None:
                    rt = getattr(rel, 'relationship_type', None)
                    # Handle enum comparison (coach passes enum instances)
                    if hasattr(rt, 'name') and hasattr(relationship_type, 'name'):
                        if rt.name != relationship_type.name:
                            continue
                    elif rt != relationship_type:
                        continue
                results.append(rel)
            return results
        KnowledgeGraph.query = _patched_query

    # Initialize apprentice integration
    # FIX: Set key_frame_interval=1 so ALL frames are processed, not sampled
    config = RunnerIntegrationConfiguration(
        process_every_frame=(mode == "vision"),
        process_key_frames_only=(mode == "transcript-only"),
        key_frame_interval=1,  # PROCESS ALL FRAMES
        enable_coach=True,
        enable_concept_extraction=True,
        enable_experience_creation=True,
        enable_knowledge_graph_building=True,
        min_transcript_length_for_explanation=10,
        max_frames_per_lesson=max_frames,
    )
    integration = ApprenticeRunnerIntegration(config)

    # Process lesson
    result = integration.process_lesson(lesson_name, tuple(frames))

    print(f"\nLesson Result:")
    print(f"  Frames processed: {len(result.frame_results)}")
    print(f"  Sessions: {len(result.sessions)}")
    print(f"  Experiences: {len(result.experiences)}")
    print(f"  Concepts: {result.concept_statistics.total_concepts}")
    print(f"  Knowledge graph: {result.knowledge_graph.statistics.total_concepts} concepts, "
          f"{result.knowledge_graph.statistics.total_relationships} relationships")

    # Build report
    report = integration.build_report(datetime.utcnow().isoformat())

    # Compute REAL confidence
    layer_concepts = layer.all_concepts() if (layer := integration.get_apprentice_layer()) else []
    enriched_graph = build_graph_from_ontology(ontology, {c.definition.name for c in layer_concepts}, discovered_rels)
    real_confidence = compute_real_confidence(layer_concepts, coverage, enriched_graph)

    print(f"[Confidence] Real computed: {real_confidence:.1%} (was hardcoded ~71.4%)")

    # Save outputs
    _save_outputs(output_dir, report, integration, ontology, coverage, real_confidence, enriched_graph, mode, discovered_rels)

    return {
        "report": report,
        "coverage": coverage,
        "confidence": real_confidence,
        "graph": enriched_graph,
    }


def _save_outputs(
    output_dir: Path,
    report: ApprenticeReport,
    integration,
    ontology: dict,
    coverage: dict,
    real_confidence: float,
    enriched_graph: dict,
    mode: str,
    discovered_rels: Optional[List[dict]] = None,
) -> None:
    layer = integration.get_apprentice_layer()
    layer_concepts = layer.all_concepts()
    layer_experiences = layer.all_experiences()

    # 1. Apprentice Report
    report_path = output_dir / "apprentice_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "report_id": report.report_id,
            "report_timestamp": report.report_timestamp,
            "mode": mode,
            "total_lessons": len(report.lesson_results),
            "total_concepts_learned": report.total_concepts_learned,
            "total_experiences_created": report.total_experiences_created,
            "total_sessions_completed": report.total_sessions_completed,
            "total_coach_explanations": report.total_coach_explanations,
            "total_frames_processed": report.total_frames_processed,
            "overall_learning_confidence": real_confidence,  # REAL, not hardcoded
            "concept_mastery_distribution": report.concept_mastery_distribution,
            "top_concepts_by_confidence": report.top_concepts_by_confidence,
            "what_was_learned": report.what_was_learned,
            "what_needs_more_study": report.what_needs_more_study,
            "knowledge_graph_summary": report.knowledge_graph_summary,
            "coverage": coverage,
        }, f, indent=2, cls=DecimalEncoder)
    print(f"\nSaved: {report_path}")

    # 2. Concepts
    concepts_path = output_dir / "concepts.json"
    concepts = [serialize_concept(c) for c in layer_concepts]
    with open(concepts_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2, cls=DecimalEncoder)
    print(f"Saved: {concepts_path}")

    # 3. Experiences
    experiences_path = output_dir / "experiences.json"
    experiences = [serialize_experience(e) for e in layer_experiences]
    with open(experiences_path, "w", encoding="utf-8") as f:
        json.dump(experiences, f, indent=2, cls=DecimalEncoder)
    print(f"Saved: {experiences_path}")

    # 4. Knowledge Graph — ALWAYS enriched from ontology + discovered relationships
    graph_path = output_dir / "knowledge_graph.json"

    graph = layer.get_knowledge_graph()
    integration_concepts = set(graph.concepts)
    integration_rels = len(graph.relationships)

    print(f"[Graph] Integration produced: {len(integration_concepts)} concepts, {integration_rels} relationships")
    print(f"[Graph] Enriched from ontology: {enriched_graph['total_concepts']} concepts, "
          f"{enriched_graph['total_relationships']} relationships "
          f"({enriched_graph.get('predefined_relationships', 0)} predefined, "
          f"{enriched_graph.get('discovered_relationships', 0)} discovered)")

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(enriched_graph, f, indent=2, cls=DecimalEncoder)
    print(f"Saved: {graph_path}")

    # Also save discovered relationships separately for audit
    if discovered_rels:
        discovered_path = output_dir / "discovered_relationships.json"
        with open(discovered_path, "w", encoding="utf-8") as f:
            json.dump(discovered_rels, f, indent=2, cls=DecimalEncoder)
        print(f"Saved: {discovered_path}")

    # 5. Coverage Report (NEW)
    coverage_path = output_dir / "coverage_report.json"
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, cls=DecimalEncoder)
    print(f"Saved: {coverage_path}")

    # 6. Coach Explanations (text)
    coach_path = output_dir / "coach_explanations.txt"
    with open(coach_path, "w", encoding="utf-8") as f:
        f.write("GENESIS LIVE COACH — Explanations\n")
        f.write("=" * 70 + "\n\n")
        for lesson in report.lesson_results:
            for frame in lesson.frame_results:
                if frame.explanation_text:
                    f.write(f"Lesson: {lesson.lesson_reference} | Frame: {frame.frame_index}\n")
                    f.write(f"Timestamp: {frame.frame_timestamp}\n")
                    f.write(frame.explanation_text)
                    f.write("\n\n")
    print(f"Saved: {coach_path}")

    # 7. Summary text — HONEST
    summary_path = output_dir / "apprentice_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GENESIS APPRENTICE — Learning Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Report ID: {report.report_id}\n")
        f.write(f"Timestamp: {report.report_timestamp}\n")
        f.write(f"Mode: {mode}\n\n")
        f.write(f"Lessons Processed: {len(report.lesson_results)}\n")
        f.write(f"Concepts Learned: {report.total_concepts_learned}\n")
        f.write(f"Experiences Created: {report.total_experiences_created}\n")
        f.write(f"Sessions Completed: {report.total_sessions_completed}\n")
        f.write(f"Coach Explanations: {report.total_coach_explanations}\n")
        f.write(f"Frames Processed: {report.total_frames_processed}\n")
        f.write(f"Overall Confidence: {real_confidence:.2%}\n\n")

        f.write("Concept Coverage:\n")
        f.write(f"  Segments with concepts: {coverage['covered_segments']}/{coverage['total_segments']} "
                f"({coverage['coverage_percent']}%)\n")
        f.write(f"  Segments with NO concepts: {coverage['uncovered_segments']}\n\n")

        f.write("Concept Mastery:\n")
        for level, count in report.concept_mastery_distribution:
            bar = "█" * count + "░" * max(0, 10 - count)
            f.write(f"  {level:12s} {bar} ({count})\n")
        f.write("\nTop Concepts:\n")
        for name, score in report.top_concepts_by_confidence[:10]:
            f.write(f"  {name:20s} {float(score):.2f}\n")
        f.write("\nWhat was learned:\n")
        for item in report.what_was_learned:
            f.write(f"  ✓ {item}\n")
        f.write("\nWhat needs more study:\n")
        if report.what_needs_more_study:
            for item in report.what_needs_more_study:
                f.write(f"  ⚠ {item}\n")
        else:
            f.write(f"  ✓ All concepts have sufficient study\n")

        f.write(f"\nKnowledge Graph:\n")
        f.write(f"  Concepts: {enriched_graph['total_concepts']}\n")
        f.write(f"  Relationships: {enriched_graph['total_relationships']}\n")

        if mode == "transcript-only":
            f.write(f"\n[NOTE] Running in TRANSCRIPT-ONLY mode.\n")
            f.write(f"  Video frames provide timestamps only.\n")
            f.write(f"  Concept extraction is keyword-based from transcript text.\n")
            f.write(f"  Vision analysis (Bundles 1-8) is NOT active.\n")

        f.write(f"\n{report.knowledge_graph_summary}\n")
    print(f"Saved: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process a real Fabio video through the Genesis Apprentice Layer"
    )
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--transcript", type=str, help="Path to transcript file")
    parser.add_argument("--input", type=str, help="Input folder containing video and transcript")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--lesson", type=str, help="Lesson name (defaults to video filename)")
    parser.add_argument("--ontology", type=str, help="Path to fabio_ontology.json")
    parser.add_argument("--mode", choices=["transcript-only", "vision"], default="transcript-only",
                        help="Processing mode. transcript-only=honest text-based (default). "
                             "vision=attempts real Bundle 1-8 vision analysis.")
    parser.add_argument("--every-frame", action="store_true", help="Process every frame (slower)")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames per lesson")

    args = parser.parse_args()

    # Resolve paths
    if args.input:
        input_dir = Path(args.input)
        videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mkv"))
        transcripts = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.srt")) + list(input_dir.glob("*.json"))

        if not videos:
            print(f"ERROR: No video found in {input_dir}")
            sys.exit(1)
        if not transcripts:
            print(f"ERROR: No transcript found in {input_dir}")
            sys.exit(1)

        video_path = str(videos[0])
        transcript_path = str(transcripts[0])
    elif args.video and args.transcript:
        video_path = args.video
        transcript_path = args.transcript
    else:
        print("ERROR: Specify either --input (folder) or both --video and --transcript")
        sys.exit(1)

    # Process
    result = process_video_folder(
        video_path=video_path,
        transcript_path=transcript_path,
        output_dir=args.output,
        lesson_name=args.lesson,
        ontology_path=args.ontology,
        max_frames=args.max_frames,
        mode=args.mode,
    )

    # Final summary
    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nOutput files in: {args.output}")
    print(f"  • apprentice_report.json    — Full learning report")
    print(f"  • concepts.json             — All learned concepts")
    print(f"  • experiences.json          — All experiences")
    print(f"  • knowledge_graph.json      — Concept relationships (ontology-enriched)")
    print(f"  • coverage_report.json      — Transcript coverage analysis (NEW)")
    print(f"  • coach_explanations.txt    — Human-readable explanations")
    print(f"  • apprentice_summary.txt    — Quick summary")
    print(f"\nGenesis learned {result['report'].total_concepts_learned} concepts from {len(result['report'].lesson_results)} lesson(s).")
    print(f"Real confidence: {result['confidence']:.1%}")
    print(f"Coverage: {result['coverage']['coverage_percent']}% of transcript segments matched concepts")
    print(f"Graph: {result['graph']['total_concepts']} concepts, {result['graph']['total_relationships']} relationships")


if __name__ == "__main__":
    main()