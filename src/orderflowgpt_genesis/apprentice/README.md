# Genesis Apprentice Layer (Bundle 14)

## Overview

The Apprentice Layer implements the Genesis 2.0 educational architecture:

- **Concept Engine** — learns, tracks, and evolves concept mastery
- **Experience Engine** — the central learning object (observe -> reason -> outcome -> reflect)
- **Learning Loop** — Observe -> Question -> Explain -> Reflect -> Experience -> Memory -> Knowledge
- **Knowledge Graph** — relationship graph between concepts (Absorption requires Aggression)
- **Decision Hierarchy** — Fabio's explanatory framework: Market State -> Location -> Aggression -> Risk -> Management -> Reflection
- **Integration Layer** — `ApprenticeLayer` wires everything to existing Bundle 1-13 infrastructure

## Design Principles

- **Immutable** — all objects are frozen dataclasses
- **Deterministic** — no randomness, no AI, no ML inside the core
- **No side effects** — no I/O, no networking, no persistence
- **No predictions** — explains first, never recommends trades
- **No breaking changes** — consumes existing `DetectionGraph` without modification

## Installation

Copy the `apprentice/` directory into `src/orderflowgpt_genesis/`:

```
src/orderflowgpt_genesis/
  apprentice/
    __init__.py
    reasoning.py
    concepts.py
    experiences.py
    knowledge_graph.py
    learning_loop.py
    integration.py
```

Then add to `src/orderflowgpt_genesis/__init__.py`:

```python
from .apprentice import (
    ApprenticeLayer,
    ApprenticeConfiguration,
    ApprenticeResult,
    # ... export what you need
)
```

## Quick Start

### Explain a chart (What is happening?)

```python
from orderflowgpt_genesis import DetectionGraph
from orderflowgpt_genesis.apprentice import ApprenticeLayer

apprentice = ApprenticeLayer()
explanation = apprentice.explain_chart(detection_graph)

print(explanation.market_state.state)      # e.g., TRENDING_UP
print(explanation.location.location)       # e.g., AT_POC
print(explanation.aggression.aggression)   # e.g., BUYER_AGGRESSIVE
print(explanation.risk.risk_level)         # e.g., MEDIUM
```

### Run a full learning session

```python
result = apprentice.process_frame(
    session_id="lesson01_frame001",
    lesson_reference="Lesson01_AbsorptionAtPOC",
    detection_graph=graph,
    teacher_explanation=(
        "Here we see absorption at the POC. The buyers are aggressive "
        "but the passive side is absorbing at this level. This tells us "
        "the market is balanced here and we should wait for a resolution."
    ),
    source_reference="Fabio Lesson01 00:03:24",
    timestamp="2026-08-04T18:00:00Z",
)

print(result.session.phases_completed)
print(result.experience_created.experience_id)
print(result.knowledge_graph.concepts)
```

### Register and query concepts

```python
from orderflowgpt_genesis.apprentice import (
    Concept, ConceptDefinition, ConceptExample, ConceptConfidenceLevel
)

concept = Concept(
    definition=ConceptDefinition(
        name="Absorption",
        definition="Passive side absorbing aggressive side volume",
        visual_appearance="Large passive volume at one price level with small wick",
        teacher_explanation="Fabio: 'When you see the passive side holding...'",
    ),
    positive_examples=(
        ConceptExample(
            description="ES 1min at 5000.00",
            chart_context="ES 1min footprint",
            evidence_reference="lesson01_frame042",
            is_positive=True,
            lesson_reference="Lesson01",
        ),
    ),
)

apprentice.register_concept(concept)
absorption = apprentice.get_concept_by_name("Absorption")
print(absorption.confidence.level)  # NOVICE -> evolves with examples
```

### Query the knowledge graph

```python
# After learning sessions, query relationships
graph = apprentice.get_knowledge_graph()

# What does Absorption require?
print(graph.what_requires("Absorption"))  # ("Aggression",)

# Find path between concepts
path = graph.shortest_path("Absorption", "MarketState")
if path:
    for rel in path.relationships:
        print(f"{rel.source_concept} --{rel.relationship_type.name}--> {rel.target_concept}")
```

## Architecture

```
DetectionGraph (Bundle 1-13)
  |
  v
DecisionHierarchyAnalyzer  ->  DecisionHierarchyResult
  |                               (Market State, Location, Aggression, Risk, Management)
  v
LearningLoop
  |-- observe()   -> LearningObservation
  |-- question()  -> LearningQuestion
  |-- explain()   -> TeacherExplanation
  |-- reflect()   -> LearningReflection
  |-- experience() -> Experience (via ExperienceEngine)
  |-- to_memory()  -> links to ConceptEngine
  |-- to_knowledge() -> KnowledgeGraph
  v
ApprenticeResult
  + hierarchy_result
  + session
  + experience_created
  + concepts_touched
  + knowledge_graph
```

## Testing

```bash
cd src/orderflowgpt_genesis/apprentice
pytest test_apprentice.py -v
```

All tests are deterministic, side-effect-free, and require no external dependencies.

## Integration with GenesisRunner

To wire the Apprentice Layer into `GenesisRunner` (Bundle 13.5):

1. In `runner.py`, instantiate `ApprenticeLayer` alongside existing engines.
2. After `DetectionGraph` is built, call `apprentice.process_frame()`.
3. Store the `ApprenticeResult` in the lesson output (`report.json`).
4. The runner already saves `report.json`, `summary.json`, `processing.log`.

Example runner integration:

```python
from orderflowgpt_genesis.apprentice import ApprenticeLayer, ApprenticeConfiguration

class GenesisRunner:
    def __init__(self, config):
        # ... existing init ...
        self.apprentice = ApprenticeLayer(ApprenticeConfiguration(
            enable_auto_concept_extraction=True,
            enable_experience_to_memory=True,
        ))

    def _process_lesson(self, video_path, transcript_path):
        # ... existing vision pipeline ...
        graph = self._build_detection_graph(video_path)

        # NEW: Apprentice Layer
        for frame_idx, frame_graph in enumerate(self._frame_graphs(graph)):
            explanation = self._get_fabio_explanation(transcript_path, frame_idx)
            apprentice_result = self.apprentice.process_frame(
                session_id=f"{lesson_id}_frame{frame_idx:04d}",
                lesson_reference=lesson_id,
                detection_graph=frame_graph,
                teacher_explanation=explanation,
                source_reference=f"{lesson_id} frame {frame_idx}",
                timestamp=frame_graph.timestamp,
            )
            self._store_apprentice_result(apprentice_result)
```


## Runner Integration (Bundle 14.5)

The `ApprenticeRunnerIntegration` wires the Apprentice Layer into your existing
`GenesisRunner` (Bundle 13.5) so batch video processing automatically builds
concepts, experiences, and the knowledge graph.

### Quick Integration

Modify your `runner.py` to use the integration after building DetectionGraphs:

```python
from orderflowgpt_genesis.apprentice import (
    ApprenticeRunnerIntegration,
    RunnerIntegrationConfiguration,
)

class GenesisRunner:
    def __init__(self, config):
        # ... existing init ...
        self.apprentice_integration = ApprenticeRunnerIntegration(
            RunnerIntegrationConfiguration(
                process_key_frames_only=True,
                key_frame_interval=30,
                enable_coach=True,
                enable_concept_extraction=True,
                enable_experience_creation=True,
                max_frames_per_lesson=50,
            )
        )

    def _process_lesson(self, video_path, transcript_path):
        # ... existing pipeline: video import, frame extraction, vision graph ...

        # Build frame data for apprentice layer
        frames = []
        for frame_idx, (frame_graph, transcript_text) in enumerate(
            zip(detection_graphs, aligned_transcripts)
        ):
            frames.append((
                frame_idx,
                frame_graph.timestamp,
                frame_graph,
                transcript_text,
            ))

        # Run apprentice layer
        lesson_result = self.apprentice_integration.process_lesson(
            lesson_reference=lesson_id,
            frames=tuple(frames),
        )

        # ... existing save logic ...

    def _save_reports(self):
        # ... existing report.json, summary.json ...

        # NEW: Apprentice learning report
        apprentice_report = self.apprentice_integration.build_report(
            timestamp=datetime.utcnow().isoformat()
        )
        self._save_json(
            "apprentice_report.json",
            self._serialize_apprentice_report(apprentice_report),
        )
```

### What Gets Produced

After processing a folder of lessons, the integration produces:

**Per Lesson:**
- `LessonApprenticeResult` with all frames processed
- Concepts extracted from Fabio's explanations
- Experiences linking observations to teacher explanations
- Knowledge graph relationships between concepts
- Coach explanations for key frames

**Aggregate Report (`apprentice_report.json`):**
```json
{
  "report_id": "a3f7b2c9d1e8f5a2",
  "total_concepts_learned": 12,
  "total_experiences_created": 45,
  "total_sessions_completed": 45,
  "total_coach_explanations": 15,
  "total_frames_processed": 150,
  "concept_mastery_distribution": [
    ["NOVICE", 5],
    ["DEVELOPING", 4],
    ["COMPETENT", 2],
    ["PROFICIENT", 1],
    ["EXPERT", 0]
  ],
  "top_concepts_by_confidence": [
    ["Absorption", 0.85],
    ["POC", 0.75],
    ["Imbalance", 0.60]
  ],
  "knowledge_graph_summary": "Knowledge graph: 12 concepts, 8 relationships, avg confidence 0.72",
  "what_was_learned": [
    "Lesson Lesson01: 5 concepts, 15 experiences",
    "Lesson Lesson02: 7 concepts, 30 experiences"
  ],
  "what_needs_more_study": [
    "DeltaDivergence (NOVICE)",
    "Exhaustion (DEVELOPING)"
  ],
  "overall_learning_confidence": 0.52,
  "report_timestamp": "2026-08-04T18:00:00Z"
}
```

### Configuration Options

```python
RunnerIntegrationConfiguration(
    process_every_frame=False,        # True = all frames, False = key frames only
    process_key_frames_only=True,     # Only process every Nth frame
    key_frame_interval=30,            # Process frame 0, 30, 60, ...
    enable_coach=True,                # Run Live Coach on processed frames
    enable_concept_extraction=True,   # Extract concepts from explanations
    enable_experience_creation=True,  # Create experiences
    enable_knowledge_graph_building=True,
    min_transcript_length_for_explanation=10,  # Skip short/empty transcripts
    max_frames_per_lesson=100,        # Cap frames per lesson
    report_concept_detail_level="summary",  # "summary" or "full"
)
```

### Frame Selection Logic

By default, the integration processes **key frames only** (every 30th frame) to
keep processing deterministic and fast. A frame is processed if:
1. It matches the key frame interval (or `process_every_frame=True`)
2. The transcript explanation is at least `min_transcript_length_for_explanation` chars
3. The lesson has not exceeded `max_frames_per_lesson`

### Accessing Results Programmatically

```python
integration = ApprenticeRunnerIntegration()

# Process lessons
for lesson_id, frames in lesson_data:
    integration.process_lesson(lesson_id, frames)

# Get the final report
report = integration.build_report(timestamp)
print(report.what_was_learned)
print(report.what_needs_more_study)

# Access underlying apprentice layer directly
layer = integration.get_apprentice_layer()
for concept in layer.all_concepts():
    print(f"{concept.definition.name}: {concept.confidence.level.name}")

# Query knowledge graph
graph = layer.get_knowledge_graph()
print(graph.what_requires("Absorption"))  # ("Aggression",)
```


## Learning Demonstration

Run the demonstration script to see Genesis learn from simulated Fabio videos:

```bash
python src/orderflowgpt_genesis/apprentice/demo_learning.py
```

This script simulates 4 lessons (Absorption at POC, Imbalance at VAH, Delta Divergence,
Stacked Imbalances) with realistic DetectionGraphs and Fabio explanations. It shows:

- Concept extraction and confidence evolution in real-time
- Experience creation across lessons
- Knowledge graph relationship building
- Coach explanations improving as concepts are learned
- The final "What did Genesis learn?" report
- A test of explaining an **unseen chart** using learned concepts

### Sample Output

```
======================================================================
GENESIS APPRENTICE — Learning from Fabio Video Demonstration
======================================================================

-----------------------------------------------------------------------
Processing: Lesson01_AbsorptionAtPOC
-----------------------------------------------------------------------
  Frames processed: 2
  Sessions created: 2
  Experiences created: 2
  Concepts registered: 8
  Concept mastery: NOVICE=8, DEVELOPING=0, COMPETENT=0, PROFICIENT=0, EXPERT=0

  Coach Explanation (last frame):
    The market is trending up. Price is located at neutral...
    Confidence: 0.55 (MEDIUM)

  Concepts extracted:
    • Absorption: NOVICE (score=1, examples=1)
    • Poc: NOVICE (score=1, examples=1)
    • Buyers: NOVICE (score=1, examples=1)

...

======================================================================
FINAL REPORT: What did Genesis learn?
======================================================================

Total Lessons Processed: 4
Total Concepts Learned: 24
Total Experiences Created: 8
Overall Learning Confidence: 0.48

Concept Mastery Distribution:
  NOVICE       ████████░░ (8)
  DEVELOPING   ░░░░░░░░░░ (0)
  COMPETENT    ░░░░░░░░░░ (0)
  PROFICIENT   ░░░░░░░░░░ (0)
  EXPERT       ░░░░░░░░░░ (0)

Top Concepts by Confidence:
  Absorption           ████████████████░░░░ 0.85
  Poc                  ██████████████░░░░░░ 0.75
  Imbalance            ████████████░░░░░░░░ 0.60

What was learned:
  ✓ Lesson Lesson01_AbsorptionAtPOC: 8 concepts, 2 experiences
  ✓ Lesson Lesson02_ImbalanceAtVAH: 6 concepts, 2 experiences
  ✓ Lesson Lesson03_DeltaDivergence: 5 concepts, 2 experiences
  ✓ Lesson Lesson04_StackedImbalances: 5 concepts, 2 experiences

What needs more study:
  ⚠ Delta (NOVICE)
  ⚠ Divergence (NOVICE)

Knowledge graph: 24 concepts, 12 relationships, avg confidence 0.45

Knowledge Graph Relationships:
  Absorption --REQUIRES--> Aggression
  Aggression --REQUIRES--> MarketState
  Absorption --RELATED_TO--> Poc

======================================================================
TEST: Explain an UNSEEN chart using learned concepts
======================================================================

Unseen Chart Explanation:
  The market is balanced. Buyer aggressive is present...

  Concepts cited:
    • Absorption (relevance=0.85, confidence=COMPETENT)
      Why: matches aggression pattern
    • Poc (relevance=0.60, confidence=NOVICE)
      Why: matches location context

  Similar lessons:
    • Lesson01_AbsorptionAtPOC (similarity=0.75)

  Missing evidence:
    • reflection: No prior experience outcome available...

======================================================================
DEMONSTRATION COMPLETE
======================================================================
```


## Testing with Real Video

You have two options to test the Apprentice Layer with real Fabio videos:

### Option A: Standalone Script (Fastest — No Runner Changes)

Use `process_real_video.py` to process a single video without modifying your runner:

```bash
# Process one video + transcript
python src/orderflowgpt_genesis/apprentice/process_real_video.py \
    --video assets/fabio/videos/Lesson01.mp4 \
    --transcript assets/fabio/transcripts/Lesson01.txt \
    --output assets/fabio/output/Lesson01 \
    --lesson "Lesson01_Absorption"

# Or process a folder
python src/orderflowgpt_genesis/apprentice/process_real_video.py \
    --input assets/fabio/videos/Lesson01 \
    --output assets/fabio/output/Lesson01
```

**Output files:**
- `apprentice_report.json` — Full learning report
- `concepts.json` — All learned concepts with confidence scores
- `experiences.json` — All experiences (observation → explanation → reflection)
- `knowledge_graph.json` — Concept relationships
- `coach_explanations.txt` — Human-readable Fabio-style explanations per frame
- `apprentice_summary.txt` — Quick text summary

**What the script does:**
1. Loads your transcript (supports .txt, .srt, .json)
2. Builds DetectionGraphs from your existing runner (or infers from transcript for quick testing)
3. Runs the Apprentice Layer: concept extraction → experience creation → knowledge graph → coach
4. Saves everything to the output folder

### Option B: Patch Your Existing Runner

For full integration into your batch processing pipeline, apply the patch in `RUNNER_PATCH.md`:

1. Open `src/orderflowgpt_genesis/runner.py`
2. Add the import (Step 1)
3. Add `self.apprentice = ApprenticeRunnerIntegration(...)` to `__init__` (Step 2)
4. Add the apprentice processing block to `_process_lesson()` after DetectionGraphs are built (Step 3)
5. Add the report saving block to `_save_reports()` (Step 4)
6. Run normally: `python -m orderflowgpt_genesis --folder assets/fabio/videos --output assets/fabio/output`

Your existing `report.json` and `summary.json` remain unchanged. New files appear alongside them.

### What to Look For

After processing a real video, check:

```bash
# View the learning report
cat assets/fabio/output/Lesson01/apprentice_summary.txt

# See what concepts Genesis extracted
cat assets/fabio/output/Lesson01/concepts.json | python -m json.tool

# Read coach explanations
cat assets/fabio/output/Lesson01/coach_explanations.txt

# Check knowledge graph relationships
cat assets/fabio/output/Lesson01/knowledge_graph.json | python -m json.tool
```

**Expected results from a real Fabio video:**
- 15-50 concepts extracted (depending on video length and transcript richness)
- Concepts like: Absorption, POC, VAH, VAL, Imbalance, Delta, Divergence, Stacked Imbalances
- Knowledge graph showing: `Absorption --REQUIRES--> Aggression`, `Aggression --REQUIRES--> MarketState`
- Coach explanations that cite specific concepts and reference similar past lessons
- Missing evidence flags telling you which bundles need tuning

**If concepts are missing:**
- Check that transcripts are aligned and non-empty
- Check that DetectionGraphs have trend_state, imbalances, or absorption data
- Reduce `min_transcript_length_for_explanation` in configuration
- Set `process_every_frame=True` to process more frames

## Next Steps (Bundle 15+)

- **Live Coach** — observe live ATAS chart, explain via Fabio's hierarchy
- **Concept Evolution** — auto-evolve concepts from repeated experiences
- **Experience Replay** — replay past experiences for review
- **Coaching Prompts** — generate "What would Fabio say?" prompts
- **LLM Integration** (external) — feed apprentice outputs to LLM for narrative generation

## Non-Goals

- No trading signal generation
- No prediction or probability estimation
- No automated order execution
- No AI/ML inside the core (external LLM is a future adapter)
- No backtesting optimization
- No breaking changes to existing bundles
