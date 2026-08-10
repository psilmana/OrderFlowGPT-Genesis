"""Genesis Apprentice Learning Demonstration.

This script demonstrates how Genesis learns from Fabio's teaching by
simulating realistic video processing scenarios. It creates synthetic
DetectionGraphs and transcript alignments based on actual Order Flow
concepts, then runs the full Apprentice Layer to show:

- Concept extraction and evolution
- Experience creation and completion
- Knowledge graph growth
- Coach explanation quality over time
- The "What did Genesis learn?" report

Run this script from your repo root:
    python src/orderflowgpt_genesis/apprentice/demo_learning.py

No video files required. All data is deterministic and synthetic.
"""

from decimal import Decimal
from dataclasses import dataclass, field
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Simulated DetectionGraph — mimics Bundle 1-13 output structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeTrendState:
    name: str

@dataclass(frozen=True)
class FakeTrend:
    trend_state: FakeTrendState

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
class FakeSession:
    session_type: str

@dataclass(frozen=True)
class FakeConfluence:
    confluence_type: str

@dataclass(frozen=True)
class FakeDetectionGraph:
    """Synthetic DetectionGraph with realistic Order Flow data."""
    graph_id: str
    timestamp: str
    trend_state: Optional[FakeTrend] = None
    trading_session: Optional[FakeSession] = None
    footprint_imbalances: Optional[FakeImbalances] = None
    absorption_result: Optional[FakeAbsorptions] = None
    footprint_delta: Optional[FakeDelta] = None
    confluence: Optional[FakeConfluence] = None


# ---------------------------------------------------------------------------
# Lesson scenarios — realistic Fabio teaching moments
# ---------------------------------------------------------------------------

LESSON_SCENARIOS = [
    {
        "lesson": "Lesson01_AbsorptionAtPOC",
        "frames": [
            {
                "timestamp": "00:01:15",
                "graph": FakeDetectionGraph(
                    graph_id="L01F001",
                    timestamp="00:01:15",
                    trend_state=FakeTrend(FakeTrendState("BALANCED")),
                    trading_session=FakeSession("RTH"),
                    footprint_imbalances=FakeImbalances((
                        FakeImbalance("BID", "5000.00"),
                        FakeImbalance("BID", "5000.25"),
                        FakeImbalance("ASK", "5000.50"),
                    )),
                    absorption_result=FakeAbsorptions((
                        FakeAbsorption("BID", "5000.00"),
                    )),
                    footprint_delta=FakeDelta(()),
                    confluence=FakeConfluence("STRONG_CONFLUENCE"),
                ),
                "explanation": (
                    "Here we see absorption at the POC. The buyers are aggressive "
                    "but the passive side is absorbing at this level. This tells us "
                    "the market is balanced here and we should wait for a resolution. "
                    "Absorption means the passive side is holding — they are not giving up. "
                    "This is a key concept in Order Flow trading."
                ),
            },
            {
                "timestamp": "00:02:30",
                "graph": FakeDetectionGraph(
                    graph_id="L01F002",
                    timestamp="00:02:30",
                    trend_state=FakeTrend(FakeTrendState("TRENDING_UP")),
                    trading_session=FakeSession("RTH"),
                    footprint_imbalances=FakeImbalances((
                        FakeImbalance("BID", "5001.00"),
                        FakeImbalance("BID", "5001.25"),
                    )),
                    absorption_result=None,
                    footprint_delta=FakeDelta(()),
                    confluence=FakeConfluence("WEAK_CONFLUENCE"),
                ),
                "explanation": (
                    "Now we broke out. The trend is up and we see buyer imbalance "
                    "without absorption. This is a continuation pattern. The POC "
                    "was defended and now price is moving higher."
                ),
            },
        ],
    },
    {
        "lesson": "Lesson02_ImbalanceAtVAH",
        "frames": [
            {
                "timestamp": "00:00:45",
                "graph": FakeDetectionGraph(
                    graph_id="L02F001",
                    timestamp="00:00:45",
                    trend_state=FakeTrend(FakeTrendState("TRENDING_UP")),
                    trading_session=FakeSession("RTH"),
                    footprint_imbalances=FakeImbalances((
                        FakeImbalance("ASK", "5010.00"),
                        FakeImbalance("ASK", "5010.25"),
                        FakeImbalance("ASK", "5010.50"),
                    )),
                    absorption_result=None,
                    footprint_delta=FakeDelta(()),
                    confluence=FakeConfluence("NO_CONFLUENCE"),
                ),
                "explanation": (
                    "We have a seller imbalance at the VAH. In a trending market, "
                    "this could be a pullback to value. The VAH is the Value Area High — "
                    "the top of the value area where 70 percent of volume traded. "
                    "When price returns to VAH in a trend, it often finds support."
                ),
            },
            {
                "timestamp": "00:01:20",
                "graph": FakeDetectionGraph(
                    graph_id="L02F002",
                    timestamp="00:01:20",
                    trend_state=FakeTrend(FakeTrendState("TRENDING_UP")),
                    trading_session=FakeSession("RTH"),
                    footprint_imbalances=FakeImbalances((
                        FakeImbalance("BID", "5010.00"),
                    )),
                    absorption_result=FakeAbsorptions((
                        FakeAbsorption("BID", "5010.00"),
                    )),
                    footprint_delta=FakeDelta(()),
                    confluence=FakeConfluence("STRONG_CONFLUENCE"),
                ),
                "explanation": (
                    "Now we see absorption at the VAH. The seller imbalance was absorbed "
                    "and buyers stepped in. This confirms the VAH as support in the uptrend. "
                    "Absorption at a key level is a strong signal."
                ),
            },
        ],
    },
    {
        "lesson": "Lesson03_DeltaDivergence",
        "frames": [
            {
                "timestamp": "00:02:10",
                "graph": FakeDetectionGraph(
                    graph_id="L03F001",
                    timestamp="00:02:10",
                    trend_state=FakeTrend(FakeTrendState("TRENDING_DOWN")),
                    trading_session=FakeSession("RTH"),
                    footprint_imbalances=FakeImbalances((
                        FakeImbalance("ASK", "4990.00"),
                    )),
                    absorption_result=None,
                    footprint_delta=FakeDelta(()),
                    confluence=FakeConfluence("WEAK_CONFLUENCE"),
                ),
                "explanation": (
                    "Delta divergence at the low. Price is making lower lows but delta "
                    "is not confirming. This means sellers are exhausted. Delta divergence "
                    "is when price goes one way but the cumulative delta goes the other way. "
                    "It is a key concept for identifying exhaustion."
                ),
            },
            {
                "timestamp": "00:03:00",
                "graph": FakeDetectionGraph(
                    graph_id="L03F002",
                    timestamp="00:03:00",
                    trend_state=FakeTrend(FakeTrendState("BALANCED")),
                    trading_session=FakeSession("RTH"),
                    footprint_imbalances=FakeImbalances(()),
                    absorption_result=FakeAbsorptions((
                        FakeAbsorption("BID", "4990.00"),
                        FakeAbsorption("BID", "4990.25"),
                    )),
                    footprint_delta=FakeDelta(()),
                    confluence=FakeConfluence("STRONG_CONFLUENCE"),
                ),
                "explanation": (
                    "The divergence resolved with absorption at the low. Buyers absorbed "
                    "the selling and the market balanced. This is how delta divergence "
                    "and absorption work together to show a potential reversal."
                ),
            },
        ],
    },
    {
        "lesson": "Lesson04_StackedImbalances",
        "frames": [
            {
                "timestamp": "00:01:00",
                "graph": FakeDetectionGraph(
                    graph_id="L04F001",
                    timestamp="00:01:00",
                    trend_state=FakeTrend(FakeTrendState("AUCTION_DOWN")),
                    trading_session=FakeSession("RTH"),
                    footprint_imbalances=FakeImbalances((
                        FakeImbalance("ASK", "4980.00"),
                        FakeImbalance("ASK", "4980.25"),
                        FakeImbalance("ASK", "4980.50"),
                        FakeImbalance("ASK", "4980.75"),
                    )),
                    absorption_result=None,
                    footprint_delta=FakeDelta(()),
                    confluence=FakeConfluence("NO_CONFLUENCE"),
                ),
                "explanation": (
                    "Stacked imbalances at support. We see four consecutive ask imbalances "
                    "which means sellers are very aggressive. But we are at a support level. "
                    "Stacked imbalances show sustained aggression. When they appear at a "
                    "key level, it is significant."
                ),
            },
            {
                "timestamp": "00:01:45",
                "graph": FakeDetectionGraph(
                    graph_id="L04F002",
                    timestamp="00:01:45",
                    trend_state=FakeTrend(FakeTrendState("AUCTION_DOWN")),
                    trading_session=FakeSession("RTH"),
                    footprint_imbalances=FakeImbalances((
                        FakeImbalance("ASK", "4980.00"),
                        FakeImbalance("ASK", "4980.25"),
                    )),
                    absorption_result=FakeAbsorptions((
                        FakeAbsorption("BID", "4980.00"),
                        FakeAbsorption("BID", "4980.25"),
                        FakeAbsorption("BID", "4980.50"),
                    )),
                    footprint_delta=FakeDelta(()),
                    confluence=FakeConfluence("STRONG_CONFLUENCE"),
                ),
                "explanation": (
                    "Now we see absorption appearing within the stacked imbalances. "
                    "The passive side is starting to hold. This is the combination of "
                    "stacked imbalances and absorption that Fabio teaches as a key setup. "
                    "The aggression is there but it is being absorbed."
                ),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo():
    """Run the full learning demonstration."""
    from orderflowgpt_genesis.apprentice import (
        ApprenticeRunnerIntegration,
        RunnerIntegrationConfiguration,
    )

    print("=" * 70)
    print("GENESIS APPRENTICE — Learning from Fabio Video Demonstration")
    print("=" * 70)
    print()

    # Initialize integration
    config = RunnerIntegrationConfiguration(
        process_every_frame=True,
        process_key_frames_only=False,
        enable_coach=True,
        enable_concept_extraction=True,
        enable_experience_creation=True,
        enable_knowledge_graph_building=True,
        min_transcript_length_for_explanation=5,
        max_frames_per_lesson=100,
    )
    integration = ApprenticeRunnerIntegration(config)

    # Process each lesson
    for scenario in LESSON_SCENARIOS:
        lesson_name = scenario["lesson"]
        print(f"\n{'─' * 70}")
        print(f"Processing: {lesson_name}")
        print(f"{'─' * 70}")

        frames = tuple(
            (i, frame["timestamp"], frame["graph"], frame["explanation"])
            for i, frame in enumerate(scenario["frames"])
        )

        result = integration.process_lesson(lesson_name, frames)

        # Show what was learned in this lesson
        print(f"  Frames processed: {len(result.frame_results)}")
        print(f"  Sessions created: {len(result.sessions)}")
        print(f"  Experiences created: {len(result.experiences)}")
        print(f"  Concepts registered: {result.concept_statistics.total_concepts}")
        print(f"  Concept mastery: NOVICE={result.concept_statistics.novice_count}, "
              f"DEVELOPING={result.concept_statistics.developing_count}, "
              f"COMPETENT={result.concept_statistics.competent_count}, "
              f"PROFICIENT={result.concept_statistics.proficient_count}, "
              f"EXPERT={result.concept_statistics.expert_count}")

        # Show coach output for the last frame
        if result.frame_results:
            last_frame = result.frame_results[-1]
            if last_frame.coach_explanation:
                print(f"\n  Coach Explanation (last frame):")
                print(f"    {last_frame.coach_explanation.narrative.summary[:120]}...")
                print(f"    Confidence: {last_frame.coach_explanation.overall_confidence} "
                      f"({last_frame.coach_explanation.explanation_confidence.name})")

        # Show concepts extracted
        if result.concepts:
            print(f"\n  Concepts extracted:")
            for concept in result.concepts[:5]:
                print(f"    • {concept.definition.name}: {concept.confidence.level.name} "
                      f"(score={concept.confidence.score}, examples={concept.confidence.evidence_count})")

    # Build final report
    print(f"\n{'=' * 70}")
    print("FINAL REPORT: What did Genesis learn?")
    print(f"{'=' * 70}")

    report = integration.build_report("2026-08-04T18:00:00Z")

    print(f"\nTotal Lessons Processed: {len(report.lesson_results)}")
    print(f"Total Concepts Learned: {report.total_concepts_learned}")
    print(f"Total Experiences Created: {report.total_experiences_created}")
    print(f"Total Sessions Completed: {report.total_sessions_completed}")
    print(f"Total Coach Explanations: {report.total_coach_explanations}")
    print(f"Total Frames Processed: {report.total_frames_processed}")
    print(f"Overall Learning Confidence: {report.overall_learning_confidence}")

    print(f"\nConcept Mastery Distribution:")
    for level, count in report.concept_mastery_distribution:
        bar = "█" * count + "░" * (10 - count)
        print(f"  {level:12s} {bar} ({count})")

    print(f"\nTop Concepts by Confidence:")
    for name, score in report.top_concepts_by_confidence[:10]:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {name:20s} {bar} {score}")

    print(f"\nWhat was learned:")
    for item in report.what_was_learned:
        print(f"  ✓ {item}")

    print(f"\nWhat needs more study:")
    if report.what_needs_more_study:
        for item in report.what_needs_more_study:
            print(f"  ⚠ {item}")
    else:
        print(f"  ✓ All concepts have sufficient study")

    print(f"\n{report.knowledge_graph_summary}")

    # Show knowledge graph relationships
    layer = integration.get_apprentice_layer()
    graph = layer.get_knowledge_graph()
    if graph.relationships:
        print(f"\nKnowledge Graph Relationships:")
        for rel in graph.relationships[:15]:
            print(f"  {rel.source_concept} --{rel.relationship_type.name}--> {rel.target_concept}")

    # Demonstrate: explain a NEW unseen chart using learned concepts
    print(f"\n{'=' * 70}")
    print("TEST: Explain an UNSEEN chart using learned concepts")
    print(f"{'=' * 70}")

    unseen_graph = FakeDetectionGraph(
        graph_id="UNSEEN001",
        timestamp="00:05:00",
        trend_state=FakeTrend(FakeTrendState("BALANCED")),
        trading_session=FakeSession("RTH"),
        footprint_imbalances=FakeImbalances((
            FakeImbalance("BID", "5005.00"),
            FakeImbalance("BID", "5005.25"),
        )),
        absorption_result=FakeAbsorptions((
            FakeAbsorption("BID", "5005.00"),
        )),
        footprint_delta=FakeDelta(()),
        confluence=FakeConfluence("STRONG_CONFLUENCE"),
    )

    explainer = layer._explainer if hasattr(layer, '_explainer') and layer._explainer else None
    if explainer is None:
        from orderflowgpt_genesis.apprentice import ChartExplainer
        explainer = ChartExplainer(layer)

    explanation = explainer.explain(unseen_graph)
    print(f"\nUnseen Chart Explanation:")
    print(f"  {explanation.narrative.summary}")
    print(f"\n  Concepts cited:")
    for citation in explanation.concept_citations:
        print(f"    • {citation.concept_name} (relevance={citation.relevance_score}, "
              f"confidence={citation.confidence_level.name})")
        print(f"      Why: {citation.why_relevant}")
    print(f"\n  Similar lessons:")
    for lesson in explanation.similar_lessons:
        print(f"    • {lesson.lesson_reference} (similarity={lesson.similarity_score})")
    print(f"\n  Missing evidence:")
    for missing in explanation.missing_evidence:
        print(f"    • {missing.field_name}: {missing.suggestion}")

    print(f"\n{'=' * 70}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_demo()
