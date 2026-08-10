"""Runner Integration for the Genesis Apprentice Layer.

This module provides deterministic hooks to wire the Apprentice Layer into
the existing GenesisRunner (Bundle 13.5). It processes lessons, builds
learning artifacts, and produces the "What did Genesis learn?" report.

No breaking changes to existing runner behavior. The integration is
compositional: the existing runner calls these hooks after building
DetectionGraphs and aligning transcripts.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Tuple

from .concepts import Concept, ConceptEngine, ConceptStatistics
from .experiences import Experience, ExperienceEngine, ExperienceStatistics
from .knowledge_graph import KnowledgeGraph, KnowledgeGraphStatistics
from .learning_loop import LearningLoop, LearningLoopStatistics, LearningSession
from .reasoning import DecisionHierarchyResult, DecisionHierarchyStatistics
from .integration import ApprenticeLayer, ApprenticeConfiguration, ApprenticeResult
from .coach import ChartExplainer, CoachConfiguration, ExplanationResult, CoachStatistics


@dataclass(frozen=True)
class FrameApprenticeResult:
    """Apprentice result for one frame within a lesson."""
    frame_index: int
    frame_timestamp: str
    session_id: str
    hierarchy_result: Optional[DecisionHierarchyResult]
    experience_id: Optional[str]
    concept_references: Tuple[str, ...]
    coach_explanation: Optional[ExplanationResult]
    explanation_text: str


@dataclass(frozen=True)
class LessonApprenticeResult:
    """Aggregate apprentice result for one complete lesson."""
    lesson_reference: str
    frame_results: Tuple[FrameApprenticeResult, ...]
    sessions: Tuple[LearningSession, ...]
    concepts: Tuple[Concept, ...]
    experiences: Tuple[Experience, ...]
    knowledge_graph: KnowledgeGraph
    concept_statistics: ConceptStatistics
    experience_statistics: ExperienceStatistics
    learning_statistics: LearningLoopStatistics
    coach_statistics: CoachStatistics


@dataclass(frozen=True)
class ApprenticeReport:
    """Immutable "What did Genesis learn?" report for one or more lessons.

    This is the primary output of runner integration. It summarizes
    everything Genesis learned, observed, and understood.
    """
    report_id: str
    lesson_results: Tuple[LessonApprenticeResult, ...]
    total_concepts_learned: int
    total_experiences_created: int
    total_sessions_completed: int
    total_coach_explanations: int
    total_frames_processed: int
    concept_mastery_distribution: Tuple[Tuple[str, int], ...]
    top_concepts_by_confidence: Tuple[Tuple[str, Decimal], ...]
    knowledge_graph_summary: str
    what_was_learned: Tuple[str, ...]
    what_needs_more_study: Tuple[str, ...]
    overall_learning_confidence: Decimal
    report_timestamp: str

    def __post_init__(self):
        if not (Decimal("0") <= self.overall_learning_confidence <= Decimal("1")):
            raise ValueError("overall_learning_confidence must be in [0, 1]")


@dataclass(frozen=True)
class RunnerIntegrationConfiguration:
    """Configuration for how the Apprentice Layer integrates with the runner."""
    apprentice_configuration: ApprenticeConfiguration = field(default_factory=ApprenticeConfiguration)
    coach_configuration: CoachConfiguration = field(default_factory=CoachConfiguration)
    process_every_frame: bool = False
    process_key_frames_only: bool = True
    key_frame_interval: int = 30
    enable_coach: bool = True
    enable_concept_extraction: bool = True
    enable_experience_creation: bool = True
    enable_knowledge_graph_building: bool = True
    min_transcript_length_for_explanation: int = 10
    max_frames_per_lesson: int = 100
    report_concept_detail_level: str = "summary"  # "summary" or "full"

    def __post_init__(self):
        if self.key_frame_interval < 1:
            raise ValueError("key_frame_interval must be at least 1")
        if self.max_frames_per_lesson < 1:
            raise ValueError("max_frames_per_lesson must be at least 1")
        if self.report_concept_detail_level not in ("summary", "full"):
            raise ValueError("report_concept_detail_level must be 'summary' or 'full'")


@dataclass(frozen=True)
class RunnerIntegrationStatistics:
    """Statistics over runner integration operations."""
    total_lessons_processed: int = 0
    total_frames_processed: int = 0
    total_frames_skipped: int = 0
    total_explanations_generated: int = 0
    total_concepts_extracted: int = 0
    total_experiences_created: int = 0
    total_sessions_completed: int = 0
    average_frames_per_lesson: Decimal = Decimal("0")
    processing_errors: Tuple[str, ...] = ()


class ApprenticeLessonProcessor:
    """Deterministic processor that runs the Apprentice Layer over one lesson.

    Given a sequence of frames (each with a DetectionGraph and aligned
    transcript text), this processor:
    1. Runs the learning loop for each frame
    2. Extracts concepts from teacher explanations
    3. Creates experiences
    4. Builds the knowledge graph
    5. Optionally runs the Live Coach

    No AI, no ML, no prediction, no trade recommendation.
    """

    def __init__(
        self,
        apprentice_layer: ApprenticeLayer,
        configuration: Optional[RunnerIntegrationConfiguration] = None,
    ):
        self._apprentice = apprentice_layer
        self._configuration = configuration or RunnerIntegrationConfiguration()
        self._explainer: Optional[ChartExplainer] = None
        if self._configuration.enable_coach:
            self._explainer = ChartExplainer(apprentice_layer, self._configuration.coach_configuration)

    def process_lesson(
        self,
        lesson_reference: str,
        frames: Tuple[Tuple[int, str, object, str], ...],
    ) -> LessonApprenticeResult:
        """Process one lesson through the apprentice layer.

        Args:
            lesson_reference: The lesson identifier (e.g., "Lesson01").
            frames: Tuple of (frame_index, frame_timestamp, detection_graph,
                    teacher_explanation) for each frame to process.

        Returns:
            LessonApprenticeResult with all learning artifacts.
        """
        frame_results = []
        processed_count = 0

        for frame_idx, timestamp, graph, explanation in frames:
            if processed_count >= self._configuration.max_frames_per_lesson:
                break

            should_process = self._should_process_frame(frame_idx, explanation)
            if not should_process:
                continue

            result = self._process_frame(lesson_reference, frame_idx, timestamp, graph, explanation)
            frame_results.append(result)
            processed_count += 1

        # Build aggregate result
        sessions = self._apprentice.all_sessions()
        concepts = self._apprentice.all_concepts()
        experiences = self._apprentice.all_experiences()
        graph = self._apprentice.get_knowledge_graph()

        return LessonApprenticeResult(
            lesson_reference=lesson_reference,
            frame_results=tuple(frame_results),
            sessions=sessions,
            concepts=concepts,
            experiences=experiences,
            knowledge_graph=graph,
            concept_statistics=self._apprentice._concept_engine._compute_statistics(),
            experience_statistics=self._apprentice._experience_engine._compute_statistics(),
            learning_statistics=self._apprentice._learning_loop._compute_statistics(),
            coach_statistics=self._explainer.get_statistics() if self._explainer else CoachStatistics(),
        )

    def _should_process_frame(self, frame_idx: int, explanation: str) -> bool:
        """Deterministic frame selection logic."""
        if not self._configuration.process_every_frame:
            if self._configuration.process_key_frames_only:
                if frame_idx % self._configuration.key_frame_interval != 0:
                    return False
        if len(explanation.strip()) < self._configuration.min_transcript_length_for_explanation:
            return False
        return True

    def _process_frame(
        self,
        lesson_reference: str,
        frame_idx: int,
        timestamp: str,
        detection_graph,
        teacher_explanation: str,
    ) -> FrameApprenticeResult:
        """Process one frame through the full apprentice pipeline."""
        session_id = f"{lesson_reference}_frame{frame_idx:04d}"

        # Run the apprentice layer for this frame
        apprentice_result = self._apprentice.process_frame(
            session_id=session_id,
            lesson_reference=lesson_reference,
            detection_graph=detection_graph,
            teacher_explanation=teacher_explanation,
            source_reference=f"{lesson_reference} frame {frame_idx}",
            timestamp=timestamp,
        )

        # Run coach if enabled
        coach_explanation = None
        explanation_text = ""
        if self._explainer is not None and self._configuration.enable_coach:
            coach_explanation = self._explainer.explain(detection_graph)
            explanation_text = self._explainer._format_explanation(coach_explanation)

        return FrameApprenticeResult(
            frame_index=frame_idx,
            frame_timestamp=timestamp,
            session_id=session_id,
            hierarchy_result=apprentice_result.hierarchy_result,
            experience_id=apprentice_result.experience_created.experience_id if apprentice_result.experience_created else None,
            concept_references=tuple(c.concept_id for c in apprentice_result.concepts_touched) if apprentice_result.concepts_touched else (),  # Fixed: return tuple
            coach_explanation=coach_explanation,
            explanation_text=explanation_text,
        )


class ApprenticeReportBuilder:
    """Deterministic builder for the "What did Genesis learn?" report.

    Consumes one or more LessonApprenticeResult objects and produces a
    canonical ApprenticeReport summarizing all learning.
    """

    def build(self, lesson_results: Tuple[LessonApprenticeResult, ...], timestamp: str) -> ApprenticeReport:
        """Build the apprentice report from lesson results."""
        if not lesson_results:
            return self._empty_report(timestamp)

        total_concepts = sum(len(lr.concepts) for lr in lesson_results)
        total_experiences = sum(len(lr.experiences) for lr in lesson_results)
        total_sessions = sum(len(lr.sessions) for lr in lesson_results)
        total_explanations = sum(lr.coach_statistics.total_explanations for lr in lesson_results)
        total_frames = sum(len(lr.frame_results) for lr in lesson_results)

        # Concept mastery distribution
        from collections import Counter
        level_counts = Counter()
        for lr in lesson_results:
            for c in lr.concepts:
                level_counts[c.confidence.level.name] += 1

        # Top concepts by confidence
        all_concepts = []
        for lr in lesson_results:
            all_concepts.extend(lr.concepts)
        top_concepts = sorted(all_concepts, key=lambda c: c.confidence.score, reverse=True)[:10]
        top_concept_tuples = tuple((c.definition.name, c.confidence.score) for c in top_concepts)

        # What was learned
        learned = []
        for lr in lesson_results:
            learned.append(f"Lesson {lr.lesson_reference}: {len(lr.concepts)} concepts, {len(lr.experiences)} experiences")
        learned = tuple(learned)

        # What needs more study
        needs_study = []
        for lr in lesson_results:
            for c in lr.concepts:
                if c.confidence.level.name in ("NOVICE", "DEVELOPING"):
                    needs_study.append(f"{c.definition.name} ({c.confidence.level.name})")
        needs_study = tuple(sorted(set(needs_study))[:20])

        # Overall confidence
        if lesson_results:
            avg_conf = sum(
                lr.learning_statistics.average_session_confidence
                for lr in lesson_results
            ) / Decimal(str(len(lesson_results)))
        else:
            avg_conf = Decimal("0")

        # Knowledge graph summary
        graph = lesson_results[-1].knowledge_graph if lesson_results else KnowledgeGraph()
        graph_summary = (
            f"Knowledge graph: {graph.statistics.total_concepts} concepts, "
            f"{graph.statistics.total_relationships} relationships, "
            f"avg confidence {graph.statistics.average_confidence}"
        )

        return ApprenticeReport(
            report_id=self._derive_report_id(lesson_results, timestamp),
            lesson_results=lesson_results,
            total_concepts_learned=total_concepts,
            total_experiences_created=total_experiences,
            total_sessions_completed=total_sessions,
            total_coach_explanations=total_explanations,
            total_frames_processed=total_frames,
            concept_mastery_distribution=tuple(level_counts.items()),
            top_concepts_by_confidence=top_concept_tuples,
            knowledge_graph_summary=graph_summary,
            what_was_learned=learned,
            what_needs_more_study=needs_study,
            overall_learning_confidence=min(avg_conf, Decimal("1")),
            report_timestamp=timestamp,
        )

    def _empty_report(self, timestamp: str) -> ApprenticeReport:
        """Return an empty report when no lessons were processed."""
        return ApprenticeReport(
            report_id=f"empty_{timestamp}",
            lesson_results=(),
            total_concepts_learned=0,
            total_experiences_created=0,
            total_sessions_completed=0,
            total_coach_explanations=0,
            total_frames_processed=0,
            concept_mastery_distribution=(),
            top_concepts_by_confidence=(),
            knowledge_graph_summary="No knowledge graph built.",
            what_was_learned=("No lessons processed.",),
            what_needs_more_study=(),
            overall_learning_confidence=Decimal("0"),
            report_timestamp=timestamp,
        )

    def _derive_report_id(self, lesson_results: Tuple[LessonApprenticeResult, ...], timestamp: str) -> str:
        """Derive a deterministic report id."""
        from hashlib import sha256
        refs = ":".join(lr.lesson_reference for lr in lesson_results)
        seed = f"report:{refs}:{timestamp}".encode("utf-8")
        return sha256(seed).hexdigest()[:16]


class ApprenticeRunnerIntegration:
    """Deterministic integration wrapper for the GenesisRunner.

    This class is designed to be composed with the existing GenesisRunner.
    The existing runner calls methods on this integration after building
    DetectionGraphs and aligning transcripts.

    Usage in runner.py:

        integration = ApprenticeRunnerIntegration()
        for lesson in lessons:
            detection_graphs = ...  # existing pipeline
            transcripts = ...       # existing alignment
            lesson_result = integration.process_lesson(lesson_id, detection_graphs, transcripts)
        report = integration.build_report(timestamp)
        # save report to report.json
    """

    def __init__(self, configuration: Optional[RunnerIntegrationConfiguration] = None):
        self._configuration = configuration or RunnerIntegrationConfiguration()
        self._apprentice = ApprenticeLayer(self._configuration.apprentice_configuration)
        self._processor = ApprenticeLessonProcessor(self._apprentice, self._configuration)
        self._report_builder = ApprenticeReportBuilder()
        self._lesson_results: list[LessonApprenticeResult] = []
        self._statistics = RunnerIntegrationStatistics()

    def process_lesson(
        self,
        lesson_reference: str,
        frames: Tuple[Tuple[int, str, object, str], ...],
    ) -> LessonApprenticeResult:
        """Process one lesson and accumulate results.

        Args:
            lesson_reference: Lesson identifier.
            frames: Tuple of (frame_index, timestamp, detection_graph, explanation).

        Returns:
            LessonApprenticeResult.
        """
        result = self._processor.process_lesson(lesson_reference, frames)
        self._lesson_results.append(result)

        # Update statistics
        processed = len(result.frame_results)
        skipped = len(frames) - processed
        errors = ()
        self._statistics = RunnerIntegrationStatistics(
            total_lessons_processed=len(self._lesson_results),
            total_frames_processed=self._statistics.total_frames_processed + processed,
            total_frames_skipped=self._statistics.total_frames_skipped + skipped,
            total_explanations_generated=self._statistics.total_explanations_generated + result.coach_statistics.total_explanations,
            total_concepts_extracted=self._statistics.total_concepts_extracted + result.concept_statistics.total_concepts,
            total_experiences_created=self._statistics.total_experiences_created + result.experience_statistics.total_experiences,
            total_sessions_completed=self._statistics.total_sessions_completed + sum(1 for s in result.sessions if s.is_complete),
            average_frames_per_lesson=Decimal(str(self._statistics.total_frames_processed + processed)) / Decimal(str(len(self._lesson_results))),
            processing_errors=self._statistics.processing_errors + errors,
        )

        return result

    def build_report(self, timestamp: str) -> ApprenticeReport:
        """Build the final "What did Genesis learn?" report."""
        return self._report_builder.build(tuple(self._lesson_results), timestamp)

    def get_statistics(self) -> RunnerIntegrationStatistics:
        """Return current integration statistics."""
        return self._statistics

    def get_apprentice_layer(self) -> ApprenticeLayer:
        """Return the underlying apprentice layer for direct access."""
        return self._apprentice

    def reset(self) -> None:
        """Reset all accumulated state. Deterministic for batch processing."""
        self._apprentice = ApprenticeLayer(self._configuration.apprentice_configuration)
        self._processor = ApprenticeLessonProcessor(self._apprentice, self._configuration)
        self._lesson_results = []
        self._statistics = RunnerIntegrationStatistics()
