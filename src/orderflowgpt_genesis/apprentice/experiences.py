"""Experience Engine for the Genesis Apprentice Layer.

Experience is the central learning object. Each Experience contains
observation, teacher explanation, evidence, reasoning, outcome, and
reflection. Experiences become Memory. Repeated Experiences become
Knowledge.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Tuple

from .reasoning import DecisionHierarchyResult


class ExperienceOutcomeType(Enum):
    """Deterministic outcome classifications for an experience."""
    VALIDATED = auto()
    PARTIALLY_VALIDATED = auto()
    INVALIDATED = auto()
    PENDING = auto()
    INSUFFICIENT_DATA = auto()


class ExperienceConfidence(Enum):
    """Deterministic confidence in an experience's quality."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@dataclass(frozen=True)
class ExperienceObservation:
    """What Genesis observed in one learning moment."""
    what_was_seen: str
    detection_graph_reference: str
    frame_reference: str
    timestamp: str
    market_context_summary: str = ""

    def __post_init__(self):
        if not self.what_was_seen:
            raise ValueError("what_was_seen is required")
        if not self.detection_graph_reference:
            raise ValueError("detection_graph_reference is required")


@dataclass(frozen=True)
class ExperienceEvidence:
    """A piece of evidence supporting or challenging an observation."""
    evidence_type: str
    source: str
    description: str
    reference: str
    confidence: Decimal = Decimal("1")

    def __post_init__(self):
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class ExperienceReasoning:
    """Reasoning chain following Fabio's hierarchy for one experience.

    This captures how Genesis reasoned about the observation at the time.
    It references the DecisionHierarchyResult but is stored independently
    so the experience remains self-contained.
    """
    market_state_summary: str
    location_summary: str
    aggression_summary: str
    risk_summary: str
    management_summary: str
    hierarchy_reference: str = ""
    reasoning_confidence: Decimal = Decimal("0")

    def __post_init__(self):
        if not (Decimal("0") <= self.reasoning_confidence <= Decimal("1")):
            raise ValueError("reasoning_confidence must be in [0, 1]")


@dataclass(frozen=True)
class ExperienceOutcome:
    """What actually happened after the observation and reasoning."""
    outcome_type: ExperienceOutcomeType
    actual_result: str
    deviation_from_expected: str = ""
    outcome_timestamp: str = ""
    validating_evidence: Tuple[str, ...] = ()

    def __post_init__(self):
        if not self.actual_result:
            raise ValueError("actual_result is required")


@dataclass(frozen=True)
class ExperienceReflection:
    """Reflection on what the experience taught Genesis."""
    what_was_right: Tuple[str, ...] = ()
    what_was_wrong: Tuple[str, ...] = ()
    what_was_missed: Tuple[str, ...] = ()
    lesson_learned: str = ""
    reflection_confidence: Decimal = Decimal("0")
    would_reason_differently: bool = False
    different_reasoning: str = ""

    def __post_init__(self):
        if not (Decimal("0") <= self.reflection_confidence <= Decimal("1")):
            raise ValueError("reflection_confidence must be in [0, 1]")


@dataclass(frozen=True)
class Experience:
    """Immutable Experience object.

    Experience is the central learning object in Genesis. One experience
    represents one complete learning cycle: observe, reason, experience
    outcome, reflect. Experiences become Memory. Repeated Experiences
    become Knowledge.
    """
    observation: ExperienceObservation
    teacher_explanation: str
    evidence: Tuple[ExperienceEvidence, ...] = ()
    reasoning: Optional[ExperienceReasoning] = None
    outcome: Optional[ExperienceOutcome] = None
    reflection: Optional[ExperienceReflection] = None
    session_reference: str = ""
    experience_id: str = ""
    concept_references: Tuple[str, ...] = ()

    def __post_init__(self):
        if not self.experience_id:
            object.__setattr__(self, "experience_id", self._derive_id())
        if not self.teacher_explanation:
            raise ValueError("teacher_explanation is required")

    def _derive_id(self) -> str:
        from hashlib import sha256
        seed = f"{self.observation.detection_graph_reference}:{self.observation.timestamp}:{self.teacher_explanation[:64]}".encode("utf-8")
        return sha256(seed).hexdigest()[:16]

    def with_reasoning(self, reasoning: ExperienceReasoning) -> "Experience":
        """Return a new Experience with reasoning attached."""
        return Experience(
            observation=self.observation,
            teacher_explanation=self.teacher_explanation,
            evidence=self.evidence,
            reasoning=reasoning,
            outcome=self.outcome,
            reflection=self.reflection,
            session_reference=self.session_reference,
            experience_id=self.experience_id,
            concept_references=self.concept_references,
        )

    def with_outcome(self, outcome: ExperienceOutcome) -> "Experience":
        """Return a new Experience with outcome attached."""
        return Experience(
            observation=self.observation,
            teacher_explanation=self.teacher_explanation,
            evidence=self.evidence,
            reasoning=self.reasoning,
            outcome=outcome,
            reflection=self.reflection,
            session_reference=self.session_reference,
            experience_id=self.experience_id,
            concept_references=self.concept_references,
        )

    def with_reflection(self, reflection: ExperienceReflection) -> "Experience":
        """Return a new Experience with reflection attached."""
        return Experience(
            observation=self.observation,
            teacher_explanation=self.teacher_explanation,
            evidence=self.evidence,
            reasoning=self.reasoning,
            outcome=self.outcome,
            reflection=reflection,
            session_reference=self.session_reference,
            experience_id=self.experience_id,
            concept_references=self.concept_references,
        )

    def with_evidence(self, evidence: ExperienceEvidence) -> "Experience":
        """Return a new Experience with additional evidence."""
        return Experience(
            observation=self.observation,
            teacher_explanation=self.teacher_explanation,
            evidence=self.evidence + (evidence,),
            reasoning=self.reasoning,
            outcome=self.outcome,
            reflection=self.reflection,
            session_reference=self.session_reference,
            experience_id=self.experience_id,
            concept_references=self.concept_references,
        )

    def with_concept_reference(self, concept_reference: str) -> "Experience":
        """Return a new Experience with an additional concept reference."""
        if concept_reference in self.concept_references:
            return self
        return Experience(
            observation=self.observation,
            teacher_explanation=self.teacher_explanation,
            evidence=self.evidence,
            reasoning=self.reasoning,
            outcome=self.outcome,
            reflection=self.reflection,
            session_reference=self.session_reference,
            experience_id=self.experience_id,
            concept_references=self.concept_references + (concept_reference,),
        )

    def is_complete(self) -> bool:
        """An experience is complete when observation, reasoning, outcome, and reflection are present."""
        return (
            self.reasoning is not None
            and self.outcome is not None
            and self.reflection is not None
        )

    def completion_level(self) -> int:
        """Return 0-4 indicating how complete the experience is."""
        level = 1  # observation always present
        if self.reasoning is not None:
            level += 1
        if self.outcome is not None:
            level += 1
        if self.reflection is not None:
            level += 1
        return level


@dataclass(frozen=True)
class ExperienceConfiguration:
    """Configuration for the Experience Engine."""
    min_evidence_for_high_confidence: int = 3
    min_reflection_length: int = 10
    require_hierarchy_reasoning: bool = True
    require_outcome_for_memory: bool = False
    require_reflection_for_memory: bool = True


@dataclass(frozen=True)
class ExperienceStatistics:
    """Statistics over a collection of experiences."""
    total_experiences: int = 0
    complete_experiences: int = 0
    incomplete_experiences: int = 0
    validated_count: int = 0
    partially_validated_count: int = 0
    invalidated_count: int = 0
    pending_count: int = 0
    average_completion_level: Decimal = Decimal("0")
    experiences_with_reflection: int = 0
    experiences_without_reflection: int = 0


@dataclass(frozen=True)
class ExperienceResult:
    """Result of an experience engine operation."""
    experiences: Tuple[Experience, ...] = ()
    created: Tuple[Experience, ...] = ()
    updated: Tuple[Experience, ...] = ()
    statistics: ExperienceStatistics = field(default_factory=ExperienceStatistics)


class ExperienceEngine:
    """Deterministic engine for managing Genesis experiences.

    The Experience Engine creates, evolves, and queries experiences. It
    performs no AI, no ML, no prediction, and no trade recommendation.
    It only organizes what Genesis has observed, reasoned about, and
    learned from outcomes.
    """

    def __init__(self, configuration: Optional[ExperienceConfiguration] = None):
        self._configuration = configuration or ExperienceConfiguration()
        self._experiences: dict[str, Experience] = {}

    def create(
        self,
        observation: ExperienceObservation,
        teacher_explanation: str,
        evidence: Tuple[ExperienceEvidence, ...] = (),
        session_reference: str = "",
    ) -> ExperienceResult:
        """Create a new experience from observation and teacher explanation."""
        experience = Experience(
            observation=observation,
            teacher_explanation=teacher_explanation,
            evidence=evidence,
            session_reference=session_reference,
        )
        self._experiences[experience.experience_id] = experience
        stats = self._compute_statistics()
        return ExperienceResult(
            experiences=tuple(self._experiences.values()),
            created=(experience,),
            updated=(),
            statistics=stats,
        )

    def add_reasoning(self, experience_id: str, reasoning: ExperienceReasoning) -> ExperienceResult:
        """Add reasoning to an existing experience."""
        experience = self._experiences.get(experience_id)
        if experience is None:
            raise ValueError(f"experience {experience_id} not found")
        updated = experience.with_reasoning(reasoning)
        self._experiences[experience_id] = updated
        stats = self._compute_statistics()
        return ExperienceResult(
            experiences=tuple(self._experiences.values()),
            created=(),
            updated=(updated,),
            statistics=stats,
        )

    def add_outcome(self, experience_id: str, outcome: ExperienceOutcome) -> ExperienceResult:
        """Add outcome to an existing experience."""
        experience = self._experiences.get(experience_id)
        if experience is None:
            raise ValueError(f"experience {experience_id} not found")
        updated = experience.with_outcome(outcome)
        self._experiences[experience_id] = updated
        stats = self._compute_statistics()
        return ExperienceResult(
            experiences=tuple(self._experiences.values()),
            created=(),
            updated=(updated,),
            statistics=stats,
        )

    def add_reflection(self, experience_id: str, reflection: ExperienceReflection) -> ExperienceResult:
        """Add reflection to an existing experience."""
        experience = self._experiences.get(experience_id)
        if experience is None:
            raise ValueError(f"experience {experience_id} not found")
        updated = experience.with_reflection(reflection)
        self._experiences[experience_id] = updated
        stats = self._compute_statistics()
        return ExperienceResult(
            experiences=tuple(self._experiences.values()),
            created=(),
            updated=(updated,),
            statistics=stats,
        )

    def add_evidence(self, experience_id: str, evidence: ExperienceEvidence) -> ExperienceResult:
        """Add evidence to an existing experience."""
        experience = self._experiences.get(experience_id)
        if experience is None:
            raise ValueError(f"experience {experience_id} not found")
        updated = experience.with_evidence(evidence)
        self._experiences[experience_id] = updated
        stats = self._compute_statistics()
        return ExperienceResult(
            experiences=tuple(self._experiences.values()),
            created=(),
            updated=(updated,),
            statistics=stats,
        )

    def link_concept(self, experience_id: str, concept_reference: str) -> ExperienceResult:
        """Link an experience to a concept."""
        experience = self._experiences.get(experience_id)
        if experience is None:
            raise ValueError(f"experience {experience_id} not found")
        updated = experience.with_concept_reference(concept_reference)
        self._experiences[experience_id] = updated
        stats = self._compute_statistics()
        return ExperienceResult(
            experiences=tuple(self._experiences.values()),
            created=(),
            updated=(updated,),
            statistics=stats,
        )

    def get(self, experience_id: str) -> Optional[Experience]:
        """Retrieve an experience by id."""
        return self._experiences.get(experience_id)

    def all_experiences(self) -> Tuple[Experience, ...]:
        """Return all experiences."""
        return tuple(self._experiences.values())

    def complete_experiences(self) -> Tuple[Experience, ...]:
        """Return only complete experiences."""
        return tuple(e for e in self._experiences.values() if e.is_complete())

    def incomplete_experiences(self) -> Tuple[Experience, ...]:
        """Return only incomplete experiences."""
        return tuple(e for e in self._experiences.values() if not e.is_complete())

    def experiences_for_concept(self, concept_reference: str) -> Tuple[Experience, ...]:
        """Return all experiences linked to a concept."""
        return tuple(
            e for e in self._experiences.values()
            if concept_reference in e.concept_references
        )

    def experiences_for_session(self, session_reference: str) -> Tuple[Experience, ...]:
        """Return all experiences from a learning session."""
        return tuple(
            e for e in self._experiences.values()
            if e.session_reference == session_reference
        )

    def experiences_ready_for_memory(self) -> Tuple[Experience, ...]:
        """Return experiences that satisfy memory requirements."""
        results = []
        for e in self._experiences.values():
            ready = True
            if self._configuration.require_outcome_for_memory and e.outcome is None:
                ready = False
            if self._configuration.require_reflection_for_memory and e.reflection is None:
                ready = False
            if ready:
                results.append(e)
        return tuple(results)

    def _compute_statistics(self) -> ExperienceStatistics:
        """Compute statistics over all experiences."""
        experiences = list(self._experiences.values())
        if not experiences:
            return ExperienceStatistics()
        complete = sum(1 for e in experiences if e.is_complete())
        validated = sum(1 for e in experiences if e.outcome is not None and e.outcome.outcome_type == ExperienceOutcomeType.VALIDATED)
        partial = sum(1 for e in experiences if e.outcome is not None and e.outcome.outcome_type == ExperienceOutcomeType.PARTIALLY_VALIDATED)
        invalidated = sum(1 for e in experiences if e.outcome is not None and e.outcome.outcome_type == ExperienceOutcomeType.INVALIDATED)
        pending = sum(1 for e in experiences if e.outcome is not None and e.outcome.outcome_type == ExperienceOutcomeType.PENDING)
        with_reflection = sum(1 for e in experiences if e.reflection is not None)
        avg_completion = sum(e.completion_level() for e in experiences) / Decimal(str(len(experiences)))
        return ExperienceStatistics(
            total_experiences=len(experiences),
            complete_experiences=complete,
            incomplete_experiences=len(experiences) - complete,
            validated_count=validated,
            partially_validated_count=partial,
            invalidated_count=invalidated,
            pending_count=pending,
            average_completion_level=min(avg_completion, Decimal("4")),
            experiences_with_reflection=with_reflection,
            experiences_without_reflection=len(experiences) - with_reflection,
        )
