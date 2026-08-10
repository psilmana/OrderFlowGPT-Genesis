"""Concept Engine for the Genesis Apprentice Layer.

Genesis learns Concepts. Each Concept contains name, definition, visual
appearance, teacher explanation, positive and negative examples, related
concepts, confidence, and lesson references. Concepts evolve over time
through repeated experience.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Tuple


class ConceptConfidenceLevel(Enum):
    """Deterministic confidence levels for concept mastery."""
    NOVICE = auto()
    DEVELOPING = auto()
    COMPETENT = auto()
    PROFICIENT = auto()
    EXPERT = auto()


@dataclass(frozen=True)
class ConceptExample:
    """A single positive or negative example of a concept."""
    description: str
    chart_context: str
    evidence_reference: str
    is_positive: bool
    lesson_reference: str = ""
    frame_reference: str = ""

    def __post_init__(self):
        if not self.description:
            raise ValueError("description is required")
        if not self.evidence_reference:
            raise ValueError("evidence_reference is required")


@dataclass(frozen=True)
class ConceptDefinition:
    """Immutable definition of what a concept is."""
    name: str
    definition: str
    visual_appearance: str
    teacher_explanation: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("name is required")
        if not self.definition:
            raise ValueError("definition is required")


@dataclass(frozen=True)
class ConceptConfidence:
    """Immutable confidence tracking for a concept."""
    level: ConceptConfidenceLevel
    score: Decimal
    evidence_count: int
    positive_example_count: int
    negative_example_count: int
    last_updated: str = ""

    def __post_init__(self):
        if not (Decimal("0") <= self.score <= Decimal("1")):
            raise ValueError("score must be in [0, 1]")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")
        if self.positive_example_count < 0:
            raise ValueError("positive_example_count must be non-negative")
        if self.negative_example_count < 0:
            raise ValueError("negative_example_count must be non-negative")


@dataclass(frozen=True)
class ConceptEvolution:
    """Immutable record of how a concept changed at one point in time."""
    timestamp: str
    change_description: str
    prior_confidence: ConceptConfidence
    new_confidence: ConceptConfidence
    trigger_experience_reference: str = ""
    trigger_lesson_reference: str = ""


@dataclass(frozen=True)
class Concept:
    """Immutable Concept object.

    A Concept is the central learning object in Genesis. It captures
    everything Genesis knows about one idea from Fabio's methodology.
    """
    definition: ConceptDefinition
    positive_examples: Tuple[ConceptExample, ...] = ()
    negative_examples: Tuple[ConceptExample, ...] = ()
    related_concepts: Tuple[str, ...] = ()
    confidence: ConceptConfidence = field(
        default_factory=lambda: ConceptConfidence(
            level=ConceptConfidenceLevel.NOVICE,
            score=Decimal("0"),
            evidence_count=0,
            positive_example_count=0,
            negative_example_count=0,
        )
    )
    lesson_references: Tuple[str, ...] = ()
    evolution_history: Tuple[ConceptEvolution, ...] = ()
    concept_id: str = ""

    def __post_init__(self):
        if not self.concept_id:
            object.__setattr__(self, "concept_id", self._derive_id())
        pos = sum(1 for e in self.positive_examples if e.is_positive)
        neg = sum(1 for e in self.negative_examples if not e.is_positive)
        if pos != len(self.positive_examples):
            raise ValueError("positive_examples contains non-positive example")
        if neg != len(self.negative_examples):
            raise ValueError("negative_examples contains positive example")

    def _derive_id(self) -> str:
        from hashlib import sha256
        seed = f"{self.definition.name}:{self.definition.definition}".encode("utf-8")
        return sha256(seed).hexdigest()[:16]

    def with_example(self, example: ConceptExample) -> "Concept":
        """Return a new Concept with the example added."""
        if example.is_positive:
            new_pos = self.positive_examples + (example,)
            new_neg = self.negative_examples
        else:
            new_pos = self.positive_examples
            new_neg = self.negative_examples + (example,)
        return self._replace_examples(new_pos, new_neg)

    def with_confidence(self, confidence: ConceptConfidence) -> "Concept":
        """Return a new Concept with updated confidence and evolution record."""
        evolution = ConceptEvolution(
            timestamp=confidence.last_updated,
            change_description=f"confidence updated to {confidence.level.name}",
            prior_confidence=self.confidence,
            new_confidence=confidence,
        )
        return Concept(
            definition=self.definition,
            positive_examples=self.positive_examples,
            negative_examples=self.negative_examples,
            related_concepts=self.related_concepts,
            confidence=confidence,
            lesson_references=self.lesson_references,
            evolution_history=self.evolution_history + (evolution,),
            concept_id=self.concept_id,
        )

    def with_related_concept(self, concept_name: str) -> "Concept":
        """Return a new Concept with an additional related concept."""
        if concept_name in self.related_concepts:
            return self
        return Concept(
            definition=self.definition,
            positive_examples=self.positive_examples,
            negative_examples=self.negative_examples,
            related_concepts=self.related_concepts + (concept_name,),
            confidence=self.confidence,
            lesson_references=self.lesson_references,
            evolution_history=self.evolution_history,
            concept_id=self.concept_id,
        )

    def with_lesson(self, lesson_reference: str) -> "Concept":
        """Return a new Concept with an additional lesson reference."""
        if lesson_reference in self.lesson_references:
            return self
        return Concept(
            definition=self.definition,
            positive_examples=self.positive_examples,
            negative_examples=self.negative_examples,
            related_concepts=self.related_concepts,
            confidence=self.confidence,
            lesson_references=self.lesson_references + (lesson_reference,),
            evolution_history=self.evolution_history,
            concept_id=self.concept_id,
        )

    def _replace_examples(
        self, positive: Tuple[ConceptExample, ...], negative: Tuple[ConceptExample, ...]
    ) -> "Concept":
        return Concept(
            definition=self.definition,
            positive_examples=positive,
            negative_examples=negative,
            related_concepts=self.related_concepts,
            confidence=self.confidence,
            lesson_references=self.lesson_references,
            evolution_history=self.evolution_history,
            concept_id=self.concept_id,
        )


@dataclass(frozen=True)
class ConceptConfiguration:
    """Configuration for the Concept Engine."""
    novice_threshold: Decimal = Decimal("0.00")
    developing_threshold: Decimal = Decimal("0.25")
    competent_threshold: Decimal = Decimal("0.50")
    proficient_threshold: Decimal = Decimal("0.75")
    expert_threshold: Decimal = Decimal("0.90")
    min_examples_for_competent: int = 3
    min_examples_for_proficient: int = 7
    min_examples_for_expert: int = 12

    def __post_init__(self):
        thresholds = [
            self.novice_threshold,
            self.developing_threshold,
            self.competent_threshold,
            self.proficient_threshold,
            self.expert_threshold,
        ]
        for i, t in enumerate(thresholds):
            if not (Decimal("0") <= t <= Decimal("1")):
                raise ValueError(f"threshold {i} must be in [0, 1]")
        for i in range(1, len(thresholds)):
            if thresholds[i] < thresholds[i - 1]:
                raise ValueError("thresholds must be non-decreasing")

    def level_for_score(self, score: Decimal, example_count: int) -> ConceptConfidenceLevel:
        """Deterministic level from score and example count."""
        if score >= self.expert_threshold and example_count >= self.min_examples_for_expert:
            return ConceptConfidenceLevel.EXPERT
        if score >= self.proficient_threshold and example_count >= self.min_examples_for_proficient:
            return ConceptConfidenceLevel.PROFICIENT
        if score >= self.competent_threshold and example_count >= self.min_examples_for_competent:
            return ConceptConfidenceLevel.COMPETENT
        if score >= self.developing_threshold:
            return ConceptConfidenceLevel.DEVELOPING
        return ConceptConfidenceLevel.NOVICE


@dataclass(frozen=True)
class ConceptStatistics:
    """Statistics over a collection of concepts."""
    total_concepts: int = 0
    novice_count: int = 0
    developing_count: int = 0
    competent_count: int = 0
    proficient_count: int = 0
    expert_count: int = 0
    total_positive_examples: int = 0
    total_negative_examples: int = 0
    total_evolution_events: int = 0
    average_confidence_score: Decimal = Decimal("0")


@dataclass(frozen=True)
class ConceptResult:
    """Result of a concept engine operation."""
    concepts: Tuple[Concept, ...] = ()
    created: Tuple[Concept, ...] = ()
    updated: Tuple[Concept, ...] = ()
    statistics: ConceptStatistics = field(default_factory=ConceptStatistics)


class ConceptEngine:
    """Deterministic engine for managing Genesis concepts.

    The Concept Engine creates, evolves, and queries concepts. It performs
    no AI, no ML, no prediction, and no trade recommendation. It only
    organizes and tracks what Genesis has learned from Fabio's teaching.
    """

    def __init__(self, configuration: Optional[ConceptConfiguration] = None):
        self._configuration = configuration or ConceptConfiguration()
        self._concepts: dict[str, Concept] = {}

    def register(self, concept: Concept) -> ConceptResult:
        """Register a concept. If it exists, merge carefully."""
        existing = self._concepts.get(concept.concept_id)
        if existing is None:
            self._concepts[concept.concept_id] = concept
            stats = self._compute_statistics()
            return ConceptResult(
                concepts=tuple(self._concepts.values()),
                created=(concept,),
                updated=(),
                statistics=stats,
            )
        merged = self._merge_concepts(existing, concept)
        self._concepts[concept.concept_id] = merged
        stats = self._compute_statistics()
        return ConceptResult(
            concepts=tuple(self._concepts.values()),
            created=(),
            updated=(merged,),
            statistics=stats,
        )

    def add_example(self, concept_id: str, example: ConceptExample) -> ConceptResult:
        """Add an example to an existing concept, updating confidence."""
        concept = self._concepts.get(concept_id)
        if concept is None:
            raise ValueError(f"concept {concept_id} not found")
        updated = concept.with_example(example)
        new_confidence = self._recalculate_confidence(updated)
        updated = updated.with_confidence(new_confidence)
        self._concepts[concept_id] = updated
        stats = self._compute_statistics()
        return ConceptResult(
            concepts=tuple(self._concepts.values()),
            created=(),
            updated=(updated,),
            statistics=stats,
        )

    def add_related_concept(self, concept_id: str, related_name: str) -> ConceptResult:
        """Add a related concept reference."""
        concept = self._concepts.get(concept_id)
        if concept is None:
            raise ValueError(f"concept {concept_id} not found")
        updated = concept.with_related_concept(related_name)
        self._concepts[concept_id] = updated
        stats = self._compute_statistics()
        return ConceptResult(
            concepts=tuple(self._concepts.values()),
            created=(),
            updated=(updated,),
            statistics=stats,
        )

    def add_lesson(self, concept_id: str, lesson_reference: str) -> ConceptResult:
        """Add a lesson reference to a concept."""
        concept = self._concepts.get(concept_id)
        if concept is None:
            raise ValueError(f"concept {concept_id} not found")
        updated = concept.with_lesson(lesson_reference)
        self._concepts[concept_id] = updated
        stats = self._compute_statistics()
        return ConceptResult(
            concepts=tuple(self._concepts.values()),
            created=(),
            updated=(updated,),
            statistics=stats,
        )

    def get(self, concept_id: str) -> Optional[Concept]:
        """Retrieve a concept by id."""
        return self._concepts.get(concept_id)

    def get_by_name(self, name: str) -> Optional[Concept]:
        """Retrieve a concept by exact name match."""
        for concept in self._concepts.values():
            if concept.definition.name == name:
                return concept
        return None

    def all_concepts(self) -> Tuple[Concept, ...]:
        """Return all registered concepts."""
        return tuple(self._concepts.values())

    def concepts_by_level(self, level: ConceptConfidenceLevel) -> Tuple[Concept, ...]:
        """Return concepts at a specific confidence level."""
        return tuple(c for c in self._concepts.values() if c.confidence.level == level)

    def related_concepts(self, concept_id: str) -> Tuple[Concept, ...]:
        """Return all concepts related to the given concept."""
        concept = self._concepts.get(concept_id)
        if concept is None:
            return ()
        related = []
        for name in concept.related_concepts:
            rc = self.get_by_name(name)
            if rc is not None:
                related.append(rc)
        return tuple(related)

    def _recalculate_confidence(self, concept: Concept) -> ConceptConfidence:
        """Deterministic confidence recalculation from examples."""
        total = len(concept.positive_examples) + len(concept.negative_examples)
        if total == 0:
            return ConceptConfidence(
                level=ConceptConfidenceLevel.NOVICE,
                score=Decimal("0"),
                evidence_count=0,
                positive_example_count=0,
                negative_example_count=0,
            )
        pos = len(concept.positive_examples)
        neg = len(concept.negative_examples)
        score = Decimal(str(pos)) / Decimal(str(total))
        level = self._configuration.level_for_score(score, total)
        return ConceptConfidence(
            level=level,
            score=score,
            evidence_count=total,
            positive_example_count=pos,
            negative_example_count=neg,
        )

    def _merge_concepts(self, existing: Concept, incoming: Concept) -> Concept:
        """Deterministic merge of two concept versions."""
        pos = existing.positive_examples
        for ex in incoming.positive_examples:
            if ex not in pos:
                pos = pos + (ex,)
        neg = existing.negative_examples
        for ex in incoming.negative_examples:
            if ex not in neg:
                neg = neg + (ex,)
        related = existing.related_concepts
        for r in incoming.related_concepts:
            if r not in related:
                related = related + (r,)
        lessons = existing.lesson_references
        for l in incoming.lesson_references:
            if l not in lessons:
                lessons = lessons + (l,)
        merged = Concept(
            definition=incoming.definition if incoming.definition.definition else existing.definition,
            positive_examples=pos,
            negative_examples=neg,
            related_concepts=related,
            confidence=existing.confidence,
            lesson_references=lessons,
            evolution_history=existing.evolution_history,
            concept_id=existing.concept_id,
        )
        return merged.with_confidence(self._recalculate_confidence(merged))

    def _compute_statistics(self) -> ConceptStatistics:
        """Compute statistics over all registered concepts."""
        concepts = list(self._concepts.values())
        if not concepts:
            return ConceptStatistics()
        level_counts = {level: 0 for level in ConceptConfidenceLevel}
        for c in concepts:
            level_counts[c.confidence.level] += 1
        total_pos = sum(len(c.positive_examples) for c in concepts)
        total_neg = sum(len(c.negative_examples) for c in concepts)
        total_evo = sum(len(c.evolution_history) for c in concepts)
        avg_score = sum(c.confidence.score for c in concepts) / Decimal(str(len(concepts)))
        return ConceptStatistics(
            total_concepts=len(concepts),
            novice_count=level_counts[ConceptConfidenceLevel.NOVICE],
            developing_count=level_counts[ConceptConfidenceLevel.DEVELOPING],
            competent_count=level_counts[ConceptConfidenceLevel.COMPETENT],
            proficient_count=level_counts[ConceptConfidenceLevel.PROFICIENT],
            expert_count=level_counts[ConceptConfidenceLevel.EXPERT],
            total_positive_examples=total_pos,
            total_negative_examples=total_neg,
            total_evolution_events=total_evo,
            average_confidence_score=min(avg_score, Decimal("1")),
        )
