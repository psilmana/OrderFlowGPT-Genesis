"""Learning Loop for the Genesis Apprentice Layer.

Observe -> Question -> Teacher Explains -> Reflection -> Experience ->
Memory -> Knowledge -> Observe Again

This loop represents continuous learning. One complete cycle produces
one LearningSession with linked experiences, concepts, and knowledge.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Tuple

from .concepts import Concept, ConceptEngine, ConceptExample
from .experiences import (
    Experience,
    ExperienceConfiguration,
    ExperienceEngine,
    ExperienceEvidence,
    ExperienceObservation,
    ExperienceOutcome,
    ExperienceOutcomeType,
    ExperienceReasoning,
    ExperienceReflection,
)
from .knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder, KnowledgeRelationship, RelationshipType
from .reasoning import DecisionHierarchyAnalyzer, DecisionHierarchyResult


class LearningPhase(Enum):
    """Deterministic phases of the learning loop."""
    OBSERVE = auto()
    QUESTION = auto()
    EXPLAIN = auto()
    REFLECT = auto()
    EXPERIENCE = auto()
    MEMORY = auto()
    KNOWLEDGE = auto()


class QuestionType(Enum):
    """Deterministic question types Genesis can ask."""
    WHAT = auto()
    WHY = auto()
    HOW = auto()
    WHEN = auto()
    WHERE = auto()
    CONFIRMATION = auto()
    CONTRADICTION = auto()


@dataclass(frozen=True)
class LearningQuestion:
    """A question asked by Genesis during learning."""
    question_text: str
    question_type: QuestionType
    context: str
    timestamp: str
    related_concepts: Tuple[str, ...] = ()
    answer_reference: str = ""

    def __post_init__(self):
        if not self.question_text:
            raise ValueError("question_text is required")


@dataclass(frozen=True)
class TeacherExplanation:
    """Fabio's explanation captured from transcript or direct input."""
    explanation_text: str
    source_reference: str
    confidence: Decimal
    related_concepts: Tuple[str, ...] = ()
    lesson_reference: str = ""

    def __post_init__(self):
        if not self.explanation_text:
            raise ValueError("explanation_text is required")
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class LearningObservation:
    """A structured observation during the learning loop."""
    observation_text: str
    evidence: Tuple[str, ...]
    timestamp: str
    detection_graph_reference: str = ""
    hierarchy_result: Optional[DecisionHierarchyResult] = None

    def __post_init__(self):
        if not self.observation_text:
            raise ValueError("observation_text is required")


@dataclass(frozen=True)
class LearningReflection:
    """Reflection produced during the learning loop."""
    reflection_text: str
    insights: Tuple[str, ...]
    confidence_change: Decimal
    concepts_reinforced: Tuple[str, ...] = ()
    concepts_challenged: Tuple[str, ...] = ()

    def __post_init__(self):
        if not self.reflection_text:
            raise ValueError("reflection_text is required")
        if not (Decimal("-1") <= self.confidence_change <= Decimal("1")):
            raise ValueError("confidence_change must be in [-1, 1]")


@dataclass(frozen=True)
class LearningSession:
    """Immutable record of one complete learning session.

    A session may not complete all phases. The phases_completed field
    tracks which phases were executed.
    """
    session_id: str
    lesson_reference: str
    phases_completed: Tuple[LearningPhase, ...] = ()
    questions: Tuple[LearningQuestion, ...] = ()
    explanations: Tuple[TeacherExplanation, ...] = ()
    observations: Tuple[LearningObservation, ...] = ()
    reflections: Tuple[LearningReflection, ...] = ()
    experience_references: Tuple[str, ...] = ()
    concept_references: Tuple[str, ...] = ()
    knowledge_graph_reference: str = ""
    session_confidence: Decimal = Decimal("0")
    is_complete: bool = False

    def __post_init__(self):
        if not self.session_id:
            raise ValueError("session_id is required")
        if not (Decimal("0") <= self.session_confidence <= Decimal("1")):
            raise ValueError("session_confidence must be in [0, 1]")

    def has_phase(self, phase: LearningPhase) -> bool:
        """Check if a phase was completed in this session."""
        return phase in self.phases_completed

    def with_phase(self, phase: LearningPhase) -> "LearningSession":
        """Return a new session with the phase marked complete."""
        if phase in self.phases_completed:
            return self
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed + (phase,),
            questions=self.questions,
            explanations=self.explanations,
            observations=self.observations,
            reflections=self.reflections,
            experience_references=self.experience_references,
            concept_references=self.concept_references,
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=self.session_confidence,
            is_complete=self.is_complete,
        )

    def with_question(self, question: LearningQuestion) -> "LearningSession":
        """Return a new session with a question added."""
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions + (question,),
            explanations=self.explanations,
            observations=self.observations,
            reflections=self.reflections,
            experience_references=self.experience_references,
            concept_references=self.concept_references,
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=self.session_confidence,
            is_complete=self.is_complete,
        )

    def with_explanation(self, explanation: TeacherExplanation) -> "LearningSession":
        """Return a new session with an explanation added."""
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions,
            explanations=self.explanations + (explanation,),
            observations=self.observations,
            reflections=self.reflections,
            experience_references=self.experience_references,
            concept_references=self.concept_references,
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=self.session_confidence,
            is_complete=self.is_complete,
        )

    def with_observation(self, observation: LearningObservation) -> "LearningSession":
        """Return a new session with an observation added."""
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions,
            explanations=self.explanations,
            observations=self.observations + (observation,),
            reflections=self.reflections,
            experience_references=self.experience_references,
            concept_references=self.concept_references,
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=self.session_confidence,
            is_complete=self.is_complete,
        )

    def with_reflection(self, reflection: LearningReflection) -> "LearningSession":
        """Return a new session with a reflection added."""
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions,
            explanations=self.explanations,
            observations=self.observations,
            reflections=self.reflections + (reflection,),
            experience_references=self.experience_references,
            concept_references=self.concept_references,
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=self.session_confidence,
            is_complete=self.is_complete,
        )

    def with_experience_reference(self, experience_reference: str) -> "LearningSession":
        """Return a new session with an experience reference added."""
        if experience_reference in self.experience_references:
            return self
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions,
            explanations=self.explanations,
            observations=self.observations,
            reflections=self.reflections,
            experience_references=self.experience_references + (experience_reference,),
            concept_references=self.concept_references,
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=self.session_confidence,
            is_complete=self.is_complete,
        )

    def with_concept_reference(self, concept_reference: str) -> "LearningSession":
        """Return a new session with a concept reference added."""
        if concept_reference in self.concept_references:
            return self
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions,
            explanations=self.explanations,
            observations=self.observations,
            reflections=self.reflections,
            experience_references=self.experience_references,
            concept_references=self.concept_references + (concept_reference,),
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=self.session_confidence,
            is_complete=self.is_complete,
        )

    def with_knowledge_graph(self, graph_reference: str) -> "LearningSession":
        """Return a new session with a knowledge graph reference."""
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions,
            explanations=self.explanations,
            observations=self.observations,
            reflections=self.reflections,
            experience_references=self.experience_references,
            concept_references=self.concept_references,
            knowledge_graph_reference=graph_reference,
            session_confidence=self.session_confidence,
            is_complete=self.is_complete,
        )

    def with_confidence(self, confidence: Decimal) -> "LearningSession":
        """Return a new session with updated confidence."""
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions,
            explanations=self.explanations,
            observations=self.observations,
            reflections=self.reflections,
            experience_references=self.experience_references,
            concept_references=self.concept_references,
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=min(confidence, Decimal("1")),
            is_complete=self.is_complete,
        )

    def mark_complete(self) -> "LearningSession":
        """Mark the session as complete."""
        return LearningSession(
            session_id=self.session_id,
            lesson_reference=self.lesson_reference,
            phases_completed=self.phases_completed,
            questions=self.questions,
            explanations=self.explanations,
            observations=self.observations,
            reflections=self.reflections,
            experience_references=self.experience_references,
            concept_references=self.concept_references,
            knowledge_graph_reference=self.knowledge_graph_reference,
            session_confidence=self.session_confidence,
            is_complete=True,
        )


@dataclass(frozen=True)
class LearningLoopConfiguration:
    """Configuration for the Learning Loop."""
    require_question_before_explanation: bool = True
    require_reflection_before_memory: bool = True
    auto_generate_questions: bool = False
    max_questions_per_session: int = 5
    max_explanations_per_session: int = 5
    concept_confidence_boost_per_session: Decimal = Decimal("0.05")

    def __post_init__(self):
        if self.max_questions_per_session < 0:
            raise ValueError("max_questions_per_session must be non-negative")
        if self.max_explanations_per_session < 0:
            raise ValueError("max_explanations_per_session must be non-negative")
        if not (Decimal("0") <= self.concept_confidence_boost_per_session <= Decimal("1")):
            raise ValueError("concept_confidence_boost_per_session must be in [0, 1]")


@dataclass(frozen=True)
class LearningLoopStatistics:
    """Statistics over learning sessions."""
    total_sessions: int = 0
    complete_sessions: int = 0
    incomplete_sessions: int = 0
    total_questions: int = 0
    total_explanations: int = 0
    total_observations: int = 0
    total_reflections: int = 0
    total_experiences_created: int = 0
    total_concepts_touched: int = 0
    average_session_confidence: Decimal = Decimal("0")


@dataclass(frozen=True)
class LearningLoopResult:
    """Result of running one learning loop iteration."""
    session: LearningSession
    experience_created: Optional[Experience] = None
    concepts_updated: Tuple[Concept, ...] = ()
    knowledge_graph: Optional[KnowledgeGraph] = None
    statistics: LearningLoopStatistics = field(default_factory=LearningLoopStatistics)


class LearningLoop:
    """Deterministic learning loop orchestrator.

    The LearningLoop coordinates the educational cycle:
    Observe -> Question -> Teacher Explains -> Reflection ->
    Experience -> Memory -> Knowledge -> Observe Again

    It consumes existing Bundle 13 deterministic outputs (DetectionGraph,
    MemoryDataset, KnowledgeDataset) and produces higher-level apprentice
    objects. No AI, no ML, no prediction, no trade recommendation.
    """

    def __init__(
        self,
        concept_engine: ConceptEngine,
        experience_engine: ExperienceEngine,
        hierarchy_analyzer: DecisionHierarchyAnalyzer,
        configuration: Optional[LearningLoopConfiguration] = None,
    ):
        self._concept_engine = concept_engine
        self._experience_engine = experience_engine
        self._hierarchy_analyzer = hierarchy_analyzer
        self._configuration = configuration or LearningLoopConfiguration()
        self._sessions: dict[str, LearningSession] = {}
        self._graph_builder = KnowledgeGraphBuilder()

    def start_session(self, session_id: str, lesson_reference: str) -> LearningSession:
        """Start a new learning session."""
        session = LearningSession(
            session_id=session_id,
            lesson_reference=lesson_reference,
        )
        self._sessions[session_id] = session
        return session

    def observe(
        self,
        session_id: str,
        detection_graph,
        timestamp: str,
    ) -> LearningLoopResult:
        """Perform the OBSERVE phase.

        Genesis observes the chart, runs the Decision Hierarchy, and
        records what it sees.
        """
        session = self._get_session(session_id)
        hierarchy = self._hierarchy_analyzer.analyze(detection_graph)
        observation_text = self._describe_hierarchy(hierarchy)
        observation = LearningObservation(
            observation_text=observation_text,
            evidence=tuple(hierarchy.missing_levels) if hierarchy.missing_levels else ("hierarchy_complete",),
            timestamp=timestamp,
            detection_graph_reference=getattr(detection_graph, "graph_id", str(id(detection_graph))),
            hierarchy_result=hierarchy,
        )
        session = session.with_observation(observation).with_phase(LearningPhase.OBSERVE)
        self._sessions[session_id] = session
        return LearningLoopResult(session=session)

    def question(
        self,
        session_id: str,
        question_text: str,
        question_type: QuestionType,
        context: str,
        timestamp: str,
    ) -> LearningLoopResult:
        """Perform the QUESTION phase.

        Genesis asks a question about what it observed.
        """
        session = self._get_session(session_id)
        if len(session.questions) >= self._configuration.max_questions_per_session:
            raise ValueError("max questions reached for this session")
        question = LearningQuestion(
            question_text=question_text,
            question_type=question_type,
            context=context,
            timestamp=timestamp,
        )
        session = session.with_question(question).with_phase(LearningPhase.QUESTION)
        self._sessions[session_id] = session
        return LearningLoopResult(session=session)

    def explain(
        self,
        session_id: str,
        explanation_text: str,
        source_reference: str,
        confidence: Decimal,
        timestamp: str,
    ) -> LearningLoopResult:
        """Perform the EXPLAIN phase.

        Fabio (or the teacher source) explains the observation.
        """
        session = self._get_session(session_id)
        if not session.observations:
            raise ValueError("observation required before explanation")
        if self._configuration.require_question_before_explanation and not session.questions:
            raise ValueError("question required before explanation")
        if len(session.explanations) >= self._configuration.max_explanations_per_session:
            raise ValueError("max explanations reached for this session")
        explanation = TeacherExplanation(
            explanation_text=explanation_text,
            source_reference=source_reference,
            confidence=confidence,
            lesson_reference=session.lesson_reference,
        )
        session = session.with_explanation(explanation).with_phase(LearningPhase.EXPLAIN)
        self._sessions[session_id] = session
        return LearningLoopResult(session=session)

    def reflect(
        self,
        session_id: str,
        reflection_text: str,
        insights: Tuple[str, ...],
        confidence_change: Decimal,
        timestamp: str,
    ) -> LearningLoopResult:
        """Perform the REFLECT phase.

        Genesis reflects on the explanation versus its observation.
        """
        session = self._get_session(session_id)
        reflection = LearningReflection(
            reflection_text=reflection_text,
            insights=insights,
            confidence_change=confidence_change,
        )
        session = session.with_reflection(reflection).with_phase(LearningPhase.REFLECT)
        self._sessions[session_id] = session
        return LearningLoopResult(session=session)

    def experience(
        self,
        session_id: str,
        outcome_type: ExperienceOutcomeType,
        actual_result: str,
        timestamp: str,
    ) -> LearningLoopResult:
        """Perform the EXPERIENCE phase.

        Create an Experience from the session's observation, explanation,
        and reflection. Attach outcome if available.
        """
        session = self._get_session(session_id)
        if not session.observations:
            raise ValueError("observation required before experience")
        if not session.explanations:
            raise ValueError("explanation required before experience")

        latest_obs = session.observations[-1]
        latest_exp = session.explanations[-1]

        observation = ExperienceObservation(
            what_was_seen=latest_obs.observation_text,
            detection_graph_reference=latest_obs.detection_graph_reference,
            frame_reference=latest_obs.detection_graph_reference,
            timestamp=timestamp,
            market_context_summary=latest_obs.observation_text[:200],
        )

        evidence = (
            ExperienceEvidence(
                evidence_type="hierarchy",
                source="DecisionHierarchyAnalyzer",
                description=latest_obs.observation_text,
                reference=latest_obs.detection_graph_reference,
            ),
        )

        result = self._experience_engine.create(
            observation=observation,
            teacher_explanation=latest_exp.explanation_text,
            evidence=evidence,
            session_reference=session_id,
        )
        experience = result.created[0] if result.created else None

        if experience is not None:
            if latest_obs.hierarchy_result is not None:
                hierarchy = latest_obs.hierarchy_result
                reasoning = ExperienceReasoning(
                    market_state_summary=hierarchy.market_state.state.name if hierarchy.market_state else "unknown",
                    location_summary=hierarchy.location.location.name if hierarchy.location else "unknown",
                    aggression_summary=hierarchy.aggression.aggression.name if hierarchy.aggression else "unknown",
                    risk_summary=hierarchy.risk.risk_level.name if hierarchy.risk else "unknown",
                    management_summary=hierarchy.management.management.name if hierarchy.management else "unknown",
                    hierarchy_reference=str(id(hierarchy)),
                    reasoning_confidence=hierarchy.overall_confidence,
                )
                self._experience_engine.add_reasoning(experience.experience_id, reasoning)

            outcome = ExperienceOutcome(
                outcome_type=outcome_type,
                actual_result=actual_result,
                outcome_timestamp=timestamp,
            )
            self._experience_engine.add_outcome(experience.experience_id, outcome)

            if session.reflections:
                latest_ref = session.reflections[-1]
                reflection = ExperienceReflection(
                    what_was_right=latest_ref.insights,
                    what_was_wrong=(),
                    what_was_missed=(),
                    lesson_learned=latest_ref.reflection_text,
                    reflection_confidence=min(
                        Decimal("1"),
                        max(Decimal("0"), Decimal("0.5") + latest_ref.confidence_change),
                    ),
                )
                self._experience_engine.add_reflection(experience.experience_id, reflection)

            session = session.with_experience_reference(experience.experience_id)

        session = session.with_phase(LearningPhase.EXPERIENCE)
        self._sessions[session_id] = session
        return LearningLoopResult(session=session, experience_created=experience)

    def to_memory(self, session_id: str) -> LearningLoopResult:
        """Perform the MEMORY phase.

        Link experiences to concepts and build memory associations.
        """
        session = self._get_session(session_id)
        ready = self._experience_engine.experiences_ready_for_memory()
        for exp in ready:
            if exp.session_reference == session_id:
                for concept in self._concept_engine.all_concepts():
                    if self._experience_mentions_concept(exp, concept):
                        self._experience_engine.link_concept(exp.experience_id, concept.concept_id)
                        session = session.with_concept_reference(concept.concept_id)
        session = session.with_phase(LearningPhase.MEMORY)
        self._sessions[session_id] = session
        return LearningLoopResult(session=session)

    def to_knowledge(self, session_id: str) -> LearningLoopResult:
        """Perform the KNOWLEDGE phase.

        Build knowledge graph relationships from the session's concepts
        and experiences.
        """
        session = self._get_session(session_id)
        for concept_id in session.concept_references:
            concept = self._concept_engine.get(concept_id)
            if concept is None:
                continue
            self._graph_builder.add_concept(concept.definition.name)
            for related in concept.related_concepts:
                self._graph_builder.add_concept(related)
                self._graph_builder.add_relationship(
                    KnowledgeRelationship(
                        source_concept=concept.definition.name,
                        relationship_type=RelationshipType.RELATED_TO,
                        target_concept=related,
                        evidence=(f"session={session_id}",),
                    )
                )
        graph = self._graph_builder.build()
        session = session.with_knowledge_graph(str(id(graph))).with_phase(LearningPhase.KNOWLEDGE)
        self._sessions[session_id] = session
        return LearningLoopResult(session=session, knowledge_graph=graph)

    def complete_session(self, session_id: str) -> LearningLoopResult:
        """Mark a session as complete and compute final confidence."""
        session = self._get_session(session_id)
        conf = self._compute_session_confidence(session)
        session = session.with_confidence(conf).mark_complete()
        self._sessions[session_id] = session
        stats = self._compute_statistics()
        return LearningLoopResult(
            session=session,
            knowledge_graph=self._graph_builder.build(),
            statistics=stats,
        )

    def get_session(self, session_id: str) -> Optional[LearningSession]:
        """Retrieve a session by id."""
        return self._sessions.get(session_id)

    def all_sessions(self) -> Tuple[LearningSession, ...]:
        """Return all sessions."""
        return tuple(self._sessions.values())

    def _get_session(self, session_id: str) -> LearningSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"session {session_id} not found")
        return session

    def _describe_hierarchy(self, hierarchy: DecisionHierarchyResult) -> str:
        """Produce a deterministic text description of the hierarchy."""
        parts = []
        if hierarchy.market_state:
            parts.append(f"market_state={hierarchy.market_state.state.name}")
        if hierarchy.location:
            parts.append(f"location={hierarchy.location.location.name}")
        if hierarchy.aggression:
            parts.append(f"aggression={hierarchy.aggression.aggression.name}")
        if hierarchy.risk:
            parts.append(f"risk={hierarchy.risk.risk_level.name}")
        if hierarchy.management:
            parts.append(f"management={hierarchy.management.management.name}")
        return "; ".join(parts) if parts else "no_hierarchy_data"

    def _experience_mentions_concept(self, experience: Experience, concept: Concept) -> bool:
        """Deterministic check if an experience mentions a concept."""
        name = concept.definition.name.lower()
        text = (
            experience.teacher_explanation.lower()
            + " "
            + experience.observation.what_was_seen.lower()
        )
        return name in text

    def _compute_session_confidence(self, session: LearningSession) -> Decimal:
        """Compute session confidence from its components."""
        if not session.phases_completed:
            return Decimal("0")
        base = Decimal(str(len(session.phases_completed))) / Decimal("7")
        if session.explanations:
            exp_conf = sum(e.confidence for e in session.explanations) / Decimal(str(len(session.explanations)))
            base = (base + exp_conf) / Decimal("2")
        if session.reflections:
            ref_conf = sum(
                min(Decimal("1"), max(Decimal("0"), Decimal("0.5") + r.confidence_change))
                for r in session.reflections
            ) / Decimal(str(len(session.reflections)))
            base = (base + ref_conf) / Decimal("2")
        return min(base, Decimal("1"))

    def _compute_statistics(self) -> LearningLoopStatistics:
        """Compute statistics over all sessions."""
        sessions = list(self._sessions.values())
        if not sessions:
            return LearningLoopStatistics()
        complete = sum(1 for s in sessions if s.is_complete)
        total_q = sum(len(s.questions) for s in sessions)
        total_e = sum(len(s.explanations) for s in sessions)
        total_o = sum(len(s.observations) for s in sessions)
        total_r = sum(len(s.reflections) for s in sessions)
        total_exp = sum(len(s.experience_references) for s in sessions)
        total_c = sum(len(s.concept_references) for s in sessions)
        avg_conf = sum(s.session_confidence for s in sessions) / Decimal(str(len(sessions)))
        return LearningLoopStatistics(
            total_sessions=len(sessions),
            complete_sessions=complete,
            incomplete_sessions=len(sessions) - complete,
            total_questions=total_q,
            total_explanations=total_e,
            total_observations=total_o,
            total_reflections=total_r,
            total_experiences_created=total_exp,
            total_concepts_touched=total_c,
            average_session_confidence=min(avg_conf, Decimal("1")),
        )
