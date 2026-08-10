"""Integration layer wiring the Apprentice Layer into existing Genesis infrastructure.

This module provides deterministic adapters that connect the Concept Engine,
Experience Engine, Learning Loop, and Knowledge Graph to Bundle 13 outputs:
DetectionGraph, MemoryDataset, KnowledgeDataset, TrainingSample, and the
GenesisRunner pipeline.

No AI, no ML, no prediction, no trade recommendation.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Tuple

from .concepts import Concept, ConceptConfidence, ConceptConfidenceLevel, ConceptDefinition, ConceptEngine, ConceptExample, ConceptResult
from .experiences import Experience, ExperienceConfiguration, ExperienceEngine, ExperienceEvidence, ExperienceObservation, ExperienceOutcome, ExperienceOutcomeType, ExperienceReasoning, ExperienceReflection, ExperienceResult
from .knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder, KnowledgeGraphConfiguration, KnowledgeRelationship, RelationshipType
from .learning_loop import LearningLoop, LearningLoopConfiguration, LearningLoopResult, LearningPhase, LearningQuestion, LearningReflection, LearningSession, TeacherExplanation, QuestionType
from .reasoning import DecisionHierarchyAnalyzer, DecisionHierarchyConfiguration, DecisionHierarchyResult, DecisionHierarchyStatistics


@dataclass(frozen=True)
class ApprenticeConfiguration:
    """Unified configuration for the entire Apprentice Layer."""
    hierarchy_configuration: DecisionHierarchyConfiguration = field(default_factory=DecisionHierarchyConfiguration)
    concept_configuration: Optional[object] = None
    experience_configuration: ExperienceConfiguration = field(default_factory=ExperienceConfiguration)
    learning_loop_configuration: LearningLoopConfiguration = field(default_factory=LearningLoopConfiguration)
    knowledge_graph_configuration: KnowledgeGraphConfiguration = field(default_factory=KnowledgeGraphConfiguration)
    enable_auto_concept_extraction: bool = True
    enable_auto_question_generation: bool = False
    enable_experience_to_memory: bool = True


@dataclass(frozen=True)
class ApprenticeStatistics:
    """Statistics across all apprentice layer components."""
    concept_statistics: object = field(default_factory=lambda: ConceptEngine()._compute_statistics() if hasattr(ConceptEngine, '_compute_statistics') else None)
    experience_statistics: object = None
    learning_statistics: object = None
    hierarchy_statistics: object = None
    knowledge_graph_statistics: object = None


@dataclass(frozen=True)
class ApprenticeResult:
    """Result of running the apprentice layer over one frame/lesson."""
    hierarchy_result: Optional[DecisionHierarchyResult] = None
    session: Optional[LearningSession] = None
    experience_created: Optional[Experience] = None
    concepts_touched: Tuple[Concept, ...] = ()
    knowledge_graph: Optional[KnowledgeGraph] = None
    statistics: ApprenticeStatistics = field(default_factory=ApprenticeStatistics)


class ApprenticeLayer:
    """Deterministic integration layer for the Genesis Apprentice system.

    The ApprenticeLayer wires together:
    - DecisionHierarchyAnalyzer (reasoning from DetectionGraph)
    - ConceptEngine (concept management)
    - ExperienceEngine (experience tracking)
    - LearningLoop (educational cycle orchestration)
    - KnowledgeGraph (relationship storage)

    It consumes existing Bundle 1-13 deterministic outputs and produces
    higher-level educational objects. No AI, no ML, no prediction, no
    trade recommendation, no side effects, no networking.
    """

    def __init__(self, configuration: Optional[ApprenticeConfiguration] = None):
        self._configuration = configuration or ApprenticeConfiguration()
        self._hierarchy_analyzer = DecisionHierarchyAnalyzer(self._configuration.hierarchy_configuration)
        self._concept_engine = ConceptEngine(self._configuration.concept_configuration)
        self._experience_engine = ExperienceEngine(self._configuration.experience_configuration)
        loop_config = self._configuration.learning_loop_configuration or LearningLoopConfiguration()
        # Runner integration is automatic — no human asks questions
        loop_config = loop_config if not hasattr(loop_config, "require_question_before_explanation") else             LearningLoopConfiguration(
                require_question_before_explanation=False,
                max_questions_per_session=loop_config.max_questions_per_session,
                max_explanations_per_session=loop_config.max_explanations_per_session,
                concept_confidence_boost_per_session=loop_config.concept_confidence_boost_per_session,
            )
        self._learning_loop = LearningLoop(
            concept_engine=self._concept_engine,
            experience_engine=self._experience_engine,
            hierarchy_analyzer=self._hierarchy_analyzer,
            configuration=loop_config,
        )

    def process_frame(
        self,
        session_id: str,
        lesson_reference: str,
        detection_graph,
        teacher_explanation: str,
        source_reference: str,
        timestamp: str,
    ) -> ApprenticeResult:
        """Process one frame through the full apprentice layer.

        This is the primary integration method. Given a DetectionGraph and
        Fabio's explanation, it runs the complete learning cycle for that
        observation.
        """
        if session_id not in [s.session_id for s in self._learning_loop.all_sessions()]:
            self._learning_loop.start_session(session_id, lesson_reference)

        self._learning_loop.observe(session_id, detection_graph, timestamp)

        if self._configuration.enable_auto_question_generation:
            self._learning_loop.question(
                session_id,
                question_text=f"What is happening in {lesson_reference}?",
                question_type=QuestionType.WHAT,
                context=lesson_reference,
                timestamp=timestamp,
            )

        self._learning_loop.explain(
            session_id,
            explanation_text=teacher_explanation,
            source_reference=source_reference,
            confidence=Decimal("0.90"),
            timestamp=timestamp,
        )

        self._learning_loop.reflect(
            session_id,
            reflection_text=f"Observed hierarchy for {lesson_reference}",
            insights=("explanation_received",),
            confidence_change=Decimal("0.05"),
            timestamp=timestamp,
        )

        self._learning_loop.experience(
            session_id,
            outcome_type=ExperienceOutcomeType.PENDING,
            actual_result="pending_outcome",
            timestamp=timestamp,
        )

        if self._configuration.enable_experience_to_memory:
            self._learning_loop.to_memory(session_id)

        self._learning_loop.to_knowledge(session_id)

        if self._configuration.enable_auto_concept_extraction:
            self._extract_concepts_from_session(session_id)

        result = self._learning_loop.complete_session(session_id)

        return ApprenticeResult(
            hierarchy_result=result.session.observations[-1].hierarchy_result if result.session.observations else None,
            session=result.session,
            experience_created=result.experience_created,
            concepts_touched=self._concepts_for_session(result.session),
            knowledge_graph=result.knowledge_graph,
        )

    def process_lesson_batch(
        self,
        lesson_reference: str,
        frames: Tuple[Tuple[str, object, str, str], ...],
    ) -> Tuple[ApprenticeResult, ...]:
        """Process multiple frames from one lesson.

        Args:
            lesson_reference: The lesson identifier.
            frames: Tuple of (session_id, detection_graph, teacher_explanation, timestamp).

        Returns:
            Tuple of ApprenticeResult, one per frame.
        """
        results = []
        for session_id, graph, explanation, timestamp in frames:
            result = self.process_frame(
                session_id=session_id,
                lesson_reference=lesson_reference,
                detection_graph=graph,
                teacher_explanation=explanation,
                source_reference=lesson_reference,
                timestamp=timestamp,
            )
            results.append(result)
        return tuple(results)

    def explain_chart(self, detection_graph) -> DecisionHierarchyResult:
        """Explain a chart through Fabio's Decision Hierarchy without running a full session.

        This is the "What is happening?" method. It produces explanation
        before any recommendation or prediction.
        """
        return self._hierarchy_analyzer.analyze(detection_graph)

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Retrieve a concept from the concept engine."""
        return self._concept_engine.get(concept_id)

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Retrieve a concept by name."""
        return self._concept_engine.get_by_name(name)

    def all_concepts(self) -> Tuple[Concept, ...]:
        """Return all registered concepts."""
        return self._concept_engine.all_concepts()

    def register_concept(self, concept: Concept) -> ConceptResult:
        """Register a concept manually."""
        return self._concept_engine.register(concept)

    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """Retrieve an experience from the experience engine."""
        return self._experience_engine.get(experience_id)

    def all_experiences(self) -> Tuple[Experience, ...]:
        """Return all experiences."""
        return self._experience_engine.all_experiences()

    def get_session(self, session_id: str) -> Optional[LearningSession]:
        """Retrieve a learning session."""
        return self._learning_loop.get_session(session_id)

    def all_sessions(self) -> Tuple[LearningSession, ...]:
        """Return all learning sessions."""
        return self._learning_loop.all_sessions()

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """Return the current knowledge graph."""
        return self._learning_loop._graph_builder.build()

    def query_knowledge(self, source: str, relationship_type: RelationshipType) -> Tuple[KnowledgeRelationship, ...]:
        """Query the knowledge graph for relationships."""
        graph = self.get_knowledge_graph()
        from .knowledge_graph import KnowledgeGraphQuery
        return graph.query(KnowledgeGraphQuery(source_concept=source, relationship_type=relationship_type))

    def _extract_concepts_from_session(self, session_id: str) -> None:
        """Deterministic concept extraction from a session's explanation text.

        Only extracts meaningful trading terminology. Filters aggressively
        against common English words and short tokens.
        """
        session = self._learning_loop.get_session(session_id)
        if session is None:
            return

        # Common English words to exclude (comprehensive blocklist)
        BLOCKLIST = {
            "about", "above", "across", "after", "again", "against", "all", "almost",
            "alone", "along", "already", "also", "although", "always", "among", "an",
            "and", "another", "any", "anyone", "anything", "are", "around", "as",
            "at", "away", "back", "be", "became", "because", "become", "becomes",
            "been", "before", "behind", "being", "below", "beside", "besides",
            "between", "beyond", "both", "but", "by", "came", "can", "cannot",
            "certainly", "come", "could", "did", "do", "does", "done", "down",
            "during", "each", "either", "else", "enough", "even", "ever", "every",
            "everyone", "everything", "except", "few", "for", "from", "further",
            "get", "gets", "got", "had", "has", "have", "having", "he", "her",
            "here", "hers", "herself", "him", "himself", "his", "how", "however",
            "into", "its", "itself", "just", "keep", "kept", "know", "known",
            "largely", "last", "later", "least", "less", "let", "lets", "like",
            "likely", "long", "made", "make", "makes", "many", "may", "maybe",
            "me", "might", "more", "most", "mostly", "much", "must", "my",
            "myself", "near", "nearly", "necessary", "neither", "never", "new",
            "next", "no", "nobody", "none", "noone", "nor", "not", "nothing",
            "now", "nowhere", "off", "often", "old", "once", "one", "only",
            "onto", "or", "other", "others", "otherwise", "our", "ours",
            "ourselves", "out", "outside", "over", "own", "part", "perhaps",
            "place", "possible", "present", "probably", "put", "quite", "rather",
            "really", "right", "said", "same", "say", "says", "see", "seem",
            "seemed", "seems", "seen", "self", "several", "shall", "she",
            "should", "show", "showed", "shows", "side", "since", "so", "some",
            "somebody", "somehow", "someone", "something", "sometime", "sometimes",
            "somewhere", "still", "such", "sure", "take", "taken", "than", "that",
            "the", "their", "them", "themselves", "then", "there", "thereby",
            "therefore", "these", "they", "this", "those", "though", "through",
            "throughout", "thus", "to", "together", "too", "toward", "towards",
            "told", "took", "under", "until", "up", "upon", "us", "use", "used",
            "uses", "using", "very", "want", "wanted", "wants", "was", "way",
            "ways", "we", "well", "went", "were", "what", "whatever", "when",
            "whenever", "where", "wherever", "whether", "which", "while", "who",
            "whoever", "whom", "whose", "why", "will", "with", "within",
            "without", "would", "yet", "you", "your", "yours", "yourself",
            "yourselves", "able", "almost", "also", "any", "are", "been",
            "being", "came", "come", "could", "did", "does", "doing", "done",
            "each", "either", "else", "even", "every", "for", "from", "get",
            "gets", "got", "had", "has", "have", "having", "her", "here",
            "him", "his", "how", "its", "just", "may", "now", "off", "old",
            "one", "only", "our", "out", "own", "put", "say", "see", "set",
            "she", "the", "too", "top", "try", "two", "use", "way", "who",
            "why", "yet", "you",
        }

        known_concepts = {c.definition.name.lower() for c in self._concept_engine.all_concepts()}

        for explanation in session.explanations:
            text = explanation.explanation_text.lower()
            words = text.replace(",", " ").replace(".", " ").replace(";", " ").replace(":", " ").split()

            for word in words:
                clean = word.strip().strip("'\"")
                # Must be 5+ chars
                if len(clean) < 5:
                    continue
                # Must not be a common word
                if clean in BLOCKLIST:
                    continue
                # Must not already exist
                if clean in known_concepts:
                    continue
                # Must look like a noun (trading terms are nouns)
                # Skip verbs and adjectives that slipped through
                if clean.endswith(("ing", "ed", "ly", "est", "er", "tion")):
                    # Allow exceptions that are valid trading terms
                    if clean not in ("absorption", "imbalance", "divergence", "exhaustion", "auction"):
                        continue

                concept = Concept(
                    definition=ConceptDefinition(
                        name=clean.capitalize(),
                        definition=f"Concept extracted from {session.lesson_reference}",
                        visual_appearance="unknown",
                        teacher_explanation=explanation.explanation_text,
                    ),
                    lesson_references=(session.lesson_reference,),
                )
                result = self._concept_engine.register(concept)
                if result.created:
                    known_concepts.add(clean)

    def _concepts_for_session(self, session: LearningSession) -> Tuple[Concept, ...]:
        """Return all concepts touched by a session."""
        results = []
        for ref in session.concept_references:
            c = self._concept_engine.get(ref)
            if c is not None:
                results.append(c)
        return tuple(results)
