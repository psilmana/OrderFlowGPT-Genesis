"""Deterministic tests for the Genesis Apprentice Layer.

These tests validate the Concept Engine, Experience Engine, Learning Loop,
Knowledge Graph, Decision Hierarchy, and Integration Layer. They perform
no I/O, no networking, no AI, no ML, and no side effects.
"""

from decimal import Decimal

import pytest

from orderflowgpt_genesis.apprentice import (
    # Runner Integration
    FrameApprenticeResult,
    LessonApprenticeResult,
    ApprenticeReport,
    RunnerIntegrationConfiguration,
    RunnerIntegrationStatistics,
    ApprenticeLessonProcessor,
    ApprenticeReportBuilder,
    ApprenticeRunnerIntegration,
    # Coach
    ExplanationConfidence,
    ConceptCitation,
    SimilarLesson,
    MissingEvidence,
    MarketNarrative,
    ExplanationResult,
    CoachConfiguration,
    CoachStatistics,
    CoachResult,
    ChartExplainer,
    # Reasoning
    MarketStateType,
    LocationType,
    AggressionType,
    RiskLevel,
    ManagementType,
    ReflectionType,
    MarketStateAssessment,
    LocationAssessment,
    AggressionAssessment,
    RiskAssessment,
    ManagementAssessment,
    ReflectionAssessment,
    DecisionHierarchyConfiguration,
    DecisionHierarchyResult,
    DecisionHierarchyAnalyzer,
    hierarchy_statistics,
    # Concepts
    ConceptConfidenceLevel,
    ConceptExample,
    ConceptDefinition,
    ConceptConfidence,
    ConceptEvolution,
    Concept,
    ConceptConfiguration,
    ConceptEngine,
    ConceptStatistics,
    # Experiences
    ExperienceOutcomeType,
    ExperienceObservation,
    ExperienceEvidence,
    ExperienceReasoning,
    ExperienceOutcome,
    ExperienceReflection,
    Experience,
    ExperienceConfiguration,
    ExperienceEngine,
    ExperienceStatistics,
    # Knowledge Graph
    RelationshipType,
    KnowledgeRelationship,
    KnowledgeGraphConfiguration,
    KnowledgeGraphQuery,
    ConceptPath,
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    # Learning Loop
    LearningPhase,
    QuestionType,
    LearningQuestion,
    TeacherExplanation,
    LearningObservation,
    LearningReflection,
    LearningSession,
    LearningLoopConfiguration,
    LearningLoop,
    LearningLoopStatistics,
    # Integration
    ApprenticeConfiguration,
    ApprenticeLayer,
)


# ============================================================================
# Reasoning / Decision Hierarchy Tests
# ============================================================================

class TestDecisionHierarchyConfiguration:
    def test_default_configuration_valid(self):
        config = DecisionHierarchyConfiguration()
        assert config.trend_confidence_threshold == Decimal("0.60")

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            DecisionHierarchyConfiguration(trend_confidence_threshold=Decimal("1.5"))

    def test_extreme_below_high_rejected(self):
        with pytest.raises(ValueError):
            DecisionHierarchyConfiguration(
                risk_high_threshold=Decimal("0.80"),
                risk_extreme_threshold=Decimal("0.70"),
            )


class TestMarketStateAssessment:
    def test_valid_assessment(self):
        assessment = MarketStateAssessment(
            state=MarketStateType.TRENDING_UP,
            evidence=("trend_confirmed",),
            confidence=Decimal("0.75"),
        )
        assert assessment.state == MarketStateType.TRENDING_UP

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValueError):
            MarketStateAssessment(
                state=MarketStateType.BALANCED,
                evidence=(),
                confidence=Decimal("1.5"),
            )


class TestDecisionHierarchyResult:
    def test_level_count_empty(self):
        result = DecisionHierarchyResult()
        assert result.level_count() == 0
        assert not result.hierarchy_complete

    def test_level_count_partial(self):
        result = DecisionHierarchyResult(
            market_state=MarketStateAssessment(
                state=MarketStateType.BALANCED,
                evidence=(),
                confidence=Decimal("0.50"),
            ),
            location=LocationAssessment(
                location=LocationType.AT_POC,
                evidence=(),
                confidence=Decimal("0.60"),
            ),
        )
        assert result.level_count() == 2

    def test_overall_confidence_capped(self):
        with pytest.raises(ValueError):
            DecisionHierarchyResult(overall_confidence=Decimal("1.1"))


class TestDecisionHierarchyAnalyzer:
    def test_analyze_empty_graph(self):
        """Analyzer should return empty result for empty graph."""
        analyzer = DecisionHierarchyAnalyzer()
        class FakeGraph:
            pass
        result = analyzer.analyze(FakeGraph())
        assert result.market_state is None
        assert result.location is None
        assert result.aggression is None
        assert result.overall_confidence == Decimal("0")

    def test_hierarchy_statistics_empty(self):
        stats = hierarchy_statistics(())
        assert stats.total_assessments == 0


# ============================================================================
# Concept Engine Tests
# ============================================================================

class TestConceptDefinition:
    def test_valid_definition(self):
        definition = ConceptDefinition(
            name="Absorption",
            definition="Passive side absorbing aggressive side volume",
            visual_appearance="Large passive volume at one price level",
            teacher_explanation="Fabio explains absorption...",
        )
        assert definition.name == "Absorption"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            ConceptDefinition(name="", definition="test", visual_appearance="", teacher_explanation="")


class TestConceptConfidence:
    def test_novice_default(self):
        conf = ConceptConfidence(
            level=ConceptConfidenceLevel.NOVICE,
            score=Decimal("0"),
            evidence_count=0,
            positive_example_count=0,
            negative_example_count=0,
        )
        assert conf.level == ConceptConfidenceLevel.NOVICE

    def test_invalid_score_rejected(self):
        with pytest.raises(ValueError):
            ConceptConfidence(
                level=ConceptConfidenceLevel.EXPERT,
                score=Decimal("1.5"),
                evidence_count=1,
                positive_example_count=1,
                negative_example_count=0,
            )


class TestConcept:
    def test_concept_id_auto_derived(self):
        concept = Concept(
            definition=ConceptDefinition(
                name="TestConcept",
                definition="A test concept",
                visual_appearance="none",
                teacher_explanation="test",
            )
        )
        assert concept.concept_id
        assert len(concept.concept_id) == 16

    def test_with_example_positive(self):
        concept = Concept(
            definition=ConceptDefinition(
                name="Test",
                definition="test",
                visual_appearance="none",
                teacher_explanation="test",
            )
        )
        example = ConceptExample(
            description="positive example",
            chart_context="ES 1min",
            evidence_reference="frame001",
            is_positive=True,
        )
        updated = concept.with_example(example)
        assert len(updated.positive_examples) == 1
        assert len(updated.evolution_history) == 0  # confidence not changed yet

    def test_with_confidence_creates_evolution(self):
        concept = Concept(
            definition=ConceptDefinition(
                name="Test",
                definition="test",
                visual_appearance="none",
                teacher_explanation="test",
            )
        )
        new_conf = ConceptConfidence(
            level=ConceptConfidenceLevel.DEVELOPING,
            score=Decimal("0.30"),
            evidence_count=2,
            positive_example_count=2,
            negative_example_count=0,
            last_updated="2026-08-04",
        )
        updated = concept.with_confidence(new_conf)
        assert len(updated.evolution_history) == 1
        assert updated.confidence.level == ConceptConfidenceLevel.DEVELOPING

    def test_positive_examples_must_be_positive(self):
        with pytest.raises(ValueError):
            Concept(
                definition=ConceptDefinition(
                    name="Test",
                    definition="test",
                    visual_appearance="none",
                    teacher_explanation="test",
                ),
                positive_examples=(
                    ConceptExample(
                        description="wrong",
                        chart_context="",
                        evidence_reference="ref",
                        is_positive=False,
                    ),
                ),
            )


class TestConceptConfiguration:
    def test_level_for_score_novice(self):
        config = ConceptConfiguration()
        assert config.level_for_score(Decimal("0.10"), 0) == ConceptConfidenceLevel.NOVICE

    def test_level_for_score_expert(self):
        config = ConceptConfiguration()
        assert config.level_for_score(Decimal("0.95"), 20) == ConceptConfidenceLevel.EXPERT

    def test_level_for_score_insufficient_examples(self):
        config = ConceptConfiguration()
        assert config.level_for_score(Decimal("0.95"), 5) == ConceptConfidenceLevel.COMPETENT

    def test_non_monotonic_thresholds_rejected(self):
        with pytest.raises(ValueError):
            ConceptConfiguration(
                developing_threshold=Decimal("0.50"),
                competent_threshold=Decimal("0.40"),
            )


class TestConceptEngine:
    def test_register_and_retrieve(self):
        engine = ConceptEngine()
        concept = Concept(
            definition=ConceptDefinition(
                name="Absorption",
                definition="Passive side absorbing aggressive side",
                visual_appearance="Large passive volume",
                teacher_explanation="Fabio explains...",
            )
        )
        result = engine.register(concept)
        assert len(result.created) == 1
        assert engine.get(concept.concept_id) is not None

    def test_get_by_name(self):
        engine = ConceptEngine()
        concept = Concept(
            definition=ConceptDefinition(
                name="POC",
                definition="Point of Control",
                visual_appearance="Highest volume row",
                teacher_explanation="Fabio explains POC...",
            )
        )
        engine.register(concept)
        found = engine.get_by_name("POC")
        assert found is not None
        assert found.definition.name == "POC"

    def test_add_example_updates_confidence(self):
        engine = ConceptEngine()
        concept = Concept(
            definition=ConceptDefinition(
                name="Test",
                definition="test",
                visual_appearance="none",
                teacher_explanation="test",
            )
        )
        engine.register(concept)
        example = ConceptExample(
            description="example",
            chart_context="ES",
            evidence_reference="frame001",
            is_positive=True,
        )
        result = engine.add_example(concept.concept_id, example)
        updated = result.updated[0]
        assert updated.confidence.evidence_count == 1
        assert updated.confidence.score == Decimal("1")

    def test_statistics(self):
        engine = ConceptEngine()
        stats = engine._compute_statistics()
        assert stats.total_concepts == 0


# ============================================================================
# Experience Engine Tests
# ============================================================================

class TestExperience:
    def test_experience_id_auto_derived(self):
        exp = Experience(
            observation=ExperienceObservation(
                what_was_seen="absorption at POC",
                detection_graph_reference="graph001",
                frame_reference="frame001",
                timestamp="2026-08-04T18:00:00Z",
            ),
            teacher_explanation="Fabio explains absorption...",
        )
        assert exp.experience_id
        assert len(exp.experience_id) == 16

    def test_is_complete_false_initially(self):
        exp = Experience(
            observation=ExperienceObservation(
                what_was_seen="test",
                detection_graph_reference="g1",
                frame_reference="f1",
                timestamp="t1",
            ),
            teacher_explanation="test",
        )
        assert not exp.is_complete()
        assert exp.completion_level() == 1

    def test_is_complete_true(self):
        exp = Experience(
            observation=ExperienceObservation(
                what_was_seen="test",
                detection_graph_reference="g1",
                frame_reference="f1",
                timestamp="t1",
            ),
            teacher_explanation="test",
            reasoning=ExperienceReasoning(
                market_state_summary="balanced",
                location_summary="at_poc",
                aggression_summary="buyer_aggressive",
                risk_summary="medium",
                management_summary="observe",
            ),
            outcome=ExperienceOutcome(
                outcome_type=ExperienceOutcomeType.VALIDATED,
                actual_result="price moved up",
            ),
            reflection=ExperienceReflection(
                lesson_learned="absorption worked",
            ),
        )
        assert exp.is_complete()
        assert exp.completion_level() == 4

    def test_with_reasoning(self):
        exp = Experience(
            observation=ExperienceObservation(
                what_was_seen="test",
                detection_graph_reference="g1",
                frame_reference="f1",
                timestamp="t1",
            ),
            teacher_explanation="test",
        )
        reasoning = ExperienceReasoning(
            market_state_summary="trending",
            location_summary="at_vah",
            aggression_summary="seller_aggressive",
            risk_summary="high",
            management_summary="wait",
        )
        updated = exp.with_reasoning(reasoning)
        assert updated.reasoning is not None
        assert updated.experience_id == exp.experience_id

    def test_empty_teacher_explanation_rejected(self):
        with pytest.raises(ValueError):
            Experience(
                observation=ExperienceObservation(
                    what_was_seen="test",
                    detection_graph_reference="g1",
                    frame_reference="f1",
                    timestamp="t1",
                ),
                teacher_explanation="",
            )


class TestExperienceEngine:
    def test_create_experience(self):
        engine = ExperienceEngine()
        observation = ExperienceObservation(
            what_was_seen="absorption",
            detection_graph_reference="g1",
            frame_reference="f1",
            timestamp="t1",
        )
        result = engine.create(observation, "Fabio explains...")
        assert len(result.created) == 1
        assert engine.get(result.created[0].experience_id) is not None

    def test_add_reasoning(self):
        engine = ExperienceEngine()
        obs = ExperienceObservation(
            what_was_seen="test",
            detection_graph_reference="g1",
            frame_reference="f1",
            timestamp="t1",
        )
        result = engine.create(obs, "test")
        exp_id = result.created[0].experience_id
        reasoning = ExperienceReasoning(
            market_state_summary="balanced",
            location_summary="at_poc",
            aggression_summary="buyer_aggressive",
            risk_summary="medium",
            management_summary="observe",
        )
        engine.add_reasoning(exp_id, reasoning)
        assert engine.get(exp_id).reasoning is not None

    def test_experiences_ready_for_memory(self):
        engine = ExperienceEngine(ExperienceConfiguration(require_reflection_for_memory=True))
        obs = ExperienceObservation(
            what_was_seen="test",
            detection_graph_reference="g1",
            frame_reference="f1",
            timestamp="t1",
        )
        result = engine.create(obs, "test")
        ready = engine.experiences_ready_for_memory()
        assert len(ready) == 0  # no reflection yet

        exp_id = result.created[0].experience_id
        engine.add_reflection(exp_id, ExperienceReflection(lesson_learned="test"))
        ready = engine.experiences_ready_for_memory()
        assert len(ready) == 1

    def test_statistics(self):
        engine = ExperienceEngine()
        stats = engine._compute_statistics()
        assert stats.total_experiences == 0


# ============================================================================
# Knowledge Graph Tests
# ============================================================================

class TestKnowledgeRelationship:
    def test_valid_relationship(self):
        rel = KnowledgeRelationship(
            source_concept="Absorption",
            relationship_type=RelationshipType.REQUIRES,
            target_concept="Aggression",
        )
        assert rel.relationship_id
        assert rel.source_concept == "Absorption"

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError):
            KnowledgeRelationship(
                source_concept="",
                relationship_type=RelationshipType.RELATED_TO,
                target_concept="Test",
            )

    def test_reverse_requires(self):
        rel = KnowledgeRelationship(
            source_concept="Absorption",
            relationship_type=RelationshipType.REQUIRES,
            target_concept="Aggression",
        )
        rev = rel.reverse()
        assert rev.source_concept == "Aggression"
        assert rev.relationship_type == RelationshipType.ENABLES

    def test_reverse_contradicts(self):
        rel = KnowledgeRelationship(
            source_concept="A",
            relationship_type=RelationshipType.CONTRADICTS,
            target_concept="B",
        )
        rev = rel.reverse()
        assert rev.relationship_type == RelationshipType.CONTRADICTS


class TestKnowledgeGraph:
    def test_add_concept(self):
        graph = KnowledgeGraph().with_concept("Absorption")
        assert "Absorption" in graph.concepts

    def test_add_relationship(self):
        graph = KnowledgeGraph().with_concept("Absorption").with_concept("Aggression")
        rel = KnowledgeRelationship(
            source_concept="Absorption",
            relationship_type=RelationshipType.REQUIRES,
            target_concept="Aggression",
        )
        graph = graph.with_relationship(rel)
        assert len(graph.relationships) == 1

    def test_query_by_source(self):
        graph = (
            KnowledgeGraph()
            .with_concept("Absorption")
            .with_concept("Aggression")
            .with_concept("MarketState")
        )
        rel1 = KnowledgeRelationship("Absorption", RelationshipType.REQUIRES, "Aggression")
        rel2 = KnowledgeRelationship("Aggression", RelationshipType.REQUIRES, "MarketState")
        graph = graph.with_relationship(rel1).with_relationship(rel2)

        query = KnowledgeGraphQuery(source_concept="Absorption")
        results = graph.query(query)
        assert len(results) == 1
        assert results[0].target_concept == "Aggression"

    def test_what_requires(self):
        graph = (
            KnowledgeGraph()
            .with_concept("Absorption")
            .with_concept("Aggression")
            .with_relationship(KnowledgeRelationship("Absorption", RelationshipType.REQUIRES, "Aggression"))
        )
        reqs = graph.what_requires("Absorption")
        assert reqs == ("Aggression",)

    def test_path_between(self):
        graph = (
            KnowledgeGraph()
            .with_concept("Absorption")
            .with_concept("Aggression")
            .with_concept("MarketState")
            .with_relationship(KnowledgeRelationship("Absorption", RelationshipType.REQUIRES, "Aggression"))
            .with_relationship(KnowledgeRelationship("Aggression", RelationshipType.REQUIRES, "MarketState"))
        )
        paths = graph.path_between("Absorption", "MarketState")
        assert len(paths) >= 1
        shortest = graph.shortest_path("Absorption", "MarketState")
        assert shortest is not None
        assert shortest.path_length == 2

    def test_unknown_concept_path(self):
        graph = KnowledgeGraph().with_concept("A")
        paths = graph.path_between("A", "B")
        assert len(paths) == 0

    def test_concept_not_in_graph_rejected(self):
        with pytest.raises(ValueError):
            KnowledgeGraph(
                concepts=("A",),
                relationships=(
                    KnowledgeRelationship("A", RelationshipType.REQUIRES, "B"),
                ),
            )


class TestKnowledgeGraphBuilder:
    def test_build_empty(self):
        builder = KnowledgeGraphBuilder()
        graph = builder.build()
        assert len(graph.concepts) == 0
        assert len(graph.relationships) == 0

    def test_build_with_chain(self):
        builder = KnowledgeGraphBuilder()
        builder.add_concept("Absorption")
        builder.add_concept("Aggression")
        builder.add_relationship(KnowledgeRelationship("Absorption", RelationshipType.REQUIRES, "Aggression"))
        graph = builder.build()
        assert len(graph.concepts) == 2
        assert len(graph.relationships) == 1


# ============================================================================
# Learning Loop Tests
# ============================================================================

class TestLearningSession:
    def test_session_creation(self):
        session = LearningSession(
            session_id="s001",
            lesson_reference="Lesson01",
        )
        assert session.session_id == "s001"
        assert not session.is_complete

    def test_with_phase(self):
        session = LearningSession(session_id="s001", lesson_reference="L01")
        updated = session.with_phase(LearningPhase.OBSERVE)
        assert LearningPhase.OBSERVE in updated.phases_completed
        assert LearningPhase.OBSERVE not in session.phases_completed

    def test_with_question(self):
        session = LearningSession(session_id="s001", lesson_reference="L01")
        question = LearningQuestion(
            question_text="What is absorption?",
            question_type=QuestionType.WHAT,
            context="Lesson01",
            timestamp="2026-08-04T18:00:00Z",
        )
        updated = session.with_question(question)
        assert len(updated.questions) == 1

    def test_mark_complete(self):
        session = LearningSession(session_id="s001", lesson_reference="L01")
        updated = session.mark_complete()
        assert updated.is_complete

    def test_empty_session_id_rejected(self):
        with pytest.raises(ValueError):
            LearningSession(session_id="", lesson_reference="L01")


class TestLearningLoop:
    def test_start_session(self):
        loop = self._make_loop()
        session = loop.start_session("s001", "Lesson01")
        assert session.session_id == "s001"
        assert loop.get_session("s001") is not None

    def test_observe_phase(self):
        loop = self._make_loop()
        loop.start_session("s001", "Lesson01")
        class FakeGraph:
            pass
        result = loop.observe("s001", FakeGraph(), "2026-08-04T18:00:00Z")
        assert result.session.has_phase(LearningPhase.OBSERVE)
        assert len(result.session.observations) == 1

    def test_explain_requires_observation(self):
        loop = self._make_loop()
        loop.start_session("s001", "Lesson01")
        with pytest.raises(ValueError):
            loop.explain("s001", "test", "src", Decimal("0.9"), "t1")

    def test_full_cycle(self):
        loop = self._make_loop()
        loop.start_session("s001", "Lesson01")
        class FakeGraph:
            pass

        loop.observe("s001", FakeGraph(), "t1")
        loop.question("s001", "What?", QuestionType.WHAT, "ctx", "t1")
        loop.explain("s001", "Explanation", "src", Decimal("0.9"), "t1")
        loop.reflect("s001", "Reflection", ("insight",), Decimal("0.05"), "t1")
        loop.experience("s001", ExperienceOutcomeType.PENDING, "pending", "t1")
        loop.to_memory("s001")
        loop.to_knowledge("s001")
        result = loop.complete_session("s001")

        assert result.session.is_complete
        assert result.session.has_phase(LearningPhase.OBSERVE)
        assert result.session.has_phase(LearningPhase.KNOWLEDGE)
        assert result.knowledge_graph is not None

    def _make_loop(self):
        from orderflowgpt_genesis.apprentice import ConceptEngine, ExperienceEngine, DecisionHierarchyAnalyzer
        return LearningLoop(
            concept_engine=ConceptEngine(),
            experience_engine=ExperienceEngine(),
            hierarchy_analyzer=DecisionHierarchyAnalyzer(),
            configuration=LearningLoopConfiguration(require_question_before_explanation=False),
        )


# ============================================================================
# Integration / ApprenticeLayer Tests
# ============================================================================

class TestApprenticeLayer:
    def test_explain_chart_empty(self):
        layer = ApprenticeLayer()
        class FakeGraph:
            pass
        result = layer.explain_chart(FakeGraph())
        assert isinstance(result, DecisionHierarchyResult)

    def test_register_and_get_concept(self):
        layer = ApprenticeLayer()
        concept = Concept(
            definition=ConceptDefinition(
                name="TestConcept",
                definition="A test",
                visual_appearance="none",
                teacher_explanation="test",
            )
        )
        layer.register_concept(concept)
        found = layer.get_concept(concept.concept_id)
        assert found is not None
        assert found.definition.name == "TestConcept"

    def test_all_concepts_empty(self):
        layer = ApprenticeLayer()
        assert layer.all_concepts() == ()

    def test_configuration_defaults(self):
        config = ApprenticeConfiguration()
        assert config.enable_auto_concept_extraction is True
        assert config.enable_experience_to_memory is True


# ============================================================================
# Live Coach / ChartExplainer Tests
# ============================================================================

class TestCoachConfiguration:
    def test_default_valid(self):
        config = CoachConfiguration()
        assert config.min_concept_relevance_score == Decimal("0.20")
        assert config.max_concept_citations == 5

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            CoachConfiguration(min_concept_relevance_score=Decimal("1.5"))


class TestConceptCitation:
    def test_valid_citation(self):
        citation = ConceptCitation(
            concept_name="Absorption",
            concept_id="abc123",
            relevance_score=Decimal("0.85"),
            teacher_explanation_snippet="Passive side holding...",
            why_relevant="matches aggression pattern",
            confidence_level=ConceptConfidenceLevel.COMPETENT,
        )
        assert citation.concept_name == "Absorption"
        assert citation.relevance_score == Decimal("0.85")

    def test_invalid_relevance_rejected(self):
        with pytest.raises(ValueError):
            ConceptCitation(
                concept_name="Test",
                concept_id="id",
                relevance_score=Decimal("1.5"),
                teacher_explanation_snippet="test",
                why_relevant="test",
                confidence_level=ConceptConfidenceLevel.NOVICE,
            )


class TestSimilarLesson:
    def test_valid_lesson(self):
        lesson = SimilarLesson(
            lesson_reference="Lesson01",
            similarity_score=Decimal("0.75"),
            teacher_explanation="Fabio explains...",
            concept_references=("absorption",),
        )
        assert lesson.lesson_reference == "Lesson01"

    def test_invalid_similarity_rejected(self):
        with pytest.raises(ValueError):
            SimilarLesson(
                lesson_reference="L01",
                similarity_score=Decimal("-0.1"),
                teacher_explanation="test",
                concept_references=(),
            )


class TestMissingEvidence:
    def test_missing_evidence(self):
        me = MissingEvidence(
            field_name="market_state",
            why_missing="no trend data",
            impact_on_confidence="high",
            suggestion="run trend engine",
        )
        assert me.field_name == "market_state"


class TestMarketNarrative:
    def test_narrative_creation(self):
        narrative = MarketNarrative(
            summary="Market is trending up at POC.",
            market_state_narrative="The market is trending up.",
            location_narrative="Price is at POC.",
            aggression_narrative="Buyers are aggressive.",
            risk_narrative="Risk is medium.",
            management_narrative="Observe for continuation.",
            reflection_narrative="No prior reflection.",
            concept_summary="Relevant concepts: Absorption, POC.",
        )
        assert "trending up" in narrative.summary


class TestExplanationResult:
    def test_valid_result(self):
        narrative = MarketNarrative(
            summary="Test summary",
            market_state_narrative="state",
            location_narrative="loc",
            aggression_narrative="agg",
            risk_narrative="risk",
            management_narrative="mgmt",
            reflection_narrative="refl",
            concept_summary="concepts",
        )
        hierarchy = DecisionHierarchyResult()
        result = ExplanationResult(
            narrative=narrative,
            hierarchy_result=hierarchy,
            concept_citations=(),
            similar_lessons=(),
            related_concepts=(),
            missing_evidence=(),
            overall_confidence=Decimal("0.75"),
            explanation_confidence=ExplanationConfidence.HIGH,
            explanation_id="exp001",
        )
        assert result.overall_confidence == Decimal("0.75")
        assert result.explanation_confidence == ExplanationConfidence.HIGH

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValueError):
            ExplanationResult(
                narrative=MarketNarrative(
                    summary="test", market_state_narrative="s", location_narrative="l",
                    aggression_narrative="a", risk_narrative="r", management_narrative="m",
                    reflection_narrative="r2", concept_summary="c",
                ),
                hierarchy_result=DecisionHierarchyResult(),
                concept_citations=(),
                similar_lessons=(),
                related_concepts=(),
                missing_evidence=(),
                overall_confidence=Decimal("1.5"),
                explanation_confidence=ExplanationConfidence.LOW,
                explanation_id="id",
            )


class TestChartExplainer:
    def test_explain_empty_graph(self):
        layer = ApprenticeLayer()
        explainer = ChartExplainer(layer)
        class FakeGraph:
            pass
        result = explainer.explain(FakeGraph())
        assert result.explanation_confidence == ExplanationConfidence.INSUFFICIENT
        assert len(result.missing_evidence) >= 1

    def test_explain_with_text(self):
        layer = ApprenticeLayer()
        explainer = ChartExplainer(layer)
        class FakeGraph:
            pass
        text = explainer.explain_with_text(FakeGraph())
        assert "GENESIS LIVE COACH" in text
        assert "What is happening?" in text

    def test_explain_with_concepts(self):
        layer = ApprenticeLayer()
        # Register a concept that matches a fake hierarchy
        concept = Concept(
            definition=ConceptDefinition(
                name="Absorption",
                definition="Passive side absorbing aggressive side",
                visual_appearance="Large passive volume",
                teacher_explanation="When you see the passive side holding at POC...",
            ),
            positive_examples=(
                ConceptExample(
                    description="example",
                    chart_context="ES",
                    evidence_reference="frame001",
                    is_positive=True,
                ),
            ),
        )
        layer.register_concept(concept)

        explainer = ChartExplainer(layer)
        class FakeGraph:
            pass
        result = explainer.explain(FakeGraph())
        # Even with empty hierarchy, the concept engine has concepts
        # but they won't be cited because hierarchy has no keywords
        assert result.explanation_id

    def test_explain_retrieves_lessons(self):
        layer = ApprenticeLayer()
        explainer = ChartExplainer(layer)

        # Create an experience with a teacher explanation
        obs = ExperienceObservation(
            what_was_seen="absorption at POC",
            detection_graph_reference="g1",
            frame_reference="f1",
            timestamp="t1",
        )
        layer._experience_engine.create(obs, "Fabio explains absorption at POC...", session_reference="Lesson01")

        class FakeGraph:
            pass
        result = explainer.explain(FakeGraph())
        # With empty hierarchy, no lessons should match
        assert result.similar_lessons == ()

    def test_statistics(self):
        layer = ApprenticeLayer()
        explainer = ChartExplainer(layer)
        class FakeGraph:
            pass
        explainer.explain(FakeGraph())
        stats = explainer.get_statistics()
        assert stats.total_explanations == 1

    def test_explanation_confidence_classification(self):
        layer = ApprenticeLayer()
        config = CoachConfiguration(
            min_hierarchy_confidence_for_high=Decimal("0.70"),
            min_hierarchy_confidence_for_medium=Decimal("0.40"),
        )
        explainer = ChartExplainer(layer, config)
        class FakeGraph:
            pass
        result = explainer.explain(FakeGraph())
        # Empty graph should be INSUFFICIENT
        assert result.explanation_confidence == ExplanationConfidence.INSUFFICIENT

    def test_coach_result(self):
        layer = ApprenticeLayer()
        explainer = ChartExplainer(layer)
        class FakeGraph:
            pass
        result = explainer.explain(FakeGraph())
        assert isinstance(result, ExplanationResult)


# ============================================================================
# Runner Integration Tests
# ============================================================================

class TestRunnerIntegrationConfiguration:
    def test_default_valid(self):
        config = RunnerIntegrationConfiguration()
        assert config.process_every_frame is False
        assert config.process_key_frames_only is True
        assert config.key_frame_interval == 30
        assert config.enable_coach is True

    def test_invalid_key_frame_interval(self):
        with pytest.raises(ValueError):
            RunnerIntegrationConfiguration(key_frame_interval=0)

    def test_invalid_max_frames(self):
        with pytest.raises(ValueError):
            RunnerIntegrationConfiguration(max_frames_per_lesson=0)

    def test_invalid_detail_level(self):
        with pytest.raises(ValueError):
            RunnerIntegrationConfiguration(report_concept_detail_level="invalid")


class TestFrameApprenticeResult:
    def test_creation(self):
        result = FrameApprenticeResult(
            frame_index=0,
            frame_timestamp="2026-08-04T18:00:00Z",
            session_id="L01_frame0000",
            hierarchy_result=None,
            experience_id=None,
            concept_references=(),
            coach_explanation=None,
            explanation_text="",
        )
        assert result.frame_index == 0
        assert result.session_id == "L01_frame0000"


class TestLessonApprenticeResult:
    def test_empty_lesson(self):
        result = LessonApprenticeResult(
            lesson_reference="Lesson01",
            frame_results=(),
            sessions=(),
            concepts=(),
            experiences=(),
            knowledge_graph=KnowledgeGraph(),
            concept_statistics=ConceptStatistics(),
            experience_statistics=ExperienceStatistics(),
            learning_statistics=LearningLoopStatistics(),
            coach_statistics=CoachStatistics(),
        )
        assert result.lesson_reference == "Lesson01"
        assert len(result.frame_results) == 0


class TestApprenticeReport:
    def test_empty_report(self):
        report = ApprenticeReport(
            report_id="r001",
            lesson_results=(),
            total_concepts_learned=0,
            total_experiences_created=0,
            total_sessions_completed=0,
            total_coach_explanations=0,
            total_frames_processed=0,
            concept_mastery_distribution=(),
            top_concepts_by_confidence=(),
            knowledge_graph_summary="empty",
            what_was_learned=(),
            what_needs_more_study=(),
            overall_learning_confidence=Decimal("0"),
            report_timestamp="t1",
        )
        assert report.total_concepts_learned == 0
        assert report.overall_learning_confidence == Decimal("0")

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValueError):
            ApprenticeReport(
                report_id="r001",
                lesson_results=(),
                total_concepts_learned=0,
                total_experiences_created=0,
                total_sessions_completed=0,
                total_coach_explanations=0,
                total_frames_processed=0,
                concept_mastery_distribution=(),
                top_concepts_by_confidence=(),
                knowledge_graph_summary="empty",
                what_was_learned=(),
                what_needs_more_study=(),
                overall_learning_confidence=Decimal("1.5"),
                report_timestamp="t1",
            )


class TestApprenticeLessonProcessor:
    def test_process_empty_lesson(self):
        apprentice = ApprenticeLayer()
        processor = ApprenticeLessonProcessor(apprentice)
        result = processor.process_lesson("Lesson01", ())
        assert result.lesson_reference == "Lesson01"
        assert len(result.frame_results) == 0
        assert len(result.concepts) == 0
        assert len(result.experiences) == 0

    def test_process_single_frame(self):
        apprentice = ApprenticeLayer()
        processor = ApprenticeLessonProcessor(apprentice)
        class FakeGraph:
            pass
        frames = (
            (0, "2026-08-04T18:00:00Z", FakeGraph(), "Fabio explains absorption at POC"),
        )
        result = processor.process_lesson("Lesson01", frames)
        assert result.lesson_reference == "Lesson01"
        assert len(result.frame_results) == 1
        assert len(result.sessions) == 1
        assert len(result.experiences) == 1

    def test_key_frame_skipping(self):
        apprentice = ApprenticeLayer()
        config = RunnerIntegrationConfiguration(
            process_key_frames_only=True,
            key_frame_interval=5,
            process_every_frame=False,
        )
        processor = ApprenticeLessonProcessor(apprentice, config)
        class FakeGraph:
            pass
        frames = tuple(
            (i, f"t{i}", FakeGraph(), f"Explanation for frame {i}")
            for i in range(10)
        )
        result = processor.process_lesson("Lesson01", frames)
        # Should process frames 0, 5 only
        assert len(result.frame_results) == 2
        assert result.frame_results[0].frame_index == 0
        assert result.frame_results[1].frame_index == 5

    def test_short_explanation_skipped(self):
        apprentice = ApprenticeLayer()
        config = RunnerIntegrationConfiguration(
            min_transcript_length_for_explanation=50,
            process_every_frame=True,
        )
        processor = ApprenticeLessonProcessor(apprentice, config)
        class FakeGraph:
            pass
        frames = (
            (0, "t1", FakeGraph(), "short"),  # too short
            (1, "t2", FakeGraph(), "This is a long enough explanation from Fabio about absorption at the POC level"),
        )
        result = processor.process_lesson("Lesson01", frames)
        assert len(result.frame_results) == 1
        assert result.frame_results[0].frame_index == 1

    def test_max_frames_limit(self):
        apprentice = ApprenticeLayer()
        config = RunnerIntegrationConfiguration(
            process_every_frame=True,
            max_frames_per_lesson=3,
        )
        processor = ApprenticeLessonProcessor(apprentice, config)
        class FakeGraph:
            pass
        frames = tuple(
            (i, f"t{i}", FakeGraph(), f"Explanation {i}")
            for i in range(10)
        )
        result = processor.process_lesson("Lesson01", frames)
        assert len(result.frame_results) == 3

    def test_concept_extraction_enabled(self):
        apprentice = ApprenticeLayer()
        config = RunnerIntegrationConfiguration(enable_concept_extraction=True)
        processor = ApprenticeLessonProcessor(apprentice, config)
        class FakeGraph:
            pass
        frames = (
            (0, "t1", FakeGraph(), "Fabio explains absorption at the POC level with buyer aggression"),
        )
        result = processor.process_lesson("Lesson01", frames)
        # Concepts should have been extracted from the explanation
        assert len(result.concepts) > 0


class TestApprenticeReportBuilder:
    def test_build_empty(self):
        builder = ApprenticeReportBuilder()
        report = builder.build((), "2026-08-04T18:00:00Z")
        assert report.total_concepts_learned == 0
        assert report.what_was_learned == ("No lessons processed.",)

    def test_build_with_lesson(self):
        builder = ApprenticeReportBuilder()
        lesson_result = LessonApprenticeResult(
            lesson_reference="Lesson01",
            frame_results=(),
            sessions=(),
            concepts=(),
            experiences=(),
            knowledge_graph=KnowledgeGraph(),
            concept_statistics=ConceptStatistics(),
            experience_statistics=ExperienceStatistics(),
            learning_statistics=LearningLoopStatistics(),
            coach_statistics=CoachStatistics(),
        )
        report = builder.build((lesson_result,), "2026-08-04T18:00:00Z")
        assert len(report.lesson_results) == 1
        assert len(report.lesson_results) == 1
        assert report.total_concepts_learned == 0

    def test_build_with_concepts(self):
        builder = ApprenticeReportBuilder()
        concept = Concept(
            definition=ConceptDefinition(
                name="Absorption",
                definition="Passive side absorbing",
                visual_appearance="Large volume",
                teacher_explanation="Fabio explains...",
            ),
            confidence=ConceptConfidence(
                level=ConceptConfidenceLevel.COMPETENT,
                score=Decimal("0.75"),
                evidence_count=5,
                positive_example_count=5,
                negative_example_count=0,
            ),
        )
        lesson_result = LessonApprenticeResult(
            lesson_reference="Lesson01",
            frame_results=(),
            sessions=(),
            concepts=(concept,),
            experiences=(),
            knowledge_graph=KnowledgeGraph(),
            concept_statistics=ConceptStatistics(total_concepts=1),
            experience_statistics=ExperienceStatistics(),
            learning_statistics=LearningLoopStatistics(),
            coach_statistics=CoachStatistics(),
        )
        report = builder.build((lesson_result,), "2026-08-04T18:00:00Z")
        assert report.total_concepts_learned == 1
        assert len(report.top_concepts_by_confidence) == 1
        assert report.top_concepts_by_confidence[0] == ("Absorption", Decimal("0.75"))


class TestApprenticeRunnerIntegration:
    def test_initialization(self):
        integration = ApprenticeRunnerIntegration()
        assert integration.get_statistics().total_lessons_processed == 0

    def test_process_lesson(self):
        integration = ApprenticeRunnerIntegration()
        class FakeGraph:
            pass
        frames = (
            (0, "t1", FakeGraph(), "Fabio explains absorption at POC"),
        )
        result = integration.process_lesson("Lesson01", frames)
        assert result.lesson_reference == "Lesson01"
        assert integration.get_statistics().total_lessons_processed == 1

    def test_build_report(self):
        integration = ApprenticeRunnerIntegration()
        class FakeGraph:
            pass
        frames = (
            (0, "t1", FakeGraph(), "Fabio explains absorption"),
        )
        integration.process_lesson("Lesson01", frames)
        report = integration.build_report("2026-08-04T18:00:00Z")
        assert len(report.lesson_results) == 1
        assert len(report.lesson_results) == 1
        assert report.report_id

    def test_reset(self):
        integration = ApprenticeRunnerIntegration()
        class FakeGraph:
            pass
        frames = ((0, "t1", FakeGraph(), "test"),)
        integration.process_lesson("L01", frames)
        assert integration.get_statistics().total_lessons_processed == 1
        integration.reset()
        assert integration.get_statistics().total_lessons_processed == 0

    def test_get_apprentice_layer(self):
        integration = ApprenticeRunnerIntegration()
        layer = integration.get_apprentice_layer()
        assert isinstance(layer, ApprenticeLayer)
