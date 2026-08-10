"""Genesis Apprentice Layer — Educational AI Apprentice system.

This package implements the Concept Engine, Experience Engine, Learning Loop,
Knowledge Graph, and Fabio's Decision Hierarchy for the Genesis 2.0
Apprentice architecture.

Modules:
    reasoning         — Fabio's Decision Hierarchy (Market State -> Location ->
                        Aggression -> Risk -> Management -> Reflection)
    concepts          — Concept Engine for learning, evolving, and tracking
                        concept mastery from Fabio's teaching
    experiences       — Experience Engine for the central learning object
    knowledge_graph   — Relationship graph between concepts (Absorption requires
                        Aggression requires Market State)
    learning_loop     — Observe -> Question -> Explain -> Reflect -> Experience ->
                        Memory -> Knowledge -> Observe Again
    integration       — ApprenticeLayer wiring everything to existing Genesis
                        Bundle 1-13 infrastructure

Usage:
    from orderflowgpt_genesis.apprentice import ApprenticeLayer
    apprentice = ApprenticeLayer()
    result = apprentice.process_frame(
        session_id="lesson01_frame001",
        lesson_reference="Lesson01",
        detection_graph=graph,
        teacher_explanation="We see absorption at the POC...",
        source_reference="Fabio Lesson01 00:03:24",
        timestamp="2026-08-04T18:00:00Z",
    )
    print(result.hierarchy_result.market_state.state)
    print(result.session.phases_completed)
"""

from .reasoning import (
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
    DecisionHierarchyStatistics,
    DecisionHierarchyResult,
    DecisionHierarchyAnalyzer,
    hierarchy_statistics,
)

from .concepts import (
    ConceptConfidenceLevel,
    ConceptExample,
    ConceptDefinition,
    ConceptConfidence,
    ConceptEvolution,
    Concept,
    ConceptConfiguration,
    ConceptStatistics,
    ConceptResult,
    ConceptEngine,
)

from .experiences import (
    ExperienceOutcomeType,
    ExperienceConfidence,
    ExperienceObservation,
    ExperienceEvidence,
    ExperienceReasoning,
    ExperienceOutcome,
    ExperienceReflection,
    Experience,
    ExperienceConfiguration,
    ExperienceStatistics,
    ExperienceResult,
    ExperienceEngine,
)

from .knowledge_graph import (
    RelationshipType,
    KnowledgeRelationship,
    KnowledgeGraphConfiguration,
    KnowledgeGraphStatistics,
    KnowledgeGraphQuery,
    ConceptPath,
    KnowledgeGraph,
    KnowledgeGraphBuilder,
)

from .learning_loop import (
    LearningPhase,
    QuestionType,
    LearningQuestion,
    TeacherExplanation,
    LearningObservation,
    LearningReflection,
    LearningSession,
    LearningLoopConfiguration,
    LearningLoopStatistics,
    LearningLoopResult,
    LearningLoop,
)

from .integration import (
    ApprenticeConfiguration,
    ApprenticeStatistics,
    ApprenticeResult,
    ApprenticeLayer,
)

from .coach import (
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
)

from .runner_integration import (
    FrameApprenticeResult,
    LessonApprenticeResult,
    ApprenticeReport,
    RunnerIntegrationConfiguration,
    RunnerIntegrationStatistics,
    ApprenticeLessonProcessor,
    ApprenticeReportBuilder,
    ApprenticeRunnerIntegration,
)

__all__ = [
    # Reasoning / Decision Hierarchy
    "MarketStateType",
    "LocationType",
    "AggressionType",
    "RiskLevel",
    "ManagementType",
    "ReflectionType",
    "MarketStateAssessment",
    "LocationAssessment",
    "AggressionAssessment",
    "RiskAssessment",
    "ManagementAssessment",
    "ReflectionAssessment",
    "DecisionHierarchyConfiguration",
    "DecisionHierarchyStatistics",
    "DecisionHierarchyResult",
    "DecisionHierarchyAnalyzer",
    "hierarchy_statistics",
    # Concepts
    "ConceptConfidenceLevel",
    "ConceptExample",
    "ConceptDefinition",
    "ConceptConfidence",
    "ConceptEvolution",
    "Concept",
    "ConceptConfiguration",
    "ConceptStatistics",
    "ConceptResult",
    "ConceptEngine",
    # Experiences
    "ExperienceOutcomeType",
    "ExperienceConfidence",
    "ExperienceObservation",
    "ExperienceEvidence",
    "ExperienceReasoning",
    "ExperienceOutcome",
    "ExperienceReflection",
    "Experience",
    "ExperienceConfiguration",
    "ExperienceStatistics",
    "ExperienceResult",
    "ExperienceEngine",
    # Knowledge Graph
    "RelationshipType",
    "KnowledgeRelationship",
    "KnowledgeGraphConfiguration",
    "KnowledgeGraphStatistics",
    "KnowledgeGraphQuery",
    "ConceptPath",
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    # Learning Loop
    "LearningPhase",
    "QuestionType",
    "LearningQuestion",
    "TeacherExplanation",
    "LearningObservation",
    "LearningReflection",
    "LearningSession",
    "LearningLoopConfiguration",
    "LearningLoopStatistics",
    "LearningLoopResult",
    "LearningLoop",
    # Integration
    "ApprenticeConfiguration",
    "ApprenticeStatistics",
    "ApprenticeResult",
    "ApprenticeLayer",
    # Coach
    "ExplanationConfidence",
    "ConceptCitation",
    "SimilarLesson",
    "MissingEvidence",
    "MarketNarrative",
    "ExplanationResult",
    "CoachConfiguration",
    "CoachStatistics",
    "CoachResult",
    "ChartExplainer",
    # Runner Integration
    "FrameApprenticeResult",
    "LessonApprenticeResult",
    "ApprenticeReport",
    "RunnerIntegrationConfiguration",
    "RunnerIntegrationStatistics",
    "ApprenticeLessonProcessor",
    "ApprenticeReportBuilder",
    "ApprenticeRunnerIntegration",
]
