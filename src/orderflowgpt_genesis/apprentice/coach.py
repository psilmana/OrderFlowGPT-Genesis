"""Live Coach for the Genesis Apprentice Layer.

The Coach takes any DetectionGraph and produces a Fabio-style explanation:
- What is the market state?
- Where are we in the structure?
- Who is aggressive?
- What is the risk?
- What should we watch?
- What concepts apply?
- What similar lessons has Fabio taught?
- What evidence is missing?

This module explains markets. It does NOT predict, recommend trades,
or estimate probabilities.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Tuple

from .reasoning import (
    DecisionHierarchyResult,
    MarketStateType,
    LocationType,
    AggressionType,
    RiskLevel,
    ManagementType,
)
from .concepts import Concept, ConceptEngine, ConceptConfidenceLevel
from .experiences import Experience, ExperienceEngine, ExperienceOutcomeType
from .knowledge_graph import KnowledgeGraph, RelationshipType
from .integration import ApprenticeLayer


class ExplanationConfidence(Enum):
    """Deterministic confidence in an explanation."""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INSUFFICIENT = auto()


@dataclass(frozen=True)
class ConceptCitation:
    """A concept cited in an explanation with relevance and context."""
    concept_name: str
    concept_id: str
    relevance_score: Decimal
    teacher_explanation_snippet: str
    why_relevant: str
    confidence_level: ConceptConfidenceLevel

    def __post_init__(self):
        if not (Decimal("0") <= self.relevance_score <= Decimal("1")):
            raise ValueError("relevance_score must be in [0, 1]")


@dataclass(frozen=True)
class SimilarLesson:
    """A past lesson from Fabio that is similar to the current observation."""
    lesson_reference: str
    similarity_score: Decimal
    teacher_explanation: str
    concept_references: Tuple[str, ...]
    experience_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not (Decimal("0") <= self.similarity_score <= Decimal("1")):
            raise ValueError("similarity_score must be in [0, 1]")


@dataclass(frozen=True)
class MissingEvidence:
    """A piece of evidence that is missing or unclear in the explanation."""
    field_name: str
    why_missing: str
    impact_on_confidence: str
    suggestion: str = ""


@dataclass(frozen=True)
class MarketNarrative:
    """Structured narrative explaining what is happening in the market.

    Each field is a sentence or paragraph in Fabio's voice. No trade
    recommendations, no predictions, no probabilities.
    """
    summary: str
    market_state_narrative: str
    location_narrative: str
    aggression_narrative: str
    risk_narrative: str
    management_narrative: str
    reflection_narrative: str
    concept_summary: str


@dataclass(frozen=True)
class ExplanationResult:
    """Complete explanation result for one chart observation.

    This is the primary output of the Live Coach. It contains everything
    Genesis understands about the current chart through Fabio's lens.
    """
    narrative: MarketNarrative
    hierarchy_result: DecisionHierarchyResult
    concept_citations: Tuple[ConceptCitation, ...]
    similar_lessons: Tuple[SimilarLesson, ...]
    related_concepts: Tuple[str, ...]
    missing_evidence: Tuple[MissingEvidence, ...]
    overall_confidence: Decimal
    explanation_confidence: ExplanationConfidence
    explanation_id: str
    concepts_referenced: Tuple[str, ...] = ()
    lessons_referenced: Tuple[str, ...] = ()

    def __post_init__(self):
        if not (Decimal("0") <= self.overall_confidence <= Decimal("1")):
            raise ValueError("overall_confidence must be in [0, 1]")


@dataclass(frozen=True)
class CoachConfiguration:
    """Configuration for the Live Coach."""
    min_concept_relevance_score: Decimal = Decimal("0.20")
    max_concept_citations: int = 5
    max_similar_lessons: int = 3
    max_related_concepts: int = 5
    max_missing_evidence_items: int = 6
    require_hierarchy_for_explanation: bool = True
    min_hierarchy_confidence_for_high: Decimal = Decimal("0.70")
    min_hierarchy_confidence_for_medium: Decimal = Decimal("0.40")
    include_incomplete_levels: bool = True
    narrative_max_length: int = 2000

    def __post_init__(self):
        for name, val in [
            ("min_concept_relevance_score", self.min_concept_relevance_score),
            ("min_hierarchy_confidence_for_high", self.min_hierarchy_confidence_for_high),
            ("min_hierarchy_confidence_for_medium", self.min_hierarchy_confidence_for_medium),
        ]:
            if not (Decimal("0") <= val <= Decimal("1")):
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class CoachStatistics:
    """Statistics over coach explanations."""
    total_explanations: int = 0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    insufficient_count: int = 0
    average_concept_citations: Decimal = Decimal("0")
    average_similar_lessons: Decimal = Decimal("0")
    average_missing_evidence: Decimal = Decimal("0")
    average_overall_confidence: Decimal = Decimal("0")


@dataclass(frozen=True)
class CoachResult:
    """Result of a coach operation."""
    explanation: ExplanationResult
    statistics: CoachStatistics


class ChartExplainer:
    """Deterministic chart explainer implementing the Genesis Live Coach.

    The ChartExplainer consumes a DetectionGraph and the current state of
    the ApprenticeLayer (concepts, experiences, knowledge graph) and
    produces a structured Fabio-style explanation.

    It answers "What is happening?" before "What should I do?"

    No AI, no ML, no prediction, no trade recommendation, no probability
    estimation inside the core. All retrieval is deterministic keyword
    matching and graph traversal.
    """

    def __init__(
        self,
        apprentice_layer: ApprenticeLayer,
        configuration: Optional[CoachConfiguration] = None,
    ):
        self._apprentice = apprentice_layer
        self._configuration = configuration or CoachConfiguration()
        self._explanation_count = 0
        self._confidence_distribution = {
            ExplanationConfidence.HIGH: 0,
            ExplanationConfidence.MEDIUM: 0,
            ExplanationConfidence.LOW: 0,
            ExplanationConfidence.INSUFFICIENT: 0,
        }

    def explain(self, detection_graph) -> ExplanationResult:
        """Explain a chart through Fabio's lens.

        This is the primary method. Given a DetectionGraph, it returns
        a complete ExplanationResult with narrative, citations, similar
        lessons, related concepts, and missing evidence.
        """
        self._explanation_count += 1
        hierarchy = self._apprentice.explain_chart(detection_graph)

        if self._configuration.require_hierarchy_for_explanation and hierarchy.level_count() == 0:
            return self._insufficient_explanation(hierarchy, "no_hierarchy_data")

        concepts = self._retrieve_concepts(hierarchy)
        lessons = self._retrieve_similar_lessons(hierarchy)
        related = self._retrieve_related_concepts(concepts)
        missing = self._identify_missing_evidence(hierarchy)
        narrative = self._build_narrative(hierarchy, concepts, lessons, missing)
        confidence = self._compute_overall_confidence(hierarchy, concepts, missing)
        expl_conf = self._classify_confidence(confidence, hierarchy, missing)

        self._explanation_count += 1
        self._confidence_distribution[expl_conf] += 1

        exp_id = self._derive_explanation_id(detection_graph)

        return ExplanationResult(
            narrative=narrative,
            hierarchy_result=hierarchy,
            concept_citations=concepts,
            similar_lessons=lessons,
            related_concepts=related,
            missing_evidence=missing,
            overall_confidence=confidence,
            explanation_confidence=expl_conf,
            explanation_id=exp_id,
            concepts_referenced=tuple(c.concept_name for c in concepts),
            lessons_referenced=tuple(l.lesson_reference for l in lessons),
        )

    def explain_with_text(self, detection_graph) -> str:
        """Produce a plain-text Fabio-style explanation.

        This is the human-readable version of explain().
        """
        result = self.explain(detection_graph)
        return self._format_explanation(result)

    def get_statistics(self) -> CoachStatistics:
        """Return statistics over all explanations produced."""
        if self._explanation_count == 0:
            return CoachStatistics()
        # These averages are approximate since we don't store all results
        return CoachStatistics(
            total_explanations=self._explanation_count,
            high_confidence_count=self._confidence_distribution[ExplanationConfidence.HIGH],
            medium_confidence_count=self._confidence_distribution[ExplanationConfidence.MEDIUM],
            low_confidence_count=self._confidence_distribution[ExplanationConfidence.LOW],
            insufficient_count=self._confidence_distribution[ExplanationConfidence.INSUFFICIENT],
            average_overall_confidence=Decimal("0"),  # Would need to track
        )

    def _retrieve_concepts(self, hierarchy: DecisionHierarchyResult) -> Tuple[ConceptCitation, ...]:
        """Deterministically retrieve relevant concepts from the concept engine.

        Matching is keyword-based: concept names and definitions are checked
        against hierarchy state names, evidence text, and aggression types.
        """
        all_concepts = self._apprentice.all_concepts()
        if not all_concepts:
            return ()

        keywords = self._extract_hierarchy_keywords(hierarchy)
        scored = []
        for concept in all_concepts:
            score = self._score_concept_relevance(concept, keywords, hierarchy)
            if score >= self._configuration.min_concept_relevance_score:
                snippet = concept.definition.teacher_explanation[:200]
                if len(concept.definition.teacher_explanation) > 200:
                    snippet += "..."
                why = self._explain_why_relevant(concept, hierarchy)
                scored.append((score, ConceptCitation(
                    concept_name=concept.definition.name,
                    concept_id=concept.concept_id,
                    relevance_score=score,
                    teacher_explanation_snippet=snippet,
                    why_relevant=why,
                    confidence_level=concept.confidence.level,
                )))

        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(c for _, c in scored[:self._configuration.max_concept_citations])

    def _retrieve_similar_lessons(self, hierarchy: DecisionHierarchyResult) -> Tuple[SimilarLesson, ...]:
        """Deterministically retrieve similar past lessons from the experience engine.

        Matching compares hierarchy state names against experience reasoning
        summaries and teacher explanations.
        """
        all_experiences = self._apprentice.all_experiences()
        if not all_experiences:
            return ()

        keywords = self._extract_hierarchy_keywords(hierarchy)
        scored = []
        for exp in all_experiences:
            score = self._score_experience_similarity(exp, keywords, hierarchy)
            if score > Decimal("0"):
                scored.append((score, SimilarLesson(
                    lesson_reference=exp.session_reference or "unknown",
                    similarity_score=score,
                    teacher_explanation=exp.teacher_explanation[:300],
                    concept_references=exp.concept_references,
                    experience_id=exp.experience_id,
                    timestamp=exp.observation.timestamp,
                )))

        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(l for _, l in scored[:self._configuration.max_similar_lessons])

    def _retrieve_related_concepts(self, citations: Tuple[ConceptCitation, ...]) -> Tuple[str, ...]:
        """Retrieve concepts related to the cited concepts from the knowledge graph."""
        graph = self._apprentice.get_knowledge_graph()
        related = set()
        for citation in citations:
            name = citation.concept_name
            for rel_type in (RelationshipType.RELATED_TO, RelationshipType.REQUIRES, RelationshipType.ENABLES):
                for rel in graph.query(source_concept=name, relationship_type=rel_type):
                    related.add(rel.target_concept)
                for rel in graph.query(target_concept=name, relationship_type=rel_type):
                    related.add(rel.source_concept)
        # Exclude concepts already cited
        cited_names = {c.concept_name for c in citations}
        related = {r for r in related if r not in cited_names}
        return tuple(sorted(related))[:self._configuration.max_related_concepts]

    def _identify_missing_evidence(self, hierarchy: DecisionHierarchyResult) -> Tuple[MissingEvidence, ...]:
        """Identify missing or weak evidence in the hierarchy."""
        missing = []
        if hierarchy.market_state is None:
            missing.append(MissingEvidence(
                field_name="market_state",
                why_missing="No trend engine or session data available",
                impact_on_confidence="Cannot determine if market is trending, balancing, or auctioning",
                suggestion="Ensure trend engine and session intelligence are running",
            ))
        elif hierarchy.market_state.confidence < Decimal("0.50"):
            missing.append(MissingEvidence(
                field_name="market_state_confidence",
                why_missing="Market state confidence is low",
                impact_on_confidence="Uncertainty about whether the market is truly trending or balancing",
                suggestion="Wait for clearer trend structure or session context",
            ))

        if hierarchy.location is None:
            missing.append(MissingEvidence(
                field_name="location",
                why_missing="No market profile or structure data available",
                impact_on_confidence="Cannot determine if price is at POC, VAH, VAL, or structure level",
                suggestion="Ensure market profile and market structure bundles are running",
            ))

        if hierarchy.aggression is None:
            missing.append(MissingEvidence(
                field_name="aggression",
                why_missing="No imbalance, delta, or absorption data available",
                impact_on_confidence="Cannot determine who is aggressive or if absorption is present",
                suggestion="Ensure footprint imbalance and delta analysis are running",
            ))

        if hierarchy.risk is None:
            missing.append(MissingEvidence(
                field_name="risk",
                why_missing="Cannot assess risk without market state, location, and aggression",
                impact_on_confidence="No risk framework available for this observation",
                suggestion="Complete lower hierarchy levels first",
            ))

        if hierarchy.management is None:
            missing.append(MissingEvidence(
                field_name="management",
                why_missing="Cannot determine management posture without risk assessment",
                impact_on_confidence="No watch conditions or scale conditions defined",
                suggestion="Complete risk assessment first",
            ))

        if hierarchy.reflection is None:
            missing.append(MissingEvidence(
                field_name="reflection",
                why_missing="No prior experience outcome available for this setup",
                impact_on_confidence="Cannot validate if this reasoning pattern has worked before",
                suggestion="Allow experience to complete with outcome and reflection",
            ))

        return tuple(missing[:self._configuration.max_missing_evidence_items])

    def _build_narrative(
        self,
        hierarchy: DecisionHierarchyResult,
        concepts: Tuple[ConceptCitation, ...],
        lessons: Tuple[SimilarLesson, ...],
        missing: Tuple[MissingEvidence, ...],
    ) -> MarketNarrative:
        """Build the structured narrative from hierarchy and retrieved knowledge."""
        ms = hierarchy.market_state
        loc = hierarchy.location
        agg = hierarchy.aggression
        risk = hierarchy.risk
        mgmt = hierarchy.management
        refl = hierarchy.reflection

        # Market State Narrative
        if ms:
            ms_text = f"The market is {ms.state.name.replace('_', ' ').lower()}."
            if ms.primary_driver:
                ms_text += f" This is driven by {ms.primary_driver}."
            if ms.evidence:
                ms_text += f" Evidence: {', '.join(ms.evidence[:3])}."
        else:
            ms_text = "The market state is unclear from the available data."

        # Location Narrative
        if loc:
            loc_text = f"Price is located at {loc.location.name.replace('_', ' ').lower()}."
            if loc.evidence:
                loc_text += f" Evidence: {', '.join(loc.evidence[:3])}."
        else:
            loc_text = "The location within market structure is not determined."

        # Aggression Narrative
        if agg:
            agg_text = f"{agg.aggression.name.replace('_', ' ').title()} is present."
            if agg.primary_zone:
                agg_text += f" {agg.primary_zone}."
            if agg.supporting_imbalances:
                agg_text += f" Supporting imbalances: {', '.join(agg.supporting_imbalances[:2])}."
            if agg.supporting_absorption:
                agg_text += f" Absorption events: {', '.join(agg.supporting_absorption[:2])}."
        else:
            agg_text = "Aggression data is not available."

        # Risk Narrative
        if risk:
            risk_text = f"Risk is assessed as {risk.risk_level.name.lower()}."
            if risk.invalidation_conditions:
                risk_text += f" Invalidation if: {', '.join(risk.invalidation_conditions[:2])}."
        else:
            risk_text = "Risk cannot be assessed with incomplete data."

        # Management Narrative
        if mgmt:
            mgmt_text = f"Management posture: {mgmt.management.name.replace('_', ' ').lower()}."
            if mgmt.watch_conditions:
                mgmt_text += f" Watch for: {', '.join(mgmt.watch_conditions[:3])}."
        else:
            mgmt_text = "No management posture can be determined."

        # Reflection Narrative
        if refl:
            refl_text = f"Prior reflection: {refl.reflection.name.replace('_', ' ').lower()}."
            if refl.lesson_learned:
                refl_text += f" Lesson: {refl.lesson_learned}"
        else:
            refl_text = "No reflection data available for this observation."

        # Concept Summary
        if concepts:
            concept_names = ", ".join(c.concept_name for c in concepts[:3])
            concept_text = f"Relevant concepts: {concept_names}."
        else:
            concept_text = "No relevant concepts have been learned yet."

        # Overall Summary
        summary_parts = [ms_text]
        if loc:
            summary_parts.append(loc_text)
        if agg:
            summary_parts.append(agg_text)
        if risk:
            summary_parts.append(risk_text)
        summary = " ".join(summary_parts)

        return MarketNarrative(
            summary=summary,
            market_state_narrative=ms_text,
            location_narrative=loc_text,
            aggression_narrative=agg_text,
            risk_narrative=risk_text,
            management_narrative=mgmt_text,
            reflection_narrative=refl_text,
            concept_summary=concept_text,
        )

    def _format_explanation(self, result: ExplanationResult) -> str:
        """Format an ExplanationResult as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("GENESIS LIVE COACH — What is happening?")
        lines.append("=" * 60)
        lines.append("")
        lines.append(result.narrative.summary)
        lines.append("")
        lines.append("— Market State —")
        lines.append(result.narrative.market_state_narrative)
        lines.append("")
        lines.append("— Location —")
        lines.append(result.narrative.location_narrative)
        lines.append("")
        lines.append("— Aggression —")
        lines.append(result.narrative.aggression_narrative)
        lines.append("")
        lines.append("— Risk —")
        lines.append(result.narrative.risk_narrative)
        lines.append("")
        lines.append("— Management —")
        lines.append(result.narrative.management_narrative)
        lines.append("")
        lines.append("— Concepts —")
        lines.append(result.narrative.concept_summary)
        if result.concept_citations:
            for c in result.concept_citations:
                lines.append(f"  • {c.concept_name} ({c.confidence_level.name}, relevance {c.relevance_score}): {c.why_relevant}")
        lines.append("")
        if result.similar_lessons:
            lines.append("— Similar Lessons —")
            for l in result.similar_lessons:
                lines.append(f"  • {l.lesson_reference} (similarity {l.similarity_score}): {l.teacher_explanation[:100]}...")
            lines.append("")
        if result.missing_evidence:
            lines.append("— Missing Evidence —")
            for m in result.missing_evidence:
                lines.append(f"  • {m.field_name}: {m.why_missing} ({m.suggestion})")
            lines.append("")
        lines.append(f"Overall Confidence: {result.overall_confidence} ({result.explanation_confidence.name})")
        lines.append(f"Explanation ID: {result.explanation_id}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _compute_overall_confidence(
        self,
        hierarchy: DecisionHierarchyResult,
        concepts: Tuple[ConceptCitation, ...],
        missing: Tuple[MissingEvidence, ...],
    ) -> Decimal:
        """Compute overall explanation confidence."""
        base = hierarchy.overall_confidence
        if concepts:
            concept_boost = min(
                Decimal("0.10"),
                Decimal(str(len(concepts))) * Decimal("0.02"),
            )
            base = min(base + concept_boost, Decimal("1"))
        if missing:
            penalty = min(
                Decimal("0.20"),
                Decimal(str(len(missing))) * Decimal("0.03"),
            )
            base = max(base - penalty, Decimal("0"))
        return base

    def _classify_confidence(
        self,
        overall: Decimal,
        hierarchy: DecisionHierarchyResult,
        missing: Tuple[MissingEvidence, ...],
    ) -> ExplanationConfidence:
        """Classify the explanation confidence level."""
        if not self._configuration.include_incomplete_levels and not hierarchy.hierarchy_complete:
            return ExplanationConfidence.INSUFFICIENT
        if missing and len(missing) >= 4:
            return ExplanationConfidence.INSUFFICIENT
        if overall >= self._configuration.min_hierarchy_confidence_for_high and len(missing) <= 1:
            return ExplanationConfidence.HIGH
        if overall >= self._configuration.min_hierarchy_confidence_for_medium:
            return ExplanationConfidence.MEDIUM
        if overall > Decimal("0"):
            return ExplanationConfidence.LOW
        return ExplanationConfidence.INSUFFICIENT

    def _extract_hierarchy_keywords(self, hierarchy: DecisionHierarchyResult) -> Tuple[str, ...]:
        """Extract searchable keywords from the hierarchy result."""
        keywords = []
        if hierarchy.market_state:
            keywords.append(hierarchy.market_state.state.name.lower())
            keywords.extend(e.lower() for e in hierarchy.market_state.evidence)
        if hierarchy.location:
            keywords.append(hierarchy.location.location.name.lower())
        if hierarchy.aggression:
            keywords.append(hierarchy.aggression.aggression.name.lower())
            keywords.extend(i.lower() for i in hierarchy.aggression.supporting_imbalances)
        if hierarchy.risk:
            keywords.append(hierarchy.risk.risk_level.name.lower())
        return tuple(set(keywords))

    def _score_concept_relevance(
        self,
        concept: Concept,
        keywords: Tuple[str, ...],
        hierarchy: DecisionHierarchyResult,
    ) -> Decimal:
        """Deterministic relevance scoring for a concept against hierarchy keywords."""
        if not keywords:
            return Decimal("0")
        text = (
            concept.definition.name.lower()
            + " "
            + concept.definition.definition.lower()
            + " "
            + concept.definition.teacher_explanation.lower()
        )
        matches = sum(1 for kw in keywords if kw in text)
        if matches == 0:
            return Decimal("0")
        score = Decimal(str(matches)) / Decimal(str(len(keywords)))
        # Boost for higher concept confidence
        if concept.confidence.level == ConceptConfidenceLevel.EXPERT:
            score = min(score + Decimal("0.10"), Decimal("1"))
        elif concept.confidence.level == ConceptConfidenceLevel.PROFICIENT:
            score = min(score + Decimal("0.05"), Decimal("1"))
        return min(score, Decimal("1"))

    def _score_experience_similarity(
        self,
        experience: Experience,
        keywords: Tuple[str, ...],
        hierarchy: DecisionHierarchyResult,
    ) -> Decimal:
        """Deterministic similarity scoring for an experience against current hierarchy."""
        if not keywords:
            return Decimal("0")
        text = experience.teacher_explanation.lower()
        if experience.reasoning:
            text += " " + experience.reasoning.market_state_summary.lower()
            text += " " + experience.reasoning.aggression_summary.lower()
        matches = sum(1 for kw in keywords if kw in text)
        if matches == 0:
            return Decimal("0")
        score = Decimal(str(matches)) / Decimal(str(len(keywords)))
        # Boost for validated outcomes
        if experience.outcome and experience.outcome.outcome_type == ExperienceOutcomeType.VALIDATED:
            score = min(score + Decimal("0.10"), Decimal("1"))
        return min(score, Decimal("1"))

    def _explain_why_relevant(self, concept: Concept, hierarchy: DecisionHierarchyResult) -> str:
        """Generate a deterministic "why relevant" sentence."""
        reasons = []
        name = concept.definition.name.lower()
        if hierarchy.market_state and name in hierarchy.market_state.state.name.lower():
            reasons.append("matches market state")
        if hierarchy.aggression and name in hierarchy.aggression.aggression.name.lower():
            reasons.append("matches aggression pattern")
        if hierarchy.location and name in hierarchy.location.location.name.lower():
            reasons.append("matches location context")
        if not reasons:
            reasons.append("keyword match in definition or explanation")
        return "; ".join(reasons)

    def _insufficient_explanation(
        self,
        hierarchy: DecisionHierarchyResult,
        reason: str,
    ) -> ExplanationResult:
        """Return an insufficient explanation when data is missing."""
        narrative = MarketNarrative(
            summary=f"Insufficient data to explain this chart. Reason: {reason}",
            market_state_narrative="No market state data available.",
            location_narrative="No location data available.",
            aggression_narrative="No aggression data available.",
            risk_narrative="No risk data available.",
            management_narrative="No management data available.",
            reflection_narrative="No reflection data available.",
            concept_summary="No concepts can be cited without hierarchy data.",
        )
        return ExplanationResult(
            narrative=narrative,
            hierarchy_result=hierarchy,
            concept_citations=(),
            similar_lessons=(),
            related_concepts=(),
            missing_evidence=(
                MissingEvidence(
                    field_name="all",
                    why_missing=f"Hierarchy analysis failed: {reason}",
                    impact_on_confidence="Explanation cannot be produced",
                    suggestion="Run full Genesis analysis pipeline first",
                ),
            ),
            overall_confidence=Decimal("0"),
            explanation_confidence=ExplanationConfidence.INSUFFICIENT,
            explanation_id=f"insufficient_{reason}",
        )

    def _derive_explanation_id(self, detection_graph) -> str:
        """Derive a deterministic explanation id from the graph."""
        from hashlib import sha256
        ref = getattr(detection_graph, "graph_id", str(id(detection_graph)))
        seed = f"explanation:{ref}:{self._explanation_count}".encode("utf-8")
        return sha256(seed).hexdigest()[:16]
