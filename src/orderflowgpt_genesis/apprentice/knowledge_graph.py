"""Knowledge Graph for the Genesis Apprentice Layer.

Genesis stores relationships between concepts. Example:
Absorption requires Aggression requires Market State.

Knowledge is represented as an evolving graph of immutable relationships.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Tuple


class RelationshipType(Enum):
    """Deterministic relationship types between concepts."""
    REQUIRES = auto()
    RELATED_TO = auto()
    SEEN_IN = auto()
    CONTRADICTS = auto()
    ENABLES = auto()
    PART_OF = auto()
    LEADS_TO = auto()
    EXAMPLE_OF = auto()
    PRECEDES = auto()
    FOLLOWS = auto()
    SIMILAR_TO = auto()
    OPPOSITE_OF = auto()


@dataclass(frozen=True)
class KnowledgeRelationship:
    """Immutable relationship between two concepts.

    Example: ("Absorption", REQUIRES, "Aggression")
    """
    source_concept: str
    relationship_type: RelationshipType
    target_concept: str
    evidence: Tuple[str, ...] = ()
    confidence: Decimal = Decimal("1")
    relationship_id: str = ""

    def __post_init__(self):
        if not self.source_concept:
            raise ValueError("source_concept is required")
        if not self.target_concept:
            raise ValueError("target_concept is required")
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")
        if not self.relationship_id:
            object.__setattr__(self, "relationship_id", self._derive_id())

    def _derive_id(self) -> str:
        from hashlib import sha256
        seed = f"{self.source_concept}:{self.relationship_type.name}:{self.target_concept}".encode("utf-8")
        return sha256(seed).hexdigest()[:16]

    def reverse(self) -> "KnowledgeRelationship":
        """Return the reverse relationship with inverted type where applicable."""
        reverse_map = {
            RelationshipType.REQUIRES: RelationshipType.ENABLES,
            RelationshipType.ENABLES: RelationshipType.REQUIRES,
            RelationshipType.PRECEDES: RelationshipType.FOLLOWS,
            RelationshipType.FOLLOWS: RelationshipType.PRECEDES,
            RelationshipType.CONTRADICTS: RelationshipType.CONTRADICTS,
            RelationshipType.SIMILAR_TO: RelationshipType.SIMILAR_TO,
            RelationshipType.OPPOSITE_OF: RelationshipType.OPPOSITE_OF,
        }
        new_type = reverse_map.get(self.relationship_type, RelationshipType.RELATED_TO)
        return KnowledgeRelationship(
            source_concept=self.target_concept,
            relationship_type=new_type,
            target_concept=self.source_concept,
            evidence=self.evidence,
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class KnowledgeGraphConfiguration:
    """Configuration for the Knowledge Graph."""
    min_confidence_for_query: Decimal = Decimal("0.30")
    max_path_length: int = 5
    allow_cycles_in_path: bool = False

    def __post_init__(self):
        if not (Decimal("0") <= self.min_confidence_for_query <= Decimal("1")):
            raise ValueError("min_confidence_for_query must be in [0, 1]")
        if self.max_path_length < 1:
            raise ValueError("max_path_length must be at least 1")


@dataclass(frozen=True)
class KnowledgeGraphStatistics:
    """Statistics for the knowledge graph."""
    total_concepts: int = 0
    total_relationships: int = 0
    relationship_type_distribution: Tuple[Tuple[RelationshipType, int], ...] = ()
    average_confidence: Decimal = Decimal("0")
    connected_components: int = 0
    longest_path_length: int = 0


@dataclass(frozen=True)
class KnowledgeGraphQuery:
    """Query parameters for the knowledge graph."""
    source_concept: Optional[str] = None
    target_concept: Optional[str] = None
    relationship_type: Optional[RelationshipType] = None
    min_confidence: Optional[Decimal] = None


@dataclass(frozen=True)
class ConceptPath:
    """A path of relationships from one concept to another."""
    start_concept: str
    end_concept: str
    relationships: Tuple[KnowledgeRelationship, ...] = ()
    path_confidence: Decimal = Decimal("0")
    path_length: int = 0

    def __post_init__(self):
        if not (Decimal("0") <= self.path_confidence <= Decimal("1")):
            raise ValueError("path_confidence must be in [0, 1]")
        object.__setattr__(self, "path_length", len(self.relationships))


@dataclass(frozen=True)
class KnowledgeGraph:
    """Immutable knowledge graph storing concepts and their relationships.

    The graph is queryable and supports path finding between concepts.
    It performs no AI, no ML, no prediction, and no trade recommendation.
    """
    concepts: Tuple[str, ...] = ()
    relationships: Tuple[KnowledgeRelationship, ...] = ()
    configuration: KnowledgeGraphConfiguration = field(default_factory=KnowledgeGraphConfiguration)
    statistics: KnowledgeGraphStatistics = field(default_factory=KnowledgeGraphStatistics)

    def __post_init__(self):
        concept_set = set(self.concepts)
        for rel in self.relationships:
            if rel.source_concept not in concept_set:
                raise ValueError(f"source concept '{rel.source_concept}' not in graph")
            if rel.target_concept not in concept_set:
                raise ValueError(f"target concept '{rel.target_concept}' not in graph")

    def with_concept(self, concept: str) -> "KnowledgeGraph":
        """Return a new graph with the concept added."""
        if concept in self.concepts:
            return self
        return KnowledgeGraph(
            concepts=self.concepts + (concept,),
            relationships=self.relationships,
            configuration=self.configuration,
            statistics=self._recompute_statistics(),
        )

    def with_relationship(self, relationship: KnowledgeRelationship) -> "KnowledgeGraph":
        """Return a new graph with the relationship added."""
        if relationship in self.relationships:
            return self
        new_concepts = list(self.concepts)
        if relationship.source_concept not in new_concepts:
            new_concepts.append(relationship.source_concept)
        if relationship.target_concept not in new_concepts:
            new_concepts.append(relationship.target_concept)
        return KnowledgeGraph(
            concepts=tuple(new_concepts),
            relationships=self.relationships + (relationship,),
            configuration=self.configuration,
            statistics=self._recompute_statistics(),
        )

    def query(self, query: KnowledgeGraphQuery) -> Tuple[KnowledgeRelationship, ...]:
        """Query relationships matching the given parameters."""
        results = []
        min_conf = query.min_confidence or self.configuration.min_confidence_for_query
        for rel in self.relationships:
            if query.source_concept is not None and rel.source_concept != query.source_concept:
                continue
            if query.target_concept is not None and rel.target_concept != query.target_concept:
                continue
            if query.relationship_type is not None and rel.relationship_type != query.relationship_type:
                continue
            if rel.confidence < min_conf:
                continue
            results.append(rel)
        return tuple(results)

    def what_requires(self, concept: str) -> Tuple[str, ...]:
        """Return concepts that this concept requires."""
        return tuple(
            rel.target_concept
            for rel in self.relationships
            if rel.source_concept == concept and rel.relationship_type == RelationshipType.REQUIRES
        )

    def what_enables(self, concept: str) -> Tuple[str, ...]:
        """Return concepts that this concept enables."""
        return tuple(
            rel.target_concept
            for rel in self.relationships
            if rel.source_concept == concept and rel.relationship_type == RelationshipType.ENABLES
        )

    def what_is_related(self, concept: str) -> Tuple[str, ...]:
        """Return concepts related to this concept."""
        related = set()
        for rel in self.relationships:
            if rel.source_concept == concept:
                related.add(rel.target_concept)
            if rel.target_concept == concept:
                related.add(rel.source_concept)
        return tuple(sorted(related))

    def what_contradicts(self, concept: str) -> Tuple[str, ...]:
        """Return concepts that contradict this concept."""
        return tuple(
            rel.target_concept if rel.source_concept == concept else rel.source_concept
            for rel in self.relationships
            if rel.relationship_type == RelationshipType.CONTRADICTS
            and (rel.source_concept == concept or rel.target_concept == concept)
        )

    def path_between(self, start: str, end: str) -> Tuple[ConceptPath, ...]:
        """Find all paths between two concepts up to max_path_length."""
        if start not in self.concepts or end not in self.concepts:
            return ()

        paths = []
        visited = set()

        def _dfs(current: str, target: str, current_path: Tuple[KnowledgeRelationship, ...], depth: int):
            if depth > self.configuration.max_path_length:
                return
            if current == target and current_path:
                conf = min(r.confidence for r in current_path) if current_path else Decimal("0")
                paths.append(ConceptPath(
                    start_concept=start,
                    end_concept=end,
                    relationships=current_path,
                    path_confidence=conf,
                ))
                return
            if not self.configuration.allow_cycles_in_path and current in visited:
                return
            visited.add(current)
            for rel in self.relationships:
                if rel.source_concept == current:
                    _dfs(rel.target_concept, target, current_path + (rel,), depth + 1)
            visited.discard(current)

        _dfs(start, end, (), 0)
        return tuple(paths)

    def shortest_path(self, start: str, end: str) -> Optional[ConceptPath]:
        """Return the shortest path between two concepts."""
        all_paths = self.path_between(start, end)
        if not all_paths:
            return None
        return min(all_paths, key=lambda p: p.path_length)

    def strongest_path(self, start: str, end: str) -> Optional[ConceptPath]:
        """Return the highest-confidence path between two concepts."""
        all_paths = self.path_between(start, end)
        if not all_paths:
            return None
        return max(all_paths, key=lambda p: p.path_confidence)

    def _recompute_statistics(self) -> KnowledgeGraphStatistics:
        """Recompute graph statistics."""
        from collections import Counter
        if not self.relationships:
            return KnowledgeGraphStatistics(total_concepts=len(self.concepts))
        type_counts = Counter(rel.relationship_type for rel in self.relationships)
        avg_conf = sum(rel.confidence for rel in self.relationships) / Decimal(str(len(self.relationships)))
        return KnowledgeGraphStatistics(
            total_concepts=len(self.concepts),
            total_relationships=len(self.relationships),
            relationship_type_distribution=tuple(type_counts.items()),
            average_confidence=min(avg_conf, Decimal("1")),
            connected_components=0,
            longest_path_length=0,
        )


class KnowledgeGraphBuilder:
    """Deterministic builder for constructing KnowledgeGraph instances.

    The builder accumulates concepts and relationships, then produces an
    immutable KnowledgeGraph. No AI, no ML, no prediction.
    """

    def __init__(self, configuration: Optional[KnowledgeGraphConfiguration] = None):
        self._configuration = configuration or KnowledgeGraphConfiguration()
        self._concepts: set[str] = set()
        self._relationships: list[KnowledgeRelationship] = []

    def add_concept(self, concept: str) -> "KnowledgeGraphBuilder":
        """Add a concept to the builder."""
        self._concepts.add(concept)
        return self

    def add_relationship(self, relationship: KnowledgeRelationship) -> "KnowledgeGraphBuilder":
        """Add a relationship to the builder."""
        self._relationships.append(relationship)
        self._concepts.add(relationship.source_concept)
        self._concepts.add(relationship.target_concept)
        return self

    def add_relationships(self, relationships: Tuple[KnowledgeRelationship, ...]) -> "KnowledgeGraphBuilder":
        """Add multiple relationships."""
        for rel in relationships:
            self.add_relationship(rel)
        return self

    def build(self) -> KnowledgeGraph:
        """Build and return the immutable KnowledgeGraph."""
        return KnowledgeGraph(
            concepts=tuple(sorted(self._concepts)),
            relationships=tuple(self._relationships),
            configuration=self._configuration,
        )
