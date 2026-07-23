"""Deterministic Fabio Learning & Memory Engine for Bundle 13.

Bundle 13 transforms immutable Bundle 12 teaching datasets into a searchable Fabio
memory. It performs no neural-network processing, LLM inference, fine tuning,
prediction, trade-signal generation, probabilistic reasoning, or strategy generation.
Fabio teaching data remains the only source of memory content.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .knowledge import KnowledgeDataset, KnowledgeObservation
from .vision import DetectionGraph


class SimilarityMetric(Enum):
    WEIGHTED_FEATURE_DISTANCE = "WEIGHTED_FEATURE_DISTANCE"
    COSINE_SIMILARITY = "COSINE_SIMILARITY"
    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MANHATTAN_DISTANCE = "MANHATTAN_DISTANCE"
    HAMMING_SIMILARITY = "HAMMING_SIMILARITY"


@dataclass(frozen=True, slots=True)
class LearningIdentifier:
    learning_id: str
    source_dataset_id: str

    def __post_init__(self) -> None:
        if not self.learning_id.strip() or not self.source_dataset_id.strip():
            raise ValueError("learning and source dataset ids are required")


@dataclass(frozen=True, slots=True)
class LearningMetadata:
    identifier: LearningIdentifier
    created_at: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    source: str = "Fabio"
    bundle: str = "bundle-13"
    rules_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("learning metadata timestamp must be timezone-aware")
        if self.source != "Fabio":
            raise ValueError("Fabio is the only teacher")
        if not self.bundle.strip() or not self.rules_version.strip():
            raise ValueError("bundle and rules version are required")


@dataclass(frozen=True, slots=True)
class MemoryIdentifier:
    memory_id: str
    source_observation_id: str
    source_dataset_id: str

    def __post_init__(self) -> None:
        if not all(
            v.strip()
            for v in (
                self.memory_id,
                self.source_observation_id,
                self.source_dataset_id,
            )
        ):
            raise ValueError("memory identifiers are required")


@dataclass(frozen=True, slots=True)
class MemoryFeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    categorical: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError("feature names and values must have equal length")
        if tuple(sorted(self.names)) != self.names or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError("feature names must be unique and sorted")
        if any(not n.strip() for n in self.names + self.categorical):
            raise ValueError("feature names and categorical features cannot be blank")
        if any(not math.isfinite(v) for v in self.values):
            raise ValueError("feature values must be finite")
        if tuple(sorted(set(self.categorical))) != self.categorical:
            raise ValueError("categorical features must be unique and sorted")

    def value(self, name: str) -> float:
        try:
            return self.values[self.names.index(name)]
        except ValueError:
            return 0.0


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    identifier: MemoryIdentifier
    feature_vector: MemoryFeatureVector
    video_id: str
    lesson_id: str
    timestamp_ms: int
    transcript: str
    knowledge_observation_id: str
    knowledge_category: str
    knowledge_topics: tuple[str, ...]
    transcript_references: tuple[str, ...]

    def __post_init__(self) -> None:
        required = (
            self.video_id,
            self.lesson_id,
            self.transcript,
            self.knowledge_observation_id,
            self.knowledge_category,
        )
        if not all(v.strip() for v in required):
            raise ValueError("memory entry fields are required")
        if self.timestamp_ms < 0:
            raise ValueError("memory timestamp cannot be negative")
        for label, values in (
            ("topics", self.knowledge_topics),
            ("transcripts", self.transcript_references),
        ):
            if tuple(sorted(set(values))) != values or any(
                not v.strip() for v in values
            ):
                raise ValueError(
                    f"memory {label} must be unique, sorted, and non-blank"
                )


@dataclass(frozen=True, slots=True)
class MemoryStatistics:
    entry_count: int
    video_count: int
    lesson_count: int
    feature_count: int
    topic_count: int
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryIndex:
    entries: tuple[MemoryEntry, ...]
    by_memory_id: Mapping[str, MemoryEntry]
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "by_memory_id", MappingProxyType(dict(self.by_memory_id))
        )
        ids = [e.identifier.memory_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("memory index entries must be unique")
        if tuple(sorted(ids)) != tuple(ids):
            raise ValueError("memory index entries must be sorted")


@dataclass(frozen=True, slots=True)
class MemoryDataset:
    dataset_id: str
    metadata: LearningMetadata
    entries: tuple[MemoryEntry, ...]
    index: MemoryIndex
    statistics: MemoryStatistics

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("memory dataset id is required")
        MemoryValidator().validate(self)


@dataclass(frozen=True, slots=True)
class LearningConfiguration:
    memory_dataset_id: str
    lesson_id: str = "fabio-lesson"

    def __post_init__(self) -> None:
        if not self.memory_dataset_id.strip() or not self.lesson_id.strip():
            raise ValueError("memory dataset and lesson ids are required")


@dataclass(frozen=True, slots=True)
class SimilarityConfiguration:
    metric: SimilarityMetric = SimilarityMetric.WEIGHTED_FEATURE_DISTANCE
    top_n: int = 5
    weights: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if any(v < 0 or not math.isfinite(v) for v in self.weights.values()):
            raise ValueError("similarity weights must be finite non-negative values")
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


@dataclass(frozen=True, slots=True)
class SimilarityScore:
    metric: SimilarityMetric
    score: float
    distance: float


@dataclass(frozen=True, slots=True)
class RetrievedExample:
    video: str
    lesson: str
    timestamp_ms: int
    transcript: str
    knowledge_observation: str
    similarity_score: SimilarityScore
    memory_identifier: MemoryIdentifier


@dataclass(frozen=True, slots=True)
class RetrievedExamples:
    examples: tuple[RetrievedExample, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query_frame_id: str
    retrieved_examples: RetrievedExamples
    statistics: MemoryStatistics
    metric: SimilarityMetric


@dataclass(frozen=True, slots=True)
class LearningResult:
    dataset: MemoryDataset
    statistics: MemoryStatistics


class MemoryBuilder:
    def __init__(self, configuration: LearningConfiguration) -> None:
        self.configuration = configuration

    def build(self, knowledge_dataset: KnowledgeDataset) -> LearningResult:
        entries = tuple(
            sorted(
                (
                    self._entry(knowledge_dataset, o)
                    for o in knowledge_dataset.timeline.observations
                ),
                key=lambda e: e.identifier.memory_id,
            )
        )
        index = MemoryIndexer().index(entries)
        stats = _statistics(entries, index.feature_names)
        dataset = MemoryDataset(
            self.configuration.memory_dataset_id,
            LearningMetadata(
                LearningIdentifier(
                    f"learning:{self.configuration.memory_dataset_id}",
                    knowledge_dataset.dataset_id,
                )
            ),
            entries,
            index,
            stats,
        )
        return LearningResult(dataset, stats)

    def _entry(
        self, dataset: KnowledgeDataset, obs: KnowledgeObservation
    ) -> MemoryEntry:
        return MemoryEntry(
            MemoryIdentifier(
                f"memory:{dataset.dataset_id}:{obs.observation_id}",
                obs.observation_id,
                dataset.dataset_id,
            ),
            feature_vector_from_observation(obs),
            obs.context.video_id,
            self.configuration.lesson_id,
            obs.context.timestamp_ms,
            obs.statement.text,
            obs.observation_id,
            obs.statement.category.value,
            tuple(sorted(obs.statement.topic_ids)),
            (obs.statement.sentence_id,),
        )


class MemoryIndexer:
    def index(self, entries: Iterable[MemoryEntry]) -> MemoryIndex:
        ordered = tuple(sorted(entries, key=lambda e: e.identifier.memory_id))
        names = tuple(sorted({n for e in ordered for n in e.feature_vector.names}))
        cats = tuple(sorted({c for e in ordered for c in e.feature_vector.categorical}))
        return MemoryIndex(
            ordered, {e.identifier.memory_id: e for e in ordered}, names, cats
        )


class MemoryDatabase:
    def __init__(self, dataset: MemoryDataset) -> None:
        self.dataset = dataset

    def search(
        self,
        graph: DetectionGraph,
        configuration: SimilarityConfiguration | None = None,
    ) -> RetrievalResult:
        return MemorySearcher(configuration or SimilarityConfiguration()).search(
            self.dataset, graph
        )


class MemorySearcher:
    def __init__(self, configuration: SimilarityConfiguration) -> None:
        self.configuration = configuration

    def search(self, dataset: MemoryDataset, graph: DetectionGraph) -> RetrievalResult:
        query = feature_vector_from_graph(graph)
        ranked = sorted(
            (
                (score_vectors(query, e.feature_vector, self.configuration), e)
                for e in dataset.entries
            ),
            key=lambda x: (
                -x[0].score,
                x[0].distance,
                x[1].timestamp_ms,
                x[1].identifier.memory_id,
            ),
        )[: self.configuration.top_n]
        examples = tuple(
            RetrievedExample(
                e.video_id,
                e.lesson_id,
                e.timestamp_ms,
                e.transcript,
                e.knowledge_observation_id,
                s,
                e.identifier,
            )
            for s, e in ranked
        )
        return RetrievalResult(
            graph.frame_id,
            RetrievedExamples(examples),
            dataset.statistics,
            self.configuration.metric,
        )


class MemoryValidator:
    def validate(self, dataset: MemoryDataset) -> None:
        ids = [e.identifier.memory_id for e in dataset.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate memory entries are not allowed")
        if tuple(sorted(ids)) != tuple(ids):
            raise ValueError("memory entries must be sorted")
        if set(ids) != set(dataset.index.by_memory_id):
            raise ValueError("memory index must reference every entry")
        if dataset.statistics.entry_count != len(dataset.entries):
            raise ValueError("memory statistics entry count mismatch")


class MemorySerializer:
    def to_dict(self, value: Any) -> Any:
        if isinstance(value, MappingProxyType):
            return {k: self.to_dict(v) for k, v in value.items()}
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, datetime):
            return value.isoformat()
        if is_dataclass(value):
            return {
                f.name: self.to_dict(getattr(value, f.name))
                for f in fields(value)
                if f.name != "by_memory_id"
            }
        if isinstance(value, tuple | list):
            return [self.to_dict(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self.to_dict(v) for k, v in value.items()}
        return value

    def dumps(self, dataset: MemoryDataset) -> str:
        return json.dumps(self.to_dict(dataset), sort_keys=True, separators=(",", ":"))

    def dump(self, dataset: MemoryDataset, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.dumps(dataset), encoding="utf-8")
        return target


class MemoryLoader:
    def loads(self, payload: str) -> MemoryDataset:
        data = json.loads(payload)
        entries = tuple(
            MemoryEntry(
                MemoryIdentifier(**e["identifier"]),
                MemoryFeatureVector(
                    tuple(e["feature_vector"]["names"]),
                    tuple(float(v) for v in e["feature_vector"]["values"]),
                    tuple(e["feature_vector"].get("categorical", ())),
                ),
                e["video_id"],
                e["lesson_id"],
                int(e["timestamp_ms"]),
                e["transcript"],
                e["knowledge_observation_id"],
                e["knowledge_category"],
                tuple(e["knowledge_topics"]),
                tuple(e["transcript_references"]),
            )
            for e in data["entries"]
        )
        metadata = LearningMetadata(
            LearningIdentifier(**data["metadata"]["identifier"]),
            datetime.fromisoformat(data["metadata"]["created_at"]),
            data["metadata"]["source"],
            data["metadata"]["bundle"],
            data["metadata"]["rules_version"],
        )
        index = MemoryIndexer().index(entries)
        return MemoryDataset(
            data["dataset_id"],
            metadata,
            entries,
            index,
            _statistics(entries, index.feature_names),
        )

    def load(self, path: str | Path) -> MemoryDataset:
        return self.loads(Path(path).read_text(encoding="utf-8"))


def feature_vector_from_observation(
    observation: KnowledgeObservation,
) -> MemoryFeatureVector:
    category = observation.statement.category.value.lower()
    cats = tuple(
        sorted(
            (f"category:{category}",)
            + tuple(f"topic:{t}" for t in observation.statement.topic_ids)
            + (f"transcript:{observation.statement.sentence_id}",)
        )
    )
    names = _FEATURE_NAMES
    values = tuple(1.0 if name == f"topic_{category}" else 0.0 for name in names)
    return MemoryFeatureVector(names, values, cats)


def feature_vector_from_graph(graph: DetectionGraph) -> MemoryFeatureVector:
    values = {
        "topic_poc": 1.0 if graph.point_of_control is not None else 0.0,
        "topic_value_area": 1.0 if graph.value_area is not None else 0.0,
        "topic_volume": float(
            (len(graph.high_volume_nodes.nodes) if graph.high_volume_nodes else 0)
            + (len(graph.low_volume_nodes.nodes) if graph.low_volume_nodes else 0)
            + (len(graph.volume_clusters.clusters) if graph.volume_clusters else 0)
        ),
        "topic_delta": (
            1.0
            if graph.footprint_delta is not None or graph.cumulative_delta is not None
            else 0.0
        ),
        "topic_stacked_imbalance": float(
            len(graph.stacked_imbalances.stacks()) if graph.stacked_imbalances else 0
        ),
        "topic_absorption": float(
            len(graph.absorption.absorptions()) if graph.absorption else 0
        ),
        "topic_trend": float(
            len(graph.trend_states()) if hasattr(graph, "trend_states") else 0
        ),
        "topic_market_structure": 1.0 if graph.market_structure is not None else 0.0,
        "topic_auction_theory": float(
            graph.unfinished_auction_statistics().total_auctions
            if hasattr(graph, "unfinished_auction_statistics")
            else 0
        ),
        "topic_session": 1.0 if graph.trading_session is not None else 0.0,
        "topic_confluence": 1.0 if graph.confluence is not None else 0.0,
        "topic_general_observation": 0.0,
        "topic_unknown": 0.0,
    }
    cats = tuple(
        sorted(
            graph.knowledge_observations
            + graph.knowledge_references
            + graph.transcript_references
        )
    )
    return MemoryFeatureVector(
        _FEATURE_NAMES, tuple(values[n] for n in _FEATURE_NAMES), cats
    )


_FEATURE_NAMES = tuple(
    sorted(
        f"topic_{c.value.lower()}"
        for c in __import__(
            "orderflowgpt_genesis.knowledge", fromlist=["KnowledgeCategory"]
        ).KnowledgeCategory
    )
)


def score_vectors(
    a: MemoryFeatureVector,
    b: MemoryFeatureVector,
    configuration: SimilarityConfiguration,
) -> SimilarityScore:
    names = tuple(sorted(set(a.names) | set(b.names)))
    diffs = [abs(a.value(n) - b.value(n)) for n in names]
    if configuration.metric is SimilarityMetric.COSINE_SIMILARITY:
        dot = sum(a.value(n) * b.value(n) for n in names)
        na = math.sqrt(sum(a.value(n) ** 2 for n in names))
        nb = math.sqrt(sum(b.value(n) ** 2 for n in names))
        score = dot / (na * nb) if na and nb else 0.0
        return SimilarityScore(configuration.metric, score, 1.0 - score)
    if configuration.metric is SimilarityMetric.EUCLIDEAN_DISTANCE:
        dist = math.sqrt(sum(d * d for d in diffs))
        return SimilarityScore(configuration.metric, 1.0 / (1.0 + dist), dist)
    if configuration.metric is SimilarityMetric.MANHATTAN_DISTANCE:
        dist = sum(diffs)
        return SimilarityScore(configuration.metric, 1.0 / (1.0 + dist), dist)
    if configuration.metric is SimilarityMetric.HAMMING_SIMILARITY:
        cats = set(a.categorical) | set(b.categorical)
        score = (
            (
                sum((c in a.categorical) == (c in b.categorical) for c in cats)
                / len(cats)
            )
            if cats
            else 1.0
        )
        return SimilarityScore(configuration.metric, score, 1.0 - score)
    dist = sum(
        configuration.weights.get(n, 1.0) * abs(a.value(n) - b.value(n)) for n in names
    )
    return SimilarityScore(configuration.metric, 1.0 / (1.0 + dist), dist)


def attach_retrieval_result(
    graph: DetectionGraph, result: RetrievalResult
) -> DetectionGraph:
    return replace(
        graph,
        memory_references=tuple(
            e.memory_identifier.memory_id for e in result.retrieved_examples.examples
        ),
        retrieval_references=tuple(
            e.knowledge_observation for e in result.retrieved_examples.examples
        ),
        retrieval_statistics=(
            len(result.retrieved_examples.examples),
            result.statistics.entry_count,
        ),
    )


def _statistics(
    entries: Sequence[MemoryEntry], feature_names: tuple[str, ...]
) -> MemoryStatistics:
    stamps = tuple(e.timestamp_ms for e in entries)
    return MemoryStatistics(
        len(entries),
        len({e.video_id for e in entries}),
        len({e.lesson_id for e in entries}),
        len(feature_names),
        len({t for e in entries for t in e.knowledge_topics}),
        min(stamps) if stamps else None,
        max(stamps) if stamps else None,
    )
