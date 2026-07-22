"""Deterministic Fabio knowledge extraction for Bundle 12.

This module turns synchronized Fabio transcript alignments and Genesis detection graphs
into immutable teaching observations. It performs no learning, prediction, strategy
generation, probabilities, neural-network processing, LLM reasoning, OCR, or OpenCV
work. Fabio transcript text is the only knowledge source; Genesis graph fields are
used only as deterministic references for already-computed market analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from .dataset import TrainingSample
from .transcript import TranscriptDataset
from .vision import DetectionGraph


class KnowledgeCategory(Enum):
    ABSORPTION = "ABSORPTION"
    STACKED_IMBALANCE = "STACKED_IMBALANCE"
    POC = "POC"
    VALUE_AREA = "VALUE_AREA"
    TREND = "TREND"
    MARKET_STRUCTURE = "MARKET_STRUCTURE"
    AUCTION_THEORY = "AUCTION_THEORY"
    DELTA = "DELTA"
    SESSION = "SESSION"
    VOLUME = "VOLUME"
    CONFLUENCE = "CONFLUENCE"
    GENERAL_OBSERVATION = "GENERAL_OBSERVATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class KnowledgeIdentifier:
    knowledge_id: str
    video_id: str
    transcript_id: str

    def __post_init__(self) -> None:
        if (
            not self.knowledge_id.strip()
            or not self.video_id.strip()
            or not self.transcript_id.strip()
        ):
            raise ValueError("knowledge, video, and transcript ids are required")


@dataclass(frozen=True, slots=True)
class KnowledgeMetadata:
    identifier: KnowledgeIdentifier
    created_at: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    source: str = "Fabio"
    bundle: str = "bundle-12"
    rules_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("knowledge metadata timestamp must be timezone-aware")
        if self.source != "Fabio":
            raise ValueError("Fabio is the only knowledge source")
        if not self.bundle.strip() or not self.rules_version.strip():
            raise ValueError("bundle and rules version are required")


@dataclass(frozen=True, slots=True)
class KnowledgeTopic:
    topic_id: str
    category: KnowledgeCategory
    label: str

    def __post_init__(self) -> None:
        if not self.topic_id.strip() or not self.label.strip():
            raise ValueError("topic id and label are required")


@dataclass(frozen=True, slots=True)
class KnowledgeStatement:
    statement_id: str
    sentence_id: str
    text: str
    category: KnowledgeCategory
    topic_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.statement_id.strip()
            or not self.sentence_id.strip()
            or not self.text.strip()
        ):
            raise ValueError("statement id, sentence id, and text are required")
        if len(set(self.topic_ids)) != len(self.topic_ids):
            raise ValueError("statement topic ids must be unique")


@dataclass(frozen=True, slots=True)
class KnowledgeContext:
    context_id: str
    video_id: str
    frame_id: str
    timestamp_ms: int
    transcript_alignment_id: str | None = None
    training_sample_id: str | None = None
    detection_graph_frame_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.context_id.strip()
            or not self.video_id.strip()
            or not self.frame_id.strip()
        ):
            raise ValueError("context id, video id, and frame id are required")
        if self.timestamp_ms < 0:
            raise ValueError("context timestamp cannot be negative")
        for value in (
            self.transcript_alignment_id,
            self.training_sample_id,
            self.detection_graph_frame_id,
        ):
            if value is not None and not value.strip():
                raise ValueError("context references cannot be blank")


@dataclass(frozen=True, slots=True)
class KnowledgeReference:
    reference_id: str
    statement_id: str
    sentence_id: str
    frame_id: str
    detection_graph_frame_id: str
    training_sample_id: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.reference_id,
            self.statement_id,
            self.sentence_id,
            self.frame_id,
            self.detection_graph_frame_id,
        )
        if not all(v.strip() for v in required):
            raise ValueError("knowledge references are required")
        if self.training_sample_id is not None and not self.training_sample_id.strip():
            raise ValueError("training sample reference cannot be blank")


@dataclass(frozen=True, slots=True)
class KnowledgeObservation:
    observation_id: str
    statement: KnowledgeStatement
    context: KnowledgeContext
    reference: KnowledgeReference

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation id is required")
        if self.statement.statement_id != self.reference.statement_id:
            raise ValueError("observation statement/reference mismatch")
        if self.statement.sentence_id != self.reference.sentence_id:
            raise ValueError("observation sentence/reference mismatch")
        if self.context.frame_id != self.reference.frame_id:
            raise ValueError("observation frame/reference mismatch")


@dataclass(frozen=True, slots=True)
class KnowledgeTimeline:
    observations: tuple[KnowledgeObservation, ...]

    def __post_init__(self) -> None:
        if (
            tuple(
                sorted(
                    self.observations,
                    key=lambda o: (
                        o.context.timestamp_ms,
                        o.context.frame_id,
                        o.observation_id,
                    ),
                )
            )
            != self.observations
        ):
            raise ValueError("knowledge observations must be timeline ordered")
        ids = [o.observation_id for o in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError("knowledge observations must be unique")


@dataclass(frozen=True, slots=True)
class KnowledgeStatistics:
    statement_count: int
    observation_count: int
    reference_count: int
    topic_count: int
    category_counts: Mapping[KnowledgeCategory, int]
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "category_counts", MappingProxyType(dict(self.category_counts))
        )


@dataclass(frozen=True, slots=True)
class KnowledgeConfiguration:
    dataset_id: str
    include_unknown: bool = True
    include_general_observations: bool = True

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("knowledge dataset id is required")


@dataclass(frozen=True, slots=True)
class KnowledgeDataset:
    dataset_id: str
    metadata: KnowledgeMetadata
    topics: tuple[KnowledgeTopic, ...]
    statements: tuple[KnowledgeStatement, ...]
    timeline: KnowledgeTimeline
    samples: tuple[TrainingSample, ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("knowledge dataset id is required")
        _unique("topic ids", [t.topic_id for t in self.topics])
        _unique("statement ids", [s.statement_id for s in self.statements])
        _unique("sample ids", [s.sample_id for s in self.samples])
        topic_ids = {t.topic_id for t in self.topics}
        if any(set(s.topic_ids) - topic_ids for s in self.statements):
            raise ValueError("statements must reference knowledge topics")
        statement_ids = {s.statement_id for s in self.statements}
        if any(
            o.statement.statement_id not in statement_ids
            for o in self.timeline.observations
        ):
            raise ValueError("observations must reference knowledge statements")

    def statistics(self) -> KnowledgeStatistics:
        counts: dict[KnowledgeCategory, int] = {c: 0 for c in KnowledgeCategory}
        for s in self.statements:
            counts[s.category] += 1
        stamps = tuple(o.context.timestamp_ms for o in self.timeline.observations)
        refs = {o.reference.reference_id for o in self.timeline.observations}
        return KnowledgeStatistics(
            len(self.statements),
            len(self.timeline.observations),
            len(refs),
            len(self.topics),
            counts,
            min(stamps) if stamps else None,
            max(stamps) if stamps else None,
        )


class KnowledgeDatasetBuilder:
    def build(
        self,
        dataset_id: str,
        metadata: KnowledgeMetadata,
        topics: Iterable[KnowledgeTopic],
        statements: Iterable[KnowledgeStatement],
        observations: Iterable[KnowledgeObservation],
        samples: Iterable[TrainingSample] = (),
    ) -> KnowledgeDataset:
        return KnowledgeDataset(
            dataset_id,
            metadata,
            tuple(sorted(topics, key=lambda t: t.topic_id)),
            tuple(sorted(statements, key=lambda s: s.statement_id)),
            KnowledgeTimeline(
                tuple(
                    sorted(
                        observations,
                        key=lambda o: (
                            o.context.timestamp_ms,
                            o.context.frame_id,
                            o.observation_id,
                        ),
                    )
                )
            ),
            tuple(sorted(samples, key=lambda s: s.sample_id)),
        )


@dataclass(frozen=True, slots=True)
class KnowledgeExtractionResult:
    dataset: KnowledgeDataset
    statistics: KnowledgeStatistics
    detection_graphs: tuple[DetectionGraph, ...]
    training_samples: tuple[TrainingSample, ...]


class KnowledgeExtractionEngine:
    def __init__(self, configuration: KnowledgeConfiguration) -> None:
        self.configuration = configuration

    def extract(
        self,
        transcript_dataset: TranscriptDataset,
        graphs: Iterable[DetectionGraph],
        samples: Iterable[TrainingSample] = (),
    ) -> KnowledgeExtractionResult:
        graph_by_frame = {g.frame_id: g for g in graphs}
        sample_by_frame = {s.metadata.identifier.frame_id: s for s in samples}
        sentence_by_id = {
            s.sentence_id: s for s in transcript_dataset.timeline.sentences
        }
        topics_by_category: dict[KnowledgeCategory, KnowledgeTopic] = {}
        statements: list[KnowledgeStatement] = []
        observations: list[KnowledgeObservation] = []
        enhanced_samples: dict[str, TrainingSample] = dict(sample_by_frame)
        enhanced_graphs: dict[str, DetectionGraph] = dict(graph_by_frame)

        for alignment in sorted(
            transcript_dataset.alignments, key=lambda a: (a.timestamp_ms, a.frame_id)
        ):
            graph = graph_by_frame.get(alignment.frame_id)
            if graph is None:
                continue
            sentence_ids = alignment.active_sentence_ids or (
                (alignment.nearest_sentence_id,)
                if alignment.nearest_sentence_id
                else ()
            )
            for sentence_id in sentence_ids:
                sentence = sentence_by_id.get(sentence_id)
                if sentence is None:
                    continue
                category = self.classify(sentence.text)
                if (
                    category is KnowledgeCategory.UNKNOWN
                    and not self.configuration.include_unknown
                ):
                    continue
                if (
                    category is KnowledgeCategory.GENERAL_OBSERVATION
                    and not self.configuration.include_general_observations
                ):
                    continue
                topic = topics_by_category.setdefault(
                    category,
                    KnowledgeTopic(
                        f"topic:{category.value.lower()}",
                        category,
                        category.value.replace("_", " ").title(),
                    ),
                )
                statement = KnowledgeStatement(
                    f"statement:{sentence.sentence_id}",
                    sentence.sentence_id,
                    sentence.text,
                    category,
                    (topic.topic_id,),
                )
                if statement.statement_id not in {s.statement_id for s in statements}:
                    statements.append(statement)
                sample = sample_by_frame.get(alignment.frame_id)
                context = KnowledgeContext(
                    f"context:{alignment.frame_id}:{sentence_id}",
                    alignment.video_id,
                    alignment.frame_id,
                    alignment.timestamp_ms,
                    f"alignment:{transcript_dataset.timeline.metadata.identifier.transcript_id}:{sentence_id}",
                    sample.sample_id if sample else None,
                    graph.frame_id,
                )
                ref = KnowledgeReference(
                    f"knowledge-ref:{alignment.frame_id}:{sentence_id}",
                    statement.statement_id,
                    sentence_id,
                    alignment.frame_id,
                    graph.frame_id,
                    sample.sample_id if sample else None,
                )
                obs = KnowledgeObservation(
                    f"knowledge-observation:{alignment.frame_id}:{sentence_id}",
                    statement,
                    context,
                    ref,
                )
                observations.append(obs)
                if sample is not None:
                    enhanced_samples[alignment.frame_id] = replace(
                        sample,
                        knowledge_observation_references=tuple(
                            sorted(
                                set(
                                    sample.knowledge_observation_references
                                    + (obs.observation_id,)
                                )
                            )
                        ),
                        knowledge_topics=tuple(
                            sorted(set(sample.knowledge_topics + (topic.topic_id,)))
                        ),
                        transcript_references=tuple(
                            sorted(set(sample.transcript_references + (sentence_id,)))
                        ),
                        frame_references=tuple(
                            sorted(set(sample.frame_references + (alignment.frame_id,)))
                        ),
                        timeline_references=tuple(
                            sorted(
                                set(sample.timeline_references + (context.context_id,))
                            )
                        ),
                    )
                enhanced_graphs[alignment.frame_id] = replace(
                    graph,
                    knowledge_observations=tuple(
                        sorted(
                            set(graph.knowledge_observations + (obs.observation_id,))
                        )
                    ),
                    knowledge_references=tuple(
                        sorted(set(graph.knowledge_references + (ref.reference_id,)))
                    ),
                    knowledge_statistics=(
                        len(set(graph.knowledge_observations + (obs.observation_id,))),
                    ),
                )

        ordered_statements = tuple(sorted(statements, key=lambda s: s.statement_id))
        timeline = KnowledgeTimeline(
            tuple(
                sorted(
                    observations,
                    key=lambda o: (
                        o.context.timestamp_ms,
                        o.context.frame_id,
                        o.observation_id,
                    ),
                )
            )
        )
        metadata = KnowledgeMetadata(
            KnowledgeIdentifier(
                f"knowledge:{self.configuration.dataset_id}",
                transcript_dataset.timeline.metadata.identifier.video_id,
                transcript_dataset.timeline.metadata.identifier.transcript_id,
            )
        )
        dataset = KnowledgeDataset(
            self.configuration.dataset_id,
            metadata,
            tuple(sorted(topics_by_category.values(), key=lambda t: t.topic_id)),
            ordered_statements,
            timeline,
            tuple(enhanced_samples[f] for f in sorted(enhanced_samples)),
        )
        return KnowledgeExtractionResult(
            dataset,
            dataset.statistics(),
            tuple(enhanced_graphs[f] for f in sorted(enhanced_graphs)),
            dataset.samples,
        )

    @staticmethod
    def classify(text: str) -> KnowledgeCategory:
        value = " ".join(text.lower().replace("-", " ").split())
        rules = (
            (
                KnowledgeCategory.STACKED_IMBALANCE,
                ("stacked imbalance", "stacked imbalances"),
            ),
            (KnowledgeCategory.ABSORPTION, ("absorption", "absorbed", "absorbing")),
            (KnowledgeCategory.VALUE_AREA, ("value area", "vah", "val")),
            (KnowledgeCategory.POC, ("poc", "point of control")),
            (
                KnowledgeCategory.MARKET_STRUCTURE,
                (
                    "market structure",
                    "break of structure",
                    "choch",
                    "higher high",
                    "lower low",
                ),
            ),
            (
                KnowledgeCategory.AUCTION_THEORY,
                ("auction", "excess", "unfinished", "poor high", "poor low"),
            ),
            (KnowledgeCategory.DELTA, ("delta", "cumulative delta")),
            (
                KnowledgeCategory.SESSION,
                ("session", "rth", "eth", "initial balance", "open"),
            ),
            (KnowledgeCategory.VOLUME, ("volume", "hvn", "lvn", "volume node")),
            (KnowledgeCategory.CONFLUENCE, ("confluence", "aligns", "alignment")),
            (KnowledgeCategory.TREND, ("trend", "pullback", "bullish", "bearish")),
            (
                KnowledgeCategory.GENERAL_OBSERVATION,
                ("look at", "notice", "watch", "see this", "observe"),
            ),
        )
        for category, phrases in rules:
            if any(p in value for p in phrases):
                return category
        return KnowledgeCategory.UNKNOWN


def _unique(label: str, values: list[str]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label} are not allowed")


KnowledgeExtractor = KnowledgeExtractionEngine
