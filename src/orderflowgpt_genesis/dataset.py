"""Deterministic dataset builder and export primitives for Bundle 9.

This module creates immutable, AI-ready training records from completed deterministic
analysis graphs. It intentionally implements no machine learning, prediction, or
probabilistic logic.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal

from .vision import DetectionGraph

ExportFormat = Literal["jsonl", "parquet", "sqlite"]


class AnnotationType(Enum):
    FABIO_DECISION = "FABIO_DECISION"
    REASONING_TEXT = "REASONING_TEXT"
    TRADE_DIRECTION = "TRADE_DIRECTION"
    CONFIDENCE = "CONFIDENCE"
    COMMENTS = "COMMENTS"
    TAGS = "TAGS"


class AnnotationStatus(Enum):
    EMPTY = "EMPTY"
    DRAFT = "DRAFT"
    REVIEWED = "REVIEWED"
    LOCKED = "LOCKED"


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    major: int
    minor: int
    patch: int
    label: str = "bundle-9"

    def __post_init__(self) -> None:
        if min(self.major, self.minor, self.patch) < 0:
            raise ValueError("dataset version numbers cannot be negative")
        if not self.label.strip():
            raise ValueError("dataset version label is required")

    @property
    def value(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}+{self.label}"


@dataclass(frozen=True, slots=True)
class VideoIdentifier:
    video_id: str
    source_uri: str = ""

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video id is required")


@dataclass(frozen=True, slots=True)
class FrameIdentifier:
    frame_id: str
    video: VideoIdentifier | None = None
    sequence_number: int = 0

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        if self.sequence_number < 0:
            raise ValueError("sequence number cannot be negative")


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    identifier: FrameIdentifier
    timestamp: datetime
    width: int | None = None
    height: int | None = None
    checksum: str = ""

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive")


@dataclass(frozen=True, slots=True)
class FeatureVector:
    graph_frame_id: str
    detection_graph: DetectionGraph
    market_profile_summary: MappingProxyType[str, Any]
    auction_theory_summary: MappingProxyType[str, Any]
    trend_engine_summary: MappingProxyType[str, Any]
    session_intelligence_summary: MappingProxyType[str, Any]
    multi_timeframe_summary: MappingProxyType[str, Any]
    delta_summary: MappingProxyType[str, Any]
    volume_cluster_summary: MappingProxyType[str, Any]
    confluence_summary: MappingProxyType[str, Any]

    def __post_init__(self) -> None:
        if self.graph_frame_id != self.detection_graph.frame_id:
            raise ValueError("feature vector frame id must match detection graph")


@dataclass(frozen=True, slots=True)
class Annotation:
    annotation_type: AnnotationType
    status: AnnotationStatus = AnnotationStatus.EMPTY
    value: str | Decimal | tuple[str, ...] | None = None
    annotator: str = ""
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is not None and self.timestamp.tzinfo is None:
            raise ValueError("annotation timestamp must be timezone-aware")
        if self.status is AnnotationStatus.EMPTY and self.value not in (
            None,
            "",
            (),
        ):  # deterministic placeholder only
            raise ValueError("empty annotations cannot contain values")

    @staticmethod
    def empty(annotation_type: AnnotationType) -> "Annotation":
        return Annotation(annotation_type)


@dataclass(frozen=True, slots=True)
class TrainingSample:
    sample_id: str
    version: DatasetVersion
    metadata: FrameMetadata
    feature_vector: FeatureVector
    annotations: tuple[Annotation, ...] = ()
    transcript_alignment_id: str | None = None
    knowledge_observation_references: tuple[str, ...] = ()
    knowledge_topics: tuple[str, ...] = ()
    transcript_references: tuple[str, ...] = ()
    transcript_text: tuple[str, ...] = ()
    frame_references: tuple[str, ...] = ()
    timeline_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample id is required")
        if self.metadata.identifier.frame_id != self.feature_vector.graph_frame_id:
            raise ValueError("sample frame id must match feature vector")
        if (
            self.transcript_alignment_id is not None
            and not self.transcript_alignment_id.strip()
        ):
            raise ValueError("transcript alignment id cannot be blank")
        for label, values in (
            ("knowledge observation references", self.knowledge_observation_references),
            ("knowledge topics", self.knowledge_topics),
            ("transcript references", self.transcript_references),
            ("transcript text", self.transcript_text),
            ("frame references", self.frame_references),
            ("timeline references", self.timeline_references),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique per sample")
            if any(not value.strip() for value in values):
                raise ValueError(f"{label} cannot contain blank references")
        seen = {a.annotation_type for a in self.annotations}
        if len(seen) != len(self.annotations):
            raise ValueError("annotation types must be unique per sample")


@dataclass(frozen=True, slots=True)
class DatasetConfiguration:
    dataset_id: str
    version: DatasetVersion = DatasetVersion(0, 1, 0)
    include_empty_annotations: bool = True

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset id is required")


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    sample_count: int
    annotated_sample_count: int
    empty_annotation_count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    dataset_id: str
    version: DatasetVersion
    samples: tuple[TrainingSample, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("dataset creation timestamp must be timezone-aware")
        DatasetValidator().validate(self)

    def statistics(self) -> DatasetStatistics:
        return DatasetStatisticsBuilder.build(self.samples)


class FeatureVectorBuilder:
    def build(self, graph: DetectionGraph) -> FeatureVector:
        return FeatureVector(
            graph.frame_id,
            graph,
            _freeze(
                {
                    "hvn": (
                        len(graph.high_volume_nodes.nodes)
                        if graph.high_volume_nodes is not None
                        else 0
                    ),
                    "lvn": (
                        len(graph.low_volume_nodes.nodes)
                        if graph.low_volume_nodes is not None
                        else 0
                    ),
                    "value_area": graph.value_area is not None,
                }
            ),
            _freeze(
                {
                    "unfinished": graph.unfinished_auction_statistics().total_auctions,
                    "excess": graph.excess_statistics().total_excesses,
                    "poor": graph.poor_auction_statistics().total_auctions,
                }
            ),
            _freeze(
                {
                    "states": len(graph.trend_states()),
                    "pullbacks": len(graph.detected_pullbacks()),
                    "bos": len(graph.bos_events()),
                    "choch": len(graph.choch_events()),
                }
            ),
            _freeze(
                {
                    "sessions": (
                        len(graph.trading_session.sessions)
                        if graph.trading_session is not None
                        else 0
                    ),
                    "initial_balances": (
                        len(graph.initial_balance.balances)
                        if graph.initial_balance is not None
                        else 0
                    ),
                    "opening_auctions": (
                        len(graph.opening_auction.auctions)
                        if graph.opening_auction is not None
                        else 0
                    ),
                }
            ),
            _freeze(
                {
                    "contexts": (
                        len(graph.timeframe_context.contexts)
                        if graph.timeframe_context is not None
                        else 0
                    ),
                    "alignments": (
                        len(graph.alignment.alignments)
                        if graph.alignment is not None
                        else 0
                    ),
                    "aggregations": (
                        len(graph.context_aggregation.aggregations)
                        if graph.context_aggregation is not None
                        else 0
                    ),
                }
            ),
            _freeze(
                {
                    "rows": (
                        len(graph.footprint_delta.rows)
                        if graph.footprint_delta is not None
                        else 0
                    ),
                    "cells": (
                        len(graph.footprint_delta.cells)
                        if graph.footprint_delta is not None
                        else 0
                    ),
                    "divergences": graph.delta_divergence_statistics().total_divergences,
                    "momentum": graph.delta_momentum_statistics().total_momentums,
                }
            ),
            _freeze(
                {
                    "clusters": (
                        len(graph.volume_clusters.clusters)
                        if graph.volume_clusters is not None
                        else 0
                    )
                }
            ),
            _freeze(
                {
                    "confluences": (
                        len(graph.confluence.confluences)
                        if graph.confluence is not None
                        else 0
                    )
                }
            ),
        )


class TrainingSampleBuilder:
    def build(
        self,
        metadata: FrameMetadata,
        feature_vector: FeatureVector,
        version: DatasetVersion,
        annotations: Iterable[Annotation] = (),
        transcript_alignment_id: str | None = None,
        transcript_references: Iterable[str] = (),
        transcript_text: Iterable[str] = (),
        frame_references: Iterable[str] = (),
        timeline_references: Iterable[str] = (),
    ) -> TrainingSample:
        return TrainingSample(
            f"sample:{metadata.identifier.frame_id}:{version.value}",
            version,
            metadata,
            feature_vector,
            tuple(annotations),
            transcript_alignment_id,
            transcript_references=tuple(transcript_references),
            transcript_text=tuple(transcript_text),
            frame_references=tuple(frame_references),
            timeline_references=tuple(timeline_references),
        )


class DatasetBuilder:
    def __init__(self, configuration: DatasetConfiguration) -> None:
        self.configuration = configuration
        self._feature_builder = FeatureVectorBuilder()
        self._sample_builder = TrainingSampleBuilder()

    def build(
        self, frames: Iterable[tuple[FrameMetadata, DetectionGraph]]
    ) -> TrainingDataset:
        samples = []
        for metadata, graph in frames:
            annotations = (
                _empty_annotations()
                if self.configuration.include_empty_annotations
                else ()
            )
            samples.append(
                self._sample_builder.build(
                    metadata,
                    self._feature_builder.build(graph),
                    self.configuration.version,
                    annotations,
                )
            )
        return TrainingDataset(
            self.configuration.dataset_id,
            self.configuration.version,
            tuple(samples),
            datetime(1970, 1, 1, tzinfo=timezone.utc),
        )


class DatasetStatisticsBuilder:
    @staticmethod
    def build(samples: tuple[TrainingSample, ...]) -> DatasetStatistics:
        stamps = tuple(sample.metadata.timestamp for sample in samples)
        return DatasetStatistics(
            len(samples),
            sum(
                any(a.status is not AnnotationStatus.EMPTY for a in s.annotations)
                for s in samples
            ),
            sum(
                a.status is AnnotationStatus.EMPTY
                for s in samples
                for a in s.annotations
            ),
            min(stamps) if stamps else None,
            max(stamps) if stamps else None,
        )


class DatasetValidator:
    def validate(self, dataset: TrainingDataset) -> None:
        if dataset.dataset_id.strip() == "":
            raise ValueError("dataset id is required")
        sample_ids = [sample.sample_id for sample in dataset.samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("training sample ids must be unique")
        frame_ids = [sample.metadata.identifier.frame_id for sample in dataset.samples]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("frame ids must be unique")
        for sample in dataset.samples:
            if sample.version != dataset.version:
                raise ValueError("sample version must match dataset version")
            if (
                sample.feature_vector.detection_graph.frame_id
                != sample.metadata.identifier.frame_id
            ):
                raise ValueError("sample graph reference must match frame metadata")


class DatasetSerializer:
    def to_dict(self, value: Any) -> Any:
        if isinstance(value, MappingProxyType):
            return {k: self.to_dict(v) for k, v in value.items()}
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return {
                field.name: self.to_dict(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, tuple | list):
            return [self.to_dict(item) for item in value]
        if isinstance(value, dict):
            return {str(k): self.to_dict(v) for k, v in value.items()}
        return value

    def dumps(self, value: Any) -> str:
        return json.dumps(self.to_dict(value), sort_keys=True, separators=(",", ":"))


class DatasetExporter:
    def __init__(self, serializer: DatasetSerializer | None = None) -> None:
        self.serializer = serializer or DatasetSerializer()

    def export(
        self,
        dataset: TrainingDataset,
        destination: str | Path,
        export_format: ExportFormat,
    ) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        if export_format == "jsonl":
            path.write_text(
                "".join(self.serializer.dumps(s) + "\n" for s in dataset.samples),
                encoding="utf-8",
            )
        elif export_format == "sqlite":
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS dataset_exports (dataset_id TEXT, version TEXT, sample_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO dataset_exports VALUES (?, ?, ?, ?)",
                    [
                        (
                            dataset.dataset_id,
                            dataset.version.value,
                            s.sample_id,
                            self.serializer.dumps(s),
                        )
                        for s in dataset.samples
                    ],
                )
        elif export_format == "parquet":
            path.write_text(
                self.serializer.dumps(
                    {
                        "format": "parquet",
                        "version": dataset.version.value,
                        "samples": dataset.samples,
                    }
                ),
                encoding="utf-8",
            )
        else:
            raise ValueError("unsupported export format")
        return path


def _empty_annotations() -> tuple[Annotation, ...]:
    return tuple(Annotation.empty(t) for t in AnnotationType)


def _freeze(data: dict[str, Any]) -> MappingProxyType[str, Any]:
    return MappingProxyType(dict(data))
