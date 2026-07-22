from __future__ import annotations

import json
import sqlite3
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from orderflowgpt_genesis import (
    Annotation,
    AnnotationStatus,
    AnnotationType,
    DatasetBuilder,
    DatasetConfiguration,
    DatasetExporter,
    DatasetSerializer,
    DatasetValidator,
    DatasetVersion,
    DetectionGraph,
    FeatureVectorBuilder,
    FrameIdentifier,
    FrameMetadata,
    TrainingDataset,
    TrainingSampleBuilder,
)


def metadata(frame_id: str, offset: int = 0) -> FrameMetadata:
    return FrameMetadata(
        FrameIdentifier(frame_id, sequence_number=offset),
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=offset),
        1920,
        1080,
        f"sha256:{frame_id}",
    )


def graph(frame_id: str) -> DetectionGraph:
    return DetectionGraph(frame_id=frame_id)


def dataset(count: int = 2) -> TrainingDataset:
    frames = [(metadata(f"frame-{i}", i), graph(f"frame-{i}")) for i in range(count)]
    return DatasetBuilder(
        DatasetConfiguration("genesis", DatasetVersion(0, 1, 0))
    ).build(frames)


def test_feature_vector_creation_and_graph_integration() -> None:
    vector = FeatureVectorBuilder().build(graph("frame-1"))
    assert vector.graph_frame_id == "frame-1"
    assert vector.detection_graph.frame_id == "frame-1"
    assert vector.delta_summary["rows"] == 0
    assert vector.confluence_summary["confluences"] == 0


def test_training_sample_creation_with_empty_annotations() -> None:
    version = DatasetVersion(0, 1, 0)
    vector = FeatureVectorBuilder().build(graph("frame-1"))
    sample = TrainingSampleBuilder().build(metadata("frame-1"), vector, version)
    assert sample.sample_id == "sample:frame-1:0.1.0+bundle-9"
    assert sample.annotations == ()


def test_dataset_builder_adds_placeholders_and_statistics() -> None:
    built = dataset(3)
    stats = built.statistics()
    assert stats.sample_count == 3
    assert stats.annotated_sample_count == 0
    assert stats.empty_annotation_count == 3 * len(AnnotationType)
    assert stats.first_timestamp == metadata("frame-0", 0).timestamp
    assert stats.last_timestamp == metadata("frame-2", 2).timestamp


def test_dataset_export_jsonl_sqlite_and_parquet_are_versioned(tmp_path) -> None:
    built = dataset(2)
    exporter = DatasetExporter()
    jsonl = exporter.export(built, tmp_path / "dataset.jsonl", "jsonl")
    sqlite = exporter.export(built, tmp_path / "dataset.sqlite", "sqlite")
    parquet = exporter.export(built, tmp_path / "dataset.parquet", "parquet")
    lines = jsonl.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["version"]["label"] == "bundle-9"
    with sqlite3.connect(sqlite) as conn:
        assert conn.execute(
            "SELECT version, count(*) FROM dataset_exports GROUP BY version"
        ).fetchone() == ("0.1.0+bundle-9", 2)
    assert "0.1.0+bundle-9" in parquet.read_text()


def test_dataset_validation_rejects_duplicates_and_bad_references() -> None:
    built = dataset(1)
    with pytest.raises(ValueError, match="sample ids must be unique"):
        TrainingDataset(
            "genesis",
            built.version,
            (built.samples[0], built.samples[0]),
            built.created_at,
        )
    vector = FeatureVectorBuilder().build(graph("other"))
    with pytest.raises(ValueError, match="sample frame id"):
        TrainingSampleBuilder().build(metadata("frame-x"), vector, built.version)
    DatasetValidator().validate(built)


def test_serialization_is_deterministic() -> None:
    built = dataset(1)
    serializer = DatasetSerializer()
    assert serializer.dumps(built) == serializer.dumps(built)
    payload = json.loads(serializer.dumps(built.samples[0]))
    assert payload["metadata"]["timestamp"] == "2026-01-01T12:00:00+00:00"


def test_versioning_validation() -> None:
    assert DatasetVersion(1, 2, 3, "bundle-9").value == "1.2.3+bundle-9"
    with pytest.raises(ValueError):
        DatasetVersion(-1, 0, 0)


def test_large_dataset_is_deterministic() -> None:
    built = dataset(250)
    assert built.statistics().sample_count == 250
    assert built.samples[-1].sample_id == "sample:frame-249:0.1.0+bundle-9"


def test_immutability() -> None:
    built = dataset(1)
    with pytest.raises(FrozenInstanceError):
        built.samples[0].sample_id = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        built.samples[0].feature_vector.delta_summary["rows"] = 2  # type: ignore[index]


def test_annotations_accept_human_labels_and_reject_non_empty_empty_status() -> None:
    ann = Annotation(
        AnnotationType.CONFIDENCE,
        AnnotationStatus.REVIEWED,
        "high",
        "Fabio",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert ann.value == "high"
    with pytest.raises(ValueError):
        Annotation(AnnotationType.COMMENTS, AnnotationStatus.EMPTY, "not empty")
