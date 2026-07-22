from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from orderflowgpt_genesis.dataset import (
    DatasetVersion,
    FeatureVector,
    FrameIdentifier,
    FrameMetadata,
    TrainingSample,
)
from orderflowgpt_genesis.knowledge import (
    KnowledgeCategory,
    KnowledgeConfiguration,
    KnowledgeDataset,
    KnowledgeExtractionEngine,
    KnowledgeIdentifier,
    KnowledgeMetadata,
    KnowledgeObservation,
    KnowledgeTimeline,
)
from orderflowgpt_genesis.transcript import (
    AlignmentConfidence,
    FrameTranscriptAlignment,
    TranscriptDataset,
    TranscriptIdentifier,
    TranscriptMetadata,
    TranscriptSentence,
    TranscriptTimeline,
)
from orderflowgpt_genesis.vision import DetectionGraph


def _timeline(*texts: str) -> TranscriptTimeline:
    return TranscriptTimeline(
        TranscriptMetadata(TranscriptIdentifier("tx1", "video1")),
        tuple(
            TranscriptSentence(f"s{i}", text, i * 1000, i * 1000 + 500)
            for i, text in enumerate(texts)
        ),
    )


def _dataset(texts: tuple[str, ...], frames: int | None = None) -> TranscriptDataset:
    timeline = _timeline(*texts)
    count = len(texts) if frames is None else frames
    alignments = tuple(
        FrameTranscriptAlignment(
            f"f{i}",
            "video1",
            i * 1000,
            f"s{min(i, len(texts) - 1)}" if texts else None,
            None,
            None,
            (f"s{min(i, len(texts) - 1)}",) if texts else (),
            AlignmentConfidence.EXACT,
        )
        for i in range(count)
    )
    return TranscriptDataset("td1", timeline, alignments)


def _graph(i: int) -> DetectionGraph:
    return DetectionGraph(f"f{i}")


def _sample(i: int) -> TrainingSample:
    graph = _graph(i)
    frame = FrameMetadata(
        FrameIdentifier(f"f{i}"), datetime.fromtimestamp(i, tz=timezone.utc)
    )
    empty: MappingProxyType[str, object] = MappingProxyType({})
    fv = FeatureVector(
        f"f{i}", graph, empty, empty, empty, empty, empty, empty, empty, empty
    )
    return TrainingSample(f"sample:f{i}", DatasetVersion(0, 1, 0), frame, fv)


def _extract(texts: tuple[str, ...], frames: int | None = None):
    return KnowledgeExtractionEngine(KnowledgeConfiguration("kd1")).extract(
        _dataset(texts, frames),
        [_graph(i) for i in range(frames or len(texts))],
        [_sample(i) for i in range(frames or len(texts))],
    )


def test_knowledge_extraction_transcript_and_frame_mapping():
    result = _extract(("Look at the absorption.",))
    obs = result.dataset.timeline.observations[0]
    assert obs.statement.text == "Look at the absorption."
    assert obs.statement.category is KnowledgeCategory.ABSORPTION
    assert obs.context.timestamp_ms == 0
    assert obs.reference.frame_id == "f0"
    assert obs.reference.detection_graph_frame_id == "f0"
    assert obs.reference.training_sample_id == "sample:f0"


@pytest.mark.parametrize(
    "text, category",
    [
        ("Stacked imbalance here", KnowledgeCategory.STACKED_IMBALANCE),
        ("The POC is important", KnowledgeCategory.POC),
        ("Inside the value area", KnowledgeCategory.VALUE_AREA),
        ("Trend changed", KnowledgeCategory.TREND),
        ("Market structure shifted", KnowledgeCategory.MARKET_STRUCTURE),
        ("Unfinished auction", KnowledgeCategory.AUCTION_THEORY),
        ("Delta is negative", KnowledgeCategory.DELTA),
        ("RTH session open", KnowledgeCategory.SESSION),
        ("High volume node", KnowledgeCategory.VOLUME),
        ("Confluence aligns", KnowledgeCategory.CONFLUENCE),
        ("Notice this behavior", KnowledgeCategory.GENERAL_OBSERVATION),
        ("Completely unrelated words", KnowledgeCategory.UNKNOWN),
    ],
)
def test_category_assignment_and_unknown_statements(
    text: str, category: KnowledgeCategory
):
    assert KnowledgeExtractionEngine.classify(text) is category


def test_multiple_knowledge_statements_and_timeline_ordering():
    result = _extract(("Delta here", "Look at the absorption", "The POC"))
    assert [o.context.timestamp_ms for o in result.dataset.timeline.observations] == [
        0,
        1000,
        2000,
    ]
    assert [s.category for s in result.dataset.statements] == [
        KnowledgeCategory.DELTA,
        KnowledgeCategory.ABSORPTION,
        KnowledgeCategory.POC,
    ]


def test_large_small_and_multiple_videos_are_deterministic():
    small = _extract(("absorption",))
    large = _extract(tuple("delta" for _ in range(75)))
    second = KnowledgeExtractionEngine(KnowledgeConfiguration("kd2")).extract(
        TranscriptDataset(
            "td2",
            TranscriptTimeline(
                TranscriptMetadata(TranscriptIdentifier("tx2", "video2")),
                (TranscriptSentence("s0", "POC", 0, 10),),
            ),
            (
                FrameTranscriptAlignment(
                    "f0",
                    "video2",
                    0,
                    "s0",
                    None,
                    None,
                    ("s0",),
                    AlignmentConfidence.EXACT,
                ),
            ),
        ),
        [DetectionGraph("f0")],
    )
    assert small.statistics.observation_count == 1
    assert large.statistics.observation_count == 75
    assert second.dataset.metadata.identifier.video_id == "video2"


def test_training_sample_and_detection_graph_integration():
    result = _extract(("absorption",))
    sample = result.training_samples[0]
    graph = result.detection_graphs[0]
    assert sample.knowledge_observation_references == ("knowledge-observation:f0:s0",)
    assert sample.knowledge_topics == ("topic:absorption",)
    assert sample.transcript_references == ("s0",)
    assert sample.frame_references == ("f0",)
    assert graph.knowledge_observations == ("knowledge-observation:f0:s0",)
    assert graph.knowledge_references == ("knowledge-ref:f0:s0",)
    assert graph.knowledge_statistics == (1,)


def test_knowledge_dataset_statistics_metadata_and_immutability():
    result = _extract(("absorption", "unknown words"))
    stats = result.dataset.statistics()
    assert stats.statement_count == 2
    assert stats.category_counts[KnowledgeCategory.ABSORPTION] == 1
    assert result.dataset.metadata.source == "Fabio"
    with pytest.raises(TypeError):
        stats.category_counts[KnowledgeCategory.ABSORPTION] = 9  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.dataset.dataset_id = "x"  # type: ignore[misc]


def test_duplicate_and_reference_validation():
    result = _extract(("absorption",))
    obs: KnowledgeObservation = result.dataset.timeline.observations[0]
    with pytest.raises(ValueError):
        KnowledgeTimeline((obs, obs))
    with pytest.raises(ValueError):
        KnowledgeDataset(
            "kd",
            KnowledgeMetadata(KnowledgeIdentifier("k", "video1", "tx1")),
            (),
            result.dataset.statements,
            result.dataset.timeline,
        )


def test_ordering_validation():
    result = _extract(("first absorption", "second delta"))
    reversed_obs = tuple(reversed(result.dataset.timeline.observations))
    with pytest.raises(ValueError):
        KnowledgeTimeline(reversed_obs)
