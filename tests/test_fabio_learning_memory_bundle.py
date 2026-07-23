from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from orderflowgpt_genesis import (
    KnowledgeCategory,
    KnowledgeContext,
    KnowledgeDataset,
    KnowledgeIdentifier,
    KnowledgeMetadata,
    KnowledgeObservation,
    KnowledgeReference,
    KnowledgeStatement,
    KnowledgeTimeline,
    KnowledgeTopic,
    LearningConfiguration,
    MemoryBuilder,
    MemoryDatabase,
    MemoryEntry,
    MemoryFeatureVector,
    MemoryIdentifier,
    MemoryIndexer,
    MemoryLoader,
    MemorySerializer,
    RetrievedExample,
    SimilarityConfiguration,
    SimilarityMetric,
    attach_retrieval_result,
    feature_vector_from_graph,
    score_vectors,
)
from orderflowgpt_genesis.vision import DetectionGraph


def _observation(i: int, video: str = "video-a", category=KnowledgeCategory.POC):
    topic = KnowledgeTopic(f"topic:{category.value.lower()}", category, category.value)
    stmt = KnowledgeStatement(
        f"statement:s{i}",
        f"s{i}",
        f"Fabio explains {category.value} {i}",
        category,
        (topic.topic_id,),
    )
    ctx = KnowledgeContext(
        f"context:f{i}:s{i}",
        video,
        f"f{i}",
        i * 1000,
        f"alignment:t:s{i}",
        f"sample:f{i}",
        f"f{i}",
    )
    ref = KnowledgeReference(
        f"knowledge-ref:f{i}:s{i}",
        stmt.statement_id,
        stmt.sentence_id,
        f"f{i}",
        f"f{i}",
        f"sample:f{i}",
    )
    return (
        topic,
        stmt,
        KnowledgeObservation(f"knowledge-observation:f{i}:s{i}", stmt, ctx, ref),
    )


def _knowledge_dataset(
    count=3,
    videos=("video-a",),
    categories=(KnowledgeCategory.POC, KnowledgeCategory.DELTA),
):
    rows = [
        _observation(i, videos[i % len(videos)], categories[i % len(categories)])
        for i in range(count)
    ]
    topics = tuple(sorted({r[0] for r in rows}, key=lambda t: t.topic_id))
    statements = tuple(r[1] for r in rows)
    observations = tuple(r[2] for r in rows)
    return KnowledgeDataset(
        "kd",
        KnowledgeMetadata(KnowledgeIdentifier("kid", videos[0], "tx")),
        topics,
        statements,
        KnowledgeTimeline(observations),
    )


def test_memory_creation_indexing_statistics_metadata_and_immutability():
    result = MemoryBuilder(LearningConfiguration("mem", "lesson-1")).build(
        _knowledge_dataset(4, ("v1", "v2"))
    )
    assert result.statistics.entry_count == 4
    assert result.statistics.video_count == 2
    assert result.statistics.lesson_count == 1
    assert result.dataset.metadata.bundle == "bundle-13"
    assert tuple(result.dataset.index.by_memory_id) == tuple(
        e.identifier.memory_id for e in result.dataset.entries
    )
    with pytest.raises(FrozenInstanceError):
        result.dataset.entries[0].video_id = "changed"  # type: ignore[misc]


def test_memory_duplicate_prevention_and_ordering():
    entry = MemoryEntry(
        MemoryIdentifier("memory:a", "obs", "kd"),
        MemoryFeatureVector(("topic_poc",), (1.0,)),
        "v",
        "l",
        0,
        "txt",
        "obs",
        "POC",
        ("topic:poc",),
        ("s",),
    )
    dup = MemoryEntry(
        MemoryIdentifier("memory:a", "obs2", "kd"),
        entry.feature_vector,
        "v",
        "l",
        1,
        "txt",
        "obs2",
        "POC",
        ("topic:poc",),
        ("s2",),
    )
    with pytest.raises(ValueError):
        MemoryIndexer().index((entry, dup))
    assert MemoryIndexer().index((entry,)).entries == (entry,)


def test_feature_vectors_and_similarity_metrics_are_deterministic():
    poc = MemoryFeatureVector(("topic_delta", "topic_poc"), (0.0, 1.0), ("a",))
    delta = MemoryFeatureVector(("topic_delta", "topic_poc"), (1.0, 0.0), ("b",))
    assert (
        score_vectors(
            poc, poc, SimilarityConfiguration(SimilarityMetric.COSINE_SIMILARITY)
        ).score
        == 1.0
    )
    assert score_vectors(
        poc, delta, SimilarityConfiguration(SimilarityMetric.EUCLIDEAN_DISTANCE)
    ).distance == pytest.approx(2**0.5)
    assert (
        score_vectors(
            poc, delta, SimilarityConfiguration(SimilarityMetric.MANHATTAN_DISTANCE)
        ).distance
        == 2.0
    )
    assert (
        score_vectors(
            poc, delta, SimilarityConfiguration(SimilarityMetric.HAMMING_SIMILARITY)
        ).score
        == 0.0
    )
    weighted = score_vectors(
        poc,
        delta,
        SimilarityConfiguration(
            SimilarityMetric.WEIGHTED_FEATURE_DISTANCE, weights={"topic_poc": 3.0}
        ),
    )
    assert weighted.distance == 4.0
    assert (
        feature_vector_from_graph(DetectionGraph("f1")).names == poc.names
        or "topic_poc" in feature_vector_from_graph(DetectionGraph("f1")).names
    )


def test_serialization_loading_small_single_and_large_databases(tmp_path: Path):
    one = (
        MemoryBuilder(LearningConfiguration("single"))
        .build(_knowledge_dataset(1))
        .dataset
    )
    loaded = MemoryLoader().loads(MemorySerializer().dumps(one))
    assert loaded.statistics.entry_count == 1
    path = MemorySerializer().dump(one, tmp_path / "memory.json")
    assert MemoryLoader().load(path).entries == loaded.entries
    large = (
        MemoryBuilder(LearningConfiguration("large"))
        .build(_knowledge_dataset(150, ("v1", "v2", "v3")))
        .dataset
    )
    assert large.statistics.entry_count == 150
    assert large.statistics.video_count == 3


def test_top_n_retrieval_multiple_lessons_videos_graph_and_pipeline_integration():
    dataset = (
        MemoryBuilder(LearningConfiguration("mem", "lesson-a"))
        .build(
            _knowledge_dataset(
                6,
                ("v1", "v2"),
                (
                    KnowledgeCategory.POC,
                    KnowledgeCategory.DELTA,
                    KnowledgeCategory.ABSORPTION,
                ),
            )
        )
        .dataset
    )
    graph = DetectionGraph("query", knowledge_observations=("category:poc",))
    result = MemoryDatabase(dataset).search(
        graph,
        SimilarityConfiguration(SimilarityMetric.WEIGHTED_FEATURE_DISTANCE, top_n=2),
    )
    assert len(result.retrieved_examples.examples) == 2
    assert all(
        isinstance(e, RetrievedExample) for e in result.retrieved_examples.examples
    )
    assert result.retrieved_examples.examples == tuple(
        sorted(
            result.retrieved_examples.examples,
            key=lambda e: (
                -e.similarity_score.score,
                e.similarity_score.distance,
                e.timestamp_ms,
                e.memory_identifier.memory_id,
            ),
        )
    )
    enhanced = attach_retrieval_result(graph, result)
    assert len(enhanced.memory_references) == 2
    assert enhanced.retrieval_statistics == (2, 6)


def test_reference_validation():
    with pytest.raises(ValueError):
        MemoryFeatureVector(("b", "a"), (1.0, 2.0))
    with pytest.raises(ValueError):
        DetectionGraph("f", memory_references=("m", "m"))
