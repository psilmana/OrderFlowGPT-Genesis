from decimal import Decimal
from types import MappingProxyType

import pytest

from orderflowgpt_genesis import (
    Alignment,
    AlignmentAnalyzer,
    AlignmentResult,
    AlignmentType,
    ConfluenceAnalyzer,
    ConfluenceConfiguration,
    ConfluenceType,
    ContextAggregation,
    ContextAggregationAnalyzer,
    ContextAggregationResult,
    DetectionGraph,
    TimeframeAnalyzer,
    TimeframeConfiguration,
    TimeframeContext,
    TimeframeResult,
    TimeframeType,
)


def test_timeframe_models_ordering_lookup_statistics_immutability_metadata():
    result = TimeframeResult(
        (
            TimeframeContext(
                "tf:1m",
                TimeframeType.ONE_MINUTE,
                ("trend:1",),
                "bullish",
                Decimal("1"),
                {"k": "v"},
            ),
            TimeframeContext(
                "tf:5m", TimeframeType.FIVE_MINUTE, ("trend:1",), "bullish"
            ),
        ),
        TimeframeConfiguration((TimeframeType.ONE_MINUTE, TimeframeType.FIVE_MINUTE)),
    )

    assert result.lookup("tf:1m") is result.contexts[0]
    assert result.by_type(TimeframeType.FIVE_MINUTE) == (result.contexts[1],)
    assert result.statistics().bullish_contexts == 2
    assert isinstance(result.contexts[0].metadata, MappingProxyType)
    with pytest.raises(Exception):
        result.contexts[0].metadata["x"] = "y"  # type: ignore[index]
    with pytest.raises(Exception):
        result.contexts[0].dominant_direction = "bearish"  # type: ignore[misc]


@pytest.mark.parametrize(
    "contexts, expected",
    [
        (("bullish", "bullish"), AlignmentType.FULLY_ALIGNED),
        (("bullish", "neutral"), AlignmentType.PARTIALLY_ALIGNED),
        (("bullish", "bearish"), AlignmentType.OPPOSING),
        (("neutral", "neutral"), AlignmentType.NEUTRAL),
    ],
)
def test_alignment_classifications(contexts, expected):
    timeframes = TimeframeResult(
        tuple(
            TimeframeContext(f"tf:{i}", tf, (f"ref:{i}",), direction)
            for i, (tf, direction) in enumerate(
                zip((TimeframeType.ONE_MINUTE, TimeframeType.FIVE_MINUTE), contexts)
            )
        ),
        TimeframeConfiguration((TimeframeType.ONE_MINUTE, TimeframeType.FIVE_MINUTE)),
    )
    result = AlignmentAnalyzer().analyze(timeframes)
    assert result.alignments[0].alignment_type == expected
    assert result.lookup("alignment:primary") == result.alignments[0]
    assert result.statistics().total_alignments == 1


def test_context_aggregation_and_confluence_classifications():
    aggregations = ContextAggregationResult(
        tuple(
            ContextAggregation(f"aggregation:{i}", "trend", (f"ref:{i}",), "bullish")
            for i in range(4)
        )
    )
    assert (
        ConfluenceAnalyzer().analyze(aggregations).confluences[0].confluence_type
        == ConfluenceType.STRONG_CONFLUENCE
    )
    assert (
        ConfluenceAnalyzer(ConfluenceConfiguration())
        .analyze(ContextAggregationResult(aggregations.aggregations[:3]))
        .confluences[0]
        .confluence_type
        == ConfluenceType.MODERATE_CONFLUENCE
    )
    assert (
        ConfluenceAnalyzer()
        .analyze(ContextAggregationResult(aggregations.aggregations[:2]))
        .confluences[0]
        .confluence_type
        == ConfluenceType.WEAK_CONFLUENCE
    )
    assert (
        ConfluenceAnalyzer()
        .analyze(
            ContextAggregationResult(
                (ContextAggregation("aggregation:none", "none", ("ref",), "neutral"),)
            )
        )
        .confluences[0]
        .confluence_type
        == ConfluenceType.NO_CONFLUENCE
    )


def test_validation_duplicates_confidence_references_ordering():
    with pytest.raises(ValueError):
        TimeframeConfiguration((TimeframeType.FIVE_MINUTE, TimeframeType.ONE_MINUTE))
    with pytest.raises(ValueError):
        TimeframeContext("tf", TimeframeType.ONE_MINUTE, ("ref", "ref"))
    with pytest.raises(ValueError):
        TimeframeContext(
            "tf", TimeframeType.ONE_MINUTE, ("ref",), confidence=Decimal("1.1")
        )
    with pytest.raises(ValueError):
        Alignment("a", AlignmentType.NEUTRAL, ())
    tf = TimeframeResult((TimeframeContext("tf", TimeframeType.ONE_MINUTE, ("ref",)),))
    with pytest.raises(ValueError):
        AlignmentResult((Alignment("a", AlignmentType.NEUTRAL, ("missing",)),), tf)


def test_large_dataset_graph_and_pipeline_helpers():
    graph = DetectionGraph(frame_id="frame-1")
    timeframe_result = TimeframeAnalyzer(
        TimeframeConfiguration(
            (
                TimeframeType.TICK,
                TimeframeType.ONE_MINUTE,
                TimeframeType.FIVE_MINUTE,
                TimeframeType.FIFTEEN_MINUTE,
                TimeframeType.THIRTY_MINUTE,
                TimeframeType.ONE_HOUR,
                TimeframeType.FOUR_HOUR,
                TimeframeType.DAILY,
            )
        )
    ).analyze(graph)
    aggregation_result = ContextAggregationAnalyzer().analyze(graph)
    confluence_result = ConfluenceAnalyzer().analyze(aggregation_result)
    graph = DetectionGraph(
        frame_id="frame-1",
        timeframe_context=timeframe_result,
        alignment=AlignmentAnalyzer().analyze(timeframe_result),
        context_aggregation=aggregation_result,
        confluence=confluence_result,
    )

    assert len(graph.timeframe_contexts()) == 8
    assert graph.lookup_timeframe_context(timeframe_result.contexts[0].context_id)
    assert graph.timeframe_statistics().total_contexts == 8
    assert graph.alignment_statistics().total_alignments == 1
    assert graph.context_aggregation_statistics().total_aggregations == 1
    assert graph.confluence_statistics().none == 1
