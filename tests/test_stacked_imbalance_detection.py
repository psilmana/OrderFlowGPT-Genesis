from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    DetectionGraph,
    StackedImbalance,
    StackedImbalanceConfiguration,
    StackedImbalanceDetector,
    StackedImbalanceResult,
    StackedImbalanceType,
)
from test_footprint_imbalance_detection import graph_with_values
from test_ocr_foundation import make_graph


def detect_stacks(columns, rows, asks, bids, configuration=None):
    graph = graph_with_values(columns, rows, asks, bids)
    result = StackedImbalanceDetector(
        configuration or StackedImbalanceConfiguration()
    ).detect(graph.footprint_matrix, graph.footprint_imbalances)
    return DetectionGraph(
        graph.frame_id,
        graph.objects,
        graph.grid_coordinate_system,
        graph.cell_classifications,
        graph.ocr_results,
        graph.footprint_interpretation,
        graph.parsed_values,
        graph.footprint_matrix,
        graph.footprint_imbalances,
        result,
    )


def test_successful_ask_stack_lookup_statistics_and_immutability():
    graph = detect_stacks(1, 4, (100, 100, 100, 1), (1, 1, 1, 1))
    stack = graph.ask_stacks()[0]
    assert stack.stack_type == StackedImbalanceType.STACKED_ASK
    assert stack.starting_cell == graph.matrix_cell(0, 0).position
    assert stack.ending_cell == graph.matrix_cell(2, 0).position
    assert stack.row_span == (0, 2)
    assert len(stack.cells) == 3
    assert stack.average_ratio == Decimal("100")
    assert stack.total_dominant_volume == Decimal("300")
    assert graph.lookup_stack(stack.stack_id) == stack
    assert graph.lookup_stacks_by_cell(graph.matrix_cell(1, 0).cell_id) == (stack,)
    stats = graph.stacked_imbalance_statistics()
    assert stats.total_stacks == 1
    assert stats.ask_stacks == 1
    assert stats.bid_stacks == 0
    assert stats.largest_stack == 3
    assert stats.maximum_stack_size == 3
    assert stats.average_stack_size == Decimal("3")
    with pytest.raises(FrozenInstanceError):
        stack.stack_id = "changed"
    with pytest.raises(TypeError):
        stack.metadata["x"] = "y"


def test_successful_bid_stack_and_single_imbalance_no_stack():
    bid_graph = detect_stacks(1, 4, (1, 1, 1, 1), (1, 100, 100, 100))
    assert len(bid_graph.bid_stacks()) == 1
    assert bid_graph.bid_stacks()[0].stack_type == StackedImbalanceType.STACKED_BID
    single = detect_stacks(
        1, 2, (100, 1), (1, 1), StackedImbalanceConfiguration(minimum_stack_size=2)
    )
    assert single.stacks() == ()


def test_minimum_stack_average_ratio_and_volume_rejections():
    assert detect_stacks(1, 4, (100, 100, 1, 1), (1, 1, 1, 1)).stacks() == ()
    config = StackedImbalanceConfiguration(
        minimum_stack_size=3, minimum_average_ratio=Decimal("101")
    )
    assert detect_stacks(1, 4, (100, 100, 100, 1), (1, 1, 1, 1), config).stacks() == ()
    volume_config = StackedImbalanceConfiguration(
        minimum_stack_size=3, minimum_total_volume=Decimal("301")
    )
    assert (
        detect_stacks(1, 4, (100, 100, 100, 1), (1, 1, 1, 1), volume_config).stacks()
        == ()
    )


def test_gap_handling_boundaries_single_row_single_column_large_matrix_and_ordering():
    no_gap = detect_stacks(
        1,
        4,
        (100, 1, 100, 1),
        (1, 1, 1, 1),
        StackedImbalanceConfiguration(minimum_stack_size=2),
    )
    assert no_gap.stacks() == ()
    gap = detect_stacks(
        1,
        4,
        (100, 1, 100, 1),
        (1, 1, 1, 1),
        StackedImbalanceConfiguration(
            minimum_stack_size=2, allow_gaps=True, maximum_gap=1
        ),
    )
    assert len(gap.stacks()) == 1
    assert gap.stacks()[0].row_span == (0, 2)
    assert detect_stacks(3, 1, (100, 100, 100), (1, 1, 1)).stacks() == ()
    assert (
        len(detect_stacks(1, 5, (100, 100, 100, 100, 1), (1, 1, 1, 1, 1)).ask_stacks())
        == 1
    )
    large = detect_stacks(5, 20, tuple([100] * 95 + [1] * 5), tuple([1] * 100))
    assert [stack.stack_id for stack in large.stacks()] == sorted(
        stack.stack_id for stack in large.stacks()
    )


def test_duplicate_stack_and_validation_rejection_pipeline_and_graph_integration():
    graph = detect_stacks(1, 4, (100, 100, 100, 1), (1, 1, 1, 1))
    stack = graph.stacks()[0]
    with pytest.raises(ValueError, match="duplicate"):
        StackedImbalanceResult(
            graph.footprint_matrix, graph.footprint_imbalances, (stack, stack)
        )
    with pytest.raises(ValueError, match="minimum stack size"):
        StackedImbalanceConfiguration(minimum_stack_size=1)
    with pytest.raises(ValueError, match="confidence"):
        StackedImbalance(
            stack.stack_id,
            stack.stack_type,
            stack.starting_cell,
            stack.ending_cell,
            stack.cells,
            stack.row_span,
            stack.average_ratio,
            stack.total_dominant_volume,
            2.0,
        )
    pipeline_graph = make_graph(2, 2)
    assert pipeline_graph.stacked_imbalances is not None
    assert pipeline_graph.stacks() == ()
    with pytest.raises(ValueError, match="reference graph matrix"):
        DetectionGraph(
            graph.frame_id,
            graph.objects,
            graph.grid_coordinate_system,
            graph.cell_classifications,
            graph.ocr_results,
            graph.footprint_interpretation,
            graph.parsed_values,
            graph.footprint_matrix,
            graph.footprint_imbalances,
            StackedImbalanceDetector().detect(
                pipeline_graph.footprint_matrix, pipeline_graph.footprint_imbalances
            ),
        )
