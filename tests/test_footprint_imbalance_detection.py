from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    DetectionGraph,
    FootprintImbalance,
    FootprintImbalanceDetector,
    FootprintImbalanceResult,
    ImbalanceConfiguration,
    ImbalanceSide,
    ImbalanceType,
    SequentialInterpretationPipeline,
)
from test_footprint_semantic_interpretation import parsed
from test_ocr_foundation import make_graph
from orderflowgpt_genesis import (
    CellSemanticRole,
    CoordinateMapper,
    FootprintMatrixBuilder,
)


def graph_with_values(columns, rows, asks, bids):
    graph = make_graph(columns, rows)
    values = []
    for c in graph.cell_classifications:
        idx = (
            c.cell_reference.coordinate.row_index * columns
            + c.cell_reference.coordinate.column_index
        )
        values.append(parsed(c, CellSemanticRole.ASK_REGION, asks[idx]))
        values.append(parsed(c, CellSemanticRole.CENTER_REGION, 0))
        values.append(parsed(c, CellSemanticRole.BID_REGION, bids[idx]))
    interpretation = SequentialInterpretationPipeline().run(
        graph.cell_classifications, tuple(values)
    )
    base = DetectionGraph(
        graph.frame_id,
        graph.objects,
        graph.grid_coordinate_system,
        graph.cell_classifications,
        graph.ocr_results,
        interpretation,
        graph.parsed_values,
    )
    matrix = FootprintMatrixBuilder().build(base, CoordinateMapper(), interpretation)
    imbalances = FootprintImbalanceDetector().detect(matrix)
    return DetectionGraph(
        base.frame_id,
        base.objects,
        base.grid_coordinate_system,
        base.cell_classifications,
        base.ocr_results,
        interpretation,
        base.parsed_values,
        matrix,
        imbalances,
    )


def test_successful_ask_and_bid_imbalance_lookup_helpers_statistics_ordering():
    graph = graph_with_values(2, 2, asks=(300, 10, 10, 10), bids=(10, 10, 10, 300))
    assert [d.imbalance_type for d in graph.imbalances()] == [
        ImbalanceType.ASK_IMBALANCE,
        ImbalanceType.BID_IMBALANCE,
    ]
    assert graph.ask_imbalances()[0].cell_id == graph.matrix_cell(0, 0).cell_id
    assert graph.bid_imbalances()[0].cell_id == graph.matrix_cell(1, 1).cell_id
    assert graph.has_imbalance(graph.matrix_cell(0, 0).cell_id)
    assert (
        graph.lookup_imbalance(graph.matrix_cell(0, 0).cell_id).side
        == ImbalanceSide.ASK
    )
    stats = graph.imbalance_statistics()
    assert stats.total_cells == 4
    assert stats.ask_imbalances == 1
    assert stats.bid_imbalances == 1
    assert stats.total_imbalances == 2
    assert stats.cells_without_imbalance == 2


def test_ratio_below_threshold_and_minimum_volume_rejection():
    matrix = graph_with_values(1, 2, asks=(20, 1), bids=(1, 10)).footprint_matrix
    assert (
        FootprintImbalanceDetector(ImbalanceConfiguration(minimum_ratio=Decimal("3")))
        .detect(matrix)
        .imbalances()
        == ()
    )
    assert (
        FootprintImbalanceDetector(ImbalanceConfiguration(minimum_volume=Decimal("25")))
        .detect(matrix)
        .imbalances()
        == ()
    )


def test_zero_opposite_volume_configuration():
    matrix = graph_with_values(1, 2, asks=(10, 1), bids=(1, 0)).footprint_matrix
    assert FootprintImbalanceDetector().detect(matrix).imbalances() == ()
    result = FootprintImbalanceDetector(
        ImbalanceConfiguration(allow_zero_opposite=True)
    ).detect(matrix)
    assert result.ask_imbalances()[0].opposite_value == 0


def test_boundaries_single_row_single_column_large_matrix():
    assert (
        graph_with_values(3, 1, asks=(100, 100, 100), bids=(1, 1, 1)).imbalances() == ()
    )
    assert (
        graph_with_values(1, 3, asks=(100, 1, 1), bids=(1, 1, 100))
        .imbalance_statistics()
        .total_imbalances
        == 2
    )
    large = graph_with_values(
        5, 5, asks=tuple([100] + [1] * 24), bids=tuple([1] * 5 + [1] + [1] * 19)
    )
    assert large.imbalance_statistics().total_cells == 25


def test_duplicate_detection_rejection_metadata_validation_and_immutability():
    result = graph_with_values(1, 2, asks=(100, 1), bids=(1, 1)).footprint_imbalances
    detection = result.imbalances()[0]
    with pytest.raises(ValueError, match="duplicate"):
        FootprintImbalanceResult(result.matrix, (detection, detection))
    with pytest.raises(ValueError, match="confidence"):
        FootprintImbalance(
            detection.cell_id,
            detection.position,
            detection.imbalance_type,
            detection.side,
            detection.ratio,
            detection.dominant_value,
            detection.opposite_value,
            2.0,
        )
    with pytest.raises(FrozenInstanceError):
        detection.cell_id = "changed"
    with pytest.raises(TypeError):
        detection.metadata["x"] = "y"


def test_pipeline_integration_produces_empty_deterministic_result():
    graph = make_graph(2, 2)
    assert graph.footprint_imbalances is not None
    assert graph.imbalances() == ()
