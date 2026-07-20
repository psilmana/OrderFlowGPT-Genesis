from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    CellDelta,
    DeltaResult,
    DeltaType,
    DetectionGraph,
    FootprintDeltaAnalyzer,
    RowDelta,
)
from test_footprint_imbalance_detection import graph_with_values
from test_ocr_foundation import make_graph


def analyze(columns, rows, asks, bids):
    graph = graph_with_values(columns, rows, asks, bids)
    result = FootprintDeltaAnalyzer().analyze(graph.footprint_matrix)
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
        graph.stacked_imbalances,
        graph.absorption,
        result,
    )


def test_single_cell_delta_and_lookup_helpers():
    graph = analyze(1, 1, asks=(7,), bids=(2,))
    cell = graph.cell_delta(graph.matrix_cell(0, 0).cell_id)
    assert cell.delta == Decimal("5")
    assert cell.absolute_delta == Decimal("5")
    assert cell.delta_type == DeltaType.POSITIVE
    assert graph.row_delta(0) == RowDelta(
        0, Decimal("2"), Decimal("7"), Decimal("5"), Decimal("5"), 1
    )
    assert graph.positive_cells() == (cell,)
    assert graph.negative_cells() == ()
    assert graph.zero_cells() == ()


def test_multiple_rows_mixed_row_matrix_aggregation_and_statistics():
    graph = analyze(2, 2, asks=(10, 1, 3, 4), bids=(1, 5, 3, 10))
    result = graph.footprint_delta
    assert [cell.delta for cell in result.cells] == [
        Decimal("9"),
        Decimal("-4"),
        Decimal("0"),
        Decimal("-6"),
    ]
    assert [row.row_delta for row in result.rows] == [Decimal("5"), Decimal("-6")]
    assert result.footprint.total_bid == Decimal("19")
    assert result.footprint.total_ask == Decimal("18")
    assert result.footprint.net_delta == Decimal("-1")
    assert result.footprint.absolute_delta == Decimal("1")
    assert result.footprint.maximum_positive_delta == Decimal("9")
    assert result.footprint.maximum_negative_delta == Decimal("-6")
    assert result.footprint.average_cell_delta == Decimal("-0.25")
    stats = graph.delta_statistics()
    assert stats.rows == 2
    assert stats.cells == 4
    assert stats.positive_cells == 1
    assert stats.negative_cells == 2
    assert stats.zero_cells == 1
    assert stats.maximum_delta == Decimal("9")
    assert stats.minimum_delta == Decimal("-6")
    assert stats.average_delta == Decimal("-0.25")


def test_large_all_positive_all_negative_and_zero_matrices():
    positive = analyze(5, 4, asks=tuple(range(10, 30)), bids=tuple(range(20)))
    assert positive.delta_statistics().positive_cells == 20
    negative = analyze(3, 3, asks=(1,) * 9, bids=(2,) * 9)
    assert negative.delta_statistics().negative_cells == 9
    zero = analyze(4, 3, asks=(5,) * 12, bids=(5,) * 12)
    assert zero.delta_statistics().zero_cells == 12


def test_validation_immutability_duplicate_cells_rows_ordering_metadata_and_references():
    graph = analyze(1, 1, asks=(3,), bids=(1,))
    result = graph.footprint_delta
    cell = result.cells[0]
    with pytest.raises(FrozenInstanceError):
        cell.delta = Decimal("0")
    with pytest.raises(TypeError):
        cell.metadata["x"] = "y"
    with pytest.raises(ValueError, match="confidence"):
        CellDelta(
            cell.cell_id,
            0,
            0,
            Decimal("1"),
            Decimal("2"),
            Decimal("1"),
            Decimal("1"),
            2.0,
        )
    with pytest.raises(ValueError, match="duplicate delta cells"):
        DeltaResult(
            result.matrix,
            (cell, cell),
            result.rows,
            result.footprint,
            result.statistics(),
        )
    with pytest.raises(ValueError, match="duplicate delta rows"):
        DeltaResult(
            result.matrix,
            result.cells,
            (result.rows[0], result.rows[0]),
            result.footprint,
            result.statistics(),
        )
    with pytest.raises(ValueError, match="ordered"):
        analyze(2, 1, asks=(1, 2), bids=(0, 0)).footprint_delta.__class__(
            result.matrix,
            tuple(reversed(analyze(2, 1, (1, 2), (0, 0)).footprint_delta.cells)),
            result.rows,
            result.footprint,
            result.statistics(),
        )
    other = analyze(1, 1, asks=(4,), bids=(1,))
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
            graph.stacked_imbalances,
            graph.absorption,
            other.footprint_delta,
        )


def test_pipeline_integration_creates_delta_result_after_absorption():
    graph = make_graph(2, 2)
    assert graph.absorption is not None
    assert graph.footprint_delta is not None
    assert graph.delta_statistics().cells == 4
    assert graph.cell_delta(graph.matrix_cell(0, 0).cell_id) is not None
