from dataclasses import FrozenInstanceError

import pytest

from orderflowgpt_genesis import (
    DetectionGraph,
    FootprintMatrix,
    FootprintMatrixBuilder,
    MatrixDimensions,
    MatrixRow,
    MatrixStatistics,
    SequentialInterpretationPipeline,
)
from test_footprint_semantic_interpretation import parse_all
from test_ocr_foundation import make_graph


def interpreted_graph(columns=2, rows=2):
    graph = make_graph(columns, rows)
    interpretation = SequentialInterpretationPipeline().run(
        graph.cell_classifications, parse_all(graph.cell_classifications)
    )
    graph = DetectionGraph(
        graph.frame_id,
        graph.objects,
        graph.grid_coordinate_system,
        graph.cell_classifications,
        graph.ocr_results,
        interpretation,
        graph.parsed_values,
    )
    matrix = FootprintMatrixBuilder().build(
        graph, __import__("orderflowgpt_genesis").CoordinateMapper(), interpretation
    )
    return DetectionGraph(
        graph.frame_id,
        graph.objects,
        graph.grid_coordinate_system,
        graph.cell_classifications,
        graph.ocr_results,
        interpretation,
        graph.parsed_values,
        matrix,
    )


def test_successful_construction_and_deterministic_ordering():
    graph = interpreted_graph(3, 2)
    matrix = graph.footprint_matrix
    assert matrix is not None
    assert matrix.dimensions() == MatrixDimensions(2, 3)
    assert [cell.cell_id for cell in matrix.cells] == sorted(
        [cell.cell_id for cell in matrix.cells],
        key=lambda value: (value.split(":")[-2], value.split(":")[-1]),
    )
    assert matrix.cell(1, 2).row_index == 1
    assert graph.matrix_cell(1, 2) == matrix.cell(1, 2)


def test_single_row_single_column_and_large_matrices():
    assert interpreted_graph(4, 1).footprint_matrix.dimensions() == MatrixDimensions(
        1, 4
    )
    assert interpreted_graph(1, 4).footprint_matrix.dimensions() == MatrixDimensions(
        4, 1
    )
    large = interpreted_graph(10, 8).footprint_matrix
    assert large.statistics().total_cells == 80
    assert large.cell(7, 9).column_index == 9


def test_missing_duplicate_incorrect_order_and_dimension_validation():
    graph = interpreted_graph(2, 2)
    matrix = graph.footprint_matrix
    with pytest.raises(ValueError, match="dimensions"):
        FootprintMatrix(
            matrix.grid_id, matrix.rows[:-1], matrix.dimensions(), matrix.statistics()
        )
    with pytest.raises(ValueError, match="duplicate"):
        MatrixRow(0, (matrix.cell(0, 0), matrix.cell(0, 0)))
    with pytest.raises(ValueError, match="ordered"):
        FootprintMatrix(
            matrix.grid_id,
            tuple(reversed(matrix.rows)),
            matrix.dimensions(),
            matrix.statistics(),
        )
    with pytest.raises(ValueError, match="dimensions"):
        FootprintMatrix(
            matrix.grid_id, matrix.rows, MatrixDimensions(3, 2), matrix.statistics()
        )
    with pytest.raises(ValueError, match="cell count"):
        FootprintMatrix(
            matrix.grid_id,
            matrix.rows,
            matrix.dimensions(),
            MatrixStatistics(2, 2, 3, 0, 0, 0, 0, 0, 0, 0),
        )


def test_neighbor_row_column_and_statistics_helpers():
    matrix = interpreted_graph(3, 3).footprint_matrix
    center = matrix.cell(1, 1)
    assert matrix.row(1).cells == (
        matrix.cell(1, 0),
        matrix.cell(1, 1),
        matrix.cell(1, 2),
    )
    assert matrix.column(2) == (matrix.cell(0, 2), matrix.cell(1, 2), matrix.cell(2, 2))
    assert matrix.above(center) == matrix.cell(0, 1)
    assert matrix.below(center) == matrix.cell(2, 1)
    assert matrix.left(center) == matrix.cell(1, 0)
    assert matrix.right(center) == matrix.cell(1, 2)
    assert matrix.neighbors(center) == (
        matrix.cell(0, 1),
        matrix.cell(2, 1),
        matrix.cell(1, 0),
        matrix.cell(1, 2),
    )
    assert matrix.diagonal_neighbors(center) == (
        matrix.cell(0, 0),
        matrix.cell(0, 2),
        matrix.cell(2, 0),
        matrix.cell(2, 2),
    )
    stats = matrix.statistics()
    assert stats.rows == 3
    assert stats.columns == 3
    assert stats.total_cells == 9
    assert stats.interpreted_cells == 9
    assert stats.empty_cells == 0
    assert stats.missing_cells == 0
    assert stats.bid_cells == 9
    assert stats.ask_cells == 9
    assert stats.delta_cells == 9


def test_pipeline_and_graph_integration_immutability():
    graph = make_graph(2, 2)
    assert graph.footprint_interpretation is not None
    assert graph.footprint_matrix is not None
    assert graph.matrix_statistics().total_cells == 4
    with pytest.raises(FrozenInstanceError):
        graph.footprint_matrix.grid_id = "changed"
