from dataclasses import FrozenInstanceError

import pytest

from orderflowgpt_genesis import (
    BoundingBox,
    CellReference,
    CoordinateMapper,
    DetectionGraph,
    DetectorRegistry,
    FootprintCellDetector,
    FootprintGridDetector,
    GridCoordinateSystem,
    ObjectType,
    SequentialObjectDetectionPipeline,
)
from test_footprint_cell_detector import detect, make_cell_frame, make_context


def mapped_grid(columns: int = 4, rows: int = 3):
    result = detect(columns=columns, rows=rows)
    return (
        CoordinateMapper().map_cells(result.detected_objects),
        result.detected_objects,
    )


def test_coordinate_generation_lookup_and_immutability():
    grid, cells = mapped_grid()
    assert isinstance(grid, GridCoordinateSystem)
    assert (grid.rows, grid.columns) == (3, 4)
    assert [coord.cell_id for coord in grid.cells] == [
        f"{grid.grid_id}:cell:{row}:{column}" for row in range(3) for column in range(4)
    ]
    assert grid.cell_at(1, 2).cell_id.endswith(":cell:1:2")
    assert grid.cell_by_id(grid.cell_at(2, 3).cell_id) == grid.cell_at(2, 3)
    with pytest.raises(FrozenInstanceError):
        grid.cell_at(0, 0).row_index = 99
    references = CoordinateMapper().references(cells)
    assert isinstance(references[0], CellReference)
    assert references[0].coordinate == grid.cell_at(0, 0)


def test_neighbors_row_and_column_helpers():
    grid, _ = mapped_grid()
    center = grid.cell_at(1, 1)
    assert [(cell.row_index, cell.column_index) for cell in grid.neighbors(center)] == [
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
    ]
    assert [(cell.row_index, cell.column_index) for cell in grid.row_cells(2)] == [
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
    ]
    assert [(cell.row_index, cell.column_index) for cell in grid.column_cells(0)] == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]
    assert [
        (cell.row_index, cell.column_index)
        for cell in grid.neighbors(grid.cell_at(0, 0))
    ] == [
        (1, 0),
        (0, 1),
    ]


def test_mapper_rejects_duplicate_and_missing_positions():
    _, cells = mapped_grid()
    duplicate = cells[1]
    duplicate = duplicate.__class__(
        object_id=duplicate.object_id,
        bounds=cells[0].bounds,
        confidence=duplicate.confidence,
        object_type=duplicate.object_type,
        frame_id=duplicate.frame_id,
        source=duplicate.source,
        parent_id=duplicate.parent_id,
        metadata=duplicate.metadata,
    )
    with pytest.raises(ValueError, match="duplicate cell positions"):
        CoordinateMapper().map_cells((cells[0], duplicate))
    with pytest.raises(ValueError, match="missing rows or columns"):
        CoordinateMapper().map_cells(cells[:-1])


def test_grid_validation_duplicate_ids_noncontinuous_and_dimensions():
    grid, _ = mapped_grid(columns=2, rows=2)
    duplicate_id = grid.cells[3].__class__(
        grid.cells[3].row_index, grid.cells[3].column_index, grid.cells[0].cell_id, grid
    )
    with pytest.raises(ValueError, match="duplicate cell ids"):
        GridCoordinateSystem(
            grid.grid_id,
            2,
            2,
            grid.cell_width,
            grid.cell_height,
            grid.bounds,
            (grid.cells[0], grid.cells[1], grid.cells[2], duplicate_id),
        )
    with pytest.raises(ValueError, match="cell dimensions"):
        GridCoordinateSystem(
            grid.grid_id, 2, 2, 0, grid.cell_height, grid.bounds, grid.cells
        )


def test_pipeline_and_detection_graph_expose_coordinate_system():
    chart = BoundingBox(50, 35, 430, 310)
    grid_box = BoundingBox(80, 70, 350, 240)
    frame = make_cell_frame(640, 420, chart, grid_box, 4, 3)
    graph = SequentialObjectDetectionPipeline(
        DetectorRegistry((FootprintGridDetector(), FootprintCellDetector()))
    ).run(make_context(frame, chart))
    assert isinstance(graph, DetectionGraph)
    assert graph.grid_coordinate_system is not None
    assert graph.grid_coordinate_system.cell_at(2, 3).cell_id.endswith(":cell:2:3")
    cell = next(
        obj for obj in graph.objects if obj.object_type == ObjectType.FOOTPRINT_CELL
    )
    assert {"row_index", "column_index", "cell_id", "grid_id"} <= set(cell.metadata)


def test_single_row_single_column_and_large_grids():
    assert mapped_grid(columns=5, rows=1)[0].rows == 1
    assert mapped_grid(columns=1, rows=4)[0].columns == 1
    large, _ = mapped_grid(columns=8, rows=6)
    assert len(large.cells) == 48
