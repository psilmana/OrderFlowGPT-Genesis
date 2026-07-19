from datetime import datetime, timezone

import pytest

from orderflowgpt_genesis import (
    BottomPanel,
    BoundingBox,
    ChartRegion,
    DetectionContext,
    DetectionGraph,
    DetectorRegistry,
    DeterministicImagePreprocessor,
    FootprintCellDetector,
    FootprintCellDetectorConfig,
    FootprintGridDetector,
    ImageFrame,
    ObjectType,
    PriceAxis,
    SequentialObjectDetectionPipeline,
    TimeAxis,
    Viewport,
    WorkspaceLayout,
)


def make_cell_frame(
    width: int,
    height: int,
    chart: BoundingBox,
    grid: BoundingBox,
    columns: int,
    rows: int,
    irregular: bool = False,
) -> ImageFrame:
    pixels = bytearray([235] * (width * height))
    for y in range(chart.y, chart.bottom):
        for x in range(chart.x, chart.right):
            pixels[y * width + x] = 35
    xs = [grid.x + round(i * grid.width / columns) for i in range(columns + 1)]
    ys = [grid.y + round(i * grid.height / rows) for i in range(rows + 1)]
    xs[-1] = grid.right - 1
    ys[-1] = grid.bottom - 1
    if irregular and len(xs) > 3:
        xs[2] += max(4, grid.width // 8)
    for x in xs:
        for y in range(grid.y, grid.bottom):
            pixels[y * width + x] = 220
    for y in ys:
        for x in range(grid.x, grid.right):
            pixels[y * width + x] = 220
    return ImageFrame(
        bytes(pixels),
        width,
        height,
        "GRAY",
        datetime(2026, 7, 19, tzinfo=timezone.utc),
        "synthetic-cells",
        f"cells-{width}-{height}-{columns}-{rows}-{irregular}",
    )


def make_context(
    frame: ImageFrame,
    chart: BoundingBox,
    price_axis: BoundingBox | None = None,
    time_axis: BoundingBox | None = None,
) -> DetectionContext:
    processed = DeterministicImagePreprocessor().preprocess(frame)
    layout = WorkspaceLayout(
        workspace_id="workspace-1",
        frame_id=frame.frame_id,
        bounds=BoundingBox(0, 0, frame.width, frame.height),
        chart_region=ChartRegion(chart, 0.9),
        price_axis=PriceAxis(
            price_axis or BoundingBox(1, 1, 1, 1), 0.0 if price_axis is None else 0.8
        ),
        time_axis=TimeAxis(
            time_axis or BoundingBox(1, 1, 1, 1), 0.0 if time_axis is None else 0.8
        ),
        viewport=Viewport(chart, 0.9),
        bottom_panels=(BottomPanel(BoundingBox(1, frame.height - 2, 1, 1), 0.1),),
    )
    return DetectionContext(processed, layout)


def detect(columns: int = 4, rows: int = 3, width: int = 640, height: int = 420):
    chart = BoundingBox(50, 35, width - 160, height - 110)
    grid = BoundingBox(chart.x + 30, chart.y + 35, chart.width - 80, chart.height - 80)
    frame = make_cell_frame(width, height, chart, grid, columns, rows)
    return FootprintCellDetector().detect(make_context(frame, chart))


def test_successful_grid_segmentation_and_deterministic_ordering():
    result = detect(columns=4, rows=3)
    cells = result.detected_objects
    assert len(cells) == 12
    assert all(cell.object_type == ObjectType.FOOTPRINT_CELL for cell in cells)
    assert [
        (cell.metadata["row_index"], cell.metadata["column_index"]) for cell in cells
    ] == [(r, c) for r in range(3) for c in range(4)]
    assert all(cell.metadata["cell_width"] == cell.bounds.width for cell in cells)
    assert result.reason.startswith("segmented footprint grid")


@pytest.mark.parametrize(
    ("width", "height", "columns", "rows"),
    [(500, 360, 2, 2), (1920, 1080, 8, 6), (800, 600, 1, 4), (800, 600, 5, 1)],
)
def test_different_chart_sizes_small_large_single_column_and_single_row(
    width, height, columns, rows
):
    result = detect(columns=columns, rows=rows, width=width, height=height)
    assert len(result.detected_objects) == columns * rows


def test_irregular_spacing_rejection():
    chart = BoundingBox(50, 35, 430, 310)
    grid = BoundingBox(80, 70, 350, 240)
    frame = make_cell_frame(640, 420, chart, grid, 4, 3, irregular=True)
    result = FootprintCellDetector().detect(make_context(frame, chart))
    assert result.detected_objects == ()
    assert "irregular" in result.reason


def test_cells_outside_grid_or_axis_rejection():
    result = detect(columns=4, rows=3)
    first = result.detected_objects[0]
    chart = BoundingBox(50, 35, 430, 310)
    grid = BoundingBox(80, 70, 350, 240)
    frame = make_cell_frame(640, 420, chart, grid, 4, 3)
    axis_result = FootprintCellDetector().detect(
        make_context(frame, chart, price_axis=first.bounds)
    )
    assert axis_result.detected_objects == ()


def test_configuration_validation_and_duplicate_helper_path():
    with pytest.raises(ValueError, match="minimum_cell_width"):
        FootprintCellDetectorConfig(minimum_cell_width=0)
    with pytest.raises(ValueError, match="maximum_spacing_variation"):
        FootprintCellDetectorConfig(maximum_spacing_variation=1.5)
    with pytest.raises(ValueError, match="minimum_confidence"):
        FootprintCellDetectorConfig(minimum_confidence=-0.1)


def test_pipeline_detection_graph_integration_and_debug_overlay(tmp_path):
    chart = BoundingBox(50, 35, 430, 310)
    grid = BoundingBox(80, 70, 350, 240)
    frame = make_cell_frame(640, 420, chart, grid, 4, 3)
    context = make_context(frame, chart)
    graph = SequentialObjectDetectionPipeline(
        DetectorRegistry((FootprintCellDetector(), FootprintGridDetector()))
    ).run(context)
    assert isinstance(graph, DetectionGraph)
    types = [obj.object_type for obj in graph.objects]
    assert ObjectType.CHART in types
    assert ObjectType.FOOTPRINT_GRID in types
    assert types.count(ObjectType.FOOTPRINT_CELL) == 12

    result = FootprintCellDetector(
        FootprintCellDetectorConfig(debug_overlay=True)
    ).detect(context)
    assert result.debug_overlay is not None
    path = tmp_path / "cells.png"
    result.debug_overlay.save_png(str(path))
    assert path.read_bytes().startswith(b"\x89PNG")


def test_duplicate_coordinate_rejection(monkeypatch):
    from orderflowgpt_genesis import vision

    chart = BoundingBox(50, 35, 430, 310)
    grid = BoundingBox(80, 70, 350, 240)
    frame = make_cell_frame(640, 420, chart, grid, 4, 3)

    def duplicate_lines(luminance, width, detected_grid, vertical):
        return (80, 80, 160) if vertical else (70, 150)

    monkeypatch.setattr(vision, "_grid_line_centers", duplicate_lines)
    result = FootprintCellDetector().detect(make_context(frame, chart))
    assert result.detected_objects == ()
    assert "duplicate" in result.reason


def test_cells_outside_grid_rejection(monkeypatch):
    from orderflowgpt_genesis import vision

    chart = BoundingBox(50, 35, 430, 310)
    grid = BoundingBox(80, 70, 350, 240)
    frame = make_cell_frame(640, 420, chart, grid, 4, 3)

    def outside_lines(luminance, width, detected_grid, vertical):
        return (
            (detected_grid.x - 4, detected_grid.x + 40)
            if vertical
            else (detected_grid.y, detected_grid.y + 40)
        )

    monkeypatch.setattr(vision, "_grid_line_centers", outside_lines)
    result = FootprintCellDetector().detect(make_context(frame, chart))
    assert result.detected_objects == ()
    assert "containment" in result.reason
