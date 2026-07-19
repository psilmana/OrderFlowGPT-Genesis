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
    FootprintGridDetector,
    FootprintGridDetectorConfig,
    ImageFrame,
    ObjectType,
    PriceAxis,
    PriceAxisDetector,
    SequentialObjectDetectionPipeline,
    TimeAxis,
    TimeAxisDetector,
    Viewport,
    WorkspaceLayout,
)


def make_grid_frame(
    width: int,
    height: int,
    chart: BoundingBox,
    grid: BoundingBox,
    low_contrast: bool = False,
) -> ImageFrame:
    pixels = bytearray([235] * (width * height))
    for y in range(chart.y, chart.bottom):
        for x in range(chart.x, chart.right):
            pixels[y * width + x] = 35
    line = 60 if low_contrast else 220
    for x in range(grid.x, grid.right, max(10, grid.width // 8)):
        for y in range(grid.y, grid.bottom):
            pixels[y * width + x] = line
    for y in range(grid.y, grid.bottom, max(8, grid.height // 10)):
        for x in range(grid.x, grid.right):
            pixels[y * width + x] = line
    return ImageFrame(
        data=bytes(pixels),
        width=width,
        height=height,
        pixel_format="GRAY",
        captured_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
        source="synthetic-footprint-grid",
        frame_id=f"footprint-grid-{width}x{height}-{grid.x}-{grid.y}",
    )


def make_context(
    frame: ImageFrame,
    chart: BoundingBox,
    workspace: BoundingBox | None = None,
    price_axis: BoundingBox | None = None,
    time_axis: BoundingBox | None = None,
) -> DetectionContext:
    processed = DeterministicImagePreprocessor().preprocess(frame)
    layout = WorkspaceLayout(
        workspace_id="workspace-1",
        frame_id=frame.frame_id,
        bounds=workspace or BoundingBox(0, 0, frame.width, frame.height),
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


def test_footprint_grid_successful_detection():
    chart = BoundingBox(50, 35, 430, 310)
    grid = BoundingBox(80, 70, 350, 240)
    result = FootprintGridDetector().detect(
        make_context(make_grid_frame(640, 420, chart, grid), chart)
    )
    assert result.detected_object is not None
    assert result.region is not None
    assert result.detected_object.object_type == ObjectType.FOOTPRINT_GRID
    assert chart.x <= result.region.x < result.region.right <= chart.right
    assert chart.y <= result.region.y < result.region.bottom <= chart.bottom
    assert set(result.detected_object.metadata) == {
        "estimated_rows",
        "estimated_columns",
        "grid_width",
        "grid_height",
        "projection_score",
        "edge_density",
    }
    assert result.confidence >= 0.35


@pytest.mark.parametrize(
    ("width", "height", "chart", "grid"),
    [
        (160, 120, BoundingBox(15, 12, 120, 90), BoundingBox(28, 25, 88, 65)),
        (
            1920,
            1080,
            BoundingBox(160, 100, 1450, 820),
            BoundingBox(260, 180, 1240, 700),
        ),
        (900, 620, BoundingBox(90, 70, 620, 420), BoundingBox(140, 120, 520, 320)),
    ],
)
def test_footprint_grid_handles_small_and_large_images(width, height, chart, grid):
    result = FootprintGridDetector().detect(
        make_context(make_grid_frame(width, height, chart, grid), chart)
    )
    assert result.detected_object is not None


def test_footprint_grid_rejects_empty_low_contrast_and_incorrect_location():
    chart = BoundingBox(50, 35, 430, 310)
    empty = ImageFrame(bytes([128] * (640 * 420)), 640, 420, "GRAY", source="empty")
    assert (
        FootprintGridDetector().detect(make_context(empty, chart)).detected_object
        is None
    )
    assert (
        FootprintGridDetector()
        .detect(
            make_context(
                make_grid_frame(
                    640, 420, chart, BoundingBox(80, 70, 350, 240), low_contrast=True
                ),
                chart,
            )
        )
        .detected_object
        is None
    )
    assert (
        FootprintGridDetector()
        .detect(
            make_context(
                make_grid_frame(640, 420, chart, BoundingBox(250, 210, 80, 70)), chart
            )
        )
        .detected_object
        is None
    )


def test_footprint_grid_rejects_workspace_chart_and_axis_overlaps():
    chart = BoundingBox(50, 35, 430, 310)
    grid = BoundingBox(80, 70, 350, 240)
    frame = make_grid_frame(640, 420, chart, grid)
    with pytest.raises(
        ValueError, match="chart region must fit within workspace layout"
    ):
        make_context(frame, chart, workspace=BoundingBox(0, 0, 360, 360))
    outside_chart = BoundingBox(500, 35, 100, 90)
    outside_frame = make_grid_frame(640, 420, outside_chart, grid)
    assert (
        FootprintGridDetector()
        .detect(make_context(outside_frame, outside_chart))
        .detected_object
        is None
    )
    assert (
        FootprintGridDetector()
        .detect(make_context(frame, chart, price_axis=BoundingBox(420, 60, 40, 260)))
        .detected_object
        is None
    )
    assert (
        FootprintGridDetector()
        .detect(make_context(frame, chart, time_axis=BoundingBox(70, 260, 370, 40)))
        .detected_object
        is None
    )


def test_footprint_grid_config_validation():
    with pytest.raises(ValueError, match="min_width_ratio"):
        FootprintGridDetectorConfig(min_width_ratio=0.0)
    with pytest.raises(ValueError, match="min_confidence"):
        FootprintGridDetectorConfig(min_confidence=1.2)


def test_footprint_grid_pipeline_graph_and_debug_overlay(tmp_path):
    chart = BoundingBox(50, 35, 430, 310)
    grid = BoundingBox(80, 70, 350, 240)
    context = make_context(make_grid_frame(640, 420, chart, grid), chart)
    graph = SequentialObjectDetectionPipeline(
        DetectorRegistry(
            (FootprintGridDetector(), TimeAxisDetector(), PriceAxisDetector())
        )
    ).run(context)
    assert isinstance(graph, DetectionGraph)
    assert ObjectType.CHART in [obj.object_type for obj in graph.objects]
    assert ObjectType.FOOTPRINT_GRID in [obj.object_type for obj in graph.objects]

    result = FootprintGridDetector(
        FootprintGridDetectorConfig(debug_overlay=True)
    ).detect(context)
    assert result.debug_overlay is not None
    path = tmp_path / "footprint-grid.png"
    result.debug_overlay.save_png(str(path))
    assert path.read_bytes().startswith(b"\x89PNG")
