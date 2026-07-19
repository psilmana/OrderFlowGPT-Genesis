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
    ImageFrame,
    ObjectType,
    PriceAxis,
    PriceAxisDetector,
    SequentialObjectDetectionPipeline,
    TimeAxis,
    TimeAxisDetector,
    TimeAxisDetectorConfig,
    Viewport,
    WorkspaceLayout,
)


def make_time_axis_frame(
    width: int,
    height: int,
    chart: BoundingBox,
    axis_height: int,
    axis_offset: int = 0,
    axis_x_offset: int = 0,
    low_contrast: bool = False,
) -> ImageFrame:
    pixels = bytearray([235] * (width * height))
    chart_value = 35
    axis_value = 70 if low_contrast else 210
    for y in range(chart.y, chart.bottom):
        for x in range(chart.x, chart.right):
            pixels[y * width + x] = chart_value
    axis = BoundingBox(
        chart.x + axis_x_offset, chart.bottom + axis_offset, chart.width, axis_height
    )
    for y in range(axis.y, min(axis.bottom, height)):
        for x in range(max(0, axis.x), min(axis.right, width)):
            pixels[y * width + x] = axis_value
    if not low_contrast:
        for x in range(
            axis.x + 8,
            min(axis.right - 2, axis.x + axis.width - 3),
            max(12, axis.width // 12),
        ):
            for y in range(
                axis.y + 3, min(axis.bottom - 2, axis.y + axis.height - 3, height)
            ):
                pixels[y * width + x] = 25
        for y in (axis.y, axis.bottom - 1):
            if 0 <= y < height:
                for x in range(max(0, axis.x), min(axis.right, width)):
                    pixels[y * width + x] = 120
    return ImageFrame(
        data=bytes(pixels),
        width=width,
        height=height,
        pixel_format="GRAY",
        captured_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
        source="synthetic-time-axis",
        frame_id=f"time-axis-{width}x{height}-{axis_offset}-{axis_x_offset}",
    )


def make_context(
    frame: ImageFrame, chart: BoundingBox, workspace: BoundingBox | None = None
) -> DetectionContext:
    processed = DeterministicImagePreprocessor().preprocess(frame)
    layout = WorkspaceLayout(
        workspace_id="workspace-1",
        frame_id=frame.frame_id,
        bounds=workspace or BoundingBox(0, 0, frame.width, frame.height),
        chart_region=ChartRegion(chart, 0.9),
        price_axis=PriceAxis(BoundingBox(1, 1, 1, 1), 0.0),
        time_axis=TimeAxis(BoundingBox(1, 1, 1, 1), 0.0),
        viewport=Viewport(chart, 0.9),
        bottom_panels=(BottomPanel(BoundingBox(1, frame.height - 2, 1, 1), 0.1),),
    )
    return DetectionContext(processed, layout)


def test_time_axis_detector_successful_detection():
    chart = BoundingBox(60, 40, 420, 300)
    result = TimeAxisDetector().detect(
        make_context(make_time_axis_frame(640, 420, chart, 36), chart)
    )

    assert result.detected_object is not None
    assert result.region is not None
    assert result.detected_object.object_type == ObjectType.TIME_AXIS
    assert result.region.y >= chart.bottom
    assert result.region.width == chart.width
    assert result.confidence >= 0.35
    assert set(result.detected_object.metadata) == {
        "estimated_height",
        "edge_density",
        "projection_score",
        "horizontal_alignment_score",
    }


@pytest.mark.parametrize(
    ("width", "height", "chart", "axis_height"),
    [
        (160, 120, BoundingBox(20, 15, 95, 75), 10),
        (1920, 1080, BoundingBox(180, 120, 1300, 760), 80),
        (900, 620, BoundingBox(90, 70, 500, 340), 42),
    ],
)
def test_time_axis_detector_handles_sizes_and_chart_shapes(
    width, height, chart, axis_height
):
    result = TimeAxisDetector().detect(
        make_context(make_time_axis_frame(width, height, chart, axis_height), chart)
    )

    assert result.detected_object is not None
    assert result.region is not None
    assert result.region.y >= chart.bottom


def test_time_axis_detector_rejects_empty_and_low_contrast_images():
    chart = BoundingBox(60, 40, 420, 300)
    empty = ImageFrame(bytes([128] * (640 * 420)), 640, 420, "GRAY", source="empty")

    assert TimeAxisDetector().detect(make_context(empty, chart)).detected_object is None
    assert (
        TimeAxisDetector()
        .detect(
            make_context(
                make_time_axis_frame(640, 420, chart, 36, low_contrast=True), chart
            )
        )
        .detected_object
        is None
    )


def test_time_axis_detector_rejects_incorrect_location_overlap_and_alignment():
    chart = BoundingBox(60, 40, 420, 300)

    assert (
        TimeAxisDetector()
        .detect(
            make_context(
                make_time_axis_frame(640, 420, chart, 36, axis_offset=50), chart
            )
        )
        .detected_object
        is None
    )
    assert (
        TimeAxisDetector()
        .detect(
            make_context(
                make_time_axis_frame(640, 420, chart, 36), BoundingBox(60, 40, 420, 310)
            )
        )
        .detected_object
        is None
    )
    assert (
        TimeAxisDetector()
        .detect(
            make_context(
                make_time_axis_frame(640, 420, chart, 36, axis_x_offset=10), chart
            )
        )
        .detected_object
        is None
    )


def test_time_axis_detector_rejects_bad_confidence_config():
    with pytest.raises(ValueError, match="max_height_ratio"):
        TimeAxisDetectorConfig(min_height_ratio=0.2, max_height_ratio=0.1)
    with pytest.raises(ValueError, match="min_confidence"):
        TimeAxisDetectorConfig(min_confidence=1.2)
    with pytest.raises(ValueError, match="alignment_tolerance"):
        TimeAxisDetectorConfig(alignment_tolerance=-1)


def test_time_axis_pipeline_detection_graph_and_debug_overlay(tmp_path):
    chart = BoundingBox(60, 40, 420, 300)
    frame = make_time_axis_frame(640, 420, chart, 36)
    context = make_context(frame, chart)

    graph = SequentialObjectDetectionPipeline(
        DetectorRegistry((TimeAxisDetector(), PriceAxisDetector()))
    ).run(context)
    assert isinstance(graph, DetectionGraph)
    assert [obj.object_type for obj in graph.objects] == [ObjectType.TIME_AXIS]

    result = TimeAxisDetector(TimeAxisDetectorConfig(debug_overlay=True)).detect(
        context
    )
    assert result.debug_overlay is not None
    assert result.debug_overlay.data.startswith(b"\x89PNG\r\n\x1a\n")
    path = tmp_path / "time-axis.png"
    result.debug_overlay.save_png(str(path))
    assert path.read_bytes().startswith(b"\x89PNG")
