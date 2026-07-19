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
    PriceAxisDetectorConfig,
    SequentialObjectDetectionPipeline,
    TimeAxis,
    TimeAxisDetector,
    Viewport,
    WorkspaceLayout,
)


def make_price_axis_frame(
    width: int,
    height: int,
    chart: BoundingBox,
    axis_width: int,
    axis_offset: int = 0,
    low_contrast: bool = False,
) -> ImageFrame:
    pixels = bytearray([235] * (width * height))
    chart_value = 35
    axis_value = 70 if low_contrast else 210
    for y in range(chart.y, chart.bottom):
        for x in range(chart.x, chart.right):
            pixels[y * width + x] = chart_value
    axis = BoundingBox(chart.right + axis_offset, chart.y, axis_width, chart.height)
    for y in range(axis.y, axis.bottom):
        for x in range(axis.x, axis.right):
            pixels[y * width + x] = axis_value
    if not low_contrast:
        for y in range(axis.y + 6, axis.bottom, max(8, axis.height // 10)):
            for x in range(axis.x + 3, min(axis.right - 2, axis.x + axis.width - 3)):
                pixels[y * width + x] = 25
        for x in (axis.x, axis.right - 1):
            for y in range(axis.y, axis.bottom):
                pixels[y * width + x] = 120
    return ImageFrame(
        data=bytes(pixels),
        width=width,
        height=height,
        pixel_format="GRAY",
        captured_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
        source="synthetic-price-axis",
        frame_id=f"price-axis-{width}x{height}",
    )


def make_context(frame: ImageFrame, chart: BoundingBox) -> DetectionContext:
    processed = DeterministicImagePreprocessor().preprocess(frame)
    layout = WorkspaceLayout(
        workspace_id="workspace-1",
        frame_id=frame.frame_id,
        bounds=BoundingBox(0, 0, frame.width, frame.height),
        chart_region=ChartRegion(chart, 0.9),
        price_axis=PriceAxis(BoundingBox(1, 1, 1, 1), 0.0),
        time_axis=TimeAxis(BoundingBox(1, 1, 1, 1), 0.0),
        viewport=Viewport(chart, 0.9),
        bottom_panels=(BottomPanel(BoundingBox(1, frame.height - 2, 1, 1), 0.1),),
    )
    return DetectionContext(processed, layout)


def test_price_axis_detector_successful_detection():
    chart = BoundingBox(60, 40, 420, 300)
    context = make_context(make_price_axis_frame(640, 420, chart, 58), chart)

    result = PriceAxisDetector().detect(context)

    assert result.detected_object is not None
    assert result.region is not None
    assert result.detected_object.object_type == ObjectType.PRICE_AXIS
    assert (
        result.detected_object.frame_id == context.processed_frame.source_frame.frame_id
    )
    assert result.region.x >= chart.right
    assert result.region.height == chart.height
    assert result.confidence >= 0.35
    assert set(result.detected_object.metadata) == {
        "estimated_width",
        "edge_density",
        "projection_score",
    }


@pytest.mark.parametrize(
    ("width", "height", "chart", "axis_width"),
    [
        (160, 120, BoundingBox(20, 15, 95, 75), 14),
        (1920, 1080, BoundingBox(180, 120, 1300, 760), 120),
        (900, 620, BoundingBox(90, 70, 500, 340), 48),
    ],
)
def test_price_axis_detector_handles_sizes_and_chart_shapes(
    width, height, chart, axis_width
):
    result = PriceAxisDetector().detect(
        make_context(make_price_axis_frame(width, height, chart, axis_width), chart)
    )

    assert result.detected_object is not None
    assert result.region is not None
    assert result.region.x >= chart.right


def test_price_axis_detector_rejects_empty_image():
    chart = BoundingBox(60, 40, 420, 300)
    frame = ImageFrame(
        data=bytes([128] * (640 * 420)),
        width=640,
        height=420,
        pixel_format="GRAY",
        captured_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
        source="empty",
    )

    result = PriceAxisDetector().detect(make_context(frame, chart))

    assert result.detected_object is None
    assert result.region is None
    assert result.confidence == 0.0


def test_price_axis_detector_rejects_low_contrast_image():
    chart = BoundingBox(60, 40, 420, 300)

    result = PriceAxisDetector().detect(
        make_context(
            make_price_axis_frame(640, 420, chart, 58, low_contrast=True), chart
        )
    )

    assert result.detected_object is None


def test_price_axis_detector_rejects_incorrect_location():
    chart = BoundingBox(120, 40, 420, 300)
    frame = make_price_axis_frame(720, 420, chart, 56, axis_offset=90)

    result = PriceAxisDetector().detect(make_context(frame, chart))

    assert result.detected_object is None


def test_price_axis_detector_rejects_excessive_overlap_and_bad_confidence_config():
    with pytest.raises(ValueError, match="max_width_ratio"):
        PriceAxisDetectorConfig(min_width_ratio=0.2, max_width_ratio=0.1)

    with pytest.raises(ValueError, match="min_confidence"):
        PriceAxisDetectorConfig(min_confidence=1.2)


def test_price_axis_pipeline_and_detection_graph_integration():
    chart = BoundingBox(60, 40, 420, 300)
    context = make_context(make_price_axis_frame(640, 420, chart, 58), chart)
    graph = SequentialObjectDetectionPipeline(
        DetectorRegistry((TimeAxisDetector(), PriceAxisDetector()))
    ).run(context)

    assert isinstance(graph, DetectionGraph)
    assert len(graph.objects) == 1
    assert graph.objects[0].object_type == ObjectType.PRICE_AXIS


def test_price_axis_detector_generates_debug_overlay(tmp_path):
    chart = BoundingBox(60, 40, 420, 300)
    context = make_context(make_price_axis_frame(640, 420, chart, 58), chart)

    result = PriceAxisDetector(PriceAxisDetectorConfig(debug_overlay=True)).detect(
        context
    )

    assert result.debug_overlay is not None
    assert result.debug_overlay.data.startswith(b"\x89PNG\r\n\x1a\n")
    path = tmp_path / "price-axis.png"
    result.debug_overlay.save_png(str(path))
    assert path.read_bytes().startswith(b"\x89PNG")
