from datetime import datetime, timezone

import pytest

from orderflowgpt_genesis import (
    BoundingBox,
    ChartDetector,
    ChartDetectorConfig,
    DetectionResult,
    DeterministicImagePreprocessor,
    ImageFrame,
    LayoutBuilder,
)


def make_chart_frame(width: int, height: int, box: BoundingBox) -> ImageFrame:
    pixels = bytearray([238] * (width * height))
    for y in range(box.y, box.bottom):
        for x in range(box.x, box.right):
            pixels[y * width + x] = 30
    for x in range(box.x, box.right, max(8, box.width // 8)):
        for y in range(box.y, box.bottom):
            pixels[y * width + x] = 190
    for y in range(box.y, box.bottom, max(8, box.height // 6)):
        row = y * width
        for x in range(box.x, box.right):
            pixels[row + x] = 190
    return ImageFrame(
        data=bytes(pixels),
        width=width,
        height=height,
        pixel_format="GRAY",
        captured_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        source="synthetic-atas",
        frame_id=f"chart-{width}x{height}",
    )


def process(frame: ImageFrame):
    return DeterministicImagePreprocessor().preprocess(frame)


def assert_close(
    actual: BoundingBox, expected: BoundingBox, tolerance: int = 4
) -> None:
    assert abs(actual.x - expected.x) <= tolerance
    assert abs(actual.y - expected.y) <= tolerance
    assert abs(actual.right - expected.right) <= tolerance
    assert abs(actual.bottom - expected.bottom) <= tolerance


@pytest.mark.parametrize(
    ("width", "height", "box"),
    [
        (640, 480, BoundingBox(80, 60, 430, 300)),
        (1280, 720, BoundingBox(140, 90, 900, 470)),
        (320, 240, BoundingBox(38, 34, 220, 150)),
    ],
)
def test_chart_detector_locates_main_chart_across_image_sizes(width, height, box):
    result = ChartDetector().detect(process(make_chart_frame(width, height, box)))

    assert result.region is not None
    assert result.confidence >= 0.7
    assert "edge_density" in result.reason
    assert_close(result.region, box)


def test_chart_detector_returns_detection_result_not_raw_rectangle():
    frame = make_chart_frame(640, 480, BoundingBox(70, 50, 420, 300))

    result = ChartDetector().detect(process(frame))

    assert isinstance(result, DetectionResult)
    assert result.detector_name == "chart-detector"


def test_chart_detector_rejects_blank_false_positive():
    frame = ImageFrame(
        data=bytes([128] * (640 * 480)),
        width=640,
        height=480,
        pixel_format="GRAY",
        captured_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        source="blank",
    )

    result = ChartDetector().detect(process(frame))

    assert result.region is None
    assert result.confidence == 0.0


def test_chart_detector_rejects_tiny_boundary_candidate():
    frame = make_chart_frame(240, 180, BoundingBox(10, 10, 38, 30))

    result = ChartDetector().detect(process(frame))

    assert result.region is None
    assert result.confidence == 0.0


def test_chart_detector_generates_png_debug_overlay(tmp_path):
    frame = make_chart_frame(400, 300, BoundingBox(50, 40, 270, 180))
    result = ChartDetector(ChartDetectorConfig(debug_overlay=True)).detect(
        process(frame)
    )

    assert result.debug_overlay is not None
    assert result.debug_overlay.data.startswith(b"\x89PNG\r\n\x1a\n")
    overlay_path = tmp_path / "overlay.png"
    result.debug_overlay.save_png(str(overlay_path))
    assert overlay_path.read_bytes().startswith(b"\x89PNG")


def test_layout_builder_integrates_chart_detection_result():
    frame = make_chart_frame(640, 480, BoundingBox(80, 60, 430, 300))
    processed = process(frame)
    result = ChartDetector().detect(processed)

    layout = LayoutBuilder().build(processed, result)

    assert layout.frame_id == frame.frame_id
    assert layout.chart_region == result.region
    assert layout.chart_confidence == result.confidence
    assert layout.detection_reason == result.reason
