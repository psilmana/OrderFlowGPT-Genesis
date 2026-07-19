# OrderFlowGPT Genesis

OrderFlowGPT Genesis is the architecture foundation for deterministic order-flow analysis and vision-driven workspace understanding. The package defines stable domain primitives, validation rules, an in-memory analysis pipeline, the Vision Foundation, Workspace Detection contracts, and the Milestone 5 deterministic chart detection framework, and the Milestone 6 Vision Object Detection Foundation, and Milestone 7 deterministic Price Axis Detector, and Milestone 8 deterministic Time Axis Detector, and Milestone 9 deterministic Footprint Grid Detector, and Milestone 10 deterministic Footprint Cell Detector, and Milestone 11 logical Footprint Cell Coordinate System, and Milestone 12 Footprint Cell Classification that future detectors and AI components can extend without changing the foundational architecture.

## Current scope

The repository currently delivers:

- A frozen package layout under `src/orderflowgpt_genesis`.
- Immutable domain models for trades, order-book levels, market snapshots, and analysis results.
- A deterministic analyzer that computes spread, mid-price, imbalance, trade bias, and confidence.
- Vision Foundation abstractions for image frames, capture/replay interfaces, bounded image caching, scene graph skeletons, and workspace detection contracts.
- Image Preprocessing contracts for grayscale, HSV, Gaussian blur, adaptive threshold, Canny edges, morphology, ROI extraction, image pyramids, and zoom normalization.
- A production `ChartDetector` that locates the main trading chart from a `ProcessedFrame` using deterministic luminance edge maps, histogram projections, geometric validation, confidence scoring, and optional PNG debug overlays.
- `DetectionResult`, `Detector`, `LayoutBuilder`, and `WorkspaceLayout` contracts so layout assembly consumes detector outputs instead of performing detection directly.
- Immutable Milestone 6 object-detection contracts: `DetectedObject`, `ObjectId`, `ObjectType`, `DetectionConfidence`, `DetectionSource`, `DetectionGraph`, `DetectionContext`, `DetectorRegistry`, `ObjectDetector`, `ObjectDetectionPipeline`, and `SequentialObjectDetectionPipeline`.
- A real `PriceAxisDetector` that uses deterministic geometry, luminance transitions, and edge/projection scores to return an `ObjectType.PRICE_AXIS` `DetectedObject` for the vertical scale immediately to the right of the chart. It intentionally detects only the axis region, not price numbers.
- A real `TimeAxisDetector` that uses deterministic geometry, horizontal edge density, brightness transitions, projection scoring, and chart alignment checks to return an `ObjectType.TIME_AXIS` `DetectedObject` for the horizontal time scale immediately below the chart. It intentionally detects only the axis region, not timestamps.
- A real `FootprintGridDetector` that uses deterministic projection analysis, edge density, histogram-style line evidence, connected grid-line regularity, and workspace/chart/axis containment validation to return an `ObjectType.FOOTPRINT_GRID` `DetectedObject`. It detects only the rectangular grid where footprint cells exist; it does not detect bid/ask numbers, volume, delta, imbalance, OCR text, or AI-derived semantics.
- A real `FootprintCellDetector` that uses the detected footprint grid and deterministic grid-line/projection analysis to emit ordered `ObjectType.FOOTPRINT_CELL` geometry for every cell. It identifies only cell rectangles and never reads numbers, performs OCR, classifies bid/ask, recognizes volume, or calculates delta.
- Immutable Milestone 11 coordinate objects: `CellCoordinate`, `GridCoordinateSystem`, `CellReference`, and `CoordinateMapper`. Genesis now maps every detected footprint cell to deterministic `row_index`, `column_index`, `cell_id`, and `grid_id` metadata, exposes ordered cell lookup helpers, and attaches the grid coordinate system to `DetectionGraph` when footprint cells are present. No OCR is performed and no numbers are interpreted.
- Immutable Milestone 12 cell-classification objects: `CellSemanticRole`, `CellRegion`, `CellLayout`, `CellLayoutAnalyzer`, and `CellClassification`. Genesis now subdivides every mapped footprint cell into configurable logical semantic regions, exposes bid/ask/center/role lookup helpers, validates non-overlapping in-cell geometry, and attaches `cell_classifications` to `DetectionGraph` without changing detector contracts. This performs no OCR, text recognition, numerical interpretation, bid/ask value recognition, delta calculation, AI, ML, or OpenCV processing.
- Placeholder object detectors for footprints, volume profiles, big trades, and absorption. They deliberately return empty `DetectionResult[DetectedObject]` instances and perform no OCR, ML, AI, capture, networking, threading, or side effects.
- Project documentation, release notes, changelog entries, and automated tests.

## Quick start

```bash
python -m pip install -e .[dev]
pytest
```

## Analysis example

```python
from decimal import Decimal
from orderflowgpt_genesis import MarketSnapshot, OrderBookLevel, OrderFlowAnalyzer, Trade

snapshot = MarketSnapshot(
    symbol="ES",
    bids=(OrderBookLevel(Decimal("5000.00"), Decimal("12")),),
    asks=(OrderBookLevel(Decimal("5000.25"), Decimal("10")),),
    trades=(Trade(price=Decimal("5000.25"), quantity=Decimal("3"), side="buy"),),
)

result = OrderFlowAnalyzer().analyze(snapshot)
print(result.bias, result.confidence)
```

## Vision Foundation example

```python
from orderflowgpt_genesis import ImageCache, ImageFrame, InMemoryFrameReplay

frame = ImageFrame(data=b"raw pixels", width=1920, height=1080, pixel_format="RGB")
cache = ImageCache(max_items=32)
cache.put(frame)

replay = InMemoryFrameReplay((frame,))
for replayed_frame in replay.frames():
    assert cache.get(replayed_frame.frame_id) == replayed_frame
```

## Image Preprocessing example

```python
from orderflowgpt_genesis import (
    BoundingBox,
    DeterministicImagePreprocessor,
    ImageFrame,
    PreprocessingConfig,
    RegionOfInterest,
)

frame = ImageFrame(data=b"raw pixels", width=1920, height=1080, pixel_format="RGB")
config = PreprocessingConfig(
    roi_regions=(RegionOfInterest("chart", BoundingBox(0, 0, 1280, 720)),),
    pyramid_scales=(1.0, 0.5, 0.25),
    zoom_normalization_scale=1.25,
)

processed = DeterministicImagePreprocessor().preprocess(frame, config)
assert processed.grayscale.pixel_format == "GRAY"
assert processed.hsv.pixel_format == "HSV"
assert processed.roi_frames["chart"].width == 1280
```

## Chart detection example

```python
from orderflowgpt_genesis import (
    ChartDetector,
    DeterministicImagePreprocessor,
    ImageFrame,
    LayoutBuilder,
)

gray_pixels = bytes([32] * (1280 * 720))
frame = ImageFrame(data=gray_pixels, width=1280, height=720, pixel_format="GRAY")
processed = DeterministicImagePreprocessor().preprocess(frame)
result = ChartDetector().detect(processed)
layout = LayoutBuilder().build(processed, result)

if layout.chart_region is not None:
    print(layout.chart_region, layout.chart_confidence)
```

## Object detection foundation example

```python
from orderflowgpt_genesis import (
    DetectorRegistry,
    PriceAxisDetector,
    SequentialObjectDetectionPipeline,
    TimeAxisDetector,
)

registry = DetectorRegistry().add(PriceAxisDetector()).add(TimeAxisDetector())
pipeline = SequentialObjectDetectionPipeline(registry)
# pipeline.run(context) returns a validated DetectionGraph.
# Milestone 7 PriceAxisDetector runs before Milestone 8 TimeAxisDetector,
# followed by Milestone 9 FootprintGridDetector and Milestone 10 FootprintCellDetector when registered.
# The graph receives chart, PRICE_AXIS, TIME_AXIS, FOOTPRINT_GRID, and FOOTPRINT_CELL objects
# when deterministic geometry finds those regions.
```

## Architecture

The architecture is intentionally small and explicit. Domain models are immutable dataclasses, services are stateless, and validation is performed at construction time. See [docs/architecture.md](docs/architecture.md) for the complete roadmap.

## Support status

This repository is at Milestone 12. It is suitable for deterministic local analysis, test fixtures, in-memory vision foundation workflows, and side-effect-free preprocessing pipeline composition, deterministic chart-region detection, object-detection pipeline composition, deterministic price-axis region detection, and deterministic time-axis region detection. OCR is intentionally not implemented yet because Milestones 7 and 8 only establish reliable axis-region geometry; reading price numbers or timestamps is a later semantic/OCR concern. Milestone 9 detects only the footprint grid rectangle. Milestone 10 detects only footprint-cell geometry. Milestone 11 introduced the complete Cell Coordinate System. Milestone 12 introduces configurable logical cell-region classification so Genesis understands each cell's semantic layout while still performing no OCR, no text recognition, no numerical interpretation, no bid/ask value recognition, no volume recognition, and no delta calculation. Milestone 13 introduces OCR Foundation. It does not connect to brokers, exchanges, live data feeds, screen capture services, storage systems, native computer-vision runtimes, or language-model providers.
