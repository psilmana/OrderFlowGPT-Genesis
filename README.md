# OrderFlowGPT Genesis

OrderFlowGPT Genesis is the architecture foundation for deterministic order-flow analysis and vision-driven workspace understanding. The package defines stable domain primitives, validation rules, an in-memory analysis pipeline, the Vision Foundation, Workspace Detection contracts, and the Milestone 5 deterministic chart detection framework, and the Milestone 6 Vision Object Detection Foundation, and Milestone 7 deterministic Price Axis Detector, and Milestone 8 deterministic Time Axis Detector, and Milestone 9 deterministic Footprint Grid Detector, and Milestone 10 deterministic Footprint Cell Detector, and Milestone 11 logical Footprint Cell Coordinate System, and Milestone 12 Footprint Cell Classification, and Milestone 13 OCR Foundation contracts, and Milestone 15 Footprint Semantic Interpretation, and Milestone 16 immutable Footprint Matrix Builder, and Milestone 17 deterministic single-cell Footprint Imbalance Detection, and Milestone 18 deterministic Stacked Footprint Imbalance Detection, and Milestone 19 deterministic Absorption Detection, and Bundle 2 immutable Auction Market Theory that future detectors and AI components can extend without changing the foundational architecture.

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
- Immutable Milestone 13 OCR objects: `OCRRequest`, `OCRResult`, `OCRWord`, `OCRLine`, `OCRPage`, `OCRRegion`, `OCRConfiguration`, and `OCRMetadata`, plus `OCREngine`, `OCRPipeline`, and `OCRProvider` contracts. Genesis can now build OCR requests from classified semantic cell regions, run the deterministic `DummyOCREngine`, attach raw `ocr_results` to `DetectionGraph`, and expose raw lookup helpers `words()`, `lines()`, `text()`, `average_confidence()`, `region_text(role)`, and `lookup(cell_id)`. No external OCR provider is implemented; Genesis performs no numeric interpretation, bid/ask recognition, volume parsing, delta calculation, AI, ML, networking, or cloud OCR. Milestone 14 introduces OCR Post Processing. Milestone 15 introduces Footprint Semantic Interpretation.

- Immutable Milestone 15 footprint semantic interpretation objects: `NumericValue`, `ParsedValue`, `FootprintSemanticType`, `FootprintValue`, `FootprintCellData`, `FootprintInterpretation`, `InterpretationResult`, and `InterpretationWarning`, plus `SemanticMapper`, `FootprintInterpreter`, and `InterpretationPipeline` contracts. Genesis now maps validated parsed numeric values from classified cell regions into bid volume, ask volume, delta, total volume, empty, unknown, and invalid semantic categories, exposes cell-level helpers (`bid()`, `ask()`, `delta()`, `total_volume()`, `is_complete()`, `is_empty()`, `missing_fields()`, and `warnings()`), and exposes graph lookup helpers for interpreted cells and values. It still does not detect imbalances, absorption, stacked imbalances, unfinished auctions, calculate signals, or generate trades. Milestone 16 builds the immutable footprint matrix.
- Immutable Milestone 16 footprint matrix objects: `FootprintMatrix`, `MatrixRow`, `MatrixCell`, `MatrixPosition`, `MatrixDimensions`, and `MatrixStatistics`, plus `FootprintMatrixBuilder`. Genesis now constructs a canonical ordered two-dimensional matrix after semantic interpretation, stores it on `DetectionGraph`, exposes row/column/cell/neighbor lookup helpers, and reports structural statistics only.
- Immutable Milestone 21 volume cluster objects: `VolumeClusterType`, `VolumeClusterConfiguration`, `VolumeCluster`, `VolumeClusterResult`, and `VolumeClusterStatistics`, plus `VolumeClusterAnalyzer`. Genesis classifies individual matrix-cell total volume only and exposes volume-cluster lookup/statistics helpers on `DetectionGraph`. No Point of Control, HVN/LVN, market profile, auction logic, market bias, trading logic, or trading signals are implemented. Milestone 22 introduces Point of Control (POC).
- Immutable Milestone 17 footprint imbalance objects: `ImbalanceType`, `ImbalanceSide`, `ImbalanceConfiguration`, `FootprintImbalance`, `FootprintImbalanceResult`, and `ImbalanceStatistics`, plus `FootprintImbalanceDetector`. Genesis detects only individual ask and bid imbalances from the immutable `FootprintMatrix`, compares ask volume with bid volume one row below and bid volume with ask volume one row above, stores results on `DetectionGraph`, and exposes imbalance lookup/statistics helpers. It performs no stacked imbalance detection, absorption detection, unfinished auction detection, POC detection, signal generation, strategy code, trading recommendations, AI, ML, OCR changes, or OpenCV additions. Milestone 18 introduces stacked imbalance detection. Milestone 19 introduces deterministic absorption detection.
- Immutable Bundle 2 Auction Market Theory objects: `DevelopingPointOfControl`, `DevelopingValueArea`, `UnfinishedAuction`, and `Excess`, plus their configurations, statistics, results, analyzers/detectors, graph helpers, and pipeline integration. Genesis now supports Developing POC, Developing Value Area, Unfinished Auctions, Excess High, and Excess Low using only immutable `FootprintMatrix` values and existing market-profile results. It still does not implement Poor High, Poor Low, Single Prints, Naked POC, trading signals, prediction, order execution, OCR changes, AI, ML, networking, threading, async, globals, randomness, or vendor-specific logic. Bundle 3 begins advanced Auction continuation analysis.
- Immutable Milestone 19 absorption objects: `AbsorptionType`, `AbsorptionSide`, `AbsorptionConfiguration`, `FootprintAbsorption`, `AbsorptionResult`, and `AbsorptionStatistics`, plus `FootprintAbsorptionDetector`. Genesis now detects deterministic absorption observations from the immutable footprint matrix and existing imbalance pressure, stores results on `DetectionGraph`, and exposes absorption lookup/statistics helpers. It performs no AI, ML, OCR changes, OpenCV additions, networking, async processing, threading, globals, randomness, vendor-specific logic, trading signals, entries, exits, or recommendations.
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

This repository is at Milestone 21. It is suitable for deterministic local analysis, test fixtures, in-memory vision foundation workflows, and side-effect-free preprocessing pipeline composition, deterministic chart-region detection, object-detection pipeline composition, deterministic price-axis region detection, and deterministic time-axis region detection. OCR is intentionally not implemented yet because Milestones 7 and 8 only establish reliable axis-region geometry; reading price numbers or timestamps is a later semantic/OCR concern. Milestone 9 detects only the footprint grid rectangle. Milestone 10 detects only footprint-cell geometry. Milestone 11 introduced the complete Cell Coordinate System. Milestone 12 introduces configurable logical cell-region classification so Genesis understands each cell's semantic layout while still performing no OCR, no text recognition, no numerical interpretation, no bid/ask value recognition, no volume recognition, and no delta calculation. Milestone 13 introduces OCR Foundation. Milestone 15 gives validated footprint numbers market meaning, Milestone 16 organizes those interpreted cells into the immutable footprint matrix, Milestone 17 detects single-cell imbalances, Milestone 18 detects stacked imbalances, Milestone 19 detects deterministic absorption observations, Milestone 20 computes deterministic delta values, and Milestone 21 classifies individual cell volume clusters while still making no trading decisions. It does not connect to brokers, exchanges, live data feeds, screen capture services, storage systems, native computer-vision runtimes, or language-model providers.

### Milestone 20 — Deterministic Delta Analysis

Milestone 20 adds deterministic Delta Analysis over the immutable `FootprintMatrix`. It reads only previously interpreted matrix values and computes `ask_volume - bid_volume` for each cell, row aggregates, whole-footprint aggregates, and immutable statistics. It does not inspect images, perform OCR, parse strings, or introduce AI/ML behavior.

Milestone 21 adds deterministic Volume Cluster Analysis over the immutable `FootprintMatrix`. It classifies individual cell volume only as `HIGH_VOLUME`, `LOW_VOLUME`, or `NORMAL_VOLUME` using deterministic percentile thresholds and a minimum-volume guard. It does not implement Point of Control, HVN/LVN, market profile, auction logic, market bias, trading logic, or trading signals. Milestone 22 introduces Point of Control (POC).

Delta Analysis intentionally computes delta values only. It does not perform divergence detection, trend analysis, market prediction, market bias, or trading signals.

### Genesis Market Profile Core (Bundle 1)

Genesis now includes the immutable Market Profile Core foundation. The core operates only on existing immutable footprint models, including `FootprintMatrix`, `VolumeClusterResult`, and footprint delta outputs; it does not inspect images, run OCR, parse strings, use networking, or introduce AI/ML behavior.

Supported deterministic analyses:

- Session Point of Control (POC): the matrix row with the highest traded volume, with deterministic lowest-row tie-breaking.
- High Volume Nodes (HVN): one-row nodes selected by configurable percentile thresholds; nearby nodes are not merged and zones are not created.
- Low Volume Nodes (LVN): one-row nodes selected by configurable percentile thresholds; zones are not created.
- Value Area (VAH / VAL): default 70% value area calculated from the POC by expanding to the larger adjacent row volume deterministically.

Still not implemented:

- Developing POC
- Developing Value Area
- Composite Profile
- Auction Theory
- Market prediction
- Trading signals

Bundle 2 begins Auction Market Theory on top of this immutable Market Profile Core.
