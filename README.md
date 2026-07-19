# OrderFlowGPT Genesis

OrderFlowGPT Genesis is the architecture foundation for deterministic order-flow analysis and vision-driven workspace understanding. The package defines stable domain primitives, validation rules, an in-memory analysis pipeline, the Vision Foundation, Workspace Detection contracts, and the Milestone 5 deterministic chart detection framework, and the Milestone 6 Vision Object Detection Foundation that future detectors and AI components can extend without changing the foundational architecture.

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
- Placeholder object detectors for price axes, time axes, footprints, volume profiles, big trades, and absorption. They deliberately return empty `DetectionResult[DetectedObject]` instances and perform no computer vision, OCR, ML, AI, capture, networking, threading, or side effects.
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
)

registry = DetectorRegistry().add(PriceAxisDetector())
pipeline = SequentialObjectDetectionPipeline(registry)
# pipeline.run(context) returns a validated DetectionGraph.
# Milestone 6 placeholder detectors intentionally return no objects.
```

## Architecture

The architecture is intentionally small and explicit. Domain models are immutable dataclasses, services are stateless, and validation is performed at construction time. See [docs/architecture.md](docs/architecture.md) for the complete roadmap.

## Support status

This repository is at Milestone 6. It is suitable for deterministic local analysis, test fixtures, in-memory vision foundation workflows, and side-effect-free preprocessing pipeline composition, deterministic chart-region detection, and contract-only object-detection pipeline composition. It does not connect to brokers, exchanges, live data feeds, screen capture services, storage systems, native computer-vision runtimes, or language-model providers.
