# Architecture Roadmap

## Purpose

OrderFlowGPT Genesis grows through narrow milestones. Milestone 1 froze the deterministic order-flow analysis core. Milestone 2 added the Vision Foundation needed to reason about captured screen frames and detected workspace structure. Milestone 3 added image preprocessing contracts that convert an `ImageFrame` into a `ProcessedFrame`. Milestone 5 added the first real vision detector for locating the main ATAS trading chart. Milestone 6 adds the contract-only Vision Object Detection Foundation that future semantic detectors will use. Milestone 7 implements the first real semantic detector: deterministic Price Axis detection. Milestone 8 implements the second real semantic detector: deterministic Time Axis detection.

## Package boundaries

- `orderflowgpt_genesis.models` owns immutable market domain objects and validation.
- `orderflowgpt_genesis.analysis` owns stateless order-flow analysis behavior.
- `orderflowgpt_genesis.vision` owns frame abstractions, vision-facing interfaces, in-memory frame replay, image caching, scene graph skeletons, workspace detection contracts, image preprocessing configuration, processed frame outputs, detector contracts, chart detection, debug overlays, layout assembly, object-detection contracts, detector registration, and object-detection pipeline orchestration.
- `orderflowgpt_genesis.__init__` exposes the supported public API.

No current module performs network I/O, file I/O, broker access, exchange access, language-model calls, image capture side effects, persistence, or serialization.

## Milestone 1: Deterministic analysis core

### Domain model

- `OrderBookLevel` represents a positive price and positive displayed quantity.
- `Trade` represents a positive price, positive quantity, and aggressor side of `buy` or `sell`.
- `MarketSnapshot` represents one symbol, at least one bid, at least one ask, and optional completed trades.
- `AnalysisResult` represents the computed mid-price, spread, book imbalance, net trade quantity, directional bias, and confidence.

### Analysis algorithm

1. Select the highest bid as the best bid.
2. Select the lowest ask as the best ask.
3. Reject crossed or locked snapshots before analysis.
4. Compute spread as best ask minus best bid.
5. Compute mid-price as the average of best bid and best ask.
6. Compute book imbalance as `(bid quantity - ask quantity) / total visible quantity`.
7. Compute net trade quantity by adding buy quantity and subtracting sell quantity.
8. Compute trade imbalance as net trade quantity divided by total trade quantity.
9. Average book imbalance and trade imbalance into a combined signal.
10. Classify the signal as bullish above `0.10`, bearish below `-0.10`, and neutral otherwise.
11. Set confidence to the absolute combined signal capped at `1.0`.

## Milestone 2: Vision Foundation

Milestone 2 implements the approved Vision Foundation only:

- `ImageFrame` is the normalized frame abstraction for raw image bytes, dimensions, pixel format, capture timestamp, source name, and frame identity.
- `FrameCapture` is the capture interface for future adapters that can provide one frame at a time.
- `FrameReplay` is the replay interface for deterministic frame sequences.
- `InMemoryFrameReplay` is a side-effect-free replay implementation for tests and local pipelines.
- `ImageCache` is a bounded in-memory least-recently-used cache keyed by frame id.
- `BoundingBox`, `SceneNode`, and `SceneGraph` provide the scene graph skeleton for visual element relationships.
- `WorkspaceDetection` and `WorkspaceDetector` define the workspace detection result and detector interface.

Milestone 2 deliberately excludes serialization, persistence, broker/exchange adapters, live screen-capture implementations, streaming infrastructure, and model-assisted interpretation.

## Milestone 3: Image Preprocessing

Milestone 3 implements a side-effect-free preprocessing pipeline shape:

```text
Image
  ↓
Preprocessing
  ↓
ProcessedFrame
```

- `PreprocessingConfig` validates Gaussian blur, adaptive threshold, Canny edge, morphology, image pyramid, zoom normalization, and ROI extraction parameters.
- `RegionOfInterest` names a bounded frame region to extract after morphology.
- `ImagePyramidLevel` represents one multi-scale image output with a scale, dimensions, and derived image bytes.
- `ProcessedFrame` groups the source image, grayscale image, HSV image, Gaussian blur output, adaptive threshold output, Canny edge output, morphology output, ROI frames, image pyramid levels, and zoom-normalized image.
- `ImagePreprocessor` defines the preprocessing interface for future native computer-vision adapters.
- `DeterministicImagePreprocessor` provides a no-I/O in-memory implementation for deterministic tests, local pipelines, and adapter contract development.

Milestone 3 deliberately excludes OpenCV bindings, GPU acceleration, model-assisted interpretation, persistence, serialization, live capture, and workspace-specific detector implementations.

## Milestone 5: First Real Vision Detector – Chart Detection

Milestone 5 implements the approved detector flow:

```text
ProcessedFrame
  ↓
ChartDetector
  ↓
DetectionResult
  ↓
LayoutBuilder
  ↓
WorkspaceLayout
```

- `Detector` defines the common `detect(frame: ProcessedFrame) -> DetectionResult` interface for future detectors.
- `DetectionResult` wraps every detector output with an optional detected region, confidence, reason, detector name, and optional PNG debug overlay. Raw rectangles are not returned directly.
- `ChartDetector` is a deterministic production detector for the main trading chart. It converts source pixels to luminance, computes an edge map, combines connected-component analysis with horizontal and vertical histogram projection analysis, validates candidates by size and area, scores edge density, and returns a confidence-rated result.
- `DebugOverlay` stores PNG bytes and can save overlays to disk. The overlay draws the detected rectangle and confidence bar for local diagnostics.
- `LayoutBuilder` consumes a chart `DetectionResult` and builds a `WorkspaceLayout`; the layout object does not run detection itself.

Milestone 5 deliberately excludes OCR, machine learning, deep learning, YOLO/TensorFlow/PyTorch, price-axis detection, toolbar detection, bottom-panel detection, volume-profile detection, footprint detection, DOM detection, and mouse automation.

## Milestone 6: Vision Object Detection Foundation

Milestone 6 extends, rather than replaces, the existing workspace and chart-detection architecture:

```text
ProcessedFrame + WorkspaceLayout + Configuration
  ↓
DetectionContext
  ↓
DetectorRegistry / ObjectDetectionPipeline
  ↓
DetectionResult[DetectedObject]
  ↓
DetectionGraph
```

- `ObjectId`, `ObjectType`, `DetectionConfidence`, and `DetectionSource` are immutable value objects for semantic detections.
- `ObjectType` initially supports price text, price axes, time axes, time labels, candles, footprint cells, bid values, ask values, delta values, volume values, POC markers, HVNs, LVNs, big trades, icebergs, absorption, stacked imbalances, volume profiles, CVD panels, delta panels, and unknown objects.
- `DetectedObject` contains a unique id, bounding box, confidence, object type, optional parent id, optional child ids, frame id, detection source, and immutable metadata mapping.
- `DetectionGraph` stores all detected objects for one frame and validates unique ids, duplicate ids, parent references, child references, and frame alignment.
- `DetectionContext` is the only input accepted by future object detectors. It contains a `ProcessedFrame`, a `WorkspaceLayout`, and immutable configuration.
- `ObjectDetector`, `ObjectDetectionPipeline`, and `DetectorRegistry` define how semantic detectors are registered and run.
- `SequentialObjectDetectionPipeline` runs registered detectors with `PriceAxisDetector` first and `TimeAxisDetector` second when they are registered, then returns a validated graph.
- `FootprintDetector`, `VolumeProfileDetector`, `BigTradeDetector`, and `AbsorptionDetector` remain placeholders that intentionally return empty object detection results. `TimeAxisDetector` becomes a real detector in Milestone 8 while preserving the same object-detector contract.

Milestone 6 deliberately excluded real detection behavior, OpenCV, OCR, machine learning, AI calls, screen capture, external libraries, side effects, globals, threading, async execution, and networking. Its purpose was to create stable extension seams for Milestone 7.

## Milestone 7: Price Axis Detector

Milestone 7 extends the object-detection foundation with the first real `ObjectDetector` implementation. `PriceAxisDetector` receives a `DetectionContext`, uses the existing `WorkspaceLayout.chart_region` as an anchor, scans only the workspace area immediately to the right of the chart, and returns `DetectionResult[DetectedObject]` with `ObjectType.PRICE_AXIS` when deterministic geometry validates a vertical axis region.

The detector uses no OCR, AI, ML, deep learning, OpenCV, networking, capture, threading, async execution, mutable globals, or external libraries. It relies on in-memory luminance, vertical edge density, brightness-transition checks, width ratios, projection scoring, overlap rejection, and confidence validation. Its metadata records `estimated_width`, `edge_density`, and `projection_score`. Optional debug overlays draw only the price-axis rectangle and do not alter chart overlays.

OCR is intentionally not implemented in Milestone 7 because the scope is region detection only. Reading price labels requires a separate text-recognition contract and validation strategy; this milestone first makes the axis geometry stable for downstream work.

## Milestone 8: Time Axis Detector

Milestone 8 adds `TimeAxisDetector`, the second real object detector. It receives `DetectionContext`, anchors itself to `WorkspaceLayout.chart_region`, scans only the workspace area immediately below the detected chart, and returns `DetectionResult[DetectedObject]` with `ObjectType.TIME_AXIS` when deterministic geometry validates a horizontal time-axis region.

The detector uses no OCR, AI, ML, deep learning, OpenCV, networking, capture, threading, async execution, mutable globals, or external libraries. It relies on in-memory luminance, horizontal edge density, brightness-transition checks, height ratios, projection scoring, workspace containment, excessive chart-overlap rejection, and horizontal alignment validation against the chart. Its metadata records `estimated_height`, `edge_density`, `projection_score`, and `horizontal_alignment_score`. Optional debug overlays draw only the time-axis rectangle and do not alter chart or price-axis overlays.

Timestamp OCR is intentionally deferred. Milestone 8 identifies where the time axis is, not what timestamp labels say, because timestamp parsing requires a separate OCR/text-recognition contract, locale/timezone handling, error modeling, and semantic validation after deterministic region geometry is reliable. Milestone 9 begins Footprint Grid Detection.

## Extension rules for future milestones

Future milestones may add adapters, persistence, streaming, model-assisted narrative generation, concrete capture providers, concrete workspace detectors, and concrete object detectors. They must keep the Milestone 1 and Milestone 2 public contracts backward compatible unless a major version explicitly documents a breaking change.
