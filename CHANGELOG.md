# Changelog

## 0.1.6 - Milestone 11

- Added immutable footprint cell coordinate system value objects and deterministic mapping from detected cell geometry to logical rows, columns, ids, and grid ids.
- Added coordinate lookup helpers and optional `DetectionGraph` coordinate-system exposure after footprint-cell detection.
- Documented that Milestone 11 performs no OCR and that Milestone 12 introduces Cell Classification.

## Unreleased

- Added Milestone 10 Footprint Cell Grid detection with immutable configuration, deterministic grid-line segmentation, cell metadata, validation, graph integration, debug overlays for grid plus cells, and documentation clarifying that OCR, bid/ask, volume, and delta are out of scope until later milestones. Milestone 11 introduces the Cell Coordinate System.
- Added Milestone 9 Footprint Grid detection with immutable configuration, deterministic projection/edge/grid-regularity scoring, workspace/chart/axis rejection, graph integration, debug overlays, and documentation clarifying that individual footprint cells are deferred to Milestone 10.
- Added Milestone 8 Time Axis detection with immutable configuration, deterministic horizontal geometry/edge/projection scoring, chart-alignment validation, object graph integration, debug overlays, and documentation clarifying that timestamp OCR is intentionally deferred. Milestone 9 begins Footprint Grid Detection.
- Added Milestone 7 Price Axis detection with immutable configuration, deterministic geometry/edge/projection scoring, object graph integration, debug overlays, and documentation clarifying that OCR is intentionally deferred until after region detection.
- Added Milestone 4 Workspace Detection contracts for workspace layouts, chart regions, price axes, time axes, bottom panels, toolbars, status bars, viewports, and future layout detector adapters.

### Added

- Added the Milestone 2 Vision Foundation with image frame abstractions, capture and replay interfaces, an in-memory replay implementation, a bounded image cache, scene graph skeletons, and workspace detection contracts.
- Added Milestone 3 Image Preprocessing contracts and an in-memory deterministic preprocessor for grayscale, HSV, Gaussian blur, adaptive threshold, Canny edges, morphology, ROI extraction, image pyramids, and zoom normalization.
- Added Milestone 5 Chart Detection with a deterministic `ChartDetector`, shared `Detector` and `DetectionResult` contracts, PNG debug overlays, `LayoutBuilder` integration, false-positive coverage, boundary tests, and multi-size integration tests.

## 0.1.0 - 2026-07-18

### Added

- Froze the Milestone 1 package architecture for deterministic order-flow analysis.
- Added immutable domain models for order-book levels, trades, market snapshots, and analysis results.
- Added a stateless analyzer for spread, mid-price, imbalance, trade pressure, bias, and confidence.
- Added architecture documentation, release notes, and automated tests.

## 0.1.1 - Milestone 6 Vision Object Detection Foundation

- Added immutable object-detection contracts for detected object ids, supported object types, confidence, detection source metadata, and frame-scoped detected objects.
- Added `DetectionGraph` validation for unique object ids plus valid parent and child references within one frame.
- Added `DetectionContext`, immutable `DetectorRegistry`, and `SequentialObjectDetectionPipeline` so future detectors receive only a processed frame, workspace layout, and configuration.
- Added placeholder object detectors for price axes, time axes, footprint cells, volume profiles, big trades, and absorption; these intentionally return empty `DetectionResult[DetectedObject]` values with no computer vision, OCR, ML, AI, capture, networking, or side effects.
- Documented how Milestone 6 extends the prior roadmap without replacing existing workspace layout or chart detection contracts.
