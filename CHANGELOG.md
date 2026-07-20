# Changelog

## 0.1.17 - Bundle 2 Auction Market Theory

- Added immutable Auction Market Theory support: Developing POC, Developing Value Area, Unfinished Auctions, Excess High, and Excess Low.
- Extended `DetectionGraph` and `SequentialObjectDetectionPipeline` with deterministic Bundle 2 results, lookup helpers, and statistics while preserving previous APIs.
- Documented that Poor High, Poor Low, Single Prints, Naked POC, trading signals, prediction, and order execution remain out of scope; Bundle 3 begins advanced Auction continuation analysis.

## 0.1.15 - Milestone 21

- Added deterministic Volume Cluster Analysis with immutable cluster type/configuration/cluster/result/statistics models, `VolumeClusterAnalyzer`, `DetectionGraph` helpers, and pipeline integration immediately after `FootprintDeltaAnalyzer`. Milestone 21 classifies individual cell volume only and explicitly excludes Point of Control, HVN/LVN, market profile, auction logic, market bias, trading logic, and trading signals. Milestone 22 introduces POC.

## 0.1.13 - Milestone 19

- Added deterministic Absorption Detection with immutable absorption type/side/configuration/detection/result/statistics models, graph and pipeline integration, lookup/statistics helpers, and tests confirming threshold behavior, validation, immutability, and deterministic no-op pipeline behavior when source footprint values do not qualify. Milestone 19 adds no AI, ML, OCR changes, OpenCV additions, networking, async processing, threading, vendor-specific logic, trading signals, entries, exits, or recommendations.

## 0.1.12 - Milestone 18

- Added deterministic Stacked Footprint Imbalance Detection with immutable stack type/configuration/detection/result/statistics models, vertical bid/ask stack detection from `FootprintImbalanceResult` plus `FootprintMatrix`, graph/pipeline integration, lookup helpers, and documentation clarifying that Milestone 18 detects stacked bid/ask imbalances only with no absorption, unfinished auction, POC, trading logic, signals, or recommendations. Milestone 19 introduces Absorption Detection.

## 0.1.11 - Milestone 17

- Added deterministic single-cell Footprint Imbalance Detection with immutable configuration, detection/result/statistics models, ask-vs-bid-below and bid-vs-ask-above comparisons from the Milestone 16 `FootprintMatrix`, graph/pipeline integration, lookup helpers, and documentation clarifying that stacked imbalance detection, absorption, unfinished auctions, POC detection, trading signals, strategy logic, and recommendations remain out of scope. Milestone 18 introduces stacked imbalance detection.

## 0.1.10 - Milestone 16

- Added the immutable Footprint Matrix Builder with `FootprintMatrix`, row/cell/position/dimension/statistics models, deterministic two-dimensional ordering, graph and pipeline integration, matrix lookup helpers, and structural statistics only. Milestone 16 adds no imbalance detection, absorption, POC detection, stacked imbalance logic, unfinished auction logic, signal generation, strategy code, or trading decisions. Milestone 17 introduces Imbalance Detection.

## 0.1.9 - Milestone 15

- Added immutable footprint semantic interpretation models, semantic mapper/interpreter/pipeline contracts, deterministic default mapping from cell roles to bid/ask/delta/total semantics, cell value helpers, graph interpretation lookups, validation, and documentation clarifying that trading decisions and imbalance/absorption detection remain out of scope until later milestones.

## 0.1.8 - Milestone 13

- Added immutable OCR Foundation models, OCR engine/pipeline/provider contracts, deterministic `DummyOCREngine`, OCR request/result validation, graph-level `ocr_results` exposure, and raw OCR lookup helpers.
- Documented that Milestone 13 implements no external OCR provider, no numeric interpretation, no bid/ask recognition, no volume parsing, no delta calculation, no AI, no ML, no networking, and that Milestone 14 introduces OCR Post Processing.

## 0.1.7 - Milestone 12

- Added immutable footprint cell classification models, configurable logical cell layouts, deterministic cell-region analysis, validation for semantic geometry, lookup helpers, and `DetectionGraph` cell-classification exposure.
- Documented that Milestone 12 performs no OCR, no text recognition, no numeric interpretation, no AI, no ML, and that Milestone 13 introduces OCR Foundation.

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

## Milestone 20 — Deterministic Delta Analysis

- Added immutable footprint delta analysis over the existing `FootprintMatrix`.
- Computes cell, row, matrix, and statistics delta values deterministically from bid and ask volumes only.
- Does not add divergence detection, trend analysis, market prediction, AI, ML, OCR, OpenCV, threading, async, networking, globals, randomness, vendor-specific logic, or trading signals.
- Milestone 21 introduces Volume Cluster Analysis.

## Unreleased

- Implemented Bundle 1 of the Genesis Market Profile Core with immutable Session Point of Control, High Volume Nodes, Low Volume Nodes, and 70% Value Area analysis.
- Extended the detection graph and sequential object detection pipeline to expose Market Profile Core results after volume cluster analysis.
- Documented that developing POC, developing value area, composite profiles, Auction Theory, market prediction, and trading signals remain out of scope; Bundle 2 begins Auction Market Theory.


### Genesis Auction Market Theory Bundle 3
Bundle 3 completes the deterministic Auction Market Theory layer. Genesis now supports Poor High, Poor Low, Single Prints, and Naked POC Tracking using only immutable footprint and point-of-control objects. After Bundle 3 the complete Auction Market Theory module is finished. Still NOT implemented: Delta Divergence, Cumulative Delta, Iceberg Detection, Exhaustion, Market Structure, AI Reasoning, and Trading Signals. Bundle 4 begins Advanced Order Flow Analytics.
