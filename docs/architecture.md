# Architecture Roadmap

## Purpose

OrderFlowGPT Genesis grows through narrow milestones. Milestone 1 froze the deterministic order-flow analysis core. Milestone 2 added the Vision Foundation needed to reason about captured screen frames and detected workspace structure. Milestone 3 added image preprocessing contracts that convert an `ImageFrame` into a `ProcessedFrame`. Milestone 5 added the first real vision detector for locating the main ATAS trading chart. Milestone 6 adds the contract-only Vision Object Detection Foundation that future semantic detectors will use. Milestone 7 implements the first real semantic detector: deterministic Price Axis detection. Milestone 8 implements the second real semantic detector: deterministic Time Axis detection. Milestone 9 implements the first detector that understands internal chart structure: deterministic Footprint Grid detection. Milestone 10 segments that grid into deterministic footprint-cell geometry. Milestone 11 assigns every detected footprint cell a stable logical coordinate and deterministic cell identifier. Milestone 12 classifies the internal logical regions of every footprint cell without OCR or numeric interpretation. Milestone 15 maps validated numeric values from classified regions into footprint market semantics without trading decisions. Milestone 16 builds the immutable footprint matrix as the canonical two-dimensional representation for future analytics. Milestone 17 detects deterministic individual bid/ask imbalances from that matrix only, with no stacked imbalance, absorption, unfinished auction, POC, or trading logic. Milestone 18 introduces stacked imbalance detection. Milestone 19 adds deterministic absorption detection. Bundle 2 introduces the immutable Auction Market Theory layer: Developing POC, Developing Value Area, Unfinished Auctions, Excess High, and Excess Low. Bundle 4 introduces deterministic Advanced Order Flow Analytics: Delta Divergence Detection, Cumulative Delta, Delta Momentum, and Exhaustion Detection. Bundle 6 introduces the Genesis Trend Engine: Trend State Detection, Pullback Detection, Break of Structure, and Change of Character.

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

## Bundle 2: Immutable Auction Market Theory

Bundle 2 operates only on immutable data already produced by previous milestones, especially `FootprintMatrix`, Point of Control, Value Area, volume-node results, and `DetectionGraph`. It adds deterministic Developing POC history, Developing Value Area history with expansion/contraction/movement statistics, boundary-only Unfinished Auction detection for top and bottom rows, and Excess High/Excess Low detection from existing bid/ask values. It does not inspect images, perform OCR, parse strings, add OpenCV, use AI/ML, start threads, use async, perform networking, create trading signals, predict markets, execute orders, or add vendor-specific logic. Poor High, Poor Low, Single Prints, and Naked POC remain unimplemented. Bundle 3 begins advanced Auction continuation analysis.

## Bundle 4: Advanced Order Flow Analytics

Bundle 4 operates only on immutable objects produced by earlier milestones, especially `FootprintMatrix`, `DeltaResult`, prior order-flow analytics, and `DetectionGraph`. It adds deterministic Delta Divergence Detection, Cumulative Delta, Delta Momentum Analysis, and Exhaustion Detection, then exposes those results through graph lookup/statistics helpers and the sequential pipeline after Auction Market Theory. Bundle 4 does not inspect images, perform OCR, parse text, add OpenCV, use AI reasoning, predict markets, estimate probabilities, apply machine learning, produce trading signals, infer trades, execute orders, perform networking, start threads, use async, use randomness, or add vendor-specific logic. Bundle 5 introduces Market Structure Analysis. Bundle 6 introduces the Trend Engine and Bundle 7 introduces Session Intelligence.

## Bundle 6: Genesis Trend Engine

Bundle 6 operates only on immutable objects produced by previous bundles, especially `MarketStructureResult`, `FootprintMatrix`, and `DetectionGraph`. It adds deterministic Trend State Detection, Pullback Detection, Break of Structure (BOS), and Change of Character (CHOCH), exposes graph lookup/statistics helpers, and extends the sequential pipeline after Market Structure. Bundle 6 does not inspect images, perform OCR, modify OpenCV behavior, use AI reasoning, predict markets, emit trading signals, estimate probabilities, apply machine learning, perform networking, start threads, use async, use randomness, generate strategies, or add vendor-specific logic. Bundle 7 introduces Session Intelligence.

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
- `ObjectType` initially supports price text, price axes, time axes, time labels, candles, footprint grids, footprint cells, bid values, ask values, delta values, volume values, POC markers, HVNs, LVNs, big trades, icebergs, absorption, stacked imbalances, volume profiles, CVD panels, delta panels, and unknown objects.
- `DetectedObject` contains a unique id, bounding box, confidence, object type, optional parent id, optional child ids, frame id, detection source, and immutable metadata mapping.
- `DetectionGraph` stores all detected objects for one frame and validates unique ids, duplicate ids, parent references, child references, and frame alignment.
- `DetectionContext` is the only input accepted by future object detectors. It contains a `ProcessedFrame`, a `WorkspaceLayout`, and immutable configuration.
- `ObjectDetector`, `ObjectDetectionPipeline`, and `DetectorRegistry` define how semantic detectors are registered and run.
- `SequentialObjectDetectionPipeline` runs registered detectors with `PriceAxisDetector` first, `TimeAxisDetector` second, `FootprintGridDetector` third, and `FootprintCellDetector` fourth when they are registered, then returns a validated graph. When footprint-grid detection participates, the graph also includes the chart object from the workspace layout so downstream relationships can include Chart, Price Axis, Time Axis, Footprint Grid, and Footprint Cells.
- `FootprintDetector`, `VolumeProfileDetector`, `BigTradeDetector`, and `AbsorptionDetector` remain placeholders that intentionally return empty object detection results. `TimeAxisDetector` becomes a real detector in Milestone 8 while preserving the same object-detector contract.

Milestone 6 deliberately excluded real detection behavior, OpenCV, OCR, machine learning, AI calls, screen capture, external libraries, side effects, globals, threading, async execution, and networking. Its purpose was to create stable extension seams for Milestone 7.

## Milestone 7: Price Axis Detector

Milestone 7 extends the object-detection foundation with the first real `ObjectDetector` implementation. `PriceAxisDetector` receives a `DetectionContext`, uses the existing `WorkspaceLayout.chart_region` as an anchor, scans only the workspace area immediately to the right of the chart, and returns `DetectionResult[DetectedObject]` with `ObjectType.PRICE_AXIS` when deterministic geometry validates a vertical axis region.

The detector uses no OCR, AI, ML, deep learning, OpenCV, networking, capture, threading, async execution, mutable globals, or external libraries. It relies on in-memory luminance, vertical edge density, brightness-transition checks, width ratios, projection scoring, overlap rejection, and confidence validation. Its metadata records `estimated_width`, `edge_density`, and `projection_score`. Optional debug overlays draw only the price-axis rectangle and do not alter chart overlays.

OCR is intentionally not implemented in Milestone 7 because the scope is region detection only. Reading price labels requires a separate text-recognition contract and validation strategy; this milestone first makes the axis geometry stable for downstream work.

## Milestone 8: Time Axis Detector

Milestone 8 adds `TimeAxisDetector`, the second real object detector. It receives `DetectionContext`, anchors itself to `WorkspaceLayout.chart_region`, scans only the workspace area immediately below the detected chart, and returns `DetectionResult[DetectedObject]` with `ObjectType.TIME_AXIS` when deterministic geometry validates a horizontal time-axis region.

The detector uses no OCR, AI, ML, deep learning, OpenCV, networking, capture, threading, async execution, mutable globals, or external libraries. It relies on in-memory luminance, horizontal edge density, brightness-transition checks, height ratios, projection scoring, workspace containment, excessive chart-overlap rejection, and horizontal alignment validation against the chart. Its metadata records `estimated_height`, `edge_density`, `projection_score`, and `horizontal_alignment_score`. Optional debug overlays draw only the time-axis rectangle and do not alter chart or price-axis overlays.

Timestamp OCR is intentionally deferred. Milestone 8 identifies where the time axis is, not what timestamp labels say, because timestamp parsing requires a separate OCR/text-recognition contract, locale/timezone handling, error modeling, and semantic validation after deterministic region geometry is reliable. Milestone 9 detects only the footprint grid rectangle. Milestone 10 detects only individual footprint-cell geometry.

## Extension rules for future milestones

Future milestones may add adapters, persistence, streaming, model-assisted narrative generation, concrete capture providers, concrete workspace detectors, and concrete object detectors. They must keep the Milestone 1 and Milestone 2 public contracts backward compatible unless a major version explicitly documents a breaking change.


## Milestone 9: Footprint Grid Detector

Milestone 9 adds `FootprintGridDetector`, the first detector that looks inside the detected chart region. It receives `DetectionContext`, anchors to `WorkspaceLayout.chart_region`, and returns `DetectionResult[DetectedObject]` with `ObjectType.FOOTPRINT_GRID` when deterministic image analysis identifies the rectangular grid where future footprint cells will exist.

The detector estimates left, right, top, and bottom grid boundaries using in-memory luminance transitions, projection-style line evidence, edge density, histogram-style row/column evidence, simple clustering, geometry regularity scoring, and containment validation. It rejects zero-size candidates, grids outside the workspace layout, grids outside the chart region, candidates overlapping detected price or time axes, confidence outside `[0, 1]`, and highly irregular geometry. Metadata records `estimated_rows`, `estimated_columns`, `grid_width`, `grid_height`, `projection_score`, and `edge_density`.

Milestone 9 deliberately detects only the footprint grid rectangle. It does not detect individual footprint cells, bid/ask numbers, volume, delta, imbalance, OCR text, AI interpretations, ML objects, or OpenCV-derived features. Milestone 10 will detect individual footprint cells after this grid boundary is stable.


## Milestone 10: Footprint Cell Grid Detector

Milestone 10 adds `FootprintCellDetector`, a deterministic object detector that segments the previously detected `ObjectType.FOOTPRINT_GRID` into individual rectangular `ObjectType.FOOTPRINT_CELL` objects. It preserves prior detector contracts and extends `DetectionResult` with a tuple of detected objects so a single detector can return every cell while remaining compatible with existing single-object detectors.

The detector uses in-memory luminance transitions, projection analysis, grid-line clustering, regular spacing validation, alignment scoring, containment checks, and axis-overlap rejection. Cells are emitted deterministically from top to bottom and left to right. Metadata records `row_index`, `column_index`, `cell_id`, `grid_id`, `cell_width`, `cell_height`, `grid_width`, and `grid_height`. Optional debug overlays draw both the footprint grid rectangle and every detected cell rectangle.

Milestone 10 detects only footprint-cell geometry. It does not read numbers, perform OCR, classify bid or ask, recognize volume, calculate delta, call AI/ML systems, use OpenCV, capture screens, perform networking, or introduce mutable globals. Milestone 11 introduces the Cell Coordinate System that downstream semantic milestones can use after this geometry is stable.


## Milestone 11: Footprint Cell Coordinate System

Milestone 11 adds immutable logical coordinate objects for detected footprint cells: `CellCoordinate`, `GridCoordinateSystem`, `CellReference`, and `CoordinateMapper`. After footprint-cell detection, Genesis deterministically orders cells from top to bottom and left to right, assigns continuous row and column indices, creates stable cell ids scoped to the detected footprint grid, and records `row_index`, `column_index`, `cell_id`, and `grid_id` metadata on every footprint cell.

`GridCoordinateSystem` validates unique coordinates, unique ids, positive dimensions, continuous indexing, and deterministic ordering. It exposes logical helpers for `cell_at(row, column)`, `cell_by_id(id)`, `neighbors(cell)`, `row_cells(row)`, and `column_cells(column)`. `DetectionGraph` now exposes the optional coordinate system when footprint cells are available, without changing existing detector APIs or replacing `DetectionGraph`.

Milestone 11 performs no OCR, reads no numbers, detects no bid/ask values, classifies no volume or delta values, and uses no AI, ML, OpenCV, networking, threading, or async execution. Milestone 12 introduces Cell Classification after the coordinate system is stable.



## Milestone 12: Footprint Cell Classification

Milestone 12 adds immutable logical cell-classification models: `CellSemanticRole`, `CellRegion`, `CellLayout`, `CellLayoutAnalyzer`, and `CellClassification`. After `CoordinateMapper` creates `CellReference` objects, Genesis deterministically subdivides each footprint cell into configurable semantic regions such as ask, center, and bid. The default layout is a generic vertical subdivision, but `CellLayout` is configurable so vendor-specific assumptions are not hardcoded.

`CellClassification` validates region confidence values, parent-cell references, frame alignment, duplicate semantic roles, overlapping semantic regions, required roles, and containment inside the parent cell. It exposes metadata for `cell_id`, row, column, semantic regions, and classification confidence, plus logical helpers for `bid_region()`, `ask_region()`, `center_region()`, `region_by_role(role)`, and `all_regions()`. `DetectionGraph` now exposes `cell_classifications` and the compatibility-style `CellClassifications` property when footprint cells are available, without changing detector contracts or replacing existing APIs.

Milestone 12 performs no OCR, no text recognition, no numeric recognition, no bid/ask value extraction, no delta calculation, no volume calculation, no AI, no ML, no OpenCV processing, no networking, no threading, and no async execution. It only teaches Genesis the logical layout of each detected footprint cell. Milestone 13 introduces OCR Foundation after the logical cell-region architecture is stable.


## Milestone 13: OCR Foundation

Milestone 13 adds immutable provider-neutral OCR contracts: `OCRRequest`, `OCRResult`, `OCRWord`, `OCRLine`, `OCRPage`, `OCRRegion`, `OCRConfiguration`, and `OCRMetadata`. The new `OCREngine`, `OCRPipeline`, and `OCRProvider` interfaces define how future OCR adapters will receive predefined semantic regions and return raw OCR output without changing detector contracts.

After Milestone 12 cell classification, `SequentialOCRPipeline` iterates through deterministic `CellClassification` results and their `CellRegion` entries, creates one `OCRRequest` per semantic region, calls the configured `OCREngine`, and returns ordered `OCRResult` values. `DetectionGraph` now exposes `ocr_results` plus raw lookup helpers `region_text(role)` and `lookup(cell_id)` without replacing previous APIs. `OCRResult`, `OCRLine`, and `OCRPage` expose helper methods for `words()`, `lines()`, `text()`, and `average_confidence()` while performing no interpretation.

`DummyOCREngine` is the only built-in engine. It returns deterministic mock OCR text for architecture tests and performs no real OCR. No Tesseract, EasyOCR, PaddleOCR, cloud API, OpenAI Vision, AI, ML, OpenCV OCR, networking, threading, or async execution is implemented. Milestone 13 does not parse numbers, validate numbers, convert values, recognize bid/ask values, calculate delta, or parse volume. Milestone 14 introduces OCR Post Processing.


## Milestone 15: Footprint Semantic Interpretation

Milestone 15 adds immutable semantic interpretation models: `NumericValue`, `ParsedValue`, `FootprintSemanticType`, `FootprintValue`, `FootprintCellData`, `FootprintInterpretation`, `InterpretationResult`, and `InterpretationWarning`. It also adds `SemanticMapper`, `FootprintInterpreter`, and `InterpretationPipeline` contracts with deterministic default implementations.

The interpretation flow is intentionally narrow:

```text
CellClassification
  ↓
ParsedValue
  ↓
SemanticMapper
  ↓
FootprintCellData
  ↓
FootprintInterpretation
```

`LayoutSemanticMapper` maps generic `CellSemanticRole` values from `CellLayout` definitions to `FootprintSemanticType` values. The default mapping treats `BID_REGION` as `BID_VOLUME`, `ASK_REGION` as `ASK_VOLUME`, `CENTER_REGION` and `DELTA_REGION` as `DELTA`, `EMPTY` as `EMPTY`, and unknown/background roles as `UNKNOWN`. This avoids hardcoded vendor-specific footprint layouts.

`FootprintCellData` validates duplicate bid/ask/delta assignments, missing parent-cell references, missing coordinates, invalid semantic assignments, multiple values assigned to the same role, and confidence bounds. It exposes helpers for `bid()`, `ask()`, `delta()`, `total_volume()`, `is_complete()`, `is_empty()`, `missing_fields()`, and `warnings()`. `DetectionGraph` now exposes detected `footprint_cells`, optional `footprint_interpretation`, and lookup helpers for interpreted cell, bid, ask, delta, and total-volume values without replacing existing graph APIs.

Milestone 15 gives Genesis semantic meaning for footprint values only. Genesis still does not detect imbalances, absorption, stacked imbalances, unfinished auctions, calculate signals, make trading decisions, or generate trades. Milestone 16 builds only the immutable footprint matrix. Milestone 17 introduces Imbalance Detection.


## Milestone 16: Footprint Matrix Builder

Milestone 16 adds immutable matrix models: `FootprintMatrix`, `MatrixRow`, `MatrixCell`, `MatrixPosition`, `MatrixDimensions`, and `MatrixStatistics`, plus the deterministic `FootprintMatrixBuilder`. The builder consumes the `DetectionGraph`, `CoordinateMapper`, and `FootprintInterpretation` and validates complete coordinates, duplicate positions, missing coordinates, deterministic ordering, bounds, row consistency, column consistency, and dimension consistency before returning the canonical two-dimensional matrix.

`SequentialObjectDetectionPipeline` now builds this matrix after semantic interpretation and stores it on `DetectionGraph` as `footprint_matrix`. Graph and matrix helpers expose deterministic lookup for cells, rows, columns, orthogonal neighbors, directional neighbors, diagonal neighbors, dimensions, and structural statistics.

Milestone 16 is structural only. It performs no imbalance detection, absorption detection, POC detection, stacked imbalance logic, unfinished auction logic, signal generation, strategy code, trading decisions, AI, ML, networking, threading, async execution, OpenCV additions, or OCR changes. Milestone 17 will introduce Imbalance Detection.


## Milestone 17: Footprint Imbalance Detection

Milestone 17 adds deterministic single-cell footprint imbalance detection after the `FootprintMatrixBuilder`. `FootprintImbalanceDetector` reads only immutable `FootprintMatrix` values that were parsed and semantically interpreted by earlier milestones. It compares ask volume against bid volume one row below, and bid volume against ask volume one row above, subject to immutable `ImbalanceConfiguration` thresholds for ratio, minimum volume, zero-opposite handling, diagonal comparison, and strict confidence mode.

The milestone exposes `FootprintImbalanceResult` on `DetectionGraph` as `footprint_imbalances`, with helpers for all imbalances, ask imbalances, bid imbalances, per-cell lookup, boolean presence checks, and immutable `ImbalanceStatistics`. It detects only individual bid/ask imbalances. It deliberately excludes stacked imbalance detection, absorption detection, unfinished auction detection, POC detection, strategy logic, trading signals, trade recommendations, AI, ML, OCR changes, OpenCV additions, networking, threading, and async execution. Milestone 18 introduces stacked imbalance detection. Milestone 19 adds deterministic absorption detection. Bundle 2 introduces the immutable Auction Market Theory layer: Developing POC, Developing Value Area, Unfinished Auctions, Excess High, and Excess Low. Bundle 4 introduces deterministic Advanced Order Flow Analytics: Delta Divergence Detection, Cumulative Delta, Delta Momentum, and Exhaustion Detection. Bundle 6 introduces the Genesis Trend Engine: Trend State Detection, Pullback Detection, Break of Structure, and Change of Character.


## Milestone 18: Stacked Footprint Imbalance Detection

Milestone 18 adds deterministic stacked imbalance detection after single-cell footprint imbalance detection. `StackedImbalanceDetector` reads only the immutable `FootprintMatrix` and the immutable `FootprintImbalanceResult` produced by Milestone 17. It finds vertically consecutive `ASK_IMBALANCE` cells as `STACKED_ASK` stacks and vertically consecutive `BID_IMBALANCE` cells as `STACKED_BID` stacks, subject only to immutable configuration thresholds for minimum stack size, optional gaps, maximum gap count, minimum average ratio, and minimum total dominant volume.

The milestone exposes `StackedImbalanceResult` on `DetectionGraph` as `stacked_imbalances`, with helpers for all stacks, ask stacks, bid stacks, stack-id lookup, cell lookup, and immutable `StackedImbalanceStatistics`. It detects stacked bid/ask imbalances only. It deliberately excludes absorption detection, unfinished auction detection, POC detection, trading logic, trading signals, trade recommendations, AI, ML, OCR changes, OpenCV additions, networking, threading, and async execution. Milestone 19 introduces Absorption Detection.

## Milestone 19: Deterministic Absorption Detection

Milestone 19 adds `FootprintAbsorptionDetector`, a deterministic analytics component that consumes the immutable `FootprintMatrix` and existing `FootprintImbalanceResult`. It detects absorption observations only when existing imbalance pressure meets the configured pressure ratio and the passive side volume in the same matrix cell meets the configured absorbed-volume threshold.

The detector emits immutable `FootprintAbsorption` objects inside an `AbsorptionResult`, with aggregate `AbsorptionStatistics` plus graph helpers for all, buy-side, sell-side, and cell-specific absorption lookup. Pipeline integration runs after matrix construction, single-cell imbalance detection, and stacked imbalance detection. Absorption results must reference the same matrix and imbalance result already attached to the `DetectionGraph`.

Milestone 19 remains side-effect free and deterministic. It adds no AI, ML, OCR changes, OpenCV additions, networking, async processing, threading, globals, randomness, vendor-specific logic, trading signals, entries, exits, or recommendations.

## Milestone 20 — Deterministic Delta Analysis

`FootprintDeltaAnalyzer` runs after Absorption Detection in `SequentialObjectDetectionPipeline` and reads only the immutable `FootprintMatrix`. It produces immutable `DeltaResult` data containing cell deltas, row deltas, whole-footprint aggregates, and deterministic statistics. `DetectionGraph` exposes the result through `footprint_delta` plus lookup helpers for cell delta, row delta, positive cells, negative cells, zero cells, and delta statistics.

## Milestone 21: Deterministic Volume Cluster Analysis

Milestone 21 adds `VolumeClusterAnalyzer`, an immutable deterministic analytics component that runs immediately after `FootprintDeltaAnalyzer`. It reads only existing semantic values from `FootprintMatrix`, computes total volume for every cell, and classifies each individual cell as `HIGH_VOLUME`, `LOW_VOLUME`, or `NORMAL_VOLUME` using `VolumeClusterConfiguration` percentile thresholds plus `minimum_volume`.

`VolumeClusterResult` stores immutable `VolumeCluster` entries, `VolumeClusterStatistics`, configuration, metadata, matrix references, deterministic ordering, and duplicate-reference validation. `DetectionGraph` exposes `volume_clusters`, `high_volume_cells()`, `low_volume_cells()`, `normal_volume_cells()`, `lookup_volume_cluster(cell_id)`, and `volume_cluster_statistics()` without replacing earlier APIs. Milestone 21 classifies individual cell volume only: no Point of Control, no HVN/LVN zones, no market profile, no auction logic, no market bias, and no trading logic or trading signals. Milestone 22 introduces Point of Control (POC).

Milestone 21 does not add Point of Control, HVN/LVN, market profile, auction logic, market prediction, market bias, trading signals, AI, ML, OCR changes, OpenCV additions, threading, async execution, networking, globals, randomness, or vendor-specific logic. Milestone 22 introduces Point of Control (POC).

## Market Profile Core (Bundle 1)

Bundle 1 adds the immutable Genesis Market Profile Core. The implementation is deterministic and consumes only existing model outputs such as `FootprintMatrix`, `VolumeClusterResult`, and `FootprintDelta`; it does not inspect images, alter OCR/OpenCV behavior, perform networking, or introduce AI/ML logic.

The sequential object detection pipeline now enriches a completed footprint matrix in this order: footprint imbalance detection, stacked imbalance detection, absorption detection, footprint delta analysis, volume cluster analysis, point of control analysis, high volume node analysis, low volume node analysis, and value area analysis.

The detection graph exposes lookup helpers and statistics for:

- Session Point of Control (POC)
- High Volume Nodes (HVN)
- Low Volume Nodes (LVN)
- Value Area (VAH / VAL)

Out of scope for Bundle 1: developing POC, developing value area, composite profiles, Auction Theory, market prediction, and trading signals. Bundle 2 begins Auction Market Theory.


### Genesis Auction Market Theory Bundle 3
Bundle 3 completes the deterministic Auction Market Theory layer. Genesis now supports Poor High, Poor Low, Single Prints, and Naked POC Tracking using only immutable footprint and point-of-control objects. After Bundle 3 the complete Auction Market Theory module is finished. Still NOT implemented: Delta Divergence, Cumulative Delta, Iceberg Detection, Exhaustion, Market Structure, AI Reasoning, and Trading Signals. Bundle 4 begins Advanced Order Flow Analytics.

## Bundle 5: Market Structure Analysis

The object detection pipeline now extends deterministic advanced order-flow analytics with `SwingDetector`, `SupportResistanceDetector`, `ZoneDetector`, and `MarketStructureAnalyzer`. These components operate only on immutable prior-bundle objects and produce frozen result models attached to `DetectionGraph` as `swing_result`, `support_resistance`, `supply_demand_zones`, and `market_structure`.

The implementation remains platform-independent and reproducible. It does not add AI reasoning, probability estimation, prediction, trading recommendations, machine learning, OCR changes, image-analysis changes, networking, threading, or async behavior. Bundle 6 introduces the Trend Engine.
