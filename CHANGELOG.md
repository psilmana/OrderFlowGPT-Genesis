# Changelog

## Unreleased

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
