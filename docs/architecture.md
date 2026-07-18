# Architecture Roadmap

## Purpose

OrderFlowGPT Genesis grows through narrow milestones. Milestone 1 froze the deterministic order-flow analysis core. Milestone 2 adds only the Vision Foundation needed to reason about captured screen frames and detected workspace structure.

## Package boundaries

- `orderflowgpt_genesis.models` owns immutable market domain objects and validation.
- `orderflowgpt_genesis.analysis` owns stateless order-flow analysis behavior.
- `orderflowgpt_genesis.vision` owns frame abstractions, vision-facing interfaces, in-memory frame replay, image caching, scene graph skeletons, and workspace detection contracts.
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

## Extension rules for future milestones

Future milestones may add adapters, persistence, streaming, model-assisted narrative generation, concrete capture providers, and concrete workspace detectors. They must keep the Milestone 1 and Milestone 2 public contracts backward compatible unless a major version explicitly documents a breaking change.
