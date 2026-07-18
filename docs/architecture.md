# Milestone 1 Architecture Freeze

## Purpose

Milestone 1 freezes the architecture for the OrderFlowGPT Genesis core. The core converts a validated `MarketSnapshot` into an `AnalysisResult` using deterministic arithmetic and explicit domain rules.

## Package boundaries

- `orderflowgpt_genesis.models` owns immutable domain objects and validation.
- `orderflowgpt_genesis.analysis` owns stateless analysis behavior.
- `orderflowgpt_genesis.__init__` exposes the supported public API.

No Milestone 1 module performs network I/O, file I/O, broker access, exchange access, or language-model calls.

## Domain model

- `OrderBookLevel` represents a positive price and positive displayed quantity.
- `Trade` represents a positive price, positive quantity, and aggressor side of `buy` or `sell`.
- `MarketSnapshot` represents one symbol, at least one bid, at least one ask, and optional completed trades.
- `AnalysisResult` represents the computed mid-price, spread, book imbalance, net trade quantity, directional bias, and confidence.

## Analysis algorithm

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

## Extension rules for future milestones

Future milestones may add adapters, persistence, streaming, and model-assisted narrative generation. They must keep the Milestone 1 public domain contracts backward compatible unless a major version explicitly documents a breaking change.
