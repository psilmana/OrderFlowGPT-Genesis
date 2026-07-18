# OrderFlowGPT Genesis

OrderFlowGPT Genesis is the architecture foundation for deterministic order-flow analysis and vision-driven workspace understanding. The package defines stable domain primitives, validation rules, an in-memory analysis pipeline, and the Milestone 2 Vision Foundation that future adapters can extend without changing the foundational contracts.

## Current scope

The repository currently delivers:

- A frozen package layout under `src/orderflowgpt_genesis`.
- Immutable domain models for trades, order-book levels, market snapshots, and analysis results.
- A deterministic analyzer that computes spread, mid-price, imbalance, trade bias, and confidence.
- Vision Foundation abstractions for image frames, capture/replay interfaces, bounded image caching, scene graph skeletons, and workspace detection contracts.
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

## Architecture

The architecture is intentionally small and explicit. Domain models are immutable dataclasses, services are stateless, and validation is performed at construction time. See [docs/architecture.md](docs/architecture.md) for the complete roadmap.

## Support status

This repository is at Milestone 2. It is suitable for deterministic local analysis, test fixtures, and in-memory vision foundation workflows. It does not connect to brokers, exchanges, live data feeds, screen capture services, storage systems, or language-model providers.
