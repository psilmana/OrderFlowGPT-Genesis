# OrderFlowGPT Genesis

OrderFlowGPT Genesis is the Milestone 1 architecture freeze for a deterministic order-flow analysis core. The package defines stable domain primitives, validation rules, and an in-memory analysis pipeline that future milestones can extend without changing the foundational contracts.

## Milestone 1 scope

Milestone 1 delivers:

- A frozen package layout under `src/orderflowgpt_genesis`.
- Immutable domain models for trades, order-book levels, market snapshots, and analysis results.
- A deterministic analyzer that computes spread, mid-price, imbalance, trade bias, and confidence.
- Project documentation, release notes, changelog entries, and automated tests.

## Quick start

```bash
python -m pip install -e .[dev]
pytest
```

## Example

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

## Architecture

The architecture is intentionally small and explicit. Domain models are immutable dataclasses, services are stateless, and validation is performed at construction time. See [docs/architecture.md](docs/architecture.md) for the complete architecture freeze.

## Support status

This repository is at Milestone 1. It is suitable for deterministic local analysis and test fixtures. It does not connect to brokers, exchanges, live data feeds, or language-model providers.
