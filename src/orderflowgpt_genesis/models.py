"""Immutable domain models for order-flow analysis."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

TradeSide = Literal["buy", "sell"]
Bias = Literal["bullish", "bearish", "neutral"]


def _require_positive(value: Decimal, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """A single price level in the visible order book."""

    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        _require_positive(self.price, "price")
        _require_positive(self.quantity, "quantity")


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed market trade with aggressor side."""

    price: Decimal
    quantity: Decimal
    side: TradeSide

    def __post_init__(self) -> None:
        _require_positive(self.price, "price")
        _require_positive(self.quantity, "quantity")
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """A point-in-time market state used as analyzer input."""

    symbol: str
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]
    trades: tuple[Trade, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if not self.bids:
            raise ValueError("at least one bid level is required")
        if not self.asks:
            raise ValueError("at least one ask level is required")
        if self.best_bid.price >= self.best_ask.price:
            raise ValueError("best bid must be lower than best ask")

    @property
    def best_bid(self) -> OrderBookLevel:
        return max(self.bids, key=lambda level: level.price)

    @property
    def best_ask(self) -> OrderBookLevel:
        return min(self.asks, key=lambda level: level.price)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Deterministic order-flow summary produced from a market snapshot."""

    symbol: str
    mid_price: Decimal
    spread: Decimal
    book_imbalance: Decimal
    net_trade_quantity: Decimal
    bias: Bias
    confidence: Decimal
