"""Stateless order-flow analyzer for Milestone 1."""

from __future__ import annotations

from decimal import Decimal

from .models import AnalysisResult, Bias, MarketSnapshot

_TWO = Decimal("2")
_ZERO = Decimal("0")
_ONE = Decimal("1")


class OrderFlowAnalyzer:
    """Compute deterministic order-flow metrics from a market snapshot."""

    def analyze(self, snapshot: MarketSnapshot) -> AnalysisResult:
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        spread = best_ask.price - best_bid.price
        mid_price = (best_bid.price + best_ask.price) / _TWO
        bid_quantity = sum((level.quantity for level in snapshot.bids), _ZERO)
        ask_quantity = sum((level.quantity for level in snapshot.asks), _ZERO)
        book_imbalance = self._ratio(
            bid_quantity - ask_quantity, bid_quantity + ask_quantity
        )
        net_trade_quantity = sum(
            (
                trade.quantity if trade.side == "buy" else -trade.quantity
                for trade in snapshot.trades
            ),
            _ZERO,
        )
        trade_quantity = sum((trade.quantity for trade in snapshot.trades), _ZERO)
        trade_imbalance = self._ratio(net_trade_quantity, trade_quantity)
        combined_signal = (book_imbalance + trade_imbalance) / _TWO
        bias = self._bias(combined_signal)
        confidence = min(abs(combined_signal), _ONE)
        return AnalysisResult(
            symbol=snapshot.symbol.upper(),
            mid_price=mid_price,
            spread=spread,
            book_imbalance=book_imbalance,
            net_trade_quantity=net_trade_quantity,
            bias=bias,
            confidence=confidence,
        )

    @staticmethod
    def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
        if denominator == _ZERO:
            return _ZERO
        return numerator / denominator

    @staticmethod
    def _bias(signal: Decimal) -> Bias:
        if signal > Decimal("0.10"):
            return "bullish"
        if signal < Decimal("-0.10"):
            return "bearish"
        return "neutral"
