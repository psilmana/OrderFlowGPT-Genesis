from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    MarketSnapshot,
    OrderBookLevel,
    OrderFlowAnalyzer,
    Trade,
)


def test_analyzer_returns_bullish_bias_for_bid_and_buy_pressure():
    snapshot = MarketSnapshot(
        symbol="es",
        bids=(OrderBookLevel(Decimal("5000.00"), Decimal("30")),),
        asks=(OrderBookLevel(Decimal("5000.25"), Decimal("10")),),
        trades=(Trade(Decimal("5000.25"), Decimal("5"), "buy"),),
    )

    result = OrderFlowAnalyzer().analyze(snapshot)

    assert result.symbol == "ES"
    assert result.mid_price == Decimal("5000.125")
    assert result.spread == Decimal("0.25")
    assert result.book_imbalance == Decimal("0.5")
    assert result.net_trade_quantity == Decimal("5")
    assert result.bias == "bullish"
    assert result.confidence == Decimal("0.75")


def test_analyzer_returns_neutral_bias_without_directional_pressure():
    snapshot = MarketSnapshot(
        symbol="NQ",
        bids=(OrderBookLevel(Decimal("18000.00"), Decimal("10")),),
        asks=(OrderBookLevel(Decimal("18000.50"), Decimal("10")),),
    )

    result = OrderFlowAnalyzer().analyze(snapshot)

    assert result.bias == "neutral"
    assert result.confidence == Decimal("0")


def test_snapshot_rejects_crossed_market():
    with pytest.raises(ValueError, match="best bid must be lower than best ask"):
        MarketSnapshot(
            symbol="CL",
            bids=(OrderBookLevel(Decimal("80.01"), Decimal("1")),),
            asks=(OrderBookLevel(Decimal("80.00"), Decimal("1")),),
        )


def test_trade_rejects_invalid_side():
    with pytest.raises(ValueError, match="side must be 'buy' or 'sell'"):
        Trade(Decimal("100"), Decimal("1"), "hold")
