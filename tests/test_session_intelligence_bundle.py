from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    DetectionGraph,
    InitialBalance,
    InitialBalanceAnalyzer,
    OpeningAuctionAnalyzer,
    OpeningAuctionType,
    SessionStatisticsAnalyzer,
    TradingSession,
    TradingSessionDetector,
    TradingSessionResult,
    TradingSessionType,
    TrendStateType,
)
from test_market_structure_bundle import bundle_graph


def session_graph(columns=5, rows=3, asks=None, bids=None, minutes=None):
    asks = asks or tuple(range(1, columns * rows + 1))
    bids = bids or (0,) * (columns * rows)
    minutes = minutes or (300, 570, 600, 970, 1260)[:columns]
    graph = bundle_graph(columns, rows, asks=asks, bids=bids)
    new_rows = []
    for row in graph.footprint_matrix.rows:
        new_cells = []
        for cell in row.cells:
            obj = replace(
                cell.original_cell,
                metadata={
                    **cell.original_cell.metadata,
                    "minute": minutes[cell.column_index],
                },
            )
            new_cells.append(replace(cell, original_cell=obj))
        new_rows.append(replace(row, cells=tuple(new_cells)))
    matrix = replace(graph.footprint_matrix, rows=tuple(new_rows))
    trend = None
    sessions = TradingSessionDetector().detect(matrix)
    stats = SessionStatisticsAnalyzer().analyze(matrix, sessions, None, None, trend)
    ib = InitialBalanceAnalyzer().analyze(matrix, sessions)
    auction = OpeningAuctionAnalyzer().analyze(matrix, sessions, ib, stats)
    return DetectionGraph(
        graph.frame_id,
        graph.objects,
        graph.grid_coordinate_system,
        graph.cell_classifications,
        graph.ocr_results,
        graph.footprint_interpretation,
        graph.parsed_values,
        matrix,
        trend_state=trend,
        trading_session=sessions,
        session_statistics=stats,
        initial_balance=ib,
        opening_auction=auction,
    )


def test_trading_session_detection_all_session_types_and_unknown():
    graph = session_graph(minutes=(300, 570, 600, 970, 1260))
    assert [s.session_type for s in graph.trading_sessions()] == [
        TradingSessionType.PRE_MARKET,
        TradingSessionType.RTH,
        TradingSessionType.POST_MARKET,
        TradingSessionType.ETH,
    ]
    unknown = session_graph(columns=1, rows=1, asks=(1,), minutes=(None,))
    assert unknown.trading_sessions()[0].session_type == TradingSessionType.UNKNOWN
    assert graph.trading_session_statistics().total_sessions == 4
    assert (
        graph.lookup_trading_session(graph.trading_sessions()[0].session_id) is not None
    )


def test_session_statistics_high_low_range_volume_delta_and_trend():
    graph = session_graph(
        columns=2,
        rows=3,
        asks=(1, 2, 3, 4, 5, 6),
        bids=(0, 1, 1, 1, 1, 1),
        minutes=(570, 600),
    )
    stats = graph.lookup_session_statistics(graph.trading_sessions()[0].session_id)
    assert stats.session_high == Decimal("2")
    assert stats.session_low == Decimal("0")
    assert stats.session_range == Decimal("2")
    assert stats.session_volume == Decimal("26")
    assert stats.session_delta == Decimal("16")
    assert stats.session_trend_state in set(TrendStateType)


def test_initial_balance_break_extension_lookup_and_immutability():
    graph = session_graph(columns=3, rows=3, minutes=(570, 600, 630))
    balance = graph.initial_balances()[0]
    assert balance.ib_high == Decimal("2")
    assert balance.ib_low == Decimal("0")
    assert balance.ib_mid == Decimal("1")
    assert balance.ib_range == Decimal("2")
    assert balance.ib_break is False
    assert graph.lookup_initial_balance(balance.initial_balance_id) == balance
    with pytest.raises(FrozenInstanceError):
        balance.ib_break = True
    with pytest.raises(TypeError):
        balance.metadata["x"] = "y"


def test_initial_balance_extension_with_partial_opening_columns():
    graph = session_graph(columns=3, rows=3, minutes=(570, 600, 630))
    result = InitialBalanceAnalyzer(
        configuration=type(graph.initial_balance.configuration)(columns=1)
    ).analyze(graph.footprint_matrix, graph.trading_session)
    assert result.balances[0].ib_break is False
    assert result.balances[0].ib_extension == Decimal("0")


def test_opening_auction_types_and_statistics():
    drive = session_graph(columns=1, rows=2, asks=(3, 3), bids=(0, 0), minutes=(570,))
    assert drive.opening_auctions()[0].auction_type == OpeningAuctionType.OPEN_DRIVE
    auction = session_graph(columns=1, rows=2, asks=(1, 1), bids=(1, 1), minutes=(300,))
    assert auction.opening_auctions()[0].auction_type == OpeningAuctionType.OPEN_AUCTION
    in_range = session_graph(
        columns=1, rows=2, asks=(1, 1), bids=(1, 1), minutes=(570,)
    )
    assert (
        in_range.opening_auctions()[0].auction_type
        == OpeningAuctionType.OPEN_AUCTION_IN_RANGE
    )
    assert drive.opening_auction_statistics().open_drives == 1


def test_ordering_duplicate_confidence_and_reference_validation():
    graph = session_graph(columns=1, rows=1, asks=(1,), minutes=(570,))
    session = graph.trading_sessions()[0]
    bad_ref = TradingSession("bad", TradingSessionType.RTH, 0, 0, ("missing",))
    with pytest.raises(ValueError):
        TradingSessionResult(graph.footprint_matrix, (bad_ref,))
    with pytest.raises(ValueError):
        TradingSession(
            "bad",
            TradingSessionType.RTH,
            0,
            0,
            (graph.footprint_matrix.cells[0].cell_id,),
            Decimal("2"),
        )
    with pytest.raises(ValueError):
        TradingSession("bad", TradingSessionType.RTH, 0, 0, ("a", "a"))
    with pytest.raises(ValueError):
        TradingSessionResult(graph.footprint_matrix, (session, session))
    with pytest.raises(ValueError):
        InitialBalance(
            "x",
            session.session_id,
            Decimal("0"),
            Decimal("1"),
            Decimal("0.5"),
            Decimal("-1"),
            Decimal("0"),
            False,
            (graph.footprint_matrix.cells[0].cell_id,),
        )
