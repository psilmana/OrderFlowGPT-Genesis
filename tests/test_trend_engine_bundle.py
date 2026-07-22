from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    BreakOfStructure,
    BreakOfStructureResult,
    BreakOfStructureType,
    CHOCHDetector,
    ChangeOfCharacter,
    ChangeOfCharacterType,
    DetectionGraph,
    Pullback,
    PullbackResult,
    PullbackType,
    TrendState,
    TrendStateResult,
    TrendStateType,
)
from test_market_structure_bundle import bundle_graph


def graph_with_bundle6(columns=3, rows=2, asks=(1, 4, 2, 1, 2, 2)):
    graph = bundle_graph(columns, rows, asks=asks, bids=(0,) * (columns * rows))
    from orderflowgpt_genesis import (
        TrendStateAnalyzer,
        PullbackDetector,
        BreakOfStructureDetector,
    )

    trend = TrendStateAnalyzer().analyze(graph.footprint_matrix, graph.market_structure)
    pullbacks = PullbackDetector().detect(
        graph.footprint_matrix, graph.market_structure, trend
    )
    bos = BreakOfStructureDetector().detect(
        graph.footprint_matrix, graph.market_structure
    )
    choch = CHOCHDetector().detect(graph.footprint_matrix, graph.market_structure, bos)
    return DetectionGraph(
        graph.frame_id,
        graph.objects,
        graph.grid_coordinate_system,
        graph.cell_classifications,
        graph.ocr_results,
        graph.footprint_interpretation,
        graph.parsed_values,
        graph.footprint_matrix,
        graph.footprint_imbalances,
        swing_result=graph.swing_result,
        support_resistance=graph.support_resistance,
        supply_demand_zones=graph.supply_demand_zones,
        market_structure=graph.market_structure,
        trend_state=trend,
        pullbacks=pullbacks,
        break_of_structure=bos,
        change_of_character=choch,
    )


def test_trend_states_bullish_bearish_neutral_large_single_boundaries():
    assert graph_with_bundle6(3, 2, (1, 2, 3, 1, 2, 3)).trend_states()[
        0
    ].state_type in (
        TrendStateType.BULLISH,
        TrendStateType.STRONG_BULLISH,
        TrendStateType.WEAK_BULLISH,
    )
    assert graph_with_bundle6(3, 2, (4, 1, 3, 2, 1, 1)).trend_states()[
        0
    ].state_type in (
        TrendStateType.BEARISH,
        TrendStateType.STRONG_BEARISH,
        TrendStateType.WEAK_BEARISH,
    )
    assert (
        graph_with_bundle6(1, 1, (5,)).trend_states()[0].state_type
        == TrendStateType.NEUTRAL
    )
    assert (
        graph_with_bundle6(8, 4, tuple(range(1, 33)))
        .trend_state_statistics()
        .total_states
        == 1
    )


def test_pullbacks_deep_shallow_completed_none_and_statistics():
    graph = graph_with_bundle6(4, 2, (1, 5, 3, 4, 1, 2, 1, 3))
    types = {x.pullback_type for x in graph.detected_pullbacks()}
    assert PullbackType.SHALLOW_PULLBACK in types or PullbackType.DEEP_PULLBACK in types
    assert PullbackType.COMPLETED_PULLBACK in types
    none_graph = graph_with_bundle6(1, 1, (5,))
    assert any(
        x.pullback_type == PullbackType.NO_PULLBACK
        for x in none_graph.detected_pullbacks()
    )
    assert graph.pullback_statistics().total_pullbacks == len(
        graph.detected_pullbacks()
    )


def test_bos_bullish_bearish_internal_external_and_choch():
    graph = graph_with_bundle6(3, 2, (1, 4, 2, 1, 2, 2))
    bos_types = {x.bos_type for x in graph.bos_events()}
    assert BreakOfStructureType.BULLISH_BOS in bos_types
    assert BreakOfStructureType.BEARISH_BOS in bos_types
    assert (
        BreakOfStructureType.INTERNAL_BOS in bos_types
        or BreakOfStructureType.EXTERNAL_BOS in bos_types
    )
    choch_types = {x.choch_type for x in graph.choch_events()}
    assert (
        ChangeOfCharacterType.BULLISH_CHOCH in choch_types
        or ChangeOfCharacterType.BEARISH_CHOCH in choch_types
    )
    neutral = graph_with_bundle6(1, 1, (5,))
    assert neutral.choch_events()[0].choch_type == ChangeOfCharacterType.NO_CHOCH


def test_graph_lookup_immutability_duplicates_confidence_metadata_references():
    graph = graph_with_bundle6()
    state = graph.trend_states()[0]
    assert graph.lookup_trend_state(state.state_id) == state
    assert graph.lookup_pullback(graph.detected_pullbacks()[0].pullback_id)
    assert graph.lookup_bos(graph.bos_events()[0].bos_id)
    assert graph.lookup_choch(graph.choch_events()[0].choch_id)
    with pytest.raises(FrozenInstanceError):
        state.confidence = Decimal("0")
    with pytest.raises(TypeError):
        state.metadata["x"] = "y"
    with pytest.raises(ValueError, match="confidence"):
        TrendState(
            "bad",
            TrendStateType.NEUTRAL,
            (graph.market_structures()[0].structure_id,),
            Decimal("2"),
        )
    with pytest.raises(ValueError, match="duplicate"):
        TrendStateResult(graph.footprint_matrix, (state, state), graph.market_structure)
    with pytest.raises(ValueError, match="references"):
        PullbackResult(
            graph.footprint_matrix,
            (Pullback("bad", PullbackType.NO_PULLBACK, ("missing",)),),
            graph.market_structure,
            graph.trend_state,
        )
    with pytest.raises(ValueError, match="confidence"):
        BreakOfStructure(
            "bad",
            BreakOfStructureType.NO_BOS,
            (graph.market_structures()[0].structure_id,),
            Decimal("-1"),
        )
    with pytest.raises(ValueError, match="duplicate"):
        BreakOfStructureResult(
            graph.footprint_matrix,
            (graph.bos_events()[0], graph.bos_events()[0]),
            graph.market_structure,
        )
    with pytest.raises(ValueError, match="confidence"):
        ChangeOfCharacter(
            "bad",
            ChangeOfCharacterType.NO_CHOCH,
            (graph.market_structures()[0].structure_id,),
            Decimal("2"),
        )
