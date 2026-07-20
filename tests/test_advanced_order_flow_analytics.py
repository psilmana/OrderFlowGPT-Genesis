from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    CumulativeDeltaAnalyzer,
    CumulativeDeltaConfiguration,
    DeltaDivergenceAnalyzer,
    DeltaDivergenceConfiguration,
    DeltaDivergenceResult,
    DeltaDivergenceType,
    DeltaMomentumAnalyzer,
    DeltaMomentumResult,
    DeltaMomentumType,
    DetectionGraph,
    ExhaustionDetector,
    ExhaustionResult,
    ExhaustionType,
    FootprintDeltaAnalyzer,
)
from test_footprint_imbalance_detection import graph_with_values
from test_ocr_foundation import make_graph


def analytics_graph(columns, rows, asks, bids):
    base = graph_with_values(columns, rows, asks=asks, bids=bids)
    delta = FootprintDeltaAnalyzer().analyze(base.footprint_matrix)
    divergence = DeltaDivergenceAnalyzer().analyze(base.footprint_matrix, delta)
    cumulative = CumulativeDeltaAnalyzer().analyze(base.footprint_matrix, delta)
    momentum = DeltaMomentumAnalyzer().analyze(base.footprint_matrix, delta)
    exhaustion = ExhaustionDetector().detect(base.footprint_matrix, delta)
    return DetectionGraph(
        base.frame_id,
        base.objects,
        base.grid_coordinate_system,
        base.cell_classifications,
        base.ocr_results,
        base.footprint_interpretation,
        base.parsed_values,
        base.footprint_matrix,
        base.footprint_imbalances,
        base.stacked_imbalances,
        base.absorption,
        delta,
        base.volume_clusters,
        base.point_of_control,
        base.high_volume_nodes,
        base.low_volume_nodes,
        base.value_area,
        base.developing_poc,
        base.developing_value_area,
        base.unfinished_auctions,
        base.excess,
        base.poor_auctions,
        base.single_prints,
        base.naked_pocs,
        divergence,
        cumulative,
        momentum,
        exhaustion,
    )


def test_delta_divergence_bullish_bearish_hidden_neutral_large_and_boundaries():
    bullish = analytics_graph(1, 2, asks=(1, 7), bids=(5, 1))
    assert bullish.lookup_delta_divergence(DeltaDivergenceType.BULLISH_DIVERGENCE)
    bearish = analytics_graph(1, 2, asks=(7, 1), bids=(1, 5))
    assert bearish.lookup_delta_divergence(DeltaDivergenceType.BEARISH_DIVERGENCE)
    base = graph_with_values(1, 2, asks=(1, 7), bids=(5, 1))
    delta = FootprintDeltaAnalyzer().analyze(base.footprint_matrix)
    hidden = DeltaDivergenceAnalyzer(DeltaDivergenceConfiguration(hidden=True)).analyze(
        base.footprint_matrix, delta
    )
    assert hidden.lookup(DeltaDivergenceType.HIDDEN_BULLISH)
    neutral = analytics_graph(1, 1, asks=(5,), bids=(5,))
    assert neutral.lookup_delta_divergence(DeltaDivergenceType.NEUTRAL)
    large = analytics_graph(10, 10, asks=tuple(range(100)), bids=(0,) * 100)
    assert large.delta_divergence.statistics().total_divergences == 1
    single_column = analytics_graph(1, 3, asks=(1, 2, 3), bids=(0, 0, 0))
    assert single_column.delta_divergence_statistics().bullish == 1


def test_cumulative_delta_running_reset_aggregation_statistics_and_lookup():
    base = graph_with_values(1, 3, asks=(3, 1, 5), bids=(1, 4, 2))
    delta = FootprintDeltaAnalyzer().analyze(base.footprint_matrix)
    result = CumulativeDeltaAnalyzer(
        CumulativeDeltaConfiguration(reset_rows=(2,))
    ).analyze(base.footprint_matrix, delta)
    assert [v.row_delta for v in result.values] == [
        Decimal("2"),
        Decimal("-3"),
        Decimal("3"),
    ]
    assert [v.running_delta for v in result.values] == [
        Decimal("2"),
        Decimal("-1"),
        Decimal("2"),
    ]
    assert [v.session_delta for v in result.values] == [
        Decimal("2"),
        Decimal("-1"),
        Decimal("3"),
    ]
    assert result.lookup(2).reset is True
    assert result.statistics().final_running_delta == Decimal("2")
    assert result.statistics().maximum_running_delta == Decimal("2")


def test_delta_momentum_acceleration_weakening_flat_and_boundary_cases():
    assert analytics_graph(1, 2, (2, 5), (1, 1)).lookup_delta_momentum(
        DeltaMomentumType.ACCELERATING_BUYING
    )
    assert analytics_graph(1, 2, (1, 1), (2, 5)).lookup_delta_momentum(
        DeltaMomentumType.ACCELERATING_SELLING
    )
    assert analytics_graph(1, 2, (5, 2), (1, 1)).lookup_delta_momentum(
        DeltaMomentumType.WEAKENING_BUYING
    )
    assert analytics_graph(1, 2, (1, 1), (5, 2)).lookup_delta_momentum(
        DeltaMomentumType.WEAKENING_SELLING
    )
    assert analytics_graph(1, 1, (5,), (5,)).lookup_delta_momentum(
        DeltaMomentumType.FLAT
    )


def test_exhaustion_buyer_seller_none_graph_pipeline_and_validations():
    buyer = analytics_graph(1, 2, (1, 6), (1, 1))
    assert buyer.lookup_exhaustion(ExhaustionType.BUYER_EXHAUSTION)
    seller = analytics_graph(1, 2, (1, 1), (1, 6))
    assert seller.lookup_exhaustion(ExhaustionType.SELLER_EXHAUSTION)
    none = analytics_graph(1, 1, (1,), (1,))
    assert none.lookup_exhaustion(ExhaustionType.NO_EXHAUSTION)
    graph = make_graph(2, 2)
    assert graph.delta_divergence is not None
    assert graph.cumulative_delta is not None
    assert graph.delta_momentum is not None
    assert graph.exhaustion is not None
    item = buyer.exhaustion.exhaustions[0]
    with pytest.raises(FrozenInstanceError):
        item.row_index = 3
    with pytest.raises(TypeError):
        item.metadata["x"] = "y"
    with pytest.raises(ValueError, match="confidence"):
        item.__class__(item.exhaustion_type, item.row_index, item.delta_value, 2.0)
    with pytest.raises(ValueError, match="duplicate"):
        ExhaustionResult(
            buyer.footprint_matrix,
            buyer.footprint_delta,
            (item, item),
            buyer.exhaustion.statistics(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        DeltaMomentumResult(
            buyer.footprint_matrix,
            buyer.footprint_delta,
            buyer.delta_momentum.momentums * 2,
            buyer.delta_momentum.statistics(),
        )
    with pytest.raises(ValueError, match="reference matrix delta"):
        DeltaDivergenceResult(
            buyer.footprint_matrix,
            seller.footprint_delta,
            buyer.delta_divergence.divergences,
            buyer.delta_divergence.statistics(),
        )
