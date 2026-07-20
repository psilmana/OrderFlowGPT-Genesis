from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    AbsorptionConfiguration,
    AbsorptionResult,
    AbsorptionSide,
    AbsorptionType,
    DetectionGraph,
    FootprintAbsorption,
    FootprintAbsorptionDetector,
)
from test_footprint_imbalance_detection import graph_with_values
from test_ocr_foundation import make_graph


def detect_absorptions(columns, rows, asks, bids, configuration=None):
    graph = graph_with_values(columns, rows, asks, bids)
    result = FootprintAbsorptionDetector(
        configuration or AbsorptionConfiguration()
    ).detect(graph.footprint_matrix, graph.footprint_imbalances)
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
        graph.stacked_imbalances,
        result,
    )


def test_successful_buy_and_sell_absorption_lookup_statistics_and_ordering():
    graph = detect_absorptions(
        2,
        2,
        asks=(300, 10, 10, 80),
        bids=(75, 10, 10, 300),
    )
    assert [d.absorption_type for d in graph.absorptions()] == [
        AbsorptionType.BUY_ABSORPTION,
        AbsorptionType.SELL_ABSORPTION,
    ]
    buy = graph.buy_absorptions()[0]
    sell = graph.sell_absorptions()[0]
    assert buy.passive_side == AbsorptionSide.BID
    assert buy.absorbed_volume == Decimal("75")
    assert buy.pressure_ratio == Decimal("30")
    assert sell.passive_side == AbsorptionSide.ASK
    assert sell.absorbed_volume == Decimal("80")
    assert graph.lookup_absorption(buy.cell_id) == buy
    stats = graph.absorption_statistics()
    assert stats.total_cells == 4
    assert stats.buy_absorptions == 1
    assert stats.sell_absorptions == 1
    assert stats.total_absorptions == 2
    assert stats.cells_without_absorption == 2


def test_absorption_threshold_rejections_and_empty_pipeline_result():
    low_passive = detect_absorptions(1, 2, asks=(300, 10), bids=(49, 10))
    assert low_passive.absorptions() == ()
    high_ratio = AbsorptionConfiguration(minimum_pressure_ratio=Decimal("31"))
    low_ratio = detect_absorptions(1, 2, (300, 10), (75, 10), high_ratio)
    assert low_ratio.absorptions() == ()
    pipeline_graph = make_graph(2, 2)
    assert pipeline_graph.absorption is not None
    assert pipeline_graph.absorptions() == ()


def test_absorption_validation_immutability_and_graph_reference_checks():
    graph = detect_absorptions(1, 2, asks=(300, 10), bids=(75, 10))
    detection = graph.absorptions()[0]
    with pytest.raises(ValueError, match="duplicate"):
        AbsorptionResult(
            graph.footprint_matrix,
            graph.footprint_imbalances,
            (detection, detection),
        )
    with pytest.raises(ValueError, match="confidence"):
        FootprintAbsorption(
            detection.cell_id,
            detection.position,
            detection.absorption_type,
            detection.passive_side,
            detection.absorbed_volume,
            detection.pressure_ratio,
            detection.source_imbalance,
            2.0,
        )
    with pytest.raises(FrozenInstanceError):
        detection.cell_id = "changed"
    with pytest.raises(TypeError):
        detection.metadata["x"] = "y"
    other = detect_absorptions(1, 2, asks=(300, 10), bids=(80, 10))
    with pytest.raises(ValueError, match="reference graph matrix"):
        DetectionGraph(
            graph.frame_id,
            graph.objects,
            graph.grid_coordinate_system,
            graph.cell_classifications,
            graph.ocr_results,
            graph.footprint_interpretation,
            graph.parsed_values,
            graph.footprint_matrix,
            graph.footprint_imbalances,
            graph.stacked_imbalances,
            other.absorption,
        )


def test_absorption_configuration_validation():
    with pytest.raises(ValueError, match="minimum absorbed volume"):
        AbsorptionConfiguration(minimum_absorbed_volume=Decimal("-1"))
    with pytest.raises(ValueError, match="minimum pressure ratio"):
        AbsorptionConfiguration(minimum_pressure_ratio=Decimal("0"))
