from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    DetectionGraph,
    MarketStructureAnalyzer,
    MarketStructureType,
    SupportResistanceDetector,
    SupportResistanceLevel,
    SupportResistanceType,
    SwingDetector,
    SwingPoint,
    SwingResult,
    SwingType,
    ZoneDetector,
    ZoneType,
)
from test_footprint_imbalance_detection import graph_with_values


def bundle_graph(columns=3, rows=2, asks=(1, 4, 2, 1, 2, 2), bids=(0, 0, 0, 0, 0, 0)):
    graph = graph_with_values(columns, rows, asks, bids)
    swings = SwingDetector().detect(graph.footprint_matrix)
    levels = SupportResistanceDetector().detect(graph.footprint_matrix, swings)
    zones = ZoneDetector().detect(graph.footprint_matrix, levels)
    structure = MarketStructureAnalyzer().analyze(graph.footprint_matrix, swings, zones)
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
        swing_result=swings,
        support_resistance=levels,
        supply_demand_zones=zones,
        market_structure=structure,
    )


def test_swing_detection_hh_hl_lh_ll_equal_large_single_boundaries():
    graph = bundle_graph(3, 2, asks=(1, 4, 2, 1, 2, 2), bids=(0,) * 6)
    types = [s.swing_type for s in graph.swings()]
    assert SwingType.HIGHER_SWING_HIGH in types
    assert SwingType.LOWER_SWING_HIGH in types
    assert SwingType.HIGHER_SWING_LOW in types
    assert SwingType.EQUAL_LOW in types
    assert graph.swing_statistics().total_swings == 6
    single = bundle_graph(1, 1, asks=(5,), bids=(0,))
    assert len(single.swings()) == 2
    large = bundle_graph(8, 4, asks=tuple(range(1, 33)), bids=(0,) * 32)
    assert large.swing_statistics().total_swings == 16


def test_support_resistance_broken_levels_zones_overlap_and_structure():
    graph = bundle_graph()
    level_types = {level.level_type for level in graph.support_resistance_levels()}
    assert SupportResistanceType.SUPPORT in level_types
    assert any(
        t in level_types
        for t in (
            SupportResistanceType.RESISTANCE,
            SupportResistanceType.BROKEN_RESISTANCE,
        )
    )
    zone_types = {zone.zone_type for zone in graph.zones()}
    assert any(t in zone_types for t in (ZoneType.SUPPLY, ZoneType.BROKEN_SUPPLY))
    assert ZoneType.DEMAND in zone_types
    assert all(zone.start_row <= zone.end_row for zone in graph.zones())
    structure_types = {s.structure_type for s in graph.market_structures()}
    assert MarketStructureType.HIGHER_HIGH in structure_types
    assert MarketStructureType.HIGHER_LOW in structure_types
    assert any(
        t in structure_types
        for t in (
            MarketStructureType.BULLISH_STRUCTURE,
            MarketStructureType.BEARISH_STRUCTURE,
            MarketStructureType.NEUTRAL_STRUCTURE,
        )
    )


def test_graph_lookup_immutability_validation_duplicates_confidence_metadata_references():
    graph = bundle_graph()
    swing = graph.swings()[0]
    assert graph.lookup_swing(swing.swing_id) == swing
    assert graph.lookup_support_resistance(
        graph.support_resistance_levels()[0].level_id
    )
    assert graph.lookup_zone(graph.zones()[0].zone_id)
    assert graph.lookup_market_structure(graph.market_structures()[0].structure_id)
    with pytest.raises(FrozenInstanceError):
        swing.value = Decimal("9")
    with pytest.raises(TypeError):
        swing.metadata["x"] = "y"
    with pytest.raises(ValueError, match="confidence"):
        SwingPoint(
            "bad",
            SwingType.EQUAL_HIGH,
            swing.cell_id,
            swing.position,
            swing.value,
            Decimal("2"),
        )
    with pytest.raises(ValueError, match="duplicate"):
        SwingResult(graph.footprint_matrix, (swing, swing))
    with pytest.raises(ValueError, match="position"):
        SupportResistanceLevel(
            "bad",
            SupportResistanceType.SUPPORT,
            "missing",
            swing.position,
            Decimal("1"),
        )
