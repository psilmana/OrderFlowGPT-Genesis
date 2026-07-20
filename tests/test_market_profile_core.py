from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    DetectionGraph,
    HighVolumeNode,
    HighVolumeNodeAnalyzer,
    HighVolumeNodeConfiguration,
    HighVolumeNodeResult,
    LowVolumeNodeAnalyzer,
    LowVolumeNodeConfiguration,
    PointOfControl,
    PointOfControlAnalyzer,
    PointOfControlType,
    ValueAreaAnalyzer,
    ValueAreaConfiguration,
)
from test_footprint_imbalance_detection import graph_with_values
from test_ocr_foundation import make_graph


def market_graph(columns, rows, totals):
    graph = graph_with_values(
        columns, rows, asks=tuple(totals), bids=(0,) * len(totals)
    )
    poc = PointOfControlAnalyzer().analyze(graph.footprint_matrix)
    hvn = HighVolumeNodeAnalyzer().analyze(graph.footprint_matrix)
    lvn = LowVolumeNodeAnalyzer().analyze(graph.footprint_matrix)
    va = ValueAreaAnalyzer().analyze(graph.footprint_matrix, poc)
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
        graph.absorption,
        graph.footprint_delta,
        graph.volume_clusters,
        poc,
        hvn,
        lvn,
        va,
    )


def test_poc_single_row_and_equal_maximum_tie_breaking():
    graph = market_graph(3, 1, (5, 10, 15))
    assert graph.session_poc().row == 0
    assert graph.session_poc().total_volume == Decimal("30")
    tied = market_graph(1, 3, (10, 10, 1))
    assert tied.session_poc().row == 0
    assert tied.point_of_control_statistics().tied_poc_rows == 2


def test_hvn_lvn_mixed_high_only_and_low_only_profiles():
    graph = market_graph(1, 5, (1, 2, 3, 4, 5))
    assert [node.row for node in graph.high_volume_nodes.nodes] == [4]
    assert [node.row for node in graph.low_volume_nodes.nodes] == [0]
    high_only = HighVolumeNodeAnalyzer(
        HighVolumeNodeConfiguration(Decimal("0"))
    ).analyze(graph.footprint_matrix)
    assert len(high_only.nodes) == 5
    low_only = LowVolumeNodeAnalyzer(
        LowVolumeNodeConfiguration(Decimal("100"))
    ).analyze(graph.footprint_matrix)
    assert len(low_only.nodes) == 5
    assert graph.lookup_high_volume_node(4).total_volume == Decimal("5")
    assert graph.lookup_low_volume_node(0).total_volume == Decimal("1")


def test_value_area_70_percent_boundaries_large_single_column():
    graph = market_graph(1, 5, (10, 20, 100, 30, 40))
    va = graph.value_area.value_area
    assert va.poc_row == 2
    assert va.val == 2
    assert va.vah == 4
    assert va.included_volume == Decimal("170")
    assert va.coverage_percentage == Decimal("85.00")
    assert graph.value_area_statistics().target_percentage == Decimal("70")
    large = market_graph(1, 25, tuple(range(1, 26)))
    assert large.high_volume_node_statistics().total_rows == 25


def test_value_area_single_row_single_column_and_minimum_rows():
    graph = market_graph(1, 1, (7,))
    va = ValueAreaAnalyzer(ValueAreaConfiguration(minimum_rows=1)).analyze(
        graph.footprint_matrix
    )
    assert va.value_area.val == va.value_area.vah == 0
    assert va.value_area.coverage_percentage == Decimal("100")


def test_immutability_duplicate_metadata_confidence_ordering_and_references():
    graph = market_graph(1, 3, (1, 2, 3))
    poc = graph.session_poc()
    with pytest.raises(FrozenInstanceError):
        poc.row = 0
    with pytest.raises(TypeError):
        poc.metadata["x"] = "y"
    with pytest.raises(ValueError, match="confidence"):
        PointOfControl(0, Decimal("1"), PointOfControlType.SESSION_POC, 2.0)
    node = graph.high_volume_nodes.nodes[0]
    with pytest.raises(ValueError, match="duplicate high volume nodes"):
        HighVolumeNodeResult(
            graph.footprint_matrix, (node, node), graph.high_volume_nodes.statistics()
        )
    with pytest.raises(ValueError, match="ordered"):
        HighVolumeNodeResult(
            graph.footprint_matrix,
            (HighVolumeNode(2, 3, 100, 1.0), HighVolumeNode(0, 1, 0, 1.0)),
            graph.high_volume_nodes.statistics(),
        )
    wrong = graph_with_values(1, 3, asks=(9, 9, 9), bids=(0, 0, 0))
    with pytest.raises(
        ValueError, match="point of control must reference graph matrix"
    ):
        DetectionGraph(
            wrong.frame_id,
            wrong.objects,
            wrong.grid_coordinate_system,
            wrong.cell_classifications,
            wrong.ocr_results,
            wrong.footprint_interpretation,
            wrong.parsed_values,
            wrong.footprint_matrix,
            None,
            None,
            None,
            None,
            None,
            graph.point_of_control,
        )


def test_pipeline_integration_exposes_market_profile_core():
    graph = make_graph(2, 2)
    assert graph.point_of_control is not None
    assert graph.high_volume_nodes is not None
    assert graph.low_volume_nodes is not None
    assert graph.value_area is not None
