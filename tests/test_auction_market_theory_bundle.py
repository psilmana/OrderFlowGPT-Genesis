from dataclasses import FrozenInstanceError
import pytest

from orderflowgpt_genesis import (
    DetectionGraph,
    DevelopingPointOfControl,
    DevelopingPointOfControlAnalyzer,
    DevelopingPointOfControlResult,
    DevelopingPointOfControlStatistics,
    DevelopingValueAreaAnalyzer,
    Excess,
    ExcessDetector,
    ExcessResult,
    ExcessStatistics,
    ExcessType,
    HighVolumeNodeAnalyzer,
    LowVolumeNodeAnalyzer,
    PointOfControlAnalyzer,
    UnfinishedAuction,
    UnfinishedAuctionDetector,
    UnfinishedAuctionResult,
    UnfinishedAuctionStatistics,
    UnfinishedAuctionType,
    ValueAreaAnalyzer,
)
from test_footprint_imbalance_detection import graph_with_values
from test_market_profile_core import market_graph
from test_ocr_foundation import make_graph


def bundle_graph(columns, rows, asks, bids):
    base = graph_with_values(columns, rows, asks=asks, bids=bids)
    poc = PointOfControlAnalyzer().analyze(base.footprint_matrix)
    hvn = HighVolumeNodeAnalyzer().analyze(base.footprint_matrix)
    lvn = LowVolumeNodeAnalyzer().analyze(base.footprint_matrix)
    va = ValueAreaAnalyzer().analyze(base.footprint_matrix, poc)
    dpoc = DevelopingPointOfControlAnalyzer().analyze(base.footprint_matrix)
    dva = DevelopingValueAreaAnalyzer().analyze(base.footprint_matrix)
    ua = UnfinishedAuctionDetector().detect(base.footprint_matrix)
    excess = ExcessDetector().detect(base.footprint_matrix)
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
        base.footprint_delta,
        base.volume_clusters,
        poc,
        hvn,
        lvn,
        va,
        dpoc,
        dva,
        ua,
        excess,
    )


def test_developing_poc_history_stable_and_moving():
    graph = bundle_graph(3, 2, asks=(5, 4, 3, 1, 10, 20), bids=(0,) * 6)
    assert [item.row for item in graph.developing_poc.history] == [0, 1, 1]
    assert graph.developing_poc.current_poc.row == 1
    assert graph.developing_poc.previous_poc.row == 1
    assert graph.developing_poc.movement_direction == "STABLE"
    assert graph.developing_poc_statistics().stable_slices == 1


def test_developing_value_area_movement_expansion_contraction_history():
    graph = bundle_graph(3, 3, asks=(10, 0, 0, 0, 10, 0, 0, 0, 10), bids=(0,) * 9)
    history = graph.developing_value_area.history
    assert len(history) == 3
    assert (
        graph.developing_value_area.current_value_area.vah
        >= graph.developing_value_area.current_value_area.val
    )
    assert graph.developing_value_area_statistics().expansion >= 0
    assert graph.developing_value_area.movement in {"STABLE", "UP", "DOWN", "MIXED"}


def test_unfinished_auctions_top_bottom_both_none_and_lookup_statistics():
    assert (
        bundle_graph(1, 2, (2, 0), (2, 0))
        .lookup_unfinished_auction(UnfinishedAuctionType.TOP)
        .row
        == 0
    )
    assert (
        bundle_graph(1, 2, (0, 2), (0, 2))
        .lookup_unfinished_auction(UnfinishedAuctionType.BOTTOM)
        .row
        == 1
    )
    both = bundle_graph(1, 2, (2, 2), (2, 2))
    assert both.unfinished_auction_statistics().total_auctions == 2
    assert bundle_graph(1, 2, (2, 0), (0, 2)).unfinished_auctions.auctions == ()


def test_excess_high_low_both_none_large_single_row_single_column():
    high = bundle_graph(1, 2, (2, 0), (0, 0))
    assert high.lookup_excess(ExcessType.EXCESS_HIGH).row == 0
    low = bundle_graph(1, 2, (0, 0), (0, 2))
    assert low.lookup_excess(ExcessType.EXCESS_LOW).row == 1
    both = bundle_graph(1, 2, (2, 0), (0, 2))
    assert both.excess_statistics().total_excesses == 2
    assert bundle_graph(1, 2, (2, 2), (2, 2)).excess.excesses == ()
    large = bundle_graph(5, 20, tuple(range(100)), tuple(0 for _ in range(100)))
    assert large.developing_poc_statistics().total_slices == 5
    single = bundle_graph(1, 1, (2,), (0,))
    assert single.developing_value_area_statistics().total_slices == 1


def test_auction_models_immutability_validation_metadata_confidence_ordering_duplicates_references():
    graph = bundle_graph(2, 2, (1, 2, 3, 4), (0, 0, 0, 0))
    item = graph.developing_poc.current_poc
    with pytest.raises(FrozenInstanceError):
        item.row = 3
    with pytest.raises(TypeError):
        item.metadata["x"] = "y"
    with pytest.raises(ValueError, match="confidence"):
        DevelopingPointOfControl(0, 0, 1, 2.0)
    with pytest.raises(ValueError, match="ordered"):
        DevelopingPointOfControlResult(
            graph.footprint_matrix,
            (item, item),
            DevelopingPointOfControlStatistics(2, item.row, item.row, 0, 1),
        )
    ua = UnfinishedAuction(UnfinishedAuctionType.TOP, 0, 1, 1, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        UnfinishedAuctionResult(
            graph.footprint_matrix, (ua, ua), UnfinishedAuctionStatistics(2, 2, 0)
        )
    ex = Excess(ExcessType.EXCESS_HIGH, 0, 0, 1, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        ExcessResult(graph.footprint_matrix, (ex, ex), ExcessStatistics(2, 2, 0))
    wrong = market_graph(1, 2, (9, 9))
    with pytest.raises(ValueError, match="developing poc must reference graph matrix"):
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
            wrong.point_of_control,
            wrong.high_volume_nodes,
            wrong.low_volume_nodes,
            wrong.value_area,
            graph.developing_poc,
        )


def test_pipeline_integration_exposes_bundle_two_results():
    graph = make_graph(2, 2)
    assert graph.developing_poc is not None
    assert graph.developing_value_area is not None
    assert graph.unfinished_auctions is not None
    assert graph.excess is not None
