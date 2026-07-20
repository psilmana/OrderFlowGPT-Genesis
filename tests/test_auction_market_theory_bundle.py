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
    NakedPointOfControlConfiguration,
    NakedPointOfControlTracker,
    PointOfControlAnalyzer,
    PoorAuction,
    PoorAuctionDetector,
    PoorAuctionResult,
    PoorAuctionStatistics,
    PoorAuctionType,
    SinglePrint,
    SinglePrintDetector,
    SinglePrintResult,
    SinglePrintStatistics,
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


def bundle3_graph(columns, rows, asks, bids):
    graph = bundle_graph(columns, rows, asks, bids)
    poor = PoorAuctionDetector().detect(graph.footprint_matrix)
    singles = SinglePrintDetector().detect(graph.footprint_matrix)
    naked = NakedPointOfControlTracker().track((graph.point_of_control,))
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
        graph.point_of_control,
        graph.high_volume_nodes,
        graph.low_volume_nodes,
        graph.value_area,
        graph.developing_poc,
        graph.developing_value_area,
        graph.unfinished_auctions,
        graph.excess,
        poor,
        singles,
        naked,
    )


def test_poor_auction_high_low_none_both_and_graph_lookup_statistics():
    high = bundle3_graph(1, 2, (2, 0), (2, 0))
    assert high.lookup_poor_auction(PoorAuctionType.POOR_HIGH).row == 0
    low = bundle3_graph(1, 2, (0, 2), (0, 2))
    assert low.lookup_poor_auction(PoorAuctionType.POOR_LOW).row == 1
    both = bundle3_graph(1, 2, (2, 2), (2, 2))
    assert both.poor_auction_statistics().total_auctions == 2
    assert bundle3_graph(1, 2, (2, 0), (0, 2)).poor_auctions.auctions == ()


def test_single_print_regions_boundaries_large_single_row_single_column():
    graph = bundle3_graph(
        3, 5, (1, 0, 0, 1, 0, 0, 3, 3, 0, 0, 0, 2, 0, 0, 3), (0,) * 15
    )
    assert [(s.start_row, s.end_row) for s in graph.single_prints.single_prints] == [
        (0, 1),
        (3, 4),
    ]
    assert graph.lookup_single_print(1).row_count == 2
    assert graph.single_print_statistics().boundary_regions == 2
    assert bundle3_graph(1, 1, (1,), (0,)).single_print_statistics().total_regions == 1
    assert (
        bundle3_graph(1, 20, tuple(range(20)), (0,) * 20)
        .single_print_statistics()
        .total_rows
        == 20
    )


def test_naked_poc_creation_tracking_tested_expired_history_multiple():
    first = bundle_graph(1, 3, (1, 9, 1), (0, 0, 0)).point_of_control
    second = bundle_graph(1, 3, (9, 1, 1), (0, 0, 0)).point_of_control
    third = bundle_graph(1, 3, (1, 9, 1), (0, 0, 0)).point_of_control
    result = NakedPointOfControlTracker().track((first, second, third))
    assert [p.state for p in result.naked_pocs] == ["tested", "active", "active"]
    assert result.naked_pocs[0].first_revisit_index == 2
    assert result.statistics().history_length == 3
    expired = NakedPointOfControlTracker(NakedPointOfControlConfiguration(1)).track(
        (first, second)
    )
    assert expired.naked_pocs[0].state == "expired"


def test_bundle_three_immutability_validation_metadata_confidence_ordering_duplicates_references():
    graph = bundle3_graph(1, 2, (2, 2), (2, 2))
    poor = graph.poor_auctions.auctions[0]
    with pytest.raises(FrozenInstanceError):
        poor.row = 5
    with pytest.raises(TypeError):
        poor.metadata["x"] = "y"
    with pytest.raises(ValueError, match="confidence"):
        PoorAuction(PoorAuctionType.POOR_HIGH, 0, 1, 1, 2.0)
    with pytest.raises(ValueError, match="duplicate"):
        PoorAuctionResult(
            graph.footprint_matrix, (poor, poor), PoorAuctionStatistics(2, 2, 0)
        )
    single = graph.single_prints.single_prints[0]
    with pytest.raises(ValueError, match="ordered"):
        SinglePrint(1, 0, (1, 0), 0, graph.footprint_matrix, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        SinglePrintResult(
            graph.footprint_matrix, (single, single), SinglePrintStatistics(2, 2, 0)
        )
    wrong = bundle_graph(1, 2, (9, 1), (0, 0))
    with pytest.raises(ValueError, match="naked pocs must reference graph matrix"):
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
            wrong.developing_poc,
            wrong.developing_value_area,
            wrong.unfinished_auctions,
            wrong.excess,
            None,
            None,
            graph.naked_pocs,
        )


def test_pipeline_integration_exposes_bundle_three_results():
    graph = make_graph(2, 2)
    assert graph.poor_auctions is not None
    assert graph.single_prints is not None
    assert graph.naked_pocs is not None
