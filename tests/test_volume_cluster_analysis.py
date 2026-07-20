from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from orderflowgpt_genesis import (
    DetectionGraph,
    VolumeCluster,
    VolumeClusterAnalyzer,
    VolumeClusterConfiguration,
    VolumeClusterResult,
    VolumeClusterType,
)
from test_footprint_imbalance_detection import graph_with_values
from test_ocr_foundation import make_graph


def analyze(columns, rows, asks, bids, configuration=None):
    graph = graph_with_values(columns, rows, asks, bids)
    result = VolumeClusterAnalyzer(
        configuration or VolumeClusterConfiguration()
    ).analyze(graph.footprint_matrix)
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
        result,
    )


def test_single_cell_is_high_volume_and_lookup():
    graph = analyze(1, 1, asks=(7,), bids=(3,))
    cell_id = graph.matrix_cell(0, 0).cell_id
    cluster = graph.lookup_volume_cluster(cell_id)
    assert cluster.total_volume == Decimal("10")
    assert cluster.cluster_type == VolumeClusterType.HIGH_VOLUME
    assert cluster.percentile == Decimal("100")
    assert graph.high_volume_cells() == (cluster,)
    assert graph.low_volume_cells() == ()
    assert graph.normal_volume_cells() == ()


def test_large_matrix_mixed_clusters_statistics_and_ordering():
    graph = analyze(5, 4, asks=tuple(range(1, 21)), bids=(0,) * 20)
    clusters = graph.volume_clusters.clusters
    assert [cluster.cell_id for cluster in clusters] == sorted(
        [cluster.cell_id for cluster in clusters],
        key=lambda cell_id: tuple(int(part) for part in cell_id.split(":")[-2:]),
    )
    assert len(graph.high_volume_cells()) == 4
    assert len(graph.low_volume_cells()) == 4
    assert len(graph.normal_volume_cells()) == 12
    stats = graph.volume_cluster_statistics()
    assert stats.total_cells == 20
    assert stats.high_volume_cells == 4
    assert stats.low_volume_cells == 4
    assert stats.normal_volume_cells == 12
    assert stats.maximum_volume == Decimal("20")
    assert stats.minimum_volume == Decimal("1")
    assert stats.average_volume == Decimal("10.5")


def test_all_high_and_all_low_with_boundary_configuration():
    all_high = analyze(
        2,
        2,
        asks=(5, 5, 5, 5),
        bids=(0, 0, 0, 0),
        configuration=VolumeClusterConfiguration(Decimal("0"), Decimal("0")),
    )
    assert len(all_high.high_volume_cells()) == 4
    all_low = analyze(
        2,
        2,
        asks=(5, 5, 5, 5),
        bids=(0, 0, 0, 0),
        configuration=VolumeClusterConfiguration(Decimal("100"), Decimal("0")),
    )
    assert len(all_low.low_volume_cells()) == 4


def test_percentile_boundaries_and_minimum_volume_rejection():
    graph = analyze(4, 1, asks=(1, 2, 3, 4), bids=(0, 0, 0, 0))
    assert [c.cluster_type for c in graph.volume_clusters.clusters] == [
        VolumeClusterType.LOW_VOLUME,
        VolumeClusterType.NORMAL_VOLUME,
        VolumeClusterType.NORMAL_VOLUME,
        VolumeClusterType.HIGH_VOLUME,
    ]
    rejected = analyze(
        2,
        1,
        asks=(1, 100),
        bids=(0, 0),
        configuration=VolumeClusterConfiguration(minimum_volume=Decimal("50")),
    )
    assert (
        rejected.volume_clusters.clusters[0].cluster_type
        == VolumeClusterType.NORMAL_VOLUME
    )
    assert (
        rejected.volume_clusters.clusters[1].cluster_type
        == VolumeClusterType.HIGH_VOLUME
    )


def test_immutability_duplicate_validation_metadata_and_graph_reference_validation():
    graph = analyze(1, 1, asks=(2,), bids=(2,))
    result = graph.volume_clusters
    cluster = result.clusters[0]
    with pytest.raises(FrozenInstanceError):
        cluster.total_volume = Decimal("0")
    with pytest.raises(TypeError):
        cluster.metadata["x"] = "y"
    with pytest.raises(ValueError, match="confidence"):
        VolumeCluster(
            cluster.cell_id,
            0,
            0,
            Decimal("1"),
            VolumeClusterType.NORMAL_VOLUME,
            Decimal("0"),
            2.0,
        )
    with pytest.raises(ValueError, match="duplicate volume clusters"):
        VolumeClusterResult(result.matrix, (cluster, cluster), result.statistics())
    wrong_graph = graph_with_values(1, 1, asks=(9,), bids=(9,))
    with pytest.raises(ValueError, match="volume clusters must reference graph matrix"):
        DetectionGraph(
            wrong_graph.frame_id,
            wrong_graph.objects,
            wrong_graph.grid_coordinate_system,
            wrong_graph.cell_classifications,
            wrong_graph.ocr_results,
            wrong_graph.footprint_interpretation,
            wrong_graph.parsed_values,
            wrong_graph.footprint_matrix,
            None,
            None,
            None,
            None,
            result,
        )


def test_pipeline_integration_exposes_volume_clusters():
    graph = make_graph(2, 2)
    assert graph.volume_clusters is not None
    assert (
        graph.volume_cluster_statistics().total_cells
        == graph.footprint_matrix.statistics().total_cells
    )
