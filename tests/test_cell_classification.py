from dataclasses import FrozenInstanceError

import pytest

from orderflowgpt_genesis import (
    BoundingBox,
    CellClassification,
    CellLayout,
    CellLayoutAnalyzer,
    CellReference,
    CellRegion,
    CellSemanticRole,
    CoordinateMapper,
    DetectionGraph,
    DetectorRegistry,
    FootprintCellDetector,
    FootprintGridDetector,
    ObjectType,
    SequentialObjectDetectionPipeline,
)
from test_footprint_cell_detector import detect, make_cell_frame, make_context


def first_reference(columns: int = 3, rows: int = 2) -> CellReference:
    result = detect(columns=columns, rows=rows)
    return CoordinateMapper().references(result.detected_objects)[0]


def classifications(columns: int = 3, rows: int = 2):
    result = detect(columns=columns, rows=rows)
    return tuple(
        CellLayoutAnalyzer().classify(reference)
        for reference in CoordinateMapper().references(result.detected_objects)
    )


def test_successful_classification_metadata_helpers_and_immutability():
    classification = CellLayoutAnalyzer().analyze(first_reference())
    assert isinstance(classification, CellClassification)
    assert [region.semantic_role for region in classification.all_regions()] == [
        CellSemanticRole.ASK_REGION,
        CellSemanticRole.CENTER_REGION,
        CellSemanticRole.BID_REGION,
    ]
    assert classification.ask_region() == classification.region_by_role(
        CellSemanticRole.ASK_REGION
    )
    assert classification.center_region() is not None
    assert classification.bid_region() is not None
    assert classification.classification_confidence == 1.0
    assert classification.validation_metadata["cell_id"] == classification.cell_id
    assert classification.validation_metadata["row"] == classification.row
    assert classification.validation_metadata["column"] == classification.column
    with pytest.raises(FrozenInstanceError):
        classification.overall_confidence = 0.5


def test_single_row_single_column_and_large_grid_classification_counts():
    assert len(classifications(columns=5, rows=1)) == 5
    assert len(classifications(columns=1, rows=4)) == 4
    large = classifications(columns=8, rows=6)
    assert len(large) == 48
    assert large == classifications(columns=8, rows=6)


def test_duplicate_semantic_roles_are_rejected():
    ref = first_reference()
    region = CellRegion(
        ref.bounds,
        CellSemanticRole.ASK_REGION,
        1.0,
        ref.coordinate.cell_id,
        ref.frame_id,
    )
    with pytest.raises(ValueError, match="duplicate semantic roles"):
        CellClassification(ref, (region, region), 1.0)
    with pytest.raises(ValueError, match="semantic roles must be unique"):
        CellLayout(
            (
                CellLayout().bands[0],
                CellLayout().bands[0],
            )
        )


def test_overlapping_regions_are_rejected():
    ref = first_reference()
    first = CellRegion(
        BoundingBox(ref.bounds.x, ref.bounds.y, ref.bounds.width, 10),
        CellSemanticRole.ASK_REGION,
        1.0,
        ref.coordinate.cell_id,
        ref.frame_id,
    )
    second = CellRegion(
        BoundingBox(ref.bounds.x, ref.bounds.y + 5, ref.bounds.width, 10),
        CellSemanticRole.BID_REGION,
        1.0,
        ref.coordinate.cell_id,
        ref.frame_id,
    )
    with pytest.raises(ValueError, match="must not overlap"):
        CellClassification(ref, (first, second), 1.0)


def test_invalid_confidence_geometry_and_parent_validation():
    ref = first_reference()
    with pytest.raises(ValueError, match="confidence"):
        CellRegion(
            ref.bounds,
            CellSemanticRole.ASK_REGION,
            1.5,
            ref.coordinate.cell_id,
            ref.frame_id,
        )
    with pytest.raises(ValueError, match="width"):
        BoundingBox(ref.bounds.x, ref.bounds.y, -1, 1)
    outside = CellRegion(
        BoundingBox(ref.bounds.right, ref.bounds.y, 1, 1),
        CellSemanticRole.ASK_REGION,
        1.0,
        ref.coordinate.cell_id,
        ref.frame_id,
    )
    with pytest.raises(ValueError, match="inside parent cell"):
        CellClassification(ref, (outside,), 1.0)
    wrong_parent = CellRegion(
        BoundingBox(ref.bounds.x, ref.bounds.y, 1, 1),
        CellSemanticRole.ASK_REGION,
        1.0,
        "other-cell",
        ref.frame_id,
    )
    with pytest.raises(ValueError, match="parent id"):
        CellClassification(ref, (wrong_parent,), 1.0)


def test_missing_required_regions_are_rejected():
    ref = first_reference()
    region = CellRegion(
        ref.bounds,
        CellSemanticRole.ASK_REGION,
        1.0,
        ref.coordinate.cell_id,
        ref.frame_id,
    )
    with pytest.raises(ValueError, match="missing required"):
        CellClassification(
            ref,
            (region,),
            1.0,
            {
                "required_roles": frozenset(
                    {CellSemanticRole.ASK_REGION, CellSemanticRole.BID_REGION}
                )
            },
        )


def test_pipeline_and_detection_graph_expose_cell_classifications():
    chart = BoundingBox(50, 35, 430, 310)
    grid_box = BoundingBox(80, 70, 350, 240)
    frame = make_cell_frame(640, 420, chart, grid_box, 4, 3)
    graph = SequentialObjectDetectionPipeline(
        DetectorRegistry((FootprintGridDetector(), FootprintCellDetector()))
    ).run(make_context(frame, chart))
    assert isinstance(graph, DetectionGraph)
    assert graph.grid_coordinate_system is not None
    assert len(graph.cell_classifications) == 12
    assert graph.CellClassifications == graph.cell_classifications
    assert graph.cell_classifications[0].row == 0
    assert graph.cell_classifications[-1].column == 3
    assert all(
        region.semantic_role != CellSemanticRole.UNKNOWN
        for classification in graph.cell_classifications
        for region in classification.semantic_regions
    )
    cell_ids = {
        str(obj.metadata.get("cell_id"))
        for obj in graph.objects
        if obj.object_type == ObjectType.FOOTPRINT_CELL
    }
    assert {
        classification.cell_id for classification in graph.cell_classifications
    } == cell_ids
