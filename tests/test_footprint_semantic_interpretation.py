import pytest

from orderflowgpt_genesis import (
    CellLayout,
    CellLayoutAnalyzer,
    CellLayoutBand,
    CellSemanticRole,
    DetectionGraph,
    FootprintCellData,
    FootprintSemanticType,
    FootprintValue,
    LayoutSemanticMapper,
    NumericValue,
    ParsedValue,
    SequentialInterpretationPipeline,
)
from test_ocr_foundation import make_graph


def parsed(classification, role, value=10, confidence=0.9):
    region = classification.region_by_role(role)
    assert region is not None
    return ParsedValue(
        NumericValue(value), role, confidence, region, classification.cell_id
    )


def parse_all(classifications):
    values = []
    for index, classification in enumerate(classifications):
        values.extend(
            (
                parsed(classification, CellSemanticRole.ASK_REGION, 100 + index),
                parsed(classification, CellSemanticRole.CENTER_REGION, index - 5),
                parsed(classification, CellSemanticRole.BID_REGION, 90 + index),
            )
        )
    return tuple(values)


def test_successful_interpretation_helpers_and_determinism():
    graph = make_graph(2, 2)
    result = SequentialInterpretationPipeline().run(
        tuple(reversed(graph.cell_classifications)),
        parse_all(graph.cell_classifications),
    )
    repeated = SequentialInterpretationPipeline().run(
        graph.cell_classifications,
        tuple(reversed(parse_all(graph.cell_classifications))),
    )
    assert result == repeated
    assert result.grid_id == graph.grid_coordinate_system.grid_id
    assert len(result.ordered_cells) == 4
    first = result.ordered_cells[0]
    assert first.ask().semantic_type == FootprintSemanticType.ASK_VOLUME
    assert first.bid().semantic_type == FootprintSemanticType.BID_VOLUME
    assert first.delta().semantic_type == FootprintSemanticType.DELTA
    assert first.total_volume() is None
    assert first.is_complete()
    assert not first.is_empty()
    assert first.missing_fields() == ()


def test_empty_cells_and_missing_values_emit_warnings():
    graph = make_graph(1, 1)
    interpretation = SequentialInterpretationPipeline().run(
        graph.cell_classifications, ()
    )
    cell = interpretation.ordered_cells[0]
    assert cell.is_empty()
    assert cell.missing_fields() == (
        FootprintSemanticType.BID_VOLUME,
        FootprintSemanticType.ASK_VOLUME,
        FootprintSemanticType.DELTA,
    )
    assert cell.warnings()


def test_unknown_semantic_roles_are_warnings_not_values():
    graph = make_graph(1, 1)
    classification = CellLayoutAnalyzer(
        CellLayout((CellLayoutBand(CellSemanticRole.UNKNOWN, required=False),))
    ).classify(graph.cell_classifications[0].cell_reference)
    region = classification.semantic_regions[0]
    value = ParsedValue(
        NumericValue(1), CellSemanticRole.UNKNOWN, 1.0, region, classification.cell_id
    )
    interpretation = SequentialInterpretationPipeline().run((classification,), (value,))
    assert interpretation.ordered_cells[0].is_empty()
    assert interpretation.ordered_cells[0].warnings()[0].code == "unknown"


def test_duplicate_semantic_assignments_and_invalid_mapping_rejected():
    graph = make_graph(1, 1)
    classification = graph.cell_classifications[0]
    one = parsed(classification, CellSemanticRole.BID_REGION, 1)
    two = parsed(classification, CellSemanticRole.BID_REGION, 2)
    with pytest.raises(ValueError, match="duplicate bid_volume values"):
        SequentialInterpretationPipeline().run((classification,), (one, two))
    bad_mapper = LayoutSemanticMapper(
        {CellSemanticRole.BID_REGION: FootprintSemanticType.ASK_VOLUME}
    )
    semantic_type = bad_mapper.map(one.semantic_role, one)
    with pytest.raises(ValueError, match="invalid semantic mapping"):
        FootprintCellData(
            classification.cell_reference,
            bid_value=FootprintValue(
                one.numeric_value,
                semantic_type,
                one.confidence,
                one.source_region,
                one.cell_id,
            ),
        )


def test_validation_rejects_parent_coordinate_confidence_and_outside_values():
    graph = make_graph(1, 1)
    classification = graph.cell_classifications[0]
    value = parsed(classification, CellSemanticRole.BID_REGION)
    with pytest.raises(ValueError, match="confidence"):
        ParsedValue(
            value.numeric_value,
            value.semantic_role,
            1.1,
            value.source_region,
            value.cell_id,
        )
    with pytest.raises(ValueError, match="parent cell"):
        ParsedValue(
            value.numeric_value,
            value.semantic_role,
            value.confidence,
            value.source_region,
            "missing",
        )
    with pytest.raises(ValueError, match="parent cell"):
        FootprintValue(
            value.numeric_value,
            FootprintSemanticType.BID_VOLUME,
            value.confidence,
            value.source_region,
            "missing",
        )


def test_single_row_single_column_large_grid_and_graph_lookup_helpers():
    single_row = make_graph(4, 1)
    assert (
        len(
            SequentialInterpretationPipeline()
            .run(
                single_row.cell_classifications,
                parse_all(single_row.cell_classifications),
            )
            .ordered_cells
        )
        == 4
    )
    single_col = make_graph(1, 4)
    assert (
        len(
            SequentialInterpretationPipeline()
            .run(
                single_col.cell_classifications,
                parse_all(single_col.cell_classifications),
            )
            .ordered_cells
        )
        == 4
    )
    large = make_graph(8, 6)
    interpretation = SequentialInterpretationPipeline().run(
        large.cell_classifications, parse_all(large.cell_classifications)
    )
    graph = DetectionGraph(
        large.frame_id,
        large.objects,
        large.grid_coordinate_system,
        large.cell_classifications,
        large.ocr_results,
        interpretation,
    )
    cell_id = interpretation.ordered_cells[-1].cell_reference.coordinate.cell_id
    assert len(graph.footprint_cells) == 48
    assert graph.lookup_cell(cell_id) == interpretation.lookup_cell(cell_id)
    assert graph.lookup_bid(cell_id) is not None
    assert graph.lookup_ask(cell_id) is not None
    assert graph.lookup_delta(cell_id) is not None
    assert graph.lookup_total_volume(cell_id) is None
