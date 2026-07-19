from dataclasses import FrozenInstanceError

import pytest

from orderflowgpt_genesis import (
    BoundingBox,
    CellSemanticRole,
    DetectionGraph,
    DetectorRegistry,
    DummyOCREngine,
    FootprintCellDetector,
    FootprintGridDetector,
    OCRConfiguration,
    OCRLine,
    OCRMetadata,
    OCRPage,
    OCRRequest,
    OCRResult,
    OCRWord,
    SequentialOCRPipeline,
    SequentialObjectDetectionPipeline,
)
from test_footprint_cell_detector import make_cell_frame, make_context


def make_graph(columns: int = 2, rows: int = 2) -> DetectionGraph:
    chart = BoundingBox(50, 35, 430, 310)
    grid_box = BoundingBox(80, 70, 350, 240)
    frame = make_cell_frame(640, 420, chart, grid_box, columns, rows)
    return SequentialObjectDetectionPipeline(
        DetectorRegistry((FootprintGridDetector(), FootprintCellDetector()))
    ).run(make_context(frame, chart))


def test_ocr_configuration_validation_and_immutability():
    config = OCRConfiguration(
        language="eng",
        minimum_confidence=0.25,
        character_whitelist="0123456789",
        character_blacklist="abc",
        engine_options={"mode": "mock"},
    )
    assert config.engine_options["mode"] == "mock"
    with pytest.raises(FrozenInstanceError):
        config.language = "spa"
    with pytest.raises(ValueError, match="minimum confidence"):
        OCRConfiguration(minimum_confidence=1.5)
    with pytest.raises(ValueError, match="language"):
        OCRConfiguration(language=" ")


def test_ocr_request_validation_rejects_invalid_inputs():
    graph = make_graph(1, 1)
    classification = graph.cell_classifications[0]
    region = classification.semantic_regions[0]
    frame = classification.cell_reference.detected_object
    image = make_context(
        make_cell_frame(
            640, 420, BoundingBox(50, 35, 430, 310), BoundingBox(80, 70, 200, 180), 1, 1
        ),
        BoundingBox(50, 35, 430, 310),
    ).processed_frame.source_frame
    OCRRequest(
        frame.frame_id,
        classification.cell_id,
        region.bounds,
        region.semantic_role,
        image,
    )
    with pytest.raises(ValueError, match="frame id"):
        OCRRequest(
            " ", classification.cell_id, region.bounds, region.semantic_role, image
        )
    with pytest.raises(ValueError, match="cell id"):
        OCRRequest(frame.frame_id, " ", region.bounds, region.semantic_role, image)
    with pytest.raises(ValueError, match="semantic role"):
        OCRRequest(
            frame.frame_id,
            classification.cell_id,
            region.bounds,
            CellSemanticRole.UNKNOWN,
            image,
        )
    with pytest.raises(ValueError, match="fit within"):
        OCRRequest(
            frame.frame_id,
            classification.cell_id,
            BoundingBox(999, 999, 10, 10),
            region.semantic_role,
            image,
        )


def test_dummy_engine_returns_deterministic_raw_output_and_helpers():
    graph = make_graph(1, 1)
    classification = graph.cell_classifications[0]
    region = classification.semantic_regions[0]
    image = classification.cell_reference.detected_object
    source_frame = make_context(
        make_cell_frame(
            640, 420, BoundingBox(50, 35, 430, 310), BoundingBox(80, 70, 200, 180), 1, 1
        ),
        BoundingBox(50, 35, 430, 310),
    ).processed_frame.source_frame
    request = OCRRequest(
        image.frame_id,
        classification.cell_id,
        region.bounds,
        region.semantic_role,
        source_frame,
    )
    result = DummyOCREngine().run(request)
    assert result.text() == f"{classification.cell_id}:{region.semantic_role.value}"
    assert result.average_confidence() == 1.0
    assert [word.text for word in result.words()] == [result.text()]
    assert result.lines()[0].text() == result.text()
    assert result.metadata.provider_name == "dummy"
    assert DummyOCREngine().run(request) == result


def test_ocr_result_validation_ordering_empty_output_and_page():
    box = BoundingBox(10, 10, 20, 10)
    first = OCRWord("b", 0.6, BoundingBox(20, 10, 5, 5))
    second = OCRWord("a", 0.8, BoundingBox(10, 10, 5, 5))
    line = OCRLine((first, second), box, 0.7)
    assert line.words() == (second, first)
    page = OCRPage((line,), OCRMetadata("engine", "provider"))
    assert page.lines() == (line,)
    result = OCRResult("frame", "cell", CellSemanticRole.ASK_REGION, "", 0.0, ())
    assert result.words() == ()
    assert result.lines() == ()
    assert result.average_confidence() == 0.0
    with pytest.raises(ValueError, match="result confidence"):
        OCRResult("frame", "cell", CellSemanticRole.ASK_REGION, "x", -0.1, (box,))
    with pytest.raises(ValueError, match="bounding boxes"):
        OCRResult("frame", "cell", CellSemanticRole.ASK_REGION, "x", 1.0, (), (first,))


def test_pipeline_graph_integration_multiple_cells_regions_lookup_and_ordering():
    graph = make_graph(3, 2)
    assert graph.ocr_results
    assert len(graph.ocr_results) == len(graph.cell_classifications) * 3
    repeated = make_graph(3, 2)
    assert [(result.cell_id, result.semantic_role) for result in graph.ocr_results] == [
        (result.cell_id, result.semantic_role) for result in repeated.ocr_results
    ]
    first_cell = graph.cell_classifications[0].cell_id
    assert len(graph.lookup(first_cell)) == 3
    assert graph.region_text(CellSemanticRole.ASK_REGION)
    assert all(
        text.endswith(CellSemanticRole.ASK_REGION.value)
        for text in graph.region_text(CellSemanticRole.ASK_REGION)
    )


def test_pipeline_empty_classifications_returns_empty_tuple():
    frame = make_context(
        make_cell_frame(
            640, 420, BoundingBox(50, 35, 430, 310), BoundingBox(80, 70, 200, 180), 1, 1
        ),
        BoundingBox(50, 35, 430, 310),
    ).processed_frame
    assert SequentialOCRPipeline().run(frame, ()) == ()
