from datetime import datetime, timezone

import pytest

from orderflowgpt_genesis import (
    AbsorptionDetector,
    BigTradeDetector,
    BottomPanel,
    BoundingBox,
    ChartRegion,
    DetectedObject,
    DetectionConfidence,
    DetectionContext,
    DetectionGraph,
    DetectionResult,
    DetectionSource,
    DetectorRegistry,
    DeterministicImagePreprocessor,
    FootprintDetector,
    ImageFrame,
    ObjectId,
    ObjectType,
    PriceAxis,
    PriceAxisDetector,
    SequentialObjectDetectionPipeline,
    TimeAxis,
    TimeAxisDetector,
    Viewport,
    VolumeProfileDetector,
    WorkspaceLayout,
)


def make_context() -> DetectionContext:
    frame = ImageFrame(
        data=b"pixels",
        width=800,
        height=600,
        pixel_format="RGB",
        captured_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
        source="unit-test",
        frame_id="frame-6",
    )
    processed = DeterministicImagePreprocessor().preprocess(frame)
    layout = WorkspaceLayout(
        workspace_id="workspace-1",
        frame_id=frame.frame_id,
        bounds=BoundingBox(0, 0, 800, 600),
        chart_region=ChartRegion(BoundingBox(80, 40, 600, 420), 0.9),
        price_axis=PriceAxis(BoundingBox(680, 40, 80, 420), 0.8),
        time_axis=TimeAxis(BoundingBox(80, 460, 600, 40), 0.8),
        viewport=Viewport(BoundingBox(80, 40, 680, 460), 0.85),
        bottom_panels=(BottomPanel(BoundingBox(80, 500, 680, 80), 0.7),),
    )
    return DetectionContext(
        processed_frame=processed,
        workspace_layout=layout,
        configuration={"profile": "unit"},
    )


def detected_object(
    object_id: str,
    object_type: ObjectType = ObjectType.PRICE_TEXT,
    parent_id: ObjectId | None = None,
    children_ids: tuple[ObjectId, ...] = (),
    metadata: dict[str, str] | None = None,
) -> DetectedObject:
    return DetectedObject(
        object_id=ObjectId(object_id),
        bounds=BoundingBox(10, 20, 30, 40),
        confidence=DetectionConfidence(0.75),
        object_type=object_type,
        frame_id="frame-6",
        source=DetectionSource("unit-detector"),
        parent_id=parent_id,
        children_ids=children_ids,
        metadata=metadata or {},
    )


def test_detected_object_validation_and_immutable_metadata():
    obj = detected_object("price-1", metadata={"text": "5000.25"})

    assert obj.object_id.value == "price-1"
    assert obj.object_type == ObjectType.PRICE_TEXT
    assert obj.metadata["text"] == "5000.25"

    with pytest.raises(TypeError):
        obj.metadata["text"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="object detection confidence"):
        DetectionConfidence(1.1)
    with pytest.raises(ValueError, match="object type"):
        ObjectType("NOT_SUPPORTED")
    with pytest.raises(ValueError, match="own parent"):
        detected_object("self", parent_id=ObjectId("self"))


def test_detection_graph_accepts_valid_parent_and_child_references():
    parent = detected_object("axis", children_ids=(ObjectId("price-1"),))
    child = detected_object("price-1", parent_id=ObjectId("axis"))

    graph = DetectionGraph(frame_id="frame-6", objects=(parent, child))

    assert graph.objects == (parent, child)


def test_detection_graph_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        DetectionGraph(
            frame_id="frame-6",
            objects=(detected_object("dup"), detected_object("dup")),
        )


def test_detection_graph_rejects_invalid_parent_references():
    child = detected_object("child", parent_id=ObjectId("missing"))

    with pytest.raises(ValueError, match="parent id"):
        DetectionGraph(frame_id="frame-6", objects=(child,))


def test_detection_graph_rejects_invalid_child_references():
    parent = detected_object("parent", children_ids=(ObjectId("missing"),))

    with pytest.raises(ValueError, match="child ids"):
        DetectionGraph(frame_id="frame-6", objects=(parent,))


def test_detection_context_validates_frame_alignment_and_freezes_configuration():
    context = make_context()

    assert context.configuration["profile"] == "unit"
    with pytest.raises(TypeError):
        context.configuration["profile"] = "changed"  # type: ignore[index]

    mismatched = WorkspaceLayout(
        workspace_id="workspace-1",
        frame_id="other-frame",
        bounds=BoundingBox(0, 0, 800, 600),
        chart_region=ChartRegion(BoundingBox(80, 40, 600, 420), 0.9),
        price_axis=PriceAxis(BoundingBox(680, 40, 80, 420), 0.8),
        time_axis=TimeAxis(BoundingBox(80, 460, 600, 40), 0.8),
        viewport=Viewport(BoundingBox(80, 40, 680, 460), 0.85),
    )
    with pytest.raises(ValueError, match="frame ids must match"):
        DetectionContext(context.processed_frame, mismatched)


def test_detector_registry_is_immutable_and_validates_unique_names():
    registry = DetectorRegistry().add(PriceAxisDetector()).add(TimeAxisDetector())

    assert DetectorRegistry().detectors == ()
    assert [detector.name for detector in registry.detectors] == [
        "price-axis-detector",
        "time-axis-detector",
    ]
    with pytest.raises(ValueError, match="unique"):
        DetectorRegistry((PriceAxisDetector(), PriceAxisDetector()))


def test_pipeline_execution_preserves_order_and_builds_graph():
    class StaticDetector:
        def __init__(self, name: str, obj: DetectedObject) -> None:
            self.name = name
            self._obj = obj

        def detect(self, context: DetectionContext) -> DetectionResult[DetectedObject]:
            return DetectionResult(
                region=self._obj.bounds,
                confidence=self._obj.confidence.value,
                reason=f"{self.name} static fixture",
                detector_name=self.name,
                detected_object=self._obj,
            )

    first = detected_object("first", object_type=ObjectType.BID_VALUE)
    second = detected_object("second", object_type=ObjectType.ASK_VALUE)
    registry = DetectorRegistry(
        (
            StaticDetector("first-detector", first),
            StaticDetector("second-detector", second),
        )
    )

    graph = SequentialObjectDetectionPipeline(registry).run(make_context())

    assert [obj.object_id.value for obj in graph.objects] == ["first", "second"]


def test_placeholder_detectors_return_empty_detection_results():
    context = make_context()
    detectors = (
        PriceAxisDetector(),
        TimeAxisDetector(),
        FootprintDetector(),
        VolumeProfileDetector(),
        BigTradeDetector(),
        AbsorptionDetector(),
    )

    results = tuple(detector.detect(context) for detector in detectors)

    assert results[0].detector_name == "price-axis-detector"
    assert all(result.detected_object is None for result in results)
    assert all(result.region is None for result in results)
    assert all(result.confidence == 0.0 for result in results)
    graph = SequentialObjectDetectionPipeline(DetectorRegistry(detectors)).run(context)
    assert graph.objects == ()
