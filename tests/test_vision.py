from datetime import datetime, timezone

import pytest

from orderflowgpt_genesis import (
    BottomPanel,
    BoundingBox,
    ChartRegion,
    DeterministicImagePreprocessor,
    ImageCache,
    ImageFrame,
    InMemoryFrameReplay,
    PreprocessingConfig,
    PriceAxis,
    RegionOfInterest,
    SceneGraph,
    SceneNode,
    StatusBar,
    TimeAxis,
    Toolbar,
    Viewport,
    WorkspaceDetection,
    WorkspaceLayout,
)


def frame(frame_id: str) -> ImageFrame:
    return ImageFrame(
        data=b"pixels",
        width=2,
        height=2,
        pixel_format="RGB",
        captured_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        source="unit-test",
        frame_id=frame_id,
    )


def test_image_frame_requires_timezone_aware_capture_time():
    with pytest.raises(ValueError, match="captured_at must be timezone-aware"):
        ImageFrame(
            data=b"pixels",
            width=2,
            height=2,
            pixel_format="RGB",
            captured_at=datetime(2026, 7, 18),
        )


def test_in_memory_replay_returns_frames_in_order():
    first = frame("first")
    second = frame("second")

    replay = InMemoryFrameReplay((first, second))

    assert tuple(replay.frames()) == (first, second)


def test_image_cache_evicts_least_recently_used_frame():
    cache = ImageCache(max_items=2)
    first = frame("first")
    second = frame("second")
    third = frame("third")

    cache.put(first)
    cache.put(second)
    assert cache.get("first") == first
    cache.put(third)

    assert "first" in cache
    assert "second" not in cache
    assert "third" in cache
    assert len(cache) == 2


def test_scene_graph_validates_references():
    root = SceneNode(
        node_id="root",
        label="chart",
        bounds=BoundingBox(x=0, y=0, width=100, height=100),
        children=("axis",),
    )
    axis = SceneNode(
        node_id="axis",
        label="price axis",
        bounds=BoundingBox(x=90, y=0, width=10, height=100),
    )

    graph = SceneGraph(frame_id="frame-1", nodes=(root, axis), root_id="root")

    assert graph.root_id == "root"


def test_scene_graph_rejects_missing_child_reference():
    node = SceneNode(
        node_id="root",
        label="chart",
        bounds=BoundingBox(x=0, y=0, width=100, height=100),
        children=("missing",),
    )

    with pytest.raises(ValueError, match="scene node children"):
        SceneGraph(frame_id="frame-1", nodes=(node,), root_id="root")


def test_workspace_detection_confidence_range():
    detection = WorkspaceDetection(
        workspace_id="main",
        frame_id="frame-1",
        bounds=BoundingBox(x=0, y=0, width=800, height=600),
        confidence=0.9,
    )

    assert detection.label == "workspace"

    with pytest.raises(ValueError, match="workspace confidence"):
        WorkspaceDetection(
            workspace_id="main",
            frame_id="frame-1",
            bounds=BoundingBox(x=0, y=0, width=800, height=600),
            confidence=1.1,
        )


def test_preprocessing_pipeline_materializes_milestone_3_stages():
    source = ImageFrame(
        data=b"rgb-pixels",
        width=640,
        height=480,
        pixel_format="RGB",
        captured_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        source="unit-test",
        frame_id="frame-3",
    )
    config = PreprocessingConfig(
        pyramid_scales=(1.0, 0.5),
        zoom_normalization_scale=1.25,
        roi_regions=(
            RegionOfInterest(
                name="chart",
                bounds=BoundingBox(x=10, y=20, width=300, height=200),
            ),
        ),
    )

    processed = DeterministicImagePreprocessor().preprocess(source, config)

    assert processed.source_frame == source
    assert processed.grayscale.pixel_format == "GRAY"
    assert processed.hsv.pixel_format == "HSV"
    assert processed.gaussian_blur.source.endswith(":gaussian-blur:5")
    assert processed.adaptive_threshold.pixel_format == "BINARY"
    assert processed.canny_edges.pixel_format == "BINARY"
    assert processed.morphology.pixel_format == "BINARY"
    assert processed.roi_frames["chart"].width == 300
    assert processed.roi_frames["chart"].height == 200
    assert [
        (level.scale, level.width, level.height) for level in processed.pyramid
    ] == [
        (1.0, 640, 480),
        (0.5, 320, 240),
    ]
    assert processed.zoom_normalized is not None
    assert processed.zoom_normalized.width == 800
    assert processed.zoom_normalized.height == 600


def test_preprocessing_config_validates_cv_parameters():
    with pytest.raises(ValueError, match="gaussian kernel size"):
        PreprocessingConfig(gaussian_kernel_size=4)

    with pytest.raises(ValueError, match="canny high threshold"):
        PreprocessingConfig(canny_low_threshold=100, canny_high_threshold=50)

    with pytest.raises(ValueError, match="pyramid scales"):
        PreprocessingConfig(pyramid_scales=(1.0, 0.0))


def test_preprocessor_rejects_roi_outside_source_frame():
    source = frame("source")
    config = PreprocessingConfig(
        roi_regions=(
            RegionOfInterest(
                name="outside",
                bounds=BoundingBox(x=1, y=1, width=2, height=2),
            ),
        ),
    )

    with pytest.raises(ValueError, match="region of interest"):
        DeterministicImagePreprocessor().preprocess(source, config)


def test_workspace_layout_models_milestone_4_detection_regions():
    layout = WorkspaceLayout(
        workspace_id="main",
        frame_id="frame-4",
        bounds=BoundingBox(x=0, y=0, width=1200, height=800),
        chart_region=ChartRegion(
            bounds=BoundingBox(x=80, y=40, width=960, height=560),
            confidence=0.94,
        ),
        price_axis=PriceAxis(
            bounds=BoundingBox(x=1040, y=40, width=80, height=560),
            confidence=0.91,
        ),
        time_axis=TimeAxis(
            bounds=BoundingBox(x=80, y=600, width=960, height=50),
            confidence=0.9,
        ),
        viewport=Viewport(
            bounds=BoundingBox(x=80, y=40, width=1040, height=610),
            confidence=0.93,
        ),
        bottom_panels=(
            BottomPanel(
                bounds=BoundingBox(x=80, y=650, width=1040, height=100),
                confidence=0.88,
            ),
        ),
        toolbars=(
            Toolbar(
                bounds=BoundingBox(x=0, y=40, width=80, height=710),
                confidence=0.86,
                position="left",
            ),
            Toolbar(
                bounds=BoundingBox(x=1120, y=40, width=80, height=710),
                confidence=0.87,
                position="right",
            ),
        ),
        status_bar=StatusBar(
            bounds=BoundingBox(x=0, y=750, width=1200, height=50),
            confidence=0.84,
        ),
        confidence=0.89,
    )

    assert layout.chart_region.label == "main_chart"
    assert layout.price_axis.label == "price_axis"
    assert layout.time_axis.label == "time_axis"
    assert layout.bottom_panels[0].label == "bottom_panel"
    assert [toolbar.position for toolbar in layout.toolbars] == ["left", "right"]
    assert layout.status_bar is not None
    assert layout.viewport.label == "viewport"


def test_workspace_layout_validates_confidence_and_containment():
    with pytest.raises(ValueError, match="toolbar position"):
        Toolbar(
            bounds=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=0.5,
            position="top",
        )

    with pytest.raises(ValueError, match="price axis confidence"):
        PriceAxis(
            bounds=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=1.5,
        )

    with pytest.raises(ValueError, match="price axis must fit"):
        WorkspaceLayout(
            workspace_id="main",
            frame_id="frame-4",
            bounds=BoundingBox(x=0, y=0, width=100, height=100),
            chart_region=ChartRegion(
                bounds=BoundingBox(x=0, y=0, width=80, height=80),
                confidence=0.9,
            ),
            price_axis=PriceAxis(
                bounds=BoundingBox(x=90, y=0, width=20, height=80),
                confidence=0.9,
            ),
            time_axis=TimeAxis(
                bounds=BoundingBox(x=0, y=80, width=80, height=20),
                confidence=0.9,
            ),
            viewport=Viewport(
                bounds=BoundingBox(x=0, y=0, width=100, height=100),
                confidence=0.9,
            ),
        )
