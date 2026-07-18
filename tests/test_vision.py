from datetime import datetime, timezone

import pytest

from orderflowgpt_genesis import (
    BoundingBox,
    ImageCache,
    ImageFrame,
    InMemoryFrameReplay,
    SceneGraph,
    SceneNode,
    WorkspaceDetection,
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
