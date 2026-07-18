"""Vision Foundation and preprocessing primitives.

This module intentionally defines in-memory abstractions and interfaces only. It does
not perform capture, replay, serialization, persistence, or platform integration.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable
from uuid import uuid4

FrameId: TypeAlias = str
SceneNodeId: TypeAlias = str
WorkspaceId: TypeAlias = str
ProcessedFrameId: TypeAlias = str


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """A captured image and its normalized frame metadata."""

    data: bytes
    width: int
    height: int
    pixel_format: str
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"
    frame_id: FrameId = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("frame data is required")
        if self.width <= 0:
            raise ValueError("frame width must be positive")
        if self.height <= 0:
            raise ValueError("frame height must be positive")
        if not self.pixel_format.strip():
            raise ValueError("pixel format is required")
        if not self.source.strip():
            raise ValueError("frame source is required")
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")


@runtime_checkable
class FrameCapture(Protocol):
    """Interface for components that provide the next frame from a source."""

    def capture_frame(self) -> ImageFrame:
        """Return one newly captured frame."""


@runtime_checkable
class FrameReplay(Protocol):
    """Interface for deterministic frame replay sources."""

    def frames(self) -> Iterable[ImageFrame]:
        """Return frames in replay order."""


class InMemoryFrameReplay:
    """Replay a fixed sequence of frames without file or network I/O."""

    def __init__(self, frames: Sequence[ImageFrame]) -> None:
        self._frames = tuple(frames)

    def frames(self) -> Iterator[ImageFrame]:
        return iter(self._frames)


class ImageCache:
    """A bounded in-memory image cache keyed by frame id."""

    def __init__(self, max_items: int = 128) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self._max_items = max_items
        self._frames: OrderedDict[FrameId, ImageFrame] = OrderedDict()

    @property
    def max_items(self) -> int:
        return self._max_items

    def put(self, frame: ImageFrame) -> None:
        if frame.frame_id in self._frames:
            self._frames.move_to_end(frame.frame_id)
        self._frames[frame.frame_id] = frame
        while len(self._frames) > self._max_items:
            self._frames.popitem(last=False)

    def get(self, frame_id: FrameId) -> ImageFrame | None:
        frame = self._frames.get(frame_id)
        if frame is not None:
            self._frames.move_to_end(frame_id)
        return frame

    def __contains__(self, frame_id: object) -> bool:
        return frame_id in self._frames

    def __len__(self) -> int:
        return len(self._frames)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space rectangle for detected visual elements."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("bounding box origin must be non-negative")
        if self.width <= 0:
            raise ValueError("bounding box width must be positive")
        if self.height <= 0:
            raise ValueError("bounding box height must be positive")

    @property
    def right(self) -> int:
        """Exclusive right edge of the box."""

        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Exclusive bottom edge of the box."""

        return self.y + self.height

    def fits_within(self, width: int, height: int) -> bool:
        """Return whether this box is contained by the supplied dimensions."""

        return self.right <= width and self.bottom <= height


@dataclass(frozen=True, slots=True)
class RegionOfInterest:
    """A named frame region extracted for focused downstream vision processing."""

    name: str
    bounds: BoundingBox

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("region of interest name is required")


@dataclass(frozen=True, slots=True)
class ImagePyramidLevel:
    """A single scale in a multi-scale image pyramid."""

    scale: float
    width: int
    height: int
    data: bytes

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("pyramid scale must be positive")
        if self.width <= 0:
            raise ValueError("pyramid width must be positive")
        if self.height <= 0:
            raise ValueError("pyramid height must be positive")
        if not self.data:
            raise ValueError("pyramid level data is required")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configuration for the side-effect-free image preprocessing pipeline."""

    gaussian_kernel_size: int = 5
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_constant: int = 2
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    morphology_kernel_size: int = 3
    pyramid_scales: tuple[float, ...] = (1.0, 0.5, 0.25)
    zoom_normalization_scale: float = 1.0
    roi_regions: tuple[RegionOfInterest, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_odd(self.gaussian_kernel_size, "gaussian kernel size")
        _require_positive_odd(
            self.adaptive_threshold_block_size,
            "adaptive threshold block size",
        )
        if self.canny_low_threshold < 0:
            raise ValueError("canny low threshold must be non-negative")
        if self.canny_high_threshold <= self.canny_low_threshold:
            raise ValueError("canny high threshold must exceed low threshold")
        _require_positive_odd(self.morphology_kernel_size, "morphology kernel size")
        if not self.pyramid_scales:
            raise ValueError("at least one pyramid scale is required")
        if any(scale <= 0 for scale in self.pyramid_scales):
            raise ValueError("pyramid scales must be positive")
        if self.zoom_normalization_scale <= 0:
            raise ValueError("zoom normalization scale must be positive")


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """A frame after deterministic preprocessing stages have been materialized."""

    source_frame: ImageFrame
    grayscale: ImageFrame
    hsv: ImageFrame
    gaussian_blur: ImageFrame
    adaptive_threshold: ImageFrame
    canny_edges: ImageFrame
    morphology: ImageFrame
    roi_frames: Mapping[str, ImageFrame] = field(default_factory=dict)
    pyramid: tuple[ImagePyramidLevel, ...] = ()
    zoom_normalized: ImageFrame | None = None
    processed_id: ProcessedFrameId = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.processed_id.strip():
            raise ValueError("processed frame id is required")
        for stage_name in (
            "grayscale",
            "hsv",
            "gaussian_blur",
            "adaptive_threshold",
            "canny_edges",
            "morphology",
        ):
            stage = getattr(self, stage_name)
            if stage.width <= 0 or stage.height <= 0:
                raise ValueError(f"{stage_name} dimensions must be positive")
        object.__setattr__(self, "roi_frames", MappingProxyType(dict(self.roi_frames)))


@runtime_checkable
class ImagePreprocessor(Protocol):
    """Interface for components that convert raw frames into processed frames."""

    def preprocess(
        self,
        frame: ImageFrame,
        config: PreprocessingConfig | None = None,
    ) -> ProcessedFrame:
        """Return the processed representation for the supplied image frame."""


class DeterministicImagePreprocessor:
    """In-memory preprocessing pipeline for deterministic tests and adapters.

    The implementation labels derived byte buffers by stage instead of invoking a
    native computer-vision runtime. This keeps the public contracts stable and
    side-effect free while preserving the Milestone 3 pipeline shape.
    """

    def preprocess(
        self,
        frame: ImageFrame,
        config: PreprocessingConfig | None = None,
    ) -> ProcessedFrame:
        active_config = config or PreprocessingConfig()
        _validate_rois(frame, active_config.roi_regions)

        grayscale = _derived_frame(frame, "grayscale", "GRAY")
        hsv = _derived_frame(frame, "hsv", "HSV")
        gaussian_blur = _derived_frame(
            grayscale,
            f"gaussian-blur:{active_config.gaussian_kernel_size}",
            "GRAY",
        )
        adaptive_threshold = _derived_frame(
            gaussian_blur,
            "adaptive-threshold:"
            f"{active_config.adaptive_threshold_block_size}:"
            f"{active_config.adaptive_threshold_constant}",
            "BINARY",
        )
        canny_edges = _derived_frame(
            gaussian_blur,
            "canny-edges:"
            f"{active_config.canny_low_threshold}:"
            f"{active_config.canny_high_threshold}",
            "BINARY",
        )
        morphology = _derived_frame(
            canny_edges,
            f"morphology:{active_config.morphology_kernel_size}",
            "BINARY",
        )
        roi_frames = {
            roi.name: _roi_frame(morphology, roi)
            for roi in active_config.roi_regions
        }
        pyramid = tuple(
            _pyramid_level(morphology, scale) for scale in active_config.pyramid_scales
        )
        zoom_normalized = _scaled_frame(
            morphology,
            active_config.zoom_normalization_scale,
            "zoom-normalized",
        )

        return ProcessedFrame(
            source_frame=frame,
            grayscale=grayscale,
            hsv=hsv,
            gaussian_blur=gaussian_blur,
            adaptive_threshold=adaptive_threshold,
            canny_edges=canny_edges,
            morphology=morphology,
            roi_frames=roi_frames,
            pyramid=pyramid,
            zoom_normalized=zoom_normalized,
        )


def _require_positive_odd(value: int, name: str) -> None:
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")


def _derived_frame(frame: ImageFrame, stage: str, pixel_format: str) -> ImageFrame:
    return ImageFrame(
        data=f"{stage}|".encode("utf-8") + frame.data,
        width=frame.width,
        height=frame.height,
        pixel_format=pixel_format,
        captured_at=frame.captured_at,
        source=f"{frame.source}:{stage}",
        frame_id=f"{frame.frame_id}:{stage}",
    )


def _scaled_frame(frame: ImageFrame, scale: float, stage: str) -> ImageFrame:
    width = max(1, round(frame.width * scale))
    height = max(1, round(frame.height * scale))
    return ImageFrame(
        data=f"{stage}:{scale}|".encode("utf-8") + frame.data,
        width=width,
        height=height,
        pixel_format=frame.pixel_format,
        captured_at=frame.captured_at,
        source=f"{frame.source}:{stage}",
        frame_id=f"{frame.frame_id}:{stage}:{scale}",
    )


def _pyramid_level(frame: ImageFrame, scale: float) -> ImagePyramidLevel:
    scaled = _scaled_frame(frame, scale, "pyramid")
    return ImagePyramidLevel(
        scale=scale,
        width=scaled.width,
        height=scaled.height,
        data=scaled.data,
    )


def _roi_frame(frame: ImageFrame, roi: RegionOfInterest) -> ImageFrame:
    return ImageFrame(
        data=f"roi:{roi.name}:{roi.bounds.x}:{roi.bounds.y}|".encode("utf-8")
        + frame.data,
        width=roi.bounds.width,
        height=roi.bounds.height,
        pixel_format=frame.pixel_format,
        captured_at=frame.captured_at,
        source=f"{frame.source}:roi:{roi.name}",
        frame_id=f"{frame.frame_id}:roi:{roi.name}",
    )


def _validate_rois(frame: ImageFrame, rois: Sequence[RegionOfInterest]) -> None:
    for roi in rois:
        if not roi.bounds.fits_within(frame.width, frame.height):
            raise ValueError("region of interest must fit within the source frame")


@dataclass(frozen=True, slots=True)
class SceneNode:
    """A visual element placeholder in the scene graph skeleton."""

    node_id: SceneNodeId
    label: str
    bounds: BoundingBox
    children: tuple[SceneNodeId, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node id is required")
        if not self.label.strip():
            raise ValueError("node label is required")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True, slots=True)
class SceneGraph:
    """A frame-scoped graph of detected visual elements."""

    frame_id: FrameId
    nodes: tuple[SceneNode, ...] = ()
    root_id: SceneNodeId | None = None

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("scene graph node ids must be unique")
        if self.root_id is not None and self.root_id not in node_ids:
            raise ValueError("root id must reference a scene node")
        for node in self.nodes:
            missing_children = set(node.children) - node_ids
            if missing_children:
                raise ValueError("scene node children must reference scene nodes")


@dataclass(frozen=True, slots=True)
class WorkspaceDetection:
    """Detected workspace region and confidence for a frame."""

    workspace_id: WorkspaceId
    frame_id: FrameId
    bounds: BoundingBox
    confidence: float
    label: str = "workspace"

    def __post_init__(self) -> None:
        if not self.workspace_id.strip():
            raise ValueError("workspace id is required")
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("workspace confidence must be between 0.0 and 1.0")
        if not self.label.strip():
            raise ValueError("workspace label is required")


@runtime_checkable
class WorkspaceDetector(Protocol):
    """Interface for components that locate workspaces in a frame."""

    def detect_workspaces(self, frame: ImageFrame) -> Sequence[WorkspaceDetection]:
        """Return workspace detections for the supplied frame."""
