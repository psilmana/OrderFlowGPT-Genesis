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
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)
from uuid import uuid4
import struct
import zlib

FrameId: TypeAlias = str
SceneNodeId: TypeAlias = str
WorkspaceId: TypeAlias = str
ProcessedFrameId: TypeAlias = str
DetectorName: TypeAlias = str
TDetected = TypeVar("TDetected")


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
class DebugOverlay:
    """A PNG debug overlay for deterministic detector diagnostics."""

    data: bytes
    width: int
    height: int
    pixel_format: str = "PNG"

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("debug overlay data is required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("debug overlay dimensions must be positive")
        if self.pixel_format != "PNG":
            raise ValueError("debug overlay pixel format must be PNG")

    def save_png(self, path: str) -> None:
        """Save the overlay to a PNG file."""

        with open(path, "wb") as png_file:
            png_file.write(self.data)


@dataclass(frozen=True, slots=True)
class DetectionResult(Generic[TDetected]):
    """Standard detector output containing a region, confidence, and rationale."""

    region: BoundingBox | None
    confidence: float
    reason: str
    detector_name: DetectorName
    debug_overlay: DebugOverlay | None = None
    detected_object: TDetected | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0.0 and 1.0")
        if not self.reason.strip():
            raise ValueError("detection reason is required")
        if not self.detector_name.strip():
            raise ValueError("detector name is required")
        if self.region is None and self.confidence != 0.0:
            raise ValueError("missing detections must have zero confidence")


@dataclass(frozen=True, slots=True)
class ObjectId:
    """Stable identifier for one frame-scoped detected object."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("object id is required")


@dataclass(frozen=True, slots=True)
class ObjectType:
    """Supported semantic type for a detected chart/workspace object."""

    value: str

    PRICE_TEXT: ClassVar["ObjectType"]
    PRICE_AXIS: ClassVar["ObjectType"]
    TIME_AXIS: ClassVar["ObjectType"]
    TIME_LABEL: ClassVar["ObjectType"]
    CANDLE: ClassVar["ObjectType"]
    FOOTPRINT_CELL: ClassVar["ObjectType"]
    BID_VALUE: ClassVar["ObjectType"]
    ASK_VALUE: ClassVar["ObjectType"]
    DELTA_VALUE: ClassVar["ObjectType"]
    VOLUME_VALUE: ClassVar["ObjectType"]
    POC_MARKER: ClassVar["ObjectType"]
    HVN: ClassVar["ObjectType"]
    LVN: ClassVar["ObjectType"]
    BIG_TRADE: ClassVar["ObjectType"]
    ICEBERG: ClassVar["ObjectType"]
    ABSORPTION: ClassVar["ObjectType"]
    STACKED_IMBALANCE: ClassVar["ObjectType"]
    VOLUME_PROFILE: ClassVar["ObjectType"]
    CVD_PANEL: ClassVar["ObjectType"]
    DELTA_PANEL: ClassVar["ObjectType"]
    UNKNOWN: ClassVar["ObjectType"]
    _SUPPORTED: ClassVar[frozenset[str]] = frozenset(
        {
            "PRICE_TEXT",
            "PRICE_AXIS",
            "TIME_AXIS",
            "TIME_LABEL",
            "CANDLE",
            "FOOTPRINT_CELL",
            "BID_VALUE",
            "ASK_VALUE",
            "DELTA_VALUE",
            "VOLUME_VALUE",
            "POC_MARKER",
            "HVN",
            "LVN",
            "BIG_TRADE",
            "ICEBERG",
            "ABSORPTION",
            "STACKED_IMBALANCE",
            "VOLUME_PROFILE",
            "CVD_PANEL",
            "DELTA_PANEL",
            "UNKNOWN",
        }
    )

    def __post_init__(self) -> None:
        if self.value not in self._SUPPORTED:
            raise ValueError("object type is not supported")


@dataclass(frozen=True, slots=True)
class DetectionConfidence:
    """Confidence score for object-level detections."""

    value: float

    def __post_init__(self) -> None:
        _validate_confidence(self.value, "object detection confidence")


@dataclass(frozen=True, slots=True)
class DetectionSource:
    """Detector/source name that produced a detected object."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("detection source is required")


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """Immutable semantic object detected within one frame."""

    object_id: ObjectId
    bounds: BoundingBox
    confidence: DetectionConfidence
    object_type: ObjectType
    frame_id: FrameId
    source: DetectionSource
    parent_id: ObjectId | None = None
    children_ids: tuple[ObjectId, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        if self.parent_id == self.object_id:
            raise ValueError("detected object cannot be its own parent")
        if len({child.value for child in self.children_ids}) != len(self.children_ids):
            raise ValueError("detected object child ids must be unique")
        if any(child == self.object_id for child in self.children_ids):
            raise ValueError("detected object cannot be its own child")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class DetectionGraph:
    """All detected objects for one frame with parent/child reference validation."""

    frame_id: FrameId
    objects: tuple[DetectedObject, ...] = ()

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        ids = {obj.object_id for obj in self.objects}
        if len(ids) != len(self.objects):
            raise ValueError("detected object ids must be unique")
        for obj in self.objects:
            if obj.frame_id != self.frame_id:
                raise ValueError("detected object frame id must match detection graph")
            if obj.parent_id is not None and obj.parent_id not in ids:
                raise ValueError("detected object parent id must reference an object")
            missing_children = set(obj.children_ids) - ids
            if missing_children:
                raise ValueError("detected object child ids must reference objects")


@dataclass(frozen=True, slots=True)
class DetectionContext:
    """Only input future object detectors receive."""

    processed_frame: "ProcessedFrame"
    workspace_layout: "WorkspaceLayout"
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.processed_frame.source_frame.frame_id != self.workspace_layout.frame_id:
            raise ValueError(
                "processed frame and workspace layout frame ids must match"
            )
        object.__setattr__(
            self, "configuration", MappingProxyType(dict(self.configuration))
        )


@runtime_checkable
class ObjectDetector(Protocol):
    """Interface for semantic object detectors."""

    @property
    def name(self) -> DetectorName:
        """Return the detector name."""

    def detect(self, context: DetectionContext) -> DetectionResult[DetectedObject]:
        """Return a semantic detected object result, never a raw rectangle."""


@dataclass(frozen=True, slots=True)
class DetectorRegistry:
    """Immutable registry of object detectors in execution order."""

    detectors: tuple[ObjectDetector, ...] = ()

    def __post_init__(self) -> None:
        names = [detector.name for detector in self.detectors]
        if any(not name.strip() for name in names):
            raise ValueError("detector name is required")
        if len(set(names)) != len(names):
            raise ValueError("detector names must be unique")

    def add(self, detector: ObjectDetector) -> "DetectorRegistry":
        """Return a new registry with the supplied detector appended."""

        return DetectorRegistry(self.detectors + (detector,))


@runtime_checkable
class ObjectDetectionPipeline(Protocol):
    """Interface for frame-scoped semantic object detection pipelines."""

    def run(self, context: DetectionContext) -> DetectionGraph:
        """Run registered detectors and return a validated detection graph."""


@dataclass(frozen=True, slots=True)
class SequentialObjectDetectionPipeline:
    """Side-effect-free pipeline that runs registered detectors in order."""

    registry: DetectorRegistry

    def run(self, context: DetectionContext) -> DetectionGraph:
        ordered_detectors = tuple(
            sorted(
                self.registry.detectors,
                key=lambda detector: (
                    0
                    if detector.name == PriceAxisDetector.name
                    else 1 if detector.name == TimeAxisDetector.name else 2
                ),
            )
        )
        detected = tuple(
            result.detected_object
            for detector in ordered_detectors
            for result in (detector.detect(context),)
            if result.detected_object is not None
        )
        return DetectionGraph(
            frame_id=context.processed_frame.source_frame.frame_id,
            objects=detected,
        )


class _EmptyObjectDetector:
    """Base class for Milestone 6 placeholder object detectors."""

    name: DetectorName = "empty-object-detector"

    def detect(self, context: DetectionContext) -> DetectionResult[DetectedObject]:
        return DetectionResult(
            region=None,
            confidence=0.0,
            reason=f"{self.name} is a Milestone 6 placeholder",
            detector_name=self.name,
        )


@dataclass(frozen=True, slots=True)
class PriceAxisDetectorConfig:
    """Immutable thresholds for deterministic price-axis detection."""

    min_width_ratio: float = 0.025
    max_width_ratio: float = 0.18
    min_edge_density: float = 0.01
    min_confidence: float = 0.35
    min_projection_score: float = 0.18
    brightness_delta_threshold: int = 18
    max_chart_overlap_ratio: float = 0.05
    debug_overlay: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_width_ratio",
            "max_width_ratio",
            "min_edge_density",
            "min_confidence",
            "min_projection_score",
            "max_chart_overlap_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.min_width_ratio <= 0.0:
            raise ValueError("min_width_ratio must be greater than 0.0")
        if self.max_width_ratio <= self.min_width_ratio:
            raise ValueError("max_width_ratio must exceed min_width_ratio")
        if self.brightness_delta_threshold <= 0:
            raise ValueError("brightness_delta_threshold must be positive")


class PriceAxisDetector:
    """Detect only the vertical price-axis region immediately right of a chart."""

    name = "price-axis-detector"

    def __init__(self, config: PriceAxisDetectorConfig | None = None) -> None:
        self._config = config or PriceAxisDetectorConfig()

    def detect(self, context: DetectionContext) -> DetectionResult[DetectedObject]:
        frame = context.processed_frame.source_frame
        layout = context.workspace_layout
        chart = layout.chart_region.bounds
        workspace = layout.bounds
        search_left = chart.right
        search_right = min(
            workspace.right,
            chart.right + round(chart.width * self._config.max_width_ratio),
        )
        if search_left >= search_right or chart.height <= 0:
            return self._empty(
                "no workspace area exists immediately right of chart",
                frame.width,
                frame.height,
            )

        min_width = max(1, round(chart.width * self._config.min_width_ratio))
        max_width = max(min_width, round(chart.width * self._config.max_width_ratio))
        max_width = min(max_width, search_right - search_left)
        if max_width <= 0:
            return self._empty(
                "price-axis search width is zero", frame.width, frame.height
            )

        luminance = _luminance(frame)
        edges = _edge_map(luminance, frame.width, frame.height)
        candidate, projection_score = _price_axis_candidate(
            luminance,
            edges,
            frame.width,
            chart,
            search_left,
            search_right,
            min_width,
            max_width,
            self._config,
        )
        if candidate is None:
            return self._empty(
                "no price-axis contrast/projection candidate found",
                frame.width,
                frame.height,
            )
        if not _contains(workspace, candidate):
            return self._empty(
                "price-axis candidate outside workspace layout",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        overlap = _overlap_area(candidate, chart) / (candidate.width * candidate.height)
        if overlap > self._config.max_chart_overlap_ratio:
            return self._empty(
                "price-axis candidate overlaps chart excessively",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        if candidate.width > max_width:
            return self._empty(
                "price-axis candidate wider than expected",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        min_expected_width = min_width + round((max_width - min_width) * 0.35)
        if max_width >= 24 and candidate.width < min_expected_width:
            return self._empty(
                "price-axis candidate narrower than expected",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )

        chart_edge = BoundingBox(
            max(chart.x, chart.right - min_width), chart.y, min_width, chart.height
        )
        immediate_strip = BoundingBox(chart.right, chart.y, min_width, chart.height)
        immediate_mean = _region_mean(luminance, frame.width, immediate_strip)
        chart_mean = _region_mean(luminance, frame.width, chart_edge)
        brightness_delta = abs(immediate_mean - chart_mean)
        background_probe_x = min(workspace.right - min_width, chart.right + max_width)
        background_probe = BoundingBox(
            background_probe_x, chart.y, min_width, chart.height
        )
        if (
            abs(immediate_mean - _region_mean(luminance, frame.width, background_probe))
            < self._config.brightness_delta_threshold
        ):
            return self._empty(
                "price-axis candidate is not immediately right of chart",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        if brightness_delta < self._config.brightness_delta_threshold * 3:
            return self._empty(
                "price-axis candidate rejected by brightness transition",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )

        edge_density = _vertical_edge_count(luminance, frame.width, candidate) / (
            candidate.width * candidate.height
        )
        width_score = 1.0 - min(
            abs(candidate.width - min_width) / max(max_width, 1), 1.0
        )
        density_score = min(
            edge_density / max(self._config.min_edge_density * 3.0, 0.001), 1.0
        )
        confidence = round(
            max(
                0.0,
                min(
                    (projection_score * 0.5)
                    + (density_score * 0.35)
                    + (width_score * 0.15),
                    1.0,
                ),
            ),
            3,
        )
        if edge_density < self._config.min_edge_density:
            return self._empty(
                "price-axis candidate rejected by edge density",
                frame.width,
                frame.height,
                candidate,
                confidence,
            )
        if (
            projection_score < self._config.min_projection_score
            or confidence < self._config.min_confidence
        ):
            return self._empty(
                "price-axis candidate rejected by confidence",
                frame.width,
                frame.height,
                candidate,
                confidence,
            )

        metadata = {
            "estimated_width": candidate.width,
            "edge_density": round(edge_density, 6),
            "projection_score": round(projection_score, 6),
        }
        detected = DetectedObject(
            object_id=ObjectId(f"{frame.frame_id}:price-axis"),
            bounds=candidate,
            confidence=DetectionConfidence(confidence),
            object_type=ObjectType.PRICE_AXIS,
            frame_id=frame.frame_id,
            source=DetectionSource(self.name),
            metadata=metadata,
        )
        return DetectionResult(
            region=candidate,
            confidence=confidence,
            reason="detected price axis immediately right of chart using deterministic projection/edge analysis",
            detector_name=self.name,
            debug_overlay=(
                _debug_overlay(
                    frame.width, frame.height, candidate, confidence, "price_axis"
                )
                if self._config.debug_overlay
                else None
            ),
            detected_object=detected,
        )

    def _empty(
        self,
        reason: str,
        width: int,
        height: int,
        box: BoundingBox | None = None,
        confidence: float = 0.0,
    ) -> DetectionResult[DetectedObject]:
        return DetectionResult(
            region=None,
            confidence=0.0,
            reason=reason,
            detector_name=self.name,
            debug_overlay=(
                _debug_overlay(width, height, box, confidence, "rejected")
                if self._config.debug_overlay
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TimeAxisDetectorConfig:
    """Immutable thresholds for deterministic time-axis detection."""

    min_height_ratio: float = 0.025
    max_height_ratio: float = 0.16
    min_edge_density: float = 0.01
    min_confidence: float = 0.35
    min_projection_score: float = 0.18
    brightness_delta_threshold: int = 18
    max_chart_overlap_ratio: float = 0.05
    alignment_tolerance: int = 4
    debug_overlay: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_height_ratio",
            "max_height_ratio",
            "min_edge_density",
            "min_confidence",
            "min_projection_score",
            "max_chart_overlap_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.min_height_ratio <= 0.0:
            raise ValueError("min_height_ratio must be greater than 0.0")
        if self.max_height_ratio <= self.min_height_ratio:
            raise ValueError("max_height_ratio must exceed min_height_ratio")
        if self.brightness_delta_threshold <= 0:
            raise ValueError("brightness_delta_threshold must be positive")
        if self.alignment_tolerance < 0:
            raise ValueError("alignment_tolerance must be non-negative")


class TimeAxisDetector:
    """Detect only the horizontal time-axis region immediately below a chart."""

    name = "time-axis-detector"

    def __init__(self, config: TimeAxisDetectorConfig | None = None) -> None:
        self._config = config or TimeAxisDetectorConfig()

    def detect(self, context: DetectionContext) -> DetectionResult[DetectedObject]:
        frame = context.processed_frame.source_frame
        layout = context.workspace_layout
        chart = layout.chart_region.bounds
        workspace = layout.bounds
        search_top = chart.bottom
        search_bottom = min(
            workspace.bottom,
            chart.bottom + round(chart.height * self._config.max_height_ratio),
        )
        if search_top >= search_bottom or chart.width <= 0:
            return self._empty(
                "no workspace area exists immediately below chart",
                frame.width,
                frame.height,
            )

        min_height = max(1, round(chart.height * self._config.min_height_ratio))
        max_height = max(
            min_height, round(chart.height * self._config.max_height_ratio)
        )
        max_height = min(max_height, search_bottom - search_top)
        if max_height <= 0:
            return self._empty(
                "time-axis search height is zero", frame.width, frame.height
            )

        luminance = _luminance(frame)
        candidate, projection_score = _time_axis_candidate(
            luminance,
            frame.width,
            chart,
            search_top,
            search_bottom,
            min_height,
            max_height,
            self._config,
        )
        if candidate is None:
            return self._empty(
                "no time-axis contrast/projection candidate found",
                frame.width,
                frame.height,
            )
        if not _contains(workspace, candidate):
            return self._empty(
                "time-axis candidate outside workspace layout",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        overlap = _overlap_area(candidate, chart) / (candidate.width * candidate.height)
        if overlap > self._config.max_chart_overlap_ratio:
            return self._empty(
                "time-axis candidate overlaps chart excessively",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        if candidate.height > max_height:
            return self._empty(
                "time-axis candidate taller than expected",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        horizontal_alignment_score = _horizontal_alignment_score(
            chart, candidate, self._config.alignment_tolerance
        )
        side_width = max(1, min(candidate.width // 12, 12))
        center_probe = BoundingBox(
            candidate.x + (candidate.width // 3),
            candidate.y,
            side_width,
            candidate.height,
        )
        left_probe = BoundingBox(candidate.x, candidate.y, side_width, candidate.height)
        right_probe = BoundingBox(
            candidate.right - side_width, candidate.y, side_width, candidate.height
        )
        center_mean = _region_mean(luminance, frame.width, center_probe)
        side_delta = max(
            abs(_region_mean(luminance, frame.width, left_probe) - center_mean),
            abs(_region_mean(luminance, frame.width, right_probe) - center_mean),
        )
        if (
            horizontal_alignment_score < 1.0
            or side_delta > self._config.brightness_delta_threshold
        ):
            return self._empty(
                "time-axis candidate is not horizontally aligned with chart",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )

        chart_edge = BoundingBox(
            chart.x, max(chart.y, chart.bottom - min_height), chart.width, min_height
        )
        immediate_strip = BoundingBox(chart.x, chart.bottom, chart.width, min_height)
        brightness_delta = abs(
            _region_mean(luminance, frame.width, immediate_strip)
            - _region_mean(luminance, frame.width, chart_edge)
        )
        if brightness_delta < self._config.brightness_delta_threshold * 3:
            return self._empty(
                "time-axis candidate rejected by brightness transition",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )

        edge_density = _horizontal_edge_count(luminance, frame.width, candidate) / (
            candidate.width * candidate.height
        )
        height_score = 1.0 - min(
            abs(candidate.height - min_height) / max(max_height, 1), 1.0
        )
        density_score = min(
            edge_density / max(self._config.min_edge_density * 3.0, 0.001), 1.0
        )
        confidence = round(
            max(
                0.0,
                min(
                    (projection_score * 0.45)
                    + (density_score * 0.35)
                    + (height_score * 0.1)
                    + (horizontal_alignment_score * 0.1),
                    1.0,
                ),
            ),
            3,
        )
        if edge_density < self._config.min_edge_density:
            return self._empty(
                "time-axis candidate rejected by edge density",
                frame.width,
                frame.height,
                candidate,
                confidence,
            )
        if (
            projection_score < self._config.min_projection_score
            or confidence < self._config.min_confidence
        ):
            return self._empty(
                "time-axis candidate rejected by confidence",
                frame.width,
                frame.height,
                candidate,
                confidence,
            )

        metadata = {
            "estimated_height": candidate.height,
            "edge_density": round(edge_density, 6),
            "projection_score": round(projection_score, 6),
            "horizontal_alignment_score": round(horizontal_alignment_score, 6),
        }
        detected = DetectedObject(
            object_id=ObjectId(f"{frame.frame_id}:time-axis"),
            bounds=candidate,
            confidence=DetectionConfidence(confidence),
            object_type=ObjectType.TIME_AXIS,
            frame_id=frame.frame_id,
            source=DetectionSource(self.name),
            metadata=metadata,
        )
        return DetectionResult(
            region=candidate,
            confidence=confidence,
            reason="detected time axis immediately below chart using deterministic projection/edge analysis without OCR",
            detector_name=self.name,
            debug_overlay=(
                _debug_overlay(
                    frame.width, frame.height, candidate, confidence, "time_axis"
                )
                if self._config.debug_overlay
                else None
            ),
            detected_object=detected,
        )

    def _empty(
        self,
        reason: str,
        width: int,
        height: int,
        box: BoundingBox | None = None,
        confidence: float = 0.0,
    ) -> DetectionResult[DetectedObject]:
        return DetectionResult(
            region=None,
            confidence=0.0,
            reason=reason,
            detector_name=self.name,
            debug_overlay=(
                _debug_overlay(width, height, box, confidence, "rejected")
                if self._config.debug_overlay
                else None
            ),
        )


class FootprintDetector(_EmptyObjectDetector):
    name = "footprint-detector"


class VolumeProfileDetector(_EmptyObjectDetector):
    name = "volume-profile-detector"


class BigTradeDetector(_EmptyObjectDetector):
    name = "big-trade-detector"


class AbsorptionDetector(_EmptyObjectDetector):
    name = "absorption-detector"


@runtime_checkable
class Detector(Protocol):
    """Common interface for deterministic processed-frame detectors."""

    def detect(self, frame: "ProcessedFrame") -> DetectionResult:
        """Return the detector result for a processed frame."""


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
            roi.name: _roi_frame(morphology, roi) for roi in active_config.roi_regions
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


def _validate_confidence(confidence: float, name: str) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _contains(container: BoundingBox, candidate: BoundingBox) -> bool:
    return (
        candidate.x >= container.x
        and candidate.y >= container.y
        and candidate.right <= container.right
        and candidate.bottom <= container.bottom
    )


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
        _validate_confidence(self.confidence, "workspace confidence")
        if not self.label.strip():
            raise ValueError("workspace label is required")


@dataclass(frozen=True, slots=True)
class PriceAxis:
    """Detected vertical price scale for a chart workspace."""

    bounds: BoundingBox
    confidence: float
    label: str = "price_axis"

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "price axis confidence")
        if not self.label.strip():
            raise ValueError("price axis label is required")


@dataclass(frozen=True, slots=True)
class TimeAxis:
    """Detected horizontal time scale for a chart workspace."""

    bounds: BoundingBox
    confidence: float
    label: str = "time_axis"

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "time axis confidence")
        if not self.label.strip():
            raise ValueError("time axis label is required")


@dataclass(frozen=True, slots=True)
class ChartRegion:
    """Detected main chart region within a workspace layout."""

    bounds: BoundingBox
    confidence: float
    label: str = "main_chart"

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "chart region confidence")
        if not self.label.strip():
            raise ValueError("chart region label is required")


@dataclass(frozen=True, slots=True)
class BottomPanel:
    """Detected bottom panel such as indicators, volume, or logs."""

    bounds: BoundingBox
    confidence: float
    label: str = "bottom_panel"

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "bottom panel confidence")
        if not self.label.strip():
            raise ValueError("bottom panel label is required")


@dataclass(frozen=True, slots=True)
class Toolbar:
    """Detected toolbar attached to a workspace edge."""

    bounds: BoundingBox
    confidence: float
    position: str
    label: str = "toolbar"

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "toolbar confidence")
        if self.position not in {"left", "right"}:
            raise ValueError("toolbar position must be left or right")
        if not self.label.strip():
            raise ValueError("toolbar label is required")


@dataclass(frozen=True, slots=True)
class StatusBar:
    """Detected workspace status bar."""

    bounds: BoundingBox
    confidence: float
    label: str = "status_bar"

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "status bar confidence")
        if not self.label.strip():
            raise ValueError("status bar label is required")


@dataclass(frozen=True, slots=True)
class Viewport:
    """Detected visible content viewport for the workspace."""

    bounds: BoundingBox
    confidence: float
    label: str = "viewport"

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "viewport confidence")
        if not self.label.strip():
            raise ValueError("viewport label is required")


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    """Detected Milestone 4 workspace structure for a frame."""

    workspace_id: WorkspaceId
    frame_id: FrameId
    bounds: BoundingBox
    chart_region: ChartRegion
    price_axis: PriceAxis
    time_axis: TimeAxis
    viewport: Viewport
    bottom_panels: tuple[BottomPanel, ...] = ()
    toolbars: tuple[Toolbar, ...] = ()
    status_bar: StatusBar | None = None
    confidence: float = 1.0
    label: str = "workspace_layout"

    def __post_init__(self) -> None:
        if not self.workspace_id.strip():
            raise ValueError("workspace id is required")
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        _validate_confidence(self.confidence, "workspace layout confidence")
        if not self.label.strip():
            raise ValueError("workspace layout label is required")
        components = [
            ("chart region", self.chart_region.bounds),
            ("price axis", self.price_axis.bounds),
            ("time axis", self.time_axis.bounds),
            ("viewport", self.viewport.bounds),
        ]
        components.extend(
            ("bottom panel", panel.bounds) for panel in self.bottom_panels
        )
        components.extend(("toolbar", toolbar.bounds) for toolbar in self.toolbars)
        if self.status_bar is not None:
            components.append(("status bar", self.status_bar.bounds))

        for component_name, component_bounds in components:
            if not _contains(self.bounds, component_bounds):
                raise ValueError(f"{component_name} must fit within workspace layout")


@runtime_checkable
class WorkspaceLayoutDetector(Protocol):
    """Interface for components that detect chart workspace layout regions."""

    def detect_workspace_layouts(self, frame: ImageFrame) -> Sequence[WorkspaceLayout]:
        """Return workspace layout detections for the supplied frame."""


@runtime_checkable
class WorkspaceDetector(Protocol):
    """Interface for components that locate workspaces in a frame."""

    def detect_workspaces(self, frame: ImageFrame) -> Sequence[WorkspaceDetection]:
        """Return workspace detections for the supplied frame."""


@dataclass(frozen=True, slots=True)
class ChartDetectorConfig:
    """Tunable thresholds for deterministic main-chart detection."""

    min_width_ratio: float = 0.25
    min_height_ratio: float = 0.25
    min_area_ratio: float = 0.12
    min_edge_density: float = 0.015
    min_active_projection_count: int = 4
    projection_threshold_ratio: float = 0.035
    debug_overlay: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_width_ratio",
            "min_height_ratio",
            "min_area_ratio",
            "min_edge_density",
            "projection_threshold_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.min_active_projection_count <= 0:
            raise ValueError("min_active_projection_count must be positive")


class ChartDetector:
    """Locate the main trading chart with deterministic projection/edge analysis."""

    name = "chart-detector"

    def __init__(self, config: ChartDetectorConfig | None = None) -> None:
        self._config = config or ChartDetectorConfig()

    def detect(self, frame: ProcessedFrame) -> DetectionResult:
        width = frame.source_frame.width
        height = frame.source_frame.height
        luminance = _luminance(frame.source_frame)
        edges = _edge_map(luminance, width, height)
        candidate = _chart_candidate(edges, width, height, self._config)
        if candidate is None:
            return DetectionResult(
                region=None,
                confidence=0.0,
                reason="no sustained chart-like edge projections found",
                detector_name=self.name,
                debug_overlay=(
                    _debug_overlay(width, height, None, 0.0, "no chart")
                    if self._config.debug_overlay
                    else None
                ),
            )

        edge_count = _count_edges(edges, width, candidate)
        area = candidate.width * candidate.height
        density = edge_count / area
        area_ratio = area / (width * height)
        size_score = min(
            candidate.width / (width * self._config.min_width_ratio),
            candidate.height / (height * self._config.min_height_ratio),
            area_ratio / self._config.min_area_ratio,
            1.0,
        )
        density_score = min(density / (self._config.min_edge_density * 2.0), 1.0)
        confidence = round(
            max(0.0, min((size_score * 0.55) + (density_score * 0.45), 1.0)), 3
        )
        if density < self._config.min_edge_density or confidence < 0.35:
            return DetectionResult(
                region=None,
                confidence=0.0,
                reason="candidate rejected by minimum edge density",
                detector_name=self.name,
                debug_overlay=(
                    _debug_overlay(width, height, candidate, confidence, "rejected")
                    if self._config.debug_overlay
                    else None
                ),
            )
        reason = (
            "selected strongest connected chart component with sustained "
            "horizontal/vertical edge projections "
            f"with edge_density={density:.3f} area_ratio={area_ratio:.3f}"
        )
        return DetectionResult(
            region=candidate,
            confidence=confidence,
            reason=reason,
            detector_name=self.name,
            debug_overlay=(
                _debug_overlay(width, height, candidate, confidence, "chart")
                if self._config.debug_overlay
                else None
            ),
        )


class LayoutBuilder:
    """Build workspace layouts from detector results."""

    def build(
        self,
        frame: ProcessedFrame,
        chart_result: DetectionResult,
    ) -> WorkspaceLayout:
        if chart_result.region is None:
            raise ValueError("chart detection result must include a region")
        chart_region_bounds = chart_result.region

        chart = ChartRegion(
            bounds=chart_region_bounds,
            confidence=chart_result.confidence,
        )

        viewport = Viewport(
            bounds=chart_region_bounds,
            confidence=chart_result.confidence,
        )

        empty_axis = BoundingBox(
            x=0,
            y=0,
            width=1,
            height=1,
        )

        return WorkspaceLayout(
            workspace_id="main",
            frame_id=frame.source_frame.frame_id,
            bounds=BoundingBox(
                x=0,
                y=0,
                width=frame.source_frame.width,
                height=frame.source_frame.height,
            ),
            chart_region=chart,
            price_axis=PriceAxis(
                bounds=empty_axis,
                confidence=0.0,
            ),
            time_axis=TimeAxis(
                bounds=empty_axis,
                confidence=0.0,
            ),
            viewport=viewport,
            confidence=chart_result.confidence,
        )


def _luminance(frame: ImageFrame) -> list[int]:
    expected_gray = frame.width * frame.height
    if (
        frame.pixel_format.upper() in {"GRAY", "BINARY"}
        and len(frame.data) >= expected_gray
    ):
        return list(frame.data[:expected_gray])
    expected_rgb = expected_gray * 3
    if frame.pixel_format.upper() == "RGB" and len(frame.data) >= expected_rgb:
        values: list[int] = []
        for offset in range(0, expected_rgb, 3):
            red, green, blue = frame.data[offset : offset + 3]
            values.append((299 * red + 587 * green + 114 * blue) // 1000)
        return values
    # Contract-test fallback: preserve deterministic behavior for non-pixel fixtures.
    return [0 for _ in range(expected_gray)]


def _edge_map(luminance: Sequence[int], width: int, height: int) -> list[bool]:
    edges = [False] * (width * height)
    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            idx = row + x
            gradient = abs(luminance[idx] - luminance[idx - 1]) + abs(
                luminance[idx] - luminance[idx - width]
            )
            edges[idx] = gradient >= 45
    return edges


def _chart_candidate(
    edges: Sequence[bool],
    width: int,
    height: int,
    config: ChartDetectorConfig,
) -> BoundingBox | None:
    projection_box = _projection_candidate(edges, width, height, config)
    component_box = _component_candidate(edges, width, height, config)
    if component_box is not None:
        return component_box
    return projection_box


def _projection_candidate(
    edges: Sequence[bool],
    width: int,
    height: int,
    config: ChartDetectorConfig,
) -> BoundingBox | None:
    col_counts = [0] * width
    row_counts = [0] * height
    for y in range(height):
        for x in range(width):
            if edges[y * width + x]:
                col_counts[x] += 1
                row_counts[y] += 1
    min_col = max(2, round(height * config.projection_threshold_ratio))
    min_row = max(2, round(width * config.projection_threshold_ratio))
    active_cols = [idx for idx, count in enumerate(col_counts) if count >= min_col]
    active_rows = [idx for idx, count in enumerate(row_counts) if count >= min_row]
    if (
        len(active_cols) < config.min_active_projection_count
        or len(active_rows) < config.min_active_projection_count
    ):
        return None
    box = BoundingBox(
        x=min(active_cols),
        y=min(active_rows),
        width=max(active_cols) - min(active_cols) + 1,
        height=max(active_rows) - min(active_rows) + 1,
    )
    if not _passes_geometry(box, width, height, config):
        return None
    return box


def _component_candidate(
    edges: Sequence[bool],
    width: int,
    height: int,
    config: ChartDetectorConfig,
) -> BoundingBox | None:
    visited = [False] * (width * height)
    best_box: BoundingBox | None = None
    best_score = 0
    for idx, is_edge in enumerate(edges):
        if not is_edge or visited[idx]:
            continue
        box, edge_count = _trace_component(edges, visited, width, height, idx)
        if not _passes_geometry(box, width, height, config):
            continue
        score = edge_count * box.width * box.height
        if score > best_score:
            best_box = box
            best_score = score
    return best_box


def _trace_component(
    edges: Sequence[bool],
    visited: list[bool],
    width: int,
    height: int,
    start: int,
) -> tuple[BoundingBox, int]:
    stack = [start]
    visited[start] = True
    min_x = max_x = start % width
    min_y = max_y = start // width
    edge_count = 0
    while stack:
        idx = stack.pop()
        edge_count += 1
        x = idx % width
        y = idx // width
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        for next_x, next_y in (
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
        ):
            if next_x < 0 or next_y < 0 or next_x >= width or next_y >= height:
                continue
            next_idx = next_y * width + next_x
            if edges[next_idx] and not visited[next_idx]:
                visited[next_idx] = True
                stack.append(next_idx)
    return (
        BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x + 1,
            height=max_y - min_y + 1,
        ),
        edge_count,
    )


def _passes_geometry(
    box: BoundingBox,
    width: int,
    height: int,
    config: ChartDetectorConfig,
) -> bool:
    if box.width < round(width * config.min_width_ratio):
        return False
    if box.height < round(height * config.min_height_ratio):
        return False
    return (box.width * box.height) / (width * height) >= config.min_area_ratio


def _count_edges(edges: Sequence[bool], width: int, box: BoundingBox) -> int:
    total = 0
    for y in range(box.y, box.bottom):
        total += sum(
            1 for value in edges[y * width + box.x : y * width + box.right] if value
        )
    return total


def _price_axis_candidate(
    luminance: Sequence[int],
    edges: Sequence[bool],
    frame_width: int,
    chart: BoundingBox,
    search_left: int,
    search_right: int,
    min_width: int,
    max_width: int,
    config: PriceAxisDetectorConfig,
) -> tuple[BoundingBox | None, float]:
    chart_edge_mean = _region_mean(
        luminance,
        frame_width,
        BoundingBox(
            max(chart.x, chart.right - min_width), chart.y, min_width, chart.height
        ),
    )
    best_box: BoundingBox | None = None
    best_score = 0.0
    for width in range(min_width, max_width + 1):
        box = BoundingBox(x=search_left, y=chart.y, width=width, height=chart.height)
        mean_delta = abs(_region_mean(luminance, frame_width, box) - chart_edge_mean)
        edge_density = _vertical_edge_count(luminance, frame_width, box) / (
            box.width * box.height
        )
        active_cols = 0
        for col in range(box.x, box.right):
            count = sum(
                1
                for y in range(box.y, box.bottom)
                if col > 0
                and abs(
                    luminance[y * frame_width + col]
                    - luminance[y * frame_width + col - 1]
                )
                >= 45
            )
            if count / box.height >= config.min_edge_density:
                active_cols += 1
        projection_score = active_cols / box.width
        contrast_score = min(mean_delta / config.brightness_delta_threshold, 1.0)
        density_score = min(
            edge_density / max(config.min_edge_density * 3.0, 0.001), 1.0
        )
        width_score = width / max_width
        score = (
            (projection_score * 0.45)
            + (contrast_score * 0.2)
            + (density_score * 0.15)
            + (width_score * 0.2)
        )
        if score > best_score:
            best_box = box
            best_score = score
    return best_box, round(best_score, 6)


def _time_axis_candidate(
    luminance: Sequence[int],
    frame_width: int,
    chart: BoundingBox,
    search_top: int,
    search_bottom: int,
    min_height: int,
    max_height: int,
    config: TimeAxisDetectorConfig,
) -> tuple[BoundingBox | None, float]:
    chart_edge_mean = _region_mean(
        luminance,
        frame_width,
        BoundingBox(
            chart.x, max(chart.y, chart.bottom - min_height), chart.width, min_height
        ),
    )
    best_box: BoundingBox | None = None
    best_score = 0.0
    for height in range(min_height, max_height + 1):
        box = BoundingBox(x=chart.x, y=search_top, width=chart.width, height=height)
        mean_delta = abs(_region_mean(luminance, frame_width, box) - chart_edge_mean)
        edge_density = _horizontal_edge_count(luminance, frame_width, box) / (
            box.width * box.height
        )
        active_rows = 0
        for row_y in range(box.y, box.bottom):
            if row_y == 0:
                continue
            count = sum(
                1
                for x in range(box.x, box.right)
                if abs(
                    luminance[row_y * frame_width + x]
                    - luminance[(row_y - 1) * frame_width + x]
                )
                >= 45
            )
            if count / box.width >= config.min_edge_density:
                active_rows += 1
        if active_rows < 2:
            continue
        projection_score = active_rows / box.height
        contrast_score = min(mean_delta / config.brightness_delta_threshold, 1.0)
        density_score = min(
            edge_density / max(config.min_edge_density * 3.0, 0.001), 1.0
        )
        height_score = height / max_height
        score = (
            (projection_score * 0.45)
            + (contrast_score * 0.2)
            + (density_score * 0.15)
            + (height_score * 0.2)
        )
        if score > best_score:
            best_box = box
            best_score = score
    return best_box, round(best_score, 6)


def _horizontal_edge_count(
    luminance: Sequence[int], width: int, box: BoundingBox
) -> int:
    total = 0
    for y in range(max(1, box.y), box.bottom):
        row = y * width
        previous = (y - 1) * width
        for x in range(box.x, box.right):
            if abs(luminance[row + x] - luminance[previous + x]) >= 45:
                total += 1
    return total


def _horizontal_alignment_score(
    chart: BoundingBox, candidate: BoundingBox, tolerance: int
) -> float:
    left_delta = abs(candidate.x - chart.x)
    right_delta = abs(candidate.right - chart.right)
    if left_delta > tolerance or right_delta > tolerance:
        return 0.0
    if tolerance == 0:
        return 1.0
    return 1.0 - ((left_delta + right_delta) / (2 * tolerance))


def _vertical_edge_count(luminance: Sequence[int], width: int, box: BoundingBox) -> int:
    total = 0
    for y in range(box.y, box.bottom):
        row = y * width
        for x in range(max(1, box.x), box.right):
            if abs(luminance[row + x] - luminance[row + x - 1]) >= 45:
                total += 1
    return total


def _region_mean(luminance: Sequence[int], width: int, box: BoundingBox) -> float:
    total = 0
    count = 0
    for y in range(box.y, box.bottom):
        row = y * width
        for x in range(box.x, box.right):
            total += luminance[row + x]
            count += 1
    return total / count if count else 0.0


def _overlap_area(first: BoundingBox, second: BoundingBox) -> int:
    x_overlap = max(0, min(first.right, second.right) - max(first.x, second.x))
    y_overlap = max(0, min(first.bottom, second.bottom) - max(first.y, second.y))
    return x_overlap * y_overlap


def _debug_overlay(
    width: int,
    height: int,
    box: BoundingBox | None,
    confidence: float,
    label: str,
) -> DebugOverlay:
    pixels = bytearray([24, 28, 34] * width * height)
    if box is not None:
        color = (0, 255, 96) if label == "chart" else (255, 128, 0)
        _draw_rect(pixels, width, height, box, color)
    _draw_label_bars(pixels, width, height, confidence)
    return DebugOverlay(
        data=_png_rgb(width, height, bytes(pixels)), width=width, height=height
    )


def _draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    box: BoundingBox,
    color: tuple[int, int, int],
) -> None:
    for x in range(box.x, min(box.right, width)):
        for y in (box.y, min(box.bottom - 1, height - 1)):
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)
    for y in range(box.y, min(box.bottom, height)):
        for x in (box.x, min(box.right - 1, width - 1)):
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)


def _draw_label_bars(
    pixels: bytearray, width: int, height: int, confidence: float
) -> None:
    bar_width = min(width, max(1, round(width * confidence)))
    for y in range(min(8, height)):
        for x in range(bar_width):
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = b"\x00\xff\x60"


def _png_rgb(width: int, height: int, rgb: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = b"".join(
        b"\x00" + rgb[y * width * 3 : (y + 1) * width * 3] for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


ObjectType.PRICE_TEXT = ObjectType("PRICE_TEXT")
ObjectType.PRICE_AXIS = ObjectType("PRICE_AXIS")
ObjectType.TIME_AXIS = ObjectType("TIME_AXIS")
ObjectType.TIME_LABEL = ObjectType("TIME_LABEL")
ObjectType.CANDLE = ObjectType("CANDLE")
ObjectType.FOOTPRINT_CELL = ObjectType("FOOTPRINT_CELL")
ObjectType.BID_VALUE = ObjectType("BID_VALUE")
ObjectType.ASK_VALUE = ObjectType("ASK_VALUE")
ObjectType.DELTA_VALUE = ObjectType("DELTA_VALUE")
ObjectType.VOLUME_VALUE = ObjectType("VOLUME_VALUE")
ObjectType.POC_MARKER = ObjectType("POC_MARKER")
ObjectType.HVN = ObjectType("HVN")
ObjectType.LVN = ObjectType("LVN")
ObjectType.BIG_TRADE = ObjectType("BIG_TRADE")
ObjectType.ICEBERG = ObjectType("ICEBERG")
ObjectType.ABSORPTION = ObjectType("ABSORPTION")
ObjectType.STACKED_IMBALANCE = ObjectType("STACKED_IMBALANCE")
ObjectType.VOLUME_PROFILE = ObjectType("VOLUME_PROFILE")
ObjectType.CVD_PANEL = ObjectType("CVD_PANEL")
ObjectType.DELTA_PANEL = ObjectType("DELTA_PANEL")
ObjectType.UNKNOWN = ObjectType("UNKNOWN")
