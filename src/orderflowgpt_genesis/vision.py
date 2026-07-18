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
import struct
import zlib

FrameId: TypeAlias = str
SceneNodeId: TypeAlias = str
WorkspaceId: TypeAlias = str
ProcessedFrameId: TypeAlias = str
DetectorName: TypeAlias = str


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


ChartRegion: TypeAlias = BoundingBox


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
class DetectionResult:
    """Standard detector output containing a region, confidence, and rationale."""

    region: ChartRegion | None
    confidence: float
    reason: str
    detector_name: DetectorName
    debug_overlay: DebugOverlay | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0.0 and 1.0")
        if not self.reason.strip():
            raise ValueError("detection reason is required")
        if not self.detector_name.strip():
            raise ValueError("detector name is required")
        if self.region is None and self.confidence != 0.0:
            raise ValueError("missing detections must have zero confidence")


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

        chart = ChartRegion(
            bounds=chart_result.region,
            confidence=chart_result.confidence,
        )

        viewport = Viewport(
            bounds=chart_result.region,
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
