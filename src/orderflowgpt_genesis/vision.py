"""Vision Foundation and preprocessing primitives.

This module intentionally defines in-memory abstractions and interfaces only. It does
not perform capture, replay, serialization, persistence, or platform integration.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from enum import Enum
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


class ImbalanceType(Enum):
    """Supported single-cell footprint imbalance classifications."""

    ASK_IMBALANCE = "ASK_IMBALANCE"
    BID_IMBALANCE = "BID_IMBALANCE"
    NONE = "NONE"


class StackedImbalanceType(Enum):
    """Supported vertically stacked footprint imbalance classifications."""

    STACKED_ASK = "STACKED_ASK"
    STACKED_BID = "STACKED_BID"
    NONE = "NONE"


class AbsorptionType(Enum):
    """Supported deterministic footprint absorption classifications."""

    BUY_ABSORPTION = "BUY_ABSORPTION"
    SELL_ABSORPTION = "SELL_ABSORPTION"
    NONE = "NONE"


class DeltaType(Enum):
    """Deterministic sign classification for footprint delta values."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    ZERO = "ZERO"


class VolumeClusterType(Enum):
    """Supported deterministic single-cell volume classifications."""

    HIGH_VOLUME = "HIGH_VOLUME"
    LOW_VOLUME = "LOW_VOLUME"
    NORMAL_VOLUME = "NORMAL_VOLUME"


class PointOfControlType(Enum):
    """Supported deterministic point-of-control classifications."""

    SESSION_POC = "SESSION_POC"


class UnfinishedAuctionType(Enum):
    """Supported deterministic unfinished-auction boundary types."""

    TOP = "TOP"
    BOTTOM = "BOTTOM"


class ExcessType(Enum):
    """Supported deterministic auction excess boundary types."""

    EXCESS_HIGH = "EXCESS_HIGH"
    EXCESS_LOW = "EXCESS_LOW"


class PoorAuctionType(Enum):
    """Supported deterministic poor-auction boundary types."""

    POOR_HIGH = "POOR_HIGH"
    POOR_LOW = "POOR_LOW"


class AbsorptionSide(Enum):
    """Passive side inferred for deterministic absorption observations."""

    BID = "BID"
    ASK = "ASK"
    NONE = "NONE"


class ImbalanceSide(Enum):
    """Dominant side for a detected single-cell imbalance."""

    ASK = "ASK"
    BID = "BID"
    NONE = "NONE"


class CellSemanticRole(Enum):
    """Logical semantic regions supported inside a footprint cell."""

    UNKNOWN = "UNKNOWN"
    BID_REGION = "BID_REGION"
    ASK_REGION = "ASK_REGION"
    DELTA_REGION = "DELTA_REGION"
    CENTER_REGION = "CENTER_REGION"
    BACKGROUND = "BACKGROUND"
    EMPTY = "EMPTY"


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
    detected_objects: tuple[TDetected, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0.0 and 1.0")
        if not self.reason.strip():
            raise ValueError("detection reason is required")
        if not self.detector_name.strip():
            raise ValueError("detector name is required")
        if self.region is None and self.confidence != 0.0:
            raise ValueError("missing detections must have zero confidence")
        objects = self.detected_objects
        if self.detected_object is not None and not objects:
            objects = (self.detected_object,)
        object.__setattr__(self, "detected_objects", tuple(objects))


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

    CHART: ClassVar["ObjectType"]
    PRICE_TEXT: ClassVar["ObjectType"]
    PRICE_AXIS: ClassVar["ObjectType"]
    TIME_AXIS: ClassVar["ObjectType"]
    TIME_LABEL: ClassVar["ObjectType"]
    CANDLE: ClassVar["ObjectType"]
    FOOTPRINT_GRID: ClassVar["ObjectType"]
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
            "CHART",
            "PRICE_TEXT",
            "PRICE_AXIS",
            "TIME_AXIS",
            "TIME_LABEL",
            "CANDLE",
            "FOOTPRINT_GRID",
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
    grid_coordinate_system: GridCoordinateSystem | None = None
    cell_classifications: tuple["CellClassification", ...] = ()
    ocr_results: tuple["OCRResult", ...] = ()
    footprint_interpretation: "FootprintInterpretation | None" = None
    parsed_values: tuple["ParsingResult", ...] = ()
    footprint_matrix: "FootprintMatrix | None" = None
    footprint_imbalances: "FootprintImbalanceResult | None" = None
    stacked_imbalances: "StackedImbalanceResult | None" = None
    absorption: "AbsorptionResult | None" = None
    footprint_delta: "DeltaResult | None" = None
    volume_clusters: "VolumeClusterResult | None" = None
    point_of_control: "PointOfControlResult | None" = None
    high_volume_nodes: "HighVolumeNodeResult | None" = None
    low_volume_nodes: "LowVolumeNodeResult | None" = None
    value_area: "ValueAreaResult | None" = None
    developing_poc: "DevelopingPointOfControlResult | None" = None
    developing_value_area: "DevelopingValueAreaResult | None" = None
    unfinished_auctions: "UnfinishedAuctionResult | None" = None
    excess: "ExcessResult | None" = None
    poor_auctions: "PoorAuctionResult | None" = None
    single_prints: "SinglePrintResult | None" = None
    naked_pocs: "NakedPointOfControlResult | None" = None

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
        if self.grid_coordinate_system is not None:
            grid_ids = {
                str(obj.metadata.get("grid_id", ""))
                for obj in self.objects
                if obj.object_type == ObjectType.FOOTPRINT_CELL
            }
            if self.grid_coordinate_system.grid_id not in grid_ids:
                raise ValueError("grid coordinate system must reference detected cells")
        cell_ids = {
            str(obj.metadata.get("cell_id", ""))
            for obj in self.objects
            if obj.object_type == ObjectType.FOOTPRINT_CELL
        }
        classification_ids = {
            classification.cell_reference.coordinate.cell_id
            for classification in self.cell_classifications
        }
        if len(classification_ids) != len(self.cell_classifications):
            raise ValueError("cell classifications must reference unique cells")
        if classification_ids - cell_ids:
            raise ValueError("cell classifications must reference detected cells")
        for classification in self.cell_classifications:
            if classification.cell_reference.frame_id != self.frame_id:
                raise ValueError("cell classification frame id must match graph")
        result_keys = {
            (result.cell_id, result.semantic_role) for result in self.ocr_results
        }
        if len(result_keys) != len(self.ocr_results):
            raise ValueError("ocr results must reference unique cell regions")
        if any(result.frame_id != self.frame_id for result in self.ocr_results):
            raise ValueError("ocr result frame id must match graph")
        parsed_cell_ids = {
            result.parsed_value.cell_id
            for result in self.parsed_values
            if result.parsed_value.cell_id
        }
        if parsed_cell_ids - cell_ids:
            raise ValueError("parsed values must reference detected cells")
        if result_keys and not result_keys.issubset(
            {
                (classification.cell_id, region.semantic_role)
                for classification in self.cell_classifications
                for region in classification.semantic_regions
            }
        ):
            raise ValueError("ocr results must reference classified cell regions")
        if self.footprint_interpretation is not None:
            if self.grid_coordinate_system is None:
                raise ValueError("footprint interpretation requires coordinate system")
            if (
                self.footprint_interpretation.grid_id
                != self.grid_coordinate_system.grid_id
            ):
                raise ValueError(
                    "footprint interpretation grid id must match coordinate system"
                )
            interpreted_ids = {
                cell.cell_reference.coordinate.cell_id
                for cell in self.footprint_interpretation.ordered_cells
            }
            if interpreted_ids - cell_ids:
                raise ValueError(
                    "footprint interpretation must reference detected cells"
                )
        if self.footprint_matrix is not None:
            if self.grid_coordinate_system is None:
                raise ValueError("footprint matrix requires coordinate system")
            if self.footprint_interpretation is None:
                raise ValueError("footprint matrix requires interpretation")
            matrix_ids = {cell.cell_id for cell in self.footprint_matrix.cells}
            if matrix_ids != cell_ids:
                raise ValueError("footprint matrix must reference detected cells")
            if (
                self.footprint_matrix.dimensions_value.rows
                != self.grid_coordinate_system.row_count
                or self.footprint_matrix.dimensions_value.columns
                != self.grid_coordinate_system.column_count
            ):
                raise ValueError("footprint matrix dimensions must match grid")
        if self.footprint_imbalances is not None:
            if self.footprint_matrix is None:
                raise ValueError("footprint imbalances require footprint matrix")
            if self.footprint_imbalances.matrix != self.footprint_matrix:
                raise ValueError("footprint imbalances must reference graph matrix")
        if self.stacked_imbalances is not None:
            if self.footprint_matrix is None or self.footprint_imbalances is None:
                raise ValueError("stacked imbalances require matrix and imbalances")
            if self.stacked_imbalances.matrix != self.footprint_matrix:
                raise ValueError("stacked imbalances must reference graph matrix")
            if self.stacked_imbalances.imbalances != self.footprint_imbalances:
                raise ValueError("stacked imbalances must reference graph imbalances")
        if self.absorption is not None:
            if self.footprint_matrix is None or self.footprint_imbalances is None:
                raise ValueError("absorption requires matrix and imbalances")
            if self.absorption.matrix != self.footprint_matrix:
                raise ValueError("absorption must reference graph matrix")
            if self.absorption.imbalances != self.footprint_imbalances:
                raise ValueError("absorption must reference graph imbalances")
        if self.footprint_delta is not None:
            if self.footprint_matrix is None:
                raise ValueError("footprint delta requires footprint matrix")
            if self.footprint_delta.matrix != self.footprint_matrix:
                raise ValueError("footprint delta must reference graph matrix")
        if self.volume_clusters is not None:
            if self.footprint_matrix is None:
                raise ValueError("volume clusters require footprint matrix")
            if self.volume_clusters.matrix != self.footprint_matrix:
                raise ValueError("volume clusters must reference graph matrix")
        if self.point_of_control is not None:
            if self.footprint_matrix is None:
                raise ValueError("point of control requires footprint matrix")
            if self.point_of_control.matrix != self.footprint_matrix:
                raise ValueError("point of control must reference graph matrix")
        if self.high_volume_nodes is not None:
            if self.footprint_matrix is None:
                raise ValueError("high volume nodes require footprint matrix")
            if self.high_volume_nodes.matrix != self.footprint_matrix:
                raise ValueError("high volume nodes must reference graph matrix")
        if self.low_volume_nodes is not None:
            if self.footprint_matrix is None:
                raise ValueError("low volume nodes require footprint matrix")
            if self.low_volume_nodes.matrix != self.footprint_matrix:
                raise ValueError("low volume nodes must reference graph matrix")
        if self.value_area is not None:
            if self.footprint_matrix is None:
                raise ValueError("value area requires footprint matrix")
            if self.value_area.matrix != self.footprint_matrix:
                raise ValueError("value area must reference graph matrix")
            if (
                self.point_of_control is not None
                and self.value_area.poc != self.point_of_control.poc
            ):
                raise ValueError("value area poc must reference graph point of control")
        for name, result in (
            ("developing poc", self.developing_poc),
            ("developing value area", self.developing_value_area),
            ("unfinished auctions", self.unfinished_auctions),
            ("excess", self.excess),
            ("poor auctions", self.poor_auctions),
            ("single prints", self.single_prints),
            ("naked pocs", self.naked_pocs),
        ):
            if result is not None:
                if self.footprint_matrix is None:
                    raise ValueError(f"{name} requires footprint matrix")
                if result.matrix != self.footprint_matrix:
                    raise ValueError(f"{name} must reference graph matrix")

    @property
    def footprint_cells(self) -> tuple[DetectedObject, ...]:
        """Return detected footprint cells in deterministic graph order."""

        return tuple(
            obj for obj in self.objects if obj.object_type == ObjectType.FOOTPRINT_CELL
        )

    @property
    def CellClassifications(self) -> tuple["CellClassification", ...]:
        """Compatibility-style alias exposing classified footprint cells."""

        return self.cell_classifications

    def region_text(self, role: "CellSemanticRole") -> tuple[str, ...]:
        """Return raw OCR text for all results matching a semantic role."""

        return tuple(
            result.text() for result in self.ocr_results if result.semantic_role == role
        )

    def lookup(self, cell_id: str) -> tuple["OCRResult", ...]:
        """Return raw OCR results for one cell id in deterministic order."""

        return tuple(result for result in self.ocr_results if result.cell_id == cell_id)

    def lookup_cell(self, cell_id: str) -> "FootprintCellData | None":
        """Return interpreted footprint cell data by cell id when available."""

        if self.footprint_interpretation is None:
            return None
        return self.footprint_interpretation.lookup_cell(cell_id)

    def lookup_bid(self, cell_id: str) -> "FootprintValue | None":
        cell = self.lookup_cell(cell_id)
        return None if cell is None else cell.bid()

    def lookup_ask(self, cell_id: str) -> "FootprintValue | None":
        cell = self.lookup_cell(cell_id)
        return None if cell is None else cell.ask()

    def lookup_delta(self, cell_id: str) -> "FootprintValue | None":
        cell = self.lookup_cell(cell_id)
        return None if cell is None else cell.delta()

    def lookup_total_volume(self, cell_id: str) -> "FootprintValue | None":
        cell = self.lookup_cell(cell_id)
        return None if cell is None else cell.total_volume()

    def lookup_parsed(self, cell_id: str) -> tuple["ParsingResult", ...]:
        """Return parsed OCR results for one cell id in deterministic order."""

        return tuple(
            result
            for result in self.parsed_values
            if result.parsed_value.cell_id == cell_id
        )

    def lookup_numeric(self, cell_id: str) -> tuple[NumericValue, ...]:
        """Return successfully parsed numeric values for one cell id."""

        return tuple(
            result.parsed_value.parsed_number
            for result in self.lookup_parsed(cell_id)
            if result.parsed_value.parsed_number is not None
        )

    def lookup_invalid(self) -> tuple["ParsingResult", ...]:
        """Return parsed OCR results that did not produce valid numbers."""

        return tuple(result for result in self.parsed_values if not result.is_valid())

    def matrix_cell(self, row: int, column: int) -> "MatrixCell":
        if self.footprint_matrix is None:
            raise ValueError("footprint matrix is not available")
        return self.footprint_matrix.cell(row, column)

    def matrix_row(self, index: int) -> "MatrixRow":
        if self.footprint_matrix is None:
            raise ValueError("footprint matrix is not available")
        return self.footprint_matrix.row(index)

    def matrix_column(self, index: int) -> tuple["MatrixCell", ...]:
        if self.footprint_matrix is None:
            raise ValueError("footprint matrix is not available")
        return self.footprint_matrix.column(index)

    def matrix_statistics(self) -> "MatrixStatistics":
        if self.footprint_matrix is None:
            raise ValueError("footprint matrix is not available")
        return self.footprint_matrix.statistics()

    def imbalances(self) -> tuple["FootprintImbalance", ...]:
        if self.footprint_imbalances is None:
            return ()
        return self.footprint_imbalances.imbalances()

    def ask_imbalances(self) -> tuple["FootprintImbalance", ...]:
        if self.footprint_imbalances is None:
            return ()
        return self.footprint_imbalances.ask_imbalances()

    def bid_imbalances(self) -> tuple["FootprintImbalance", ...]:
        if self.footprint_imbalances is None:
            return ()
        return self.footprint_imbalances.bid_imbalances()

    def lookup_imbalance(self, cell_id: str) -> "FootprintImbalance | None":
        if self.footprint_imbalances is None:
            return None
        return self.footprint_imbalances.lookup(cell_id)

    def has_imbalance(self, cell_id: str) -> bool:
        return self.lookup_imbalance(cell_id) is not None

    def imbalance_statistics(self) -> "ImbalanceStatistics":
        if self.footprint_imbalances is None:
            if self.footprint_matrix is None:
                raise ValueError("footprint imbalances are not available")
            total = self.footprint_matrix.statistics().total_cells
            return ImbalanceStatistics(total, 0, 0, 0, total)
        return self.footprint_imbalances.statistics()

    def stacks(self) -> tuple["StackedImbalance", ...]:
        if self.stacked_imbalances is None:
            return ()
        return self.stacked_imbalances.stacks()

    def ask_stacks(self) -> tuple["StackedImbalance", ...]:
        if self.stacked_imbalances is None:
            return ()
        return self.stacked_imbalances.ask_stacks()

    def bid_stacks(self) -> tuple["StackedImbalance", ...]:
        if self.stacked_imbalances is None:
            return ()
        return self.stacked_imbalances.bid_stacks()

    def lookup_stack(self, stack_id: str) -> "StackedImbalance | None":
        if self.stacked_imbalances is None:
            return None
        return self.stacked_imbalances.lookup(stack_id)

    def lookup_stacks_by_cell(self, cell_id: str) -> tuple["StackedImbalance", ...]:
        if self.stacked_imbalances is None:
            return ()
        return self.stacked_imbalances.lookup_by_cell(cell_id)

    def stacked_imbalance_statistics(self) -> "StackedImbalanceStatistics":
        if self.stacked_imbalances is None:
            return StackedImbalanceStatistics(0, 0, 0, 0, Decimal("0"), 0)
        return self.stacked_imbalances.statistics()

    def absorptions(self) -> tuple["FootprintAbsorption", ...]:
        if self.absorption is None:
            return ()
        return self.absorption.absorptions()

    def buy_absorptions(self) -> tuple["FootprintAbsorption", ...]:
        if self.absorption is None:
            return ()
        return self.absorption.buy_absorptions()

    def sell_absorptions(self) -> tuple["FootprintAbsorption", ...]:
        if self.absorption is None:
            return ()
        return self.absorption.sell_absorptions()

    def lookup_absorption(self, cell_id: str) -> "FootprintAbsorption | None":
        if self.absorption is None:
            return None
        return self.absorption.lookup(cell_id)

    def absorption_statistics(self) -> "AbsorptionStatistics":
        if self.absorption is None:
            total = (
                0
                if self.footprint_matrix is None
                else self.footprint_matrix.statistics().total_cells
            )
            return AbsorptionStatistics(total, 0, 0, 0, total)
        return self.absorption.statistics()

    def cell_delta(self, cell_id: str) -> "CellDelta | None":
        if self.footprint_delta is None:
            return None
        return self.footprint_delta.cell_delta(cell_id)

    def row_delta(self, row_index: int) -> "RowDelta | None":
        if self.footprint_delta is None:
            return None
        return self.footprint_delta.row_delta(row_index)

    def positive_cells(self) -> tuple["CellDelta", ...]:
        if self.footprint_delta is None:
            return ()
        return self.footprint_delta.positive_cells()

    def negative_cells(self) -> tuple["CellDelta", ...]:
        if self.footprint_delta is None:
            return ()
        return self.footprint_delta.negative_cells()

    def zero_cells(self) -> tuple["CellDelta", ...]:
        if self.footprint_delta is None:
            return ()
        return self.footprint_delta.zero_cells()

    def delta_statistics(self) -> "DeltaStatistics":
        if self.footprint_delta is None:
            if self.footprint_matrix is None:
                raise ValueError("footprint delta is not available")
            cells = self.footprint_matrix.statistics().total_cells
            return DeltaStatistics(
                self.footprint_matrix.dimensions_value.rows,
                cells,
                0,
                0,
                cells,
                Decimal("0"),
                Decimal("0"),
                Decimal("0"),
            )
        return self.footprint_delta.statistics()

    def high_volume_cells(self) -> tuple["VolumeCluster", ...]:
        if self.volume_clusters is None:
            return ()
        return self.volume_clusters.high_volume_cells()

    def low_volume_cells(self) -> tuple["VolumeCluster", ...]:
        if self.volume_clusters is None:
            return ()
        return self.volume_clusters.low_volume_cells()

    def normal_volume_cells(self) -> tuple["VolumeCluster", ...]:
        if self.volume_clusters is None:
            return ()
        return self.volume_clusters.normal_volume_cells()

    def lookup_volume_cluster(self, cell_id: str) -> "VolumeCluster | None":
        if self.volume_clusters is None:
            return None
        return self.volume_clusters.lookup(cell_id)

    def volume_cluster_statistics(self) -> "VolumeClusterStatistics":
        if self.volume_clusters is None:
            if self.footprint_matrix is None:
                raise ValueError("volume clusters are not available")
            total = self.footprint_matrix.statistics().total_cells
            return VolumeClusterStatistics(
                total, 0, 0, total, Decimal("0"), Decimal("0"), Decimal("0")
            )
        return self.volume_clusters.statistics()

    def session_poc(self) -> "PointOfControl | None":
        if self.point_of_control is None:
            return None
        return self.point_of_control.poc

    def point_of_control_statistics(self) -> "PointOfControlStatistics":
        if self.point_of_control is None:
            if self.footprint_matrix is None:
                raise ValueError("point of control is not available")
            return PointOfControlAnalyzer().analyze(self.footprint_matrix).statistics()
        return self.point_of_control.statistics()

    def lookup_high_volume_node(self, row: int) -> "HighVolumeNode | None":
        if self.high_volume_nodes is None:
            return None
        return self.high_volume_nodes.lookup(row)

    def lookup_low_volume_node(self, row: int) -> "LowVolumeNode | None":
        if self.low_volume_nodes is None:
            return None
        return self.low_volume_nodes.lookup(row)

    def high_volume_node_statistics(self) -> "HighVolumeNodeStatistics":
        if self.high_volume_nodes is None:
            if self.footprint_matrix is None:
                raise ValueError("high volume nodes are not available")
            rows = self.footprint_matrix.dimensions_value.rows
            return HighVolumeNodeStatistics(rows, 0, Decimal("0"), Decimal("80"))
        return self.high_volume_nodes.statistics()

    def low_volume_node_statistics(self) -> "LowVolumeNodeStatistics":
        if self.low_volume_nodes is None:
            if self.footprint_matrix is None:
                raise ValueError("low volume nodes are not available")
            rows = self.footprint_matrix.dimensions_value.rows
            return LowVolumeNodeStatistics(rows, 0, Decimal("0"), Decimal("20"))
        return self.low_volume_nodes.statistics()

    def value_area_statistics(self) -> "ValueAreaStatistics":
        if self.value_area is None:
            if self.footprint_matrix is None:
                raise ValueError("value area is not available")
            return ValueAreaAnalyzer().analyze(self.footprint_matrix).statistics()
        return self.value_area.statistics()

    def developing_poc_statistics(self) -> "DevelopingPointOfControlStatistics":
        if self.developing_poc is None:
            if self.footprint_matrix is None:
                raise ValueError("developing poc is not available")
            return (
                DevelopingPointOfControlAnalyzer()
                .analyze(self.footprint_matrix)
                .statistics()
            )
        return self.developing_poc.statistics()

    def developing_value_area_statistics(self) -> "DevelopingValueAreaStatistics":
        if self.developing_value_area is None:
            if self.footprint_matrix is None:
                raise ValueError("developing value area is not available")
            return (
                DevelopingValueAreaAnalyzer()
                .analyze(self.footprint_matrix)
                .statistics()
            )
        return self.developing_value_area.statistics()

    def lookup_unfinished_auction(
        self, auction_type: "UnfinishedAuctionType"
    ) -> "UnfinishedAuction | None":
        if self.unfinished_auctions is None:
            return None
        return self.unfinished_auctions.lookup(auction_type)

    def unfinished_auction_statistics(self) -> "UnfinishedAuctionStatistics":
        if self.unfinished_auctions is None:
            return UnfinishedAuctionStatistics(0, 0, 0)
        return self.unfinished_auctions.statistics()

    def lookup_excess(self, excess_type: "ExcessType") -> "Excess | None":
        if self.excess is None:
            return None
        return self.excess.lookup(excess_type)

    def excess_statistics(self) -> "ExcessStatistics":
        if self.excess is None:
            return ExcessStatistics(0, 0, 0)
        return self.excess.statistics()

    def lookup_poor_auction(
        self, auction_type: "PoorAuctionType"
    ) -> "PoorAuction | None":
        if self.poor_auctions is None:
            return None
        return self.poor_auctions.lookup(auction_type)

    def poor_auction_statistics(self) -> "PoorAuctionStatistics":
        if self.poor_auctions is None:
            return PoorAuctionStatistics(0, 0, 0)
        return self.poor_auctions.statistics()

    def lookup_single_print(self, row: int) -> "SinglePrint | None":
        if self.single_prints is None:
            return None
        return self.single_prints.lookup(row)

    def single_print_statistics(self) -> "SinglePrintStatistics":
        if self.single_prints is None:
            return SinglePrintStatistics(0, 0, 0)
        return self.single_prints.statistics()

    def lookup_naked_poc(self, row: int) -> "NakedPointOfControl | None":
        if self.naked_pocs is None:
            return None
        return self.naked_pocs.lookup(row)

    def naked_poc_statistics(self) -> "NakedPointOfControlStatistics":
        if self.naked_pocs is None:
            return NakedPointOfControlStatistics(0, 0, 0, 0, 0)
        return self.naked_pocs.statistics()


@dataclass(frozen=True, slots=True)
class CellCoordinate:
    """Immutable logical position for one footprint cell."""

    row_index: int
    column_index: int
    cell_id: str
    grid: "GridCoordinateSystem" = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.row_index < 0:
            raise ValueError("row_index must be non-negative")
        if self.column_index < 0:
            raise ValueError("column_index must be non-negative")
        if not self.cell_id.strip():
            raise ValueError("cell_id is required")


@dataclass(frozen=True, slots=True)
class GridCoordinateSystem:
    """Logical row/column structure for an entire detected footprint grid."""

    grid_id: str
    row_count: int
    column_count: int
    cell_width: int
    cell_height: int
    bounds: BoundingBox
    cells: tuple[CellCoordinate, ...] = ()

    def __post_init__(self) -> None:
        if not self.grid_id.strip():
            raise ValueError("grid_id is required")
        if self.row_count <= 0 or self.column_count <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.cell_width <= 0 or self.cell_height <= 0:
            raise ValueError("cell dimensions must be positive")
        expected_count = self.row_count * self.column_count
        if len(self.cells) != expected_count:
            raise ValueError("coordinate count must match grid dimensions")
        positions = {(cell.row_index, cell.column_index) for cell in self.cells}
        if len(positions) != len(self.cells):
            raise ValueError("duplicate coordinates are not allowed")
        ids = {cell.cell_id for cell in self.cells}
        if len(ids) != len(self.cells):
            raise ValueError("duplicate cell ids are not allowed")
        expected_positions = {
            (row, column)
            for row in range(self.row_count)
            for column in range(self.column_count)
        }
        if positions != expected_positions:
            raise ValueError("coordinate indexing must be continuous")
        ordered = tuple(
            sorted(self.cells, key=lambda cell: (cell.row_index, cell.column_index))
        )
        if self.cells != ordered:
            object.__setattr__(self, "cells", ordered)

    @property
    def rows(self) -> int:
        return self.row_count

    @property
    def columns(self) -> int:
        return self.column_count

    def cell_at(self, row: int, column: int) -> CellCoordinate:
        for cell in self.cells:
            if cell.row_index == row and cell.column_index == column:
                return cell
        raise KeyError(f"cell coordinate not found: {row},{column}")

    def cell_by_id(self, cell_id: str) -> CellCoordinate:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(f"cell id not found: {cell_id}")

    def row_cells(self, row: int) -> tuple[CellCoordinate, ...]:
        return tuple(cell for cell in self.cells if cell.row_index == row)

    def column_cells(self, column: int) -> tuple[CellCoordinate, ...]:
        return tuple(cell for cell in self.cells if cell.column_index == column)

    def neighbors(self, cell: CellCoordinate) -> tuple[CellCoordinate, ...]:
        candidates = (
            (cell.row_index - 1, cell.column_index),
            (cell.row_index + 1, cell.column_index),
            (cell.row_index, cell.column_index - 1),
            (cell.row_index, cell.column_index + 1),
        )
        return tuple(
            self.cell_at(row, column)
            for row, column in candidates
            if 0 <= row < self.row_count and 0 <= column < self.column_count
        )


@dataclass(frozen=True, slots=True)
class CellReference:
    """Immutable link from a logical cell coordinate to detected geometry."""

    coordinate: CellCoordinate
    bounds: BoundingBox
    detected_object: DetectedObject
    frame_id: FrameId

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        if self.detected_object.frame_id != self.frame_id:
            raise ValueError("cell reference frame id must match detected object")
        if self.detected_object.bounds != self.bounds:
            raise ValueError("cell reference bounds must match detected object")


@dataclass(frozen=True, slots=True)
class CellRegion:
    """Immutable semantic sub-region inside a footprint cell."""

    bounds: BoundingBox
    semantic_role: CellSemanticRole
    confidence: float
    parent_cell_id: str
    frame_id: FrameId

    def __post_init__(self) -> None:
        if not self.parent_cell_id.strip():
            raise ValueError("parent cell id is required")
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        _validate_confidence(self.confidence, "cell region confidence")


@dataclass(frozen=True, slots=True)
class CellLayoutBand:
    """One configurable vertical logical band in a footprint-cell layout."""

    semantic_role: CellSemanticRole
    weight: int = 1
    required: bool = True

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("cell layout band weight must be positive")


@dataclass(frozen=True, slots=True)
class CellLayout:
    """Configurable expected logical subdivision of a footprint cell."""

    bands: tuple[CellLayoutBand, ...] = (
        CellLayoutBand(CellSemanticRole.ASK_REGION),
        CellLayoutBand(CellSemanticRole.CENTER_REGION),
        CellLayoutBand(CellSemanticRole.BID_REGION),
    )

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("cell layout requires at least one band")
        roles = [band.semantic_role for band in self.bands]
        if len(set(roles)) != len(roles):
            raise ValueError("cell layout semantic roles must be unique")
        required_roles = [band.semantic_role for band in self.bands if band.required]
        if len(set(required_roles)) != len(required_roles):
            raise ValueError("required semantic roles must be unique")

    @property
    def required_roles(self) -> frozenset[CellSemanticRole]:
        return frozenset(band.semantic_role for band in self.bands if band.required)


@dataclass(frozen=True, slots=True)
class CellClassification:
    """Immutable logical classification result for one footprint cell."""

    cell_reference: CellReference
    detected_cell_regions: tuple[CellRegion, ...]
    overall_confidence: float
    validation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_confidence(self.overall_confidence, "cell classification confidence")
        if not self.detected_cell_regions:
            raise ValueError("cell classification requires semantic regions")
        roles = [region.semantic_role for region in self.detected_cell_regions]
        if len(set(roles)) != len(roles):
            raise ValueError("duplicate semantic roles are not allowed")
        required = frozenset(self.validation_metadata.get("required_roles", ()))
        if required and not required.issubset(set(roles)):
            raise ValueError("missing required semantic regions")
        parent_id = self.cell_reference.coordinate.cell_id
        for region in self.detected_cell_regions:
            if region.parent_cell_id != parent_id:
                raise ValueError("cell region parent id must match cell reference")
            if region.frame_id != self.cell_reference.frame_id:
                raise ValueError("cell region frame id must match cell reference")
            if not _contains(self.cell_reference.bounds, region.bounds):
                raise ValueError("cell region must be inside parent cell")
        for index, first in enumerate(self.detected_cell_regions):
            for second in self.detected_cell_regions[index + 1 :]:
                if _overlap_area(first.bounds, second.bounds) > 0:
                    raise ValueError("semantic regions must not overlap")
        ordered = tuple(
            sorted(
                self.detected_cell_regions,
                key=lambda region: (
                    region.bounds.y,
                    region.bounds.x,
                    region.semantic_role.value,
                ),
            )
        )
        if self.detected_cell_regions != ordered:
            object.__setattr__(self, "detected_cell_regions", ordered)
        metadata = {
            **dict(self.validation_metadata),
            "cell_id": parent_id,
            "row": self.cell_reference.coordinate.row_index,
            "column": self.cell_reference.coordinate.column_index,
            "semantic_regions": tuple(role.value for role in roles),
            "classification_confidence": self.overall_confidence,
        }
        object.__setattr__(self, "validation_metadata", MappingProxyType(metadata))

    @property
    def cell_id(self) -> str:
        return self.cell_reference.coordinate.cell_id

    @property
    def row(self) -> int:
        return self.cell_reference.coordinate.row_index

    @property
    def column(self) -> int:
        return self.cell_reference.coordinate.column_index

    @property
    def semantic_regions(self) -> tuple[CellRegion, ...]:
        return self.detected_cell_regions

    @property
    def classification_confidence(self) -> float:
        return self.overall_confidence

    def region_by_role(self, role: CellSemanticRole) -> CellRegion | None:
        for region in self.detected_cell_regions:
            if region.semantic_role == role:
                return region
        return None

    def bid_region(self) -> CellRegion | None:
        return self.region_by_role(CellSemanticRole.BID_REGION)

    def ask_region(self) -> CellRegion | None:
        return self.region_by_role(CellSemanticRole.ASK_REGION)

    def center_region(self) -> CellRegion | None:
        return self.region_by_role(CellSemanticRole.CENTER_REGION)

    def all_regions(self) -> tuple[CellRegion, ...]:
        return self.detected_cell_regions


@dataclass(frozen=True, slots=True)
class CellLayoutAnalyzer:
    """Create deterministic logical footprint-cell regions without OCR."""

    layout: CellLayout = field(default_factory=CellLayout)

    def classify(self, cell_reference: CellReference) -> CellClassification:
        total_weight = sum(band.weight for band in self.layout.bands)
        top = cell_reference.bounds.y
        regions = []
        for index, band in enumerate(self.layout.bands):
            remaining_height = cell_reference.bounds.bottom - top
            if index == len(self.layout.bands) - 1:
                height = remaining_height
            else:
                height = max(
                    1,
                    round(cell_reference.bounds.height * band.weight / total_weight),
                )
                height = min(
                    height, remaining_height - (len(self.layout.bands) - index - 1)
                )
            bounds = BoundingBox(
                cell_reference.bounds.x,
                top,
                cell_reference.bounds.width,
                height,
            )
            regions.append(
                CellRegion(
                    bounds=bounds,
                    semantic_role=band.semantic_role,
                    confidence=1.0,
                    parent_cell_id=cell_reference.coordinate.cell_id,
                    frame_id=cell_reference.frame_id,
                )
            )
            top = bounds.bottom
        return CellClassification(
            cell_reference=cell_reference,
            detected_cell_regions=tuple(regions),
            overall_confidence=1.0,
            validation_metadata={"required_roles": self.layout.required_roles},
        )

    def analyze(self, cell_reference: CellReference) -> CellClassification:
        return self.classify(cell_reference)


class NumericType(Enum):
    """Raw OCR numeric shape after deterministic normalization."""

    UNKNOWN = "UNKNOWN"
    INTEGER = "INTEGER"
    DECIMAL = "DECIMAL"
    SIGNED_INTEGER = "SIGNED_INTEGER"
    SIGNED_DECIMAL = "SIGNED_DECIMAL"
    EMPTY = "EMPTY"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class NumericValue:
    """Validated numeric value parsed from normalized OCR text."""

    value: int | Decimal
    numeric_type: NumericType = NumericType.INTEGER

    def __post_init__(self) -> None:
        if self.numeric_type not in {
            NumericType.INTEGER,
            NumericType.DECIMAL,
            NumericType.SIGNED_INTEGER,
            NumericType.SIGNED_DECIMAL,
        }:
            raise ValueError("numeric value requires a numeric type")
        if isinstance(self.value, bool) or not isinstance(self.value, int | Decimal):
            raise ValueError("numeric value must be int or Decimal")
        if isinstance(self.value, Decimal) and not self.value.is_finite():
            raise ValueError("numeric value must be finite")


@dataclass(frozen=True, slots=True)
class ParsingError:
    """Positioned deterministic parsing error for raw OCR text."""

    reason: str
    original_text: str
    position: int | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("parsing error reason is required")
        if self.position is not None and self.position < 0:
            raise ValueError("parsing error position must be non-negative")


@dataclass(frozen=True, slots=True)
class NormalizationRule:
    """Single deterministic text-normalization replacement rule."""

    source: str
    replacement: str
    description: str = "replacement"

    def __post_init__(self) -> None:
        if self.source == "":
            raise ValueError("normalization rule source is required")
        if not self.description.strip():
            raise ValueError("normalization rule description is required")

    def apply(self, text: str) -> str:
        return text.replace(self.source, self.replacement)


@dataclass(frozen=True, slots=True)
class OCRNormalizationConfiguration:
    """Immutable configuration for OCR numeric text normalization."""

    replacement_table: Mapping[str, str] = field(
        default_factory=lambda: {
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "S": "5",
            "B": "8",
            "−": "-",
            "–": "-",
            "—": "-",
            "，": ",",
            "٫": ".",
        }
    )
    allowed_characters: str = "0123456789.-"
    allow_negative: bool = True
    allow_decimal: bool = True
    maximum_length: int = 32
    minimum_length: int = 1
    strict_mode: bool = True

    def __post_init__(self) -> None:
        if self.maximum_length <= 0:
            raise ValueError("maximum length must be positive")
        if self.minimum_length < 0 or self.minimum_length > self.maximum_length:
            raise ValueError("minimum length must be between zero and maximum length")
        if not self.allowed_characters:
            raise ValueError("allowed characters are required")
        object.__setattr__(
            self, "replacement_table", MappingProxyType(dict(self.replacement_table))
        )

    @property
    def rules(self) -> tuple[NormalizationRule, ...]:
        return tuple(
            NormalizationRule(source, replacement)
            for source, replacement in self.replacement_table.items()
        )


@dataclass(frozen=True, slots=True, init=False)
class ParsedValue:
    """Validated numeric interpretation of one raw OCR result."""

    raw_text: str
    normalized_text: str
    numeric_type: NumericType
    parsed_number: NumericValue | None
    confidence: float
    validation_messages: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        raw_text: str | NumericValue,
        normalized_text: str | CellSemanticRole,
        numeric_type: NumericType | float,
        parsed_number: NumericValue | CellRegion | None,
        confidence: float | str,
        validation_messages: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if (isinstance(raw_text, NumericValue) or callable(raw_text)) and isinstance(
            normalized_text, CellSemanticRole
        ):
            value = (
                raw_text
                if isinstance(raw_text, NumericValue)
                else NumericValue(raw_text())
            )
            role = normalized_text
            if isinstance(numeric_type, NumericType):
                raise ValueError("parsed value confidence is required")
            parsed_confidence = float(numeric_type)
            source_region = parsed_number
            cell_id = str(confidence)
            if not isinstance(source_region, CellRegion):
                raise ValueError("parsed value source region is required")
            if not cell_id.strip():
                raise ValueError("cell id is required")
            if source_region.parent_cell_id != cell_id:
                raise ValueError(
                    "parsed value source region must reference parent cell"
                )
            if source_region.semantic_role != role:
                raise ValueError("parsed value semantic role must match source region")
            object.__setattr__(self, "raw_text", str(value.value))
            object.__setattr__(self, "normalized_text", str(value.value))
            object.__setattr__(self, "numeric_type", value.numeric_type)
            object.__setattr__(self, "parsed_number", value)
            object.__setattr__(self, "confidence", parsed_confidence)
            object.__setattr__(self, "validation_messages", ())
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(
                    {
                        "semantic_role": role.value,
                        "source_region": source_region,
                        "cell_id": cell_id,
                    }
                ),
            )
        else:
            object.__setattr__(self, "raw_text", str(raw_text))
            object.__setattr__(self, "normalized_text", str(normalized_text))
            if not isinstance(numeric_type, NumericType):
                raise ValueError("numeric type is required")
            object.__setattr__(self, "numeric_type", numeric_type)
            if parsed_number is not None and not isinstance(
                parsed_number, NumericValue
            ):
                raise ValueError("parsed number must be a NumericValue")
            object.__setattr__(self, "parsed_number", parsed_number)
            object.__setattr__(self, "confidence", float(confidence))
            object.__setattr__(self, "validation_messages", tuple(validation_messages))
            object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "parsed value confidence")
        object.__setattr__(self, "validation_messages", tuple(self.validation_messages))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def is_numeric(self) -> bool:
        return self.parsed_number is not None

    def is_valid(self) -> bool:
        return self.numeric_type not in {NumericType.INVALID, NumericType.EMPTY}

    def is_empty(self) -> bool:
        return self.numeric_type == NumericType.EMPTY

    def numeric_value(self) -> int | Decimal | None:
        return self.parsed_number.value if self.parsed_number is not None else None

    @property
    def semantic_role(self) -> CellSemanticRole:
        role = self.metadata.get("semantic_role", CellSemanticRole.UNKNOWN.value)
        return (
            role if isinstance(role, CellSemanticRole) else CellSemanticRole(str(role))
        )

    @property
    def source_region(self) -> CellRegion:
        region = self.metadata.get("source_region")
        if not isinstance(region, CellRegion):
            raise ValueError("parsed value source region is required")
        return region

    @property
    def cell_id(self) -> str:
        return str(self.metadata.get("cell_id", ""))


@dataclass(frozen=True, slots=True)
class ParsingResult:
    """Success/failure wrapper for post-processed OCR numeric parsing."""

    parsed_value: ParsedValue
    success: bool
    failure_reason: str = ""
    warnings: tuple[str, ...] = ()
    parsing_errors: tuple[ParsingError, ...] = ()

    def __post_init__(self) -> None:
        if self.success and self.failure_reason:
            raise ValueError("successful parsing result cannot have a failure reason")
        if not self.success and not self.failure_reason.strip():
            raise ValueError("failed parsing result requires a failure reason")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "parsing_errors", tuple(self.parsing_errors))

    def is_valid(self) -> bool:
        return self.success and self.parsed_value.is_valid()

    def is_numeric(self) -> bool:
        return self.success and self.parsed_value.is_numeric()

    def is_empty(self) -> bool:
        return self.parsed_value.is_empty()

    def numeric_value(self) -> int | Decimal | None:
        return self.parsed_value.numeric_value()

    def errors(self) -> tuple[ParsingError, ...]:
        return self.parsing_errors


@runtime_checkable
class NormalizationPipeline(Protocol):
    """Normalize raw OCR text deterministically before validation."""

    def normalize(self, raw_text: str) -> tuple[str, tuple[str, ...]]:
        """Return normalized text and warnings."""


@runtime_checkable
class NumericParser(Protocol):
    """Parse already-normalized text into a numeric parsing result."""

    def parse(self, text: str, confidence: float = 1.0) -> ParsingResult:
        """Return parsed numeric result."""


@runtime_checkable
class OCRPostProcessor(Protocol):
    """Post-process raw OCR output into validated numeric values."""

    def process(self, result: "OCRResult") -> ParsingResult:
        """Return normalized and parsed OCR result."""


@dataclass(frozen=True, slots=True)
class DeterministicNumericParser:
    """Deterministic parser for integer and decimal OCR numeric text only."""

    configuration: OCRNormalizationConfiguration = field(
        default_factory=OCRNormalizationConfiguration
    )

    def parse(self, text: str, confidence: float = 1.0) -> ParsingResult:
        _validate_confidence(confidence, "parsing confidence")
        error = _numeric_error(text, self.configuration)
        if error is not None:
            ntype = NumericType.EMPTY if text == "" else NumericType.INVALID
            parsed = ParsedValue(text, text, ntype, None, 0.0, (error.reason,), {})
            return ParsingResult(parsed, False, error.reason, (), (error,))
        ntype = _numeric_type(text)
        try:
            value: int | Decimal = int(text) if "." not in text else Decimal(text)
        except (InvalidOperation, ValueError) as exc:
            error = ParsingError("invalid decimal", text, None)
            parsed = ParsedValue(
                text, text, NumericType.INVALID, None, 0.0, (str(exc),), {}
            )
            return ParsingResult(parsed, False, error.reason, (), (error,))
        if isinstance(value, Decimal) and not value.is_finite():
            error = ParsingError("non-finite decimal", text, None)
            parsed = ParsedValue(
                text, text, NumericType.INVALID, None, 0.0, (error.reason,), {}
            )
            return ParsingResult(parsed, False, error.reason, (), (error,))
        parsed = ParsedValue(
            text, text, ntype, NumericValue(value, ntype), confidence, (), {}
        )
        return ParsingResult(parsed, True)


@dataclass(frozen=True, slots=True)
class DeterministicOCRPostProcessor:
    """Normalize, validate, and parse raw OCR text without market semantics."""

    configuration: OCRNormalizationConfiguration = field(
        default_factory=OCRNormalizationConfiguration
    )
    parser: NumericParser | None = None

    def __post_init__(self) -> None:
        if self.parser is None:
            object.__setattr__(
                self, "parser", DeterministicNumericParser(self.configuration)
            )

    def normalize(self, raw_text: str) -> tuple[str, tuple[str, ...]]:
        warnings: list[str] = []
        normalized = raw_text
        for rule in self.configuration.rules:
            updated = rule.apply(normalized)
            if updated != normalized:
                warnings.append(rule.description)
            normalized = updated
        compact = "".join(normalized.split())
        if compact != normalized:
            warnings.append("whitespace removed")
        normalized = compact.replace(",", "")
        normalized, decimal_warning = _normalize_decimal_points(normalized)
        if decimal_warning:
            warnings.append(decimal_warning)
        normalized, sign_warning = _cleanup_duplicate_leading_signs(normalized)
        if sign_warning:
            warnings.append(sign_warning)
        trimmed = _trim_garbage(normalized, self.configuration)
        if trimmed != normalized:
            warnings.append("garbage trimmed")
        return trimmed, tuple(warnings)

    def process(self, result: "OCRResult") -> ParsingResult:
        normalized, warnings = self.normalize(result.text())
        parsed = self.parser.parse(normalized, result.confidence)  # type: ignore[union-attr]
        value = ParsedValue(
            raw_text=result.text(),
            normalized_text=normalized,
            numeric_type=parsed.parsed_value.numeric_type,
            parsed_number=parsed.parsed_value.parsed_number,
            confidence=parsed.parsed_value.confidence,
            validation_messages=parsed.parsed_value.validation_messages,
            metadata={
                "frame_id": result.frame_id,
                "cell_id": result.cell_id,
                "semantic_role": result.semantic_role.value,
            },
        )
        return ParsingResult(
            value,
            parsed.success,
            parsed.failure_reason,
            warnings + parsed.warnings,
            parsed.parsing_errors,
        )


def normalize(
    raw_text: str, configuration: OCRNormalizationConfiguration | None = None
) -> str:
    return DeterministicOCRPostProcessor(
        configuration or OCRNormalizationConfiguration()
    ).normalize(raw_text)[0]


def parse(
    raw_text: str, configuration: OCRNormalizationConfiguration | None = None
) -> ParsingResult:
    config = configuration or OCRNormalizationConfiguration()
    text = normalize(raw_text, config)
    return DeterministicNumericParser(config).parse(text)


def is_numeric(value: ParsedValue | ParsingResult) -> bool:
    return value.is_numeric()


def is_valid(value: ParsedValue | ParsingResult) -> bool:
    return value.is_valid()


def is_empty(value: ParsedValue | ParsingResult) -> bool:
    return value.is_empty()


def numeric_value(value: ParsedValue | ParsingResult) -> int | Decimal | None:
    return value.numeric_value()


def errors(result: ParsingResult) -> tuple[ParsingError, ...]:
    return result.errors()


def warnings(result: ParsingResult) -> tuple[str, ...]:
    return result.warnings


@dataclass(frozen=True, slots=True)
class OCRConfiguration:
    """Immutable OCR execution configuration without provider assumptions."""

    language: str = "eng"
    minimum_confidence: float = 0.0
    character_whitelist: str = ""
    character_blacklist: str = ""
    engine_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("ocr language is required")
        _validate_confidence(self.minimum_confidence, "ocr minimum confidence")
        object.__setattr__(
            self, "engine_options", MappingProxyType(dict(self.engine_options))
        )


@dataclass(frozen=True, slots=True)
class OCRMetadata:
    """Immutable provider-neutral metadata for a raw OCR result."""

    engine_name: str
    provider_name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.engine_name.strip():
            raise ValueError("ocr engine name is required")
        if not self.provider_name.strip():
            raise ValueError("ocr provider name is required")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class OCRWord:
    """One raw OCR word with provider-reported geometry and confidence."""

    text_value: str
    confidence: float
    bounding_box: BoundingBox

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "ocr word confidence")

    @property
    def text(self) -> str:
        return self.text_value


@dataclass(frozen=True, slots=True)
class OCRLine:
    """One raw OCR line containing ordered OCR words."""

    words_value: tuple[OCRWord, ...]
    bounding_box: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "ocr line confidence")
        ordered = tuple(
            sorted(
                self.words_value,
                key=lambda word: (word.bounding_box.y, word.bounding_box.x),
            )
        )
        if self.words_value != ordered:
            object.__setattr__(self, "words_value", ordered)

    def words(self) -> tuple[OCRWord, ...]:
        return self.words_value

    def text(self) -> str:
        return " ".join(word.text for word in self.words_value)


@dataclass(frozen=True, slots=True)
class OCRPage:
    """Raw OCR page containing lines and metadata."""

    lines_value: tuple[OCRLine, ...] = ()
    metadata: OCRMetadata = field(
        default_factory=lambda: OCRMetadata("unknown", "unknown")
    )

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                self.lines_value,
                key=lambda line: (line.bounding_box.y, line.bounding_box.x),
            )
        )
        if self.lines_value != ordered:
            object.__setattr__(self, "lines_value", ordered)

    def lines(self) -> tuple[OCRLine, ...]:
        return self.lines_value


@dataclass(frozen=True, slots=True)
class OCRRegion:
    """Semantic OCR target region linked to one classified cell region."""

    frame_id: FrameId
    cell_id: str
    bounding_box: BoundingBox
    semantic_role: CellSemanticRole

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        if not self.cell_id.strip():
            raise ValueError("cell id is required")
        if self.semantic_role == CellSemanticRole.UNKNOWN:
            raise ValueError("semantic role is required")


@dataclass(frozen=True, slots=True)
class OCRRequest:
    """Immutable input for an OCR engine over one predefined semantic region."""

    frame_id: FrameId
    cell_id: str
    bounding_box: BoundingBox
    semantic_role: CellSemanticRole
    image_region: ImageFrame
    configuration: OCRConfiguration = field(default_factory=OCRConfiguration)

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        if not self.cell_id.strip():
            raise ValueError("cell id is required")
        if self.semantic_role == CellSemanticRole.UNKNOWN:
            raise ValueError("semantic role is required")
        if not self.bounding_box.fits_within(
            self.image_region.width, self.image_region.height
        ):
            raise ValueError("ocr bounding box must fit within image region")


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Provider-neutral raw OCR output with no parsing or interpretation."""

    frame_id: FrameId
    cell_id: str
    semantic_role: CellSemanticRole
    extracted_text: str
    confidence: float
    bounding_boxes: tuple[BoundingBox, ...]
    detected_words: tuple[OCRWord, ...] = ()
    detected_lines: tuple[OCRLine, ...] = ()
    metadata: OCRMetadata = field(
        default_factory=lambda: OCRMetadata("unknown", "unknown")
    )

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame id is required")
        if not self.cell_id.strip():
            raise ValueError("cell id is required")
        if self.semantic_role == CellSemanticRole.UNKNOWN:
            raise ValueError("semantic role is required")
        _validate_confidence(self.confidence, "ocr result confidence")
        if not self.bounding_boxes and (self.detected_words or self.detected_lines):
            raise ValueError("ocr bounding boxes are required for non-empty output")
        object.__setattr__(
            self,
            "detected_words",
            tuple(
                sorted(
                    self.detected_words,
                    key=lambda word: (word.bounding_box.y, word.bounding_box.x),
                )
            ),
        )
        object.__setattr__(
            self,
            "detected_lines",
            tuple(
                sorted(
                    self.detected_lines,
                    key=lambda line: (line.bounding_box.y, line.bounding_box.x),
                )
            ),
        )

    def words(self) -> tuple[OCRWord, ...]:
        return self.detected_words

    def lines(self) -> tuple[OCRLine, ...]:
        return self.detected_lines

    def text(self) -> str:
        return self.extracted_text

    def average_confidence(self) -> float:
        confidences = [word.confidence for word in self.detected_words]
        return sum(confidences) / len(confidences) if confidences else self.confidence


class FootprintSemanticType(Enum):
    """Market meaning assigned to a validated footprint numeric value."""

    UNKNOWN = "UNKNOWN"
    BID_VOLUME = "BID_VOLUME"
    ASK_VOLUME = "ASK_VOLUME"
    DELTA = "DELTA"
    TOTAL_VOLUME = "TOTAL_VOLUME"
    EMPTY = "EMPTY"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class InterpretationWarning:
    """Non-fatal footprint interpretation validation warning."""

    code: str
    message: str
    cell_id: str = ""

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("warning code is required")
        if not self.message.strip():
            raise ValueError("warning message is required")


@dataclass(frozen=True, slots=True)
class FootprintValue:
    """A numeric value with market semantics inside a footprint cell."""

    numeric_value: NumericValue
    semantic_type: FootprintSemanticType
    confidence: float
    source_region: CellRegion
    cell_id: str

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("cell id is required")
        _validate_confidence(self.confidence, "footprint value confidence")
        if self.source_region.parent_cell_id != self.cell_id:
            raise ValueError("footprint value source region must reference parent cell")
        if self.semantic_type in (
            FootprintSemanticType.UNKNOWN,
            FootprintSemanticType.INVALID,
            FootprintSemanticType.EMPTY,
        ):
            raise ValueError(
                "footprint value requires a concrete numeric semantic type"
            )


@dataclass(frozen=True, slots=True)
class FootprintCellData:
    """Interpreted market data for one footprint cell."""

    cell_reference: CellReference
    bid_value: FootprintValue | None = None
    ask_value: FootprintValue | None = None
    delta_value: FootprintValue | None = None
    total_volume_value: FootprintValue | None = None
    missing_values: tuple[FootprintSemanticType, ...] = ()
    interpretation_warnings: tuple[InterpretationWarning, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = tuple(
            v
            for v in (
                self.bid_value,
                self.ask_value,
                self.delta_value,
                self.total_volume_value,
            )
            if v is not None
        )
        cell_id = self.cell_reference.coordinate.cell_id
        if self.cell_reference.coordinate is None:
            raise ValueError("missing coordinate")
        for value in values:
            if value.cell_id != cell_id:
                raise ValueError("footprint value cell id must match parent cell")
        semantic_types = [value.semantic_type for value in values]
        if len(set(semantic_types)) != len(semantic_types):
            raise ValueError("duplicate semantic assignments are not allowed")
        expected = {
            "bid_value": FootprintSemanticType.BID_VOLUME,
            "ask_value": FootprintSemanticType.ASK_VOLUME,
            "delta_value": FootprintSemanticType.DELTA,
            "total_volume_value": FootprintSemanticType.TOTAL_VOLUME,
        }
        for attr, semantic_type in expected.items():
            value = getattr(self, attr)
            if value is not None and value.semantic_type != semantic_type:
                raise ValueError("invalid semantic mapping")
        object.__setattr__(self, "missing_values", tuple(self.missing_values))
        object.__setattr__(
            self, "interpretation_warnings", tuple(self.interpretation_warnings)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def bid(self) -> FootprintValue | None:
        return self.bid_value

    def ask(self) -> FootprintValue | None:
        return self.ask_value

    def delta(self) -> FootprintValue | None:
        return self.delta_value

    def total_volume(self) -> FootprintValue | None:
        return self.total_volume_value

    def is_complete(self) -> bool:
        return (
            not self.missing_values
            and self.bid_value is not None
            and self.ask_value is not None
            and self.delta_value is not None
        )

    def is_empty(self) -> bool:
        return not any(
            (self.bid_value, self.ask_value, self.delta_value, self.total_volume_value)
        )

    def missing_fields(self) -> tuple[FootprintSemanticType, ...]:
        return self.missing_values

    def warnings(self) -> tuple[InterpretationWarning, ...]:
        return self.interpretation_warnings


@dataclass(frozen=True, slots=True)
class FootprintInterpretation:
    """Interpreted footprint data for an entire footprint grid."""

    grid_id: str
    ordered_cells: tuple[FootprintCellData, ...]
    interpretation_confidence: float
    interpretation_warnings: tuple[InterpretationWarning, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.grid_id.strip():
            raise ValueError("grid id is required")
        _validate_confidence(
            self.interpretation_confidence, "interpretation confidence"
        )
        ids = [cell.cell_reference.coordinate.cell_id for cell in self.ordered_cells]
        if len(set(ids)) != len(ids):
            raise ValueError("interpreted cells must be unique")
        ordered = tuple(
            sorted(
                self.ordered_cells,
                key=lambda c: (
                    c.cell_reference.coordinate.row_index,
                    c.cell_reference.coordinate.column_index,
                    c.cell_reference.coordinate.cell_id,
                ),
            )
        )
        object.__setattr__(self, "ordered_cells", ordered)
        object.__setattr__(
            self, "interpretation_warnings", tuple(self.interpretation_warnings)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def lookup_cell(self, cell_id: str) -> FootprintCellData | None:
        return next(
            (
                cell
                for cell in self.ordered_cells
                if cell.cell_reference.coordinate.cell_id == cell_id
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class MatrixDimensions:
    """Immutable row/column dimensions for a footprint matrix."""

    rows: int
    columns: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("matrix dimensions must be positive")


@dataclass(frozen=True, slots=True)
class MatrixPosition:
    """Immutable logical matrix position for one footprint cell."""

    row_index: int
    column_index: int
    cell_id: str
    coordinate: CellCoordinate

    def __post_init__(self) -> None:
        if self.row_index < 0 or self.column_index < 0:
            raise ValueError("matrix position indexes must be non-negative")
        if not self.cell_id.strip():
            raise ValueError("matrix position cell id is required")
        if self.coordinate.row_index != self.row_index:
            raise ValueError("matrix position row must match coordinate")
        if self.coordinate.column_index != self.column_index:
            raise ValueError("matrix position column must match coordinate")
        if self.coordinate.cell_id != self.cell_id:
            raise ValueError("matrix position cell id must match coordinate")


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """Canonical immutable matrix entry referencing all cell-level source data."""

    position: MatrixPosition
    interpretation: FootprintCellData
    parsed_values: tuple[ParsingResult, ...]
    classification: CellClassification
    original_cell: DetectedObject

    def __post_init__(self) -> None:
        cell_id = self.position.cell_id
        if self.interpretation.cell_reference.coordinate.cell_id != cell_id:
            raise ValueError("matrix cell interpretation must match position")
        if self.classification.cell_id != cell_id:
            raise ValueError("matrix cell classification must match position")
        if self.original_cell.object_type != ObjectType.FOOTPRINT_CELL:
            raise ValueError("matrix cell original object must be a footprint cell")
        if str(self.original_cell.metadata.get("cell_id", "")) != cell_id:
            raise ValueError("matrix cell original object must match position")
        if any(result.parsed_value.cell_id != cell_id for result in self.parsed_values):
            raise ValueError("matrix cell parsed values must match position")
        object.__setattr__(self, "parsed_values", tuple(self.parsed_values))

    @property
    def row_index(self) -> int:
        return self.position.row_index

    @property
    def column_index(self) -> int:
        return self.position.column_index

    @property
    def cell_id(self) -> str:
        return self.position.cell_id

    @property
    def coordinate(self) -> CellCoordinate:
        return self.position.coordinate


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """Immutable row of matrix cells."""

    index: int
    cells: tuple[MatrixCell, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("matrix row index must be non-negative")
        if not self.cells:
            raise ValueError("matrix row must contain cells")
        if any(cell.row_index != self.index for cell in self.cells):
            raise ValueError("matrix row cells must share the row index")
        ordered = tuple(sorted(self.cells, key=lambda cell: cell.column_index))
        if self.cells != ordered:
            raise ValueError("matrix row cells must be column ordered")
        columns = tuple(cell.column_index for cell in self.cells)
        if len(set(columns)) != len(columns):
            raise ValueError("duplicate matrix row columns are not allowed")
        if columns != tuple(range(len(self.cells))):
            raise ValueError("matrix row columns must be continuous")
        object.__setattr__(self, "cells", tuple(self.cells))


@dataclass(frozen=True, slots=True)
class MatrixStatistics:
    """Structural statistics for a footprint matrix."""

    rows: int
    columns: int
    total_cells: int
    interpreted_cells: int
    empty_cells: int
    missing_cells: int
    bid_cells: int
    ask_cells: int
    delta_cells: int
    unknown_cells: int

    def __post_init__(self) -> None:
        values = (
            self.rows,
            self.columns,
            self.total_cells,
            self.interpreted_cells,
            self.empty_cells,
            self.missing_cells,
            self.bid_cells,
            self.ask_cells,
            self.delta_cells,
            self.unknown_cells,
        )
        if any(value < 0 for value in values):
            raise ValueError("matrix statistics cannot be negative")


@dataclass(frozen=True, slots=True)
class ImbalanceConfiguration:
    """Immutable settings for single-cell footprint imbalance detection."""

    minimum_ratio: Decimal = Decimal("3")
    minimum_volume: Decimal = Decimal("1")
    compare_diagonal: bool = True
    allow_zero_opposite: bool = False
    strict_mode: bool = True

    def __post_init__(self) -> None:
        ratio = Decimal(str(self.minimum_ratio))
        volume = Decimal(str(self.minimum_volume))
        if not ratio.is_finite() or ratio <= 0:
            raise ValueError("minimum ratio must be positive and finite")
        if not volume.is_finite() or volume < 0:
            raise ValueError("minimum volume must be non-negative and finite")
        object.__setattr__(self, "minimum_ratio", ratio)
        object.__setattr__(self, "minimum_volume", volume)


@dataclass(frozen=True, slots=True)
class FootprintImbalance:
    """One deterministic bid/ask imbalance detected from matrix cell values."""

    cell_id: str
    position: "MatrixPosition"
    imbalance_type: ImbalanceType
    side: ImbalanceSide
    ratio: Decimal
    dominant_value: Decimal
    opposite_value: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("imbalance cell id is required")
        if self.position.cell_id != self.cell_id:
            raise ValueError("imbalance position must reference cell id")
        if self.imbalance_type == ImbalanceType.NONE:
            raise ValueError("detected imbalance cannot have NONE type")
        if (
            self.imbalance_type == ImbalanceType.ASK_IMBALANCE
            and self.side != ImbalanceSide.ASK
        ):
            raise ValueError("ask imbalance must use ask side")
        if (
            self.imbalance_type == ImbalanceType.BID_IMBALANCE
            and self.side != ImbalanceSide.BID
        ):
            raise ValueError("bid imbalance must use bid side")
        ratio = Decimal(str(self.ratio))
        dominant = Decimal(str(self.dominant_value))
        opposite = Decimal(str(self.opposite_value))
        if not ratio.is_finite() or ratio <= 0:
            raise ValueError("imbalance ratio must be positive and finite")
        if not dominant.is_finite() or dominant < 0:
            raise ValueError("dominant value must be non-negative and finite")
        if not opposite.is_finite() or opposite < 0:
            raise ValueError("opposite value must be non-negative and finite")
        _validate_confidence(self.confidence, "imbalance confidence")
        object.__setattr__(self, "ratio", ratio)
        object.__setattr__(self, "dominant_value", dominant)
        object.__setattr__(self, "opposite_value", opposite)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ImbalanceStatistics:
    """Aggregate counts for detected single-cell footprint imbalances."""

    total_cells: int
    ask_imbalances: int
    bid_imbalances: int
    total_imbalances: int
    cells_without_imbalance: int

    def __post_init__(self) -> None:
        values = (
            self.total_cells,
            self.ask_imbalances,
            self.bid_imbalances,
            self.total_imbalances,
            self.cells_without_imbalance,
        )
        if any(value < 0 for value in values):
            raise ValueError("imbalance statistics cannot be negative")
        if self.total_imbalances != self.ask_imbalances + self.bid_imbalances:
            raise ValueError("total imbalances must equal ask plus bid imbalances")
        if self.cells_without_imbalance + self.total_imbalances != self.total_cells:
            raise ValueError("imbalance statistics must account for all cells")


@dataclass(frozen=True, slots=True)
class FootprintImbalanceResult:
    """Immutable result set for single-cell footprint imbalance detection."""

    matrix: "FootprintMatrix"
    detections: tuple[FootprintImbalance, ...]
    configuration: ImbalanceConfiguration = field(
        default_factory=ImbalanceConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                self.detections,
                key=lambda d: (
                    d.position.row_index,
                    d.position.column_index,
                    d.imbalance_type.value,
                ),
            )
        )
        if self.detections != ordered:
            raise ValueError("imbalance detections must be ordered")
        ids = [d.cell_id for d in self.detections]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate imbalance detections are not allowed")
        matrix_ids = {cell.cell_id for cell in self.matrix.cells}
        if set(ids) - matrix_ids:
            raise ValueError("imbalance detections must reference matrix cells")
        object.__setattr__(self, "detections", ordered)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def imbalances(self) -> tuple[FootprintImbalance, ...]:
        return self.detections

    def ask_imbalances(self) -> tuple[FootprintImbalance, ...]:
        return tuple(
            d
            for d in self.detections
            if d.imbalance_type == ImbalanceType.ASK_IMBALANCE
        )

    def bid_imbalances(self) -> tuple[FootprintImbalance, ...]:
        return tuple(
            d
            for d in self.detections
            if d.imbalance_type == ImbalanceType.BID_IMBALANCE
        )

    def lookup(self, cell_id: str) -> FootprintImbalance | None:
        return next((d for d in self.detections if d.cell_id == cell_id), None)

    def has_imbalance(self, cell_id: str) -> bool:
        return self.lookup(cell_id) is not None

    def statistics(self) -> ImbalanceStatistics:
        ask_count = len(self.ask_imbalances())
        bid_count = len(self.bid_imbalances())
        total = len(self.detections)
        return ImbalanceStatistics(
            self.matrix.statistics().total_cells,
            ask_count,
            bid_count,
            total,
            self.matrix.statistics().total_cells - total,
        )


@dataclass(frozen=True, slots=True)
class FootprintImbalanceDetector:
    """Detect deterministic individual bid/ask footprint imbalances from a matrix."""

    configuration: ImbalanceConfiguration = field(
        default_factory=ImbalanceConfiguration
    )

    def detect(self, matrix: "FootprintMatrix") -> FootprintImbalanceResult:
        detections = []
        for cell in matrix.cells:
            ask = self._decimal(cell.interpretation.ask())
            bid = self._decimal(cell.interpretation.bid())
            if ask is not None:
                opposite_cell = (
                    matrix.below(cell) if self.configuration.compare_diagonal else cell
                )
                opposite = (
                    None
                    if opposite_cell is None
                    else self._decimal(opposite_cell.interpretation.bid())
                )
                detection = self._build(
                    cell, ImbalanceType.ASK_IMBALANCE, ImbalanceSide.ASK, ask, opposite
                )
                if detection is not None:
                    detections.append(detection)
                    continue
            if bid is not None:
                opposite_cell = (
                    matrix.above(cell) if self.configuration.compare_diagonal else cell
                )
                opposite = (
                    None
                    if opposite_cell is None
                    else self._decimal(opposite_cell.interpretation.ask())
                )
                detection = self._build(
                    cell, ImbalanceType.BID_IMBALANCE, ImbalanceSide.BID, bid, opposite
                )
                if detection is not None:
                    detections.append(detection)
        return FootprintImbalanceResult(
            matrix,
            tuple(detections),
            self.configuration,
            {"detector": "FootprintImbalanceDetector"},
        )

    @staticmethod
    def _decimal(value: FootprintValue | None) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value.numeric_value.value))

    def _build(
        self,
        cell: "MatrixCell",
        imbalance_type: ImbalanceType,
        side: ImbalanceSide,
        dominant: Decimal,
        opposite: Decimal | None,
    ) -> FootprintImbalance | None:
        if dominant < self.configuration.minimum_volume:
            return None
        if opposite is None:
            return None
        if opposite == 0:
            if not self.configuration.allow_zero_opposite:
                return None
            ratio = dominant
        else:
            ratio = dominant / opposite
        if ratio < self.configuration.minimum_ratio:
            return None
        confidence = (
            min(1.0, float(ratio / self.configuration.minimum_ratio))
            if self.configuration.strict_mode
            else 1.0
        )
        return FootprintImbalance(
            cell.cell_id,
            cell.position,
            imbalance_type,
            side,
            ratio,
            dominant,
            opposite,
            confidence,
            {"compare_diagonal": self.configuration.compare_diagonal},
        )


@dataclass(frozen=True, slots=True)
class StackedImbalanceConfiguration:
    """Immutable settings for vertically stacked imbalance detection."""

    minimum_stack_size: int = 3
    allow_gaps: bool = False
    maximum_gap: int = 0
    minimum_average_ratio: Decimal = Decimal("3")
    minimum_total_volume: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if self.minimum_stack_size <= 1:
            raise ValueError("minimum stack size must be greater than one")
        if self.maximum_gap < 0:
            raise ValueError("maximum gap must be non-negative")
        if not self.allow_gaps and self.maximum_gap != 0:
            raise ValueError("maximum gap requires gaps to be allowed")
        ratio = Decimal(str(self.minimum_average_ratio))
        volume = Decimal(str(self.minimum_total_volume))
        if not ratio.is_finite() or ratio <= 0:
            raise ValueError("minimum average ratio must be positive and finite")
        if not volume.is_finite() or volume < 0:
            raise ValueError("minimum total volume must be non-negative and finite")
        object.__setattr__(self, "minimum_average_ratio", ratio)
        object.__setattr__(self, "minimum_total_volume", volume)


@dataclass(frozen=True, slots=True)
class StackedImbalance:
    """One deterministic vertical sequence of same-side footprint imbalances."""

    stack_id: str
    stack_type: StackedImbalanceType
    starting_cell: MatrixPosition
    ending_cell: MatrixPosition
    cells: tuple[FootprintImbalance, ...]
    row_span: tuple[int, int]
    average_ratio: Decimal
    total_dominant_volume: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stack_id.strip():
            raise ValueError("stack id is required")
        if self.stack_type == StackedImbalanceType.NONE:
            raise ValueError("detected stack cannot have NONE type")
        if len(self.cells) <= 1:
            raise ValueError("stack must contain at least two cells")
        ordered = tuple(
            sorted(
                self.cells,
                key=lambda d: (
                    d.position.column_index,
                    d.position.row_index,
                    d.cell_id,
                ),
            )
        )
        if self.cells != ordered:
            raise ValueError("stack cells must be ordered")
        columns = {cell.position.column_index for cell in self.cells}
        if len(columns) != 1:
            raise ValueError("stack cells must share one matrix column")
        expected_type = (
            ImbalanceType.ASK_IMBALANCE
            if self.stack_type == StackedImbalanceType.STACKED_ASK
            else ImbalanceType.BID_IMBALANCE
        )
        if any(cell.imbalance_type != expected_type for cell in self.cells):
            raise ValueError("stack cells must match stack type")
        if (
            self.starting_cell != self.cells[0].position
            or self.ending_cell != self.cells[-1].position
        ):
            raise ValueError("stack boundaries must match included cells")
        if self.row_span != (self.starting_cell.row_index, self.ending_cell.row_index):
            raise ValueError("row span must match stack boundaries")
        ratio = Decimal(str(self.average_ratio))
        volume = Decimal(str(self.total_dominant_volume))
        if not ratio.is_finite() or ratio <= 0:
            raise ValueError("average ratio must be positive and finite")
        if not volume.is_finite() or volume < 0:
            raise ValueError("total dominant volume must be non-negative and finite")
        _validate_confidence(self.confidence, "stack confidence")
        object.__setattr__(self, "average_ratio", ratio)
        object.__setattr__(self, "total_dominant_volume", volume)
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class StackedImbalanceStatistics:
    """Aggregate counts for deterministic stacked footprint imbalances."""

    total_stacks: int
    ask_stacks: int
    bid_stacks: int
    largest_stack: int
    average_stack_size: Decimal
    maximum_stack_size: int

    def __post_init__(self) -> None:
        values = (
            self.total_stacks,
            self.ask_stacks,
            self.bid_stacks,
            self.largest_stack,
            self.maximum_stack_size,
        )
        if any(value < 0 for value in values):
            raise ValueError("stack statistics cannot be negative")
        if self.total_stacks != self.ask_stacks + self.bid_stacks:
            raise ValueError("total stacks must equal ask plus bid stacks")
        average = Decimal(str(self.average_stack_size))
        if not average.is_finite() or average < 0:
            raise ValueError("average stack size must be non-negative and finite")
        if self.largest_stack != self.maximum_stack_size:
            raise ValueError("largest stack must equal maximum stack size")
        object.__setattr__(self, "average_stack_size", average)


@dataclass(frozen=True, slots=True)
class StackedImbalanceResult:
    """Immutable result set for stacked imbalance detection."""

    matrix: "FootprintMatrix"
    imbalances: FootprintImbalanceResult
    detections: tuple[StackedImbalance, ...]
    configuration: StackedImbalanceConfiguration = field(
        default_factory=StackedImbalanceConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.imbalances.matrix != self.matrix:
            raise ValueError("stacked result imbalances must reference matrix")
        ordered = tuple(
            sorted(
                self.detections,
                key=lambda d: (
                    d.starting_cell.column_index,
                    d.starting_cell.row_index,
                    d.stack_type.value,
                    d.stack_id,
                ),
            )
        )
        if self.detections != ordered:
            raise ValueError("stacked imbalance detections must be ordered")
        ids = [d.stack_id for d in self.detections]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate stacks are not allowed")
        matrix_ids = {cell.cell_id for cell in self.matrix.cells}
        for stack in self.detections:
            cell_ids = {cell.cell_id for cell in stack.cells}
            if cell_ids - matrix_ids:
                raise ValueError("stack cells must reference matrix cells")
        object.__setattr__(self, "detections", ordered)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def stacks(self) -> tuple[StackedImbalance, ...]:
        return self.detections

    def ask_stacks(self) -> tuple[StackedImbalance, ...]:
        return tuple(
            d
            for d in self.detections
            if d.stack_type == StackedImbalanceType.STACKED_ASK
        )

    def bid_stacks(self) -> tuple[StackedImbalance, ...]:
        return tuple(
            d
            for d in self.detections
            if d.stack_type == StackedImbalanceType.STACKED_BID
        )

    def lookup(self, stack_id: str) -> StackedImbalance | None:
        return next((d for d in self.detections if d.stack_id == stack_id), None)

    def lookup_by_cell(self, cell_id: str) -> tuple[StackedImbalance, ...]:
        return tuple(
            d
            for d in self.detections
            if any(cell.cell_id == cell_id for cell in d.cells)
        )

    def statistics(self) -> StackedImbalanceStatistics:
        sizes = tuple(len(stack.cells) for stack in self.detections)
        total = len(sizes)
        maximum = max(sizes, default=0)
        average = Decimal("0") if total == 0 else Decimal(sum(sizes)) / Decimal(total)
        return StackedImbalanceStatistics(
            total,
            len(self.ask_stacks()),
            len(self.bid_stacks()),
            maximum,
            average,
            maximum,
        )


@dataclass(frozen=True, slots=True)
class StackedImbalanceDetector:
    """Detect vertical stacks of same-side footprint imbalances from immutable data."""

    configuration: StackedImbalanceConfiguration = field(
        default_factory=StackedImbalanceConfiguration
    )

    def detect(
        self, matrix: "FootprintMatrix", imbalances: FootprintImbalanceResult
    ) -> StackedImbalanceResult:
        if imbalances.matrix != matrix:
            raise ValueError(
                "stacked imbalance detector inputs must reference the same matrix"
            )
        detections: list[StackedImbalance] = []
        lookup = {imbalance.cell_id: imbalance for imbalance in imbalances.imbalances()}
        for column in range(matrix.dimensions_value.columns):
            for imbalance_type, stack_type in (
                (ImbalanceType.ASK_IMBALANCE, StackedImbalanceType.STACKED_ASK),
                (ImbalanceType.BID_IMBALANCE, StackedImbalanceType.STACKED_BID),
            ):
                detections.extend(
                    self._column_stacks(
                        matrix, lookup, column, imbalance_type, stack_type
                    )
                )
        return StackedImbalanceResult(
            matrix,
            imbalances,
            tuple(detections),
            self.configuration,
            {"detector": "StackedImbalanceDetector"},
        )

    def _column_stacks(
        self,
        matrix: "FootprintMatrix",
        lookup: Mapping[str, FootprintImbalance],
        column: int,
        imbalance_type: ImbalanceType,
        stack_type: StackedImbalanceType,
    ) -> tuple[StackedImbalance, ...]:
        found: list[StackedImbalance] = []
        current: list[FootprintImbalance] = []
        gaps = 0
        for row in range(matrix.dimensions_value.rows):
            cell = matrix.cell(row, column)
            imbalance = lookup.get(cell.cell_id)
            if imbalance is not None and imbalance.imbalance_type == imbalance_type:
                current.append(imbalance)
                gaps = 0
                continue
            if (
                current
                and self.configuration.allow_gaps
                and gaps < self.configuration.maximum_gap
            ):
                gaps += 1
                continue
            self._append_if_valid(found, current, stack_type)
            current = []
            gaps = 0
        self._append_if_valid(found, current, stack_type)
        return tuple(found)

    def _append_if_valid(
        self,
        found: list[StackedImbalance],
        cells: Sequence[FootprintImbalance],
        stack_type: StackedImbalanceType,
    ) -> None:
        if len(cells) < self.configuration.minimum_stack_size:
            return
        average_ratio = sum((cell.ratio for cell in cells), Decimal("0")) / Decimal(
            len(cells)
        )
        total_volume = sum((cell.dominant_value for cell in cells), Decimal("0"))
        if (
            average_ratio < self.configuration.minimum_average_ratio
            or total_volume < self.configuration.minimum_total_volume
        ):
            return
        first = cells[0]
        last = cells[-1]
        stack_id = f"{stack_type.value}:{first.position.column_index}:{first.position.row_index}-{last.position.row_index}"
        confidence = min(
            1.0, float(average_ratio / self.configuration.minimum_average_ratio)
        )
        found.append(
            StackedImbalance(
                stack_id,
                stack_type,
                first.position,
                last.position,
                tuple(cells),
                (first.position.row_index, last.position.row_index),
                average_ratio,
                total_volume,
                confidence,
                {"minimum_stack_size": self.configuration.minimum_stack_size},
            )
        )


@dataclass(frozen=True, slots=True)
class AbsorptionConfiguration:
    """Immutable settings for deterministic absorption detection."""

    minimum_absorbed_volume: Decimal = Decimal("50")
    minimum_pressure_ratio: Decimal = Decimal("3")

    def __post_init__(self) -> None:
        volume = Decimal(str(self.minimum_absorbed_volume))
        ratio = Decimal(str(self.minimum_pressure_ratio))
        if not volume.is_finite() or volume < 0:
            raise ValueError("minimum absorbed volume must be non-negative and finite")
        if not ratio.is_finite() or ratio <= 0:
            raise ValueError("minimum pressure ratio must be positive and finite")
        object.__setattr__(self, "minimum_absorbed_volume", volume)
        object.__setattr__(self, "minimum_pressure_ratio", ratio)


@dataclass(frozen=True, slots=True)
class FootprintAbsorption:
    """One deterministic absorption observation derived from footprint values."""

    cell_id: str
    position: MatrixPosition
    absorption_type: AbsorptionType
    passive_side: AbsorptionSide
    absorbed_volume: Decimal
    pressure_ratio: Decimal
    source_imbalance: FootprintImbalance
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("absorption cell id is required")
        if self.position.cell_id != self.cell_id:
            raise ValueError("absorption position must reference cell id")
        if self.source_imbalance.cell_id != self.cell_id:
            raise ValueError("absorption source imbalance must reference cell id")
        if self.absorption_type == AbsorptionType.NONE:
            raise ValueError("detected absorption cannot have NONE type")
        if (
            self.absorption_type == AbsorptionType.BUY_ABSORPTION
            and self.passive_side != AbsorptionSide.BID
        ):
            raise ValueError("buy absorption must use bid passive side")
        if (
            self.absorption_type == AbsorptionType.SELL_ABSORPTION
            and self.passive_side != AbsorptionSide.ASK
        ):
            raise ValueError("sell absorption must use ask passive side")
        volume = Decimal(str(self.absorbed_volume))
        ratio = Decimal(str(self.pressure_ratio))
        if not volume.is_finite() or volume < 0:
            raise ValueError("absorbed volume must be non-negative and finite")
        if not ratio.is_finite() or ratio <= 0:
            raise ValueError("pressure ratio must be positive and finite")
        _validate_confidence(self.confidence, "absorption confidence")
        object.__setattr__(self, "absorbed_volume", volume)
        object.__setattr__(self, "pressure_ratio", ratio)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class AbsorptionStatistics:
    """Aggregate counts for deterministic absorption detections."""

    total_cells: int
    buy_absorptions: int
    sell_absorptions: int
    total_absorptions: int
    cells_without_absorption: int

    def __post_init__(self) -> None:
        values = (
            self.total_cells,
            self.buy_absorptions,
            self.sell_absorptions,
            self.total_absorptions,
            self.cells_without_absorption,
        )
        if any(value < 0 for value in values):
            raise ValueError("absorption statistics cannot be negative")
        if self.total_absorptions != self.buy_absorptions + self.sell_absorptions:
            raise ValueError("total absorptions must equal buy plus sell absorptions")
        if self.cells_without_absorption + self.total_absorptions != self.total_cells:
            raise ValueError("absorption statistics must account for all cells")


@dataclass(frozen=True, slots=True)
class AbsorptionResult:
    """Immutable result set for deterministic absorption detection."""

    matrix: FootprintMatrix
    imbalances: FootprintImbalanceResult
    detections: tuple[FootprintAbsorption, ...]
    configuration: AbsorptionConfiguration = field(
        default_factory=AbsorptionConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.imbalances.matrix != self.matrix:
            raise ValueError("absorption result imbalances must reference matrix")
        ordered = tuple(
            sorted(
                self.detections,
                key=lambda d: (
                    d.position.row_index,
                    d.position.column_index,
                    d.absorption_type.value,
                ),
            )
        )
        if self.detections != ordered:
            raise ValueError("absorption detections must be ordered")
        ids = [d.cell_id for d in self.detections]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate absorption detections are not allowed")
        matrix_ids = {cell.cell_id for cell in self.matrix.cells}
        if set(ids) - matrix_ids:
            raise ValueError("absorption detections must reference matrix cells")
        object.__setattr__(self, "detections", ordered)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def absorptions(self) -> tuple[FootprintAbsorption, ...]:
        return self.detections

    def buy_absorptions(self) -> tuple[FootprintAbsorption, ...]:
        return tuple(
            d
            for d in self.detections
            if d.absorption_type == AbsorptionType.BUY_ABSORPTION
        )

    def sell_absorptions(self) -> tuple[FootprintAbsorption, ...]:
        return tuple(
            d
            for d in self.detections
            if d.absorption_type == AbsorptionType.SELL_ABSORPTION
        )

    def lookup(self, cell_id: str) -> FootprintAbsorption | None:
        return next((d for d in self.detections if d.cell_id == cell_id), None)

    def statistics(self) -> AbsorptionStatistics:
        buy = len(self.buy_absorptions())
        sell = len(self.sell_absorptions())
        total = len(self.detections)
        cells = self.matrix.statistics().total_cells
        return AbsorptionStatistics(cells, buy, sell, total, cells - total)


@dataclass(frozen=True, slots=True)
class FootprintAbsorptionDetector:
    """Detect deterministic absorption from matrix values and imbalance pressure."""

    configuration: AbsorptionConfiguration = field(
        default_factory=AbsorptionConfiguration
    )

    def detect(
        self, matrix: FootprintMatrix, imbalances: FootprintImbalanceResult
    ) -> AbsorptionResult:
        if imbalances.matrix != matrix:
            raise ValueError(
                "absorption detector inputs must reference the same matrix"
            )
        found = []
        for imbalance in imbalances.imbalances():
            cell = matrix.cell(
                imbalance.position.row_index, imbalance.position.column_index
            )
            detection = self._build(cell, imbalance)
            if detection is not None:
                found.append(detection)
        return AbsorptionResult(
            matrix,
            imbalances,
            tuple(found),
            self.configuration,
            {"detector": "FootprintAbsorptionDetector"},
        )

    @staticmethod
    def _decimal(value: FootprintValue | None) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value.numeric_value.value))

    def _build(
        self, cell: MatrixCell, imbalance: FootprintImbalance
    ) -> FootprintAbsorption | None:
        if imbalance.ratio < self.configuration.minimum_pressure_ratio:
            return None
        if imbalance.imbalance_type == ImbalanceType.ASK_IMBALANCE:
            passive = self._decimal(cell.interpretation.bid())
            absorption_type = AbsorptionType.BUY_ABSORPTION
            passive_side = AbsorptionSide.BID
        else:
            passive = self._decimal(cell.interpretation.ask())
            absorption_type = AbsorptionType.SELL_ABSORPTION
            passive_side = AbsorptionSide.ASK
        if passive is None or passive < self.configuration.minimum_absorbed_volume:
            return None
        confidence = min(
            1.0, float(imbalance.ratio / self.configuration.minimum_pressure_ratio)
        )
        return FootprintAbsorption(
            cell.cell_id,
            cell.position,
            absorption_type,
            passive_side,
            passive,
            imbalance.ratio,
            imbalance,
            confidence,
            {"source": "footprint_imbalance"},
        )


@dataclass(frozen=True, slots=True)
class VolumeClusterConfiguration:
    """Immutable settings for deterministic volume cluster analysis."""

    high_volume_percentile: Decimal = Decimal("80")
    low_volume_percentile: Decimal = Decimal("20")
    minimum_volume: Decimal = Decimal("0")
    strict_mode: bool = True

    def __post_init__(self) -> None:
        high = Decimal(str(self.high_volume_percentile))
        low = Decimal(str(self.low_volume_percentile))
        minimum = Decimal(str(self.minimum_volume))
        if any(not value.is_finite() for value in (high, low, minimum)):
            raise ValueError("volume cluster configuration values must be finite")
        if high < 0 or high > 100 or low < 0 or low > 100:
            raise ValueError("volume cluster percentiles must be between 0 and 100")
        if low > high:
            raise ValueError(
                "low volume percentile cannot exceed high volume percentile"
            )
        if minimum < 0:
            raise ValueError("minimum volume cannot be negative")
        object.__setattr__(self, "high_volume_percentile", high)
        object.__setattr__(self, "low_volume_percentile", low)
        object.__setattr__(self, "minimum_volume", minimum)


@dataclass(frozen=True, slots=True)
class VolumeCluster:
    """Deterministic volume classification for one footprint matrix cell."""

    cell_id: str
    row: int
    column: int
    total_volume: Decimal
    cluster_type: VolumeClusterType
    percentile: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("volume cluster cell id is required")
        if self.row < 0 or self.column < 0:
            raise ValueError("volume cluster position must be non-negative")
        total = Decimal(str(self.total_volume))
        percentile = Decimal(str(self.percentile))
        if not total.is_finite() or total < 0:
            raise ValueError(
                "volume cluster total volume must be non-negative and finite"
            )
        if not percentile.is_finite() or percentile < 0 or percentile > 100:
            raise ValueError("volume cluster percentile must be between 0 and 100")
        _validate_confidence(self.confidence, "volume cluster confidence")
        object.__setattr__(self, "total_volume", total)
        object.__setattr__(self, "percentile", percentile)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class VolumeClusterStatistics:
    """Aggregate counts and values for deterministic volume clusters."""

    total_cells: int
    high_volume_cells: int
    low_volume_cells: int
    normal_volume_cells: int
    maximum_volume: Decimal
    minimum_volume: Decimal
    average_volume: Decimal

    def __post_init__(self) -> None:
        counts = (
            self.total_cells,
            self.high_volume_cells,
            self.low_volume_cells,
            self.normal_volume_cells,
        )
        if any(count < 0 for count in counts):
            raise ValueError("volume cluster statistics cannot be negative")
        if (
            self.high_volume_cells + self.low_volume_cells + self.normal_volume_cells
            != self.total_cells
        ):
            raise ValueError("volume cluster statistics must account for all cells")
        maximum = Decimal(str(self.maximum_volume))
        minimum = Decimal(str(self.minimum_volume))
        average = Decimal(str(self.average_volume))
        if any(not value.is_finite() for value in (maximum, minimum, average)):
            raise ValueError("volume cluster statistics values must be finite")
        if maximum < minimum:
            raise ValueError("maximum volume must be greater than minimum volume")
        object.__setattr__(self, "maximum_volume", maximum)
        object.__setattr__(self, "minimum_volume", minimum)
        object.__setattr__(self, "average_volume", average)


@dataclass(frozen=True, slots=True)
class VolumeClusterResult:
    """Immutable result set for deterministic volume cluster analysis."""

    matrix: "FootprintMatrix"
    clusters: tuple[VolumeCluster, ...]
    statistics_value: VolumeClusterStatistics
    configuration: VolumeClusterConfiguration = field(
        default_factory=VolumeClusterConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(self.clusters, key=lambda c: (c.row, c.column, c.cell_id))
        )
        if self.clusters != ordered:
            raise ValueError("volume clusters must be ordered")
        if len({cluster.cell_id for cluster in self.clusters}) != len(self.clusters):
            raise ValueError("duplicate volume clusters are not allowed")
        matrix_cells = {
            (cell.cell_id, cell.row_index, cell.column_index)
            for cell in self.matrix.cells
        }
        cluster_cells = {
            (cluster.cell_id, cluster.row, cluster.column) for cluster in self.clusters
        }
        if matrix_cells != cluster_cells:
            raise ValueError("volume clusters must reference matrix cells")
        if self.statistics_value.total_cells != len(self.clusters):
            raise ValueError("volume cluster statistics must match clusters")
        object.__setattr__(self, "clusters", ordered)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def high_volume_cells(self) -> tuple[VolumeCluster, ...]:
        return tuple(
            c for c in self.clusters if c.cluster_type == VolumeClusterType.HIGH_VOLUME
        )

    def low_volume_cells(self) -> tuple[VolumeCluster, ...]:
        return tuple(
            c for c in self.clusters if c.cluster_type == VolumeClusterType.LOW_VOLUME
        )

    def normal_volume_cells(self) -> tuple[VolumeCluster, ...]:
        return tuple(
            c
            for c in self.clusters
            if c.cluster_type == VolumeClusterType.NORMAL_VOLUME
        )

    def lookup(self, cell_id: str) -> VolumeCluster | None:
        return next(
            (cluster for cluster in self.clusters if cluster.cell_id == cell_id), None
        )

    def statistics(self) -> VolumeClusterStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class VolumeClusterAnalyzer:
    """Classify individual matrix cells by deterministic total-volume percentiles."""

    configuration: VolumeClusterConfiguration = field(
        default_factory=VolumeClusterConfiguration
    )

    def analyze(self, matrix: "FootprintMatrix") -> VolumeClusterResult:
        volumes = tuple(self._total_volume(cell) for cell in matrix.cells)
        sorted_volumes = tuple(sorted(volumes))
        clusters = tuple(
            self._cluster(cell, volume, sorted_volumes)
            for cell, volume in zip(matrix.cells, volumes, strict=True)
        )
        stats = VolumeClusterStatistics(
            len(clusters),
            sum(c.cluster_type == VolumeClusterType.HIGH_VOLUME for c in clusters),
            sum(c.cluster_type == VolumeClusterType.LOW_VOLUME for c in clusters),
            sum(c.cluster_type == VolumeClusterType.NORMAL_VOLUME for c in clusters),
            max(volumes, default=Decimal("0")),
            min(volumes, default=Decimal("0")),
            (
                Decimal("0")
                if not volumes
                else sum(volumes, Decimal("0")) / Decimal(len(volumes))
            ),
        )
        return VolumeClusterResult(
            matrix,
            clusters,
            stats,
            self.configuration,
            {"detector": "VolumeClusterAnalyzer"},
        )

    @staticmethod
    def _total_volume(cell: "MatrixCell") -> Decimal:
        total = cell.interpretation.total_volume()
        if total is not None:
            return Decimal(str(total.numeric_value.value))
        bid = cell.interpretation.bid()
        ask = cell.interpretation.ask()
        bid_value = (
            Decimal("0") if bid is None else Decimal(str(bid.numeric_value.value))
        )
        ask_value = (
            Decimal("0") if ask is None else Decimal(str(ask.numeric_value.value))
        )
        return bid_value + ask_value

    def _cluster(
        self, cell: "MatrixCell", volume: Decimal, sorted_volumes: tuple[Decimal, ...]
    ) -> VolumeCluster:
        percentile = self._percentile(volume, sorted_volumes)
        if volume < self.configuration.minimum_volume:
            cluster_type = VolumeClusterType.NORMAL_VOLUME
        elif percentile >= self.configuration.high_volume_percentile:
            cluster_type = VolumeClusterType.HIGH_VOLUME
        elif percentile <= self.configuration.low_volume_percentile:
            cluster_type = VolumeClusterType.LOW_VOLUME
        else:
            cluster_type = VolumeClusterType.NORMAL_VOLUME
        return VolumeCluster(
            cell.cell_id,
            cell.row_index,
            cell.column_index,
            volume,
            cluster_type,
            percentile,
            1.0,
            {"source": "footprint_matrix"},
        )

    @staticmethod
    def _percentile(volume: Decimal, sorted_volumes: tuple[Decimal, ...]) -> Decimal:
        if len(sorted_volumes) <= 1:
            return Decimal("100")
        less = sum(candidate < volume for candidate in sorted_volumes)
        return (Decimal(less) / Decimal(len(sorted_volumes) - 1)) * Decimal("100")


@dataclass(frozen=True, slots=True)
class PointOfControlConfiguration:
    """Immutable settings for deterministic session POC analysis."""

    poc_type: PointOfControlType = PointOfControlType.SESSION_POC
    strict_mode: bool = True

    def __post_init__(self) -> None:
        if self.poc_type != PointOfControlType.SESSION_POC:
            raise ValueError("only SESSION_POC is supported")


@dataclass(frozen=True, slots=True)
class PointOfControl:
    """The matrix row with the greatest traded volume."""

    row: int
    total_volume: Decimal
    poc_type: PointOfControlType
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("point of control row must be non-negative")
        volume = Decimal(str(self.total_volume))
        if not volume.is_finite() or volume < 0:
            raise ValueError("point of control volume must be non-negative and finite")
        if self.poc_type != PointOfControlType.SESSION_POC:
            raise ValueError("only SESSION_POC is supported")
        _validate_confidence(self.confidence, "point of control confidence")
        object.__setattr__(self, "total_volume", volume)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PointOfControlStatistics:
    """Aggregate statistics for deterministic session POC analysis."""

    total_rows: int
    poc_row: int
    poc_volume: Decimal
    total_volume: Decimal
    tied_poc_rows: int

    def __post_init__(self) -> None:
        if self.total_rows <= 0 or self.poc_row < 0 or self.poc_row >= self.total_rows:
            raise ValueError("point of control statistics row bounds are invalid")
        if self.tied_poc_rows <= 0:
            raise ValueError("point of control tied row count must be positive")
        poc = Decimal(str(self.poc_volume))
        total = Decimal(str(self.total_volume))
        if any(not value.is_finite() or value < 0 for value in (poc, total)):
            raise ValueError(
                "point of control statistics volumes must be non-negative and finite"
            )
        if poc > total:
            raise ValueError("point of control volume cannot exceed total volume")
        object.__setattr__(self, "poc_volume", poc)
        object.__setattr__(self, "total_volume", total)


@dataclass(frozen=True, slots=True)
class PointOfControlResult:
    """Immutable result for deterministic session POC analysis."""

    matrix: "FootprintMatrix"
    poc: PointOfControl
    statistics_value: PointOfControlStatistics
    configuration: PointOfControlConfiguration = field(
        default_factory=PointOfControlConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.poc.row >= self.matrix.dimensions_value.rows:
            raise ValueError("point of control must reference matrix row")
        if (
            self.statistics_value.poc_row != self.poc.row
            or self.statistics_value.poc_volume != self.poc.total_volume
        ):
            raise ValueError("point of control statistics must match poc")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def statistics(self) -> PointOfControlStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class PointOfControlAnalyzer:
    """Find the session POC by highest matrix-row traded volume."""

    configuration: PointOfControlConfiguration = field(
        default_factory=PointOfControlConfiguration
    )

    def analyze(self, matrix: "FootprintMatrix") -> PointOfControlResult:
        row_volumes = _matrix_row_volumes(matrix)
        max_volume = max(row_volumes)
        poc_row = next(
            index for index, volume in enumerate(row_volumes) if volume == max_volume
        )
        poc = PointOfControl(
            poc_row,
            max_volume,
            self.configuration.poc_type,
            1.0,
            {"source": "footprint_matrix", "tie_break": "lowest_row"},
        )
        stats = PointOfControlStatistics(
            len(row_volumes),
            poc_row,
            max_volume,
            sum(row_volumes, Decimal("0")),
            sum(v == max_volume for v in row_volumes),
        )
        return PointOfControlResult(
            matrix,
            poc,
            stats,
            self.configuration,
            {"detector": "PointOfControlAnalyzer"},
        )


@dataclass(frozen=True, slots=True)
class HighVolumeNodeConfiguration:
    """Immutable settings for deterministic HVN row analysis."""

    percentile_threshold: Decimal = Decimal("80")
    minimum_volume: Decimal = Decimal("0")
    strict_mode: bool = True

    def __post_init__(self) -> None:
        threshold = Decimal(str(self.percentile_threshold))
        minimum = Decimal(str(self.minimum_volume))
        if (
            any(not value.is_finite() for value in (threshold, minimum))
            or threshold < 0
            or threshold > 100
            or minimum < 0
        ):
            raise ValueError("high volume node configuration values are invalid")
        object.__setattr__(self, "percentile_threshold", threshold)
        object.__setattr__(self, "minimum_volume", minimum)


@dataclass(frozen=True, slots=True)
class LowVolumeNodeConfiguration:
    """Immutable settings for deterministic LVN row analysis."""

    percentile_threshold: Decimal = Decimal("20")
    minimum_volume: Decimal = Decimal("0")
    strict_mode: bool = True

    def __post_init__(self) -> None:
        threshold = Decimal(str(self.percentile_threshold))
        minimum = Decimal(str(self.minimum_volume))
        if (
            any(not value.is_finite() for value in (threshold, minimum))
            or threshold < 0
            or threshold > 100
            or minimum < 0
        ):
            raise ValueError("low volume node configuration values are invalid")
        object.__setattr__(self, "percentile_threshold", threshold)
        object.__setattr__(self, "minimum_volume", minimum)


@dataclass(frozen=True, slots=True)
class HighVolumeNode:
    row: int
    total_volume: Decimal
    percentile: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_node(self, "high volume node")


@dataclass(frozen=True, slots=True)
class LowVolumeNode:
    row: int
    total_volume: Decimal
    percentile: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_node(self, "low volume node")


def _validate_node(node: HighVolumeNode | LowVolumeNode, name: str) -> None:
    if node.row < 0:
        raise ValueError(f"{name} row must be non-negative")
    volume = Decimal(str(node.total_volume))
    percentile = Decimal(str(node.percentile))
    if (
        not volume.is_finite()
        or volume < 0
        or not percentile.is_finite()
        or percentile < 0
        or percentile > 100
    ):
        raise ValueError(f"{name} values are invalid")
    _validate_confidence(node.confidence, f"{name} confidence")
    object.__setattr__(node, "total_volume", volume)
    object.__setattr__(node, "percentile", percentile)
    object.__setattr__(node, "metadata", MappingProxyType(dict(node.metadata)))


@dataclass(frozen=True, slots=True)
class HighVolumeNodeStatistics:
    total_rows: int
    node_count: int
    maximum_volume: Decimal
    threshold_percentile: Decimal

    def __post_init__(self) -> None:
        _validate_node_stats(self, "high volume node statistics")


@dataclass(frozen=True, slots=True)
class LowVolumeNodeStatistics:
    total_rows: int
    node_count: int
    minimum_volume: Decimal
    threshold_percentile: Decimal

    def __post_init__(self) -> None:
        _validate_node_stats(self, "low volume node statistics")


def _validate_node_stats(
    stats: HighVolumeNodeStatistics | LowVolumeNodeStatistics, name: str
) -> None:
    if (
        stats.total_rows <= 0
        or stats.node_count < 0
        or stats.node_count > stats.total_rows
    ):
        raise ValueError(f"{name} counts are invalid")
    for field_name in ("maximum_volume", "minimum_volume"):
        if hasattr(stats, field_name):
            value = Decimal(str(getattr(stats, field_name)))
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} volume is invalid")
            object.__setattr__(stats, field_name, value)
    threshold = Decimal(str(stats.threshold_percentile))
    if not threshold.is_finite() or threshold < 0 or threshold > 100:
        raise ValueError(f"{name} threshold is invalid")
    object.__setattr__(stats, "threshold_percentile", threshold)


@dataclass(frozen=True, slots=True)
class HighVolumeNodeResult:
    matrix: "FootprintMatrix"
    nodes: tuple[HighVolumeNode, ...]
    statistics_value: HighVolumeNodeStatistics
    configuration: HighVolumeNodeConfiguration = field(
        default_factory=HighVolumeNodeConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_node_result(self, self.nodes, "high volume nodes")

    def lookup(self, row: int) -> HighVolumeNode | None:
        return next((node for node in self.nodes if node.row == row), None)

    def statistics(self) -> HighVolumeNodeStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class LowVolumeNodeResult:
    matrix: "FootprintMatrix"
    nodes: tuple[LowVolumeNode, ...]
    statistics_value: LowVolumeNodeStatistics
    configuration: LowVolumeNodeConfiguration = field(
        default_factory=LowVolumeNodeConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_node_result(self, self.nodes, "low volume nodes")

    def lookup(self, row: int) -> LowVolumeNode | None:
        return next((node for node in self.nodes if node.row == row), None)

    def statistics(self) -> LowVolumeNodeStatistics:
        return self.statistics_value


def _validate_node_result(
    result: HighVolumeNodeResult | LowVolumeNodeResult,
    nodes: tuple[Any, ...],
    name: str,
) -> None:
    ordered = tuple(sorted(nodes, key=lambda node: node.row))
    if nodes != ordered:
        raise ValueError(f"{name} must be ordered")
    if len({node.row for node in nodes}) != len(nodes):
        raise ValueError(f"duplicate {name} are not allowed")
    if any(node.row >= result.matrix.dimensions_value.rows for node in nodes):
        raise ValueError(f"{name} must reference matrix rows")
    if (
        result.statistics_value.total_rows != result.matrix.dimensions_value.rows
        or result.statistics_value.node_count != len(nodes)
    ):
        raise ValueError(f"{name} statistics must match nodes")
    object.__setattr__(result, "nodes", ordered)
    object.__setattr__(result, "metadata", MappingProxyType(dict(result.metadata)))


@dataclass(frozen=True, slots=True)
class HighVolumeNodeAnalyzer:
    configuration: HighVolumeNodeConfiguration = field(
        default_factory=HighVolumeNodeConfiguration
    )

    def analyze(self, matrix: "FootprintMatrix") -> HighVolumeNodeResult:
        volumes = _matrix_row_volumes(matrix)
        sorted_volumes = tuple(sorted(volumes))
        nodes = tuple(
            HighVolumeNode(
                row,
                volume,
                _volume_percentile(volume, sorted_volumes),
                1.0,
                {"source": "footprint_matrix"},
            )
            for row, volume in enumerate(volumes)
            if volume >= self.configuration.minimum_volume
            and _volume_percentile(volume, sorted_volumes)
            >= self.configuration.percentile_threshold
        )
        stats = HighVolumeNodeStatistics(
            len(volumes),
            len(nodes),
            max(volumes),
            self.configuration.percentile_threshold,
        )
        return HighVolumeNodeResult(
            matrix,
            nodes,
            stats,
            self.configuration,
            {"detector": "HighVolumeNodeAnalyzer"},
        )


@dataclass(frozen=True, slots=True)
class LowVolumeNodeAnalyzer:
    configuration: LowVolumeNodeConfiguration = field(
        default_factory=LowVolumeNodeConfiguration
    )

    def analyze(self, matrix: "FootprintMatrix") -> LowVolumeNodeResult:
        volumes = _matrix_row_volumes(matrix)
        sorted_volumes = tuple(sorted(volumes))
        nodes = tuple(
            LowVolumeNode(
                row,
                volume,
                _volume_percentile(volume, sorted_volumes),
                1.0,
                {"source": "footprint_matrix"},
            )
            for row, volume in enumerate(volumes)
            if volume >= self.configuration.minimum_volume
            and _volume_percentile(volume, sorted_volumes)
            <= self.configuration.percentile_threshold
        )
        stats = LowVolumeNodeStatistics(
            len(volumes),
            len(nodes),
            min(volumes),
            self.configuration.percentile_threshold,
        )
        return LowVolumeNodeResult(
            matrix,
            nodes,
            stats,
            self.configuration,
            {"detector": "LowVolumeNodeAnalyzer"},
        )


@dataclass(frozen=True, slots=True)
class ValueAreaConfiguration:
    value_area_percentage: Decimal = Decimal("70")
    strict_mode: bool = True
    minimum_rows: int = 1

    def __post_init__(self) -> None:
        pct = Decimal(str(self.value_area_percentage))
        if not pct.is_finite() or pct <= 0 or pct > 100:
            raise ValueError("value area percentage must be between 0 and 100")
        if self.minimum_rows <= 0:
            raise ValueError("value area minimum rows must be positive")
        object.__setattr__(self, "value_area_percentage", pct)


@dataclass(frozen=True, slots=True)
class ValueArea:
    vah: int
    val: int
    poc_row: int
    included_rows: tuple[int, ...]
    included_volume: Decimal
    target_volume: Decimal
    coverage_percentage: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.val < 0
            or self.vah < self.val
            or self.poc_row < self.val
            or self.poc_row > self.vah
        ):
            raise ValueError("value area boundaries are invalid")
        ordered = tuple(sorted(self.included_rows))
        if self.included_rows != ordered:
            raise ValueError("value area rows must be ordered")
        if len(set(self.included_rows)) != len(self.included_rows):
            raise ValueError("duplicate value area rows are not allowed")
        if self.val != self.included_rows[0] or self.vah != self.included_rows[-1]:
            raise ValueError("value area boundaries must match rows")
        included = Decimal(str(self.included_volume))
        target = Decimal(str(self.target_volume))
        coverage = Decimal(str(self.coverage_percentage))
        if any(not v.is_finite() or v < 0 for v in (included, target, coverage)):
            raise ValueError("value area volumes must be non-negative and finite")
        _validate_confidence(self.confidence, "value area confidence")
        object.__setattr__(self, "included_volume", included)
        object.__setattr__(self, "target_volume", target)
        object.__setattr__(self, "coverage_percentage", coverage)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ValueAreaStatistics:
    total_rows: int
    included_rows: int
    total_volume: Decimal
    included_volume: Decimal
    target_percentage: Decimal
    coverage_percentage: Decimal

    def __post_init__(self) -> None:
        if (
            self.total_rows <= 0
            or self.included_rows <= 0
            or self.included_rows > self.total_rows
        ):
            raise ValueError("value area statistics counts are invalid")
        for name in (
            "total_volume",
            "included_volume",
            "target_percentage",
            "coverage_percentage",
        ):
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite() or value < 0:
                raise ValueError("value area statistics values are invalid")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ValueAreaResult:
    matrix: "FootprintMatrix"
    value_area: ValueArea
    poc: PointOfControl
    statistics_value: ValueAreaStatistics
    configuration: ValueAreaConfiguration = field(
        default_factory=ValueAreaConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value_area.vah >= self.matrix.dimensions_value.rows:
            raise ValueError("value area must reference matrix rows")
        if self.poc.row != self.value_area.poc_row:
            raise ValueError("value area poc must match")
        if (
            self.statistics_value.total_rows != self.matrix.dimensions_value.rows
            or self.statistics_value.included_rows != len(self.value_area.included_rows)
        ):
            raise ValueError("value area statistics must match")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def statistics(self) -> ValueAreaStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class ValueAreaAnalyzer:
    configuration: ValueAreaConfiguration = field(
        default_factory=ValueAreaConfiguration
    )

    def analyze(
        self, matrix: "FootprintMatrix", poc_result: PointOfControlResult | None = None
    ) -> ValueAreaResult:
        poc_result = (
            PointOfControlAnalyzer().analyze(matrix)
            if poc_result is None
            else poc_result
        )
        if poc_result.matrix != matrix:
            raise ValueError("value area poc must reference matrix")
        volumes = _matrix_row_volumes(matrix)
        total = sum(volumes, Decimal("0"))
        target = total * self.configuration.value_area_percentage / Decimal("100")
        low = high = poc_result.poc.row
        included = volumes[low]
        while (
            included < target or (high - low + 1) < self.configuration.minimum_rows
        ) and (low > 0 or high + 1 < len(volumes)):
            up = volumes[low - 1] if low > 0 else None
            down = volumes[high + 1] if high + 1 < len(volumes) else None
            if up is not None and (down is None or up >= down):
                low -= 1
                included += up
            elif down is not None:
                high += 1
                included += down
        rows = tuple(range(low, high + 1))
        coverage = Decimal("0") if total == 0 else included / total * Decimal("100")
        va = ValueArea(
            high,
            low,
            poc_result.poc.row,
            rows,
            included,
            target,
            coverage,
            1.0,
            {"source": "footprint_matrix"},
        )
        stats = ValueAreaStatistics(
            len(volumes),
            len(rows),
            total,
            included,
            self.configuration.value_area_percentage,
            coverage,
        )
        return ValueAreaResult(
            matrix,
            va,
            poc_result.poc,
            stats,
            self.configuration,
            {"detector": "ValueAreaAnalyzer"},
        )


@dataclass(frozen=True, slots=True)
class DevelopingPointOfControlConfiguration:
    strict_mode: bool = True


@dataclass(frozen=True, slots=True)
class DevelopingPointOfControl:
    slice_index: int
    row: int
    total_volume: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.slice_index < 0 or self.row < 0:
            raise ValueError("developing poc indexes must be non-negative")
        volume = Decimal(str(self.total_volume))
        if not volume.is_finite() or volume < 0:
            raise ValueError("developing poc volume must be non-negative and finite")
        _validate_confidence(self.confidence, "developing poc confidence")
        object.__setattr__(self, "total_volume", volume)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class DevelopingPointOfControlStatistics:
    total_slices: int
    current_row: int
    previous_row: int | None
    movement_distance: int
    stable_slices: int

    def __post_init__(self) -> None:
        if self.total_slices <= 0 or self.current_row < 0 or self.movement_distance < 0:
            raise ValueError("developing poc statistics are invalid")
        if self.previous_row is not None and self.previous_row < 0:
            raise ValueError("developing poc previous row is invalid")


@dataclass(frozen=True, slots=True)
class DevelopingPointOfControlResult:
    matrix: "FootprintMatrix"
    history: tuple[DevelopingPointOfControl, ...]
    statistics_value: DevelopingPointOfControlStatistics
    configuration: DevelopingPointOfControlConfiguration = field(
        default_factory=DevelopingPointOfControlConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_developing_history(self.matrix, self.history, "developing poc")
        if (
            self.statistics_value.total_slices != len(self.history)
            or self.statistics_value.current_row != self.current_poc.row
        ):
            raise ValueError("developing poc statistics must match history")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def current_poc(self) -> DevelopingPointOfControl:
        return self.history[-1]

    @property
    def previous_poc(self) -> DevelopingPointOfControl | None:
        return None if len(self.history) == 1 else self.history[-2]

    @property
    def movement_direction(self) -> str:
        if self.previous_poc is None or self.current_poc.row == self.previous_poc.row:
            return "STABLE"
        return "UP" if self.current_poc.row < self.previous_poc.row else "DOWN"

    @property
    def movement_distance(self) -> int:
        return (
            0
            if self.previous_poc is None
            else abs(self.current_poc.row - self.previous_poc.row)
        )

    def statistics(self) -> DevelopingPointOfControlStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class DevelopingPointOfControlAnalyzer:
    configuration: DevelopingPointOfControlConfiguration = field(
        default_factory=DevelopingPointOfControlConfiguration
    )

    def analyze(self, matrix: "FootprintMatrix") -> DevelopingPointOfControlResult:
        history = tuple(
            self._slice_poc(matrix, column)
            for column in range(matrix.dimensions_value.columns)
        )
        previous = None if len(history) == 1 else history[-2].row
        stats = DevelopingPointOfControlStatistics(
            len(history),
            history[-1].row,
            previous,
            0 if previous is None else abs(history[-1].row - previous),
            sum(
                1 for left, right in zip(history, history[1:]) if left.row == right.row
            ),
        )
        return DevelopingPointOfControlResult(
            matrix,
            history,
            stats,
            self.configuration,
            {"detector": "DevelopingPointOfControlAnalyzer"},
        )

    @staticmethod
    def _slice_poc(matrix: "FootprintMatrix", column: int) -> DevelopingPointOfControl:
        volumes = tuple(
            sum(
                (
                    VolumeClusterAnalyzer._total_volume(row.cells[c])
                    for c in range(column + 1)
                ),
                Decimal("0"),
            )
            for row in matrix.rows
        )
        max_volume = max(volumes)
        row = next(
            index for index, volume in enumerate(volumes) if volume == max_volume
        )
        return DevelopingPointOfControl(
            column, row, max_volume, 1.0, {"source": "footprint_matrix"}
        )


@dataclass(frozen=True, slots=True)
class DevelopingValueAreaConfiguration(ValueAreaConfiguration):
    pass


@dataclass(frozen=True, slots=True)
class DevelopingValueArea:
    slice_index: int
    vah: int
    val: int
    included_rows: tuple[int, ...]
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.slice_index < 0 or self.val < 0 or self.vah < self.val:
            raise ValueError("developing value area boundaries are invalid")
        if self.included_rows != tuple(sorted(self.included_rows)) or len(
            set(self.included_rows)
        ) != len(self.included_rows):
            raise ValueError(
                "developing value area rows must be ordered without duplicates"
            )
        if self.included_rows and (
            self.val != self.included_rows[0] or self.vah != self.included_rows[-1]
        ):
            raise ValueError("developing value area boundaries must match rows")
        _validate_confidence(self.confidence, "developing value area confidence")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class DevelopingValueAreaStatistics:
    total_slices: int
    current_vah: int
    current_val: int
    previous_vah: int | None
    previous_val: int | None
    expansion: int
    contraction: int

    def __post_init__(self) -> None:
        if (
            self.total_slices <= 0
            or self.current_vah < self.current_val
            or self.expansion < 0
            or self.contraction < 0
        ):
            raise ValueError("developing value area statistics are invalid")


@dataclass(frozen=True, slots=True)
class DevelopingValueAreaResult:
    matrix: "FootprintMatrix"
    history: tuple[DevelopingValueArea, ...]
    statistics_value: DevelopingValueAreaStatistics
    configuration: DevelopingValueAreaConfiguration = field(
        default_factory=DevelopingValueAreaConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_developing_history(self.matrix, self.history, "developing value area")
        if self.statistics_value.total_slices != len(self.history):
            raise ValueError("developing value area statistics must match history")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def current_value_area(self) -> DevelopingValueArea:
        return self.history[-1]

    @property
    def previous_value_area(self) -> DevelopingValueArea | None:
        return None if len(self.history) == 1 else self.history[-2]

    @property
    def movement(self) -> str:
        previous = self.previous_value_area
        current = self.current_value_area
        if previous is None or (current.vah, current.val) == (
            previous.vah,
            previous.val,
        ):
            return "STABLE"
        if current.vah <= previous.vah and current.val < previous.val:
            return "UP"
        if current.vah > previous.vah and current.val >= previous.val:
            return "DOWN"
        return "MIXED"

    def statistics(self) -> DevelopingValueAreaStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class DevelopingValueAreaAnalyzer:
    configuration: DevelopingValueAreaConfiguration = field(
        default_factory=DevelopingValueAreaConfiguration
    )

    def analyze(self, matrix: "FootprintMatrix") -> DevelopingValueAreaResult:
        history = []
        for column in range(matrix.dimensions_value.columns):
            sliced = _prefix_row_volumes(matrix, column)
            va = _value_area_from_volumes(sliced, self.configuration)
            history.append(
                DevelopingValueArea(
                    column,
                    va.vah,
                    va.val,
                    va.included_rows,
                    1.0,
                    {"source": "footprint_matrix"},
                )
            )
        hist = tuple(history)
        prev = None if len(hist) == 1 else hist[-2]
        cur = hist[-1]
        expansion = (
            0 if prev is None else max(0, (cur.vah - cur.val) - (prev.vah - prev.val))
        )
        contraction = (
            0 if prev is None else max(0, (prev.vah - prev.val) - (cur.vah - cur.val))
        )
        stats = DevelopingValueAreaStatistics(
            len(hist),
            cur.vah,
            cur.val,
            None if prev is None else prev.vah,
            None if prev is None else prev.val,
            expansion,
            contraction,
        )
        return DevelopingValueAreaResult(
            matrix,
            hist,
            stats,
            self.configuration,
            {"detector": "DevelopingValueAreaAnalyzer"},
        )


def _validate_developing_history(
    matrix: "FootprintMatrix", history: tuple[Any, ...], name: str
) -> None:
    if not history:
        raise ValueError(f"{name} history is required")
    if tuple(item.slice_index for item in history) != tuple(range(len(history))):
        raise ValueError(f"{name} history must be ordered")
    if len({item.slice_index for item in history}) != len(history):
        raise ValueError(f"duplicate {name} history entries are not allowed")
    if len(history) != matrix.dimensions_value.columns:
        raise ValueError(f"{name} history must reference matrix slices")


def _prefix_row_volumes(matrix: "FootprintMatrix", column: int) -> tuple[Decimal, ...]:
    return tuple(
        sum(
            (
                VolumeClusterAnalyzer._total_volume(row.cells[c])
                for c in range(column + 1)
            ),
            Decimal("0"),
        )
        for row in matrix.rows
    )


def _value_area_from_volumes(
    volumes: tuple[Decimal, ...], configuration: ValueAreaConfiguration
) -> DevelopingValueArea:
    max_volume = max(volumes)
    poc_row = next(
        index for index, volume in enumerate(volumes) if volume == max_volume
    )
    total = sum(volumes, Decimal("0"))
    target = total * configuration.value_area_percentage / Decimal("100")
    low = high = poc_row
    included = volumes[poc_row]
    while (included < target or (high - low + 1) < configuration.minimum_rows) and (
        low > 0 or high + 1 < len(volumes)
    ):
        up = volumes[low - 1] if low > 0 else None
        down = volumes[high + 1] if high + 1 < len(volumes) else None
        if up is not None and (down is None or up >= down):
            low -= 1
            included += up
        elif down is not None:
            high += 1
            included += down
    return DevelopingValueArea(0, high, low, tuple(range(low, high + 1)), 1.0)


@dataclass(frozen=True, slots=True)
class UnfinishedAuction:
    auction_type: UnfinishedAuctionType
    row: int
    bid_volume: Decimal
    ask_volume: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("unfinished auction row must be non-negative")
        bid, ask = Decimal(str(self.bid_volume)), Decimal(str(self.ask_volume))
        if any(not value.is_finite() or value < 0 for value in (bid, ask)):
            raise ValueError("unfinished auction volumes are invalid")
        _validate_confidence(self.confidence, "unfinished auction confidence")
        object.__setattr__(self, "bid_volume", bid)
        object.__setattr__(self, "ask_volume", ask)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class UnfinishedAuctionConfiguration:
    minimum_boundary_volume: Decimal = Decimal("1")
    strict_mode: bool = True

    def __post_init__(self) -> None:
        value = Decimal(str(self.minimum_boundary_volume))
        if not value.is_finite() or value < 0:
            raise ValueError("unfinished auction minimum volume is invalid")
        object.__setattr__(self, "minimum_boundary_volume", value)


@dataclass(frozen=True, slots=True)
class UnfinishedAuctionStatistics:
    total_auctions: int
    top_count: int
    bottom_count: int


@dataclass(frozen=True, slots=True)
class UnfinishedAuctionResult:
    matrix: "FootprintMatrix"
    auctions: tuple[UnfinishedAuction, ...]
    statistics_value: UnfinishedAuctionStatistics
    configuration: UnfinishedAuctionConfiguration = field(
        default_factory=UnfinishedAuctionConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_boundary_result(self, self.auctions, "unfinished auctions")

    def lookup(self, auction_type: UnfinishedAuctionType) -> UnfinishedAuction | None:
        return next(
            (
                auction
                for auction in self.auctions
                if auction.auction_type == auction_type
            ),
            None,
        )

    def statistics(self) -> UnfinishedAuctionStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class UnfinishedAuctionDetector:
    configuration: UnfinishedAuctionConfiguration = field(
        default_factory=UnfinishedAuctionConfiguration
    )

    def detect(self, matrix: "FootprintMatrix") -> UnfinishedAuctionResult:
        auctions = []
        for auction_type, row in (
            (UnfinishedAuctionType.TOP, 0),
            (UnfinishedAuctionType.BOTTOM, matrix.dimensions_value.rows - 1),
        ):
            bid, ask = _row_bid_ask(matrix.row(row))
            if (
                bid >= self.configuration.minimum_boundary_volume
                and ask >= self.configuration.minimum_boundary_volume
            ):
                auctions.append(
                    UnfinishedAuction(
                        auction_type, row, bid, ask, 1.0, {"source": "footprint_matrix"}
                    )
                )
        result = tuple(auctions)
        stats = UnfinishedAuctionStatistics(
            len(result),
            sum(a.auction_type == UnfinishedAuctionType.TOP for a in result),
            sum(a.auction_type == UnfinishedAuctionType.BOTTOM for a in result),
        )
        return UnfinishedAuctionResult(
            matrix,
            result,
            stats,
            self.configuration,
            {"detector": "UnfinishedAuctionDetector"},
        )


@dataclass(frozen=True, slots=True)
class ExcessConfiguration:
    minimum_opposite_volume: Decimal = Decimal("1")
    strict_mode: bool = True

    def __post_init__(self) -> None:
        value = Decimal(str(self.minimum_opposite_volume))
        if not value.is_finite() or value < 0:
            raise ValueError("excess minimum opposite volume is invalid")
        object.__setattr__(self, "minimum_opposite_volume", value)


@dataclass(frozen=True, slots=True)
class Excess:
    excess_type: ExcessType
    row: int
    bid_volume: Decimal
    ask_volume: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        UnfinishedAuction(
            (
                UnfinishedAuctionType.TOP
                if self.excess_type == ExcessType.EXCESS_HIGH
                else UnfinishedAuctionType.BOTTOM
            ),
            self.row,
            self.bid_volume,
            self.ask_volume,
            self.confidence,
            self.metadata,
        )
        object.__setattr__(self, "bid_volume", Decimal(str(self.bid_volume)))
        object.__setattr__(self, "ask_volume", Decimal(str(self.ask_volume)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ExcessStatistics:
    total_excesses: int
    high_count: int
    low_count: int


@dataclass(frozen=True, slots=True)
class ExcessResult:
    matrix: "FootprintMatrix"
    excesses: tuple[Excess, ...]
    statistics_value: ExcessStatistics
    configuration: ExcessConfiguration = field(default_factory=ExcessConfiguration)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_boundary_result(self, self.excesses, "excess")

    def lookup(self, excess_type: ExcessType) -> Excess | None:
        return next(
            (excess for excess in self.excesses if excess.excess_type == excess_type),
            None,
        )

    def statistics(self) -> ExcessStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class ExcessDetector:
    configuration: ExcessConfiguration = field(default_factory=ExcessConfiguration)

    def detect(self, matrix: "FootprintMatrix") -> ExcessResult:
        excesses = []
        top_bid, top_ask = _row_bid_ask(matrix.row(0))
        if top_ask >= self.configuration.minimum_opposite_volume and top_bid == 0:
            excesses.append(
                Excess(
                    ExcessType.EXCESS_HIGH,
                    0,
                    top_bid,
                    top_ask,
                    1.0,
                    {"source": "footprint_matrix"},
                )
            )
        bottom_bid, bottom_ask = _row_bid_ask(
            matrix.row(matrix.dimensions_value.rows - 1)
        )
        if bottom_bid >= self.configuration.minimum_opposite_volume and bottom_ask == 0:
            excesses.append(
                Excess(
                    ExcessType.EXCESS_LOW,
                    matrix.dimensions_value.rows - 1,
                    bottom_bid,
                    bottom_ask,
                    1.0,
                    {"source": "footprint_matrix"},
                )
            )
        result = tuple(excesses)
        stats = ExcessStatistics(
            len(result),
            sum(e.excess_type == ExcessType.EXCESS_HIGH for e in result),
            sum(e.excess_type == ExcessType.EXCESS_LOW for e in result),
        )
        return ExcessResult(
            matrix, result, stats, self.configuration, {"detector": "ExcessDetector"}
        )


@dataclass(frozen=True, slots=True)
class PoorAuctionConfiguration:
    minimum_boundary_volume: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        value = Decimal(str(self.minimum_boundary_volume))
        if not value.is_finite() or value < 0:
            raise ValueError("poor auction minimum boundary volume is invalid")
        object.__setattr__(self, "minimum_boundary_volume", value)


@dataclass(frozen=True, slots=True)
class PoorAuction:
    auction_type: PoorAuctionType
    row: int
    bid_volume: Decimal
    ask_volume: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("poor auction row must be non-negative")
        bid, ask = Decimal(str(self.bid_volume)), Decimal(str(self.ask_volume))
        if any(not value.is_finite() or value < 0 for value in (bid, ask)):
            raise ValueError("poor auction volumes are invalid")
        _validate_confidence(self.confidence, "poor auction confidence")
        object.__setattr__(self, "bid_volume", bid)
        object.__setattr__(self, "ask_volume", ask)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PoorAuctionStatistics:
    total_auctions: int
    high_count: int
    low_count: int


@dataclass(frozen=True, slots=True)
class PoorAuctionResult:
    matrix: "FootprintMatrix"
    auctions: tuple[PoorAuction, ...]
    statistics_value: PoorAuctionStatistics
    configuration: PoorAuctionConfiguration = field(
        default_factory=PoorAuctionConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_boundary_result(self, self.auctions, "poor auctions")

    def lookup(self, auction_type: PoorAuctionType) -> PoorAuction | None:
        return next((a for a in self.auctions if a.auction_type == auction_type), None)

    def statistics(self) -> PoorAuctionStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class PoorAuctionDetector:
    configuration: PoorAuctionConfiguration = field(
        default_factory=PoorAuctionConfiguration
    )

    def detect(self, matrix: "FootprintMatrix") -> PoorAuctionResult:
        found: list[PoorAuction] = []
        for auction_type, row in (
            (PoorAuctionType.POOR_HIGH, 0),
            (PoorAuctionType.POOR_LOW, matrix.dimensions_value.rows - 1),
        ):
            bid, ask = _row_bid_ask(matrix.row(row))
            if (
                bid >= self.configuration.minimum_boundary_volume
                and ask >= self.configuration.minimum_boundary_volume
            ):
                found.append(
                    PoorAuction(
                        auction_type, row, bid, ask, 1.0, {"source": "footprint_matrix"}
                    )
                )
        result = tuple(found)
        return PoorAuctionResult(
            matrix,
            result,
            PoorAuctionStatistics(
                len(result),
                sum(a.auction_type == PoorAuctionType.POOR_HIGH for a in result),
                sum(a.auction_type == PoorAuctionType.POOR_LOW for a in result),
            ),
            self.configuration,
            {"detector": "PoorAuctionDetector"},
        )


@dataclass(frozen=True, slots=True)
class SinglePrintConfiguration:
    maximum_active_cells_per_row: int = 1
    minimum_cell_volume: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        value = Decimal(str(self.minimum_cell_volume))
        if self.maximum_active_cells_per_row <= 0 or not value.is_finite() or value < 0:
            raise ValueError("single print configuration is invalid")
        object.__setattr__(self, "minimum_cell_volume", value)


@dataclass(frozen=True, slots=True)
class SinglePrint:
    start_row: int
    end_row: int
    price_range: tuple[int, int]
    row_count: int
    matrix: "FootprintMatrix"
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_row < 0 or self.end_row < self.start_row:
            raise ValueError("single print rows must be ordered")
        if self.row_count != self.end_row - self.start_row + 1:
            raise ValueError("single print row count must match range")
        _validate_confidence(self.confidence, "single print confidence")
        object.__setattr__(self, "price_range", tuple(self.price_range))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SinglePrintStatistics:
    total_regions: int
    total_rows: int
    boundary_regions: int


@dataclass(frozen=True, slots=True)
class SinglePrintResult:
    matrix: "FootprintMatrix"
    single_prints: tuple[SinglePrint, ...]
    statistics_value: SinglePrintStatistics
    configuration: SinglePrintConfiguration = field(
        default_factory=SinglePrintConfiguration
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        last = -1
        seen: set[tuple[int, int]] = set()
        for item in self.single_prints:
            if item.matrix != self.matrix:
                raise ValueError("single prints must reference matrix")
            if item.end_row >= self.matrix.dimensions_value.rows:
                raise ValueError("single prints must reference matrix rows")
            key = (item.start_row, item.end_row)
            if key in seen:
                raise ValueError("duplicate single prints are not allowed")
            if item.start_row <= last:
                raise ValueError("single prints must be ordered")
            seen.add(key)
            last = item.end_row
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def lookup(self, row: int) -> SinglePrint | None:
        return next(
            (s for s in self.single_prints if s.start_row <= row <= s.end_row), None
        )

    def statistics(self) -> SinglePrintStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class SinglePrintDetector:
    configuration: SinglePrintConfiguration = field(
        default_factory=SinglePrintConfiguration
    )

    def detect(self, matrix: "FootprintMatrix") -> SinglePrintResult:
        active = [
            sum(
                VolumeClusterAnalyzer._total_volume(c)
                >= self.configuration.minimum_cell_volume
                for c in r.cells
            )
            <= self.configuration.maximum_active_cells_per_row
            for r in matrix.rows
        ]
        regions: list[SinglePrint] = []
        start: int | None = None
        for idx, is_single in enumerate(active + [False]):
            if is_single and start is None:
                start = idx
            if not is_single and start is not None:
                end = idx - 1
                regions.append(
                    SinglePrint(
                        start,
                        end,
                        (start, end),
                        end - start + 1,
                        matrix,
                        1.0,
                        {"source": "footprint_matrix"},
                    )
                )
                start = None
        result = tuple(regions)
        stats = SinglePrintStatistics(
            len(result),
            sum(s.row_count for s in result),
            sum(
                s.start_row == 0 or s.end_row == matrix.dimensions_value.rows - 1
                for s in result
            ),
        )
        return SinglePrintResult(
            matrix,
            result,
            stats,
            self.configuration,
            {"detector": "SinglePrintDetector"},
        )


@dataclass(frozen=True, slots=True)
class NakedPointOfControlConfiguration:
    expiration_period: int = 0

    def __post_init__(self) -> None:
        if self.expiration_period < 0:
            raise ValueError("naked poc expiration period must be non-negative")


@dataclass(frozen=True, slots=True)
class NakedPointOfControl:
    row: int
    total_volume: Decimal
    creation_index: int
    first_revisit_index: int | None
    state: str
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.row < 0 or self.creation_index < 0:
            raise ValueError("naked poc references are invalid")
        if (
            self.first_revisit_index is not None
            and self.first_revisit_index <= self.creation_index
        ):
            raise ValueError("naked poc revisit index must follow creation index")
        if self.state not in {"active", "tested", "expired"}:
            raise ValueError("naked poc state is invalid")
        _validate_confidence(self.confidence, "naked poc confidence")
        object.__setattr__(self, "total_volume", Decimal(str(self.total_volume)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class NakedPointOfControlStatistics:
    total_pocs: int
    active_count: int
    tested_count: int
    expired_count: int
    history_length: int


@dataclass(frozen=True, slots=True)
class NakedPointOfControlResult:
    matrix: "FootprintMatrix"
    naked_pocs: tuple[NakedPointOfControl, ...]
    statistics_value: NakedPointOfControlStatistics
    configuration: NakedPointOfControlConfiguration = field(
        default_factory=NakedPointOfControlConfiguration
    )
    history: tuple[PointOfControlResult, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.history and self.history[-1].matrix != self.matrix:
            raise ValueError("naked pocs must reference graph matrix")
        keys = tuple((p.row, p.creation_index) for p in self.naked_pocs)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate naked pocs are not allowed")
        if (
            tuple(sorted(self.naked_pocs, key=lambda p: (p.creation_index, p.row)))
            != self.naked_pocs
        ):
            raise ValueError("naked pocs must be ordered")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def lookup(self, row: int) -> NakedPointOfControl | None:
        return next((p for p in self.naked_pocs if p.row == row), None)

    def statistics(self) -> NakedPointOfControlStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class NakedPointOfControlTracker:
    configuration: NakedPointOfControlConfiguration = field(
        default_factory=NakedPointOfControlConfiguration
    )

    def track(
        self, history: Sequence[PointOfControlResult]
    ) -> NakedPointOfControlResult:
        if not history:
            raise ValueError("naked poc history is required")
        rows = history[-1].matrix.dimensions_value.rows
        pocs: list[NakedPointOfControl] = []
        for i, poc_result in enumerate(history):
            if poc_result.poc.row >= rows:
                raise ValueError("naked poc history references incompatible matrices")
            revisit = next(
                (
                    j
                    for j, later in enumerate(history[i + 1 :], i + 1)
                    if later.poc.row == poc_result.poc.row
                ),
                None,
            )
            if revisit is not None:
                state = "tested"
            elif (
                self.configuration.expiration_period
                and len(history) - 1 - i >= self.configuration.expiration_period
            ):
                state = "expired"
            else:
                state = "active"
            pocs.append(
                NakedPointOfControl(
                    poc_result.poc.row,
                    poc_result.poc.total_volume,
                    i,
                    revisit,
                    state,
                    1.0,
                    {"source": "point_of_control"},
                )
            )
        tracked = tuple(pocs)
        stats = NakedPointOfControlStatistics(
            len(tracked),
            sum(p.state == "active" for p in tracked),
            sum(p.state == "tested" for p in tracked),
            sum(p.state == "expired" for p in tracked),
            len(history),
        )
        return NakedPointOfControlResult(
            history[-1].matrix,
            tracked,
            stats,
            self.configuration,
            tuple(history),
            {"tracker": "NakedPointOfControlTracker"},
        )


def _row_bid_ask(row: "MatrixRow") -> tuple[Decimal, Decimal]:
    bid = ask = Decimal("0")
    for cell in row.cells:
        bid_value, ask_value = cell.interpretation.bid(), cell.interpretation.ask()
        bid += (
            Decimal("0")
            if bid_value is None
            else Decimal(str(bid_value.numeric_value.value))
        )
        ask += (
            Decimal("0")
            if ask_value is None
            else Decimal(str(ask_value.numeric_value.value))
        )
    return bid, ask


def _validate_boundary_result(result: Any, entries: tuple[Any, ...], name: str) -> None:
    rows = result.matrix.dimensions_value.rows
    if any(entry.row >= rows for entry in entries):
        raise ValueError(f"{name} must reference matrix rows")
    keys = tuple(_boundary_key(entry) for entry in entries)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate {name} are not allowed")
    if (
        tuple(sorted(entries, key=lambda entry: _boundary_order(_boundary_key(entry))))
        != entries
    ):
        raise ValueError(f"{name} must be ordered")
    object.__setattr__(result, "metadata", MappingProxyType(dict(result.metadata)))


def _matrix_row_volumes(matrix: "FootprintMatrix") -> tuple[Decimal, ...]:
    return tuple(
        sum(
            (VolumeClusterAnalyzer._total_volume(cell) for cell in row.cells),
            Decimal("0"),
        )
        for row in matrix.rows
    )


def _volume_percentile(volume: Decimal, sorted_volumes: tuple[Decimal, ...]) -> Decimal:
    if len(sorted_volumes) <= 1:
        return Decimal("100")
    less = sum(candidate < volume for candidate in sorted_volumes)
    return Decimal(less) / Decimal(len(sorted_volumes) - 1) * Decimal("100")


@dataclass(frozen=True, slots=True)
class DeltaConfiguration:
    """Immutable settings for deterministic footprint delta analysis."""

    include_empty_cells: bool = True


@dataclass(frozen=True, slots=True)
class CellDelta:
    """Deterministic bid/ask delta for one footprint matrix cell."""

    cell_id: str
    row: int
    column: int
    bid: Decimal
    ask: Decimal
    delta: Decimal
    absolute_delta: Decimal
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("delta cell id is required")
        if self.row < 0 or self.column < 0:
            raise ValueError("delta cell position must be non-negative")
        bid = Decimal(str(self.bid))
        ask = Decimal(str(self.ask))
        delta = Decimal(str(self.delta))
        absolute = Decimal(str(self.absolute_delta))
        if not bid.is_finite() or bid < 0:
            raise ValueError("delta bid must be non-negative and finite")
        if not ask.is_finite() or ask < 0:
            raise ValueError("delta ask must be non-negative and finite")
        if not delta.is_finite() or not absolute.is_finite():
            raise ValueError("delta values must be finite")
        if delta != ask - bid:
            raise ValueError("cell delta must equal ask minus bid")
        if absolute != abs(delta):
            raise ValueError("absolute cell delta must equal abs(delta)")
        _validate_confidence(self.confidence, "delta confidence")
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "absolute_delta", absolute)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def delta_type(self) -> DeltaType:
        if self.delta > 0:
            return DeltaType.POSITIVE
        if self.delta < 0:
            return DeltaType.NEGATIVE
        return DeltaType.ZERO


@dataclass(frozen=True, slots=True)
class RowDelta:
    """Aggregate deterministic delta for one matrix row."""

    row_index: int
    total_bid: Decimal
    total_ask: Decimal
    row_delta: Decimal
    absolute_delta: Decimal
    cell_count: int

    def __post_init__(self) -> None:
        if self.row_index < 0 or self.cell_count <= 0:
            raise ValueError("row delta indexes and counts must be valid")
        bid = Decimal(str(self.total_bid))
        ask = Decimal(str(self.total_ask))
        delta = Decimal(str(self.row_delta))
        absolute = Decimal(str(self.absolute_delta))
        if any(not value.is_finite() for value in (bid, ask, delta, absolute)):
            raise ValueError("row delta values must be finite")
        if bid < 0 or ask < 0:
            raise ValueError("row delta volumes must be non-negative")
        if delta != ask - bid or absolute != abs(delta):
            raise ValueError("row delta aggregates are inconsistent")
        object.__setattr__(self, "total_bid", bid)
        object.__setattr__(self, "total_ask", ask)
        object.__setattr__(self, "row_delta", delta)
        object.__setattr__(self, "absolute_delta", absolute)


@dataclass(frozen=True, slots=True)
class FootprintDelta:
    """Aggregate deterministic delta for the entire footprint matrix."""

    total_bid: Decimal
    total_ask: Decimal
    net_delta: Decimal
    absolute_delta: Decimal
    maximum_positive_delta: Decimal
    maximum_negative_delta: Decimal
    average_cell_delta: Decimal

    def __post_init__(self) -> None:
        values = tuple(
            Decimal(str(v))
            for v in (
                self.total_bid,
                self.total_ask,
                self.net_delta,
                self.absolute_delta,
                self.maximum_positive_delta,
                self.maximum_negative_delta,
                self.average_cell_delta,
            )
        )
        if any(not value.is_finite() for value in values):
            raise ValueError("footprint delta values must be finite")
        bid, ask, net, absolute, max_pos, max_neg, average = values
        if bid < 0 or ask < 0:
            raise ValueError("footprint delta volumes must be non-negative")
        if net != ask - bid or absolute != abs(net):
            raise ValueError("footprint delta aggregates are inconsistent")
        if max_pos < 0 or max_neg > 0:
            raise ValueError("maximum deltas have invalid signs")
        for name, value in zip(self.__slots__, values, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class DeltaStatistics:
    """Aggregate counts and basic values for deterministic delta analysis."""

    rows: int
    cells: int
    positive_cells: int
    negative_cells: int
    zero_cells: int
    maximum_delta: Decimal
    minimum_delta: Decimal
    average_delta: Decimal

    def __post_init__(self) -> None:
        if any(
            v < 0
            for v in (
                self.rows,
                self.cells,
                self.positive_cells,
                self.negative_cells,
                self.zero_cells,
            )
        ):
            raise ValueError("delta statistics counts cannot be negative")
        if self.positive_cells + self.negative_cells + self.zero_cells != self.cells:
            raise ValueError("delta statistics must account for all cells")
        maximum = Decimal(str(self.maximum_delta))
        minimum = Decimal(str(self.minimum_delta))
        average = Decimal(str(self.average_delta))
        if any(not value.is_finite() for value in (maximum, minimum, average)):
            raise ValueError("delta statistics values must be finite")
        if maximum < minimum:
            raise ValueError("maximum delta must be greater than minimum delta")
        object.__setattr__(self, "maximum_delta", maximum)
        object.__setattr__(self, "minimum_delta", minimum)
        object.__setattr__(self, "average_delta", average)


@dataclass(frozen=True, slots=True)
class DeltaResult:
    """Immutable result set for deterministic footprint delta analysis."""

    matrix: "FootprintMatrix"
    cells: tuple[CellDelta, ...]
    rows: tuple[RowDelta, ...]
    footprint: FootprintDelta
    statistics_value: DeltaStatistics
    configuration: DeltaConfiguration = field(default_factory=DeltaConfiguration)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered_cells = tuple(
            sorted(self.cells, key=lambda c: (c.row, c.column, c.cell_id))
        )
        if self.cells != ordered_cells:
            raise ValueError("delta cells must be ordered")
        ordered_rows = tuple(sorted(self.rows, key=lambda r: r.row_index))
        if self.rows != ordered_rows:
            raise ValueError("delta rows must be ordered")
        if len({c.cell_id for c in self.cells}) != len(self.cells):
            raise ValueError("duplicate delta cells are not allowed")
        if len({r.row_index for r in self.rows}) != len(self.rows):
            raise ValueError("duplicate delta rows are not allowed")
        matrix_ids = {cell.cell_id for cell in self.matrix.cells}
        if {cell.cell_id for cell in self.cells} != matrix_ids:
            raise ValueError("delta cells must reference matrix cells")
        if tuple(r.row_index for r in self.rows) != tuple(
            range(self.matrix.dimensions_value.rows)
        ):
            raise ValueError("delta rows must reference matrix rows")
        object.__setattr__(self, "cells", ordered_cells)
        object.__setattr__(self, "rows", ordered_rows)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def cell_delta(self, cell_id: str) -> CellDelta | None:
        return next((cell for cell in self.cells if cell.cell_id == cell_id), None)

    def row_delta(self, row_index: int) -> RowDelta | None:
        return next((row for row in self.rows if row.row_index == row_index), None)

    def positive_cells(self) -> tuple[CellDelta, ...]:
        return tuple(cell for cell in self.cells if cell.delta > 0)

    def negative_cells(self) -> tuple[CellDelta, ...]:
        return tuple(cell for cell in self.cells if cell.delta < 0)

    def zero_cells(self) -> tuple[CellDelta, ...]:
        return tuple(cell for cell in self.cells if cell.delta == 0)

    def statistics(self) -> DeltaStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class FootprintDeltaAnalyzer:
    """Compute deterministic delta values from an immutable footprint matrix."""

    configuration: DeltaConfiguration = field(default_factory=DeltaConfiguration)

    def analyze(self, matrix: "FootprintMatrix") -> DeltaResult:
        cells = tuple(self._cell_delta(cell) for cell in matrix.cells)
        rows = tuple(self._row_delta(row, cells) for row in matrix.rows)
        total_bid = sum((cell.bid for cell in cells), Decimal("0"))
        total_ask = sum((cell.ask for cell in cells), Decimal("0"))
        deltas = tuple(cell.delta for cell in cells)
        net = total_ask - total_bid
        count = Decimal(len(cells))
        footprint = FootprintDelta(
            total_bid,
            total_ask,
            net,
            abs(net),
            max((d for d in deltas if d > 0), default=Decimal("0")),
            min((d for d in deltas if d < 0), default=Decimal("0")),
            Decimal("0") if not cells else net / count,
        )
        stats = DeltaStatistics(
            matrix.dimensions_value.rows,
            len(cells),
            sum(d > 0 for d in deltas),
            sum(d < 0 for d in deltas),
            sum(d == 0 for d in deltas),
            max(deltas, default=Decimal("0")),
            min(deltas, default=Decimal("0")),
            Decimal("0") if not cells else net / count,
        )
        return DeltaResult(
            matrix,
            cells,
            rows,
            footprint,
            stats,
            self.configuration,
            {"detector": "FootprintDeltaAnalyzer"},
        )

    @staticmethod
    def _decimal(value: FootprintValue | None) -> Decimal:
        if value is None:
            return Decimal("0")
        return Decimal(str(value.numeric_value.value))

    def _cell_delta(self, cell: "MatrixCell") -> CellDelta:
        bid = self._decimal(cell.interpretation.bid())
        ask = self._decimal(cell.interpretation.ask())
        delta = ask - bid
        return CellDelta(
            cell.cell_id,
            cell.row_index,
            cell.column_index,
            bid,
            ask,
            delta,
            abs(delta),
            1.0,
            {"source": "footprint_matrix"},
        )

    @staticmethod
    def _row_delta(row: "MatrixRow", cells: Sequence[CellDelta]) -> RowDelta:
        row_cells = tuple(cell for cell in cells if cell.row == row.index)
        bid = sum((cell.bid for cell in row_cells), Decimal("0"))
        ask = sum((cell.ask for cell in row_cells), Decimal("0"))
        delta = ask - bid
        return RowDelta(row.index, bid, ask, delta, abs(delta), len(row_cells))


@dataclass(frozen=True, slots=True)
class FootprintMatrix:
    """Canonical immutable two-dimensional footprint-cell representation."""

    grid_id: str
    rows: tuple[MatrixRow, ...]
    dimensions_value: MatrixDimensions
    statistics_value: MatrixStatistics

    def __post_init__(self) -> None:
        if not self.grid_id.strip():
            raise ValueError("matrix grid id is required")
        if len(self.rows) != self.dimensions_value.rows:
            raise ValueError("matrix row count must match dimensions")
        ordered_rows = tuple(sorted(self.rows, key=lambda row: row.index))
        if self.rows != ordered_rows:
            raise ValueError("matrix rows must be row ordered")
        if tuple(row.index for row in self.rows) != tuple(
            range(self.dimensions_value.rows)
        ):
            raise ValueError("matrix rows must be continuous")
        cells = tuple(cell for row in self.rows for cell in row.cells)
        if any(len(row.cells) != self.dimensions_value.columns for row in self.rows):
            raise ValueError("matrix rows must match column dimensions")
        positions = {(cell.row_index, cell.column_index) for cell in cells}
        if len(positions) != len(cells):
            raise ValueError("duplicate matrix positions are not allowed")
        expected_positions = {
            (row, column)
            for row in range(self.dimensions_value.rows)
            for column in range(self.dimensions_value.columns)
        }
        if positions != expected_positions:
            raise ValueError("matrix coordinates must be continuous")
        ids = {cell.cell_id for cell in cells}
        if len(ids) != len(cells):
            raise ValueError("duplicate matrix cell ids are not allowed")
        if self.statistics_value.total_cells != len(cells):
            raise ValueError("matrix statistics must match cell count")
        object.__setattr__(self, "rows", tuple(self.rows))

    @property
    def cells(self) -> tuple[MatrixCell, ...]:
        return tuple(cell for row in self.rows for cell in row.cells)

    def cell(self, row: int, column: int) -> MatrixCell:
        if row < 0 or column < 0:
            raise KeyError(f"matrix cell not found: {row},{column}")
        return self.rows[row].cells[column]

    def row(self, index: int) -> MatrixRow:
        return self.rows[index]

    def column(self, index: int) -> tuple[MatrixCell, ...]:
        return tuple(row.cells[index] for row in self.rows)

    def above(self, cell: MatrixCell) -> MatrixCell | None:
        return (
            None
            if cell.row_index == 0
            else self.cell(cell.row_index - 1, cell.column_index)
        )

    def below(self, cell: MatrixCell) -> MatrixCell | None:
        if cell.row_index + 1 >= self.dimensions_value.rows:
            return None
        return self.cell(cell.row_index + 1, cell.column_index)

    def left(self, cell: MatrixCell) -> MatrixCell | None:
        return (
            None
            if cell.column_index == 0
            else self.cell(cell.row_index, cell.column_index - 1)
        )

    def right(self, cell: MatrixCell) -> MatrixCell | None:
        if cell.column_index + 1 >= self.dimensions_value.columns:
            return None
        return self.cell(cell.row_index, cell.column_index + 1)

    def neighbors(self, cell: MatrixCell) -> tuple[MatrixCell, ...]:
        return tuple(
            candidate
            for candidate in (
                self.above(cell),
                self.below(cell),
                self.left(cell),
                self.right(cell),
            )
            if candidate is not None
        )

    def diagonal_neighbors(self, cell: MatrixCell) -> tuple[MatrixCell, ...]:
        candidates = (
            (cell.row_index - 1, cell.column_index - 1),
            (cell.row_index - 1, cell.column_index + 1),
            (cell.row_index + 1, cell.column_index - 1),
            (cell.row_index + 1, cell.column_index + 1),
        )
        return tuple(
            self.cell(row, column)
            for row, column in candidates
            if 0 <= row < self.dimensions_value.rows
            and 0 <= column < self.dimensions_value.columns
        )

    def dimensions(self) -> MatrixDimensions:
        return self.dimensions_value

    def statistics(self) -> MatrixStatistics:
        return self.statistics_value


@dataclass(frozen=True, slots=True)
class FootprintMatrixBuilder:
    """Build and validate the canonical immutable footprint matrix."""

    def build(
        self,
        detection_graph: DetectionGraph,
        coordinate_mapper: CoordinateMapper,
        interpretation: FootprintInterpretation,
    ) -> FootprintMatrix:
        cells = detection_graph.footprint_cells
        grid = coordinate_mapper.map_cells(cells)
        if grid.grid_id != interpretation.grid_id:
            raise ValueError("matrix grid id must match interpretation")
        by_coordinate = {(c.row_index, c.column_index): c for c in grid.cells}
        expected = {
            (row, column)
            for row in range(grid.row_count)
            for column in range(grid.column_count)
        }
        if set(by_coordinate) != expected:
            raise ValueError("matrix coordinates must not have gaps")
        by_cell_id = {cell.cell_id: cell for cell in grid.cells}
        interpreted = {
            cell.cell_reference.coordinate.cell_id: cell
            for cell in interpretation.ordered_cells
        }
        if set(interpreted) != set(by_cell_id):
            raise ValueError("every coordinate must have one interpretation")
        classifications = {c.cell_id: c for c in detection_graph.cell_classifications}
        originals = {str(obj.metadata.get("cell_id", "")): obj for obj in cells}
        parsed = {
            cell_id: tuple(
                result
                for result in detection_graph.parsed_values
                if result.parsed_value.cell_id == cell_id
            )
            for cell_id in by_cell_id
        }
        matrix_cells = tuple(
            MatrixCell(
                MatrixPosition(
                    coordinate.row_index,
                    coordinate.column_index,
                    coordinate.cell_id,
                    coordinate,
                ),
                interpreted[coordinate.cell_id],
                parsed[coordinate.cell_id],
                classifications[coordinate.cell_id],
                originals[coordinate.cell_id],
            )
            for coordinate in grid.cells
        )
        rows = tuple(
            MatrixRow(
                row,
                tuple(cell for cell in matrix_cells if cell.row_index == row),
            )
            for row in range(grid.row_count)
        )
        statistics = self._statistics(grid.row_count, grid.column_count, matrix_cells)
        return FootprintMatrix(
            grid.grid_id,
            rows,
            MatrixDimensions(grid.row_count, grid.column_count),
            statistics,
        )

    @staticmethod
    def _statistics(
        rows: int, columns: int, cells: Sequence[MatrixCell]
    ) -> MatrixStatistics:
        total = rows * columns
        return MatrixStatistics(
            rows=rows,
            columns=columns,
            total_cells=total,
            interpreted_cells=sum(not cell.interpretation.is_empty() for cell in cells),
            empty_cells=sum(cell.interpretation.is_empty() for cell in cells),
            missing_cells=total - len(cells),
            bid_cells=sum(cell.interpretation.bid() is not None for cell in cells),
            ask_cells=sum(cell.interpretation.ask() is not None for cell in cells),
            delta_cells=sum(cell.interpretation.delta() is not None for cell in cells),
            unknown_cells=sum(
                any(
                    warning.code in ("unknown", "missing_values")
                    for warning in cell.interpretation.warnings()
                )
                for cell in cells
            ),
        )


@dataclass(frozen=True, slots=True)
class InterpretationResult:
    """Result wrapper for footprint semantic interpretation."""

    interpretation: FootprintInterpretation
    warnings: tuple[InterpretationWarning, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))


@runtime_checkable
class SemanticMapper(Protocol):
    """Map classified cell roles and parsed values to footprint semantics."""

    def map(
        self, role: CellSemanticRole, parsed_value: ParsedValue
    ) -> FootprintSemanticType:
        """Return the footprint semantic type for one parsed value."""


@dataclass(frozen=True, slots=True)
class LayoutSemanticMapper:
    """Generic mapper driven by cell layout roles rather than vendor layouts."""

    role_mapping: Mapping[CellSemanticRole, FootprintSemanticType] = field(
        default_factory=lambda: MappingProxyType(
            {
                CellSemanticRole.BID_REGION: FootprintSemanticType.BID_VOLUME,
                CellSemanticRole.ASK_REGION: FootprintSemanticType.ASK_VOLUME,
                CellSemanticRole.CENTER_REGION: FootprintSemanticType.DELTA,
                CellSemanticRole.DELTA_REGION: FootprintSemanticType.DELTA,
                CellSemanticRole.EMPTY: FootprintSemanticType.EMPTY,
                CellSemanticRole.UNKNOWN: FootprintSemanticType.UNKNOWN,
                CellSemanticRole.BACKGROUND: FootprintSemanticType.UNKNOWN,
            }
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "role_mapping", MappingProxyType(dict(self.role_mapping))
        )

    def map(
        self, role: CellSemanticRole, parsed_value: ParsedValue
    ) -> FootprintSemanticType:
        if role != parsed_value.semantic_role:
            raise ValueError("invalid semantic mapping")
        return self.role_mapping.get(role, FootprintSemanticType.UNKNOWN)


@runtime_checkable
class FootprintInterpreter(Protocol):
    """Interpret parsed numeric values for one classified footprint cell."""

    def interpret(
        self, classification: CellClassification, parsed_values: Sequence[ParsedValue]
    ) -> FootprintCellData:
        """Return interpreted cell data."""


@dataclass(frozen=True, slots=True)
class DefaultFootprintInterpreter:
    """Deterministic semantic interpreter for one footprint cell."""

    mapper: SemanticMapper = field(default_factory=LayoutSemanticMapper)

    def interpret(
        self, classification: CellClassification, parsed_values: Sequence[ParsedValue]
    ) -> FootprintCellData:
        cell_id = classification.cell_id
        if classification.cell_reference is None:
            raise ValueError("missing parent cell")
        by_type: dict[FootprintSemanticType, FootprintValue] = {}
        warnings: list[InterpretationWarning] = []
        for parsed in parsed_values:
            if parsed.cell_id != cell_id:
                raise ValueError("parsed value missing parent cell")
            semantic_type = self.mapper.map(parsed.semantic_role, parsed)
            if semantic_type == FootprintSemanticType.INVALID:
                raise ValueError("invalid semantic mapping")
            if semantic_type in (
                FootprintSemanticType.UNKNOWN,
                FootprintSemanticType.EMPTY,
            ):
                warnings.append(
                    InterpretationWarning(
                        semantic_type.value.lower(),
                        f"{semantic_type.value} semantic role",
                        cell_id,
                    )
                )
                continue
            if semantic_type in by_type:
                raise ValueError(f"duplicate {semantic_type.value.lower()} values")
            if parsed.parsed_number is None:
                warnings.append(
                    InterpretationWarning(
                        "missing_numeric_value",
                        "parsed value does not contain a numeric value",
                        cell_id,
                    )
                )
                continue
            by_type[semantic_type] = FootprintValue(
                parsed.parsed_number,
                semantic_type,
                parsed.confidence,
                parsed.source_region,
                cell_id,
            )
        missing = tuple(
            t
            for t in (
                FootprintSemanticType.BID_VOLUME,
                FootprintSemanticType.ASK_VOLUME,
                FootprintSemanticType.DELTA,
            )
            if t not in by_type
        )
        if missing:
            warnings.append(
                InterpretationWarning(
                    "missing_values",
                    "one or more footprint values are missing",
                    cell_id,
                )
            )
        return FootprintCellData(
            classification.cell_reference,
            by_type.get(FootprintSemanticType.BID_VOLUME),
            by_type.get(FootprintSemanticType.ASK_VOLUME),
            by_type.get(FootprintSemanticType.DELTA),
            by_type.get(FootprintSemanticType.TOTAL_VOLUME),
            missing,
            tuple(warnings),
            {
                "cell_id": cell_id,
                "row": classification.row,
                "column": classification.column,
            },
        )


@runtime_checkable
class InterpretationPipeline(Protocol):
    """Run classification, parsed values, semantic mapping, and grid interpretation."""

    def run(
        self,
        cell_classifications: Sequence[CellClassification],
        parsed_values: Sequence[ParsedValue],
    ) -> FootprintInterpretation:
        """Return grid-level interpretation."""


@dataclass(frozen=True, slots=True)
class SequentialInterpretationPipeline:
    """Deterministic footprint semantic interpretation pipeline."""

    interpreter: FootprintInterpreter = field(
        default_factory=DefaultFootprintInterpreter
    )

    def run(
        self,
        cell_classifications: Sequence[CellClassification],
        parsed_values: Sequence[ParsedValue],
    ) -> FootprintInterpretation:
        if not cell_classifications:
            raise ValueError("at least one cell classification is required")
        ordered_classifications = tuple(
            sorted(cell_classifications, key=lambda c: (c.row, c.column, c.cell_id))
        )
        grid_id = ordered_classifications[0].cell_reference.coordinate.grid.grid_id
        by_cell: dict[str, list[ParsedValue]] = {
            c.cell_id: [] for c in ordered_classifications
        }
        for parsed in parsed_values:
            if parsed.cell_id not in by_cell:
                raise ValueError("parsed value missing parent cell")
            by_cell[parsed.cell_id].append(parsed)
        cells = tuple(
            self.interpreter.interpret(c, tuple(by_cell[c.cell_id]))
            for c in ordered_classifications
        )
        warnings = tuple(w for cell in cells for w in cell.warnings())
        values = tuple(
            v
            for cell in cells
            for v in (cell.bid(), cell.ask(), cell.delta(), cell.total_volume())
            if v is not None
        )
        confidence = sum(v.confidence for v in values) / len(values) if values else 1.0
        return FootprintInterpretation(
            grid_id, cells, confidence, warnings, {"cell_count": len(cells)}
        )


@runtime_checkable
class OCRProvider(Protocol):
    """Abstract provider interface only; implementations live outside Genesis."""

    @property
    def name(self) -> str:
        """Return provider name."""

    def recognize(self, request: OCRRequest) -> OCRResult:
        """Return raw OCR output for one request."""


@runtime_checkable
class OCREngine(Protocol):
    """OCR engine contract accepting requests and returning raw results."""

    def run(self, request: OCRRequest) -> OCRResult:
        """Run OCR over one predefined semantic region."""


@dataclass(frozen=True, slots=True)
class DummyOCREngine:
    """Deterministic mock OCR engine for architecture tests; performs no OCR."""

    engine_name: str = "dummy-ocr-engine"

    def run(self, request: OCRRequest) -> OCRResult:
        text = f"{request.cell_id}:{request.semantic_role.value}"
        word = OCRWord(text, 1.0, request.bounding_box)
        line = OCRLine((word,), request.bounding_box, 1.0)
        metadata = OCRMetadata(self.engine_name, "dummy", {"mock": True})
        return OCRResult(
            request.frame_id,
            request.cell_id,
            request.semantic_role,
            text,
            1.0,
            (request.bounding_box,),
            (word,),
            (line,),
            metadata,
        )


@runtime_checkable
class OCRPipeline(Protocol):
    """Contract for executing OCR after cell classification."""

    def run(
        self,
        processed_frame: "ProcessedFrame",
        cell_classifications: Sequence[CellClassification],
    ) -> tuple[OCRResult, ...]:
        """Return raw OCR results for classified cell regions."""


@dataclass(frozen=True, slots=True)
class SequentialOCRPipeline:
    """Deterministically builds OCR requests from classified cell regions."""

    engine: OCREngine = field(default_factory=DummyOCREngine)
    configuration: OCRConfiguration = field(default_factory=OCRConfiguration)

    def run(
        self,
        processed_frame: "ProcessedFrame",
        cell_classifications: Sequence[CellClassification],
    ) -> tuple[OCRResult, ...]:
        if not cell_classifications:
            return ()
        results = []
        for classification in sorted(
            cell_classifications, key=lambda item: (item.row, item.column, item.cell_id)
        ):
            for region in classification.semantic_regions:
                request = OCRRequest(
                    frame_id=classification.cell_reference.frame_id,
                    cell_id=classification.cell_id,
                    bounding_box=region.bounds,
                    semantic_role=region.semantic_role,
                    image_region=processed_frame.source_frame,
                    configuration=self.configuration,
                )
                results.append(self.engine.run(request))
        return tuple(results)


class CoordinateMapper:
    """Deterministically map footprint-cell detections to logical coordinates."""

    def map_cells(self, cells: Sequence[DetectedObject]) -> GridCoordinateSystem:
        if not cells:
            raise ValueError("at least one footprint cell is required")
        ordered = tuple(sorted(cells, key=lambda obj: (obj.bounds.y, obj.bounds.x)))
        if any(obj.object_type != ObjectType.FOOTPRINT_CELL for obj in ordered):
            raise ValueError("only footprint cells can be mapped")
        frame_ids = {obj.frame_id for obj in ordered}
        if len(frame_ids) != 1:
            raise ValueError("all footprint cells must belong to one frame")
        grid_ids = {str(obj.metadata.get("grid_id", "")) for obj in ordered}
        if len(grid_ids) != 1 or not next(iter(grid_ids)).strip():
            raise ValueError("all footprint cells must reference one grid_id")
        duplicate_positions = {(obj.bounds.x, obj.bounds.y) for obj in ordered}
        if len(duplicate_positions) != len(ordered):
            raise ValueError("duplicate cell positions are not allowed")
        xs = tuple(sorted({obj.bounds.x for obj in ordered}))
        ys = tuple(sorted({obj.bounds.y for obj in ordered}))
        if len(xs) * len(ys) != len(ordered):
            raise ValueError("missing rows or columns in footprint cells")
        widths = tuple(obj.bounds.width for obj in ordered)
        heights = tuple(obj.bounds.height for obj in ordered)
        if max(widths) / min(widths) > 1.25 or max(heights) / min(heights) > 1.25:
            raise ValueError("cell dimensions must be consistent")
        by_position = {(obj.bounds.y, obj.bounds.x): obj for obj in ordered}
        cell_width = round(sum(widths) / len(widths))
        cell_height = round(sum(heights) / len(heights))
        bounds = BoundingBox(
            xs[0],
            ys[0],
            xs[-1] - xs[0] + cell_width,
            ys[-1] - ys[0] + cell_height,
        )
        grid = object.__new__(GridCoordinateSystem)
        object.__setattr__(grid, "grid_id", next(iter(grid_ids)))
        object.__setattr__(grid, "row_count", len(ys))
        object.__setattr__(grid, "column_count", len(xs))
        object.__setattr__(grid, "cell_width", cell_width)
        object.__setattr__(grid, "cell_height", cell_height)
        object.__setattr__(grid, "bounds", bounds)
        coords = tuple(
            CellCoordinate(row, column, f"{grid.grid_id}:cell:{row}:{column}", grid)
            for row, y in enumerate(ys)
            for column, x in enumerate(xs)
            if (y, x) in by_position
        )
        object.__setattr__(grid, "cells", coords)
        GridCoordinateSystem.__post_init__(grid)
        return grid

    def references(self, cells: Sequence[DetectedObject]) -> tuple[CellReference, ...]:
        grid = self.map_cells(cells)
        by_coordinate = {
            (int(obj.metadata["row_index"]), int(obj.metadata["column_index"])): obj
            for obj in cells
        }
        return tuple(
            CellReference(
                coordinate=coordinate,
                bounds=by_coordinate[
                    (coordinate.row_index, coordinate.column_index)
                ].bounds,
                detected_object=by_coordinate[
                    (coordinate.row_index, coordinate.column_index)
                ],
                frame_id=by_coordinate[
                    (coordinate.row_index, coordinate.column_index)
                ].frame_id,
            )
            for coordinate in grid.cells
        )


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
                    else (
                        1
                        if detector.name == TimeAxisDetector.name
                        else (
                            2
                            if detector.name == FootprintGridDetector.name
                            else 3 if detector.name == FootprintCellDetector.name else 4
                        )
                    )
                ),
            )
        )
        detector_objects = tuple(
            detected
            for detector in ordered_detectors
            for result in (detector.detect(context),)
            for detected in result.detected_objects
        )
        if any(
            detector.name == FootprintGridDetector.name
            for detector in ordered_detectors
        ):
            chart_object = DetectedObject(
                object_id=ObjectId(
                    f"{context.processed_frame.source_frame.frame_id}:chart"
                ),
                bounds=context.workspace_layout.chart_region.bounds,
                confidence=DetectionConfidence(
                    context.workspace_layout.chart_region.confidence
                ),
                object_type=ObjectType.CHART,
                frame_id=context.processed_frame.source_frame.frame_id,
                source=DetectionSource("workspace-layout"),
                metadata={"role": "chart"},
            )
            detected = (chart_object,) + detector_objects
        else:
            detected = detector_objects
        footprint_cells = tuple(
            obj for obj in detected if obj.object_type == ObjectType.FOOTPRINT_CELL
        )
        coordinate_system = (
            CoordinateMapper().map_cells(footprint_cells) if footprint_cells else None
        )
        cell_classifications = (
            tuple(
                CellLayoutAnalyzer().classify(reference)
                for reference in CoordinateMapper().references(footprint_cells)
            )
            if footprint_cells
            else ()
        )
        ocr_results = SequentialOCRPipeline().run(
            context.processed_frame, cell_classifications
        )
        parsed_values = tuple(
            DeterministicOCRPostProcessor().process(result) for result in ocr_results
        )
        footprint_interpretation = (
            SequentialInterpretationPipeline().run(
                cell_classifications,
                tuple(result.parsed_value for result in parsed_values),
            )
            if cell_classifications
            else None
        )
        graph = DetectionGraph(
            frame_id=context.processed_frame.source_frame.frame_id,
            objects=detected,
            grid_coordinate_system=coordinate_system,
            cell_classifications=cell_classifications,
            ocr_results=ocr_results,
            footprint_interpretation=footprint_interpretation,
            parsed_values=parsed_values,
        )
        footprint_matrix = (
            FootprintMatrixBuilder().build(
                graph,
                CoordinateMapper(),
                footprint_interpretation,
            )
            if footprint_interpretation is not None
            else None
        )
        footprint_imbalances = (
            FootprintImbalanceDetector().detect(footprint_matrix)
            if footprint_matrix is not None
            else None
        )
        stacked_imbalances = (
            StackedImbalanceDetector().detect(footprint_matrix, footprint_imbalances)
            if footprint_matrix is not None and footprint_imbalances is not None
            else None
        )
        absorption = (
            FootprintAbsorptionDetector().detect(footprint_matrix, footprint_imbalances)
            if footprint_matrix is not None and footprint_imbalances is not None
            else None
        )
        footprint_delta = (
            FootprintDeltaAnalyzer().analyze(footprint_matrix)
            if footprint_matrix is not None and absorption is not None
            else None
        )
        volume_clusters = (
            VolumeClusterAnalyzer().analyze(footprint_matrix)
            if footprint_matrix is not None and footprint_delta is not None
            else None
        )
        point_of_control = (
            PointOfControlAnalyzer().analyze(footprint_matrix)
            if footprint_matrix is not None and volume_clusters is not None
            else None
        )
        high_volume_nodes = (
            HighVolumeNodeAnalyzer().analyze(footprint_matrix)
            if footprint_matrix is not None and point_of_control is not None
            else None
        )
        low_volume_nodes = (
            LowVolumeNodeAnalyzer().analyze(footprint_matrix)
            if footprint_matrix is not None and high_volume_nodes is not None
            else None
        )
        value_area = (
            ValueAreaAnalyzer().analyze(footprint_matrix, point_of_control)
            if footprint_matrix is not None
            and low_volume_nodes is not None
            and point_of_control is not None
            else None
        )
        developing_poc = (
            DevelopingPointOfControlAnalyzer().analyze(footprint_matrix)
            if footprint_matrix is not None and value_area is not None
            else None
        )
        developing_value_area = (
            DevelopingValueAreaAnalyzer().analyze(footprint_matrix)
            if footprint_matrix is not None and developing_poc is not None
            else None
        )
        unfinished_auctions = (
            UnfinishedAuctionDetector().detect(footprint_matrix)
            if footprint_matrix is not None and developing_value_area is not None
            else None
        )
        excess = (
            ExcessDetector().detect(footprint_matrix)
            if footprint_matrix is not None and unfinished_auctions is not None
            else None
        )
        poor_auctions = (
            PoorAuctionDetector().detect(footprint_matrix)
            if footprint_matrix is not None and excess is not None
            else None
        )
        single_prints = (
            SinglePrintDetector().detect(footprint_matrix)
            if footprint_matrix is not None and poor_auctions is not None
            else None
        )
        naked_pocs = (
            NakedPointOfControlTracker().track((point_of_control,))
            if footprint_matrix is not None
            and single_prints is not None
            and point_of_control is not None
            else None
        )
        return DetectionGraph(
            frame_id=graph.frame_id,
            objects=graph.objects,
            grid_coordinate_system=graph.grid_coordinate_system,
            cell_classifications=graph.cell_classifications,
            ocr_results=graph.ocr_results,
            footprint_interpretation=graph.footprint_interpretation,
            parsed_values=graph.parsed_values,
            footprint_matrix=footprint_matrix,
            footprint_imbalances=footprint_imbalances,
            stacked_imbalances=stacked_imbalances,
            absorption=absorption,
            footprint_delta=footprint_delta,
            volume_clusters=volume_clusters,
            point_of_control=point_of_control,
            high_volume_nodes=high_volume_nodes,
            low_volume_nodes=low_volume_nodes,
            value_area=value_area,
            developing_poc=developing_poc,
            developing_value_area=developing_value_area,
            unfinished_auctions=unfinished_auctions,
            excess=excess,
            poor_auctions=poor_auctions,
            single_prints=single_prints,
            naked_pocs=naked_pocs,
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


@dataclass(frozen=True, slots=True)
class FootprintGridDetectorConfig:
    """Immutable thresholds for deterministic footprint-grid detection."""

    min_width_ratio: float = 0.35
    min_height_ratio: float = 0.35
    max_margin_from_chart: float = 0.25
    min_edge_density: float = 0.006
    min_confidence: float = 0.35
    projection_threshold_ratio: float = 0.05
    grid_regularity_tolerance: float = 0.55
    debug_overlay: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_width_ratio",
            "min_height_ratio",
            "max_margin_from_chart",
            "min_edge_density",
            "min_confidence",
            "projection_threshold_ratio",
            "grid_regularity_tolerance",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.min_width_ratio <= 0.0:
            raise ValueError("min_width_ratio must be greater than 0.0")
        if self.min_height_ratio <= 0.0:
            raise ValueError("min_height_ratio must be greater than 0.0")
        if self.max_margin_from_chart <= 0.0:
            raise ValueError("max_margin_from_chart must be greater than 0.0")
        if self.min_edge_density <= 0.0:
            raise ValueError("min_edge_density must be greater than 0.0")
        if self.min_confidence <= 0.0:
            raise ValueError("min_confidence must be greater than 0.0")


class FootprintGridDetector:
    """Detect the rectangular footprint grid inside the chart without OCR/AI/ML."""

    name = "footprint-grid-detector"

    def __init__(self, config: FootprintGridDetectorConfig | None = None) -> None:
        self._config = config or FootprintGridDetectorConfig()

    def detect(self, context: DetectionContext) -> DetectionResult[DetectedObject]:
        frame = context.processed_frame.source_frame
        layout = context.workspace_layout
        chart = layout.chart_region.bounds
        luminance = _luminance(frame)
        edges = _edge_map(luminance, frame.width, frame.height)
        candidate, projection_score, rows, columns = _footprint_grid_candidate(
            luminance, frame.width, chart, self._config
        )
        if candidate is None:
            return self._empty(
                "no regular footprint-grid projections found", frame.width, frame.height
            )
        if not _contains(layout.bounds, candidate):
            return self._empty(
                "footprint grid outside workspace layout",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        if not _contains(chart, candidate):
            return self._empty(
                "footprint grid outside chart region",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        if _axis_overlap(
            candidate, layout.price_axis.bounds, layout.price_axis.confidence
        ):
            return self._empty(
                "footprint grid overlaps price axis",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        if _axis_overlap(
            candidate, layout.time_axis.bounds, layout.time_axis.confidence
        ):
            return self._empty(
                "footprint grid overlaps time axis",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        if not _inside_chart_margins(candidate, chart, self._config):
            return self._empty(
                "footprint grid has highly irregular geometry",
                frame.width,
                frame.height,
                candidate,
                projection_score,
            )
        edge_density = _count_edges(edges, frame.width, candidate) / (
            candidate.width * candidate.height
        )
        regularity = _grid_regularity_score(rows, columns)
        size_score = min(
            candidate.width / (chart.width * self._config.min_width_ratio),
            candidate.height / (chart.height * self._config.min_height_ratio),
            1.0,
        )
        density_score = min(
            edge_density / max(self._config.min_edge_density * 3.0, 0.001), 1.0
        )
        confidence = round(
            min(
                (projection_score * 0.4)
                + (density_score * 0.25)
                + (regularity * 0.2)
                + (size_score * 0.15),
                1.0,
            ),
            3,
        )
        if (
            edge_density < self._config.min_edge_density
            or confidence < self._config.min_confidence
            or regularity < (1.0 - self._config.grid_regularity_tolerance)
        ):
            return self._empty(
                "footprint grid rejected by density/confidence/regularity",
                frame.width,
                frame.height,
                candidate,
                confidence,
            )
        metadata = {
            "estimated_rows": len(rows),
            "estimated_columns": len(columns),
            "grid_width": candidate.width,
            "grid_height": candidate.height,
            "projection_score": round(projection_score, 6),
            "edge_density": round(edge_density, 6),
        }
        detected = DetectedObject(
            object_id=ObjectId(f"{frame.frame_id}:footprint-grid"),
            bounds=candidate,
            confidence=DetectionConfidence(confidence),
            object_type=ObjectType.FOOTPRINT_GRID,
            frame_id=frame.frame_id,
            source=DetectionSource(self.name),
            metadata=metadata,
        )
        return DetectionResult(
            region=candidate,
            confidence=confidence,
            reason="detected footprint grid using deterministic edge/projection analysis only",
            detector_name=self.name,
            debug_overlay=(
                _debug_overlay(
                    frame.width, frame.height, candidate, confidence, "footprint_grid"
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
class FootprintCellDetectorConfig:
    """Immutable thresholds for deterministic footprint-cell grid detection."""

    minimum_cell_width: int = 6
    minimum_cell_height: int = 6
    maximum_spacing_variation: float = 0.25
    maximum_alignment_deviation: int = 2
    minimum_confidence: float = 0.35
    debug_overlay: bool = False

    def __post_init__(self) -> None:
        if self.minimum_cell_width <= 0:
            raise ValueError("minimum_cell_width must be positive")
        if self.minimum_cell_height <= 0:
            raise ValueError("minimum_cell_height must be positive")
        if not 0.0 <= self.maximum_spacing_variation <= 1.0:
            raise ValueError("maximum_spacing_variation must be between 0.0 and 1.0")
        if self.maximum_alignment_deviation < 0:
            raise ValueError("maximum_alignment_deviation must be non-negative")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0.0 and 1.0")


class FootprintCellDetector:
    """Segment a detected footprint grid into deterministic cell rectangles only."""

    name = "footprint-cell-detector"

    def __init__(self, config: FootprintCellDetectorConfig | None = None) -> None:
        self._config = config or FootprintCellDetectorConfig()

    def detect(self, context: DetectionContext) -> DetectionResult[DetectedObject]:
        frame = context.processed_frame.source_frame
        grid_result = FootprintGridDetector().detect(context)
        grid_object = grid_result.detected_object
        if grid_result.region is None or grid_object is None:
            return self._empty(
                "footprint grid is required before cell detection",
                frame.width,
                frame.height,
            )
        grid = grid_result.region
        chart = context.workspace_layout.chart_region.bounds
        luminance = _luminance(frame)
        columns = _grid_line_centers(luminance, frame.width, grid, vertical=True)
        rows = _grid_line_centers(luminance, frame.width, grid, vertical=False)
        if len(columns) < 2 or len(rows) < 2:
            return self._empty(
                "insufficient grid lines for footprint cells",
                frame.width,
                frame.height,
                grid,
                grid_result.confidence,
            )
        if _has_duplicates(columns) or _has_duplicates(rows):
            return self._empty(
                "duplicate row/column coordinates rejected",
                frame.width,
                frame.height,
                grid,
                grid_result.confidence,
            )
        if not _spacing_regular(
            columns, self._config.maximum_spacing_variation
        ) or not _spacing_regular(rows, self._config.maximum_spacing_variation):
            return self._empty(
                "irregular footprint-cell spacing rejected",
                frame.width,
                frame.height,
                grid,
                grid_result.confidence,
            )
        cell_widths = tuple(right - left for left, right in zip(columns, columns[1:]))
        cell_heights = tuple(bottom - top for top, bottom in zip(rows, rows[1:]))
        if (
            min(cell_widths) < self._config.minimum_cell_width
            or min(cell_heights) < self._config.minimum_cell_height
        ):
            return self._empty(
                "footprint cells smaller than configured minimum",
                frame.width,
                frame.height,
                grid,
                grid_result.confidence,
            )
        cells: list[DetectedObject] = []
        for row_index, (top, bottom) in enumerate(zip(rows, rows[1:])):
            for column_index, (left, right) in enumerate(zip(columns, columns[1:])):
                box = BoundingBox(left, top, right - left, bottom - top)
                if not self._valid_cell(box, grid, chart, context.workspace_layout):
                    return self._empty(
                        "footprint cell failed containment/axis validation",
                        frame.width,
                        frame.height,
                        box,
                        grid_result.confidence,
                    )
                confidence = round(
                    min(
                        grid_result.confidence
                        * _cell_alignment_score(box, cell_widths, cell_heights),
                        1.0,
                    ),
                    3,
                )
                if confidence < self._config.minimum_confidence:
                    return self._empty(
                        "footprint cell confidence below minimum",
                        frame.width,
                        frame.height,
                        box,
                        confidence,
                    )
                cells.append(
                    DetectedObject(
                        object_id=ObjectId(
                            f"{frame.frame_id}:footprint-cell:{row_index}:{column_index}"
                        ),
                        bounds=box,
                        confidence=DetectionConfidence(confidence),
                        object_type=ObjectType.FOOTPRINT_CELL,
                        frame_id=frame.frame_id,
                        source=DetectionSource(self.name),
                        parent_id=grid_object.object_id,
                        metadata={
                            "row_index": row_index,
                            "column_index": column_index,
                            "cell_id": f"{grid_object.object_id.value}:cell:{row_index}:{column_index}",
                            "grid_id": grid_object.object_id.value,
                            "cell_width": box.width,
                            "cell_height": box.height,
                            "grid_width": grid.width,
                            "grid_height": grid.height,
                        },
                    )
                )
        result_confidence = min((cell.confidence.value for cell in cells), default=0.0)
        return DetectionResult(
            region=grid,
            confidence=result_confidence,
            reason="segmented footprint grid into deterministic cell geometry only; no OCR, AI, ML, bid/ask, volume, or delta classification",
            detector_name=self.name,
            debug_overlay=(
                _debug_overlay_many(
                    frame.width,
                    frame.height,
                    grid,
                    tuple(cell.bounds for cell in cells),
                    result_confidence,
                )
                if self._config.debug_overlay
                else None
            ),
            detected_object=cells[0] if cells else None,
            detected_objects=tuple(cells),
        )

    def _valid_cell(
        self,
        box: BoundingBox,
        grid: BoundingBox,
        chart: BoundingBox,
        layout: WorkspaceLayout,
    ) -> bool:
        return (
            _contains(grid, box)
            and _contains(chart, box)
            and not _axis_overlap(
                box, layout.price_axis.bounds, layout.price_axis.confidence
            )
            and not _axis_overlap(
                box, layout.time_axis.bounds, layout.time_axis.confidence
            )
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


def _footprint_grid_candidate(
    luminance: Sequence[int],
    frame_width: int,
    chart: BoundingBox,
    config: FootprintGridDetectorConfig,
) -> tuple[BoundingBox | None, float, tuple[int, ...], tuple[int, ...]]:
    min_col_edges = max(2, round(chart.height * config.projection_threshold_ratio))
    min_row_edges = max(2, round(chart.width * config.projection_threshold_ratio))
    border_x = max(2, round(chart.width * 0.01))
    border_y = max(2, round(chart.height * 0.01))
    cols = tuple(
        x
        for x in range(chart.x + border_x, chart.right - border_x)
        if x > 0
        and sum(
            1
            for y in range(chart.y + border_y, chart.bottom - border_y)
            if abs(luminance[y * frame_width + x] - luminance[y * frame_width + x - 1])
            >= 45
        )
        >= min_col_edges
    )
    rows = tuple(
        y
        for y in range(chart.y + border_y, chart.bottom - border_y)
        if y > 0
        and sum(
            1
            for x in range(chart.x + border_x, chart.right - border_x)
            if abs(
                luminance[y * frame_width + x] - luminance[(y - 1) * frame_width + x]
            )
            >= 45
        )
        >= min_row_edges
    )
    if len(cols) < 2 or len(rows) < 2:
        return None, 0.0, (), ()
    column_centers = _cluster_centers(cols)
    row_centers = _cluster_centers(rows)
    if len(column_centers) < 2 or len(row_centers) < 2:
        return None, 0.0, row_centers, column_centers
    box = BoundingBox(
        min(cols), min(rows), max(cols) - min(cols) + 1, max(rows) - min(rows) + 1
    )
    if box.width < round(chart.width * config.min_width_ratio) or box.height < round(
        chart.height * config.min_height_ratio
    ):
        return None, 0.0, row_centers, column_centers
    col_score = len(column_centers) / max(chart.width * 0.08, 1.0)
    row_score = len(row_centers) / max(chart.height * 0.08, 1.0)
    return (
        box,
        round(min((col_score + row_score) / 2.0, 1.0), 6),
        row_centers,
        column_centers,
    )


def _cluster_centers(indices: Sequence[int]) -> tuple[int, ...]:
    if not indices:
        return ()
    clusters: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index <= previous + 2:
            previous = index
            continue
        clusters.append((start, previous))
        start = previous = index
    clusters.append((start, previous))
    return tuple((start + end) // 2 for start, end in clusters)


def _grid_line_centers(
    luminance: Sequence[int], width: int, grid: BoundingBox, vertical: bool
) -> tuple[int, ...]:
    indices: list[int] = []
    if vertical:
        threshold = max(2, round(grid.height * 0.45))
        for x in range(grid.x, grid.right):
            if x == 0:
                continue
            count = sum(
                1
                for y in range(grid.y, grid.bottom)
                if abs(luminance[y * width + x] - luminance[y * width + x - 1]) >= 45
            )
            if count >= threshold:
                indices.append(x)
    else:
        threshold = max(2, round(grid.width * 0.45))
        for y in range(grid.y, grid.bottom):
            if y == 0:
                continue
            count = sum(
                1
                for x in range(grid.x, grid.right)
                if abs(luminance[y * width + x] - luminance[(y - 1) * width + x]) >= 45
            )
            if count >= threshold:
                indices.append(y)
    return _cluster_centers(tuple(indices))


def _has_duplicates(indices: Sequence[int]) -> bool:
    return len(set(indices)) != len(indices)


def _spacing_regular(indices: Sequence[int], tolerance: float) -> bool:
    if len(indices) < 3:
        return True
    gaps = [right - left for left, right in zip(indices, indices[1:])]
    if any(gap <= 0 for gap in gaps):
        return False
    average = sum(gaps) / len(gaps)
    if average <= 0:
        return False
    return all(abs(gap - average) / average <= tolerance for gap in gaps)


def _cell_alignment_score(
    box: BoundingBox, cell_widths: Sequence[int], cell_heights: Sequence[int]
) -> float:
    average_width = sum(cell_widths) / len(cell_widths)
    average_height = sum(cell_heights) / len(cell_heights)
    width_score = 1.0 - min(abs(box.width - average_width) / average_width, 1.0)
    height_score = 1.0 - min(abs(box.height - average_height) / average_height, 1.0)
    return max(0.0, min((width_score + height_score) / 2.0, 1.0))


def _axis_overlap(
    candidate: BoundingBox, axis: BoundingBox, axis_confidence: float
) -> bool:
    return axis_confidence > 0.0 and _overlap_area(candidate, axis) > 0


def _inside_chart_margins(
    candidate: BoundingBox, chart: BoundingBox, config: FootprintGridDetectorConfig
) -> bool:
    max_margin_x = round(chart.width * config.max_margin_from_chart)
    max_margin_y = round(chart.height * config.max_margin_from_chart)
    return (
        candidate.x - chart.x <= max_margin_x + 2
        and chart.right - candidate.right <= max_margin_x + 2
        and candidate.y - chart.y <= max_margin_y + 2
        and chart.bottom - candidate.bottom <= max_margin_y + 2
    )


def _grid_regularity_score(rows: Sequence[int], columns: Sequence[int]) -> float:
    return min(_axis_regularity_score(rows), _axis_regularity_score(columns))


def _axis_regularity_score(indices: Sequence[int]) -> float:
    if len(indices) < 3:
        return 1.0
    gaps = [right - left for left, right in zip(indices, indices[1:]) if right > left]
    if not gaps:
        return 0.0
    average = sum(gaps) / len(gaps)
    if average == 0:
        return 0.0
    mean_deviation = sum(abs(gap - average) for gap in gaps) / len(gaps)
    return max(0.0, 1.0 - min(mean_deviation / average, 1.0))


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


def _debug_overlay_many(
    width: int,
    height: int,
    grid: BoundingBox,
    cells: tuple[BoundingBox, ...],
    confidence: float,
) -> DebugOverlay:
    pixels = bytearray([24, 28, 34] * width * height)
    _draw_rect(pixels, width, height, grid, (255, 128, 0))
    for cell in cells:
        _draw_rect(pixels, width, height, cell, (0, 180, 255))
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


ObjectType.CHART = ObjectType("CHART")
ObjectType.PRICE_TEXT = ObjectType("PRICE_TEXT")
ObjectType.PRICE_AXIS = ObjectType("PRICE_AXIS")
ObjectType.TIME_AXIS = ObjectType("TIME_AXIS")
ObjectType.TIME_LABEL = ObjectType("TIME_LABEL")
ObjectType.CANDLE = ObjectType("CANDLE")
ObjectType.FOOTPRINT_GRID = ObjectType("FOOTPRINT_GRID")
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


def _normalize_decimal_points(text: str) -> tuple[str, str]:
    if text.count(".") <= 1:
        return text, ""
    first = text.find(".")
    return (
        text[: first + 1] + text[first + 1 :].replace(".", ""),
        "multiple decimal separators normalized",
    )


def _cleanup_duplicate_leading_signs(text: str) -> tuple[str, str]:
    if len(text) > 1 and set(text[:2]) <= {"+", "-"} and text[0] == text[1]:
        sign = "-" if text[0] == "-" else ""
        return sign + text.lstrip("+-"), "duplicate sign cleanup"
    return text, ""


def _trim_garbage(text: str, configuration: OCRNormalizationConfiguration) -> str:
    allowed = set(configuration.allowed_characters) | {",", "+"}
    start = 0
    end = len(text)
    while start < end and text[start] not in allowed:
        start += 1
    while end > start and text[end - 1] not in allowed:
        end -= 1
    return text[start:end]


def _numeric_error(
    text: str, configuration: OCRNormalizationConfiguration
) -> ParsingError | None:
    if text == "":
        return ParsingError("empty normalized text", text, 0)
    if len(text) < configuration.minimum_length:
        return ParsingError(
            "normalized text shorter than minimum length", text, len(text)
        )
    if len(text) > configuration.maximum_length:
        return ParsingError("overflow", text, configuration.maximum_length)
    lowered = text.lower()
    if lowered in {"nan", "inf", "infinity", "+inf", "-inf", "+infinity", "-infinity"}:
        return ParsingError("non-finite value", text, 0)
    allowed = set(configuration.allowed_characters) | {"+"}
    for index, char in enumerate(text):
        if char not in allowed:
            return ParsingError(
                "alphabetic or disallowed character after normalization", text, index
            )
    if not configuration.allow_negative and "-" in text:
        return ParsingError("negative values are not allowed", text, text.find("-"))
    if not configuration.allow_decimal and "." in text:
        return ParsingError("decimal values are not allowed", text, text.find("."))
    if text.count("-") + text.count("+") > 1:
        return ParsingError("multiple signs", text, 0)
    if ("-" in text and not text.startswith("-")) or (
        "+" in text and not text.startswith("+")
    ):
        return ParsingError(
            "sign must be leading", text, max(text.find("-"), text.find("+"))
        )
    unsigned = text[1:] if text.startswith(("-", "+")) else text
    if unsigned == "":
        return ParsingError("empty normalized text", text, len(text) - 1)
    if unsigned.count(".") > 1:
        return ParsingError(
            "multiple decimal points", text, text.find(".", text.find(".") + 1)
        )
    if unsigned.startswith("."):
        return ParsingError("leading decimal without digits", text, text.find("."))
    if unsigned.endswith("."):
        return ParsingError("trailing decimal without digits", text, len(text) - 1)
    if not any(char.isdigit() for char in unsigned):
        return ParsingError("empty normalized text", text, 0)
    return None


def _numeric_type(text: str) -> NumericType:
    signed = text.startswith(("-", "+"))
    decimal = "." in text
    if signed and decimal:
        return NumericType.SIGNED_DECIMAL
    if signed:
        return NumericType.SIGNED_INTEGER
    if decimal:
        return NumericType.DECIMAL
    return NumericType.INTEGER


def _boundary_key(entry: Any) -> UnfinishedAuctionType | ExcessType | PoorAuctionType:
    if isinstance(entry, UnfinishedAuction):
        return entry.auction_type
    if isinstance(entry, Excess):
        return entry.excess_type
    if isinstance(entry, PoorAuction):
        return entry.auction_type
    raise ValueError("boundary result entry type is invalid")


def _boundary_order(key: UnfinishedAuctionType | ExcessType | PoorAuctionType) -> int:
    order = {
        UnfinishedAuctionType.TOP: 0,
        UnfinishedAuctionType.BOTTOM: 1,
        ExcessType.EXCESS_HIGH: 0,
        ExcessType.EXCESS_LOW: 1,
        PoorAuctionType.POOR_HIGH: 0,
        PoorAuctionType.POOR_LOW: 1,
    }
    return order[key]
