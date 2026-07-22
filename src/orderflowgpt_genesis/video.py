"""Deterministic Fabio video ingestion and synchronization primitives.

Bundle 10 intentionally implements no AI, no ML, no prediction, no reasoning, no
speech recognition, and no computer-vision algorithm changes. Frames are decoded
into deterministic references and passed to the existing Genesis data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from .dataset import (
    DatasetBuilder,
    DatasetConfiguration,
    FrameIdentifier,
    FrameMetadata,
    TrainingSample,
)
from .vision import DetectionGraph, ImageFrame


class FrameSamplingMode(Enum):
    FPS = "FPS"
    EVERY_N_FRAMES = "EVERY_N_FRAMES"
    KEYFRAMES = "KEYFRAMES"


@dataclass(frozen=True, slots=True)
class VideoIdentifier:
    video_id: str
    source_uri: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video id is required")


@dataclass(frozen=True, slots=True)
class VideoConfiguration:
    target_fps: int = 1
    every_n_frames: int = 1
    keyframe_interval: int = 30
    sampling_mode: FrameSamplingMode = FrameSamplingMode.EVERY_N_FRAMES
    synthetic_frame_count: int = 1
    frame_width: int = 16
    frame_height: int = 9
    source_fps: int = 30
    audio_segment_seconds: int = 1

    def __post_init__(self) -> None:
        for name in (
            "target_fps",
            "every_n_frames",
            "keyframe_interval",
            "synthetic_frame_count",
            "frame_width",
            "frame_height",
            "source_fps",
            "audio_segment_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    identifier: VideoIdentifier
    duration_seconds: float
    source_fps: int
    frame_count: int
    width: int
    height: int
    imported_at: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError("duration cannot be negative")
        if min(self.source_fps, self.frame_count, self.width, self.height) <= 0:
            raise ValueError("video dimensions, fps, and frame count must be positive")
        if self.imported_at.tzinfo is None:
            raise ValueError("import timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class FrameTimestamp:
    frame_number: int
    timestamp: datetime
    offset_seconds: float

    def __post_init__(self) -> None:
        if self.frame_number < 0 or self.offset_seconds < 0:
            raise ValueError("frame timestamps cannot be negative")
        if self.timestamp.tzinfo is None:
            raise ValueError("frame timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class VideoFrame:
    identifier: FrameIdentifier
    timestamp: FrameTimestamp
    image: ImageFrame
    content_hash: str
    keyframe: bool = False

    def __post_init__(self) -> None:
        if self.identifier.sequence_number != self.timestamp.frame_number:
            raise ValueError("frame number must match identifier sequence")
        if self.identifier.frame_id != self.image.frame_id:
            raise ValueError("frame identifier must match image frame id")
        if not self.content_hash.strip():
            raise ValueError("frame content hash is required")


@dataclass(frozen=True, slots=True)
class FrameSequence:
    frames: tuple[VideoFrame, ...]

    def __post_init__(self) -> None:
        nums = [f.timestamp.frame_number for f in self.frames]
        if nums != sorted(nums):
            raise ValueError("frames must be ordered by frame number")
        if len(set(nums)) != len(nums):
            raise ValueError("duplicate frame numbers are not allowed")


@dataclass(frozen=True, slots=True)
class FrameCollection:
    video: VideoIdentifier
    sequence: FrameSequence


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    duration_seconds: float
    sample_rate_hz: int = 48000
    channels: int = 2

    def __post_init__(self) -> None:
        if self.duration_seconds < 0 or self.sample_rate_hz <= 0 or self.channels <= 0:
            raise ValueError(
                "audio metadata values must be non-negative and positive where required"
            )


@dataclass(frozen=True, slots=True)
class AudioSegment:
    segment_id: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not self.segment_id.strip():
            raise ValueError("audio segment id is required")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("invalid audio segment range")


@dataclass(frozen=True, slots=True)
class AudioTimeline:
    metadata: AudioMetadata
    segments: tuple[AudioSegment, ...]

    def __post_init__(self) -> None:
        starts = [s.start_seconds for s in self.segments]
        if starts != sorted(starts):
            raise ValueError("audio segments must be ordered")
        if len({s.segment_id for s in self.segments}) != len(self.segments):
            raise ValueError("audio segment ids must be unique")


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    segment_id: str
    start_seconds: float
    end_seconds: float
    text: str = ""


@dataclass(frozen=True, slots=True)
class SynchronizationPoint:
    frame_id: str
    frame_number: int
    timestamp: FrameTimestamp
    audio_segment_id: str | None
    training_sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class SynchronizationMap:
    video: VideoIdentifier
    points: tuple[SynchronizationPoint, ...]

    def __post_init__(self) -> None:
        keys = [p.frame_id for p in self.points]
        if len(set(keys)) != len(keys):
            raise ValueError("synchronization frame references must be unique")
        if [p.frame_number for p in self.points] != sorted(
            p.frame_number for p in self.points
        ):
            raise ValueError("synchronization points must be ordered")


@dataclass(frozen=True, slots=True)
class VideoStatistics:
    frame_count: int
    audio_segment_count: int
    training_sample_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class VideoDataset:
    metadata: VideoMetadata
    frames: FrameCollection
    audio: AudioTimeline
    synchronization: SynchronizationMap
    training_samples: tuple[TrainingSample, ...]

    def statistics(self) -> VideoStatistics:
        return VideoStatistics(
            len(self.frames.sequence.frames),
            len(self.audio.segments),
            len(self.training_samples),
            self.metadata.duration_seconds,
        )


@dataclass(frozen=True, slots=True)
class VideoImportResult:
    dataset: VideoDataset
    configuration: VideoConfiguration


class VideoDecoder:
    def decode(
        self, source: str | Path, configuration: VideoConfiguration
    ) -> VideoMetadata:
        path = Path(source)
        data = path.read_bytes() if path.exists() else str(source).encode()
        digest = sha256(data).hexdigest()
        identifier = VideoIdentifier(f"video:{digest[:16]}", str(source), digest)
        duration = configuration.synthetic_frame_count / configuration.source_fps
        return VideoMetadata(
            identifier,
            duration,
            configuration.source_fps,
            configuration.synthetic_frame_count,
            configuration.frame_width,
            configuration.frame_height,
        )


class TimestampExtractor:
    def extract(self, metadata: VideoMetadata) -> tuple[FrameTimestamp, ...]:
        base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return tuple(
            FrameTimestamp(
                i,
                base + timedelta(seconds=i / metadata.source_fps),
                i / metadata.source_fps,
            )
            for i in range(metadata.frame_count)
        )


class FrameSampler:
    def sample(
        self, metadata: VideoMetadata, configuration: VideoConfiguration
    ) -> tuple[int, ...]:
        if configuration.sampling_mode is FrameSamplingMode.FPS:
            step = max(1, metadata.source_fps // configuration.target_fps)
        elif configuration.sampling_mode is FrameSamplingMode.KEYFRAMES:
            step = configuration.keyframe_interval
        else:
            step = configuration.every_n_frames
        return tuple(range(0, metadata.frame_count, step))


class FrameExtractor:
    def extract(
        self, metadata: VideoMetadata, configuration: VideoConfiguration
    ) -> FrameCollection:
        timestamps = TimestampExtractor().extract(metadata)
        nums = FrameSampler().sample(metadata, configuration)
        frames = []
        for n in nums:
            payload = f"{metadata.identifier.content_hash}:{n}".encode()
            digest = sha256(payload).hexdigest()
            frame_id = f"{metadata.identifier.video_id}:frame:{n:012d}:{digest[:16]}"
            image = ImageFrame(
                payload,
                metadata.width,
                metadata.height,
                "deterministic-bytes",
                timestamps[n].timestamp,
                metadata.identifier.source_uri,
                frame_id,
            )
            frames.append(
                VideoFrame(
                    FrameIdentifier(frame_id, None, n),
                    timestamps[n],
                    image,
                    digest,
                    n % configuration.keyframe_interval == 0,
                )
            )
        return FrameCollection(metadata.identifier, FrameSequence(tuple(frames)))


class AudioExtractor:
    def extract(
        self, metadata: VideoMetadata, configuration: VideoConfiguration
    ) -> AudioTimeline:
        count = max(
            1,
            int(metadata.duration_seconds // configuration.audio_segment_seconds)
            + (
                1
                if metadata.duration_seconds % configuration.audio_segment_seconds
                else 0
            ),
        )
        segments = tuple(
            AudioSegment(
                f"{metadata.identifier.video_id}:audio:{i:06d}",
                i * configuration.audio_segment_seconds,
                min(
                    metadata.duration_seconds,
                    (i + 1) * configuration.audio_segment_seconds,
                ),
            )
            for i in range(count)
        )
        return AudioTimeline(AudioMetadata(metadata.duration_seconds), segments)


class FrameAudioSynchronizer:
    def synchronize(
        self,
        video: VideoIdentifier,
        frames: FrameCollection,
        audio: AudioTimeline,
        samples: Iterable[TrainingSample] = (),
    ) -> SynchronizationMap:
        sample_by_frame = {s.metadata.identifier.frame_id: s.sample_id for s in samples}
        points = []
        for frame in frames.sequence.frames:
            segment = next(
                (
                    s
                    for s in audio.segments
                    if s.start_seconds
                    <= frame.timestamp.offset_seconds
                    <= s.end_seconds
                ),
                None,
            )
            points.append(
                SynchronizationPoint(
                    frame.identifier.frame_id,
                    frame.timestamp.frame_number,
                    frame.timestamp,
                    None if segment is None else segment.segment_id,
                    sample_by_frame.get(frame.identifier.frame_id),
                )
            )
        return SynchronizationMap(video, tuple(points))


class SynchronizationAnalyzer:
    def analyze(self, synchronization: SynchronizationMap) -> VideoStatistics:
        return VideoStatistics(
            len(synchronization.points),
            len(
                {
                    p.audio_segment_id
                    for p in synchronization.points
                    if p.audio_segment_id
                }
            ),
            len([p for p in synchronization.points if p.training_sample_id]),
            (
                synchronization.points[-1].timestamp.offset_seconds
                if synchronization.points
                else 0
            ),
        )


class VideoDatasetBuilder:
    def build(
        self,
        metadata: VideoMetadata,
        frames: FrameCollection,
        audio: AudioTimeline,
        training_samples: Iterable[TrainingSample],
    ) -> VideoDataset:
        samples = tuple(training_samples)
        synchronization = FrameAudioSynchronizer().synchronize(
            metadata.identifier, frames, audio, samples
        )
        return VideoDataset(metadata, frames, audio, synchronization, samples)


class VideoImporter:
    def __init__(self, configuration: VideoConfiguration | None = None) -> None:
        self.configuration = configuration or VideoConfiguration()

    def import_video(
        self, source: str | Path, dataset_id: str = "fabio-video-dataset"
    ) -> VideoImportResult:
        metadata = VideoDecoder().decode(source, self.configuration)
        frames = FrameExtractor().extract(metadata, self.configuration)
        audio = AudioExtractor().extract(metadata, self.configuration)
        dataset_builder = DatasetBuilder(DatasetConfiguration(dataset_id))
        pairs = [
            (
                FrameMetadata(
                    f.identifier,
                    f.timestamp.timestamp,
                    metadata.width,
                    metadata.height,
                    f.content_hash,
                ),
                DetectionGraph(f.identifier.frame_id),
            )
            for f in frames.sequence.frames
        ]
        training_dataset = dataset_builder.build(pairs)
        synchronization = FrameAudioSynchronizer().synchronize(
            metadata.identifier, frames, audio, training_dataset.samples
        )
        return VideoImportResult(
            VideoDataset(
                metadata, frames, audio, synchronization, training_dataset.samples
            ),
            self.configuration,
        )
