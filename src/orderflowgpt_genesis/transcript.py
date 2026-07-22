"""Deterministic Fabio transcript import and frame alignment primitives.

Bundle 11 intentionally performs no AI reasoning, learning, prediction, strategy
creation, or probabilistic inference. Transcript text is parsed and aligned only by
explicit timestamps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Literal

from .dataset import FrameMetadata, TrainingSample

TranscriptFormat = Literal["srt", "vtt", "txt", "json"]


class AlignmentConfidence(Enum):
    EXACT = "EXACT"
    WITHIN_WINDOW = "WITHIN_WINDOW"
    NEAREST = "NEAREST"
    UNALIGNED = "UNALIGNED"


@dataclass(frozen=True, slots=True)
class TranscriptIdentifier:
    transcript_id: str
    video_id: str
    source_uri: str = ""

    def __post_init__(self) -> None:
        if not self.transcript_id.strip():
            raise ValueError("transcript id is required")
        if not self.video_id.strip():
            raise ValueError("video id is required")


@dataclass(frozen=True, slots=True)
class TranscriptMetadata:
    identifier: TranscriptIdentifier
    language: str = "und"
    created_at: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    format: TranscriptFormat = "txt"
    checksum: str = ""

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("transcript creation timestamp must be timezone-aware")
        if not self.language.strip():
            raise ValueError("transcript language is required")


@dataclass(frozen=True, slots=True)
class TranscriptToken:
    token_id: str
    text: str
    start_ms: int
    end_ms: int
    sentence_id: str

    def __post_init__(self) -> None:
        _validate_span(self.start_ms, self.end_ms)
        if not self.token_id.strip() or not self.sentence_id.strip():
            raise ValueError("token id and sentence id are required")
        if not self.text.strip():
            raise ValueError("token text is required")


@dataclass(frozen=True, slots=True)
class TranscriptSentence:
    sentence_id: str
    text: str
    start_ms: int
    end_ms: int
    token_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_span(self.start_ms, self.end_ms)
        if not self.sentence_id.strip():
            raise ValueError("sentence id is required")
        if not self.text.strip():
            raise ValueError("sentence text is required")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("sentence token ids must be unique")


@dataclass(frozen=True, slots=True)
class TranscriptParagraph:
    paragraph_id: str
    sentence_ids: tuple[str, ...]
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        _validate_span(self.start_ms, self.end_ms)
        if not self.paragraph_id.strip():
            raise ValueError("paragraph id is required")
        if not self.sentence_ids:
            raise ValueError("paragraph requires at least one sentence")
        if len(set(self.sentence_ids)) != len(self.sentence_ids):
            raise ValueError("paragraph sentence ids must be unique")


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    segment_id: str
    text: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        _validate_span(self.start_ms, self.end_ms)
        if not self.segment_id.strip():
            raise ValueError("segment id is required")
        if not self.text.strip():
            raise ValueError("segment text is required")


@dataclass(frozen=True, slots=True)
class TranscriptStatistics:
    sentence_count: int
    token_count: int
    paragraph_count: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class TranscriptConfiguration:
    transcript_id: str
    video_id: str
    language: str = "und"
    timestamp_tolerance_ms: int = 500
    active_window_ms: int = 2_000

    def __post_init__(self) -> None:
        if not self.transcript_id.strip() or not self.video_id.strip():
            raise ValueError("transcript and video ids are required")
        if self.timestamp_tolerance_ms < 0 or self.active_window_ms < 0:
            raise ValueError("alignment windows cannot be negative")


@dataclass(frozen=True, slots=True)
class TranscriptTimeline:
    metadata: TranscriptMetadata
    sentences: tuple[TranscriptSentence, ...]
    tokens: tuple[TranscriptToken, ...] = ()
    paragraphs: tuple[TranscriptParagraph, ...] = ()
    segments: tuple[TranscriptSegment, ...] = ()

    def __post_init__(self) -> None:
        ids = [s.sentence_id for s in self.sentences]
        if len(set(ids)) != len(ids):
            raise ValueError("sentence ids must be unique")
        if (
            tuple(
                sorted(
                    self.sentences, key=lambda s: (s.start_ms, s.end_ms, s.sentence_id)
                )
            )
            != self.sentences
        ):
            raise ValueError("sentences must be timestamp ordered")
        token_ids = [t.token_id for t in self.tokens]
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("token ids must be unique")
        known = set(ids)
        if any(t.sentence_id not in known for t in self.tokens):
            raise ValueError("tokens must reference transcript sentences")
        if any(sid not in known for p in self.paragraphs for sid in p.sentence_ids):
            raise ValueError("paragraphs must reference transcript sentences")

    def statistics(self) -> TranscriptStatistics:
        duration = max((s.end_ms for s in self.sentences), default=0)
        return TranscriptStatistics(
            len(self.sentences), len(self.tokens), len(self.paragraphs), duration
        )


@dataclass(frozen=True, slots=True)
class TranscriptAlignment:
    alignment_id: str
    transcript_id: str
    video_id: str
    sentence_id: str
    start_ms: int
    end_ms: int
    confidence: AlignmentConfidence

    def __post_init__(self) -> None:
        _validate_span(self.start_ms, self.end_ms)
        if not all(
            v.strip()
            for v in (
                self.alignment_id,
                self.transcript_id,
                self.video_id,
                self.sentence_id,
            )
        ):
            raise ValueError("alignment references are required")


@dataclass(frozen=True, slots=True)
class FrameTranscriptAlignment:
    frame_id: str
    video_id: str
    timestamp_ms: int
    nearest_sentence_id: str | None
    previous_sentence_id: str | None
    next_sentence_id: str | None
    active_sentence_ids: tuple[str, ...]
    confidence: AlignmentConfidence

    def __post_init__(self) -> None:
        if not self.frame_id.strip() or not self.video_id.strip():
            raise ValueError("frame and video ids are required")
        if self.timestamp_ms < 0:
            raise ValueError("frame timestamp cannot be negative")
        if len(set(self.active_sentence_ids)) != len(self.active_sentence_ids):
            raise ValueError("active sentence ids must be unique")


@dataclass(frozen=True, slots=True)
class TranscriptDataset:
    dataset_id: str
    timeline: TranscriptTimeline
    alignments: tuple[FrameTranscriptAlignment, ...]
    samples: tuple[TrainingSample, ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("transcript dataset id is required")
        frame_ids = [a.frame_id for a in self.alignments]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("frame transcript alignments must be unique per frame")
        known = {s.sentence_id for s in self.timeline.sentences}
        refs = {
            r
            for a in self.alignments
            for r in (
                a.nearest_sentence_id,
                a.previous_sentence_id,
                a.next_sentence_id,
                *a.active_sentence_ids,
            )
            if r is not None
        }
        if refs - known:
            raise ValueError("frame alignments must reference transcript sentences")


class TranscriptImporter:
    def import_text(
        self,
        payload: str,
        configuration: TranscriptConfiguration,
        transcript_format: TranscriptFormat,
    ) -> TranscriptTimeline:
        segments = _parse_segments(payload, transcript_format)
        return _timeline_from_segments(segments, configuration, transcript_format)

    def import_file(
        self,
        path: str | Path,
        configuration: TranscriptConfiguration,
        transcript_format: TranscriptFormat | None = None,
    ) -> TranscriptTimeline:
        p = Path(path)
        fmt = transcript_format or p.suffix.lower().lstrip(".")
        if fmt not in ("srt", "vtt", "txt", "json"):
            raise ValueError("unsupported transcript format")
        return self.import_text(p.read_text(encoding="utf-8"), configuration, fmt)  # type: ignore[arg-type]


class SentenceMapper:
    def map(
        self, timeline: TranscriptTimeline
    ) -> MappingProxyType[str, TranscriptSentence]:
        return MappingProxyType({s.sentence_id: s for s in timeline.sentences})


class TimestampMatcher:
    def nearest(
        self, timestamp_ms: int, sentences: tuple[TranscriptSentence, ...]
    ) -> TranscriptSentence | None:
        if not sentences:
            return None
        return min(
            sentences,
            key=lambda s: (
                (
                    0
                    if s.start_ms <= timestamp_ms <= s.end_ms
                    else min(
                        abs(timestamp_ms - s.start_ms), abs(timestamp_ms - s.end_ms)
                    )
                ),
                s.start_ms,
                s.sentence_id,
            ),
        )


class TimelineSynchronizer:
    def synchronize(
        self, timeline: TranscriptTimeline
    ) -> tuple[TranscriptAlignment, ...]:
        return tuple(
            TranscriptAlignment(
                f"alignment:{timeline.metadata.identifier.transcript_id}:{s.sentence_id}",
                timeline.metadata.identifier.transcript_id,
                timeline.metadata.identifier.video_id,
                s.sentence_id,
                s.start_ms,
                s.end_ms,
                AlignmentConfidence.EXACT,
            )
            for s in timeline.sentences
        )


class FrameAligner:
    def __init__(self, configuration: TranscriptConfiguration) -> None:
        self.configuration = configuration
        self.matcher = TimestampMatcher()

    def align(
        self, frame: FrameMetadata, timeline: TranscriptTimeline
    ) -> FrameTranscriptAlignment:
        timestamp_ms = int(frame.timestamp.timestamp() * 1000)
        sentences = timeline.sentences
        nearest = self.matcher.nearest(timestamp_ms, sentences)
        previous = next(
            (s for s in reversed(sentences) if s.end_ms <= timestamp_ms), None
        )
        next_sentence = next((s for s in sentences if s.start_ms >= timestamp_ms), None)
        active = tuple(
            s.sentence_id
            for s in sentences
            if s.start_ms - self.configuration.active_window_ms
            <= timestamp_ms
            <= s.end_ms + self.configuration.active_window_ms
        )
        confidence = (
            AlignmentConfidence.UNALIGNED
            if nearest is None
            else (
                AlignmentConfidence.EXACT
                if nearest.start_ms <= timestamp_ms <= nearest.end_ms
                else AlignmentConfidence.NEAREST
            )
        )
        return FrameTranscriptAlignment(
            frame.identifier.frame_id,
            timeline.metadata.identifier.video_id,
            timestamp_ms,
            nearest.sentence_id if nearest else None,
            previous.sentence_id if previous else None,
            next_sentence.sentence_id if next_sentence else None,
            active,
            confidence,
        )


class TranscriptAligner:
    def __init__(self, configuration: TranscriptConfiguration) -> None:
        self.configuration = configuration
        self.frame_aligner = FrameAligner(configuration)
        self.timeline_synchronizer = TimelineSynchronizer()

    def align(
        self, frames: Iterable[FrameMetadata], timeline: TranscriptTimeline
    ) -> TranscriptDataset:
        alignments = tuple(
            self.frame_aligner.align(frame, timeline) for frame in frames
        )
        return TranscriptDataset(
            f"transcript-dataset:{self.configuration.video_id}:{self.configuration.transcript_id}",
            timeline,
            alignments,
        )


def _validate_span(start_ms: int, end_ms: int) -> None:
    if start_ms < 0 or end_ms < 0:
        raise ValueError("timestamps cannot be negative")
    if end_ms < start_ms:
        raise ValueError(
            "end timestamp must be greater than or equal to start timestamp"
        )


def _parse_segments(
    payload: str, transcript_format: TranscriptFormat
) -> tuple[TranscriptSegment, ...]:
    if transcript_format == "json":
        raw = json.loads(payload)
        rows = raw.get("segments", raw if isinstance(raw, list) else [])
        return tuple(
            TranscriptSegment(
                f"segment:{i}",
                str(r["text"]).strip(),
                int(r.get("start_ms", r.get("start", 0))),
                int(r.get("end_ms", r.get("end", 0))),
            )
            for i, r in enumerate(rows)
        )
    if transcript_format in ("srt", "vtt"):
        return _parse_cue_text(payload)
    if transcript_format == "txt":
        return _parse_timestamped_txt(payload)
    raise ValueError("unsupported transcript format")


def _parse_cue_text(payload: str) -> tuple[TranscriptSegment, ...]:
    blocks = re.split(r"\n\s*\n", payload.replace("\ufeff", "").strip())
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip() and line.strip() != "WEBVTT"
        ]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timing_index == -1:
            continue
        start, end = [
            part.strip().split()[0] for part in lines[timing_index].split("-->", 1)
        ]
        text = " ".join(lines[timing_index + 1 :]).strip()
        if text:
            segments.append(
                TranscriptSegment(
                    f"segment:{len(segments)}",
                    text,
                    _parse_timestamp(start),
                    _parse_timestamp(end),
                )
            )
    return tuple(segments)


def _parse_timestamped_txt(payload: str) -> tuple[TranscriptSegment, ...]:
    segments: list[TranscriptSegment] = []
    for line in payload.splitlines():
        match = re.match(
            r"^\s*\[?([0-9:.;,]+)\s*(?:-->|-)\s*([0-9:.;,]+)\]?\s*(.+)$", line
        )
        if match:
            segments.append(
                TranscriptSegment(
                    f"segment:{len(segments)}",
                    match.group(3).strip(),
                    _parse_timestamp(match.group(1)),
                    _parse_timestamp(match.group(2)),
                )
            )
    return tuple(segments)


def _parse_timestamp(value: str) -> int:
    normalized = value.replace(",", ".").replace(";", ".")
    parts = normalized.split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) > 1 else 0
    hours = int(parts[-3]) if len(parts) > 2 else 0
    return int(round(((hours * 60 + minutes) * 60 + seconds) * 1000))


def _timeline_from_segments(
    segments: tuple[TranscriptSegment, ...],
    configuration: TranscriptConfiguration,
    fmt: TranscriptFormat,
) -> TranscriptTimeline:
    ordered = tuple(
        sorted(segments, key=lambda s: (s.start_ms, s.end_ms, s.segment_id))
    )
    sentences = tuple(
        TranscriptSentence(
            f"sentence:{i}",
            s.text,
            s.start_ms,
            s.end_ms,
            tuple(f"token:{i}:{j}" for j, _ in enumerate(s.text.split())),
        )
        for i, s in enumerate(ordered)
    )
    tokens = tuple(
        TranscriptToken(
            token_id,
            word,
            sentences[i].start_ms,
            sentences[i].end_ms,
            sentences[i].sentence_id,
        )
        for i, sentence in enumerate(sentences)
        for token_id, word in zip(
            sentence.token_ids, sentence.text.split(), strict=True
        )
    )
    paragraphs = (
        (
            TranscriptParagraph(
                "paragraph:0",
                tuple(s.sentence_id for s in sentences),
                sentences[0].start_ms,
                sentences[-1].end_ms,
            ),
        )
        if sentences
        else ()
    )
    metadata = TranscriptMetadata(
        TranscriptIdentifier(configuration.transcript_id, configuration.video_id),
        configuration.language,
        format=fmt,
    )
    return TranscriptTimeline(metadata, sentences, tokens, paragraphs, ordered)
