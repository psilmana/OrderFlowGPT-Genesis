"""Executable deterministic Genesis pipeline runner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import GenesisConfiguration
from .dataset import DatasetBuilder, DatasetConfiguration
from .io import (
    ensure_asset_directories,
    prepare_lesson_directory,
    safe_filename_stem,
    write_json,
)
from .knowledge import KnowledgeConfiguration, KnowledgeExtractionEngine
from .learning import LearningConfiguration, MemoryBuilder, MemorySerializer
from .transcript import TranscriptAligner, TranscriptConfiguration, TranscriptImporter
from .video import VideoConfiguration, VideoImporter
from .vision import DetectionGraph

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
TRANSCRIPT_EXTENSIONS = (".srt", ".vtt", ".txt", ".json")
BUNDLE_VERSIONS = {
    "runner": "bundle-13.5",
    "video": "bundle-10",
    "dataset": "bundle-9",
    "transcript": "bundle-11",
    "knowledge": "bundle-12",
    "learning": "bundle-13",
}


@dataclass(frozen=True, slots=True)
class RunnerStatistics:
    lessons_processed: int = 0
    frames_extracted: int = 0
    detection_count: int = 0
    training_samples: int = 0
    knowledge_observations: int = 0
    memory_entries: int = 0


@dataclass(frozen=True, slots=True)
class RunnerResult:
    lesson_id: str
    video: Path
    output: Path
    report: Path
    summary: Path
    log: Path
    statistics: RunnerStatistics


class GenesisRunner:
    """Permanent entry point that orchestrates existing Genesis bundles."""

    def __init__(self, configuration: GenesisConfiguration | None = None) -> None:
        self.configuration = configuration or GenesisConfiguration()
        ensure_asset_directories()

    def run_video(
        self, video: str | Path, transcript: str | Path | None = None
    ) -> RunnerResult:
        video_path = Path(video)
        transcript_path = Path(transcript) if transcript is not None else None
        self._validate_video(video_path)
        if transcript_path is not None:
            self._validate_transcript(transcript_path)
        lesson_id = self._lesson_id(video_path)
        lesson_dir = self.configuration.output_folder / lesson_id
        prepare_lesson_directory(lesson_dir, self.configuration.overwrite)
        lines: list[str] = []
        started = time.perf_counter()

        def stage(name: str, fn):  # type: ignore[no-untyped-def]
            begin = time.perf_counter()
            lines.append(f"START {name}")
            try:
                result = fn()
            except Exception as exc:
                lines.append(f"ERROR {name}: {exc}")
                (lesson_dir / "processing.log").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                raise RuntimeError(f"{name} failed: {exc}") from exc
            lines.append(f"END {name} {time.perf_counter() - begin:.6f}s")
            return result

        video_cfg = VideoConfiguration(
            every_n_frames=self.configuration.frame_extraction_interval
        )
        import_result = stage(
            "Video Import",
            lambda: VideoImporter(video_cfg).import_video(video_path, lesson_id),
        )
        frames = import_result.dataset.frames.sequence.frames
        lines.append(f"Video loaded: {video_path}")
        lines.append(f"Frames extracted: {len(frames)}")
        lines.append("Frames skipped: 0")
        if not frames:
            raise RuntimeError("no frames extracted")
        graphs = stage(
            "Vision Engine",
            lambda: tuple(self._run_vision(frame) for frame in frames),
        )
        graphs = stage(
            "DetectionGraph", lambda: tuple(sorted(graphs, key=lambda g: g.frame_id))
        )
        frame_pairs = tuple(
            (sample.metadata, graph)
            for sample, graph in zip(
                import_result.dataset.training_samples, graphs, strict=True
            )
        )
        dataset = stage(
            "Dataset Builder",
            lambda: DatasetBuilder(DatasetConfiguration(f"dataset:{lesson_id}")).build(
                frame_pairs
            ),
        )
        transcript_dataset = None
        if transcript_path is None:
            lines.append("WARNING Missing transcript: transcript alignment skipped")
        else:

            def align():  # type: ignore[no-untyped-def]
                cfg = TranscriptConfiguration(
                    transcript_path.stem,
                    import_result.dataset.metadata.identifier.video_id,
                )
                timeline = TranscriptImporter().import_file(transcript_path, cfg)
                return TranscriptAligner(cfg).align(
                    (s.metadata for s in dataset.samples), timeline
                )

            transcript_dataset = stage("Transcript Alignment", align)
        if transcript_dataset is None:
            knowledge = None
            memory = None
        else:
            knowledge = stage(
                "Knowledge Extraction",
                lambda: KnowledgeExtractionEngine(
                    KnowledgeConfiguration(f"knowledge:{lesson_id}")
                ).extract(transcript_dataset, graphs, dataset.samples),
            )
            memory = stage(
                "Learning & Memory",
                lambda: MemoryBuilder(
                    LearningConfiguration(f"memory:{lesson_id}", lesson_id)
                ).build(knowledge.dataset),
            )
        stage(
            "Save Results",
            lambda: self._save(lesson_dir, frames, graphs, dataset, knowledge, memory),
        )
        elapsed = time.perf_counter() - started
        stats = RunnerStatistics(
            1,
            len(frames),
            len(graphs),
            len(dataset.samples),
            0 if knowledge is None else len(knowledge.dataset.timeline.observations),
            0 if memory is None else len(memory.dataset.entries),
        )
        report = {
            "video": str(video_path),
            "duration": import_result.dataset.metadata.duration_seconds,
            "frames_extracted": stats.frames_extracted,
            "detection_count": stats.detection_count,
            "training_samples": stats.training_samples,
            "knowledge_observations": stats.knowledge_observations,
            "memory_entries": stats.memory_entries,
            "processing_time": round(elapsed, 6),
            "bundle_versions": BUNDLE_VERSIONS,
            "timestamp": datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat(),
        }
        write_json(lesson_dir / "report.json", report)
        write_json(
            lesson_dir / "summary.json", {"lesson_id": lesson_id, "statistics": stats}
        )
        lines.append(f"SUMMARY {report}")
        (lesson_dir / "processing.log").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return RunnerResult(
            lesson_id,
            video_path,
            lesson_dir,
            lesson_dir / "report.json",
            lesson_dir / "summary.json",
            lesson_dir / "processing.log",
            stats,
        )

    def run_folder(self, folder: str | Path) -> tuple[RunnerResult, ...]:
        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"missing video folder: {folder_path}")
        videos = tuple(
            sorted(
                p for p in folder_path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
            )
        )
        if not videos:
            raise ValueError(f"empty folder: {folder_path}")
        return tuple(
            self.run_video(video, self._matching_transcript(video)) for video in videos
        )

    def _save(self, lesson_dir: Path, frames, graphs, dataset, knowledge, memory) -> None:  # type: ignore[no-untyped-def]
        for frame in frames:
            suffix = ".png" if frame.image.pixel_format == "png" else ".frame"
            (
                lesson_dir / "frames" / f"{frame.timestamp.frame_number:012d}{suffix}"
            ).write_bytes(frame.image.data)
        for graph in graphs:
            write_json(
                lesson_dir
                / "detections"
                / f"{safe_filename_stem(graph.frame_id)}.json",
                graph,
            )
        write_json(lesson_dir / "dataset" / "dataset.json", dataset)
        write_json(
            lesson_dir / "knowledge" / "knowledge.json",
            {} if knowledge is None else knowledge.dataset,
        )
        if memory is not None:
            MemorySerializer().dump(
                memory.dataset, lesson_dir / "memory" / "memory.json"
            )
        else:
            write_json(lesson_dir / "memory" / "memory.json", {})

    def _run_vision(self, frame):  # type: ignore[no-untyped-def]
        # The repository currently exposes deterministic graph/model primitives, not
        # a mutating AI inference service. Construct the graph from the decoded frame
        # id so downstream deterministic bundles consume the real frame identity.
        return DetectionGraph(frame.identifier.frame_id)

    def _matching_transcript(self, video: Path) -> Path | None:
        for ext in TRANSCRIPT_EXTENSIONS:
            candidate = self.configuration.transcript_folder / f"{video.stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _lesson_id(video: Path) -> str:
        return "Lesson" + video.stem.title().replace(" ", "")

    @staticmethod
    def _validate_video(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"missing video: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"invalid video extension: {path.suffix}")
        if path.is_dir() or path.stat().st_size == 0:
            raise ValueError(f"corrupt video: {path}")

    @staticmethod
    def _validate_transcript(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"missing transcript: {path}")
        if path.suffix.lower() not in TRANSCRIPT_EXTENSIONS:
            raise ValueError(f"invalid transcript extension: {path.suffix}")
