from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from orderflowgpt_genesis import GenesisCLI, GenesisConfiguration, GenesisRunner
from orderflowgpt_genesis.io import safe_filename_stem


def make_video(tmp_path, name="lesson01.mp4"):
    path = tmp_path / name
    path.write_bytes(b"deterministic video")
    return path


def make_transcript(tmp_path, name="lesson01.txt"):
    path = tmp_path / name
    path.write_text(
        "[00:00:00.000] deterministic teaching observation", encoding="utf-8"
    )
    return path


def config(tmp_path, overwrite=False):
    return GenesisConfiguration(
        video_folder=tmp_path / "videos",
        transcript_folder=tmp_path / "transcripts",
        output_folder=tmp_path / "output",
        frame_extraction_interval=1,
        overwrite=overwrite,
    )


def test_single_video_pipeline_report_logging_and_outputs(tmp_path):
    video = make_video(tmp_path)
    transcript = make_transcript(tmp_path)
    result = GenesisRunner(config(tmp_path)).run_video(video, transcript)
    assert result.statistics.lessons_processed == 1
    assert result.statistics.frames_extracted == 1
    assert result.statistics.training_samples == 1
    assert result.report.exists()
    assert (result.output / "frames").is_dir()
    assert (result.output / "detections").is_dir()
    assert (result.output / "dataset" / "dataset.json").exists()
    assert (result.output / "knowledge" / "knowledge.json").exists()
    assert (result.output / "memory" / "memory.json").exists()
    report = json.loads(result.report.read_text(encoding="utf-8"))
    assert report["bundle_versions"]["runner"] == "bundle-13.5"
    assert report["timestamp"] == "1970-01-01T00:00:00+00:00"
    assert "START Video Import" in result.log.read_text(encoding="utf-8")


def test_missing_transcript_is_warning_not_failure(tmp_path):
    result = GenesisRunner(config(tmp_path)).run_video(make_video(tmp_path))
    assert result.statistics.knowledge_observations == 0
    assert "WARNING Missing transcript" in result.log.read_text(encoding="utf-8")


def test_detection_artifact_names_are_cross_platform_safe(tmp_path):
    result = GenesisRunner(config(tmp_path)).run_video(make_video(tmp_path))
    detection_files = tuple((result.output / "detections").glob("*.json"))
    assert len(detection_files) == result.statistics.frames_extracted
    assert detection_files[0].name == (
        "video_2efba8d2c67eb939_frame_000000000000_910abc823cde4c7d.json"
    )
    assert safe_filename_stem("video:abc/frame:0?x*") == "video_abc_frame_0_x_"


def test_missing_video_and_invalid_file_errors(tmp_path):
    runner = GenesisRunner(config(tmp_path))
    with pytest.raises(FileNotFoundError, match="missing video"):
        runner.run_video(tmp_path / "missing.mp4")
    invalid = tmp_path / "bad.txt"
    invalid.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid video extension"):
        runner.run_video(invalid)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="corrupt video"):
        runner.run_video(empty)


def test_overwrite_guard(tmp_path):
    video = make_video(tmp_path)
    GenesisRunner(config(tmp_path)).run_video(video)
    with pytest.raises(FileExistsError):
        GenesisRunner(config(tmp_path)).run_video(video)
    assert (
        GenesisRunner(config(tmp_path, overwrite=True)).run_video(video).report.exists()
    )


def test_folder_processing_ordering_and_determinism(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    make_video(folder, "b.mp4")
    make_video(folder, "a.mp4")
    cfg = config(tmp_path, overwrite=True)
    runner = GenesisRunner(cfg)
    first = runner.run_folder(folder)
    second = runner.run_folder(folder)
    assert [r.video.name for r in first] == ["a.mp4", "b.mp4"]
    assert [
        json.loads(r.report.read_text(encoding="utf-8"))["timestamp"] for r in first
    ] == [
        "1970-01-01T00:00:00+00:00",
        "1970-01-01T00:00:00+00:00",
    ]
    assert [r.lesson_id for r in first] == [r.lesson_id for r in second]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty folder"):
        runner.run_folder(empty)


def test_configuration_immutability_and_cli_parsing(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(FrozenInstanceError):
        cfg.overwrite = True  # type: ignore[misc]
    args = GenesisCLI().parse(
        ["--video", "x.mp4", "--output", str(tmp_path), "--overwrite", "--verbose"]
    )
    assert args.video.name == "x.mp4"
    assert args.output == tmp_path
    assert args.overwrite is True


def test_transcript_integrates_through_knowledge_and_memory(tmp_path):
    video = make_video(tmp_path)
    transcript = make_transcript(tmp_path)
    result = GenesisRunner(config(tmp_path)).run_video(video, transcript)

    report = json.loads(result.report.read_text(encoding="utf-8"))
    assert report["knowledge_observations"] == 1
    assert report["memory_entries"] == 1

    dataset = json.loads(
        (result.output / "dataset" / "dataset.json").read_text(encoding="utf-8")
    )
    sample = dataset["samples"][0]
    assert sample["transcript_alignment_id"].startswith("falign:")
    assert sample["transcript_references"] == ["sentence:0"]
    assert sample["transcript_text"] == ["deterministic teaching observation"]

    detections = tuple((result.output / "detections").glob("*.json"))
    detection = json.loads(detections[0].read_text(encoding="utf-8"))
    assert detection["frame_transcript_references"] == [
        sample["transcript_alignment_id"]
    ]

    knowledge = json.loads(
        (result.output / "knowledge" / "knowledge.json").read_text(encoding="utf-8")
    )
    assert len(knowledge["timeline"]["observations"]) == 1

    memory = json.loads(
        (result.output / "memory" / "memory.json").read_text(encoding="utf-8")
    )
    assert len(memory["entries"]) == 1
    assert memory["entries"][0]["transcript"] == "deterministic teaching observation"

    log = result.log.read_text(encoding="utf-8")
    assert "Transcript loaded:" in log
    assert "Transcript segments: 1" in log
    assert "Frames matched: 1" in log
    assert "Knowledge observations created: 1" in log
    assert "Memory entries created: 1" in log


def test_timestamp_mismatch_is_deterministic_nearest_match(tmp_path):
    video = make_video(tmp_path)
    transcript = tmp_path / "lesson01.txt"
    transcript.write_text("[00:10:00.000] far away transcript", encoding="utf-8")
    result = GenesisRunner(config(tmp_path)).run_video(video, transcript)
    assert result.statistics.knowledge_observations == 1
    dataset = json.loads(
        (result.output / "dataset" / "dataset.json").read_text(encoding="utf-8")
    )
    assert dataset["samples"][0]["transcript_text"] == ["far away transcript"]
