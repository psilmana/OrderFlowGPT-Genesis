from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from orderflowgpt_genesis import GenesisCLI, GenesisConfiguration, GenesisRunner


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
