from dataclasses import FrozenInstanceError

import pytest

from orderflowgpt_genesis.video import (
    AudioExtractor,
    FrameAudioSynchronizer,
    FrameExtractor,
    FrameSampler,
    FrameSamplingMode,
    SynchronizationMap,
    VideoConfiguration,
    VideoDecoder,
    VideoFrame,
    VideoImporter,
)


def test_video_metadata_is_deterministic(tmp_path):
    source = tmp_path / "FabioVideo.mp4"
    source.write_bytes(b"fabio")
    config = VideoConfiguration(synthetic_frame_count=60, source_fps=30)
    first = VideoDecoder().decode(source, config)
    second = VideoDecoder().decode(source, config)
    assert first == second
    assert first.duration_seconds == 2
    assert first.identifier.video_id.startswith("video:")


def test_frame_extraction_numbering_and_ordering(tmp_path):
    source = tmp_path / "FabioVideo.mp4"
    source.write_bytes(b"frames")
    config = VideoConfiguration(synthetic_frame_count=10, every_n_frames=2)
    metadata = VideoDecoder().decode(source, config)
    collection = FrameExtractor().extract(metadata, config)
    assert [f.timestamp.frame_number for f in collection.sequence.frames] == [
        0,
        2,
        4,
        6,
        8,
    ]
    assert (
        collection.sequence.frames[0].identifier.frame_id
        < collection.sequence.frames[-1].identifier.frame_id
    )


def test_timestamp_ordering(tmp_path):
    result = VideoImporter(VideoConfiguration(synthetic_frame_count=4)).import_video(
        tmp_path / "missing.mp4"
    )
    offsets = [
        p.timestamp.offset_seconds for p in result.dataset.synchronization.points
    ]
    assert offsets == sorted(offsets)


def test_audio_mapping_and_synchronization(tmp_path):
    result = VideoImporter(
        VideoConfiguration(
            synthetic_frame_count=61, source_fps=30, audio_segment_seconds=1
        )
    ).import_video(tmp_path / "a.mp4")
    assert len(result.dataset.audio.segments) == 3
    assert all(
        point.audio_segment_id for point in result.dataset.synchronization.points
    )
    assert all(
        point.training_sample_id for point in result.dataset.synchronization.points
    )


def test_large_small_and_single_frame_videos(tmp_path):
    assert (
        len(
            VideoImporter(VideoConfiguration(synthetic_frame_count=1))
            .import_video(tmp_path / "one.mp4")
            .dataset.frames.sequence.frames
        )
        == 1
    )
    assert (
        len(
            VideoImporter(VideoConfiguration(synthetic_frame_count=2))
            .import_video(tmp_path / "two.mp4")
            .dataset.frames.sequence.frames
        )
        == 2
    )
    assert (
        len(
            VideoImporter(
                VideoConfiguration(synthetic_frame_count=1000, every_n_frames=100)
            )
            .import_video(tmp_path / "large.mp4")
            .dataset.frames.sequence.frames
        )
        == 10
    )


def test_multiple_videos_have_distinct_references(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    importer = VideoImporter(VideoConfiguration(synthetic_frame_count=1))
    assert (
        importer.import_video(a).dataset.metadata.identifier
        != importer.import_video(b).dataset.metadata.identifier
    )


def test_dataset_and_training_sample_integration(tmp_path):
    result = VideoImporter(VideoConfiguration(synthetic_frame_count=3)).import_video(
        tmp_path / "dataset.mp4"
    )
    assert len(result.dataset.training_samples) == 3
    assert [
        s.metadata.identifier.frame_id for s in result.dataset.training_samples
    ] == [p.frame_id for p in result.dataset.synchronization.points]
    assert result.dataset.statistics().training_sample_count == 3


def test_immutability(tmp_path):
    metadata = (
        VideoImporter(VideoConfiguration())
        .import_video(tmp_path / "immutable.mp4")
        .dataset.metadata
    )
    with pytest.raises(FrozenInstanceError):
        metadata.frame_count = 99  # type: ignore[misc]


def test_duplicate_validation_and_reference_validation(tmp_path):
    result = VideoImporter(VideoConfiguration(synthetic_frame_count=2)).import_video(
        tmp_path / "dup.mp4"
    )
    points = result.dataset.synchronization.points
    with pytest.raises(ValueError):
        SynchronizationMap(result.dataset.metadata.identifier, (points[0], points[0]))
    frame = result.dataset.frames.sequence.frames[0]
    with pytest.raises(ValueError):
        VideoFrame(
            frame.identifier,
            result.dataset.frames.sequence.frames[1].timestamp,
            frame.image,
            frame.content_hash,
        )


def test_sampling_modes(tmp_path):
    metadata = VideoDecoder().decode(
        tmp_path / "sampler.mp4",
        VideoConfiguration(synthetic_frame_count=90, source_fps=30),
    )
    assert FrameSampler().sample(
        metadata,
        VideoConfiguration(
            synthetic_frame_count=90,
            source_fps=30,
            sampling_mode=FrameSamplingMode.FPS,
            target_fps=10,
        ),
    ) == tuple(range(0, 90, 3))
    assert FrameSampler().sample(
        metadata,
        VideoConfiguration(
            synthetic_frame_count=90,
            sampling_mode=FrameSamplingMode.KEYFRAMES,
            keyframe_interval=30,
        ),
    ) == (0, 30, 60)


def test_audio_extractor_metadata(tmp_path):
    config = VideoConfiguration(synthetic_frame_count=30, source_fps=30)
    metadata = VideoDecoder().decode(tmp_path / "audio.mp4", config)
    audio = AudioExtractor().extract(metadata, config)
    assert audio.metadata.duration_seconds == 1
    assert audio.segments[0].start_seconds == 0


def test_manual_synchronizer_without_samples(tmp_path):
    config = VideoConfiguration(synthetic_frame_count=3)
    metadata = VideoDecoder().decode(tmp_path / "sync.mp4", config)
    frames = FrameExtractor().extract(metadata, config)
    audio = AudioExtractor().extract(metadata, config)
    sync = FrameAudioSynchronizer().synchronize(metadata.identifier, frames, audio)
    assert [p.training_sample_id for p in sync.points] == [None, None, None]
