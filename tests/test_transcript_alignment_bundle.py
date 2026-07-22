from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from orderflowgpt_genesis import (
    AlignmentConfidence,
    DatasetVersion,
    FeatureVectorBuilder,
    FrameIdentifier,
    FrameMetadata,
    FrameTranscriptAlignment,
    TimelineSynchronizer,
    TrainingSampleBuilder,
    TranscriptAligner,
    TranscriptConfiguration,
    TranscriptIdentifier,
    TranscriptImporter,
    TranscriptSentence,
    TranscriptTimeline,
)
from orderflowgpt_genesis.vision import DetectionGraph


def config(video="video-1"):
    return TranscriptConfiguration("tx-1", video, "en", active_window_ms=100)


def frame(fid, seconds, video="video-1"):
    return FrameMetadata(
        FrameIdentifier(fid, sequence_number=int(seconds * 10)),
        datetime.fromtimestamp(seconds, tz=timezone.utc),
    )


def test_srt_parsing_metadata_and_ordering():
    text = "1\n00:00:00,000 --> 00:00:01,000\nFirst sentence.\n\n2\n00:00:02,000 --> 00:00:03,000\nSecond sentence.\n"
    timeline = TranscriptImporter().import_text(text, config(), "srt")
    assert [s.text for s in timeline.sentences] == [
        "First sentence.",
        "Second sentence.",
    ]
    assert timeline.metadata.identifier == TranscriptIdentifier("tx-1", "video-1")
    assert timeline.statistics().sentence_count == 2
    assert timeline.statistics().token_count == 4


def test_vtt_parsing():
    text = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nFabio explains delta.\n"
    timeline = TranscriptImporter().import_text(text, config(), "vtt")
    assert timeline.sentences[0].start_ms == 1000
    assert timeline.sentences[0].end_ms == 2000


def test_txt_and_json_parsing_are_deterministic():
    txt = "[00:00:00.000 --> 00:00:01.000] One\n00:00:01.000 - 00:00:02.000 Two"
    js = '{"segments":[{"start_ms":0,"end_ms":1000,"text":"One"},{"start_ms":1000,"end_ms":2000,"text":"Two"}]}'
    importer = TranscriptImporter()
    assert [s.text for s in importer.import_text(txt, config(), "txt").sentences] == [
        "One",
        "Two",
    ]
    assert [s.text for s in importer.import_text(js, config(), "json").sentences] == [
        "One",
        "Two",
    ]


def test_timestamp_ordering_and_duplicate_validation():
    a = TranscriptSentence("a", "A", 1000, 2000)
    b = TranscriptSentence("b", "B", 0, 500)
    with pytest.raises(ValueError, match="timestamp ordered"):
        TranscriptTimeline(
            TranscriptImporter().import_text("", config(), "txt").metadata, (a, b)
        )
    with pytest.raises(ValueError, match="unique"):
        TranscriptTimeline(
            TranscriptImporter().import_text("", config(), "txt").metadata, (a, a)
        )


def test_alignment_nearest_previous_next_and_active_window():
    timeline = TranscriptImporter().import_text(
        "00:00:00.000 --> 00:00:01.000 First\n00:00:02.000 --> 00:00:03.000 Second",
        config(),
        "txt",
    )
    dataset = TranscriptAligner(config()).align([frame("f1", 1.5)], timeline)
    alignment = dataset.alignments[0]
    assert alignment.nearest_sentence_id == "sentence:0"
    assert alignment.previous_sentence_id == "sentence:0"
    assert alignment.next_sentence_id == "sentence:1"
    assert alignment.active_sentence_ids == ()
    assert alignment.confidence is AlignmentConfidence.NEAREST


def test_single_sentence_small_and_large_transcripts():
    importer = TranscriptImporter()
    small = importer.import_text("00:00:00 --> 00:00:01 Only", config(), "txt")
    assert (
        TranscriptAligner(config())
        .align([frame("f1", 0.5)], small)
        .alignments[0]
        .nearest_sentence_id
        == "sentence:0"
    )
    large_text = "\n".join(
        f"00:00:{i:02d}.000 --> 00:00:{i+1:02d}.000 Line {i}" for i in range(50)
    )
    large = importer.import_text(large_text, config(), "txt")
    assert large.statistics().sentence_count == 50
    assert (
        TranscriptAligner(config())
        .align([frame("f49", 49.5)], large)
        .alignments[0]
        .nearest_sentence_id
        == "sentence:49"
    )


def test_multiple_videos_reference_validation_and_immutability():
    t1 = TranscriptImporter().import_text(
        "00:00:00 --> 00:00:01 One", config("v1"), "txt"
    )
    t2 = TranscriptImporter().import_text(
        "00:00:00 --> 00:00:01 Two", config("v2"), "txt"
    )
    assert t1.metadata.identifier.video_id == "v1"
    assert t2.metadata.identifier.video_id == "v2"
    with pytest.raises(FrozenInstanceError):
        t1.sentences[0].text = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="reference transcript sentences"):
        from orderflowgpt_genesis import TranscriptDataset

        TranscriptDataset(
            "bad",
            t1,
            (
                FrameTranscriptAlignment(
                    "f", "v1", 0, "missing", None, None, (), AlignmentConfidence.NEAREST
                ),
            ),
        )


def test_training_sample_integration_and_detection_graph_references():
    graph = DetectionGraph(
        "f1",
        transcript_references=("tx-1",),
        frame_transcript_references=("falign:f1",),
        transcript_alignment_references=("alignment:tx-1:sentence:0",),
    )
    sample = TrainingSampleBuilder().build(
        frame("f1", 0),
        FeatureVectorBuilder().build(graph),
        DatasetVersion(0, 1, 0),
        transcript_alignment_id="falign:f1",
    )
    assert sample.transcript_alignment_id == "falign:f1"
    assert graph.transcript_references == ("tx-1",)
    with pytest.raises(ValueError, match="unique"):
        DetectionGraph("f2", transcript_references=("tx", "tx"))


def test_timeline_synchronizer_sentence_alignment_references():
    timeline = TranscriptImporter().import_text(
        "00:00:00 --> 00:00:01 One", config(), "txt"
    )
    alignments = TimelineSynchronizer().synchronize(timeline)
    assert alignments[0].sentence_id == "sentence:0"
    assert alignments[0].confidence is AlignmentConfidence.EXACT
