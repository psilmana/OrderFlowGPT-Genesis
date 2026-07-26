from dataclasses import FrozenInstanceError
import json

import pytest

from orderflowgpt.apprentice import (
    Concept,
    Experience,
    LearningSession,
    Lesson,
    Observation,
    Teacher,
)


def _teacher() -> Teacher:
    return Teacher(
        "teacher:fabio", "Fabio", (Concept("concept:poc", "POC"),), ("lesson-01",)
    )


def _lesson() -> Lesson:
    return Lesson(
        id="lesson-01",
        video_id="video-01",
        transcript="Fabio explains what he sees.",
        frames=("frame-001",),
        concepts_introduced=("concept:poc",),
        summary="A first apprentice lesson.",
        teacher_comments=("Look at the evidence first.",),
    )


def _observation(timestamp: int = 1000) -> Observation:
    return Observation(
        id="observation-001",
        timestamp_ms=timestamp,
        frame_reference="frame-001",
        transcript_reference="transcript-001",
        visual_evidence=("visible footprint imbalance", "highlighted price area"),
        teacher_statement="Fabio explains the visible evidence.",
        market_context={"instrument": "ES", "session": "morning"},
    )


def test_teacher_lesson_concept_and_observation_are_immutable_learning_models():
    teacher = _teacher()
    lesson = _lesson()
    observation = _observation()

    assert teacher.name == "Fabio"
    assert lesson.video_id == "video-01"
    assert observation.market_context["instrument"] == "ES"
    with pytest.raises(FrozenInstanceError):
        teacher.name = "Someone else"  # type: ignore[misc]
    with pytest.raises(TypeError):
        observation.market_context["instrument"] = "NQ"  # type: ignore[index]


def test_experience_has_stable_id_and_rule_based_reflection():
    experience = Experience.from_observation(_observation(), "lesson-01", "concept:poc")

    assert experience.id == "experience:lesson-01:observation-001"
    assert experience.teacher_explanation == "Fabio explains the visible evidence."
    assert experience.reflection.id == "reflection:experience:lesson-01:observation-001"
    assert experience.reflection.evidence_contradicted == ()
    assert experience.reflection.concept_demonstrated == "concept:poc"


def test_learning_session_outputs_deterministic_experience_json(tmp_path):
    second = Observation(
        id="observation-000",
        timestamp_ms=500,
        frame_reference="frame-000",
        transcript_reference="transcript-000",
        visual_evidence=("prior visual evidence",),
        teacher_statement="Fabio starts with what is visible.",
        market_context={"instrument": "ES"},
    )
    bundle = LearningSession(
        _teacher(), _lesson(), (_observation(), second)
    ).build_experience_bundle()

    assert [observation.id for observation in bundle.observations] == [
        "observation-000",
        "observation-001",
    ]
    assert (
        bundle.summary
        == "I observed 2 item(s). Fabio explained them. I stored 2 experience(s). I reflected on them."
    )
    output = bundle.dump(tmp_path)
    assert output.name == "experience.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert tuple(payload) == (
        "experiences",
        "lesson",
        "observations",
        "reflections",
        "summary",
        "teacher",
    )
    assert (
        payload["experiences"][0]["reflection"]["concept_demonstrated"] == "concept:poc"
    )


def test_validation_rejects_blank_and_predictive_free_models():
    with pytest.raises(ValueError):
        Concept(" ", "POC")
    with pytest.raises(ValueError):
        Observation("o", -1, "f", "t", (), "", {})
    with pytest.raises(ValueError):
        Experience.from_observation(_observation(), " ", "concept:poc")
