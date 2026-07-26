"""Learning session orchestration for the Apprentice Layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .experience import Experience
from .lesson import Lesson
from .observation import Observation
from .reflection import Reflection
from .teacher import Teacher


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, MappingProxyType):
        return dict(value)
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExperienceBundle:
    """Serializable Apprentice output bundle."""

    lesson: Lesson
    teacher: Teacher
    observations: tuple[Observation, ...]
    experiences: tuple[Experience, ...]
    reflections: tuple[Reflection, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible dictionary."""

        return _jsonable(self)

    def dump(self, path: Path) -> Path:
        """Write experience.json without replacing existing Genesis bundles."""

        target = path if path.name == "experience.json" else path / "experience.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


@dataclass(frozen=True, slots=True)
class LearningSession:
    """Orchestrates teacher, lesson, observations, experiences, and reflections."""

    teacher: Teacher
    lesson: Lesson
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("learning session requires at least one observation")

    def build_experience_bundle(self) -> ExperienceBundle:
        """Convert Fabio teaching observations into deterministic experiences."""

        default_concept = (
            self.lesson.concepts_introduced[0]
            if self.lesson.concepts_introduced
            else "concept:unassigned"
        )
        experiences = tuple(
            Experience.from_observation(observation, self.lesson.id, default_concept)
            for observation in sorted(
                self.observations, key=lambda item: (item.timestamp_ms, item.id)
            )
        )
        reflections = tuple(experience.reflection for experience in experiences)
        summary = (
            f"I observed {len(self.observations)} item(s). Fabio explained them. "
            f"I stored {len(experiences)} experience(s). I reflected on them."
        )
        return ExperienceBundle(
            lesson=self.lesson,
            teacher=self.teacher,
            observations=tuple(experience.observation for experience in experiences),
            experiences=experiences,
            reflections=reflections,
            summary=summary,
        )
