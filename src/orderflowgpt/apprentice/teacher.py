"""Teacher model for Fabio in the Genesis Apprentice Layer."""

from __future__ import annotations

from dataclasses import dataclass

from .concept import Concept


@dataclass(frozen=True, slots=True)
class Teacher:
    """A deterministic representation of Fabio as teacher, not predictor."""

    id: str
    name: str
    concepts: tuple[Concept, ...] = ()
    lessons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("teacher id and name are required")
        if any(not lesson.strip() for lesson in self.lessons):
            raise ValueError("teacher lesson ids cannot be blank")
