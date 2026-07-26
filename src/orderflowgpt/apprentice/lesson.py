"""Lesson model for one Fabio educational unit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Lesson:
    """One educational unit in the Apprentice Layer."""

    id: str
    video_id: str
    transcript: str
    frames: tuple[str, ...] = ()
    concepts_introduced: tuple[str, ...] = ()
    summary: str = ""
    teacher_comments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.video_id.strip():
            raise ValueError("lesson id and video id are required")
        for label, values in (
            ("frames", self.frames),
            ("concepts", self.concepts_introduced),
            ("teacher comments", self.teacher_comments),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"lesson {label} cannot contain blank values")
