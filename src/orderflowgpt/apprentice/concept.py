"""Concept models for the Genesis Apprentice Layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Concept:
    """A teachable idea that grows through deterministic experiences."""

    id: str
    name: str
    description: str = ""
    examples: tuple[str, ...] = ()
    counter_examples: tuple[str, ...] = ()
    related_concepts: tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("concept id and name are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("concept confidence must be between 0.0 and 1.0")
        for label, values in (
            ("examples", self.examples),
            ("counter examples", self.counter_examples),
            ("related concepts", self.related_concepts),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"concept {label} cannot contain blank values")
