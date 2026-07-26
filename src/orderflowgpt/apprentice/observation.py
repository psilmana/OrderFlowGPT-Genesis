"""Observation model for educational evidence without conclusions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Observation:
    """A timestamped educational unit that records evidence only."""

    id: str
    timestamp_ms: int
    frame_reference: str
    transcript_reference: str
    visual_evidence: tuple[str, ...]
    teacher_statement: str
    market_context: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("observation timestamp cannot be negative")
        required = (self.id, self.frame_reference, self.transcript_reference)
        if not all(value.strip() for value in required):
            raise ValueError(
                "observation id, frame, and transcript references are required"
            )
        if any(not value.strip() for value in self.visual_evidence):
            raise ValueError("visual evidence cannot contain blank values")
        if any(
            not key.strip() or not value.strip()
            for key, value in self.market_context.items()
        ):
            raise ValueError("market context keys and values cannot be blank")
        object.__setattr__(
            self,
            "market_context",
            MappingProxyType(dict(sorted(self.market_context.items()))),
        )
