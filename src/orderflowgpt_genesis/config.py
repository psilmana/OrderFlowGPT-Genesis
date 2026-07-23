"""Immutable configuration for the Genesis executable runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GenesisConfiguration:
    """Configuration for deterministic Genesis pipeline execution."""

    video_folder: Path = Path("assets/fabio/videos")
    transcript_folder: Path = Path("assets/fabio/transcripts")
    output_folder: Path = Path("assets/fabio/output")
    frame_extraction_interval: int = 1
    logging: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_folder", Path(self.video_folder))
        object.__setattr__(self, "transcript_folder", Path(self.transcript_folder))
        object.__setattr__(self, "output_folder", Path(self.output_folder))
        if self.frame_extraction_interval <= 0:
            raise ValueError("frame extraction interval must be positive")
