"""Deterministic filesystem helpers for the Genesis runner."""

from __future__ import annotations

import json
import shutil
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

_WINDOWS_RESERVED_FILENAME_CHARS = frozenset('<>:"/\\|?*')


def safe_filename_stem(value: str) -> str:
    """Return a deterministic filename stem that is valid on Windows and POSIX."""
    safe = "".join(
        "_" if char in _WINDOWS_RESERVED_FILENAME_CHARS or ord(char) < 32 else char
        for char in value
    ).strip(" .")
    return safe or "unnamed"


ASSET_DIRECTORIES = (
    Path("assets"),
    Path("assets/fabio"),
    Path("assets/fabio/videos"),
    Path("assets/fabio/transcripts"),
    Path("assets/fabio/output"),
    Path("assets/fabio/output/frames"),
    Path("assets/fabio/output/datasets"),
    Path("assets/fabio/output/memory"),
    Path("assets/fabio/output/logs"),
)


def ensure_asset_directories(root: Path = Path(".")) -> None:
    for directory in ASSET_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)


def prepare_lesson_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    for child in ("frames", "detections", "dataset", "knowledge", "memory"):
        (path / child).mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple | list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
