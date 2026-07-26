"""Rule-based reflections for Apprentice experiences."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reflection:
    """A deterministic reflection generated after an experience."""

    id: str
    what_happened: str
    why_fabio_thought_this: str
    evidence_supported: tuple[str, ...]
    evidence_contradicted: tuple[str, ...]
    concept_demonstrated: str
    what_genesis_should_remember: str

    def __post_init__(self) -> None:
        required = (
            self.id,
            self.what_happened,
            self.why_fabio_thought_this,
            self.concept_demonstrated,
            self.what_genesis_should_remember,
        )
        if not all(value.strip() for value in required):
            raise ValueError("reflection core fields are required")
        for label, values in (
            ("supported evidence", self.evidence_supported),
            ("contradicted evidence", self.evidence_contradicted),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"reflection {label} cannot contain blank values")


def reflect_experience(
    experience_id: str,
    teacher_statement: str,
    evidence: tuple[str, ...],
    reasoning: str,
    outcome: str,
    concept_id: str,
) -> Reflection:
    """Create a rule-based reflection without LLM or predictive inference."""

    return Reflection(
        id=f"reflection:{experience_id}",
        what_happened=outcome,
        why_fabio_thought_this=teacher_statement or reasoning,
        evidence_supported=evidence,
        evidence_contradicted=(),
        concept_demonstrated=concept_id,
        what_genesis_should_remember=(
            f"Remember concept '{concept_id}' through Fabio's explanation and the recorded evidence."
        ),
    )
