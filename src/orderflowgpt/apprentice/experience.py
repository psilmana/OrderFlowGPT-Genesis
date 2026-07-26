"""Experience model for the Genesis Apprentice Layer."""

from __future__ import annotations

from dataclasses import dataclass

from .observation import Observation
from .reflection import Reflection, reflect_experience


@dataclass(frozen=True, slots=True)
class Experience:
    """Observation plus Fabio explanation, evidence, reasoning, outcome, reflection."""

    id: str
    observation: Observation
    teacher_explanation: str
    evidence: tuple[str, ...]
    reasoning: str
    outcome: str
    concept_id: str
    reflection: Reflection

    def __post_init__(self) -> None:
        required = (
            self.id,
            self.teacher_explanation,
            self.reasoning,
            self.outcome,
            self.concept_id,
        )
        if not all(value.strip() for value in required):
            raise ValueError("experience core fields are required")
        if any(not value.strip() for value in self.evidence):
            raise ValueError("experience evidence cannot contain blank values")
        if self.reflection.id != f"reflection:{self.id}":
            raise ValueError(
                "experience reflection id must be derived from experience id"
            )

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        lesson_id: str,
        concept_id: str,
        reasoning: str = "Fabio connected the statement to the recorded evidence.",
        outcome: str = "Genesis stored the taught observation as an experience.",
    ) -> "Experience":
        """Build a stable, deterministic experience from an observation."""

        if not lesson_id.strip():
            raise ValueError("lesson id is required")
        experience_id = f"experience:{lesson_id}:{observation.id}"
        evidence = observation.visual_evidence
        return cls(
            id=experience_id,
            observation=observation,
            teacher_explanation=observation.teacher_statement,
            evidence=evidence,
            reasoning=reasoning,
            outcome=outcome,
            concept_id=concept_id,
            reflection=reflect_experience(
                experience_id,
                observation.teacher_statement,
                evidence,
                reasoning,
                outcome,
                concept_id,
            ),
        )
