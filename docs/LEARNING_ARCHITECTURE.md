# Learning Architecture

Genesis Learning Architecture v1 (GLA v1) reorganises the system around apprenticeship learning.

The primary flow is:

```text
Teacher
  ↓
Lesson
  ↓
Observation
  ↓
Reasoning
  ↓
Decision
  ↓
Outcome
  ↓
Reflection
  ↓
Experience
  ↓
Memory
  ↓
Knowledge
```

## Architectural Layers

### 1. Teacher Layer

The Teacher Layer represents Fabio Valentini's methodology, lesson narration, corrections, examples, and discretionary judgement.

It supplies:

- Lessons.
- Commentary.
- Corrections.
- Trade reviews.
- Methodology principles.
- Examples of expert reasoning.

### 2. Student Layer

The Student Layer represents Genesis as an apprentice.

It is responsible for:

- Observing lessons.
- Asking what matters.
- Forming tentative interpretations.
- Comparing reasoning against Fabio's explanation.
- Identifying gaps.
- Improving through reflection.

### 3. Perception Layer

The Perception Layer contains existing operational modules:

- Vision.
- Video.
- Transcript.
- Timeline.
- Dataset.

These modules convert raw lesson material into observations and evidence. They are not removed, renamed, or treated as obsolete. They continue to serve the system by producing inputs for learning.

### 4. Experience Layer

The Experience Layer is the centre of GLA v1.

An Experience captures:

- Observation: what Genesis perceived.
- Reasoning: how Genesis interpreted the situation.
- Decision: what action, bias, or judgement was considered.
- Outcome: what happened next.
- Reflection: what was learned.

### 5. Memory Layer

The Memory Layer stores learning over time.

It should preserve:

- Lessons.
- Experiences.
- Corrections.
- Mistakes.
- Repeated market patterns.
- Contextual cues.
- Reflection notes.
- Knowledge objects derived from repeated experience.

### 6. Reasoning Layer

The Reasoning Layer follows Fabio's hierarchy:

1. Market State.
2. Location.
3. Aggression.
4. Risk.
5. Management.
6. Reflection.

This hierarchy ensures Genesis does not jump directly from detection to prediction. It should first explain context, location, behaviour, risk, management logic, and lessons learned.

### 7. Knowledge Layer

The Knowledge Layer is emergent. Knowledge is not the first product of the system; it is the refined product of experience, memory, and reflection.

Knowledge objects should be organised by Fabio's reasoning hierarchy rather than by raw detector output alone.

## Non-Functional Architecture Commitments

GLA v1 is additive and non-breaking:

- No runtime behaviour changes.
- No current JSON schema changes.
- No package renames.
- No dataset removal.
- No memory removal.
- No existing bundle rewrites.
- No trading automation.
