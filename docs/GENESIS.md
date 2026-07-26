# Genesis Learning Architecture (GLA v1)

Genesis is evolving from a video-processing pipeline into an AI Apprentice learning system.

This document defines the top-level architecture for Genesis Learning Architecture v1 (GLA v1). It is documentation-only and does not change runtime behaviour, existing modules, package names, JSON schemas, or public APIs.

## Architectural Reframe

### Previous System Model

```text
Video
  ↓
Frames
  ↓
OCR
  ↓
Knowledge
```

The previous model treated extracted knowledge as an early product of perception. Video and frame processing were central, and learning artifacts were downstream of OCR, timeline alignment, dataset construction, and knowledge extraction.

### New System Model

```text
Teacher
  ↓
Lesson
  ↓
Experience
  ↓
Reflection
  ↓
Memory
  ↓
Knowledge
```

GLA v1 makes knowledge an emergent product of learning. Genesis should learn from Fabio Valentini's discretionary Order Flow methodology through lessons, observations, reasoning, outcomes, reflection, and memory.

## Genesis Is an AI Apprentice

Genesis is not a trading bot, indicator, or automated execution system. Genesis is an AI Apprentice whose purpose is to learn, explain, remember, reflect, and develop discretionary Order Flow intuition.

Genesis should:

1. Learn from Fabio.
2. Develop intuition through repeated experiences.
3. Remember contextual market episodes.
4. Reflect after every lesson.
5. Explain before predicting.
6. Treat knowledge as something earned through learning, not as raw extraction output.

## Core Architecture

GLA v1 consists of the following conceptual components:

| Component | Role |
| --- | --- |
| Teacher | Source of methodology, lessons, corrections, and discretionary judgement. |
| Student | Apprentice identity that observes, asks, practices, and improves. |
| Brain | Coordinates reasoning, reflection, memory access, and knowledge emergence. |
| Experience | Central learning unit built from observation, reasoning, decision, outcome, and reflection. |
| Memory | Persistent store of lessons, experiences, mistakes, corrections, and learned patterns. |
| Reasoning | Structured interpretation process following Fabio's Order Flow hierarchy. |
| Reflection | Post-lesson and post-outcome review that turns experience into learning. |
| Vision | Perception capability for chart, footprint, DOM, and visual market evidence. |
| Audio | Perception capability for teacher commentary and lesson narration. |
| Live Coach | Future guidance mode where Genesis assists by explaining context and reasoning. |

## Perception Layer

Existing modules remain operational and are now documented as the Perception Layer rather than the Brain.

The Perception Layer includes:

- Vision
- Video
- Transcript
- Timeline
- Dataset

These modules continue to provide observations, inputs, evidence, alignment, and structured source material. They do not define the apprentice's intelligence by themselves.

## Experience-Centred Learning

Genesis should revolve around Experience instead of Frames.

### Old Unit of Progress

```text
Frame
  ↓
Detection
  ↓
Knowledge
```

### New Unit of Progress

```text
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
```

An Experience represents a complete learning episode. It is not merely a detected chart state; it contains what Genesis saw, how it reasoned, what decision was implied or discussed, what happened next, and what reflection changed in memory.

## Learning Hierarchy

Genesis should reason using Fabio's playbook as the architectural principle:

1. Market State
2. Location
3. Aggression
4. Risk
5. Management
6. Reflection

Knowledge objects should be organised around this hierarchy so that Genesis learns the structure of discretionary thinking rather than isolated signals.

## Compatibility Commitments

GLA v1 is additive only:

- Existing bundles are not rewritten.
- Current JSON schemas are not changed.
- Dataset remains part of the system.
- Memory remains part of the system.
- Packages are not renamed.
- Runtime behaviour is not modified.
- No breaking API changes are introduced.

