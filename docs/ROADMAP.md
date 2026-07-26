# Genesis Learning Architecture Roadmap

This roadmap defines the staged evolution from a video-processing system toward an AI Apprentice learning system.

The roadmap is architectural and documentation-oriented. It does not introduce trading automation, runtime behaviour changes, package renames, schema changes, or breaking APIs.

## Stage 1: Observation

Genesis learns to observe lesson material reliably.

Focus:

- Preserve existing Vision, Video, Transcript, Timeline, and Dataset modules as the Perception Layer.
- Convert lesson material into observations.
- Keep perception separate from reasoning.
- Maintain compatibility with existing bundles and schemas.

Outcome:

Genesis can describe what it sees and hears without claiming complete understanding.

## Stage 2: Understanding

Genesis learns to map observations to Fabio's methodology.

Focus:

- Relate observations to Market State, Location, Aggression, Risk, Management, and Reflection.
- Distinguish raw detection from meaningful context.
- Capture Fabio's explanations and corrections.

Outcome:

Genesis can explain why an observation matters within a lesson.

## Stage 3: Reasoning

Genesis learns to reason through the hierarchy before prediction.

Focus:

- Apply the reasoning order consistently.
- Compare Genesis's interpretation with Fabio's teaching.
- Record decision logic, non-action logic, and uncertainty.

Outcome:

Genesis can produce structured explanations before forming directional expectations.

## Stage 4: Generalisation

Genesis learns from repeated reflected experiences.

Focus:

- Connect similar experiences.
- Identify recurring principles.
- Separate durable knowledge from one-off examples.
- Organise knowledge objects around the Fabio hierarchy.

Outcome:

Genesis begins to develop transferable intuition from memory.

## Stage 5: Live Coaching

Genesis becomes capable of educational live coaching.

Focus:

- Explain context in real time.
- Recall relevant experiences.
- Highlight uncertainty and risk.
- Coach reasoning without becoming an automated trading system.

Outcome:

Genesis can assist a learner by explaining Order Flow context, reasoning paths, and lessons remembered from Fabio, while remaining an AI Apprentice and not a trading bot.
