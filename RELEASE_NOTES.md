# Release Notes

## Bundle 10: Fabio Video Ingestion

Genesis now supports deterministic Fabio video ingestion and synchronization for sources such as `FabioVideo.mp4`.

The Bundle 10 ingestion pipeline imports a video reference, decodes deterministic metadata, extracts deterministic frame references, maps timestamps, extracts audio timeline metadata, synchronizes frames to audio segments, runs the existing Genesis deterministic Vision-to-dataset path, and produces synchronized `TrainingSample` records in a `VideoDataset`.

Bundle 10 is ingestion-only and synchronization-only:

- NO speech recognition.
- NO AI.
- NO learning.
- NO reasoning.
- NO ML or neural networks.
- NO prediction, strategy generation, or trade recommendations.
- NO OCR modifications and NO OpenCV algorithm changes.

Bundle 11 introduces Transcript Alignment.

## Bundle 12 - Fabio Knowledge Extraction

Bundle 12 creates deterministic Fabio teaching datasets from Bundle 10 video datasets, Bundle 11 transcript alignment, Genesis DetectionGraph references, and TrainingSample references. It adds immutable knowledge models, statistics, configuration, results, and extraction APIs.

It performs NO learning, NO prediction, NO AI reasoning, NO strategy generation, and NO market forecasting. Bundle 13 introduces the Learning Engine.
