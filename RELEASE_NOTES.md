# Release Notes

## Bundle 7: Session Intelligence

Bundle 7 implements the Genesis Session Intelligence Engine milestones for trading session detection, deterministic session statistics, Initial Balance (IB), and opening auction analysis.

Genesis remains deterministic, immutable, reproducible, and platform independent. It does not perform AI reasoning, prediction, trading signals, probability estimation, machine learning, OCR changes, OpenCV changes, networking, threading, async execution, randomness, or vendor-specific logic.

Bundle 8 introduces Multi-Timeframe Context.

## Bundle 9: Dataset Builder & Annotation Infrastructure

Bundle 9 creates immutable AI-ready training datasets from deterministic Bundle 1-8 analysis graphs. Every analyzed frame can now become a complete `TrainingSample` with frame identity, timestamp metadata, graph-backed `FeatureVector` summaries, Fabio annotation placeholders, validation, statistics, serialization, and versioned JSONL, SQLite, or Parquet-compatible export.

Bundle 9 performs no machine learning, AI reasoning, neural-network training, LLM calls, prediction, probabilities, strategy generation, vendor-specific logic, OCR modifications, or OpenCV changes. Bundle 10 introduces the Learning Engine.
