================================================================================
GENESIS RUNNER INTEGRATION PATCH
================================================================================

Add these lines to your existing src/orderflowgpt_genesis/runner.py

Step 1: Add import at the top of runner.py
--------------------------------------------------------------------------------

# Add this import near the top of runner.py, after existing imports:
from orderflowgpt_genesis.apprentice import (
    ApprenticeRunnerIntegration,
    RunnerIntegrationConfiguration,
)

Step 2: Add to GenesisRunner.__init__
--------------------------------------------------------------------------------

# In GenesisRunner.__init__, add after existing engine initialization:
self.apprentice = ApprenticeRunnerIntegration(
    RunnerIntegrationConfiguration(
        process_key_frames_only=True,
        key_frame_interval=30,
        enable_coach=True,
        enable_concept_extraction=True,
        enable_experience_creation=True,
        enable_knowledge_graph_building=True,
        min_transcript_length_for_explanation=10,
        max_frames_per_lesson=100,
    )
)

Step 3: Modify _process_lesson() method
--------------------------------------------------------------------------------

Find your _process_lesson() method. After building the DetectionGraph
and aligning transcripts, add this block:

    def _process_lesson(self, video_path, transcript_path):
        # ... your existing code ...
        
        # EXISTING: Build detection graphs and align transcripts
        # detection_graphs = [...]
        # aligned_transcripts = [...]
        
        # ================================================================
        # NEW: Apprentice Layer Integration
        # ================================================================
        lesson_id = Path(video_path).stem
        
        apprentice_frames = []
        for frame_idx, (frame_graph, transcript_text) in enumerate(
            zip(detection_graphs, aligned_transcripts)
        ):
            apprentice_frames.append((
                frame_idx,
                getattr(frame_graph, 'timestamp', f'frame_{frame_idx}'),
                frame_graph,
                transcript_text if transcript_text else '',
            ))
        
        lesson_result = self.apprentice.process_lesson(
            lesson_reference=lesson_id,
            frames=tuple(apprentice_frames),
        )
        
        self.logger.info(
            f'Apprentice: {lesson_id} - '
            f'{len(lesson_result.frame_results)} frames, '
            f'{lesson_result.concept_statistics.total_concepts} concepts, '
            f'{lesson_result.experience_statistics.total_experiences} experiences'
        )
        # ================================================================
        
        # ... rest of your existing save logic ...

Step 4: Modify _save_reports() method
--------------------------------------------------------------------------------

    def _save_reports(self, output_dir, lesson_id):
        # ... your existing report.json and summary.json save logic ...
        
        # ================================================================
        # NEW: Save Apprentice Report
        # ================================================================
        apprentice_report = self.apprentice.build_report(
            timestamp=datetime.utcnow().isoformat()
        )
        
        apprentice_report_path = Path(output_dir) / 'apprentice_report.json'
        with open(apprentice_report_path, 'w', encoding='utf-8') as f:
            import json
            from decimal import Decimal
            
            class DecimalEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Decimal):
                        return float(obj)
                    if isinstance(obj, tuple):
                        return list(obj)
                    return super().default(obj)
            
            json.dump({
                'report_id': apprentice_report.report_id,
                'timestamp': apprentice_report.report_timestamp,
                'total_lessons': len(apprentice_report.lesson_results),
                'total_concepts_learned': apprentice_report.total_concepts_learned,
                'total_experiences_created': apprentice_report.total_experiences_created,
                'total_sessions_completed': apprentice_report.total_sessions_completed,
                'total_coach_explanations': apprentice_report.total_coach_explanations,
                'total_frames_processed': apprentice_report.total_frames_processed,
                'overall_learning_confidence': float(apprentice_report.overall_learning_confidence),
                'concept_mastery_distribution': apprentice_report.concept_mastery_distribution,
                'top_concepts_by_confidence': [
                    {'name': name, 'score': float(score)}
                    for name, score in apprentice_report.top_concepts_by_confidence[:10]
                ],
                'what_was_learned': list(apprentice_report.what_was_learned),
                'what_needs_more_study': list(apprentice_report.what_needs_more_study),
                'knowledge_graph_summary': apprentice_report.knowledge_graph_summary,
            }, f, indent=2, cls=DecimalEncoder)
        
        self.logger.info(f'Saved apprentice report: {apprentice_report_path}')
        # ================================================================

Step 5: Run as usual
--------------------------------------------------------------------------------

Your normal command now produces apprentice outputs automatically:

    python -m orderflowgpt_genesis --folder assets/fabio/videos --output assets/fabio/output

New output files per lesson:
    output/Lesson01/apprentice_report.json
    output/Lesson01/concepts.json
    output/Lesson01/experiences.json
    output/Lesson01/knowledge_graph.json
    output/Lesson01/coach_explanations.txt
    output/Lesson01/apprentice_summary.txt