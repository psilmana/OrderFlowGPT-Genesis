#!/usr/bin/env python3
"""
video_frame_debug.py

Diagnoses and fixes video frame extraction issues for OrderFlowGPT-Genesis
Bundle 10 / Apprentice video processing.

Usage:
    # Diagnose a video
    python video_frame_debug.py --video path/to/Lesson01.mp4

    # Extract frames with robust fallback
    python video_frame_debug.py --video path/to/Lesson01.mp4 --extract --output frames/

    # Use as a drop-in replacement in your pipeline
    from video_frame_debug import RobustFrameExtractor
    extractor = RobustFrameExtractor("Lesson01.mp4")
    frames = extractor.extract_key_frames(interval_sec=5.0, max_frames=100)
"""

import argparse
import sys
import os
import cv2
import math
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass
import shutil


@dataclass
class VideoInfo:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float
    fourcc: str
    codec_name: str
    is_readable: bool
    actual_frames_read: int = 0


def diagnose_video(video_path: str) -> VideoInfo:
    """Deep diagnostic of a video file using multiple strategies."""
    path = Path(video_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"VIDEO DIAGNOSTIC: {path.name}")
    print(f"{'='*60}")
    print(f"Full path: {path.resolve()}")
    print(f"File size: {path.stat().st_size / (1024*1024):.2f} MB")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("[FATAL] OpenCV cannot open this video.")
        print("Possible causes:")
        print("  - Missing codec (install K-Lite Codec Pack or ffmpeg)")
        print("  - Corrupted file")
        print("  - Unsupported container format")
        cap.release()
        sys.exit(1)

    # Primary properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_raw >> 8 * i) & 0xFF) for i in range(4)])

    # Duration calculation
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    else:
        duration = 0.0

    print(f"\n[OpenCV Properties]")
    print(f"  Frame count (reported): {frame_count}")
    print(f"  FPS (reported):         {fps:.3f}")
    print(f"  Resolution:             {width}x{height}")
    print(f"  Duration (calc):        {duration:.2f}s ({duration/60:.2f} min)")
    print(f"  FourCC:                 {fourcc!r}")

    # Test 1: Sequential read (ground truth frame count)
    print(f"\n[Test 1] Sequential frame read...")
    actual_frames = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        actual_frames += 1
        if actual_frames <= 3:
            print(f"  Frame {actual_frames}: shape={frame.shape}, mean={frame.mean():.1f}")

    print(f"  Actual frames readable: {actual_frames}")
    if actual_frames == 0:
        print("  [FAIL] Zero frames readable. Video is likely corrupted or codec is missing.")
    elif frame_count > 0 and actual_frames != frame_count:
        print(f"  [WARN] Reported {frame_count} but read {actual_frames}. OpenCV estimate may be wrong.")

    # Test 2: Key frame seek
    print(f"\n[Test 2] Key-frame seek test...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if ret:
        print(f"  Seek to frame 0: OK")
    else:
        print(f"  Seek to frame 0: FAILED")

    # Test 3: Random access
    if actual_frames > 10:
        target = actual_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        print(f"  Seek to frame {target}: {'OK' if ret else 'FAILED'}")

    cap.release()

    # Codec lookup
    codec_map = {
        "avc1": "H.264", "H264": "H.264", "h264": "H.264",
        "hev1": "H.265/HEVC", "hvc1": "H.265/HEVC",
        "mp4v": "MPEG-4", "MP42": "MPEG-4",
        "MJPG": "Motion JPEG", "mjpa": "Motion JPEG",
        "XVID": "Xvid", "DIVX": "DivX",
        "VP80": "VP8", "VP90": "VP9",
    }
    codec_name = codec_map.get(fourcc, f"Unknown ({fourcc})")
    print(f"  Codec: {codec_name}")

    # Recommendations
    print(f"\n[Recommendations]")
    if actual_frames == 0:
        print("  1. Install ffmpeg and ensure it's in PATH")
        print("  2. Try converting: ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset fast output.mp4")
        print("  3. Check if the file plays in a media player")
    elif actual_frames == 1:
        print("  [ISSUE] Only 1 frame readable. This explains your '1 frame' result.")
        print("  Possible causes:")
        print("    - Video is actually a single image")
        print("    - Key-frame interval logic is discarding everything")
        print("    - Frame extraction loop has a break/return too early")
    elif actual_frames < 10:
        print(f"  [WARN] Very short video ({actual_frames} frames).")
    else:
        print(f"  Video looks healthy. Check your extraction logic.")

    print(f"{'='*60}\n")

    return VideoInfo(
        path=str(path),
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration,
        fourcc=fourcc,
        codec_name=codec_name,
        is_readable=actual_frames > 0,
        actual_frames_read=actual_frames,
    )


class RobustFrameExtractor:
    """
    Drop-in replacement for fragile frame extraction.
    Handles codec issues, incorrect frame counts, and seek failures.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self._info: Optional[VideoInfo] = None

    def info(self) -> VideoInfo:
        if self._info is None:
            self._info = diagnose_video(str(self.video_path))
        return self._info

    def extract_all_frames(self, output_dir: Optional[str] = None) -> List[Tuple[int, any]]:
        """Extract every readable frame. Returns list of (frame_number, image)."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        frames = []
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((frame_num, frame))
            if output_dir:
                out = Path(output_dir) / f"frame_{frame_num:06d}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), frame)
            frame_num += 1

        cap.release()
        print(f"[RobustFrameExtractor] Extracted {len(frames)} frames")
        return frames

    def extract_key_frames(
        self,
        interval_sec: float = 5.0,
        max_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> List[Tuple[int, any]]:
        """
        Extract frames at regular time intervals.
        This is safer than using CAP_PROP_POS_FRAMES which can fail on some codecs.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # fallback

        frame_interval = int(round(fps * interval_sec))
        if frame_interval < 1:
            frame_interval = 1

        frames = []
        frame_num = 0
        next_target = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num >= next_target:
                frames.append((frame_num, frame))
                if output_dir:
                    out = Path(output_dir) / f"keyframe_{frame_num:06d}_t{frame_num/fps:.1f}s.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), frame)
                next_target += frame_interval

                if max_frames and len(frames) >= max_frames:
                    break

            frame_num += 1

        cap.release()
        print(f"[RobustFrameExtractor] Extracted {len(frames)} key frames @ {interval_sec}s interval")
        return frames

    def extract_by_scene_change(
        self,
        threshold: float = 30.0,
        max_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> List[Tuple[int, any]]:
        """
        Extract frames where significant visual changes occur.
        Useful for chart videos where the scene changes when switching timeframes.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        frames = []
        prev_frame = None
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = diff.mean()
                if mean_diff > threshold:
                    frames.append((frame_num, frame))
                    if output_dir:
                        out = Path(output_dir) / f"scene_{frame_num:06d}_diff{mean_diff:.1f}.png"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out), frame)
                    if max_frames and len(frames) >= max_frames:
                        break
            else:
                # Always include first frame
                frames.append((frame_num, frame))

            prev_frame = gray
            frame_num += 1

        cap.release()
        print(f"[RobustFrameExtractor] Extracted {len(frames)} scene-change frames")
        return frames


def main():
    parser = argparse.ArgumentParser(description="Video frame extraction diagnostic & fix")
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument("--extract", "-e", action="store_true", help="Extract frames")
    parser.add_argument("--output", "-o", default="frames", help="Output directory")
    parser.add_argument("--mode", choices=["all", "key", "scene"], default="key",
                        help="Extraction mode: all=every frame, key=time interval, scene=visual change")
    parser.add_argument("--interval", type=float, default=5.0, help="Key frame interval in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to extract")
    args = parser.parse_args()

    # Always run diagnostic first
    info = diagnose_video(args.video)

    if args.extract:
        extractor = RobustFrameExtractor(args.video)
        out_dir = Path(args.output)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "all":
            extractor.extract_all_frames(str(out_dir))
        elif args.mode == "key":
            extractor.extract_key_frames(
                interval_sec=args.interval,
                max_frames=args.max_frames,
                output_dir=str(out_dir),
            )
        elif args.mode == "scene":
            extractor.extract_by_scene_change(
                max_frames=args.max_frames,
                output_dir=str(out_dir),
            )

        print(f"\nFrames saved to: {out_dir.resolve()}")
        print(f"Frame files: {len(list(out_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()cl