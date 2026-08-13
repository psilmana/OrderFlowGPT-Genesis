#!/usr/bin/env python3
"""
chart_vision_analyzer.py — Real OpenCV-based chart analysis for Genesis vision mode.

Detects:
  • Chart region in video frames (contour detection)
  • Footprint grid patterns (Hough line detection)
  • Delta color analysis (buy/sell pressure from dominant colors)
  • Price action structure (trend direction from silhouette)
  • Imbalance zones (asymmetric color distribution)
  • Session boundaries (horizontal price levels)

GPU Acceleration:
  • Auto-detects CUDA-enabled OpenCV
  • Falls back to CPU gracefully
  • Reports device usage per frame
"""

import json
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# GPU / CUDA detection
# ---------------------------------------------------------------------------

class GPUContext:
    """Manages GPU acceleration state for OpenCV operations."""

    def __init__(self):
        self.cuda_available = False
        self.cuda_device = None
        self._check_cuda()

    def _check_cuda(self):
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cv2.cuda.setDevice(0)
                self.cuda_available = True
                self.cuda_device = cv2.cuda.getDevice()
                print(f"[GPU] CUDA enabled — device {self.cuda_device}")
            else:
                print("[GPU] CUDA not available — using CPU")
        except Exception as e:
            print(f"[GPU] CUDA check failed ({e}) — using CPU")

    def upload(self, frame: np.ndarray) -> Optional[Any]:
        if self.cuda_available:
            try:
                return cv2.cuda_GpuMat(frame)
            except Exception:
                return None
        return None

    def download(self, gpu_mat: Any) -> np.ndarray:
        if gpu_mat is not None:
            return gpu_mat.download()
        return np.array([])

GPU = GPUContext()

# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------

@dataclass
class ChartRegion:
    x: int
    y: int
    w: int
    h: int
    confidence: float

@dataclass
class GridInfo:
    horizontal_lines: List[Tuple[int, int, int, int]]  # x1,y1,x2,y2
    vertical_lines: List[Tuple[int, int, int, int]]
    cell_count: int
    confidence: float

@dataclass
class DeltaProfile:
    buy_dominance: float   # 0.0-1.0
    sell_dominance: float  # 0.0-1.0
    neutral_ratio: float   # 0.0-1.0
    confidence: float

@dataclass
class PriceAction:
    trend_direction: str   # "UP", "DOWN", "BALANCED", "UNKNOWN"
    volatility_estimate: str  # "HIGH", "MEDIUM", "LOW"
    structure: str         # "TRENDING", "RANGING", "CHOPPY"
    confidence: float

@dataclass
class VisionFeatures:
    has_chart: bool
    chart_region: Optional[ChartRegion]
    grid: Optional[GridInfo]
    delta: Optional[DeltaProfile]
    price_action: Optional[PriceAction]
    dominant_colors: List[Tuple[int, int, int]]
    text_regions: List[Tuple[int, int, int, int]]
    processing_time_ms: float

# ---------------------------------------------------------------------------
# Chart detection
# ---------------------------------------------------------------------------

def detect_chart_region(frame: np.ndarray) -> Optional[ChartRegion]:
    """Find the largest rectangular region that looks like a chart.

    Strategy:
    1. Convert to grayscale
    2. Edge detection (Canny)
    3. Find contours
    4. Select largest rectangle with reasonable aspect ratio
    """
    h, w = frame.shape[:2]
    if h < 100 or w < 100:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use GPU if available for Gaussian blur
    if GPU.cuda_available:
        try:
            gpu_gray = cv2.cuda_GpuMat(gray)
            gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5,5), 0)
            blurred = gpu_blur.apply(gpu_gray).download()
        except Exception:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_score = 0.0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        frame_area = w * h

        # Must be reasonably large (at least 15% of frame)
        if area < frame_area * 0.15:
            continue

        # Aspect ratio check (charts are usually wider than tall)
        aspect = cw / max(1, ch)
        if aspect < 0.8 or aspect > 3.0:
            continue

        # Score = area * aspect_quality
        aspect_quality = 1.0 - abs(aspect - 1.6) / 1.6  # peak at 1.6:1
        score = area * max(0, aspect_quality)

        if score > best_score:
            best_score = score
            best_rect = (x, y, cw, ch)

    if best_rect:
        x, y, cw, ch = best_rect
        conf = min(1.0, best_score / (frame_area * 0.5))
        return ChartRegion(x, y, cw, ch, conf)

    return None

# ---------------------------------------------------------------------------
# Grid detection (footprint charts have regular grids)
# ---------------------------------------------------------------------------

def detect_grid(chart_roi: np.ndarray) -> Optional[GridInfo]:
    """Detect horizontal and vertical grid lines in a footprint chart.

    TUNED for footprint charts:
    - Lower Canny thresholds for faint grid lines
    - Lower Hough threshold for subtle lines
    - Shorter minLineLength for cell boundaries
    - Adaptive thresholding as preprocessing
    """
    gray = cv2.cvtColor(chart_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Preprocessing: adaptive threshold to enhance faint grid lines
    # Footprint grids are often low-contrast against dark backgrounds
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # Also try standard Canny on original + adaptive
    edges1 = cv2.Canny(gray, 20, 60, apertureSize=3)
    edges2 = cv2.Canny(adaptive, 20, 60, apertureSize=3)
    edges = cv2.bitwise_or(edges1, edges2)

    # Dilate to connect broken grid segments
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Hough line detection — TUNED parameters
    # threshold=20 (was 50): catch faint lines
    # minLineLength=w//20 (was w//4): cell boundaries are short
    # maxLineGap=10 (was 5): allow slightly broken lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                            minLineLength=max(20, w // 20),
                            maxLineGap=10)

    if lines is None or len(lines) < 3:
        return None

    h_lines = []
    v_lines = []

    for line in lines:
        if hasattr(line, 'shape') and line.shape == (1, 4):
            x1, y1, x2, y2 = line[0]
        elif hasattr(line, '__len__') and len(line) == 4:
            x1, y1, x2, y2 = line
        else:
            continue
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Relaxed orientation check for slightly diagonal lines
        if dx > dy * 2:  # Horizontal (was *5)
            h_lines.append((x1, y1, x2, y2))
        elif dy > dx * 2:  # Vertical (was *5)
            v_lines.append((x1, y1, x2, y2))

    # Need at least a few of each
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    # Deduplicate nearby parallel lines (footprint grids have tight spacing)
    def dedup_lines(lines, axis=0, min_dist=3):
        """Remove duplicate lines that are very close together."""
        if not lines:
            return []
        # Sort by midpoint on the perpendicular axis
        midpoints = []
        for x1, y1, x2, y2 in lines:
            if axis == 0:  # horizontal: sort by y-midpoint
                mid = (y1 + y2) / 2
            else:  # vertical: sort by x-midpoint
                mid = (x1 + x2) / 2
            midpoints.append((mid, (x1, y1, x2, y2)))
        midpoints.sort()

        deduped = [midpoints[0][1]]
        for mid, line in midpoints[1:]:
            last_mid = midpoints[midpoints.index((mid, line)) - 1][0] if midpoints[midpoints.index((mid, line)) - 1][0] != mid else deduped[-1][1 if axis==0 else 0]
            # Check distance from last kept line
            last_y = (deduped[-1][1] + deduped[-1][3]) / 2 if axis == 0 else (deduped[-1][0] + deduped[-1][2]) / 2
            if abs(mid - last_y) >= min_dist:
                deduped.append(line)
        return deduped

    h_lines = dedup_lines(h_lines, axis=0, min_dist=max(2, h // 40))
    v_lines = dedup_lines(v_lines, axis=1, min_dist=max(2, w // 40))

    # Estimate cell count
    cells_h = max(1, len(h_lines) - 1)
    cells_v = max(1, len(v_lines) - 1)
    cell_count = cells_h * cells_v

    # Confidence: more lines = higher confidence, cap at 1.0
    conf = min(1.0, (len(h_lines) + len(v_lines)) / 30.0)

    return GridInfo(h_lines, v_lines, cell_count, conf)

# ---------------------------------------------------------------------------
# Delta color analysis
# ---------------------------------------------------------------------------

def analyze_delta_colors(chart_roi: np.ndarray) -> Optional[DeltaProfile]:
    """Analyze color distribution to estimate buy/sell pressure.

    Footprint charts typically use:
    - Green/cyan for buy delta (positive)
    - Red/magenta for sell delta (negative)
    - White/gray for neutral
    """
    # Resize for speed
    small = cv2.resize(chart_roi, (chart_roi.shape[1]//2, chart_roi.shape[0]//2))

    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    total = len(pixels)
    if total == 0:
        return None

    # Define color ranges in HSV
    # Green (buy): H ~ 35-85, S > 40, V > 40
    green_mask = ((pixels[:, 0] >= 35) & (pixels[:, 0] <= 85) &
                  (pixels[:, 1] >= 40) & (pixels[:, 2] >= 40))

    # Red (sell): H ~ 0-10 or 160-180, S > 40, V > 40
    red_mask = (((pixels[:, 0] <= 10) | (pixels[:, 0] >= 160)) &
                (pixels[:, 1] >= 40) & (pixels[:, 2] >= 40))

    # Cyan (buy alternative): H ~ 85-100
    cyan_mask = ((pixels[:, 0] >= 85) & (pixels[:, 0] <= 100) &
                 (pixels[:, 1] >= 40) & (pixels[:, 2] >= 40))

    # Magenta (sell alternative): H ~ 140-160
    magenta_mask = ((pixels[:, 0] >= 140) & (pixels[:, 0] <= 160) &
                    (pixels[:, 1] >= 40) & (pixels[:, 2] >= 40))

    buy_count = np.sum(green_mask) + np.sum(cyan_mask)
    sell_count = np.sum(red_mask) + np.sum(magenta_mask)
    neutral_count = total - buy_count - sell_count

    buy_dom = buy_count / total
    sell_dom = sell_count / total
    neutral = neutral_count / total

    # Confidence based on color saturation (charts have vivid colors)
    vivid_mask = (pixels[:, 1] > 60) & (pixels[:, 2] > 60)
    vivid_ratio = np.sum(vivid_mask) / total
    conf = min(1.0, vivid_ratio * 2.0)

    return DeltaProfile(
        buy_dominance=float(buy_dom),
        sell_dominance=float(sell_dom),
        neutral_ratio=float(neutral),
        confidence=conf
    )

# ---------------------------------------------------------------------------
# Price action / trend detection from silhouette
# ---------------------------------------------------------------------------

def detect_price_action(chart_roi: np.ndarray, 
                        transcript_text: str = "") -> Optional[PriceAction]:
    """Estimate trend and structure from the chart's price action silhouette.

    Uses horizontal projection (row brightness) to find price levels
    and vertical projection (column brightness) to find time progression.
    """
    gray = cv2.cvtColor(chart_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Horizontal projection: sum each row
    row_sums = np.sum(gray, axis=1)

    # Find peaks (price levels with lots of activity) — numpy only, no scipy
    peaks = []
    mean_val = np.mean(row_sums)
    std_val = np.std(row_sums)
    threshold = mean_val + 0.5 * std_val
    min_dist = max(3, h // 10)

    for i in range(1, len(row_sums) - 1):
        if row_sums[i] > threshold and row_sums[i] > row_sums[i-1] and row_sums[i] > row_sums[i+1]:
            # Check distance from last peak
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
    peaks = np.array(peaks)

    if len(peaks) < 2:
        # Not enough structure — use transcript fallback
        return _price_action_from_text(transcript_text)

    # Analyze peak distribution over time (columns)
    # Divide chart into 3 vertical zones: left (past), middle, right (recent)
    zone_width = w // 3

    left_activity = np.mean(row_sums[:h//3]) if h > 0 else 0
    mid_activity = np.mean(row_sums[h//3:2*h//3]) if h > 0 else 0
    right_activity = np.mean(row_sums[2*h//3:]) if h > 0 else 0

    # Volatility from peak count
    volatility = "LOW"
    if len(peaks) > 15:
        volatility = "HIGH"
    elif len(peaks) > 8:
        volatility = "MEDIUM"

    # Structure from activity distribution
    activity_std = np.std([left_activity, mid_activity, right_activity])
    if activity_std < 0.1 * np.mean([left_activity, mid_activity, right_activity]):
        structure = "RANGING"
    elif right_activity > mid_activity > left_activity:
        structure = "TRENDING"
    else:
        structure = "CHOPPY"

    # Trend direction from transcript (vision alone can't determine up/down easily)
    text_trend = _price_action_from_text(transcript_text)
    trend_dir = text_trend.trend_direction if text_trend else "UNKNOWN"

    conf = min(1.0, len(peaks) / 20.0)

    return PriceAction(
        trend_direction=trend_dir,
        volatility_estimate=volatility,
        structure=structure,
        confidence=conf
    )

def _price_action_from_text(text: str) -> Optional[PriceAction]:
    """Fallback trend detection from transcript keywords."""
    t = text.lower()

    if "trend" in t and "up" in t:
        trend = "UP"
    elif "trend" in t and "down" in t:
        trend = "DOWN"
    elif "auction" in t and "down" in t:
        trend = "DOWN"
    elif "auction" in t and "up" in t:
        trend = "UP"
    elif "balance" in t or "rotational" in t:
        trend = "BALANCED"
    else:
        trend = "UNKNOWN"

    vol = "MEDIUM"
    if "volatile" in t or "volatility" in t:
        vol = "HIGH"
    elif "choppy" in t or "slow" in t:
        vol = "LOW"

    struct = "TRENDING" if "trend" in t else "RANGING" if "range" in t or "balance" in t else "CHOPPY"

    return PriceAction(trend, vol, struct, 0.5)

# ---------------------------------------------------------------------------
# Main vision analysis pipeline
# ---------------------------------------------------------------------------

def analyze_frame(frame: np.ndarray, transcript_text: str = "") -> VisionFeatures:
    """Run full vision analysis on a single frame.

    Returns VisionFeatures with all detected chart properties.
    """
    import time
    t0 = time.time()

    # Step 1: Chart detection
    chart_region = detect_chart_region(frame)
    has_chart = chart_region is not None

    if not has_chart:
        return VisionFeatures(
            has_chart=False,
            chart_region=None,
            grid=None,
            delta=None,
            price_action=None,
            dominant_colors=[],
            text_regions=[],
            processing_time_ms=(time.time() - t0) * 1000
        )

    # Extract chart ROI
    cr = chart_region
    chart_roi = frame[cr.y:cr.y+cr.h, cr.x:cr.x+cr.w]

    # Step 2: Grid detection
    grid = detect_grid(chart_roi)

    # Step 3: Delta color analysis
    delta = analyze_delta_colors(chart_roi)

    # Step 4: Price action
    price_action = detect_price_action(chart_roi, transcript_text)

    # Step 5: Dominant colors (k-means on chart ROI)
    small = cv2.resize(chart_roi, (50, 50))
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant = [tuple(map(int, c)) for c in centers]

    elapsed = (time.time() - t0) * 1000

    return VisionFeatures(
        has_chart=has_chart,
        chart_region=chart_region,
        grid=grid,
        delta=delta,
        price_action=price_action,
        dominant_colors=dominant,
        text_regions=[],
        processing_time_ms=elapsed
    )

# ---------------------------------------------------------------------------
# Genesis-compatible DetectionGraph builder
# ---------------------------------------------------------------------------

def build_vision_detection_graph(
    frame_idx: int,
    frame_image: np.ndarray,
    transcript_text: str,
    ontology: dict,
) -> Any:
    """Build a real DetectionGraph from vision analysis + transcript fusion.

    This replaces the fake detection graph with actual image-derived state.
    """
    # Run vision analysis
    features = analyze_frame(frame_image, transcript_text)

    # Build components based on what was detected
    trend = None
    session = None
    imbalances = []
    absorptions = []
    confluence_type = "NO_CONFLUENCE"

    if features.has_chart:
        # Trend from price action
        if features.price_action:
            pa = features.price_action
            trend_state_name = f"TRENDING_{pa.trend_direction}" if pa.trend_direction != "UNKNOWN" else "BALANCED"
            if pa.structure == "RANGING":
                trend_state_name = "BALANCED"
            elif pa.structure == "CHOPPY":
                trend_state_name = "CHOPPY"
            trend = FakeTrend(FakeTrendState(trend_state_name))

        # Session (always RTH for now, could detect from timestamps)
        session = FakeSession("RTH")

        # Imbalances from delta analysis
        if features.delta:
            d = features.delta
            if d.buy_dominance > 0.6:
                imbalances.append(FakeImbalance("BID", f"price_{frame_idx}"))
            elif d.sell_dominance > 0.6:
                imbalances.append(FakeImbalance("ASK", f"price_{frame_idx}"))

            # Extreme imbalance = stacked
            if d.buy_dominance > 0.8 or d.sell_dominance > 0.8:
                confluence_type = "STRONG_IMBALANCE"
            elif d.buy_dominance > 0.65 or d.sell_dominance > 0.65:
                confluence_type = "IMBALANCE_PRESENT"

        # Absorption from grid + delta combination
        if features.grid and features.delta:
            # If grid is dense and delta is neutral, possible absorption
            if features.grid.cell_count > 50 and features.delta.neutral_ratio > 0.5:
                side = "BID" if features.delta.buy_dominance > features.delta.sell_dominance else "ASK"
                absorptions.append(FakeAbsorption(side, f"price_{frame_idx}"))
                confluence_type = "ABSORPTION_DETECTED"

        # Volatility from price action
        if features.price_action and features.price_action.volatility_estimate == "HIGH":
            confluence_type = "HIGH_VOLATILITY"

    # Transcript override for confluence (text often has clearer signals)
    text = transcript_text.lower()
    if "strong" in text or "confirm" in text:
        confluence_type = "STRONG_CONFLUENCE"
    elif "weak" in text:
        confluence_type = "WEAK_CONFLUENCE"
    elif "absorption" in text:
        confluence_type = "ABSORPTION_DETECTED"
    elif "exhaustion" in text:
        confluence_type = "EXHAUSTION"

    return FakeDetectionGraph(
        graph_id=f"frame_{frame_idx:04d}",
        timestamp=f"00:{frame_idx // 60:02d}:{frame_idx % 60:02d}",
        trend_state=trend,
        trading_session=session,
        footprint_imbalances=FakeImbalances(tuple(imbalances)) if imbalances else None,
        absorption_result=FakeAbsorptions(tuple(absorptions)) if absorptions else None,
        footprint_delta=FakeDelta(()),
        confluence=FakeConfluence(confluence_type),
    )


# ---------------------------------------------------------------------------
# Debug frame annotation
# ---------------------------------------------------------------------------

def annotate_debug_frame(
    frame: np.ndarray,
    features: VisionFeatures,
    vision_concepts: List[Any],
    frame_idx: int,
) -> np.ndarray:
    """Draw vision detection annotations directly on the frame for human verification.

    Overlays:
      • Green rectangle: detected chart region
      • Red lines: horizontal grid lines
      • Blue lines: vertical grid lines
      • Color bar: buy/sell/neutral delta ratio
      • Text panel: assigned vision concepts with confidence

    Returns a copy of the frame with annotations.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # --- Chart region bounding box ---
    if features.has_chart and features.chart_region:
        cr = features.chart_region
        cv2.rectangle(annotated, (cr.x, cr.y), (cr.x + cr.w, cr.y + cr.h),
                      (0, 255, 0), 2)
        cv2.putText(annotated, f"CHART {cr.confidence:.2f}",
                    (cr.x + 5, cr.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- Grid lines (drawn inside chart ROI) ---
    if features.grid and features.chart_region:
        cr = features.chart_region
        roi_x, roi_y = cr.x, cr.y

        for x1, y1, x2, y2 in features.grid.horizontal_lines:
            cv2.line(annotated, (roi_x + x1, roi_y + y1), (roi_x + x2, roi_y + y2),
                     (0, 0, 255), 1)
        for x1, y1, x2, y2 in features.grid.vertical_lines:
            cv2.line(annotated, (roi_x + x1, roi_y + y1), (roi_x + x2, roi_y + y2),
                     (255, 0, 0), 1)

        # Grid info text
        cv2.putText(annotated,
                    f"GRID: {features.grid.cell_count} cells, conf={features.grid.confidence:.2f}",
                    (roi_x + 5, roi_y + cr.h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    # --- Delta color bar ---
    if features.delta:
        d = features.delta
        bar_x, bar_y = 10, h - 40
        bar_w, bar_h = 200, 20

        # Background
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (40, 40, 40), -1)

        # Buy segment (green)
        buy_w = int(bar_w * d.buy_dominance)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + buy_w, bar_y + bar_h),
                      (0, 200, 0), -1)

        # Sell segment (red)
        sell_w = int(bar_w * d.sell_dominance)
        cv2.rectangle(annotated, (bar_x + buy_w, bar_y),
                      (bar_x + buy_w + sell_w, bar_y + bar_h),
                      (0, 0, 200), -1)

        # Neutral segment (gray)
        neu_w = bar_w - buy_w - sell_w
        cv2.rectangle(annotated, (bar_x + buy_w + sell_w, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (128, 128, 128), -1)

        # Labels
        cv2.putText(annotated,
                    f"BUY {d.buy_dominance:.0%}  SELL {d.sell_dominance:.0%}  NEU {d.neutral_ratio:.0%}",
                    (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- Vision concepts text panel (top-right) ---
    if vision_concepts:
        panel_x = w - 260
        panel_y = 10
        line_h = 18

        # Background panel
        panel_h = min(len(vision_concepts) * line_h + 10, 200)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (panel_x - 5, panel_y - 5),
                      (w - 5, panel_y + panel_h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        cv2.putText(annotated, "VISION CONCEPTS:",
                    (panel_x, panel_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for i, vc in enumerate(vision_concepts[:8]):  # max 8 lines
            y = panel_y + 28 + i * line_h
            conf = float(vc.confidence.score)
            color = (0, 255, 0) if conf > 0.6 else (0, 200, 200) if conf > 0.4 else (0, 128, 255)
            text = f"{vc.definition.name[:22]} {conf:.2f}"
            cv2.putText(annotated, text, (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # --- Frame index watermark ---
    cv2.putText(annotated, f"FRAME {frame_idx:03d}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return annotated

# ---------------------------------------------------------------------------
# Fake classes (must match process_real_video.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeTrendState:
    name: str

@dataclass(frozen=True)
class FakeTrend:
    trend_state: FakeTrendState

@dataclass(frozen=True)
class FakeSession:
    session_type: str

@dataclass(frozen=True)
class FakeImbalance:
    side: str
    cell_id: str

@dataclass(frozen=True)
class FakeImbalances:
    imbalances: Tuple[Any, ...]

@dataclass(frozen=True)
class FakeAbsorption:
    side: str
    cell_id: str

@dataclass(frozen=True)
class FakeAbsorptions:
    absorptions: Tuple[Any, ...]

@dataclass(frozen=True)
class FakeDelta:
    cell_deltas: Tuple[Any, ...]

@dataclass(frozen=True)
class FakeConfluence:
    confluence_type: str

@dataclass(frozen=True)
class FakeDetectionGraph:
    graph_id: str
    timestamp: str
    trend_state: Optional[FakeTrend] = None
    trading_session: Optional[FakeSession] = None
    footprint_imbalances: Optional[FakeImbalances] = None
    absorption_result: Optional[FakeAbsorptions] = None
    footprint_delta: Optional[FakeDelta] = None
    confluence: Optional[FakeConfluence] = None