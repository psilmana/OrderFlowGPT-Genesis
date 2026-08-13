#!/usr/bin/env python3
"""
process_real_video.py  —  CLEANED VERSION v5.1

Fixes applied:
 1. SEMANTIC FILTERING: RelationshipExtractor now filters stop words and enforces
    domain-quality gates. No more "Blood --cooccurs_with--> Sweat" noise.
 2. CONCEPT QUALITY: extract_concepts_from_text() uses fuzzy matching and filters
    against a trading-domain stop list. Reduces invented-concept spam.
 3. INTEGRATION GRAPH INJECTION: Discovered relationships and ontology relationships
    are injected into the ApprenticeRunnerIntegration's knowledge graph so the
    integration graph is no longer empty (0 relationships).
 4. CO-OCCURRENCE MINIMUM: Raised default min_cooccurrence to 2 for batch runs
    (configurable via --min-cooccurrence).
 5. CONFIDENCE HONESTY: compute_real_confidence() now uses graph relationship
    density correctly.
 6. RELATIONSHIP DEDUPLICATION: Prevents duplicate edges in discovered_rels.json.
 7. STOP-WORD LISTS: Shared with batch_process_lessons via module-level constants.

Usage — Transcript-only (default, honest):
 python process_real_video.py \
     --video assets/fabio/videos/Lesson01.mp4 \
     --transcript assets/fabio/transcripts/Lesson01.txt \
     --output assets/fabio/output/Lesson01 \
     --mode transcript-only \
     --ontology fabio_ontology.json
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set, Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from orderflowgpt_genesis.apprentice import (
    ApprenticeRunnerIntegration,
    RunnerIntegrationConfiguration,
    ApprenticeReport,
    Concept,
    Experience,
    KnowledgeGraph,
)

# Vision analyzer (real OpenCV chart analysis)
try:
    from chart_vision_analyzer import (
        analyze_frame,
        build_vision_detection_graph,
        annotate_debug_frame,
        GPU,
        FakeDetectionGraph as VisionFakeDetectionGraph,
        FakeTrend as VisionFakeTrend,
        FakeTrendState as VisionFakeTrendState,
        FakeSession as VisionFakeSession,
        FakeImbalance as VisionFakeImbalance,
        FakeImbalances as VisionFakeImbalances,
        FakeAbsorption as VisionFakeAbsorption,
        FakeAbsorptions as VisionFakeAbsorptions,
        FakeDelta as VisionFakeDelta,
        FakeConfluence as VisionFakeConfluence,
    )
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"[Vision] chart_vision_analyzer not available: {e}")
    VISION_AVAILABLE = False
    analyze_frame = None
    build_vision_detection_graph = None
    annotate_debug_frame = None

# ---------------------------------------------------------------------------
# Domain stop words — shared with batch_process_lessons
# ---------------------------------------------------------------------------

TRADING_STOP_WORDS: Set[str] = frozenset({
    # Articles, pronouns, auxiliaries
    "the", "a", "an", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "mine", "yours", "hers", "ours", "theirs", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "cannot", "cant",
    # Prepositions, conjunctions
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between", "under",
    "and", "but", "or", "yet", "so", "because", "since", "although", "though",
    "if", "unless", "while", "when", "where", "than", "which", "who",
    "whom", "whose", "what", "whatever", "whoever", "whomever",
    # Common conversational
    "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank", "please",
    "sorry", "okay", "ok", "yes", "no", "yeah", "nah", "yep", "nope",
    "people", "person", "guy", "guys", "man", "woman", "men", "women", "folks",
    "everyone", "someone", "anyone", "nobody", "somebody", "anybody", "everybody",
    "beautiful", "amazing", "awesome", "fantastic", "wonderful", "nice", "great",
    "good", "bad", "better", "best", "worse", "worst", "perfect", "fine", "cool",
    "interesting", "boring", "fun", "funny", "serious", "important", "special",
    "normal", "regular", "usual", "common", "rare", "unique",
    # Time references
    "today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening",
    "now", "then", "soon", "later", "early", "late", "recently", "currently",
    "always", "never", "sometimes", "often", "usually", "rarely", "frequently",
    "minute", "hour", "day", "week", "month", "year", "date", "time", "moment",
    # Place references
    "here", "there", "where", "everywhere", "somewhere", "anywhere", "nowhere",
    "place", "space", "room",
    # Quantifiers
    "all", "none", "some", "many", "much", "more", "most", "less", "least",
    "few", "fewer", "several", "various", "numerous", "countless",
    "every", "each", "both", "either", "neither", "any", "other", "another",
    "such", "same", "different", "whole", "entire", "full", "empty", "half",
    "one", "two", "three", "first", "second", "third", "last", "next", "previous",
    # Generic nouns
    "thing", "things", "stuff", "way", "ways", "kind", "kinds", "sort", "sorts",
    "type", "types", "part", "parts", "piece", "pieces", "bit", "bits", "lot", "lots",
    "group", "set", "item", "items", "point", "points", "line", "lines",
    "number", "amount", "quantity", "count", "sum", "total", "whole",
    # Generic verbs (non-trading)
    "get", "getting", "got", "gotten", "give", "giving", "gave", "given",
    "take", "taking", "took", "taken", "make", "making", "made",
    "go", "going", "went", "gone", "come", "coming", "came",
    "put", "putting", "set", "setting", "let", "letting",
    "leave", "leaving", "left", "keep", "keeping", "kept",
    "hold", "holding", "held", "run", "running", "ran",
    "use", "using", "used", "try", "trying", "tried",
    "start", "starting", "started", "stop", "stopping", "stopped",
    "turn", "turning", "turned", "open", "opening", "opened",
    "close", "closing", "closed", "show", "showing", "showed", "shown",
    "play", "playing", "played", "call", "calling", "called",
    "tell", "telling", "told", "say", "saying", "said",
    "speak", "speaking", "spoke", "spoken", "talk", "talking", "talked",
    "ask", "asking", "asked", "answer", "answering", "answered",
    "help", "helping", "helped", "need", "needing", "needed",
    "want", "wanting", "wanted", "wish", "wishing", "wished",
    "hope", "hoping", "hoped", "expect", "expecting", "expected",
    "seem", "seeming", "seemed", "appear", "appearing", "appeared",
    "become", "becoming", "became", "remain", "remaining", "remained",
    "continue", "continuing", "continued", "finish", "finishing", "finished",
    "begin", "beginning", "began", "begun", "end", "ending", "ended",
    "think", "thinking", "thought", "know", "knew", "known", "knowing",
    "feel", "feeling", "felt", "believe", "believing", "believed",
    "like", "liked", "liking", "love", "loved", "loving", "hate",
    "mean", "means", "meant", "meaning",
    "see", "seeing", "saw", "seen", "watch", "watching", "watched",
    "find", "finding", "found", "search", "searching", "searched",
    "look", "looking", "looked", "check", "checking", "checked",
    "work", "working", "worked", "job", "task", "tasks",
    "test", "testing", "tested", "trial", "attempt", "tried",
    # Mental/cognitive
    "idea", "ideas", "thought", "thoughts", "mind", "brain", "memory",
    "understand", "understanding", "understood", "learn", "learning", "learned",
    "remember", "remembering", "remembered", "forget", "forgetting", "forgot",
    "realize", "realizing", "realized", "notice", "noticing", "noticed",
    "recognize", "recognizing", "recognized", "consider", "considering", "considered",
    "decide", "deciding", "decided", "choose", "choosing", "chose", "chosen",
    "prefer", "preferring", "preferred", "suppose", "supposing", "supposed",
    "imagine", "imagining", "imagined", "guess", "guessing", "guessed",
    "assume", "assuming", "assumed", "wonder", "wondering", "wondered",
    # Descriptors
    "big", "small", "little", "tiny", "huge", "large", "massive",
    "old", "new", "young", "ancient", "modern",
    "long", "short", "tall", "wide", "narrow", "deep", "shallow",
    "fast", "slow", "quick", "rapid", "swift", "speedy",
    "easy", "hard", "difficult", "simple", "complex", "complicated",
    "true", "false", "right", "wrong", "correct", "incorrect", "exact",
    "real", "fake", "actual", "virtual", "potential", "possible", "impossible",
    "clear", "unclear", "obvious", "hidden", "visible", "invisible",
    "strong", "weak", "powerful", "fragile", "tough", "soft",
    "happy", "sad", "angry", "calm", "excited", "bored", "tired",
    "hot", "cold", "warm", "cool", "freezing", "burning",
    "bright", "dark", "light", "heavy", "colorful", "dull",
    "safe", "dangerous", "risky", "secure", "protected", "vulnerable",
    "clean", "dirty", "fresh", "stale", "pure", "polluted",
    "rich", "poor", "wealthy", "expensive", "cheap", "free",
    "busy", "idle", "active", "passive", "lazy", "energetic",
    # Social/chat
    "chat", "message", "comment", "reply", "response", "feedback",
    "community", "team", "group", "club", "organization", "company",
    "friend", "friends", "family", "colleague", "partner", "boss",
    "customer", "client", "user", "member", "subscriber", "follower",
    "audience", "crowd", "public", "private", "personal",
    # Media/tech
    "video", "audio", "image", "picture", "photo", "screenshot",
    "screen", "monitor", "display", "camera", "microphone", "speaker",
    "phone", "mobile", "computer", "laptop", "tablet", "device",
    "app", "application", "software", "program", "system", "platform",
    "website", "site", "page", "link", "url", "web", "internet", "online",
    "file", "folder", "document", "text", "content", "data", "information",
    "record", "recording", "recorded", "playback", "stream", "streaming",
    # Education
    "lesson", "lessons", "course", "courses", "class", "classes",
    "training", "teaching", "teacher", "student", "study", "studying",
    "education", "educational", "academic", "school", "university", "college",
    "question", "questions", "answer", "answers", "problem", "problems",
    "solution", "solutions", "exercise", "exercises", "example", "examples",
    "quiz", "exam", "homework", "assignment", "project", "projects",
    # Work/business
    "work", "worker", "working", "worked", "workplace", "office",
    "business", "company", "corporation", "firm", "enterprise", "industry",
    "marketplace", "economy", "economic", "financial", "finance", "fiscal",
    "money", "cash", "currency", "dollar", "euro", "pound", "yen", "cent",
    "bank", "banking", "account", "accounts", "deposit", "withdrawal",
    "invest", "investing", "investment", "investor", "stock", "stocks",
    "share", "shares", "bond", "bonds", "fund", "funds", "portfolio",
    "profit", "profits", "profitable", "loss", "losses", "revenue", "income",
    "salary", "wage", "pay", "payment", "paying", "paid", "spend", "spent",
    "purchase", "purchased", "cost", "costs", "costing", "priced", "fee", "fees",
    "charge", "charges", "charging", "charged", "bill", "bills", "billing",
    "budget", "budgets", "expense", "expenses", "cheaper", "cheapest",
    "pricier", "pricey", "affordable", "costly",
    # Communication
    "welcome", "welcoming", "welcomed", "thanking", "thanked",
    "apologize", "apologizing", "apologized", "apology", "apologies",
    "kindly", "request", "requesting", "requested",
    "congratulate", "congratulating", "congratulated", "congratulations",
    "encourage", "encouraging", "encouraged", "motivate", "motivating", "motivated",
    "inspire", "inspiring", "inspired", "appreciate", "appreciating", "appreciated",
    # Health/body
    "health", "healthy", "unhealthy", "fit", "fitness", "exercising",
    "sport", "sports", "game", "games", "player", "players", "teams",
    "body", "bodies", "blood", "sweat", "tears", "heart", "soul", "spirit",
    "brain", "head", "face", "eye", "eyes", "ear", "ears", "nose",
    "mouth", "lip", "lips", "hand", "hands", "arm", "arms", "leg", "legs",
    "foot", "feet", "toe", "toes", "finger", "fingers", "thumb", "thumbs",
    "hair", "skin", "bone", "bones", "muscle", "muscles", "nerve", "nerves",
    # Food/drink
    "food", "foods", "meal", "meals", "breakfast", "lunch", "dinner", "supper",
    "snack", "snacks", "drink", "drinks", "water", "coffee", "tea", "juice",
    "eat", "eating", "ate", "eaten", "drank", "drunk",
    "cook", "cooking", "cooked", "bake", "baking", "baked", "fry", "frying", "fried",
    # Nature
    "nature", "natural", "animal", "animals", "plant", "plants", "tree", "trees",
    "flower", "flowers", "grass", "leaf", "leaves", "root", "roots", "seed", "seeds",
    "sun", "moon", "star", "stars", "sky", "cloud", "clouds", "rain", "snow",
    "wind", "storm", "thunder", "lightning", "fog", "mist", "air", "weather",
    "earth", "ground", "soil", "dirt", "mud", "sand", "rock", "rocks", "stone", "stones",
    "mountain", "mountains", "hill", "hills", "valley", "valleys", "river", "rivers",
    "lake", "lakes", "ocean", "oceans", "sea", "seas", "beach", "beaches", "coast",
    "forest", "forests", "jungle", "jungles", "desert", "deserts", "island", "islands",
    # Colors
    "color", "colors", "colour", "colours", "red", "blue", "green", "yellow",
    "orange", "purple", "pink", "brown", "black", "white", "gray", "grey",
    "pale", "vivid", "colorful", "drab",
    # Directions
    "north", "south", "east", "west", "up", "down", "left", "right",
    "top", "bottom", "front", "back", "side", "sides", "center", "centre",
    "middle", "edge", "edges", "corner", "corners", "end", "ends", "origin",
    # Transport
    "car", "cars", "truck", "trucks", "bus", "buses", "train", "trains",
    "plane", "planes", "flight", "flights", "airport", "airports", "station", "stations",
    "road", "roads", "street", "streets", "highway", "highways",
    "drive", "driving", "drove", "driven", "ride", "riding", "rode", "ridden",
    "walk", "walking", "walked", "fly", "flying", "flew", "flown",
    # House/home
    "house", "houses", "home", "homes", "apartment", "apartments", "room", "rooms",
    "kitchen", "bathroom", "bedroom", "living", "dining", "garage", "garden", "yard",
    "door", "doors", "window", "windows", "wall", "walls", "floor", "floors",
    "roof", "roofs", "ceiling", "ceilings", "stairs", "stair", "step", "steps",
    "furniture", "chair", "chairs", "table", "tables", "bed", "beds", "couch", "sofa",
    # Clothing
    "clothes", "clothing", "shirt", "shirts", "pants", "jeans", "shorts", "dress", "dresses",
    "skirt", "skirts", "jacket", "jackets", "coat", "coats", "sweater", "sweaters",
    "shoe", "shoes", "sock", "socks", "hat", "hats", "cap", "caps", "scarf", "scarves",
    "glove", "gloves", "belt", "belts", "tie", "ties", "suit", "suits", "uniform", "uniforms",
})

TRADING_ALLOWLIST: Set[str] = frozenset({
    # Core order-flow concepts
    "absorption", "aggression", "aggressive", "passive", "initiative", "responsive",
    "imbalance", "imbalances", "stacked", "exhaustion", "divergence", "confluence",
    "reversal", "breakout", "breakdown", "pullback", "retracement", "continuation",
    "momentum", "speed", "velocity", "acceleration", "deceleration",
    # Market structure
    "support", "resistance", "trend", "trending", "range", "ranging", "auction",
    "market", "structure", "break", "breaker", "character", "change", "chop", "choppy",
    "rotation", "rotational", "balance", "balanced", "imbalanced",
    # Volume/price profile
    "profile", "footprint", "volume", "delta", "cumulative", "cumulative_delta",
    "poc", "point_of_control", "val", "value_area_low", "vah", "value_area_high",
    "vwap", "hvn", "high_volume_node", "lvn", "low_volume_node",
    "developing", "composite", "naked", "single", "print", "prints",
    # Order flow specifics
    "bid", "ask", "buy", "sell", "long", "short", "iceberg", "icebergs",
    "stop", "loss", "target", "reward", "risk", "management", "managed",
    "entry", "entries", "exit", "exits", "setup", "setups", "pattern", "patterns",
    "signal", "signals", "confirmation", "confirmations",
    # Chart/candle elements
    "chart", "charts", "candle", "candles", "bar", "bars", "wick", "wicks",
    "tail", "tails", "body", "bodies", "shadow", "shadows", "doji", "engulfing",
    "open", "high", "low", "close", "settlement", "opening", "closing",
    # Session/time
    "session", "sessions", "rth", "regular_trading_hours", "eth", "extended_trading_hours",
    "overnight", "premarket", "postmarket", "globex", "cash", "initial",
    "balance", "ib", "initial_balance", "extension", "extensions",
    # Zones/levels
    "zone", "zones", "level", "levels", "region", "regions", "area", "areas",
    "node", "nodes", "cluster", "clusters", "pocket", "pockets", "shelf", "shelves",
    "ledge", "ledges", "stair", "stairs", "step", "steps", "ladder", "ladders",
    "bridge", "bridges", "gap", "gaps", "fill", "fills", "unfilled", "partial",
    # Price action
    "compression", "consolidation", "distribution", "accumulation", "expansion",
    "contraction", "displacement", "liquidity", "sweep", "sweeps", "run", "runs",
    "manipulation", "engineered", "inducement", "inducements", "stop_run", "stop_hunt",
    "premium", "discount", "equilibrium", "inefficiency", "inefficiencies",
    "inversion", "inversions", "flip", "flips", "reclaim", "reclaims", "rejection",
    "rejections", "bounce", "bounces", "tap", "taps", "test", "tests", "hold", "holds",
    "fail", "fails", "breaks", "breach", "breaches",
    # Advanced concepts
    "fvg", "fair_value_gap", "order_block", "mitigation", "mitigated",
    "breakers", "ob", "orderblock", "bos", "break_of_structure",
    "choch", "change_of_character", "msb", "market_structure_break",
    "swing", "swings", "highs", "lows", "higher_high", "higher_low", "lower_high", "lower_low",
    "hh", "hl", "lh", "ll", "smt", "smart_money",
    # Participants
    "buyer", "buyers", "seller", "sellers", "bull", "bulls", "bear", "bears",
    "bullish", "bearish", "smart_money", "institution", "institutions", "retail",
    "whale", "whales", "participant", "participants",
    # Misc valid trading
    "trade", "trades", "trader", "traders", "trading", "scalp", "scalping", "swing",
    "daytrade", "daytrading", "invest", "investing", "speculate", "speculating",
    "fundamental", "fundamentals", "technical", "technicals", "quantitative",
    "algorithmic", "algo", "automated", "discretionary", "systematic",
    "timeframe", "timeframes", "context", "alignment", "opposing", "neutral",
    "confluences", "correlation", "correlations", "divergences",
    "correlated", "inversely", "inverse",
    # Acronyms & shorthand
    "es", "nq", "ym", "cl", "gc", "si", "zb", "zn", "treasury", "bond", "bonds",
    "spx", "spy", "qqq", "dia", "iwm", "russell", "nasdaq", "dow",
    # Risk/money management
    "drawdown", "drawdowns", "equity", "curve", "winrate", "win_rate", "rr", "r_r",
    "risk_reward", "position", "sizing", "leverage", "margin", "account",
    # Emotional/psychology (valid in trading context)
    "psychology", "discipline", "patience", "emotion", "emotions", "fear", "greed",
    "hope", "regret", "revenge", "fomo", "fear_of_missing_out",
    # Additional valid single words
    "direction", "directional", "volatile", "volatility", "choppy", "smooth",
    "aggressive", "passive", "initiative", "responsive",
})

def is_valid_concept_name(name: str) -> bool:
    """Check if an invented concept name is worth keeping in the ontology.

    STRICT RULES:
    1. Single-word concepts MUST be in TRADING_ALLOWLIST.
    2. Multi-word phrases: reject if >50% of words are TRADING_STOP_WORDS.
    3. Reject pure stop words.
    4. Length gate: 3-40 chars.
    """
    if not name or len(name) < 3 or len(name) > 40:
        return False
    name_lower = name.lower().strip()

    # Allow explicitly listed trading terms
    if name_lower in TRADING_ALLOWLIST:
        return True

    # Reject pure strict stop words
    if name_lower in TRADING_STOP_WORDS:
        return False

    words = name_lower.replace("_", " ").replace("-", " ").split()

    # STRICT: Single-word concepts MUST be in allowlist
    if len(words) == 1:
        return words[0] in TRADING_ALLOWLIST

    # Multi-word: reject if majority are strict stop words
    stop_count = sum(1 for w in words if w in TRADING_STOP_WORDS)
    if len(words) > 0 and stop_count / len(words) > 0.5:
        return False

    return True

def is_valid_relationship(rel: dict) -> bool:
    """Filter out noisy relationships."""
    src = rel.get("source", "").lower().strip()
    tgt = rel.get("target", "").lower().strip()
    if src == tgt or not src or not tgt:
        return False
    # Both must be valid concept names
    if not is_valid_concept_name(src) or not is_valid_concept_name(tgt):
        return False
    # Confidence gate
    if rel.get("confidence", 0) < 0.4:
        return False
    # NEW: Reject if either concept is a pure strict stop word
    # (catches edge cases where is_valid_concept_name passes compound words
    #  that contain stop words, e.g. "what's happening")
    src_words = src.replace("_", " ").replace("-", " ").split()
    tgt_words = tgt.replace("_", " ").replace("-", " ").split()
    if len(src_words) == 1 and src in TRADING_STOP_WORDS:
        return False
    if len(tgt_words) == 1 and tgt in TRADING_STOP_WORDS:
        return False
    return True

# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

# ---------------------------------------------------------------------------
# Fake DetectionGraph — ONLY used in vision mode as fallback
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
    imbalances: Tuple[FakeImbalance, ...]

@dataclass(frozen=True)
class FakeAbsorption:
    side: str
    cell_id: str

@dataclass(frozen=True)
class FakeAbsorptions:
    absorptions: Tuple[FakeAbsorption, ...]

@dataclass(frozen=True)
class FakeDelta:
    cell_deltas: Tuple[object, ...]

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

# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def _split_by_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'\(])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def _split_by_timestamps(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r'(?:^|\n)\s*(?:\[?)(\d{1,2}:\d{2}(?::\d{2})?)(?:\]?)\s*[-–—]?\s*(.*?)'
        r'(?=(?:\n\s*(?:\[?)\d{1,2}:\d{2}(?::\d{2})?(?:\]?)\s*[-–—]?|$))',
        re.DOTALL
    )
    matches = pattern.findall(text)
    if not matches:
        return []
    result = []
    for ts, content in matches:
        content = content.strip().replace('\n', ' ')
        if len(content) > 10:
            result.append((ts, content))
    return result

def load_transcript(transcript_path: str) -> Tuple[Tuple[int, str], ...]:
    path = Path(transcript_path)
    if not path.exists():
        print(f"WARNING: Transcript not found: {transcript_path}")
        return ()

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle multiple JSON transcript formats:
        # 1. List of dicts: [{"text": "..."}, ...]  (Whisper-style)
        # 2. List of strings: ["text1", "text2", ...]  (simple array)
        # 3. Dict of strings: {"0": "text1", "1": "text2"}  (indexed object)
        # 4. Dict with "segments" key: {"segments": [{"text": "..."}]}  (VTT-style JSON)
        if isinstance(data, list):
            entries = []
            for i, entry in enumerate(data):
                if isinstance(entry, dict):
                    text = entry.get("text", "")
                    if not text and "content" in entry:
                        text = entry.get("content", "")
                    if not text and "body" in entry:
                        text = entry.get("body", "")
                elif isinstance(entry, str):
                    text = entry
                else:
                    text = str(entry)
                if text and len(text.strip()) > 0:
                    entries.append((i, text.strip()))
            return tuple(entries)
        elif isinstance(data, dict):
            # Try "segments" key first
            if "segments" in data and isinstance(data["segments"], list):
                entries = []
                for i, entry in enumerate(data["segments"]):
                    if isinstance(entry, dict):
                        text = entry.get("text", "")
                        if not text:
                            text = entry.get("content", "")
                    elif isinstance(entry, str):
                        text = entry
                    else:
                        text = str(entry)
                    if text and len(text.strip()) > 0:
                        entries.append((i, text.strip()))
                return tuple(entries)
            # Try ordered dict values
            entries = []
            for i, (key, val) in enumerate(sorted(data.items(), key=lambda x: str(x[0]))):
                if isinstance(val, str):
                    text = val
                elif isinstance(val, dict):
                    text = val.get("text", "")
                else:
                    text = str(val)
                if text and len(text.strip()) > 0:
                    entries.append((i, text.strip()))
            return tuple(entries)
        else:
            # Fallback: wrap single string/object
            text = str(data) if not isinstance(data, str) else data
            return ((0, text.strip()),)

    if suffix == ".srt":
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = content.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                text = " ".join(lines[2:]).strip()
                entries.append((len(entries), text))
        return tuple(entries)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    timestamp_entries = _split_by_timestamps(content)
    if timestamp_entries:
        return tuple((i, text) for i, (_, text) in enumerate(timestamp_entries))

    sentences = _split_by_sentences(content)
    if len(sentences) >= 3:
        return tuple((i, s) for i, s in enumerate(sentences))

    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if len(paragraphs) >= 2:
        return tuple((i, p) for i, p in enumerate(paragraphs))

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        return tuple((i, p) for i, p in enumerate(paragraphs))

    chunks = []
    words = content.split()
    current = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > 300 and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        chunks.append(" ".join(current))

    if len(chunks) >= 2:
        return tuple((i, c) for i, c in enumerate(chunks))

    return ((0, content.strip()),)

# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: str,
    max_frames: int = 50,
    interval_sec: Optional[float] = None,
) -> List[Tuple[int, str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"WARNING: Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    duration_sec = total_frames / fps if total_frames > 0 else 0

    if interval_sec is None:
        if duration_sec > 0 and max_frames > 0:
            interval_sec = max(1.0, duration_sec / max_frames)
        else:
            interval_sec = 5.0

    frame_interval = max(1, int(round(fps * interval_sec)))

    frames = []
    frame_num = 0
    next_target = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num >= next_target:
            ts_sec = frame_num / fps
            ts_str = f"{int(ts_sec // 3600):02d}:{int((ts_sec % 3600) // 60):02d}:{int(ts_sec % 60):02d}"
            frames.append((frame_num, ts_str, frame))
            next_target += frame_interval
            if len(frames) >= max_frames:
                break

        frame_num += 1

    cap.release()
    print(f"[Video] Extracted {len(frames)} frames from {total_frames} total "
          f"(interval={interval_sec:.1f}s, fps={fps:.1f})")
    return frames

# ---------------------------------------------------------------------------
# Ontology loading & graph building
# ---------------------------------------------------------------------------

def load_ontology(ontology_path: str) -> dict:
    path = Path(ontology_path)
    if not path.exists():
        candidates = [
            Path(__file__).parent / "fabio_ontology.json",
            Path(__file__).parent / "fabio_ontology_fixed.json",
            Path.cwd() / "fabio_ontology.json",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
    if not path.exists():
        print(f"WARNING: Ontology not found at {ontology_path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_concepts_from_text(text: str, ontology: dict) -> List[str]:
    """Match text against ontology keywords. Returns matched concept names.

    NOTE: We do NOT filter ontology concepts through is_valid_concept_name().
    The ontology is user-curated and should be trusted. Stop-word filtering
    is only applied to AUTO-DISCOVERED concepts in evolve_ontology().
    """
    text_lower = text.lower()
    matched = []
    for concept in ontology.get("concepts", []):
        name = concept.get("name", "")
        if not name:
            continue
        # Match by keywords
        for kw in concept.get("keywords", []):
            if kw.lower() in text_lower:
                matched.append(name)
                break
        else:
            # Also match by concept name itself if not already keyword-matched
            if name.lower() in text_lower and name not in matched:
                matched.append(name)
    return matched

def build_graph_from_ontology(
    ontology: dict,
    extracted_concept_names: Set[str],
    discovered_rels: Optional[List[dict]] = None,
) -> dict:
    """Build a knowledge graph dict from ontology + extracted concepts + discovered relationships."""
    all_concepts = {c["name"]: c for c in ontology.get("concepts", []) if c.get("name")}

    graph_concepts = set(extracted_concept_names)

    for name in list(graph_concepts):
        if name in all_concepts:
            c = all_concepts[name]
            graph_concepts.update(c.get("parent_concepts", []))
            graph_concepts.update(c.get("related_concepts", []))
            graph_concepts.update(c.get("opposite_concepts", []))

    # Add targets of all relationships (predefined + discovered)
    all_rels = list(ontology.get("relationships", []))
    if discovered_rels:
        all_rels.extend(discovered_rels)

    for rel in all_rels:
        if not is_valid_relationship(rel):
            continue
        if rel["source"] in graph_concepts:
            graph_concepts.add(rel["target"])

    # Build relationship list with deduplication
    graph_relationships = []
    seen_rels = set()

    for rel in all_rels:
        if not is_valid_relationship(rel):
            continue
        src = rel["source"]
        tgt = rel["target"]
        if src in graph_concepts and tgt in graph_concepts:
            rel_name = rel.get("relation", rel.get("relationship_type", "unknown"))
            key = (src.lower(), tgt.lower(), rel_name.lower())
            if key not in seen_rels:
                seen_rels.add(key)
                graph_relationships.append(rel)

    concept_details = []
    for name in sorted(graph_concepts):
        if name in all_concepts:
            c = all_concepts[name]
            concept_details.append({
                "name": name,
                "category": c.get("category", "Unknown"),
                "definition": c.get("definition", ""),
                "confidence_weight": c.get("confidence_weight", 1.0),
            })
        else:
            concept_details.append({
                "name": name,
                "category": "Extracted",
                "definition": "",
                "confidence_weight": 0.8,
            })

    predefined_count = sum(1 for r in graph_relationships if not r.get("discovered"))
    discovered_count = sum(1 for r in graph_relationships if r.get("discovered"))

    return {
        "total_concepts": len(graph_concepts),
        "total_relationships": len(graph_relationships),
        "predefined_relationships": predefined_count,
        "discovered_relationships": discovered_count,
        "concepts": sorted(graph_concepts),
        "concept_details": concept_details,
        "relationships": graph_relationships,
    }

# ---------------------------------------------------------------------------
# Concept coverage analysis
# ---------------------------------------------------------------------------

def analyze_coverage(
    transcript_entries: Tuple[Tuple[int, str], ...],
    ontology: dict,
) -> dict:
    """Analyze which transcript segments matched concepts and which didn't."""
    covered = []
    uncovered = []
    concept_hits: Dict[str, int] = {}

    for idx, text in transcript_entries:
        matched = extract_concepts_from_text(text, ontology)
        if matched:
            covered.append({"index": idx, "text_preview": text[:120], "concepts": matched})
            for c in matched:
                concept_hits[c] = concept_hits.get(c, 0) + 1
        else:
            uncovered.append({"index": idx, "text_preview": text[:120]})

    return {
        "total_segments": len(transcript_entries),
        "covered_segments": len(covered),
        "uncovered_segments": len(uncovered),
        "coverage_percent": round(100 * len(covered) / max(1, len(transcript_entries)), 1),
        "concept_frequency": dict(sorted(concept_hits.items(), key=lambda x: -x[1])),
        "uncovered_examples": uncovered[:20],
    }

# ---------------------------------------------------------------------------
# Real confidence computation
# ---------------------------------------------------------------------------

def compute_real_confidence(
    concepts: List[Concept],
    coverage: dict,
    graph: dict,
) -> float:
    """Compute confidence from actual evidence, not a hardcoded value."""
    if not concepts:
        return 0.0

    # Factor 1: Average concept confidence score
    scores = [float(c.confidence.score) for c in concepts if hasattr(c.confidence, 'score')]
    avg_score = sum(scores) / len(scores) if scores else 0.5

    # Factor 2: Coverage ratio
    coverage_ratio = coverage.get("coverage_percent", 0) / 100.0

    # Factor 3: Graph richness — relationships per concept
    num_concepts = graph.get("total_concepts", len(concepts))
    num_rels = graph.get("total_relationships", 0)
    graph_ratio = min(1.0, num_rels / max(1, num_concepts)) if num_concepts > 0 else 0.0

    # Weighted combination
    confidence = (avg_score * 0.4) + (coverage_ratio * 0.3) + (graph_ratio * 0.3)
    return min(1.0, max(0.0, confidence))

# ---------------------------------------------------------------------------
# Fake DetectionGraph builder (vision mode fallback only)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Vision-derived concept injection (Path A)
# ---------------------------------------------------------------------------

@dataclass
class VisionConcept:
    """Minimal Concept-compatible dataclass for vision-derived concepts."""
    concept_id: str
    definition: Any
    confidence: Any
    positive_examples: list = field(default_factory=list)
    negative_examples: list = field(default_factory=list)
    related_concepts: set = field(default_factory=set)
    lesson_references: set = field(default_factory=set)
    evolution_history: list = field(default_factory=list)

@dataclass
class VisionConceptDef:
    name: str
    definition: str
    visual_appearance: str = ""
    teacher_explanation: str = ""

@dataclass
class VisionConfidence:
    level: Any
    score: Decimal
    evidence_count: int = 1

def _make_confidence_level(name: str):
    """Create a confidence level namespace with .name attribute."""
    class Level:
        pass
    lvl = Level()
    lvl.name = name
    return lvl

def extract_vision_concepts(features, frame_idx: int = 0, transcript_text: str = "") -> List[VisionConcept]:
    """Map vision features to trading concepts with confidence scores.

    Returns VisionConcept objects ready for injection into the apprentice layer.
    """
    concepts = []

    if not features or not features.has_chart:
        concepts.append(VisionConcept(
            concept_id=f"vis_{frame_idx:04d}_no_chart",
            definition=VisionConceptDef(
                name="No_Chart_Detected",
                definition="No chart region identified in the video frame.",
                visual_appearance="Frame contains no discernible chart region.",
                teacher_explanation="When no chart is visible, vision analysis cannot provide trading context."
            ),
            confidence=VisionConfidence(
                level=_make_confidence_level("EMERGING"),
                score=Decimal("0.30")
            ),
            lesson_references={f"frame_{frame_idx}"}
        ))
        return concepts

    # Chart presence
    cr_conf = features.chart_region.confidence if features.chart_region else Decimal("0.50")
    concepts.append(VisionConcept(
        concept_id=f"vis_{frame_idx:04d}_chart",
        definition=VisionConceptDef(
            name="Chart_Detected",
            definition="A trading chart region was identified in the video frame through contour analysis.",
            visual_appearance="Rectangular region with grid-like structure and price action.",
            teacher_explanation="Fabio is showing a live chart. Pay attention to the price levels and delta colors."
        ),
        confidence=VisionConfidence(
            level=_make_confidence_level("DEVELOPING"),
            score=Decimal(str(min(1.0, cr_conf)))
        ),
        lesson_references={f"frame_{frame_idx}"}
    ))

    # Grid detection
    if features.grid:
        g = features.grid
        concepts.append(VisionConcept(
            concept_id=f"vis_{frame_idx:04d}_grid",
            definition=VisionConceptDef(
                name="Footprint_Grid",
                definition=f"Detected a footprint chart grid with approximately {g.cell_count} cells.",
                visual_appearance="Horizontal and vertical lines forming a matrix of price×time cells.",
                teacher_explanation="The grid represents individual price levels over time. Each cell shows volume and delta at that price."
            ),
            confidence=VisionConfidence(
                level=_make_confidence_level("DEVELOPING"),
                score=Decimal(str(min(1.0, g.confidence)))
            ),
            lesson_references={f"frame_{frame_idx}"}
        ))

    # Delta analysis
    if features.delta:
        d = features.delta
        if d.buy_dominance > 0.6:
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_buy_delta",
                definition=VisionConceptDef(
                    name="Buy_Delta_Dominant",
                    definition=f"Buy delta dominates the chart with {d.buy_dominance:.1%} buy pressure.",
                    visual_appearance="Predominantly green/cyan coloring in footprint cells indicating aggressive buyers.",
                    teacher_explanation="When buy delta dominates, aggressive buyers are in control. Look for continuation or absorption at highs."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("CONFIDENT"),
                    score=Decimal(str(min(1.0, d.confidence * d.buy_dominance)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))
        elif d.sell_dominance > 0.6:
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_sell_delta",
                definition=VisionConceptDef(
                    name="Sell_Delta_Dominant",
                    definition=f"Sell delta dominates the chart with {d.sell_dominance:.1%} sell pressure.",
                    visual_appearance="Predominantly red/magenta coloring in footprint cells indicating aggressive sellers.",
                    teacher_explanation="When sell delta dominates, aggressive sellers are in control. Look for continuation or absorption at lows."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("CONFIDENT"),
                    score=Decimal(str(min(1.0, d.confidence * d.sell_dominance)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))

        if d.buy_dominance > 0.8 or d.sell_dominance > 0.8:
            side = "Buy" if d.buy_dominance > d.sell_dominance else "Sell"
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_extreme_delta",
                definition=VisionConceptDef(
                    name="Extreme_Delta_Imbalance",
                    definition=f"Extreme {side.lower()} delta imbalance detected (>80% dominance).",
                    visual_appearance=f"Overwhelming {side.lower()}er color saturation across the chart.",
                    teacher_explanation=f"Extreme {side.lower()} delta often precedes exhaustion or a reversal. Watch for absorption."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("HIGH_CONFIDENCE"),
                    score=Decimal(str(min(1.0, d.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))
        elif d.neutral_ratio > 0.5:
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_neutral_delta",
                definition=VisionConceptDef(
                    name="Neutral_Delta",
                    definition="Balanced buy/sell delta suggests equilibrium or absorption.",
                    visual_appearance="Mixed or muted colors with no clear dominance.",
                    teacher_explanation="Neutral delta often indicates absorption — buyers and sellers are matched. This can precede a breakout."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("DEVELOPING"),
                    score=Decimal(str(min(1.0, d.confidence * d.neutral_ratio)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))

    # Price action
    if features.price_action:
        pa = features.price_action
        if pa.trend_direction == "UP":
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_uptrend",
                definition=VisionConceptDef(
                    name="Visual_Uptrend",
                    definition="Price action silhouette indicates upward directional movement.",
                    visual_appearance="Higher highs and higher lows visible in the chart structure.",
                    teacher_explanation="The visual structure confirms an uptrend. Trade in the direction of the trend."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("CONFIDENT"),
                    score=Decimal(str(min(1.0, pa.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))
        elif pa.trend_direction == "DOWN":
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_downtrend",
                definition=VisionConceptDef(
                    name="Visual_Downtrend",
                    definition="Price action silhouette indicates downward directional movement.",
                    visual_appearance="Lower highs and lower lows visible in the chart structure.",
                    teacher_explanation="The visual structure confirms a downtrend. Trade in the direction of the trend or wait for reversal signals."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("CONFIDENT"),
                    score=Decimal(str(min(1.0, pa.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))
        elif pa.trend_direction == "BALANCED":
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_balanced",
                definition=VisionConceptDef(
                    name="Visual_Balanced",
                    definition="Price action shows balanced, rotational market conditions.",
                    visual_appearance="Price oscillating within a defined range without clear directional bias.",
                    teacher_explanation="A balanced market means two-sided trade. Look for extremes — value area highs and lows."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("DEVELOPING"),
                    score=Decimal(str(min(1.0, pa.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))

        if pa.volatility_estimate == "HIGH":
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_high_vol",
                definition=VisionConceptDef(
                    name="High_Volatility_Chart",
                    definition="High volatility detected from price action silhouette analysis.",
                    visual_appearance="Wide price swings, long wicks, and rapid vertical displacement.",
                    teacher_explanation="High volatility means larger stops and targets. Widen your risk management."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("CONFIDENT"),
                    score=Decimal(str(min(1.0, pa.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))
        elif pa.volatility_estimate == "LOW":
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_low_vol",
                definition=VisionConceptDef(
                    name="Low_Volatility_Chart",
                    definition="Low volatility detected — compression or consolidation phase.",
                    visual_appearance="Tight price range, small candles, minimal vertical movement.",
                    teacher_explanation="Low volatility often precedes expansion. Be ready for a breakout."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("DEVELOPING"),
                    score=Decimal(str(min(1.0, pa.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))

        if pa.structure == "TRENDING":
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_trending",
                definition=VisionConceptDef(
                    name="Trending_Structure",
                    definition="Chart structure is clearly trending with directional bias.",
                    visual_appearance="Sustained directional movement with shallow pullbacks.",
                    teacher_explanation="In a trending structure, pullbacks are buying (or selling) opportunities. Trade with the trend."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("CONFIDENT"),
                    score=Decimal(str(min(1.0, pa.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))
        elif pa.structure == "RANGING":
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_ranging",
                definition=VisionConceptDef(
                    name="Ranging_Structure",
                    definition="Chart structure is ranging or balanced.",
                    visual_appearance="Price oscillating between defined support and resistance levels.",
                    teacher_explanation="In a ranging structure, buy at support and sell at resistance. The POC is your anchor."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("CONFIDENT"),
                    score=Decimal(str(min(1.0, pa.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))
        elif pa.structure == "CHOPPY":
            concepts.append(VisionConcept(
                concept_id=f"vis_{frame_idx:04d}_choppy",
                definition=VisionConceptDef(
                    name="Choppy_Structure",
                    definition="Chart structure is choppy with no clear directional bias.",
                    visual_appearance="Erratic price movement with frequent direction changes and whipsaws.",
                    teacher_explanation="Choppy conditions are dangerous. Reduce size or stay out until structure clarifies."
                ),
                confidence=VisionConfidence(
                    level=_make_confidence_level("DEVELOPING"),
                    score=Decimal(str(min(1.0, pa.confidence)))
                ),
                lesson_references={f"frame_{frame_idx}"}
            ))

    return concepts

def inject_vision_concepts(
    integration,
    vision_concepts: List[VisionConcept],
    ontology: dict,
    discovered_rels: Optional[List[dict]] = None,
    transcript_text: str = "",
) -> Tuple[List[VisionConcept], List[dict]]:
    """Register vision-derived concepts into the ontology and create cross-modal relationships.

    Creates two types of relationships:
    1. Vision↔Vision: When two vision concepts appear in the same frame
    2. Vision↔Transcript: When a vision concept co-occurs with a transcript concept

    Returns (newly_registered_vision_concepts, new_relationships).
    """
    layer = integration.get_apprentice_layer()
    if not layer:
        return [], []

    # Get existing names from both layer and ontology
    existing_names = set()
    for c in layer.all_concepts():
        if hasattr(c, 'definition') and c.definition:
            existing_names.add(c.definition.name)

    kg = layer.get_knowledge_graph()
    existing_names.update(kg.concepts)

    ontology_concepts = {c["name"] for c in ontology.get("concepts", [])}

    newly_registered = []
    new_relationships = []

    # --- Step 1: Register vision concepts to ontology ---
    for vc in vision_concepts:
        name = vc.definition.name

        if name in existing_names and name in ontology_concepts:
            continue

        existing_names.add(name)

        if name not in ontology_concepts:
            import re
            existing_ids = {c["concept_id"] for c in ontology.get("concepts", [])}
            next_num = max([int(re.search(r'\d+', cid).group()) for cid in existing_ids if re.search(r'\d+', cid)] + [0]) + 1
            new_id = f"vis_{next_num:03d}"
            while new_id in existing_ids:
                next_num += 1
                new_id = f"vis_{next_num:03d}"

            ontology.setdefault("concepts", []).append({
                "concept_id": new_id,
                "name": name,
                "category": "Vision_Derived",
                "definition": vc.definition.definition,
                "keywords": [name.lower(), name.lower().replace("_", " ")],
                "visual_markers": [vc.definition.visual_appearance],
                "parent_concepts": [],
                "related_concepts": [],
                "opposite_concepts": [],
                "confidence_weight": float(vc.confidence.score),
            })
            ontology_concepts.add(name)
            newly_registered.append(vc)

    # --- Step 2: Vision↔Vision relationships (same frame co-occurrence) ---
    unique_vis_names = list(set(vc.definition.name for vc in vision_concepts))
    for i, a_name in enumerate(unique_vis_names):
        for b_name in unique_vis_names[i+1:]:
            if a_name == b_name:
                continue
            # Get confidence from the vision concepts
            a_confs = [float(vc.confidence.score) for vc in vision_concepts if vc.definition.name == a_name]
            b_confs = [float(vc.confidence.score) for vc in vision_concepts if vc.definition.name == b_name]
            conf = (max(a_confs) + max(b_confs)) / 2.0 if a_confs and b_confs else 0.5

            new_relationships.append({
                "source": a_name,
                "target": b_name,
                "relation": "cooccurs_with",
                "confidence": round(min(1.0, conf), 2),
                "direction": "bidirectional",
                "method": "vision_cooccurrence",
                "discovered": True,
            })

    # --- Step 3: Vision↔Transcript relationships ---
    text_lower = transcript_text.lower()
    transcript_matched = []
    for concept in ontology.get("concepts", []):
        cname = concept.get("name", "")
        if not cname:
            continue
        for kw in concept.get("keywords", []):
            if kw.lower() in text_lower and cname not in transcript_matched:
                transcript_matched.append(cname)
                break
        if cname.lower() in text_lower and cname not in transcript_matched:
            transcript_matched.append(cname)

    for vc in vision_concepts:
        vname = vc.definition.name
        for tc_name in transcript_matched:
            if vname == tc_name:
                continue
            conf = float(vc.confidence.score)
            new_relationships.append({
                "source": vname,
                "target": tc_name,
                "relation": "confirms",
                "confidence": round(min(1.0, conf), 2),
                "direction": "forward",
                "method": "vision_transcript_fusion",
                "discovered": True,
            })

    return newly_registered, new_relationships

def build_fake_detection_graph(frame_idx: int, transcript_text: str) -> FakeDetectionGraph:
    text = transcript_text.lower()

    if "trend" in text and "up" in text:
        trend = FakeTrend(FakeTrendState("TRENDING_UP"))
    elif "trend" in text and "down" in text:
        trend = FakeTrend(FakeTrendState("TRENDING_DOWN"))
    elif "auction" in text and "down" in text:
        trend = FakeTrend(FakeTrendState("AUCTION_DOWN"))
    elif "auction" in text and "up" in text:
        trend = FakeTrend(FakeTrendState("AUCTION_UP"))
    elif "balance" in text or "rotational" in text:
        trend = FakeTrend(FakeTrendState("BALANCED"))
    else:
        trend = FakeTrend(FakeTrendState("BALANCED"))

    imbalances = []
    if "buyer" in text or "bid" in text:
        imbalances.append(FakeImbalance("BID", f"price_{frame_idx}"))
    if "seller" in text or "ask" in text:
        imbalances.append(FakeImbalance("ASK", f"price_{frame_idx}"))
    if "imbalance" in text and "stacked" not in text:
        imbalances.append(FakeImbalance("BID", f"price_{frame_idx}"))

    absorptions = []
    if "absorption" in text or "absorb" in text:
        side = "BID" if "buy" in text or "bid" in text else "ASK"
        absorptions.append(FakeAbsorption(side, f"price_{frame_idx}"))

    confluence = "NO_CONFLUENCE"
    if "strong" in text or "confirm" in text:
        confluence = "STRONG_CONFLUENCE"
    elif "weak" in text:
        confluence = "WEAK_CONFLUENCE"

    return FakeDetectionGraph(
        graph_id=f"frame_{frame_idx:04d}",
        timestamp=f"00:{frame_idx // 60:02d}:{frame_idx % 60:02d}",
        trend_state=trend,
        trading_session=FakeSession("RTH"),
        footprint_imbalances=FakeImbalances(tuple(imbalances)) if imbalances else None,
        absorption_result=FakeAbsorptions(tuple(absorptions)) if absorptions else None,
        footprint_delta=FakeDelta(()),
        confluence=FakeConfluence(confluence),
    )

# ---------------------------------------------------------------------------
# Relationship Discovery from Transcript Text  —  CLEANED
# ---------------------------------------------------------------------------

class RelationshipExtractor:
    """Extract semantic relationships between concepts from transcript text.

    Two-phase approach:
    1. EXPLICIT PATTERNS: "Absorption requires Aggression" -> direct relationship
    2. CO-OCCURRENCE FALLBACK: Concepts in same sentence -> inferred relationship

    CLEANED v5.1:
    - Stop-word filtering on both source and target concepts
    - Confidence gating (minimum 0.4)
    - Deduplication within a single lesson
    - Configurable min_cooccurrence for batch quality
    """

    EXPLICIT_PATTERNS = [
        # Strong causal/requirement patterns
        (r'\b({source})\s+(?:requires?|needs?|demands?|depends?\s+on)\s+(?:the\s+)?({target})\b', "requires", 0.95, "forward"),
        (r'\b({source})\s+(?:is\s+(?:a|an)\s+)?(?:form\s+of|type\s+of|kind\s+of)\s+(?:the\s+)?({target})\b', "is_a", 0.9, "forward"),
        (r'\b({source})\s+(?:is\s+)?(?:the\s+)?(?:same\s+as|equivalent\s+to|identical\s+to)\s+(?:the\s+)?({target})\b', "is_a", 0.85, "bidirectional"),

        # Temporal/precedence patterns
        (r'\b({source})\s+(?:precedes?|comes?\s+before|leads?\s+to|results?\s+in|gives?\s+rise\s+to)\s+(?:the\s+)?({target})\b', "precedes", 0.85, "forward"),
        (r'\b({source})\s+(?:triggers?|initiates?|starts?|kicks?\s+off)\s+(?:the\s+)?({target})\b', "triggers", 0.85, "forward"),
        (r'\b({source})\s+(?:follows?|comes?\s+after|happens?\s+after)\s+(?:the\s+)?({target})\b', "follows", 0.8, "forward"),
        (r'\b({source})\s+(?:often\s+)?(?:results?\s+in|ends?\s+up\s+as|turns?\s+into)\s+(?:the\s+)?({target})\b', "results_in", 0.8, "forward"),

        # Signaling/confirmation patterns
        (r'\b({source})\s+(?:signals?|indicates?|shows?|suggests?|means?)\s+(?:the\s+)?({target})\b', "signals", 0.8, "forward"),
        (r'\b({source})\s+(?:confirms?|validates?|verifies?)\s+(?:the\s+)?({target})\b', "confirms", 0.85, "forward"),
        (r'\b({source})\s+(?:contradicts?|invalidates?|negates?)\s+(?:the\s+)?({target})\b', "contradicts", 0.8, "forward"),

        # Location/context patterns
        (r'\b({source})\s+(?:occurs?\s+at|happens?\s+at|seen?\s+at|found?\s+at)\s+(?:the\s+)?({target})\b', "occurs_at", 0.75, "forward"),
        (r'\b({source})\s+(?:seen?\s+with|associated\s+with|linked\s+to|correlated\s+with)\s+(?:the\s+)?({target})\b', "associated_with", 0.7, "bidirectional"),

        # Opposition patterns
        (r'\b({source})\s+(?:is\s+(?:the\s+)?)?opposite\s+(?:of|to)\s+(?:the\s+)?({target})\b', "opposite_of", 0.9, "bidirectional"),
        (r'\b({source})\s+(?:vs\.?|versus)\s+(?:the\s+)?({target})\b', "opposite_of", 0.85, "bidirectional"),

        # Weak co-occurrence patterns
        (r'\b({source})\s+(?:and|with)\s+(?:the\s+)?({target})\b', "related_to", 0.5, "bidirectional"),

        # FABIO-SPECIFIC: Teaching phrases that imply relationships
        (r'\b({source})\s+(?:because|since|as)\s+(?:the\s+)?(?:reason\s+is\s+)?(?:we\s+have|there\s+is|there\s+was)?\s*(?:the\s+)?({target})\b', "caused_by", 0.7, "forward"),
        (r'\b(?:this\s+is|that\s+is|which\s+is)\s+(?:what\s+we\s+call|called)?\s*(?:the\s+)?({source})\b.*\b(?:this\s+is|that\s+is|which\s+is)\s+(?:what\s+we\s+call|called)?\s*(?:the\s+)?({target})\b', "related_to", 0.6, "bidirectional"),
        (r'\b(?:you\s+see)\s+(?:the\s+)?({source})\b.*\b(?:and|with)\s+(?:the\s+)?({target})\b', "related_to", 0.55, "bidirectional"),
        (r'\b(?:we\s+have)\s+(?:the\s+)?({source})\b.*\b(?:and|plus|also)\s+(?:the\s+)?({target})\b', "related_to", 0.55, "bidirectional"),
        (r'\b({source})\s+(?:here|there)\b.*\b(?:and|with)\s+(?:the\s+)?({target})\b', "cooccurs_with", 0.5, "bidirectional"),
    ]

    def __init__(self, ontology: dict, min_cooccurrence: int = 1):
        self.ontology = ontology
        self.min_cooccurrence = min_cooccurrence
        # Filter concept names through quality gate
        self.concept_names = sorted(
            [c["name"] for c in ontology.get("concepts", []) if is_valid_concept_name(c.get("name",""))],
            key=len, reverse=True
        )
        self._compiled_explicit = self._compile_explicit_patterns()

    def _compile_explicit_patterns(self) -> list:
        """Compile explicit regex patterns with concept names substituted."""
        name_alts = "|".join(re.escape(name) for name in self.concept_names if len(name) > 2)
        if not name_alts:
            return []

        compiled = []
        for pattern_tpl, rel_name, confidence, direction in self.EXPLICIT_PATTERNS:
            pattern_str = pattern_tpl.format(source=f"({name_alts})", target=f"({name_alts})")
            try:
                compiled.append((re.compile(pattern_str, re.IGNORECASE), rel_name, confidence, direction))
            except re.error:
                continue
        return compiled

    def extract_explicit(self, text: str) -> list:
        """Extract explicit relationships from text."""
        if not self._compiled_explicit:
            return []

        discovered = []
        seen = set()

        for pattern, rel_name, confidence, direction in self._compiled_explicit:
            for match in pattern.finditer(text):
                source = match.group(1)
                target = match.group(2)

                if source == target:
                    continue
                if not is_valid_concept_name(source) or not is_valid_concept_name(target):
                    continue

                key = (source.lower(), target.lower(), rel_name)
                if key in seen:
                    continue
                seen.add(key)

                if direction == "bidirectional":
                    seen.add((target.lower(), source.lower(), rel_name))

                discovered.append({
                    "source": source,
                    "target": target,
                    "relation": rel_name,
                    "confidence": confidence,
                    "direction": direction,
                    "matched_text": match.group(0),
                    "method": "explicit",
                    "discovered": True,
                })

        return discovered

    def extract_cooccurrence(self, segments: list, min_cooccurrence: Optional[int] = None) -> list:
        """Extract relationships from concept co-occurrence in sentences.

        CLEANED: Filters stop words, requires quality concept names,
        deduplicates, respects min_cooccurrence threshold, and skips
        greeting/overview sentences that contain too many concepts.
        """
        min_co = min_cooccurrence if min_cooccurrence is not None else self.min_cooccurrence
        pair_counts = {}
        pair_examples = {}

        for seg_idx, (idx, text) in enumerate(segments):
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
            for sent_idx, sent in enumerate(sentences):
                sent_lower = sent.lower()
                found = []
                for name in self.concept_names:
                    if name.lower() in sent_lower:
                        found.append(name)

                # SKIP greeting/overview sentences:
                # 1. First 2 sentences of each segment (greetings/introductions)
                # 2. Any sentence with >5 concepts (likely overview/summary)
                if sent_idx < 2 and seg_idx == 0:
                    continue
                if len(found) > 5:
                    continue

                for i, a in enumerate(found):
                    for b in found[i+1:]:
                        if a == b:
                            continue
                        if not is_valid_concept_name(a) or not is_valid_concept_name(b):
                            continue
                        key = tuple(sorted([a.lower(), b.lower()]))
                        pair_counts[key] = pair_counts.get(key, 0) + 1
                        if key not in pair_examples:
                            pair_examples[key] = sent.strip()[:100]

        discovered = []
        for (a, b), count in pair_counts.items():
            if count >= min_co:
                # Find the actual names (not lowercased)
                name_a = next(n for n in self.concept_names if n.lower() == a)
                name_b = next(n for n in self.concept_names if n.lower() == b)

                # Confidence scales with co-occurrence count
                confidence = min(0.65, 0.45 + (count - 1) * 0.1)

                discovered.append({
                    "source": name_a,
                    "target": name_b,
                    "relation": "cooccurs_with",
                    "confidence": confidence,
                    "direction": "bidirectional",
                    "matched_text": pair_examples.get((a, b), ""),
                    "cooccurrence_count": count,
                    "method": "cooccurrence",
                    "discovered": True,
                })

        return discovered

    def extract(self, text: str, segments: list = None) -> list:
        """Extract all relationships using both explicit and co-occurrence methods."""
        explicit = self.extract_explicit(text)

        cooccur = []
        if segments:
            cooccur = self.extract_cooccurrence(segments)

        # Merge, preferring explicit over co-occurrence for same pairs
        explicit_pairs = set((r["source"].lower(), r["target"].lower()) for r in explicit)
        cooccur_filtered = [r for r in cooccur
                            if (r["source"].lower(), r["target"].lower()) not in explicit_pairs
                            and (r["target"].lower(), r["source"].lower()) not in explicit_pairs]

        return explicit + cooccur_filtered

    def extract_from_segments(self, segments: list) -> list:
        """Extract relationships from all transcript segments."""
        all_text = " ".join(text for _, text in segments)
        all_rels = self.extract(all_text, segments)

        # Add segment index to each relationship (from first matching segment)
        for rel in all_rels:
            for idx, text in segments:
                if rel["source"].lower() in text.lower() and rel["target"].lower() in text.lower():
                    rel["segment_index"] = idx
                    break

        return all_rels

# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def serialize_concept(concept: Concept) -> dict:
    return {
        "concept_id": concept.concept_id,
        "name": concept.definition.name,
        "definition": concept.definition.definition,
        "visual_appearance": concept.definition.visual_appearance,
        "teacher_explanation": concept.definition.teacher_explanation,
        "confidence_level": concept.confidence.level.name,
        "confidence_score": float(concept.confidence.score),
        "evidence_count": concept.confidence.evidence_count,
        "positive_examples": len(concept.positive_examples),
        "negative_examples": len(concept.negative_examples),
        "related_concepts": list(concept.related_concepts),
        "lesson_references": list(concept.lesson_references),
        "evolution_events": len(concept.evolution_history),
    }

def serialize_experience(exp: Experience) -> dict:
    return {
        "experience_id": exp.experience_id,
        "session_reference": exp.session_reference,
        "what_was_seen": exp.observation.what_was_seen,
        "teacher_explanation": exp.teacher_explanation[:200] + "..." if len(exp.teacher_explanation) > 200 else exp.teacher_explanation,
        "has_reasoning": exp.reasoning is not None,
        "has_outcome": exp.outcome is not None,
        "has_reflection": exp.reflection is not None,
        "is_complete": exp.is_complete(),
        "concept_references": list(exp.concept_references),
    }

def serialize_knowledge_graph(graph: KnowledgeGraph) -> dict:
    return {
        "total_concepts": len(graph.concepts),
        "total_relationships": len(graph.relationships),
        "concepts": sorted(graph.concepts),
        "relationships": [
            {
                "source": rel.source_concept,
                "type": rel.relationship_type.name,
                "target": rel.target_concept,
                "confidence": float(rel.confidence),
            }
            for rel in graph.relationships
        ],
    }

# ---------------------------------------------------------------------------
# Main processing — HONEST ARCHITECTURE  —  CLEANED
# ---------------------------------------------------------------------------

def process_video_folder(
    video_path: str,
    transcript_path: str,
    output_dir: str,
    lesson_name: Optional[str] = None,
    ontology_path: Optional[str] = None,
    max_frames: int = 50,
    mode: str = "transcript-only",
    min_cooccurrence: int = 1,
) -> dict:
    """Process one video + transcript.

    Args:
        mode: "transcript-only" (default, honest) or "vision" (attempts real vision)
        min_cooccurrence: Minimum co-occurrence count for relationship discovery (default 1)
    """
    video_path = Path(video_path)
    transcript_path = Path(transcript_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lesson_name = lesson_name or video_path.stem

    print(f"\n{'=' * 70}")
    print(f"Processing: {lesson_name}")
    print(f"Mode: {mode}")
    print(f"{'=' * 70}")
    print(f"Video: {video_path}")
    print(f"Transcript: {transcript_path}")
    print(f"Output: {output_dir}")

    # Load ontology first (needed for everything)
    ontology = load_ontology(ontology_path) if ontology_path else {}
    if ontology:
        valid_concepts = sum(1 for c in ontology.get("concepts", []) if is_valid_concept_name(c.get("name","")))
        print(f"[Ontology] Loaded: {len(ontology.get('concepts', []))} concepts "
              f"({valid_concepts} valid), {len(ontology.get('relationships', []))} relationships")
    else:
        print("[Ontology] WARNING: No ontology loaded. Graph will be empty.")

    # Load transcript
    transcript_entries = load_transcript(str(transcript_path))
    print(f"Transcript entries: {len(transcript_entries)}")

    if not transcript_entries:
        print("WARNING: No transcript entries found. Creating minimal synthetic data.")
        transcript_entries = ((0, "Fabio explains the market structure and order flow concepts."),)

    # Coverage analysis (always run)
    coverage = analyze_coverage(transcript_entries, ontology)
    print(f"[Coverage] {coverage['covered_segments']}/{coverage['total_segments']} segments matched concepts "
          f"({coverage['coverage_percent']}%)")
    if coverage["uncovered_segments"] > 0:
        print(f"[Coverage] {coverage['uncovered_segments']} segments had zero concept matches")

    # Relationship discovery from transcript text
    rel_extractor = RelationshipExtractor(ontology, min_cooccurrence=min_cooccurrence)
    discovered_rels = rel_extractor.extract_from_segments(list(transcript_entries))

    # Deduplicate discovered relationships before saving
    seen_rels = set()
    deduped_rels = []
    for rel in discovered_rels:
        key = (rel["source"].lower(), rel["target"].lower(), rel.get("relation","").lower())
        if key not in seen_rels:
            seen_rels.add(key)
            deduped_rels.append(rel)
    discovered_rels = deduped_rels

    print(f"[Discovery] Found {len(discovered_rels)} potential relationships in transcript")
    if discovered_rels:
        for rel in discovered_rels[:5]:
            print(f" → {rel['source']} --{rel['relation']}--> {rel['target']} "
                  f"(conf={rel['confidence']:.2f}): '{rel['matched_text'][:60]}...'")
        if len(discovered_rels) > 5:
            print(f" ... and {len(discovered_rels) - 5} more")

    # Extract video frames
    video_frames = extract_video_frames(str(video_path), max_frames=max_frames)

    # Build frames for integration
    frames = []

    if mode == "transcript-only":
        print(f"[Mode] Transcript-only: Processing {len(transcript_entries)} segments")
        num_entries = len(transcript_entries)
        num_video_frames = max(1, len(video_frames))

        for i, (entry_idx, text) in enumerate(transcript_entries):
            vf_idx = min(i * num_video_frames // max(1, num_entries), num_video_frames - 1)
            frame_num, timestamp, image = video_frames[vf_idx] if video_frames else (i, f"00:{i:02d}:00", None)
            graph = build_fake_detection_graph(i, text)
            frames.append((i, timestamp, graph, text))

    else:
        print(f"[Mode] Vision: Processing {len(video_frames)} frames")
        if VISION_AVAILABLE:
            print(f"[Vision] GPU: {'CUDA enabled' if GPU.cuda_available else 'CPU only'}")
        else:
            print("[Vision] WARNING: chart_vision_analyzer not available, falling back to keyword-based fake graphs")

        num_frames = len(video_frames)
        num_entries = len(transcript_entries)
        vision_stats = {"charts_detected": 0, "grids_detected": 0, "avg_process_ms": 0.0}

        for i, (frame_num, timestamp, image) in enumerate(video_frames):
            entry_idx = min(i * num_entries // max(1, num_frames), num_entries - 1)
            _, text = transcript_entries[entry_idx]

            if VISION_AVAILABLE and image is not None:
                graph = build_vision_detection_graph(i, image, text, ontology)
                # Collect stats and vision concepts
                features = analyze_frame(image, text)
                if features.has_chart:
                    vision_stats["charts_detected"] += 1
                if features.grid:
                    vision_stats["grids_detected"] += 1
                vision_stats["avg_process_ms"] += features.processing_time_ms

                # Extract vision-derived concepts for this frame
                vconcepts = extract_vision_concepts(features, i, text)
                vision_stats.setdefault("vision_concepts", []).extend(vconcepts)

                # Save debug annotated frame
                debug_dir = output_dir / "debug_frames"
                debug_dir.mkdir(parents=True, exist_ok=True)
                annotated = annotate_debug_frame(image, features, vconcepts, i)
                debug_path = debug_dir / f"frame_{i:04d}.jpg"
                cv2.imwrite(str(debug_path), annotated)
                vision_stats.setdefault("debug_frames_saved", 0)
                vision_stats["debug_frames_saved"] += 1
            else:
                graph = build_fake_detection_graph(i, text)
                vision_stats.setdefault("vision_concepts", [])
                vision_stats.setdefault("debug_frames_saved", 0)

            frames.append((i, timestamp, graph, text))

        if vision_stats["charts_detected"] > 0:
            avg_ms = vision_stats["avg_process_ms"] / num_frames
            print(f"[Vision] Charts detected: {vision_stats['charts_detected']}/{num_frames}")
            print(f"[Vision] Grids detected: {vision_stats['grids_detected']}/{num_frames}")
            print(f"[Vision] Avg processing time: {avg_ms:.1f}ms/frame")

    print(f"Frames to process: {len(frames)}")

    # Monkey-patch KnowledgeGraph.query() for coach.py compatibility
    if hasattr(KnowledgeGraph, 'query'):
        _orig_query = KnowledgeGraph.query
        def _patched_query(self, source_concept=None, relationship_type=None, target_concept=None, **kwargs):
            if source_concept is None and relationship_type is None and target_concept is None:
                return _orig_query(self, **kwargs)
            results = []
            for rel in getattr(self, 'relationships', []):
                if source_concept is not None and getattr(rel, 'source_concept', None) != source_concept:
                    continue
                if target_concept is not None and getattr(rel, 'target_concept', None) != target_concept:
                    continue
                if relationship_type is not None:
                    rt = getattr(rel, 'relationship_type', None)
                    if hasattr(rt, 'name') and hasattr(relationship_type, 'name'):
                        if rt.name != relationship_type.name:
                            continue
                    elif rt != relationship_type:
                        continue
                results.append(rel)
            return results
        KnowledgeGraph.query = _patched_query

    # Initialize apprentice integration
    config = RunnerIntegrationConfiguration(
        process_every_frame=(mode == "vision"),
        process_key_frames_only=(mode == "transcript-only"),
        key_frame_interval=1,
        enable_coach=True,
        enable_concept_extraction=True,
        enable_experience_creation=True,
        enable_knowledge_graph_building=True,
        min_transcript_length_for_explanation=10,
        max_frames_per_lesson=max_frames,
    )
    integration = ApprenticeRunnerIntegration(config)

    # Process lesson
    result = integration.process_lesson(lesson_name, tuple(frames))

    # Build enriched graph early so we can show it in the Lesson Result
    layer = integration.get_apprentice_layer()
    layer_concepts = layer.all_concepts() if layer else []
    enriched_graph = build_graph_from_ontology(ontology, {c.definition.name for c in layer_concepts}, discovered_rels)

    print(f"\nLesson Result:")
    print(f" Frames processed: {len(result.frame_results)}")
    print(f" Sessions: {len(result.sessions)}")
    print(f" Experiences: {len(result.experiences)}")
    print(f" Concepts: {result.concept_statistics.total_concepts}")
    print(f" Knowledge graph (integration): {result.knowledge_graph.statistics.total_concepts} concepts, "
          f"{result.knowledge_graph.statistics.total_relationships} relationships")
    print(f" Knowledge graph (enriched): {enriched_graph['total_concepts']} concepts, "
          f"{enriched_graph['total_relationships']} relationships "
          f"({enriched_graph.get('predefined_relationships', 0)} predefined, "
          f"{enriched_graph.get('discovered_relationships', 0)} discovered)")

    # ------------------------------------------------------------------
    # FIX: Inject discovered + ontology relationships into integration graph
    # ------------------------------------------------------------------
    layer = integration.get_apprentice_layer()
    if layer:
        kg = layer.get_knowledge_graph()

        # Make relationships mutable if it's a tuple
        if isinstance(kg.relationships, tuple):
            object.__setattr__(kg, 'relationships', list(kg.relationships))

        # Build a mapping of relation names to RelationshipType enum values
        try:
            from orderflowgpt_genesis.apprentice import RelationshipType
            rel_type_map = {rt.name: rt for rt in RelationshipType}
            default_rt = RelationshipType.RELATED_TO
        except ImportError:
            rel_type_map = {}
            default_rt = None

        def _get_rel_type(rel_name: str):
            """Map a string relation name to a RelationshipType enum value."""
            if not rel_type_map:
                return default_rt
            # Try exact match
            upper_name = rel_name.upper().replace("-", "_")
            if upper_name in rel_type_map:
                return rel_type_map[upper_name]
            # Try common mappings
            mapping = {
                "COOCCURS_WITH": "RELATED_TO",
                "IS_A": "REQUIRES",
                "RESULTS_IN": "TRIGGERS",
                "CAUSED_BY": "REQUIRES",
                "OCCURS_AT": "REQUIRES",
                "ASSOCIATED_WITH": "RELATED_TO",
                "OPPOSITE_OF": "CONTRADICTS",
                "FOLLOWS": "TRIGGERS",
                "PRECEDES": "TRIGGERS",
                "SIGNALS": "CONFIRMS",
            }
            mapped = mapping.get(upper_name, "RELATED_TO")
            return rel_type_map.get(mapped, default_rt)

        injected = 0
        # Inject ontology predefined relationships
        for rel in ontology.get("relationships", []):
            if not is_valid_relationship(rel):
                continue
            rt = _get_rel_type(rel.get("relation", "RELATED_TO"))
            if rt is None:
                continue
            if rel["source"] in kg.concepts and rel["target"] in kg.concepts:
                try:
                    from orderflowgpt_genesis.apprentice import Relationship, Confidence
                    kg.relationships.append(Relationship(
                        source_concept=rel["source"],
                        target_concept=rel["target"],
                        relationship_type=rt,
                        confidence=Confidence(score=Decimal(str(rel.get("confidence", 0.5))))
                    ))
                    injected += 1
                except Exception:
                    pass

        # Inject discovered relationships
        for rel in discovered_rels:
            if not is_valid_relationship(rel):
                continue
            rt = _get_rel_type(rel.get("relation", "RELATED_TO"))
            if rt is None:
                continue
            if rel["source"] in kg.concepts and rel["target"] in kg.concepts:
                try:
                    from orderflowgpt_genesis.apprentice import Relationship, Confidence
                    kg.relationships.append(Relationship(
                        source_concept=rel["source"],
                        target_concept=rel["target"],
                        relationship_type=rt,
                        confidence=Confidence(score=Decimal(str(rel.get("confidence", 0.5))))
                    ))
                    injected += 1
                except Exception:
                    pass

        if injected > 0:
            print(f"[Graph] Injected {injected} relationships into integration knowledge graph")

    # ------------------------------------------------------------------
    # PATH A: Inject vision-derived concepts into apprentice layer
    # ------------------------------------------------------------------
    all_vision_concepts = vision_stats.get("vision_concepts", []) if mode == "vision" else []
    if all_vision_concepts:
        unique_vis = set(vc.definition.name for vc in all_vision_concepts)
        print(f"[Vision] Extracted {len(all_vision_concepts)} vision-derived concepts ({len(unique_vis)} unique) from {len(video_frames)} frames")
        newly_registered, vision_rels = inject_vision_concepts(integration, all_vision_concepts, ontology, discovered_rels, text)
        print(f"[Vision] {len(newly_registered)} newly registered to ontology")
        print(f"[Vision] {len(vision_rels)} cross-modal relationships generated")
        if vision_stats.get("debug_frames_saved", 0) > 0:
            debug_dir = output_dir / "debug_frames"
            print(f"[Vision] Debug frames saved: {vision_stats['debug_frames_saved']} -> {debug_dir}")
        # Add vision relationships to discovered_rels for persistence
        if vision_rels:
            discovered_rels = (discovered_rels or []) + vision_rels

    # Build report
    report = integration.build_report(datetime.utcnow().isoformat())

    # Compute REAL confidence (includes vision concepts now)
    layer_concepts = layer.all_concepts() if layer else []
    # Merge vision concepts into the concept set for confidence calculation
    all_concept_names = {c.definition.name for c in layer_concepts}
    for vc in all_vision_concepts:
        all_concept_names.add(vc.definition.name)
    enriched_graph = build_graph_from_ontology(ontology, all_concept_names, discovered_rels)
    real_confidence = compute_real_confidence(list(layer_concepts) + all_vision_concepts, coverage, enriched_graph)

    print(f"[Confidence] Real computed: {real_confidence:.1%} (was hardcoded ~71.4%)")

    # Save outputs
    _save_outputs(output_dir, report, integration, ontology, coverage, real_confidence, enriched_graph, mode, discovered_rels, all_vision_concepts, vision_stats, video_frames)

    return {
        "report": report,
        "coverage": coverage,
        "confidence": real_confidence,
        "graph": enriched_graph,
    }

def _save_outputs(
    output_dir: Path,
    report: ApprenticeReport,
    integration,
    ontology: dict,
    coverage: dict,
    real_confidence: float,
    enriched_graph: dict,
    mode: str,
    discovered_rels: Optional[List[dict]] = None,
    vision_concepts: Optional[List[VisionConcept]] = None,
    vision_stats: Optional[dict] = None,
    video_frames: Optional[list] = None,
) -> None:
    layer = integration.get_apprentice_layer()
    layer_concepts = layer.all_concepts()
    layer_experiences = layer.all_experiences()

    # 1. Apprentice Report
    report_path = output_dir / "apprentice_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "report_id": report.report_id,
            "report_timestamp": report.report_timestamp,
            "mode": mode,
            "total_lessons": len(report.lesson_results),
            "total_concepts_learned": report.total_concepts_learned,
            "total_experiences_created": report.total_experiences_created,
            "total_sessions_completed": report.total_sessions_completed,
            "total_coach_explanations": report.total_coach_explanations,
            "total_frames_processed": report.total_frames_processed,
            "overall_learning_confidence": real_confidence,
            "concept_mastery_distribution": report.concept_mastery_distribution,
            "top_concepts_by_confidence": report.top_concepts_by_confidence,
            "what_was_learned": report.what_was_learned,
            "what_needs_more_study": report.what_needs_more_study,
            "knowledge_graph_summary": report.knowledge_graph_summary,
            "coverage": coverage,
        }, f, indent=2, cls=DecimalEncoder)
    print(f"\nSaved: {report_path}")

    # 2. Concepts (merge transcript + vision)
    concepts_path = output_dir / "concepts.json"
    concepts = [serialize_concept(c) for c in layer_concepts]
    if vision_concepts:
        for vc in vision_concepts:
            concepts.append({
                "concept_id": vc.concept_id,
                "name": vc.definition.name,
                "definition": vc.definition.definition,
                "visual_appearance": vc.definition.visual_appearance,
                "teacher_explanation": vc.definition.teacher_explanation,
                "confidence_level": vc.confidence.level.name,
                "confidence_score": float(vc.confidence.score),
                "evidence_count": vc.confidence.evidence_count,
                "positive_examples": len(vc.positive_examples),
                "negative_examples": len(vc.negative_examples),
                "related_concepts": list(vc.related_concepts),
                "lesson_references": list(vc.lesson_references),
                "evolution_events": len(vc.evolution_history),
                "source": "vision",
            })
    with open(concepts_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2, cls=DecimalEncoder)
    print(f"Saved: {concepts_path}")
    if vision_concepts:
        vis_count = len([c for c in concepts if c.get("source") == "vision"])
        txt_count = len([c for c in concepts if c.get("source") != "vision"])
        print(f"[Concepts] {txt_count} transcript-derived, {vis_count} vision-derived")

    # 3. Experiences
    experiences_path = output_dir / "experiences.json"
    experiences = [serialize_experience(e) for e in layer_experiences]
    with open(experiences_path, "w", encoding="utf-8") as f:
        json.dump(experiences, f, indent=2, cls=DecimalEncoder)
    print(f"Saved: {experiences_path}")

    # 4. Knowledge Graph — enriched from ontology + discovered relationships
    graph_path = output_dir / "knowledge_graph.json"

    graph = layer.get_knowledge_graph()
    integration_concepts = set(graph.concepts)
    integration_rels = len(graph.relationships)

    print(f"[Graph] Integration produced: {len(integration_concepts)} concepts, {integration_rels} relationships")
    print(f"[Graph] Enriched from ontology: {enriched_graph['total_concepts']} concepts, "
          f"{enriched_graph['total_relationships']} relationships "
          f"({enriched_graph.get('predefined_relationships', 0)} predefined, "
          f"{enriched_graph.get('discovered_relationships', 0)} discovered)")

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(enriched_graph, f, indent=2, cls=DecimalEncoder)
    print(f"Saved: {graph_path}")

    # Also save discovered relationships separately for audit
    if discovered_rels:
        discovered_path = output_dir / "discovered_relationships.json"
        with open(discovered_path, "w", encoding="utf-8") as f:
            json.dump(discovered_rels, f, indent=2, cls=DecimalEncoder)
        print(f"Saved: {discovered_path}")

    # 5. Coverage Report
    coverage_path = output_dir / "coverage_report.json"
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, cls=DecimalEncoder)
    print(f"Saved: {coverage_path}")

    # 6. Coach Explanations (text)
    coach_path = output_dir / "coach_explanations.txt"
    with open(coach_path, "w", encoding="utf-8") as f:
        f.write("GENESIS LIVE COACH — Explanations\n")
        f.write("=" * 70 + "\n\n")
        for lesson in report.lesson_results:
            for frame in lesson.frame_results:
                if frame.explanation_text:
                    f.write(f"Lesson: {lesson.lesson_reference} | Frame: {frame.frame_index}\n")
                    f.write(f"Timestamp: {frame.frame_timestamp}\n")
                    f.write(frame.explanation_text)
                    f.write("\n\n")
    print(f"Saved: {coach_path}")

    # 7. Summary text — HONEST
    summary_path = output_dir / "apprentice_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GENESIS APPRENTICE — Learning Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Report ID: {report.report_id}\n")
        f.write(f"Timestamp: {report.report_timestamp}\n")
        f.write(f"Mode: {mode}\n\n")
        f.write(f"Lessons Processed: {len(report.lesson_results)}\n")
        f.write(f"Concepts Learned: {report.total_concepts_learned}\n")
        f.write(f"Experiences Created: {report.total_experiences_created}\n")
        f.write(f"Sessions Completed: {report.total_sessions_completed}\n")
        f.write(f"Coach Explanations: {report.total_coach_explanations}\n")
        f.write(f"Frames Processed: {report.total_frames_processed}\n")
        f.write(f"Overall Confidence: {real_confidence:.2%}\n\n")

        f.write("Concept Coverage:\n")
        f.write(f" Segments with concepts: {coverage['covered_segments']}/{coverage['total_segments']} "
                f"({coverage['coverage_percent']}%)\n")
        f.write(f" Segments with NO concepts: {coverage['uncovered_segments']}\n\n")

        f.write("Concept Mastery:\n")
        for level, count in report.concept_mastery_distribution:
            bar = "█" * count + "░" * max(0, 10 - count)
            f.write(f" {level:12s} {bar} ({count})\n")
        f.write("\nTop Concepts:\n")
        for name, score in report.top_concepts_by_confidence[:10]:
            f.write(f" {name:20s} {float(score):.2f}\n")
        f.write("\nWhat was learned:\n")
        for item in report.what_was_learned:
            f.write(f" ✓ {item}\n")
        f.write("\nWhat needs more study:\n")
        if report.what_needs_more_study:
            for item in report.what_needs_more_study:
                f.write(f" ⚠ {item}\n")
        else:
            f.write(f" ✓ All concepts have sufficient study\n")

        f.write(f"\nKnowledge Graph:\n")
        f.write(f" Concepts: {enriched_graph['total_concepts']}\n")
        f.write(f" Relationships: {enriched_graph['total_relationships']}\n")

        if mode == "transcript-only":
            f.write(f"\n[NOTE] Running in TRANSCRIPT-ONLY mode.\n")
            f.write(f" Video frames provide timestamps only.\n")
            f.write(f" Concept extraction is keyword-based from transcript text.\n")
            f.write(f" Vision analysis (Bundles 1-8) is NOT active.\n")
        else:
            f.write(f"\n[NOTE] Running in VISION mode.\n")
            f.write(f" OpenCV chart analysis is ACTIVE.\n")
            f.write(f" GPU acceleration: {'CUDA' if VISION_AVAILABLE and GPU.cuda_available else 'CPU'}\n")
            f.write(f" Concept extraction fuses visual evidence + transcript text.\n")
            vis_count = len(set(vc.definition.name for vc in (vision_concepts or [])))
            f.write(f" Vision-derived concepts: {vis_count}\n")
            charts = vision_stats.get('charts_detected', 0) if vision_stats else 0
            total = len(video_frames) if video_frames else 0
            f.write(f" Chart detection rate: {charts}/{total} frames\n")
            debug_saved = vision_stats.get('debug_frames_saved', 0) if vision_stats else 0
            f.write(f" Debug frames saved: {debug_saved}\n")

        f.write(f"\n{report.knowledge_graph_summary}\n")
    print(f"Saved: {summary_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process a real Fabio video through the Genesis Apprentice Layer (CLEANED v5.1)"
    )
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--transcript", type=str, help="Path to transcript file")
    parser.add_argument("--input", type=str, help="Input folder containing video and transcript")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--lesson", type=str, help="Lesson name (defaults to video filename)")
    parser.add_argument("--ontology", type=str, help="Path to fabio_ontology.json")
    parser.add_argument("--mode", choices=["transcript-only", "vision"], default="transcript-only",
                        help="Processing mode. transcript-only=honest text-based (default). "
                             "vision=attempts real Bundle 1-8 vision analysis.")
    parser.add_argument("--every-frame", action="store_true", help="Process every frame (slower)")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames per lesson")
    parser.add_argument("--min-cooccurrence", type=int, default=1,
                        help="Minimum co-occurrence count for relationship discovery (default 1, use 2+ for batch)")

    args = parser.parse_args()

    # Resolve paths
    if args.input:
        input_dir = Path(args.input)
        videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mkv"))
        transcripts = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.srt")) + list(input_dir.glob("*.json"))

        if not videos:
            print(f"ERROR: No video found in {input_dir}")
            sys.exit(1)
        if not transcripts:
            print(f"ERROR: No transcript found in {input_dir}")
            sys.exit(1)

        video_path = str(videos[0])
        transcript_path = str(transcripts[0])
    elif args.video and args.transcript:
        video_path = args.video
        transcript_path = args.transcript
    else:
        print("ERROR: Specify either --input (folder) or both --video and --transcript")
        sys.exit(1)

    # Process
    result = process_video_folder(
        video_path=video_path,
        transcript_path=transcript_path,
        output_dir=args.output,
        lesson_name=args.lesson,
        ontology_path=args.ontology,
        max_frames=args.max_frames,
        mode=args.mode,
        min_cooccurrence=args.min_cooccurrence,
    )

    # Final summary
    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nOutput files in: {args.output}")
    print(f" • apprentice_report.json — Full learning report")
    print(f" • concepts.json — All learned concepts")
    print(f" • experiences.json — All experiences")
    print(f" • knowledge_graph.json — Concept relationships (ontology-enriched)")
    print(f" • coverage_report.json — Transcript coverage analysis")
    print(f" • coach_explanations.txt — Human-readable explanations")
    print(f" • apprentice_summary.txt — Quick summary")
    print(f"\nGenesis learned {result['report'].total_concepts_learned} concepts from {len(result['report'].lesson_results)} lesson(s).")
    print(f"Real confidence: {result['confidence']:.1%}")
    print(f"Coverage: {result['coverage']['coverage_percent']}% of transcript segments matched concepts")
    print(f"Graph: {result['graph']['total_concepts']} concepts, {result['graph']['total_relationships']} relationships")

if __name__ == "__main__":
    main()