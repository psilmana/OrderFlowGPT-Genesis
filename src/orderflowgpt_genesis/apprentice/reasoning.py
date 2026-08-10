"""Deterministic reasoning layer implementing Fabio's Decision Hierarchy.

Market State -> Location -> Aggression -> Risk -> Management -> Reflection

This module explains markets the way Fabio teaches. It does NOT generate
trading signals, predictions, probabilities, or recommendations.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Tuple


class MarketStateType(Enum):
    """Deterministic market state classifications."""
    BALANCED = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    AUCTION_UP = auto()
    AUCTION_DOWN = auto()
    ROTATIONAL = auto()
    UNKNOWN = auto()


class LocationType(Enum):
    """Deterministic location classifications within market structure."""
    AT_POC = auto()
    AT_VAH = auto()
    AT_VAL = auto()
    ABOVE_VAH = auto()
    BELOW_VAL = auto()
    IN_VALUE_AREA = auto()
    AT_STRUCTURE_HIGH = auto()
    AT_STRUCTURE_LOW = auto()
    AT_SUPPORT = auto()
    AT_RESISTANCE = auto()
    AT_SUPPLY_ZONE = auto()
    AT_DEMAND_ZONE = auto()
    NEUTRAL = auto()


class AggressionType(Enum):
    """Deterministic aggression classifications from order flow."""
    BUYER_AGGRESSIVE = auto()
    SELLER_AGGRESSIVE = auto()
    BALANCED = auto()
    ABSORPTION_BUY = auto()
    ABSORPTION_SELL = auto()
    DELTA_DIVERGENCE_BULLISH = auto()
    DELTA_DIVERGENCE_BEARISH = auto()
    EXHAUSTION_BUY = auto()
    EXHAUSTION_SELL = auto()
    UNKNOWN = auto()


class RiskLevel(Enum):
    """Deterministic risk level classifications."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    EXTREME = auto()
    UNKNOWN = auto()


class ManagementType(Enum):
    """Deterministic management posture classifications.

    These are explanatory postures, not trade recommendations.
    """
    WAIT = auto()
    OBSERVE = auto()
    SCALE_IN = auto()
    SCALE_OUT = auto()
    EXIT = auto()
    UNKNOWN = auto()


class ReflectionType(Enum):
    """Deterministic reflection classifications on prior reasoning."""
    CORRECT = auto()
    PARTIAL = auto()
    INCORRECT = auto()
    INSUFFICIENT_DATA = auto()
    PREMATURE = auto()
    DELAYED = auto()


@dataclass(frozen=True)
class MarketStateAssessment:
    """Assessment of current market state with supporting evidence."""
    state: MarketStateType
    evidence: Tuple[str, ...]
    confidence: Decimal
    primary_driver: str = ""
    secondary_drivers: Tuple[str, ...] = ()

    def __post_init__(self):
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class LocationAssessment:
    """Assessment of current location within market structure."""
    location: LocationType
    evidence: Tuple[str, ...]
    confidence: Decimal
    distance_from_poc: Optional[Decimal] = None
    distance_from_vah: Optional[Decimal] = None
    distance_from_val: Optional[Decimal] = None

    def __post_init__(self):
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class AggressionAssessment:
    """Assessment of who is aggressive and how."""
    aggression: AggressionType
    evidence: Tuple[str, ...]
    confidence: Decimal
    primary_zone: str = ""
    supporting_imbalances: Tuple[str, ...] = ()
    supporting_absorption: Tuple[str, ...] = ()

    def __post_init__(self):
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class RiskAssessment:
    """Assessment of risk given market state, location, and aggression."""
    risk_level: RiskLevel
    evidence: Tuple[str, ...]
    confidence: Decimal
    stop_location_reference: str = ""
    invalidation_conditions: Tuple[str, ...] = ()

    def __post_init__(self):
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class ManagementAssessment:
    """Assessment of management posture given prior levels.

    This explains what Fabio would watch, not what to trade.
    """
    management: ManagementType
    evidence: Tuple[str, ...]
    confidence: Decimal
    watch_conditions: Tuple[str, ...] = ()
    scale_conditions: Tuple[str, ...] = ()

    def __post_init__(self):
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class ReflectionAssessment:
    """Assessment of what the prior reasoning got right or wrong."""
    reflection: ReflectionType
    evidence: Tuple[str, ...]
    confidence: Decimal
    what_was_right: Tuple[str, ...] = ()
    what_was_wrong: Tuple[str, ...] = ()
    what_was_missed: Tuple[str, ...] = ()
    lesson_learned: str = ""

    def __post_init__(self):
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class DecisionHierarchyConfiguration:
    """Configuration for deterministic decision hierarchy analysis."""
    trend_confidence_threshold: Decimal = Decimal("0.60")
    auction_confidence_threshold: Decimal = Decimal("0.55")
    imbalance_threshold_for_aggression: Decimal = Decimal("2.0")
    absorption_threshold_for_aggression: Decimal = Decimal("1.5")
    risk_high_threshold: Decimal = Decimal("0.75")
    risk_extreme_threshold: Decimal = Decimal("0.90")
    require_all_levels: bool = False

    def __post_init__(self):
        for name, val in [
            ("trend_confidence_threshold", self.trend_confidence_threshold),
            ("auction_confidence_threshold", self.auction_confidence_threshold),
            ("risk_high_threshold", self.risk_high_threshold),
            ("risk_extreme_threshold", self.risk_extreme_threshold),
        ]:
            if not (Decimal("0") <= val <= Decimal("1")):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.risk_extreme_threshold < self.risk_high_threshold:
            raise ValueError("risk_extreme_threshold must be >= risk_high_threshold")


@dataclass(frozen=True)
class DecisionHierarchyStatistics:
    """Statistics over a collection of hierarchy results."""
    total_assessments: int = 0
    market_state_distribution: Tuple[Tuple[MarketStateType, int], ...] = ()
    location_distribution: Tuple[Tuple[LocationType, int], ...] = ()
    aggression_distribution: Tuple[Tuple[AggressionType, int], ...] = ()
    risk_distribution: Tuple[Tuple[RiskLevel, int], ...] = ()
    average_confidence: Decimal = Decimal("0")
    complete_hierarchy_count: int = 0
    incomplete_hierarchy_count: int = 0


@dataclass(frozen=True)
class DecisionHierarchyResult:
    """Complete Fabio Decision Hierarchy result for one observation."""
    market_state: Optional[MarketStateAssessment] = None
    location: Optional[LocationAssessment] = None
    aggression: Optional[AggressionAssessment] = None
    risk: Optional[RiskAssessment] = None
    management: Optional[ManagementAssessment] = None
    reflection: Optional[ReflectionAssessment] = None
    overall_confidence: Decimal = Decimal("0")
    hierarchy_complete: bool = False
    missing_levels: Tuple[str, ...] = ()

    def __post_init__(self):
        if not (Decimal("0") <= self.overall_confidence <= Decimal("1")):
            raise ValueError("overall_confidence must be in [0, 1]")

    def level_count(self) -> int:
        """Return the number of completed hierarchy levels."""
        count = 0
        for level in (
            self.market_state,
            self.location,
            self.aggression,
            self.risk,
            self.management,
            self.reflection,
        ):
            if level is not None:
                count += 1
        return count


class DecisionHierarchyAnalyzer:
    """Deterministic analyzer that explains a chart through Fabio's hierarchy.

    This analyzer consumes immutable DetectionGraph results produced by
    prior bundles and explains what is happening. It never predicts,
    recommends trades, or estimates probabilities.
    """

    def __init__(self, configuration: Optional[DecisionHierarchyConfiguration] = None):
        self._configuration = configuration or DecisionHierarchyConfiguration()

    def analyze(self, detection_graph) -> DecisionHierarchyResult:
        """Analyze a DetectionGraph through Fabio's Decision Hierarchy.

        Args:
            detection_graph: A DetectionGraph from prior bundle analysis.

        Returns:
            DecisionHierarchyResult with all available levels populated.
        """
        market_state = self._assess_market_state(detection_graph)
        location = self._assess_location(detection_graph)
        aggression = self._assess_aggression(detection_graph)
        risk = self._assess_risk(detection_graph, market_state, location, aggression)
        management = self._assess_management(detection_graph, market_state, location, aggression, risk)
        reflection = self._assess_reflection(detection_graph)

        levels = [market_state, location, aggression, risk, management, reflection]
        present = [l for l in levels if l is not None]
        overall_confidence = Decimal("0")
        if present:
            total = sum(l.confidence for l in present)
            overall_confidence = min(total / Decimal(str(len(present))), Decimal("1"))

        missing = []
        if market_state is None:
            missing.append("market_state")
        if location is None:
            missing.append("location")
        if aggression is None:
            missing.append("aggression")
        if risk is None:
            missing.append("risk")
        if management is None:
            missing.append("management")
        if reflection is None:
            missing.append("reflection")

        hierarchy_complete = len(missing) == 0
        if self._configuration.require_all_levels and not hierarchy_complete:
            overall_confidence = Decimal("0")

        return DecisionHierarchyResult(
            market_state=market_state,
            location=location,
            aggression=aggression,
            risk=risk,
            management=management,
            reflection=reflection,
            overall_confidence=overall_confidence,
            hierarchy_complete=hierarchy_complete,
            missing_levels=tuple(missing),
        )

    def _assess_market_state(self, graph) -> Optional[MarketStateAssessment]:
        """Determine market state from trend and auction evidence."""
        evidence = []
        state = MarketStateType.UNKNOWN
        confidence = Decimal("0")
        primary = ""
        secondary = []

        trend = getattr(graph, "trend_state", None)
        if trend is not None:
            trend_state = getattr(trend, "trend_state", None)
            if trend_state is not None:
                name = getattr(trend_state, "name", str(trend_state))
                if "UP" in name or "BULL" in name:
                    state = MarketStateType.TRENDING_UP
                    primary = "trend engine bullish"
                    confidence = Decimal("0.70")
                elif "DOWN" in name or "BEAR" in name:
                    state = MarketStateType.TRENDING_DOWN
                    primary = "trend engine bearish"
                    confidence = Decimal("0.70")
                elif "ROTATION" in name or "RANGE" in name:
                    state = MarketStateType.ROTATIONAL
                    primary = "trend engine rotational"
                    confidence = Decimal("0.60")
                else:
                    state = MarketStateType.BALANCED
                    primary = "trend engine balanced"
                    confidence = Decimal("0.50")
                evidence.append(f"trend_state={name}")

        session = getattr(graph, "trading_session", None)
        if session is not None:
            session_type = getattr(session, "session_type", None)
            if session_type is not None:
                evidence.append(f"session={session_type}")
                if state == MarketStateType.UNKNOWN:
                    state = MarketStateType.BALANCED
                    confidence = Decimal("0.40")

        auction = getattr(graph, "unfinished_auctions", None)
        if auction is not None:
            auctions = getattr(auction, "unfinished_auctions", ())
            if auctions:
                evidence.append(f"unfinished_auctions={len(auctions)}")
                secondary.append("unfinished auction present")
                if state in (MarketStateType.BALANCED, MarketStateType.UNKNOWN):
                    top = any(getattr(a, "is_top", False) for a in auctions)
                    bottom = any(getattr(a, "is_bottom", False) for a in auctions)
                    if top and not bottom:
                        state = MarketStateType.AUCTION_DOWN
                        primary = "unfinished auction at top"
                        confidence = Decimal("0.55")
                    elif bottom and not top:
                        state = MarketStateType.AUCTION_UP
                        primary = "unfinished auction at bottom"
                        confidence = Decimal("0.55")

        if state == MarketStateType.UNKNOWN:
            return None

        return MarketStateAssessment(
            state=state,
            evidence=tuple(evidence),
            confidence=min(confidence, Decimal("1")),
            primary_driver=primary,
            secondary_drivers=tuple(secondary),
        )

    def _assess_location(self, graph) -> Optional[LocationAssessment]:
        """Determine location from market profile and structure."""
        evidence = []
        location = LocationType.NEUTRAL
        confidence = Decimal("0")

        poc = getattr(graph, "point_of_control", None)
        va = getattr(graph, "value_area", None)
        structure = getattr(graph, "market_structure", None)
        zones = getattr(graph, "supply_demand_zones", None)

        if poc is not None and va is not None:
            evidence.append("poc_and_value_area_available")
            confidence = Decimal("0.60")

        if structure is not None:
            swings = getattr(structure, "swing_points", ())
            if swings:
                evidence.append(f"swing_points={len(swings)}")
                confidence = max(confidence, Decimal("0.50"))

        if zones is not None:
            zone_list = getattr(zones, "zones", ())
            if zone_list:
                evidence.append(f"supply_demand_zones={len(zone_list)}")

        if not evidence:
            return None

        return LocationAssessment(
            location=location,
            evidence=tuple(evidence),
            confidence=min(confidence, Decimal("1")),
        )

    def _assess_aggression(self, graph) -> Optional[AggressionAssessment]:
        """Determine aggression from imbalances, delta, and absorption."""
        evidence = []
        aggression = AggressionType.UNKNOWN
        confidence = Decimal("0")
        primary_zone = ""
        imbalances = []
        absorptions = []

        imb = getattr(graph, "footprint_imbalances", None)
        if imb is not None:
            imbs = getattr(imb, "imbalances", ())
            ask_count = sum(1 for i in imbs if getattr(i, "side", None) and "ASK" in str(getattr(i, "side", "")))
            bid_count = sum(1 for i in imbs if getattr(i, "side", None) and "BID" in str(getattr(i, "side", "")))
            evidence.append(f"imbalances_ask={ask_count}_bid={bid_count}")
            if ask_count > bid_count:
                aggression = AggressionType.SELLER_AGGRESSIVE
                confidence = Decimal("0.60")
                primary_zone = "ask imbalances dominant"
            elif bid_count > ask_count:
                aggression = AggressionType.BUYER_AGGRESSIVE
                confidence = Decimal("0.60")
                primary_zone = "bid imbalances dominant"
            else:
                aggression = AggressionType.BALANCED
                confidence = Decimal("0.40")
                primary_zone = "imbalances balanced"
            imbalances = [f"{getattr(i, 'side', '?')} at {getattr(i, 'cell_id', '?')}" for i in imbs[:3]]

        stacked = getattr(graph, "stacked_imbalances", None)
        if stacked is not None:
            stacks = getattr(stacked, "stacks", ())
            if stacks:
                evidence.append(f"stacked_imbalances={len(stacks)}")
                confidence = min(confidence + Decimal("0.10"), Decimal("1"))

        delta = getattr(graph, "footprint_delta", None)
        if delta is not None:
            cell_deltas = getattr(delta, "cell_deltas", ())
            positive = sum(1 for d in cell_deltas if getattr(d, "delta", Decimal("0")) > Decimal("0"))
            negative = sum(1 for d in cell_deltas if getattr(d, "delta", Decimal("0")) < Decimal("0"))
            evidence.append(f"delta_positive={positive}_negative={negative}")

        absorp = getattr(graph, "absorption_result", None)
        if absorp is not None:
            abss = getattr(absorp, "absorptions", ())
            if abss:
                evidence.append(f"absorption_events={len(abss)}")
                absorptions = [f"{getattr(a, 'side', '?')} at {getattr(a, 'cell_id', '?')}" for a in abss[:3]]
                if aggression == AggressionType.SELLER_AGGRESSIVE:
                    aggression = AggressionType.ABSORPTION_SELL
                    primary_zone = "seller aggression with absorption"
                elif aggression == AggressionType.BUYER_AGGRESSIVE:
                    aggression = AggressionType.ABSORPTION_BUY
                    primary_zone = "buyer aggression with absorption"

        if not evidence:
            return None

        return AggressionAssessment(
            aggression=aggression,
            evidence=tuple(evidence),
            confidence=min(confidence, Decimal("1")),
            primary_zone=primary_zone,
            supporting_imbalances=tuple(imbalances),
            supporting_absorption=tuple(absorptions),
        )

    def _assess_risk(
        self,
        graph,
        market_state: Optional[MarketStateAssessment],
        location: Optional[LocationAssessment],
        aggression: Optional[AggressionAssessment],
    ) -> Optional[RiskAssessment]:
        """Determine risk from state, location, and aggression."""
        evidence = []
        risk = RiskLevel.UNKNOWN
        confidence = Decimal("0")
        invalidation = []

        if market_state is None or aggression is None:
            return None

        if market_state.state in (MarketStateType.TRENDING_UP, MarketStateType.TRENDING_DOWN):
            evidence.append("trending_market")
            risk = RiskLevel.MEDIUM
            confidence = Decimal("0.50")
            invalidation.append("break of structure")
        elif market_state.state in (MarketStateType.AUCTION_UP, MarketStateType.AUCTION_DOWN):
            evidence.append("auction_market")
            risk = RiskLevel.HIGH
            confidence = Decimal("0.65")
            invalidation.append("unfinished auction fills")
        elif market_state.state == MarketStateType.ROTATIONAL:
            evidence.append("rotational_market")
            risk = RiskLevel.MEDIUM
            confidence = Decimal("0.55")
            invalidation.append("acceptance outside range")
        else:
            evidence.append("balanced_market")
            risk = RiskLevel.LOW
            confidence = Decimal("0.40")

        if aggression.aggression in (AggressionType.BUYER_AGGRESSIVE, AggressionType.SELLER_AGGRESSIVE):
            evidence.append("clear_aggression")
            confidence = min(confidence + Decimal("0.10"), Decimal("1"))
        elif aggression.aggression in (AggressionType.ABSORPTION_BUY, AggressionType.ABSORPTION_SELL):
            evidence.append("absorption_present")
            risk = RiskLevel.HIGH
            confidence = min(confidence + Decimal("0.15"), Decimal("1"))
            invalidation.append("absorption fails")

        confluence = getattr(graph, "confluence", None)
        if confluence is not None:
            ctype = getattr(confluence, "confluence_type", None)
            if ctype is not None:
                evidence.append(f"confluence={ctype}")
                if "STRONG" in str(ctype).upper():
                    confidence = min(confidence + Decimal("0.10"), Decimal("1"))
                elif "NO" in str(ctype).upper() or "WEAK" in str(ctype).upper():
                    risk = RiskLevel.HIGH
                    confidence = min(confidence + Decimal("0.05"), Decimal("1"))

        return RiskAssessment(
            risk_level=risk,
            evidence=tuple(evidence),
            confidence=min(confidence, Decimal("1")),
            invalidation_conditions=tuple(invalidation),
        )

    def _assess_management(
        self,
        graph,
        market_state: Optional[MarketStateAssessment],
        location: Optional[LocationAssessment],
        aggression: Optional[AggressionAssessment],
        risk: Optional[RiskAssessment],
    ) -> Optional[ManagementAssessment]:
        """Determine management posture from prior levels."""
        if market_state is None or risk is None:
            return None

        evidence = []
        management = ManagementType.OBSERVE
        confidence = Decimal("0.40")
        watch = []
        scale = []

        if risk.risk_level == RiskLevel.EXTREME:
            management = ManagementType.WAIT
            confidence = Decimal("0.60")
            watch.append("wait for clarity")
        elif risk.risk_level == RiskLevel.HIGH:
            management = ManagementType.OBSERVE
            confidence = Decimal("0.55")
            watch.append("watch for invalidation")
        elif risk.risk_level == RiskLevel.LOW:
            management = ManagementType.OBSERVE
            confidence = Decimal("0.45")
            watch.append("monitor continuation")
        else:
            management = ManagementType.OBSERVE
            confidence = Decimal("0.50")

        if aggression is not None:
            if aggression.aggression in (AggressionType.ABSORPTION_BUY, AggressionType.ABSORPTION_SELL):
                watch.append("absorption resolution")
                scale.append("scale after confirmation")
                confidence = min(confidence + Decimal("0.10"), Decimal("1"))

        session = getattr(graph, "trading_session", None)
        if session is not None:
            ib = getattr(graph, "initial_balance", None)
            if ib is not None:
                ib_break = getattr(ib, "ib_break", None)
                if ib_break is not None:
                    evidence.append(f"ib_break={ib_break}")
                    watch.append("ib extension behavior")

        evidence.append(f"risk={risk.risk_level.name}")
        evidence.append(f"state={market_state.state.name}")

        return ManagementAssessment(
            management=management,
            evidence=tuple(evidence),
            confidence=min(confidence, Decimal("1")),
            watch_conditions=tuple(watch),
            scale_conditions=tuple(scale),
        )

    def _assess_reflection(self, graph) -> Optional[ReflectionAssessment]:
        """Reflection is empty by default; populated by ExperienceEngine later."""
        return None


def hierarchy_statistics(results: Tuple[DecisionHierarchyResult, ...]) -> DecisionHierarchyStatistics:
    """Compute statistics over a collection of DecisionHierarchyResult objects."""
    from collections import Counter

    if not results:
        return DecisionHierarchyStatistics()

    ms_counts = Counter(r.market_state.state for r in results if r.market_state is not None)
    loc_counts = Counter(r.location.location for r in results if r.location is not None)
    agg_counts = Counter(r.aggression.aggression for r in results if r.aggression is not None)
    risk_counts = Counter(r.risk.risk_level for r in results if r.risk is not None)

    complete = sum(1 for r in results if r.hierarchy_complete)
    avg_conf = sum(r.overall_confidence for r in results) / Decimal(str(len(results)))

    return DecisionHierarchyStatistics(
        total_assessments=len(results),
        market_state_distribution=tuple(ms_counts.items()),
        location_distribution=tuple(loc_counts.items()),
        aggression_distribution=tuple(agg_counts.items()),
        risk_distribution=tuple(risk_counts.items()),
        average_confidence=min(avg_conf, Decimal("1")),
        complete_hierarchy_count=complete,
        incomplete_hierarchy_count=len(results) - complete,
    )
