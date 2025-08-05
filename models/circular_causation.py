"""
Circular and cumulative causation modeling for Social Fabric Matrix analysis.

This module implements Hayden's circular and cumulative causation (CCC) methodology
for analyzing feedback loops, cumulative effects, and system dynamics in institutional
systems. CCC is a core concept in institutional economics that explains how 
institutional changes create feedback effects that reinforce or modify the initial change.

Key Components:
- CausalLink: Individual causal relationship between system elements
- CausalChain: Sequence of causal links forming a pathway
- FeedbackLoop: Circular causal pathway that reinforces system behavior
- CumulativeProcess: Process that builds up effects over time
- CCCAnalyzer: Tools for analyzing circular and cumulative causation patterns
"""

# pylint: disable=too-many-instance-attributes,too-many-public-methods  # Complex circular causation analysis requires many attributes

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import math

from models.base_nodes import Node
# from models.meta_entities import TimeSlice
# from models.metadata_models import TemporalDynamics
from models.sfm_enums import (
    FeedbackPolarity,
    # FeedbackType,
    # SystemArchetype,
    # ChangeType,
    PathDependencyType,
)


class CausalDirection(Enum):
    """Direction of causal influence."""
    
    POSITIVE = auto()     # Increase in cause leads to increase in effect
    NEGATIVE = auto()     # Increase in cause leads to decrease in effect
    NEUTRAL = auto()      # No clear directional relationship
    COMPLEX = auto()      # Non-linear or context-dependent relationship


class CausalStrength(Enum):
    """Strength of causal relationship."""
    
    VERY_STRONG = auto()  # Strong, consistent causal influence
    STRONG = auto()       # Consistent causal influence
    MODERATE = auto()     # Moderate causal influence
    WEAK = auto()         # Weak causal influence
    VERY_WEAK = auto()    # Minimal causal influence


class CumulativeType(Enum):
    """Types of cumulative processes."""
    
    LINEAR_ACCUMULATION = auto()      # Effects accumulate linearly
    EXPONENTIAL_GROWTH = auto()       # Effects grow exponentially
    THRESHOLD_EFFECTS = auto()        # Effects appear after threshold
    SATURATION_EFFECTS = auto()       # Effects level off at saturation
    OSCILLATING_CUMULATION = auto()   # Effects oscillate while accumulating
    DEGRADATION_CUMULATION = auto()   # Effects decay while accumulating


class LoopType(Enum):
    """Types of feedback loops."""
    
    REINFORCING = auto()    # Positive feedback that amplifies change
    BALANCING = auto()      # Negative feedback that stabilizes system
    MIXED = auto()          # Combination of reinforcing and balancing
    DELAYED = auto()        # Feedback with significant time delays
    CONDITIONAL = auto()    # Feedback that depends on conditions


@dataclass
class CausalLink(Node):
    """Individual causal relationship between two system elements."""
    
    cause_element_id: uuid.UUID  # What causes the effect
    effect_element_id: uuid.UUID  # What is affected
    
    # Causal properties
    causal_direction: CausalDirection = CausalDirection.POSITIVE
    causal_strength: CausalStrength = CausalStrength.MODERATE
    causal_coefficient: Optional[float] = None  # Quantitative measure if available
    
    # Temporal properties
    time_delay: Optional[timedelta] = None  # Delay between cause and effect
    duration: Optional[timedelta] = None    # How long the effect lasts
    decay_rate: Optional[float] = None      # Rate at which effect decays
    
    # Conditions and context
    enabling_conditions: List[str] = field(default_factory=list)
    inhibiting_conditions: List[str] = field(default_factory=list)
    contextual_factors: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence and validation
    evidence_strength: float = 0.5  # Strength of evidence for this link (0-1)
    empirical_support: List[str] = field(default_factory=list)
    theoretical_basis: Optional[str] = None
    
    # SFM Matrix Integration (Enhanced)
    institutional_mediation: List[uuid.UUID] = field(default_factory=list)  # Institutions that mediate this link
    matrix_cells_affected: List[uuid.UUID] = field(default_factory=list)  # Matrix cells influenced
    delivery_causal_effects: Dict[uuid.UUID, str] = field(default_factory=dict)  # Delivery impacts
    
    # Ceremonial-Instrumental Analysis
    ceremonial_component: Optional[float] = None  # Ceremonial aspects of causation
    instrumental_component: Optional[float] = None  # Instrumental aspects of causation
    ci_balance: Optional[float] = None  # CI balance (-1 to +1)
    
    # Policy and Political Integration
    policy_mediated_causation: List[uuid.UUID] = field(default_factory=list)  # Policies affecting link
    political_action_influences: List[uuid.UUID] = field(default_factory=list)  # Political actions
    
    # Cross-Matrix Effects
    cross_matrix_propagation: List[str] = field(default_factory=list)  # How effects cross matrix
    system_level_causation: Optional[str] = None  # System-wide causal role
    
    def calculate_effective_strength(self, context: Dict[str, Any]) -> float:
        """Calculate effective causal strength given context."""
        base_strength_values = {
            CausalStrength.VERY_STRONG: 0.9,
            CausalStrength.STRONG: 0.7,
            CausalStrength.MODERATE: 0.5,
            CausalStrength.WEAK: 0.3,
            CausalStrength.VERY_WEAK: 0.1
        }
        
        base_strength = base_strength_values[self.causal_strength]
        
        # Adjust for enabling/inhibiting conditions
        condition_modifier = 1.0
        
        for condition in self.enabling_conditions:
            if context.get(condition, False):
                condition_modifier *= 1.2  # Boost strength
        
        for condition in self.inhibiting_conditions:
            if context.get(condition, False):
                condition_modifier *= 0.8  # Reduce strength
        
        # Adjust for evidence strength
        evidence_modifier = (self.evidence_strength + 1.0) / 2.0
        
        effective_strength = base_strength * condition_modifier * evidence_modifier
        return min(effective_strength, 1.0)
    
    def calculate_time_adjusted_effect(self, elapsed_time: timedelta) -> float:
        """Calculate effect strength adjusted for time delays and decay."""
        if self.time_delay and elapsed_time < self.time_delay:
            return 0.0  # Effect hasn't started yet
        
        effective_time = elapsed_time
        if self.time_delay:
            effective_time = elapsed_time - self.time_delay
        
        # Apply decay if specified
        if self.decay_rate and effective_time.total_seconds() > 0:
            decay_factor = math.exp(-self.decay_rate * effective_time.total_seconds() / 3600)  # Decay per hour
            return decay_factor
        
        # Check if effect has duration limit
        if self.duration and effective_time > self.duration:
            return 0.0  # Effect has ended
        
        return 1.0  # Full effect
    
    def is_active(self, context: Dict[str, Any]) -> bool:
        """Check if this causal link is currently active."""
        # Check enabling conditions
        for condition in self.enabling_conditions:
            if not context.get(condition, False):
                return False
        
        # Check inhibiting conditions
        for condition in self.inhibiting_conditions:
            if context.get(condition, False):
                return False
        
        return True


@dataclass
class CausalChain(Node):
    """Sequence of causal links forming a pathway."""
    
    causal_links: List[uuid.UUID] = field(default_factory=list)  # Ordered list of link IDs
    chain_strength: Optional[float] = None  # Overall strength of the chain
    
    # Chain properties
    total_delay: Optional[timedelta] = None  # Total time delay through chain
    amplification_factor: Optional[float] = None  # Net amplification/dampening
    
    # Path characteristics
    is_complete_path: bool = False  # Whether chain forms complete causal path
    has_branches: bool = False      # Whether chain has branching paths
    
    def calculate_chain_strength(self, links: Dict[uuid.UUID, CausalLink], 
                                context: Dict[str, Any]) -> float:
        """Calculate overall strength of the causal chain."""
        if not self.causal_links:
            return 0.0
        
        # Chain strength is product of link strengths (weakest link principle)
        chain_strength = 1.0
        
        for link_id in self.causal_links:
            if link_id in links:
                link = links[link_id]
                if link.is_active(context):
                    link_strength = link.calculate_effective_strength(context)
                    chain_strength *= link_strength
                else:
                    chain_strength = 0.0  # Broken chain
                    break
        
        self.chain_strength = chain_strength
        return chain_strength
    
    def calculate_total_delay(self, links: Dict[uuid.UUID, CausalLink]) -> timedelta:
        """Calculate total time delay through the chain."""
        total_delay = timedelta()
        
        for link_id in self.causal_links:
            if link_id in links:
                link = links[link_id]
                if link.time_delay:
                    total_delay += link.time_delay
        
        self.total_delay = total_delay
        return total_delay
    
    def trace_path(self, links: Dict[uuid.UUID, CausalLink]) -> List[uuid.UUID]:
        """Trace the path of elements connected by this chain."""
        if not self.causal_links:
            return []
        
        path = []
        
        # Add first element
        first_link_id = self.causal_links[0]
        if first_link_id in links:
            first_link = links[first_link_id]
            path.append(first_link.cause_element_id)
        
        # Add subsequent elements
        for link_id in self.causal_links:
            if link_id in links:
                link = links[link_id]
                path.append(link.effect_element_id)
        
        return path


@dataclass
class FeedbackLoop(Node):
    """Circular causal pathway that creates feedback effects."""
    
    loop_type: LoopType = LoopType.REINFORCING
    causal_chain: Optional[CausalChain] = None  # type: ignore[misc]
    
    # Loop properties
    loop_polarity: Optional[FeedbackPolarity] = None  # type: ignore[misc]
    loop_gain: Optional[float] = None  # Amplification factor per cycle
    cycle_time: Optional[timedelta] = None  # Time for one complete cycle
    
    # Stability properties
    is_stable: Optional[bool] = None  # Whether loop reaches equilibrium
    equilibrium_point: Optional[float] = None  # Stable equilibrium value
    instability_threshold: Optional[float] = None  # Point where loop becomes unstable
    
    # Loop dynamics
    oscillation_period: Optional[timedelta] = None  # Period of oscillations
    damping_coefficient: Optional[float] = None    # Rate of oscillation decay
    growth_rate: Optional[float] = None            # Rate of exponential growth/decay
    
    # SFM integration
    system_archetype: Optional[str] = None  # type: ignore[misc]
    institutional_reinforcement: List[uuid.UUID] = field(default_factory=list)
    
    def calculate_loop_gain(self, links: Dict[uuid.UUID, CausalLink], 
                           context: Dict[str, Any]) -> float:
        """Calculate the gain of one complete loop cycle."""
        chain_strength = self.causal_chain.calculate_chain_strength(links, context)  # type: ignore[misc]
        
        # For reinforcing loops, gain > 1 amplifies, gain < 1 dampens
        # For balancing loops, gain represents correction strength
        if self.loop_type == LoopType.REINFORCING:
            # Positive feedback: effects amplify initial change
            loop_gain = 1.0 + chain_strength
        elif self.loop_type == LoopType.BALANCING:
            # Negative feedback: effects counteract initial change
            loop_gain = max(0.0, 1.0 - chain_strength)
        else:
            # Mixed or other types
            loop_gain = chain_strength
        
        self.loop_gain = loop_gain
        return loop_gain
    
    def simulate_loop_behavior(self, initial_value: float, num_cycles: int,
                              links: Dict[uuid.UUID, CausalLink], 
                              context: Dict[str, Any]) -> List[float]:
        """Simulate loop behavior over multiple cycles."""
        if num_cycles <= 0:
            return [initial_value]
        
        values = [initial_value]
        current_value = initial_value
        loop_gain = self.calculate_loop_gain(links, context)
        
        for cycle in range(num_cycles):
            if self.loop_type == LoopType.REINFORCING:
                # Exponential growth/decay
                current_value *= loop_gain
            elif self.loop_type == LoopType.BALANCING:
                # Approach equilibrium
                if self.equilibrium_point is not None:
                    difference = self.equilibrium_point - current_value
                    current_value += difference * loop_gain
                else:
                    current_value *= loop_gain
            
            # Apply damping if specified
            if self.damping_coefficient:
                damping_factor = math.exp(-self.damping_coefficient * cycle)
                current_value *= damping_factor
            
            values.append(current_value)
        
        return values
    
    def assess_loop_stability(self, links: Dict[uuid.UUID, CausalLink], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess stability characteristics of the feedback loop."""
        loop_gain = self.calculate_loop_gain(links, context)
        
        stability_assessment = {
            'loop_gain': loop_gain,
            'is_stable': None,
            'behavior_type': 'unknown',
            'risk_level': 'medium'
        }
        
        if self.loop_type == LoopType.REINFORCING:
            if loop_gain > 1.1:
                stability_assessment['is_stable'] = False
                stability_assessment['behavior_type'] = 'exponential_growth'
                stability_assessment['risk_level'] = 'high'
            elif loop_gain < 0.9:
                stability_assessment['is_stable'] = True
                stability_assessment['behavior_type'] = 'decay_to_zero'
                stability_assessment['risk_level'] = 'low'
            else:
                stability_assessment['is_stable'] = True
                stability_assessment['behavior_type'] = 'stable_growth'
                stability_assessment['risk_level'] = 'medium'
        
        elif self.loop_type == LoopType.BALANCING:
            if loop_gain > 0.5:
                stability_assessment['is_stable'] = True
                stability_assessment['behavior_type'] = 'stable_equilibrium'
                stability_assessment['risk_level'] = 'low'
            else:
                stability_assessment['is_stable'] = False
                stability_assessment['behavior_type'] = 'weak_correction'
                stability_assessment['risk_level'] = 'medium'
        
        self.is_stable = stability_assessment['is_stable']
        return stability_assessment


@dataclass
class CumulativeProcess(Node):
    """Process that accumulates effects over time through repeated causation."""
    
    cumulative_type: CumulativeType = CumulativeType.LINEAR_ACCUMULATION
    base_causal_links: List[uuid.UUID] = field(default_factory=list)
    
    # Accumulation parameters
    accumulation_rate: Optional[float] = None  # Rate of accumulation
    carrying_capacity: Optional[float] = None  # Maximum possible accumulation
    threshold_value: Optional[float] = None    # Threshold for effects to appear
    
    # Current state
    current_accumulated_value: float = 0.0
    accumulation_history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Temporal properties
    accumulation_period: Optional[timedelta] = None  # Period over which accumulation occurs
    decay_half_life: Optional[timedelta] = None      # Half-life of accumulated effects
    
    # Process characteristics
    has_threshold_effects: bool = False
    has_saturation_effects: bool = False
    exhibits_path_dependency: bool = False
    path_dependency_type: Optional[PathDependencyType] = None
    
    def add_accumulation_event(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add an accumulation event to the process."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Apply accumulation based on type
        if self.cumulative_type == CumulativeType.LINEAR_ACCUMULATION:
            self.current_accumulated_value += value
        
        elif self.cumulative_type == CumulativeType.EXPONENTIAL_GROWTH:
            if self.accumulation_rate:
                growth_factor = 1 + self.accumulation_rate
                self.current_accumulated_value = self.current_accumulated_value * growth_factor + value
            else:
                self.current_accumulated_value += value
        
        elif self.cumulative_type == CumulativeType.SATURATION_EFFECTS:
            if self.carrying_capacity:
                # Logistic growth towards carrying capacity
                remaining_capacity = self.carrying_capacity - self.current_accumulated_value
                growth = value * (remaining_capacity / self.carrying_capacity)
                self.current_accumulated_value += growth
            else:
                self.current_accumulated_value += value
        
        elif self.cumulative_type == CumulativeType.THRESHOLD_EFFECTS:
            self.current_accumulated_value += value
            # Effects only appear after threshold
            if self.threshold_value and self.current_accumulated_value < self.threshold_value:
                effective_value = 0.0  # type: ignore[misc]  # noqa: F841
            else:
                effective_value = self.current_accumulated_value  # type: ignore[misc]  # noqa: F841
        
        # Apply carrying capacity constraint if specified
        if self.carrying_capacity:
            self.current_accumulated_value = min(self.current_accumulated_value, self.carrying_capacity)
        
        # Record in history
        self.accumulation_history.append((timestamp, self.current_accumulated_value))
    
    def calculate_decay(self, time_elapsed: timedelta) -> float:
        """Calculate decay of accumulated effects over time."""
        if not self.decay_half_life:
            return 1.0  # No decay
        
        # Exponential decay
        decay_constant = math.log(2) / self.decay_half_life.total_seconds()
        decay_factor = math.exp(-decay_constant * time_elapsed.total_seconds())
        
        return decay_factor
    
    def apply_decay(self, timestamp: Optional[datetime] = None) -> None:
        """Apply decay to the current accumulated value."""
        if not self.accumulation_history or not self.decay_half_life:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        last_timestamp = self.accumulation_history[-1][0]
        time_elapsed = timestamp - last_timestamp
        
        decay_factor = self.calculate_decay(time_elapsed)
        self.current_accumulated_value *= decay_factor
        
        # Record decayed value
        self.accumulation_history.append((timestamp, self.current_accumulated_value))
    
    def get_effective_value(self) -> float:
        """Get effective value considering thresholds and other factors."""
        if self.has_threshold_effects and self.threshold_value:
            if self.current_accumulated_value < self.threshold_value:
                return 0.0
        
        return self.current_accumulated_value
    
    def calculate_accumulation_rate(self, time_window: Optional[timedelta] = None) -> Optional[float]:
        """Calculate rate of accumulation over a time window."""
        if len(self.accumulation_history) < 2:
            return None
        
        if time_window is None:
            # Use entire history
            start_time, start_value = self.accumulation_history[0]
            end_time, end_value = self.accumulation_history[-1]
        else:
            # Use recent history within time window
            cutoff_time = datetime.now() - time_window
            recent_history = [(t, v) for t, v in self.accumulation_history if t >= cutoff_time]
            
            if len(recent_history) < 2:
                return None
            
            start_time, start_value = recent_history[0]
            end_time, end_value = recent_history[-1]
        
        time_diff = (end_time - start_time).total_seconds()
        if time_diff == 0:
            return None
        
        value_diff = end_value - start_value
        rate = value_diff / time_diff  # Change per second
        
        return rate


@dataclass
class CCCAnalyzer(Node):
    """Analyzer for circular and cumulative causation patterns."""
    
    causal_links: Dict[uuid.UUID, CausalLink] = field(default_factory=dict)
    feedback_loops: Dict[uuid.UUID, FeedbackLoop] = field(default_factory=dict)
    cumulative_processes: Dict[uuid.UUID, CumulativeProcess] = field(default_factory=dict)
    
    # Analysis results
    detected_loops: List[uuid.UUID] = field(default_factory=list)
    system_stability: Optional[float] = None  # Overall system stability (0-1)
    cumulative_risk_score: Optional[float] = None  # Risk from cumulative effects (0-1)
    
    def add_causal_link(self, link: CausalLink) -> None:
        """Add a causal link to the analysis."""
        self.causal_links[link.id] = link
    
    def add_feedback_loop(self, loop: FeedbackLoop) -> None:
        """Add a feedback loop to the analysis."""
        self.feedback_loops[loop.id] = loop
        self.detected_loops.append(loop.id)
    
    def add_cumulative_process(self, process: CumulativeProcess) -> None:
        """Add a cumulative process to the analysis."""
        self.cumulative_processes[process.id] = process
    
    def detect_feedback_loops(self) -> List[FeedbackLoop]:
        """Automatically detect feedback loops from causal links."""
        # This is a simplified implementation - a full version would use
        # graph algorithms to detect cycles
        
        detected_loops = []
        
        # Build adjacency graph
        graph = {}
        for link in self.causal_links.values():
            if link.cause_element_id not in graph:
                graph[link.cause_element_id] = []
            graph[link.cause_element_id].append((link.effect_element_id, link.id))
        
        # Find cycles using DFS
        visited = set()
        rec_stack = set()
        
        def dfs_find_cycles(node: Any, path: Any, link_path: Any) -> None:  # type: ignore[misc]
            visited.add(node)  # type: ignore[arg-type]
            rec_stack.add(node)  # type: ignore[arg-type]
            path.append(node)
            
            if node in graph:
                for neighbor, link_id in graph[node]:
                    if neighbor in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(neighbor)
                        cycle_nodes = path[cycle_start:]  # type: ignore[misc]  # noqa: F841
                        cycle_links = link_path[cycle_start:]
                        
                        # Create feedback loop
                        chain = CausalChain(  # type: ignore[arg-type]
                            label=f"Detected Loop {len(detected_loops) + 1}",  # type: ignore[arg-type]
                            causal_links=cycle_links  # type: ignore[arg-type]
                        )
                        
                        loop = FeedbackLoop(
                            label=f"Feedback Loop {len(detected_loops) + 1}",  # type: ignore[arg-type]
                            causal_chain=chain
                        )
                        
                        detected_loops.append(loop)
                        self.add_feedback_loop(loop)
                    
                    elif neighbor not in visited:
                        link_path.append(link_id)
                        dfs_find_cycles(neighbor, path.copy(), link_path.copy())
                        link_path.pop()
            
            rec_stack.remove(node)
        
        # Run DFS from each unvisited node
        for node in graph:
            if node not in visited:
                dfs_find_cycles(node, [], [])
        
        return detected_loops
    
    def analyze_system_stability(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall system stability."""
        if not self.feedback_loops:
            return {'stability_score': 1.0, 'risk_level': 'low', 'unstable_loops': []}
        
        stability_factors = []
        unstable_loops = []
        
        for loop in self.feedback_loops.values():
            stability_assessment = loop.assess_loop_stability(self.causal_links, context)
            
            if stability_assessment['is_stable'] is False:
                unstable_loops.append({
                    'loop_id': loop.id,
                    'loop_label': loop.label,
                    'risk_level': stability_assessment['risk_level'],
                    'behavior_type': stability_assessment['behavior_type']
                })
            
            # Convert stability to numeric score
            if stability_assessment['is_stable'] is True:
                stability_factors.append(0.8)
            elif stability_assessment['is_stable'] is False:
                if stability_assessment['risk_level'] == 'high':
                    stability_factors.append(0.1)
                else:
                    stability_factors.append(0.3)
            else:
                stability_factors.append(0.5)
        
        # Overall stability score
        if stability_factors:
            overall_stability = sum(stability_factors) / len(stability_factors)  # type: ignore[arg-type]
        else:
            overall_stability = 1.0
        
        self.system_stability = overall_stability
        
        # Determine risk level
        if overall_stability > 0.7:
            risk_level = 'low'
        elif overall_stability > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'stability_score': overall_stability,
            'risk_level': risk_level,
            'unstable_loops': unstable_loops,
            'total_loops': len(self.feedback_loops),
            'stable_loops': len(self.feedback_loops) - len(unstable_loops)  # type: ignore[arg-type]
        }
    
    def analyze_cumulative_effects(self) -> Dict[str, Any]:
        """Analyze cumulative effects across all processes."""
        if not self.cumulative_processes:
            return {'cumulative_risk': 0.0, 'active_processes': 0}
        
        risk_factors = []
        active_processes = 0
        high_risk_processes = []
        
        for process in self.cumulative_processes.values():
            if process.current_accumulated_value > 0:
                active_processes += 1
                
                # Calculate risk based on accumulation relative to thresholds/capacity
                risk_score = 0.0
                
                if process.threshold_value:
                    threshold_ratio = process.current_accumulated_value / process.threshold_value
                    if threshold_ratio > 0.8:  # Approaching threshold
                        risk_score += 0.6
                
                if process.carrying_capacity:
                    capacity_ratio = process.current_accumulated_value / process.carrying_capacity
                    if capacity_ratio > 0.7:  # Approaching capacity
                        risk_score += 0.4
                
                # High accumulation rate increases risk
                recent_rate = process.calculate_accumulation_rate(timedelta(hours=24))
                if recent_rate and recent_rate > 0:
                    risk_score += min(recent_rate * 0.1, 0.3)
                
                risk_factors.append(risk_score)
                
                if risk_score > 0.7:
                    high_risk_processes.append({
                        'process_id': process.id,
                        'process_label': process.label,
                        'risk_score': risk_score,
                        'current_value': process.current_accumulated_value
                    })
        
        # Overall cumulative risk
        if risk_factors:
            cumulative_risk = sum(risk_factors) / len(risk_factors)  # type: ignore[arg-type]
        else:
            cumulative_risk = 0.0
        
        self.cumulative_risk_score = cumulative_risk
        
        return {
            'cumulative_risk': cumulative_risk,
            'active_processes': active_processes,
            'high_risk_processes': high_risk_processes,
            'total_processes': len(self.cumulative_processes)
        }
    
    def generate_ccc_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive circular and cumulative causation report."""
        # Analyze system components
        stability_analysis = self.analyze_system_stability(context)
        cumulative_analysis = self.analyze_cumulative_effects()
        
        # Identify dominant patterns
        dominant_loops = []
        for loop_id, loop in self.feedback_loops.items():
            loop_gain = loop.calculate_loop_gain(self.causal_links, context)
            if loop_gain > 1.2 or loop_gain < 0.8:  # Significant impact
                dominant_loops.append({
                    'loop_id': loop_id,
                    'loop_label': loop.label,
                    'loop_type': loop.loop_type.name,
                    'loop_gain': loop_gain,
                    'influence_level': 'high' if abs(loop_gain - 1.0) > 0.3 else 'medium'
                })
        
        # System archetype analysis
        archetype_patterns = {}
        for loop in self.feedback_loops.values():
            if loop.system_archetype:
                archetype = loop.system_archetype  # type: ignore[misc]
                archetype_patterns[archetype] = archetype_patterns.get(archetype, 0) + 1  # type: ignore[arg-type]
        
        return {
            'system_overview': {
                'total_causal_links': len(self.causal_links),
                'total_feedback_loops': len(self.feedback_loops),
                'total_cumulative_processes': len(self.cumulative_processes),
                'overall_stability': stability_analysis['stability_score'],
                'cumulative_risk': cumulative_analysis['cumulative_risk']
            },
            'stability_analysis': stability_analysis,
            'cumulative_analysis': cumulative_analysis,
            'dominant_loops': dominant_loops,
            'system_archetypes': archetype_patterns,
            'recommendations': self._generate_recommendations(stability_analysis, cumulative_analysis)
        }
    
    def _generate_recommendations(self, stability_analysis: Dict[str, Any], 
                                 cumulative_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on CCC analysis."""
        recommendations = []
        
        if stability_analysis['risk_level'] == 'high':
            recommendations.append("Monitor unstable feedback loops closely and consider interventions")
            recommendations.append("Strengthen balancing loops to improve system stability")
        
        if cumulative_analysis['cumulative_risk'] > 0.6:
            recommendations.append("Address high-risk cumulative processes before they reach critical thresholds")
            recommendations.append("Implement decay mechanisms to reduce harmful accumulations")
        
        if len(stability_analysis['unstable_loops']) > len(self.feedback_loops) * 0.3:
            recommendations.append("Review institutional rules that may be creating excessive positive feedback")
        
        if not recommendations:
            recommendations.append("System appears stable - continue monitoring for emerging patterns")
        
        return recommendations