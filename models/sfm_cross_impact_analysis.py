"""
Cross-Impact Analysis Framework for Social Fabric Matrix.

This module implements Hayden's methodology for analyzing complex interactions
and interdependencies between institutions, criteria, delivery systems, and
other SFM elements. It provides systematic approaches to understanding system-wide
effects, feedback loops, and emergent properties in social fabric analysis.

Key Components:
- CrossImpactAnalysis: Comprehensive cross-impact analysis framework
- ImpactRelationship: Individual impact relationships between elements
- SystemInteraction: System-level interaction patterns
- FeedbackLoop: Identification and analysis of feedback mechanisms
- EmergentEffect: Analysis of emergent system properties
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum, auto
import statistics

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.sfm_enums import (
    SystemLevel,
    ValueCategory,
    EvidenceQuality,
    ValidationMethod,
    StatisticalMethod,
    InstitutionalScope,
)

class ImpactType(Enum):
    """Types of cross-impacts in SFM analysis."""

    REINFORCING = auto()             # Mutually reinforcing impacts
    INHIBITING = auto()              # Mutually inhibiting impacts
    AMPLIFYING = auto()              # Amplifying/accelerating impacts
    DAMPENING = auto()               # Dampening/moderating impacts
    TRANSFORMATIVE = auto()          # Transformative/qualitative impacts
    NEUTRAL = auto()                 # Minimal or no impact

class ImpactDirection(Enum):
    """Direction of impact relationships."""

    UNIDIRECTIONAL = auto()          # One-way impact
    BIDIRECTIONAL = auto()           # Two-way mutual impact
    MULTI_DIRECTIONAL = auto()       # Complex multi-element impacts
    CYCLICAL = auto()                # Cyclical impact patterns

class ImpactTimeframe(Enum):
    """Timeframe for impact manifestation."""

    IMMEDIATE = auto()               # Immediate/real-time impacts
    SHORT_TERM = auto()              # Days to weeks
    MEDIUM_TERM = auto()             # Months to years
    LONG_TERM = auto()               # Multiple years
    GENERATIONAL = auto()            # Across generations

class ImpactStrength(Enum):
    """Strength levels of cross-impacts."""

    VERY_STRONG = auto()             # Dominant influence
    STRONG = auto()                  # Major influence
    MODERATE = auto()                # Significant influence
    WEAK = auto()                    # Minor influence
    NEGLIGIBLE = auto()              # Minimal influence

class InteractionPattern(Enum):
    """Patterns of system interaction."""

    LINEAR = auto()                  # Linear cause-effect relationships
    NONLINEAR = auto()               # Nonlinear relationships
    THRESHOLD = auto()               # Threshold-based interactions
    NETWORK = auto()                 # Network effect patterns
    EMERGENT = auto()                # Emergent interaction patterns

class FeedbackType(Enum):
    """Types of feedback loops."""

    POSITIVE_FEEDBACK = auto()       # Self-reinforcing feedback
    NEGATIVE_FEEDBACK = auto()       # Self-correcting feedback
    BALANCING = auto()               # Balancing feedback loops
    RUNAWAY = auto()                 # Runaway/exponential feedback
    OSCILLATING = auto()             # Oscillating feedback patterns

@dataclass
class ImpactRelationship(Node):  # pylint: disable=too-many-instance-attributes
    """Individual impact relationship between SFM elements."""

    source_element_id: uuid.UUID
    target_element_id: uuid.UUID
    impact_type: ImpactType = ImpactType.NEUTRAL

    # Impact characteristics
    impact_direction: ImpactDirection = ImpactDirection.UNIDIRECTIONAL
    impact_strength: ImpactStrength = ImpactStrength.MODERATE
    impact_timeframe: ImpactTimeframe = ImpactTimeframe.MEDIUM_TERM

    # Impact quantification
    impact_coefficient: Optional[float] = None     # Quantified impact strength (-1 to 1)
    impact_probability: Optional[float] = None     # Probability of impact occurring (0-1)
    impact_confidence: Optional[float] = None      # Confidence in impact assessment (0-1)

    # Impact mechanisms
    causal_mechanism: Optional[str] = None         # Description of causal mechanism
    transmission_pathways: List[str] = field(default_factory=list)
    mediating_factors: List[str] = field(default_factory=list)
    moderating_conditions: List[str] = field(default_factory=list)

    # Contextual factors
    contextual_dependencies: List[str] = field(default_factory=list)
    scope_conditions: Dict[str, Any] = field(default_factory=dict)
    temporal_dynamics: Optional[str] = None
    spatial_variation: Optional[str] = None

    # Supporting evidence
    empirical_evidence: List[str] = field(default_factory=list)
    theoretical_justification: Optional[str] = None
    evidence_quality: EvidenceQuality = EvidenceQuality.MEDIUM
    validation_methods: List[ValidationMethod] = field(default_factory=list)

    # Impact measurement
    impact_indicators: List[str] = field(default_factory=list)
    measurement_approach: Optional[str] = None
    historical_observations: List[Dict[str, Any]] = field(default_factory=list)

    # Uncertainty and sensitivity
    uncertainty_factors: List[str] = field(default_factory=list)
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    scenario_variations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def assess_impact_significance(self) -> Dict[str, Any]:
        """Assess overall significance of the impact relationship."""
        significance_assessment = {
            'strength_score': 0.0,
            'probability_score': 0.0,
            'evidence_score': 0.0,
            'mechanism_clarity': 0.0,
            'overall_significance': 0.0,
            'significance_level': 'low',
            'key_factors': [],
            'limitations': []
        }

        # Strength score
        strength_scores = {
            ImpactStrength.VERY_STRONG: 1.0,
            ImpactStrength.STRONG: 0.8,
            ImpactStrength.MODERATE: 0.6,
            ImpactStrength.WEAK: 0.4,
            ImpactStrength.NEGLIGIBLE: 0.2
        }
        significance_assessment['strength_score'] = strength_scores.get(self.impact_strength, 0.5)

        # Probability score
        significance_assessment['probability_score'] = self.impact_probability or 0.5

        # Evidence score
        evidence_scores = {
            EvidenceQuality.HIGH: 1.0,
            EvidenceQuality.MEDIUM: 0.7,
            EvidenceQuality.LOW: 0.4
        }
        significance_assessment['evidence_score'] = evidence_scores.get(self.evidence_quality, 0.5)

        # Mechanism clarity
        mechanism_score = 0.0
        if self.causal_mechanism:
            mechanism_score += 0.4
        if self.transmission_pathways:
            mechanism_score += 0.3
        if self.theoretical_justification:
            mechanism_score += 0.3
        significance_assessment['mechanism_clarity'] = mechanism_score

        # Overall significance
        significance_factors = [
            significance_assessment['strength_score'] * 0.3,
            significance_assessment['probability_score'] * 0.25,
            significance_assessment['evidence_score'] * 0.25,
            significance_assessment['mechanism_clarity'] * 0.2
        ]
        significance_assessment['overall_significance'] = sum(significance_factors)

        # Categorize significance level
        if significance_assessment['overall_significance'] >= 0.8:
            significance_assessment['significance_level'] = 'very_high'
        elif significance_assessment['overall_significance'] >= 0.6:
            significance_assessment['significance_level'] = 'high'
        elif significance_assessment['overall_significance'] >= 0.4:
            significance_assessment['significance_level'] = 'moderate'
        else:
            significance_assessment['significance_level'] = 'low'

        # Identify key factors
        if significance_assessment['strength_score'] > 0.7:
            significance_assessment['key_factors'].append('Strong impact strength')
        if significance_assessment['evidence_score'] > 0.8:
            significance_assessment['key_factors'].append('High-quality evidence')
        if significance_assessment['mechanism_clarity'] > 0.7:
            significance_assessment['key_factors'].append('Clear causal mechanism')

        # Identify limitations
        if significance_assessment['evidence_score'] < 0.5:
            significance_assessment['limitations'].append('Limited supporting evidence')
        if not self.causal_mechanism:
            significance_assessment['limitations'].append('Unclear causal mechanism')
        if self.uncertainty_factors:
            significance_assessment['limitations'].append('High uncertainty factors')

        return significance_assessment

    def analyze_temporal_dynamics(self) -> Dict[str, Any]:
        """Analyze temporal aspects of the impact relationship."""
        temporal_analysis = {
            'impact_delay': 0.0,
            'impact_duration': 'unknown',
            'temporal_pattern': 'linear',
            'time_sensitivity': 0.0,
            'temporal_stability': 0.0,
            'dynamic_factors': []
        }

        # Estimate impact delay based on timeframe
        timeframe_delays = {
            ImpactTimeframe.IMMEDIATE: 0.0,
            ImpactTimeframe.SHORT_TERM: 0.1,
            ImpactTimeframe.MEDIUM_TERM: 0.5,
            ImpactTimeframe.LONG_TERM: 2.0,
            ImpactTimeframe.GENERATIONAL: 20.0
        }
        temporal_analysis['impact_delay'] = timeframe_delays.get(self.impact_timeframe, 1.0)

        # Analyze historical observations for patterns
        if self.historical_observations:
            # Simplified pattern analysis
            temporal_analysis['temporal_pattern'] = 'variable'  # Would analyze actual patterns
            temporal_analysis['temporal_stability'] = 0.7      # Would calculate from data

        # Identify dynamic factors
        if self.temporal_dynamics:
            temporal_analysis['dynamic_factors'].append(self.temporal_dynamics)
        if self.contextual_dependencies:
            temporal_analysis['dynamic_factors'].extend(self.contextual_dependencies)

        return temporal_analysis

@dataclass
class SystemInteraction(Node):
    """System-level interaction patterns between multiple elements."""

    interaction_name: Optional[str] = None
    interaction_scope: Optional[str] = None

    # Participating elements
    participating_elements: List[uuid.UUID] = field(default_factory=list)
    element_roles: Dict[uuid.UUID, str] = field(default_factory=dict)  # Element -> role
    interaction_pattern: InteractionPattern = InteractionPattern.LINEAR

    # Interaction characteristics
    interaction_complexity: Optional[float] = None  # Complexity measure (0-1)
    interaction_stability: Optional[float] = None   # Stability over time (0-1)
    interaction_predictability: Optional[float] = None  # Predictability (0-1)

    # System properties
    emergent_properties: List[str] = field(default_factory=list)
    system_outcomes: Dict[str, Any] = field(default_factory=dict)
    interaction_effects: List[uuid.UUID] = field(default_factory=list)  # ImpactRelationship IDs

    # Network characteristics
    interaction_density: Optional[float] = None     # Density of interactions
    centralization: Optional[float] = None          # Degree of centralization
    clustering: Optional[float] = None              # Clustering coefficient

    # Dynamic properties
    adaptation_capacity: Optional[float] = None     # System adaptation capacity
    resilience_factors: List[str] = field(default_factory=list)
    vulnerability_points: List[str] = field(default_factory=list)

    # Intervention points
    leverage_points: List[str] = field(default_factory=list)
    intervention_opportunities: Dict[str, float] = field(default_factory=dict)
    policy_implications: List[str] = field(default_factory=list)

    def analyze_interaction_dynamics(self) -> Dict[str, Any]:
        """Analyze dynamics of system interactions."""
        dynamics_analysis = {
            'complexity_assessment': {},
            'stability_assessment': {},
            'adaptability_assessment': {},
            'intervention_analysis': {},
            'system_health': 'unknown'
        }

        # Complexity assessment
        complexity_factors = []
        if len(self.participating_elements) > 5:
            complexity_factors.append('High number of elements')
        if self.interaction_pattern in [InteractionPattern.NONLINEAR, InteractionPattern.EMERGENT]:
            complexity_factors.append('Complex interaction patterns')
        if len(self.emergent_properties) > 0:
            complexity_factors.append('Emergent properties present')

        dynamics_analysis['complexity_assessment'] = {
            'complexity_level': self.interaction_complexity or 0.5,
            'complexity_factors': complexity_factors,
            'management_difficulty': 'high' if len(complexity_factors) > 2 else 'moderate'
        }

        # Stability assessment
        dynamics_analysis['stability_assessment'] = {
            'stability_level': self.interaction_stability or 0.5,
            'predictability': self.interaction_predictability or 0.5,
            'vulnerability_count': len(self.vulnerability_points),
            'resilience_factors': len(self.resilience_factors)
        }

        # Adaptability assessment
        dynamics_analysis['adaptability_assessment'] = {
            'adaptation_capacity': self.adaptation_capacity or 0.5,
            'leverage_points': len(self.leverage_points),
            'intervention_opportunities': len(self.intervention_opportunities)
        }

        # Overall system health
        health_score = 0.0
        if self.interaction_stability:
            health_score += self.interaction_stability * 0.3
        if self.adaptation_capacity:
            health_score += self.adaptation_capacity * 0.3
        if self.resilience_factors:
            health_score += min(len(self.resilience_factors) / 5.0, 1.0) * 0.2
        if self.vulnerability_points:
            health_score -= min(len(self.vulnerability_points) / 10.0, 0.2)

        health_score = max(0.0, min(1.0, health_score))

        if health_score >= 0.8:
            dynamics_analysis['system_health'] = 'excellent'
        elif health_score >= 0.6:
            dynamics_analysis['system_health'] = 'good'
        elif health_score >= 0.4:
            dynamics_analysis['system_health'] = 'fair'
        else:
            dynamics_analysis['system_health'] = 'poor'

        return dynamics_analysis

    def identify_intervention_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for system intervention."""
        interventions = []

        # Leverage point interventions
        for leverage_point in self.leverage_points:
            interventions.append({
                'intervention_type': 'leverage_point',
                'description': leverage_point,
                'potential_impact': 'high',
                'intervention_difficulty': 'moderate',
                'timeframe': 'medium_term'
            })

        # Vulnerability-based interventions
        for vulnerability in self.vulnerability_points:
            interventions.append({
                'intervention_type': 'vulnerability_mitigation',
                'description': f'Address {vulnerability}',
                'potential_impact': 'protective',
                'intervention_difficulty': 'variable',
                'timeframe': 'short_to_medium_term'
            })

        # Emergent property enhancement
        for property_name in self.emergent_properties:
            interventions.append({
                'intervention_type': 'emergent_enhancement',
                'description': f'Strengthen {property_name}',
                'potential_impact': 'systemic',
                'intervention_difficulty': 'high',
                'timeframe': 'long_term'
            })

        return interventions

    def assess_system_resilience(self) -> Dict[str, Any]:
        """Assess resilience of the interaction system."""
        resilience_assessment = {
            'structural_resilience': 0.0,
            'functional_resilience': 0.0,
            'adaptive_resilience': 0.0,
            'overall_resilience': 0.0,
            'resilience_sources': [],
            'resilience_gaps': [],
            'enhancement_strategies': []
        }

        # Structural resilience
        if self.interaction_density and self.interaction_density > 0.5:
            resilience_assessment['structural_resilience'] += 0.3
        if len(self.participating_elements) > 3:  # Redundancy
            resilience_assessment['structural_resilience'] += 0.4
        if self.clustering and self.clustering > 0.4:
            resilience_assessment['structural_resilience'] += 0.3

        # Functional resilience
        if self.interaction_stability and self.interaction_stability > 0.6:
            resilience_assessment['functional_resilience'] += 0.5
        if len(self.resilience_factors) > 2:
            resilience_assessment['functional_resilience'] += 0.5

        # Adaptive resilience
        if self.adaptation_capacity and self.adaptation_capacity > 0.6:
            resilience_assessment['adaptive_resilience'] = self.adaptation_capacity

        # Overall resilience
        resilience_factors = [
            resilience_assessment['structural_resilience'],
            resilience_assessment['functional_resilience'],
            resilience_assessment['adaptive_resilience']
        ]
        valid_factors = [f for f in resilience_factors if f > 0]
        if valid_factors:
            resilience_assessment['overall_resilience'] = sum(valid_factors) / len(valid_factors)

        # Identify resilience sources
        if len(self.resilience_factors) > 0:
            resilience_assessment['resilience_sources'] = self.resilience_factors.copy()

        # Identify resilience gaps
        if resilience_assessment['structural_resilience'] < 0.5:
            resilience_assessment['resilience_gaps'].append('Low structural resilience')
        if resilience_assessment['adaptive_resilience'] < 0.5:
            resilience_assessment['resilience_gaps'].append('Limited adaptive capacity')

        return resilience_assessment

@dataclass
class FeedbackLoop(Node):
    """Feedback loops in SFM system interactions."""

    loop_name: Optional[str] = None
    loop_description: Optional[str] = None

    # Loop structure
    loop_elements: List[uuid.UUID] = field(default_factory=list)  # Elements in the loop
    loop_sequence: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)  # Element connections
    feedback_type: FeedbackType = FeedbackType.POSITIVE_FEEDBACK

    # Loop characteristics
    loop_strength: Optional[float] = None       # Overall loop strength (0-1)
    loop_delay: Optional[float] = None          # Time delay in loop (normalized)
    loop_stability: Optional[float] = None      # Stability of loop behavior

    # Dynamic properties
    amplification_factor: Optional[float] = None  # How much the loop amplifies changes
    equilibrium_point: Optional[float] = None     # Loop equilibrium (if applicable)
    oscillation_tendency: Optional[float] = None  # Tendency to oscillate

    # Loop behavior
    historical_behavior: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_patterns: List[str] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)

    # Impact assessment
    system_impact: Optional[float] = None       # Impact on overall system (0-1)
    beneficial_effects: List[str] = field(default_factory=list)
    detrimental_effects: List[str] = field(default_factory=list)
    unintended_consequences: List[str] = field(default_factory=list)

    # Management considerations
    controllability: Optional[float] = None     # How controllable the loop is
    intervention_points: List[str] = field(default_factory=list)
    management_strategies: List[str] = field(default_factory=list)

    def analyze_loop_dynamics(self) -> Dict[str, Any]:
        """Analyze dynamics of the feedback loop."""
        loop_analysis = {
            'loop_characteristics': {},
            'behavioral_analysis': {},
            'stability_analysis': {},
            'impact_analysis': {},
            'management_implications': {}
        }

        # Loop characteristics
        loop_analysis['loop_characteristics'] = {
            'feedback_type': self.feedback_type.name,
            'loop_length': len(self.loop_elements),
            'loop_strength': self.loop_strength or 0.0,
            'loop_delay': self.loop_delay or 0.0
        }

        # Behavioral analysis
        dominant_pattern = 'stable'
        if self.oscillation_tendency and self.oscillation_tendency > 0.6:
            dominant_pattern = 'oscillating'
        elif self.feedback_type == FeedbackType.RUNAWAY:
            dominant_pattern = 'exponential'

        loop_analysis['behavioral_analysis'] = {
            'dominant_pattern': dominant_pattern,
            'behavioral_patterns': self.behavioral_patterns.copy(),
            'trigger_conditions': len(self.trigger_conditions),
            'predictability': 1.0 - (self.oscillation_tendency or 0.0)
        }

        # Stability analysis
        stability_score = self.loop_stability or 0.5
        if self.feedback_type == FeedbackType.BALANCING:
            stability_score += 0.2
        elif self.feedback_type == FeedbackType.RUNAWAY:
            stability_score -= 0.3

        loop_analysis['stability_analysis'] = {
            'stability_score': max(0.0, min(1.0, stability_score)),
            'equilibrium_exists': self.equilibrium_point is not None,
            'oscillation_risk': self.oscillation_tendency or 0.0,
            'runaway_risk': 1.0 if self.feedback_type == FeedbackType.RUNAWAY else 0.2
        }

        # Impact analysis
        net_impact = 0.0
        if self.beneficial_effects:
            net_impact += len(self.beneficial_effects) * 0.2
        if self.detrimental_effects:
            net_impact -= len(self.detrimental_effects) * 0.3

        loop_analysis['impact_analysis'] = {
            'system_impact': self.system_impact or 0.0,
            'net_impact': max(-1.0, min(1.0, net_impact)),
            'beneficial_effects_count': len(self.beneficial_effects),
            'detrimental_effects_count': len(self.detrimental_effects),
            'unintended_consequences': len(self.unintended_consequences)
        }

        # Management implications
        loop_analysis['management_implications'] = {
            'controllability': self.controllability or 0.5,
            'intervention_points': len(self.intervention_points),
            'management_complexity': 'high' if len(self.loop_elements) > 5 else 'moderate',
            'priority_level': self._assess_management_priority()
        }

        return loop_analysis

    def _assess_management_priority(self) -> str:
        """Assess management priority for this feedback loop."""
        priority_score = 0.0

        # High impact loops need attention
        if self.system_impact and self.system_impact > 0.7:
            priority_score += 0.4

        # Unstable loops need management
        if self.loop_stability and self.loop_stability < 0.4:
            priority_score += 0.3

        # Runaway loops are high priority
        if self.feedback_type == FeedbackType.RUNAWAY:
            priority_score += 0.5

        # Detrimental effects increase priority
        if len(self.detrimental_effects) > 2:
            priority_score += 0.3

        # Low controllability increases priority
        if self.controllability and self.controllability < 0.3:
            priority_score += 0.2

        if priority_score >= 0.8:
            return 'critical'
        elif priority_score >= 0.6:
            return 'high'
        elif priority_score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def recommend_interventions(self) -> List[Dict[str, Any]]:
        """Recommend interventions for managing the feedback loop."""
        interventions = []

        # Based on feedback type
        if self.feedback_type == FeedbackType.RUNAWAY:
            interventions.append({
                'intervention_type': 'circuit_breaker',
                'description': 'Implement circuit breakers to prevent runaway effects',
                'urgency': 'high',
                'complexity': 'medium'
            })

        elif self.feedback_type == FeedbackType.POSITIVE_FEEDBACK and len(self.beneficial_effects) > 0:
            interventions.append({
                'intervention_type': 'amplification',
                'description': 'Strengthen positive feedback mechanisms',
                'urgency': 'medium',
                'complexity': 'low'
            })

        # Based on stability
        if self.loop_stability and self.loop_stability < 0.4:
            interventions.append({
                'intervention_type': 'stabilization',
                'description': 'Implement stabilization mechanisms',
                'urgency': 'high',
                'complexity': 'medium'
            })

        # Based on controllability
        if self.controllability and self.controllability < 0.5:
            interventions.append({
                'intervention_type': 'control_enhancement',
                'description': 'Develop better control mechanisms',
                'urgency': 'medium',
                'complexity': 'high'
            })

        return interventions

@dataclass
class EmergentEffect(Node):
    """Analysis of emergent effects in SFM systems."""

    effect_name: Optional[str] = None
    effect_description: Optional[str] = None

    # Emergence characteristics
    emergence_source: List[uuid.UUID] = field(default_factory=list)  # Source elements
    emergence_mechanism: Optional[str] = None
    emergence_conditions: List[str] = field(default_factory=list)

    # Effect properties
    effect_magnitude: Optional[float] = None    # Magnitude of emergent effect
    effect_persistence: Optional[float] = None # How persistent the effect is
    effect_predictability: Optional[float] = None  # How predictable

    # System implications
    system_level_impacts: Dict[SystemLevel, str] = field(default_factory=dict)
    value_implications: Dict[ValueCategory, float] = field(default_factory=dict)
    institutional_effects: Dict[uuid.UUID, str] = field(default_factory=dict)

    # Dynamic aspects
    evolution_pattern: Optional[str] = None
    triggering_events: List[str] = field(default_factory=list)
    sustaining_factors: List[str] = field(default_factory=list)
    terminating_conditions: List[str] = field(default_factory=list)

    # Policy implications
    policy_relevance: Optional[float] = None    # Relevance for policy (0-1)
    intervention_opportunities: List[str] = field(default_factory=list)
    policy_recommendations: List[str] = field(default_factory=list)

    def analyze_emergence_dynamics(self) -> Dict[str, Any]:
        """Analyze the dynamics of emergent effects."""
        emergence_analysis = {
            'emergence_assessment': {},
            'system_impact_analysis': {},
            'policy_implications': {},
            'management_considerations': {}
        }

        # Emergence assessment
        emergence_strength = 0.0
        if self.effect_magnitude:
            emergence_strength += self.effect_magnitude * 0.4
        if self.effect_persistence:
            emergence_strength += self.effect_persistence * 0.3
        if len(self.sustaining_factors) > 0:
            emergence_strength += min(len(self.sustaining_factors) / 5.0, 0.3)

        emergence_analysis['emergence_assessment'] = {
            'emergence_strength': emergence_strength,
            'predictability': self.effect_predictability or 0.0,
            'sustainability': self.effect_persistence or 0.0,
            'complexity': len(self.emergence_source) / 10.0,  # Normalize complexity
            'emergence_type': self._classify_emergence_type()
        }

        # System impact analysis
        emergence_analysis['system_impact_analysis'] = {
            'affected_system_levels': len(self.system_level_impacts),
            'value_implications': len(self.value_implications),
            'institutional_effects': len(self.institutional_effects),
            'overall_significance': self._assess_overall_significance()
        }

        # Policy implications
        emergence_analysis['policy_implications'] = {
            'policy_relevance': self.policy_relevance or 0.0,
            'intervention_opportunities': len(self.intervention_opportunities),
            'policy_complexity': 'high' if len(self.emergence_source) > 5 else 'moderate',
            'urgency_level': self._assess_policy_urgency()
        }

        return emergence_analysis

    def _classify_emergence_type(self) -> str:
        """Classify the type of emergence."""
        if self.effect_predictability and self.effect_predictability > 0.7:
            return 'predictable_emergence'
        elif self.effect_magnitude and self.effect_magnitude > 0.8:
            return 'strong_emergence'
        elif len(self.emergence_source) > 5:
            return 'complex_emergence'
        else:
            return 'weak_emergence'

    def _assess_overall_significance(self) -> float:
        """Assess overall significance of the emergent effect."""
        significance = 0.0

        if self.effect_magnitude:
            significance += self.effect_magnitude * 0.3

        if self.system_level_impacts:
            significance += min(len(self.system_level_impacts) / 5.0, 0.3)

        if self.value_implications:
            avg_value_impact = sum(abs(impact) for impact in self.value_implications.values()) / len(self.value_implications)
            significance += avg_value_impact * 0.2

        if self.policy_relevance:
            significance += self.policy_relevance * 0.2

        return min(significance, 1.0)

    def _assess_policy_urgency(self) -> str:
        """Assess urgency level for policy response."""
        urgency_score = 0.0

        if self.effect_magnitude and self.effect_magnitude > 0.8:
            urgency_score += 0.4

        if len(self.institutional_effects) > 3:
            urgency_score += 0.3

        if self.policy_relevance and self.policy_relevance > 0.7:
            urgency_score += 0.3

        if urgency_score >= 0.7:
            return 'high'
        elif urgency_score >= 0.4:
            return 'medium'
        else:
            return 'low'

@dataclass
class CrossImpactAnalysis(Node):
    """Comprehensive cross-impact analysis framework for SFM."""

    analysis_scope: Optional[str] = None
    analysis_purpose: Optional[str] = None

    # Analysis components
    impact_relationships: List[uuid.UUID] = field(default_factory=list)  # ImpactRelationship IDs
    system_interactions: List[uuid.UUID] = field(default_factory=list)   # SystemInteraction IDs
    feedback_loops: List[uuid.UUID] = field(default_factory=list)        # FeedbackLoop IDs
    emergent_effects: List[uuid.UUID] = field(default_factory=list)      # EmergentEffect IDs

    # Analysis scope
    analyzed_elements: List[uuid.UUID] = field(default_factory=list)
    analysis_timeframe: Optional[TimeSlice] = None
    analysis_context: Optional[str] = None

    # Cross-impact matrix
    impact_matrix: Dict[Tuple[uuid.UUID, uuid.UUID], float] = field(default_factory=dict)
    interaction_strengths: Dict[Tuple[uuid.UUID, uuid.UUID], float] = field(default_factory=dict)
    network_properties: Dict[str, float] = field(default_factory=dict)

    # System-level findings
    dominant_interactions: List[Tuple[uuid.UUID, uuid.UUID, float]] = field(default_factory=list)
    critical_feedback_loops: List[uuid.UUID] = field(default_factory=list)
    key_emergent_properties: List[uuid.UUID] = field(default_factory=list)

    # Analysis quality
    analysis_completeness: Optional[float] = None   # Coverage completeness
    analysis_confidence: Optional[float] = None     # Overall confidence
    validation_status: Optional[str] = None

    # Policy implications
    policy_insights: List[str] = field(default_factory=list)
    intervention_priorities: List[Dict[str, Any]] = field(default_factory=list)
    system_leverage_points: List[str] = field(default_factory=list)

    def conduct_comprehensive_analysis(self) -> Dict[str, Any]:
        """Conduct comprehensive cross-impact analysis."""
        analysis_results = {
            'analysis_overview': {},
            'impact_analysis': {},
            'system_interactions': {},
            'feedback_analysis': {},
            'emergent_effects': {},
            'synthesis': {},
            'recommendations': []
        }

        # Analysis overview
        analysis_results['analysis_overview'] = {
            'elements_analyzed': len(self.analyzed_elements),
            'relationships_identified': len(self.impact_relationships),
            'interactions_mapped': len(self.system_interactions),
            'feedback_loops_found': len(self.feedback_loops),
            'emergent_effects_identified': len(self.emergent_effects),
            'analysis_completeness': self.analysis_completeness or 0.0
        }

        # Impact analysis
        if self.impact_matrix:
            strong_impacts = [(
                source,
                target,
                strength) for (source,
                target),
                strength in self.impact_matrix.items() if abs(strength) > 0.6]
            analysis_results['impact_analysis'] = {
                'total_impact_relationships': len(self.impact_matrix),
                'strong_impact_relationships': len(strong_impacts),
                'network_density': len(
                    self.impact_matrix) / max(len(self.analyzed_elements) ** 2,
                    1),
                'dominant_impacts': [(str(s), str(t), st) for s, t, st in strong_impacts[:10]]
            }

        # System interactions
        analysis_results['system_interactions'] = {
            'interaction_count': len(self.system_interactions),
            'complex_interactions': 0,  # Would analyze actual interactions
            'stable_interactions': 0,   # Would analyze stability
        }

        # Feedback analysis
        analysis_results['feedback_analysis'] = {
            'total_feedback_loops': len(self.feedback_loops),
            'critical_loops': len(self.critical_feedback_loops),
            'reinforcing_loops': 0,  # Would categorize loop types
            'balancing_loops': 0
        }

        # Synthesis
        analysis_results['synthesis'] = {
            'system_complexity': self._assess_system_complexity(),
            'system_stability': self._assess_system_stability(),
            'intervention_leverage': len(self.system_leverage_points),
            'policy_priority_areas': len(self.intervention_priorities)
        }

        return analysis_results

    def _assess_system_complexity(self) -> float:
        """Assess overall system complexity."""
        complexity_score = 0.0

        # Structural complexity
        if self.analyzed_elements:
            element_ratio = len(self.impact_relationships) / len(self.analyzed_elements)
            complexity_score += min(element_ratio / 10.0, 0.3)

        # Interaction complexity
        complexity_score += min(len(self.system_interactions) / 20.0, 0.3)

        # Feedback complexity
        complexity_score += min(len(self.feedback_loops) / 10.0, 0.2)

        # Emergent complexity
        complexity_score += min(len(self.emergent_effects) / 5.0, 0.2)

        return min(complexity_score, 1.0)

    def _assess_system_stability(self) -> float:
        """Assess overall system stability."""
        stability_score = 0.5  # Default moderate stability

        # Feedback loop stability impact
        if len(self.critical_feedback_loops) > len(self.feedback_loops) * 0.3:
            stability_score -= 0.2  # Many critical loops reduce stability

        # Emergent effects impact
        if len(self.emergent_effects) > 3:
            stability_score -= 0.1  # Emergent effects can reduce predictability

        # Network density impact
        if self.network_properties.get('density', 0) > 0.7:
            stability_score -= 0.1  # Very dense networks can be unstable

        return max(0.0, min(stability_score, 1.0))

    def identify_system_leverage_points(self) -> List[Dict[str, Any]]:
        """Identify high-leverage intervention points in the system."""
        leverage_points = []

        # High-impact elements
        if self.dominant_interactions:
            for source, target, strength in self.dominant_interactions[:5]:
                leverage_points.append({
                    'type': 'dominant_interaction',
                    'description': f'Interaction between {source} and {target}',
                    'leverage_score': abs(strength),
                    'intervention_type': 'relationship_modification'
                })

        # Critical feedback loops
        for loop_id in self.critical_feedback_loops:
            leverage_points.append({
                'type': 'critical_feedback_loop',
                'description': f'Critical feedback loop {loop_id}',
                'leverage_score': 0.8,  # High leverage by definition
                'intervention_type': 'feedback_management'
            })

        # System leverage points
        for leverage_point in self.system_leverage_points:
            leverage_points.append({
                'type': 'system_leverage_point',
                'description': leverage_point,
                'leverage_score': 0.7,
                'intervention_type': 'system_restructuring'
            })

        # Sort by leverage score
        leverage_points.sort(key=lambda x: x['leverage_score'], reverse=True)

        return leverage_points

    def generate_policy_recommendations(self) -> Dict[str, Any]:
        """Generate policy recommendations based on cross-impact analysis."""
        policy_recommendations = {
            'strategic_recommendations': [],
            'intervention_priorities': [],
            'system_management': [],
            'monitoring_requirements': [],
            'implementation_considerations': []
        }

        # Strategic recommendations based on system properties
        complexity = self._assess_system_complexity()
        stability = self._assess_system_stability()

        if complexity > 0.7:
            policy_recommendations['strategic_recommendations'].append(
                'Adopt adaptive management approaches for high-complexity system'
            )

        if stability < 0.4:
            policy_recommendations['strategic_recommendations'].append(
                'Implement stability mechanisms to reduce system volatility'
            )

        # Intervention priorities
        leverage_points = self.identify_system_leverage_points()
        for point in leverage_points[:3]:  # Top 3 priorities
            policy_recommendations['intervention_priorities'].append({
                'priority': point['description'],
                'approach': point['intervention_type'],
                'expected_impact': 'high' if point['leverage_score'] > 0.7 else 'medium'
            })

        # System management recommendations
        if len(self.critical_feedback_loops) > 0:
            policy_recommendations['system_management'].append(
                'Establish feedback loop monitoring and management systems'
            )

        if len(self.emergent_effects) > 2:
            policy_recommendations['system_management'].append(
                'Develop capacity for managing emergent system properties'
            )

        return policy_recommendations
