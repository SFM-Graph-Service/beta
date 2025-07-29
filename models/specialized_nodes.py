"""
Specialized nodes for SFM modeling.

This module defines specialized nodes for belief systems, technology systems,
indicators, feedback loops, and other specialized SFM entities based on
F. Gregory Hayden's Social Fabric Matrix framework.

## Key SFM Components Implemented:

- **MatrixCell**: Core SFM component representing institution-criteria relationships
- **SFMCriteria**: Evaluation criteria for institutional analysis
- **ToolSkillTechnologyComplex**: Hayden's integrated technology system concept
- **CeremonialInstrumentalClassification**: Central SFM behavioral distinction
- **DigraphAnalysis**: Institutional dependency analysis
- **ValueJudgment**: Explicit value judgments in policy analysis
- **SFM-specific attributes**: For ceremonial/instrumental analysis
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from models.base_nodes import Node
from models.metadata_models import TemporalDynamics, ValidationRule
from models.sfm_enums import (
    FeedbackPolarity,
    FeedbackType,
    TechnologyReadinessLevel,
    ValueCategory,
    SystemPropertyType,
    PolicyInstrumentType,
    EnumValidator,
    CorrelationType,
    EvidenceQuality,
    CriteriaType,
    MeasurementApproach,
    CeremonialInstrumentalType,
    ValueJudgmentType,
    ToolSkillTechnologyType,
    ProblemSolvingStage,
    InstitutionalScope,
    GovernanceMechanism,
    CrossImpactType,
    EnforcementType,
    DecisionMakingType,
    TransactionCostType,
    CoordinationMechanismType,
    CoordinationScope,
    CommonsGovernanceType,
    SocialValueDimension,
    SystemArchetype,
    PathDependencyType,
    # New enums for missing components
    ValueSystemType,
    SocialFabricIndicatorType,
    SocialCostType,
    InstitutionalLevel,
    NormativeFramework,
    EvolutionaryStage,
    DependencyStrength,
    CriteriaPriority,
    CorrelationScale,
    BoundaryType,
    ProvisioningStage,
    ConflictType,
)


@dataclass
class BeliefSystem(Node):
    """Cultural myths, ideology or worldview that guides decision-making."""

    strength: Optional[float] = None  # Cultural embeddedness (0-1)
    domain: Optional[str] = None  # Area of society where belief operates
    # SFM-specific additions:
    ceremonial_function: bool = True  # Most beliefs are ceremonial in SFM
    supporting_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    conflicting_beliefs: List[uuid.UUID] = field(default_factory=lambda: [])
    change_resistance: Optional[float] = None  # 0-1 scale
    legitimacy_provider: bool = False  # Does it provide institutional legitimacy?
    problem_solving_hindrance: Optional[float] = None  # How much it blocks adaptation


@dataclass
class FeedbackLoop(Node):
    """Represents a feedback loop in the Social Fabric Matrix."""

    relationships: List[uuid.UUID] = field(default_factory=lambda: [])
    description: Optional[str] = None
    polarity: Optional[FeedbackPolarity] = None  # "reinforcing" or "balancing"
    strength: Optional[float] = None  # Measure of loop strength/impact
    type: Optional[FeedbackType] = None  # e.g. "positive", "negative", "neutral"
    # SFM-specific additions:
    ceremonial_reinforcement: Optional[bool] = None  # Does it reinforce status quo?
    institutional_impact: List[uuid.UUID] = field(default_factory=lambda: [])  # Affected institutions
    matrix_cell_effects: List[uuid.UUID] = field(default_factory=lambda: [])  # Affected matrix cells
    time_delay: Optional[float] = None  # Lag time in feedback (in appropriate time units)
    system_level_effect: Optional[SystemPropertyType] = None


@dataclass
class TechnologySystem(Node):
    """Coherent system of techniques, tools and knowledge."""

    maturity: Optional[TechnologyReadinessLevel] = None  # Technology readiness level
    compatibility: Dict[str, float] = field(default_factory=lambda: {})  # Fit with other systems
    # SFM-specific additions:
    institutional_requirements: List[uuid.UUID] = field(default_factory=lambda: [])  # Required institutions
    skill_requirements: List[str] = field(default_factory=lambda: [])  # Required skills
    resource_requirements: List[uuid.UUID] = field(default_factory=lambda: [])  # Required resources
    ceremonial_aspects: Optional[float] = None  # How much is ceremonial vs functional (0-1)
    problem_solving_capacity: Optional[float] = None  # Effectiveness for problem solving (0-1)
    adaptation_barriers: List[str] = field(default_factory=lambda: [])  # Barriers to adoption


@dataclass
class Indicator(Node):
    """Measurable proxy for system performance."""

    value_category: Optional[ValueCategory] = (
        None  # Non-default field moved to the beginning
    )
    measurement_unit: Optional[str] = None  # Non-default field moved to the beginning
    current_value: Optional[float] = None
    target_value: Optional[float] = None
    threshold_values: Dict[str, float] = field(default_factory=lambda: {})
    temporal_dynamics: Optional[TemporalDynamics] = None  # Track changes over time

    def __post_init__(self) -> None:
        """Validate indicator configuration after initialization."""
        # Validate value category context if measurement unit suggests measurement type
        if self.value_category and self.measurement_unit:
            # Infer measurement context from measurement unit
            measurement_context = "quantitative"  # Default assumption
            if any(qual_indicator in self.measurement_unit.lower() for qual_indicator in
                   ['scale', 'rating', 'level', 'score', 'index']):
                measurement_context = "qualitative"

            EnumValidator.validate_value_category_context(
                self.value_category, measurement_context
            )


@dataclass
class AnalyticalContext(Node):  # pylint: disable=too-many-instance-attributes
    """Contains metadata about analysis parameters and configuration."""

    methods_used: List[str] = field(default_factory=lambda: [])
    assumptions: Dict[str, str] = field(default_factory=lambda: {})
    data_sources: Dict[str, str] = field(default_factory=lambda: {})
    validation_approach: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=lambda: {})
    validation_rules: List[ValidationRule] = field(default_factory=lambda: [])


@dataclass
class SystemProperty(Node):
    """Represents a system-level property or metric of the SFM."""

    property_type: SystemPropertyType = SystemPropertyType.STRUCTURAL
    value: Any = None
    unit: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    # Nodes that this property applies to
    affected_nodes: List[uuid.UUID] = field(default_factory=lambda: [])
    # Relationships that contribute to this property
    contributing_relationships: List[uuid.UUID] = field(default_factory=lambda: [])
    # id, name, and description are inherited from Node


@dataclass
class PolicyInstrument(Node):
    """Specific tools used to implement policies."""

    # regulatory, economic, voluntary, information
    instrument_type: PolicyInstrumentType = PolicyInstrumentType.REGULATORY
    target_behavior: Optional[str] = None
    compliance_mechanism: Optional[str] = None
    effectiveness_measure: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate policy instrument configuration after initialization."""
        # Validate instrument type if target behavior is specified
        if self.target_behavior:
            EnumValidator.validate_policy_instrument_combination(
                self.instrument_type, self.target_behavior
            )


@dataclass
class MatrixCell(Node):
    """Represents a cell in the Social Fabric Matrix showing institution-criteria relationship with enhanced SFM integration."""
    
    institution_id: Optional[uuid.UUID] = None  # Made optional with default to fix dataclass ordering
    criteria_id: Optional[uuid.UUID] = None     # Made optional with default to fix dataclass ordering
    correlation_type: CorrelationType = CorrelationType.UNKNOWN
    correlation_strength: Optional[float] = None  # 0-1 scale
    correlation_scale: CorrelationScale = CorrelationScale.NEUTRAL  # Hayden's standardized -3 to +3 scale
    evidence_quality: EvidenceQuality = EvidenceQuality.LOW
    justification: Optional[str] = None
    data_sources: List[str] = field(default_factory=lambda: [])
    confidence_level: Optional[float] = None  # 0-1 scale
    last_updated: datetime = field(default_factory=datetime.now)
    reviewed_by: Optional[str] = None
    review_date: Optional[datetime] = None
    
    # Enhanced SFM Integration - Delivery System Modeling
    deliveries_provided: List[Dict[str, Any]] = field(default_factory=lambda: [])  # What this cell delivers to others
    deliveries_received: List[Dict[str, Any]] = field(default_factory=lambda: [])  # What this cell receives from others
    delivery_quality: Optional[float] = None  # Quality of deliveries (0-1)
    delivery_reliability: Optional[float] = None  # Reliability of deliveries (0-1)
    delivery_capacity: Optional[float] = None  # Capacity for deliveries (0-1)
    
    # Cultural Integration - Hayden's cultural values/beliefs/attitudes framework
    cultural_values_influence: Dict[str, float] = field(default_factory=lambda: {})  # Value type -> influence strength
    social_beliefs_alignment: Dict[str, float] = field(default_factory=lambda: {})  # Belief type -> alignment level
    attitude_mediation: Dict[str, float] = field(default_factory=lambda: {})  # Attitude type -> mediation effect
    cultural_legitimacy: Optional[float] = None  # Cultural legitimacy of the relationship (0-1)
    
    # Ceremonial vs Instrumental Analysis
    ceremonial_component: Optional[float] = None  # Ceremonial aspects of the relationship (0-1)
    instrumental_component: Optional[float] = None  # Instrumental aspects of the relationship (0-1)
    ceremonial_barriers: List[str] = field(default_factory=lambda: [])  # Ceremonial obstacles
    instrumental_enablers: List[str] = field(default_factory=lambda: [])  # Instrumental facilitators
    
    # Technology Integration
    technology_dependencies: List[uuid.UUID] = field(default_factory=lambda: [])  # Required technologies
    technology_compatibility: Optional[float] = None  # How well technology supports relationship (0-1)
    technological_change_sensitivity: Optional[float] = None  # Sensitivity to tech changes (0-1)
    
    # Ecological System Integration
    ecological_constraints: List[str] = field(default_factory=lambda: [])  # Environmental constraints
    ecological_impact: Optional[float] = None  # Environmental impact (-1 to +1)
    ecological_sustainability: Optional[float] = None  # Sustainability rating (0-1)
    resource_flows: Dict[str, float] = field(default_factory=lambda: {})  # Resource type -> flow amount
    
    # Network Process Integration
    network_position: Optional[str] = None  # Position in network ("central", "peripheral", "bridge")
    network_influence: Optional[float] = None  # Influence within network (0-1)
    feedback_loops: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Feedback relationships
    system_criticality: Optional[float] = None  # Criticality to overall system (0-1)
    
    # Social Indicator Integration
    social_indicators: Dict[str, float] = field(default_factory=lambda: {})  # Indicator type -> value
    indicator_trends: Dict[str, str] = field(default_factory=lambda: {})  # Indicator -> trend direction
    monitoring_frequency: Optional[str] = None  # How often indicators are updated
    
    # Temporal Sequencing
    temporal_sequence_position: Optional[int] = None  # Position in temporal sequence
    sequence_dependencies: List[uuid.UUID] = field(default_factory=lambda: [])  # Dependent cells in sequence
    temporal_coordination_requirements: List[str] = field(default_factory=lambda: [])  # Timing requirements
    
    def __post_init__(self) -> None:
        """Validate that required fields are provided."""
        if self.institution_id is None:
            raise ValueError("institution_id is required for MatrixCell")
        if self.criteria_id is None:
            raise ValueError("criteria_id is required for MatrixCell")
    
    def calculate_weighted_score(self, criteria_weight: float) -> Optional[float]:
        """Calculate weighted contribution to overall matrix score."""
        if self.correlation_strength is None:
            return None
        
        multiplier = 1.0 if self.correlation_type == CorrelationType.POSITIVE else -1.0
        if self.correlation_type == CorrelationType.NEUTRAL:
            multiplier = 0.0
        elif self.correlation_type == CorrelationType.UNKNOWN:
            multiplier = 0.0  # Unknown treated as neutral for scoring
            
        return self.correlation_strength * multiplier * criteria_weight

    def get_confidence_adjusted_score(self) -> Optional[float]:
        """Get score adjusted for confidence level."""
        if self.correlation_strength is None or self.confidence_level is None:
            return None
        return self.correlation_strength * self.confidence_level
    
    def get_evidence_weighted_score(self) -> Optional[float]:
        """Get score weighted by evidence quality."""
        if self.correlation_strength is None:
            return None
        
        # Weight evidence quality: LOW=0.5, MEDIUM=0.7, HIGH=0.9, VERIFIED=1.0
        evidence_weights = {
            EvidenceQuality.LOW: 0.5,
            EvidenceQuality.MEDIUM: 0.7,
            EvidenceQuality.HIGH: 0.9,
            EvidenceQuality.VERIFIED: 1.0
        }
        
        evidence_weight = evidence_weights.get(self.evidence_quality, 0.5)
        return self.correlation_strength * evidence_weight
    
    def get_standardized_correlation_value(self) -> float:
        """Convert correlation scale to standardized numeric value (-3 to +3)."""
        scale_values = {
            CorrelationScale.STRONGLY_NEGATIVE: -3.0,
            CorrelationScale.MODERATELY_NEGATIVE: -2.0,
            CorrelationScale.WEAKLY_NEGATIVE: -1.0,
            CorrelationScale.NEUTRAL: 0.0,
            CorrelationScale.WEAKLY_POSITIVE: 1.0,
            CorrelationScale.MODERATELY_POSITIVE: 2.0,
            CorrelationScale.STRONGLY_POSITIVE: 3.0
        }
        return scale_values.get(self.correlation_scale, 0.0)
    
    def set_correlation_from_numeric(self, value: float) -> None:
        """Set correlation scale from numeric value (-3 to +3)."""
        if value <= -2.5:
            self.correlation_scale = CorrelationScale.STRONGLY_NEGATIVE
        elif value <= -1.5:
            self.correlation_scale = CorrelationScale.MODERATELY_NEGATIVE
        elif value <= -0.5:
            self.correlation_scale = CorrelationScale.WEAKLY_NEGATIVE
        elif value <= 0.5:
            self.correlation_scale = CorrelationScale.NEUTRAL
        elif value <= 1.5:
            self.correlation_scale = CorrelationScale.WEAKLY_POSITIVE
        elif value <= 2.5:
            self.correlation_scale = CorrelationScale.MODERATELY_POSITIVE
        else:
            self.correlation_scale = CorrelationScale.STRONGLY_POSITIVE
    
    def assess_cell_quality(self) -> Dict[str, Any]:
        """Comprehensive assessment of matrix cell quality."""
        assessment: Dict[str, Any] = {
            "data_quality": "unknown",
            "confidence": "medium",
            "completeness": "partial",
            "consistency": "consistent",
            "review_status": "needs_review"
        }
        
        # Assess data quality
        if len(self.data_sources) >= 3 and self.evidence_quality in [EvidenceQuality.HIGH, EvidenceQuality.VERIFIED]:
            assessment["data_quality"] = "high"
        elif len(self.data_sources) >= 2 and self.evidence_quality == EvidenceQuality.MEDIUM:
            assessment["data_quality"] = "moderate"
        else:
            assessment["data_quality"] = "low"
        
        # Assess confidence
        if self.confidence_level:
            if self.confidence_level >= 0.8:
                assessment["confidence"] = "high"
            elif self.confidence_level >= 0.6:
                assessment["confidence"] = "medium"
            else:
                assessment["confidence"] = "low"
        
        # Assess completeness
        required_fields: list[Optional[Any]] = [self.institution_id, self.criteria_id, self.justification]
        completed_fields = sum(1 for field in required_fields if field is not None)
        if completed_fields == len(required_fields):
            assessment["completeness"] = "complete"
        elif completed_fields >= len(required_fields) // 2:
            assessment["completeness"] = "partial"
        else:
            assessment["completeness"] = "incomplete"
        
        # Check consistency between correlation measures
        if self.correlation_strength is not None:
            numeric_correlation = self.get_standardized_correlation_value()
            expected_strength = abs(numeric_correlation) / 3.0  # Normalize to 0-1
            
            if abs(self.correlation_strength - expected_strength) > 0.3:
                assessment["consistency"] = "inconsistent"
        
        # Check review status
        if self.reviewed_by and self.review_date:
            assessment["review_status"] = "reviewed"
            
        return assessment
    
    def analyze_delivery_relationships(self) -> Dict[str, Any]:
        """Analyze delivery relationships for this matrix cell - core to Hayden's SFM."""
        delivery_analysis: Dict[str, Any] = {
            "deliveries_provided_count": len(self.deliveries_provided),
            "deliveries_received_count": len(self.deliveries_received),
            "delivery_balance": "unknown",
            "delivery_effectiveness": 0.0,
            "delivery_criticality": 0.0
        }
        
        # Assess delivery balance
        provided = len(self.deliveries_provided)
        received = len(self.deliveries_received)
        if provided > received * 1.5:
            delivery_analysis["delivery_balance"] = "net_provider"
        elif received > provided * 1.5:
            delivery_analysis["delivery_balance"] = "net_receiver"
        else:
            delivery_analysis["delivery_balance"] = "balanced"
        
        # Calculate delivery effectiveness
        if self.delivery_quality and self.delivery_reliability:
            delivery_analysis["delivery_effectiveness"] = (self.delivery_quality + self.delivery_reliability) / 2
        
        # Assess criticality based on system criticality and network influence
        if self.system_criticality and self.network_influence:
            delivery_analysis["delivery_criticality"] = (self.system_criticality + self.network_influence) / 2
        
        return delivery_analysis
    
    def assess_cultural_integration(self) -> Dict[str, Any]:
        """Assess cultural integration aspects - Hayden's values/beliefs/attitudes framework."""
        cultural_assessment: Dict[str, Any] = {
            "cultural_values_count": len(self.cultural_values_influence),
            "social_beliefs_count": len(self.social_beliefs_alignment),
            "attitude_mediation_count": len(self.attitude_mediation),
            "cultural_coherence": 0.0,
            "cultural_legitimacy_level": "unknown"
        }
        
        # Calculate cultural coherence (alignment between values, beliefs, attitudes)
        total_cultural_elements = (
            len(self.cultural_values_influence) +
            len(self.social_beliefs_alignment) +
            len(self.attitude_mediation)
        )
        
        if total_cultural_elements > 0:
            values_avg = sum(self.cultural_values_influence.values()) / len(self.cultural_values_influence) if self.cultural_values_influence else 0
            beliefs_avg = sum(self.social_beliefs_alignment.values()) / len(self.social_beliefs_alignment) if self.social_beliefs_alignment else 0
            attitudes_avg = sum(self.attitude_mediation.values()) / len(self.attitude_mediation) if self.attitude_mediation else 0
            
            # Coherence is inversely related to variance
            cultural_scores = [values_avg, beliefs_avg, attitudes_avg]
            if len([s for s in cultural_scores if s > 0]) > 1:
                variance = sum((s - sum(cultural_scores)/3)**2 for s in cultural_scores) / len(cultural_scores)
                cultural_assessment["cultural_coherence"] = max(0.0, 1.0 - variance)
        
        # Assess cultural legitimacy level
        if self.cultural_legitimacy:
            if self.cultural_legitimacy >= 0.8:
                cultural_assessment["cultural_legitimacy_level"] = "high"
            elif self.cultural_legitimacy >= 0.6:
                cultural_assessment["cultural_legitimacy_level"] = "moderate"
            else:
                cultural_assessment["cultural_legitimacy_level"] = "low"
        
        return cultural_assessment
    
    def analyze_ceremonial_instrumental_balance(self) -> Dict[str, Any]:
        """Analyze ceremonial vs instrumental balance - central to Hayden's framework."""
        balance_analysis: Dict[str, Any] = {
            "ceremonial_score": self.ceremonial_component or 0.0,
            "instrumental_score": self.instrumental_component or 0.0,
            "balance_type": "unknown",
            "dominant_aspect": "unknown",
            "barriers_count": len(self.ceremonial_barriers),
            "enablers_count": len(self.instrumental_enablers),
            "adjustment_recommendations": []
        }
        
        # Determine balance type
        ceremonial = self.ceremonial_component or 0.0
        instrumental = self.instrumental_component or 0.0
        
        if ceremonial > instrumental * 1.5:
            balance_analysis["balance_type"] = "ceremonial_dominated"
            balance_analysis["dominant_aspect"] = "ceremonial"
            balance_analysis["adjustment_recommendations"].append("Increase instrumental problem-solving capacity")
        elif instrumental > ceremonial * 1.5:
            balance_analysis["balance_type"] = "instrumental_dominated"
            balance_analysis["dominant_aspect"] = "instrumental"
            balance_analysis["adjustment_recommendations"].append("Consider ceremonial stabilization needs")
        else:
            balance_analysis["balance_type"] = "balanced"
            balance_analysis["dominant_aspect"] = "balanced"
        
        # Assess barriers vs enablers
        if len(self.ceremonial_barriers) > len(self.instrumental_enablers):
            balance_analysis["adjustment_recommendations"].append("Address ceremonial barriers")
        elif len(self.instrumental_enablers) > len(self.ceremonial_barriers):
            balance_analysis["adjustment_recommendations"].append("Leverage instrumental enablers")
        
        return balance_analysis
    
    def assess_ecological_integration(self) -> Dict[str, Any]:
        """Assess ecological system integration - part of Hayden's comprehensive approach."""
        ecological_assessment: Dict[str, Any] = {
            "ecological_constraints_count": len(self.ecological_constraints),
            "ecological_impact_level": "unknown",
            "sustainability_rating": "unknown",
            "resource_flows_count": len(self.resource_flows),
            "environmental_health": 0.0
        }
        
        # Assess ecological impact level
        if self.ecological_impact is not None:
            if self.ecological_impact >= 0.5:
                ecological_assessment["ecological_impact_level"] = "positive"
            elif self.ecological_impact <= -0.5:
                ecological_assessment["ecological_impact_level"] = "negative"
            else:
                ecological_assessment["ecological_impact_level"] = "neutral"
        
        # Assess sustainability rating
        if self.ecological_sustainability is not None:
            if self.ecological_sustainability >= 0.8:
                ecological_assessment["sustainability_rating"] = "high"
            elif self.ecological_sustainability >= 0.6:
                ecological_assessment["sustainability_rating"] = "moderate"
            else:
                ecological_assessment["sustainability_rating"] = "low"
        
        # Calculate environmental health score
        if self.ecological_impact is not None and self.ecological_sustainability is not None:
            # Weight sustainability higher than impact
            ecological_assessment["environmental_health"] = (
                self.ecological_sustainability * 0.7 + 
                max(0, self.ecological_impact) * 0.3
            )
        
        return ecological_assessment
    
    def analyze_social_indicators(self) -> Dict[str, Any]:
        """Analyze social indicators - key to Hayden's methodology."""
        indicator_analysis: Dict[str, Any] = {
            "indicators_count": len(self.social_indicators),
            "trending_indicators": {"improving": [], "declining": [], "stable": []},
            "indicator_coverage": "unknown",
            "monitoring_adequacy": "unknown"
        }
        
        # Categorize trends
        for indicator, trend in self.indicator_trends.items():
            if trend in ["improving", "increasing", "positive"]:
                indicator_analysis["trending_indicators"]["improving"].append(indicator)
            elif trend in ["declining", "decreasing", "negative"]:
                indicator_analysis["trending_indicators"]["declining"].append(indicator)
            else:
                indicator_analysis["trending_indicators"]["stable"].append(indicator)
        
        # Assess indicator coverage
        if len(self.social_indicators) >= 5:
            indicator_analysis["indicator_coverage"] = "comprehensive"
        elif len(self.social_indicators) >= 3:
            indicator_analysis["indicator_coverage"] = "adequate"
        else:
            indicator_analysis["indicator_coverage"] = "limited"
        
        # Assess monitoring adequacy
        if self.monitoring_frequency in ["daily", "weekly", "monthly"]:
            indicator_analysis["monitoring_adequacy"] = "adequate"
        else:
            indicator_analysis["monitoring_adequacy"] = "insufficient"
        
        return indicator_analysis


@dataclass
class SFMCriteria(Node):
    """Evaluation criteria used in the Social Fabric Matrix with Hayden's priority system."""
    
    criteria_type: CriteriaType = CriteriaType.SOCIAL
    measurement_approach: MeasurementApproach = MeasurementApproach.QUALITATIVE
    priority: CriteriaPriority = CriteriaPriority.SECONDARY  # Hayden's primary/secondary distinction
    weight: Optional[float] = None  # Relative importance (0-1)
    sub_criteria: List[uuid.UUID] = field(default_factory=lambda: [])
    evaluation_method: Optional[str] = None
    data_requirements: List[str] = field(default_factory=lambda: [])
    measurement_frequency: Optional[str] = None
    responsible_party: Optional[str] = None
    
    # Hayden's priority system integration
    life_process_relevance: Optional[float] = None  # Relevance to life process enhancement (0-1)
    instrumental_capacity: Optional[float] = None  # Problem-solving capacity measure (0-1)
    ceremonial_bias_risk: Optional[float] = None  # Risk of ceremonial distortion (0-1)
    normative_justification: Optional[str] = None  # Why this criterion matters normatively
    
    def calculate_hayden_priority_score(self) -> float:
        """Calculate priority score using Hayden's framework."""
        if self.priority == CriteriaPriority.PRIMARY:
            base_score = 1.0
        elif self.priority == CriteriaPriority.SECONDARY:
            base_score = 0.7
        else:  # TERTIARY
            base_score = 0.4
        
        # Adjust based on life process relevance
        if self.life_process_relevance is not None:
            base_score *= (0.5 + 0.5 * self.life_process_relevance)
        
        # Penalize for ceremonial bias risk
        if self.ceremonial_bias_risk is not None:
            base_score *= (1.0 - 0.3 * self.ceremonial_bias_risk)
        
        return max(0.0, min(1.0, base_score))
    
    def assess_criterion_quality(self) -> Dict[str, Any]:
        """Assess the quality and appropriateness of this criterion."""
        assessment: Dict[str, Any] = {
            "priority_appropriateness": "appropriate",
            "measurement_feasibility": "feasible",
            "normative_strength": "moderate",
            "bias_risk": "low"
        }
        
        # Check priority appropriateness
        if self.priority == CriteriaPriority.PRIMARY and (self.life_process_relevance or 0) < 0.5:
            assessment["priority_appropriateness"] = "questionable"
        
        # Check measurement feasibility
        if self.measurement_approach == MeasurementApproach.QUANTITATIVE and not self.data_requirements:
            assessment["measurement_feasibility"] = "difficult"
        
        # Check normative strength
        if self.normative_justification:
            assessment["normative_strength"] = "strong"
        elif not self.normative_justification and self.priority == CriteriaPriority.PRIMARY:
            assessment["normative_strength"] = "weak"
        
        # Check bias risk
        if self.ceremonial_bias_risk and self.ceremonial_bias_risk > 0.7:
            assessment["bias_risk"] = "high"
        elif self.ceremonial_bias_risk and self.ceremonial_bias_risk > 0.3:
            assessment["bias_risk"] = "moderate"
        
        return assessment


@dataclass
class ToolSkillTechnologyComplex(Node):
    """Hayden's integrated tool-skill-technology system."""
    
    technology_type: ToolSkillTechnologyType = ToolSkillTechnologyType.PHYSICAL_TOOL
    physical_tools: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to TechnologySystem
    required_skills: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to skill nodes
    knowledge_base: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to information resources
    integration_level: Optional[float] = None  # How well integrated (0-1)
    problem_solving_capacity: Optional[float] = None  # Effectiveness (0-1)
    institutional_support_required: List[uuid.UUID] = field(default_factory=lambda: [])
    compatibility_constraints: Dict[str, str] = field(default_factory=lambda: {})
    maintenance_requirements: List[str] = field(default_factory=lambda: [])


@dataclass
class CeremonialInstrumentalClassification(Node):
    """Classifies behaviors/institutions as ceremonial vs instrumental."""
    
    classification: CeremonialInstrumentalType = CeremonialInstrumentalType.MIXED
    ceremonial_score: Optional[float] = None  # 0-1, higher = more ceremonial
    instrumental_score: Optional[float] = None  # 0-1, higher = more instrumental
    change_resistance: Optional[float] = None  # Resistance to adaptive change (0-1)
    problem_solving_contribution: Optional[float] = None  # Contribution to problem solving (0-1)
    status_quo_reinforcement: Optional[float] = None  # Reinforces existing patterns (0-1)
    adaptive_potential: Optional[float] = None  # Capacity for change (0-1)
    supporting_evidence: List[str] = field(default_factory=lambda: [])
    classification_rationale: Optional[str] = None
    temporal_dynamics: Optional[TemporalDynamics] = None


@dataclass
class DigraphAnalysis(Node):
    """Enhanced digraph analysis with sequence analysis for institutional dependency tracking."""
    
    analyzed_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    dependency_matrix: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})
    cycle_detection: List[List[uuid.UUID]] = field(default_factory=lambda: [])  # Circular dependencies
    path_analysis: Dict[str, List[uuid.UUID]] = field(default_factory=lambda: {})
    critical_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # High dependency nodes
    leverage_points: List[uuid.UUID] = field(default_factory=lambda: [])  # High influence nodes
    stability_score: Optional[float] = None  # System stability measure (0-1)
    complexity_measure: Optional[float] = None  # System complexity (0-1)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    methodology_notes: Optional[str] = None
    
    # Enhanced sequence analysis capabilities
    propagation_sequences: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {})  # Change propagation paths
    temporal_dependencies: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Time-lagged dependencies
    sequence_patterns: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Recurring sequence patterns
    cascade_potential: Dict[str, float] = field(default_factory=lambda: {})  # Institution -> cascade risk
    
    # Dynamic sequence properties
    sequence_stability: Optional[float] = None  # Stability of propagation sequences (0-1)
    adaptation_pathways: List[List[uuid.UUID]] = field(default_factory=lambda: [])  # Paths for system adaptation
    bottleneck_sequences: List[List[uuid.UUID]] = field(default_factory=lambda: [])  # Sequence bottlenecks
    
    def analyze_propagation_sequences(self, initial_change: uuid.UUID, time_steps: int = 5) -> List[Dict[str, Any]]:
        """Analyze how changes propagate through the institutional network over time."""
        sequences: List[Dict[str, Any]] = []
        
        # Start with initial change
        current_affected: Dict[str, float] = {str(initial_change): 1.0}  # Institution ID -> impact strength
        sequence_step: Dict[str, Any] = {
            "step": 0,
            "affected_institutions": current_affected.copy(),
            "new_impacts": current_affected.copy(),
            "cumulative_impact": sum(current_affected.values())
        }
        sequences.append(sequence_step)
        
        # Propagate through time steps
        for step in range(1, time_steps + 1):
            new_impacts: Dict[str, float] = {}
            
            # For each currently affected institution, find its dependencies
            for institution_id, impact_strength in current_affected.items():
                if institution_id in self.dependency_matrix:
                    for dependent_id, dependency_strength in self.dependency_matrix[institution_id].items():
                        # Calculate propagated impact (with decay)
                        decay_factor = 0.8 ** step  # Impact decays over time
                        propagated_impact = impact_strength * dependency_strength * decay_factor
                        
                        if propagated_impact > 0.1:  # Threshold for significant impact
                            if dependent_id not in new_impacts:
                                new_impacts[dependent_id] = 0.0
                            new_impacts[dependent_id] += propagated_impact
            
            # Add new impacts to cumulative
            for inst_id, impact in new_impacts.items():
                if inst_id not in current_affected:
                    current_affected[inst_id] = 0.0
                current_affected[inst_id] += impact
            
            sequence_step = {
                "step": step,
                "affected_institutions": current_affected.copy(),
                "new_impacts": new_impacts,
                "cumulative_impact": sum(current_affected.values())
            }
            sequences.append(sequence_step)
            
            # Stop if no new significant impacts
            if not new_impacts:
                break
        
        return sequences
    
    def identify_critical_sequences(self) -> List[Dict[str, Any]]:
        """Identify sequences that are critical for system functioning."""
        critical_sequences: List[Dict[str, Any]] = []
        
        # Analyze each potential starting point
        for institution_id in self.analyzed_institutions:
            sequences = self.analyze_propagation_sequences(institution_id)
            
            # Calculate sequence criticality
            max_impact = max(seq["cumulative_impact"] for seq in sequences)
            affected_count = len(sequences[-1]["affected_institutions"]) if sequences else 0
            
            if max_impact > 2.0 or affected_count > len(self.analyzed_institutions) * 0.5:
                critical_sequences.append({
                    "starting_institution": institution_id,
                    "max_impact": max_impact,
                    "institutions_affected": affected_count,
                    "sequence_length": len(sequences),
                    "criticality_score": max_impact * (affected_count / len(self.analyzed_institutions))
                })
        
        return sorted(critical_sequences, key=lambda x: x["criticality_score"], reverse=True)
    
    def detect_sequence_patterns(self) -> List[Dict[str, Any]]:
        """Detect recurring patterns in propagation sequences."""
        patterns: List[Dict[str, Any]] = []
        
        # Analyze sequences for common patterns
        all_sequences: List[List[Dict[str, Any]]] = []
        for institution_id in self.analyzed_institutions[:10]:  # Limit for performance
            sequences = self.analyze_propagation_sequences(institution_id, time_steps=3)
            all_sequences.append(sequences)
        
        # Look for common propagation paths
        path_frequency: Dict[str, int] = {}
        for sequences in all_sequences:
            for i in range(len(sequences) - 1):
                current_step = sequences[i]
                next_step = sequences[i + 1]
                
                # Create path signature
                current_institutions = set(current_step["affected_institutions"].keys())
                new_institutions = set(next_step["new_impacts"].keys())
                
                if new_institutions:  # Only if there are new impacts
                    path_key = f"{len(current_institutions)}->{len(new_institutions)}"
                    if path_key not in path_frequency:
                        path_frequency[path_key] = 0
                    path_frequency[path_key] += 1
        
        # Identify frequent patterns
        total_sequences = len(all_sequences)
        for path_pattern, frequency in path_frequency.items():
            if frequency / total_sequences > 0.3:  # Appears in >30% of sequences
                patterns.append({
                    "pattern": path_pattern,
                    "frequency": frequency,
                    "prevalence": frequency / total_sequences,
                    "description": f"Pattern where {path_pattern} institutions are affected"
                })
        
        return sorted(patterns, key=lambda x: x["prevalence"], reverse=True)
    
    def assess_sequence_stability(self) -> Dict[str, Any]:
        """Assess the stability of propagation sequences."""
        stability_assessment: Dict[str, Any] = {
            "overall_stability": "unknown",
            "vulnerable_sequences": [],
            "stable_sequences": [],
            "stability_factors": []
        }
        
        # Analyze critical sequences for stability
        critical_sequences = self.identify_critical_sequences()
        
        for seq in critical_sequences:
            institution_id = seq["starting_institution"]
            
            # Check if starting institution is in leverage points (more vulnerable)
            if institution_id in self.leverage_points:
                stability_assessment["vulnerable_sequences"].append({
                    "institution": institution_id,
                    "reason": "High leverage point - changes here affect many others",
                    "impact_potential": seq["criticality_score"]
                })
            else:
                stability_assessment["stable_sequences"].append({
                    "institution": institution_id,
                    "stability_factor": seq["criticality_score"]
                })
        
        # Overall stability assessment
        vulnerable_count = len(stability_assessment["vulnerable_sequences"])
        total_critical = len(critical_sequences)
        
        if total_critical > 0:
            stability_ratio = 1.0 - (vulnerable_count / total_critical)
            if stability_ratio > 0.7:
                stability_assessment["overall_stability"] = "high"
            elif stability_ratio > 0.4:
                stability_assessment["overall_stability"] = "moderate"
            else:
                stability_assessment["overall_stability"] = "low"
        
        return stability_assessment
    
    def recommend_sequence_interventions(self) -> List[Dict[str, Any]]:
        """Recommend interventions to improve sequence stability."""
        recommendations: List[Dict[str, Any]] = []
        
        stability_assessment = self.assess_sequence_stability()
        
        # Recommendations for vulnerable sequences
        for vulnerable_seq in stability_assessment["vulnerable_sequences"]:
            institution_id = vulnerable_seq["institution"]
            
            recommendations.append({
                "type": "risk_mitigation",
                "target_institution": institution_id,
                "intervention": "Create redundant pathways to reduce dependency",
                "priority": "high" if vulnerable_seq["impact_potential"] > 3.0 else "medium",
                "rationale": f"Institution is high-leverage point affecting many others"
            })
        
        # Recommendations for bottlenecks
        for bottleneck_sequence in self.bottleneck_sequences:
            if bottleneck_sequence:  # Check if sequence is not empty
                bottleneck_institution = bottleneck_sequence[len(bottleneck_sequence) // 2]  # Middle of sequence
                
                recommendations.append({
                    "type": "bottleneck_resolution",
                    "target_institution": bottleneck_institution,
                    "intervention": "Strengthen capacity or create alternative pathways",
                    "priority": "medium",
                    "rationale": "Institution is a bottleneck in critical sequences"
                })
        
        return sorted(recommendations, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)


@dataclass
class ValueJudgment(Node):
    """Explicit value judgments in SFM policy analysis."""
    
    judgment_type: ValueJudgmentType = ValueJudgmentType.EFFICIENCY
    value_categories_affected: List[ValueCategory] = field(default_factory=lambda: [])
    trade_offs: Dict[str, float] = field(default_factory=lambda: {})  # What's traded for what
    stakeholder_impacts: Dict[str, float] = field(default_factory=lambda: {})  # Impact on different groups
    ethical_framework: Optional[str] = None  # Underlying ethical approach
    justification: Optional[str] = None  # Rationale for the judgment
    controversy_level: Optional[float] = None  # How contested this judgment is (0-1)
    alternative_judgments: List[str] = field(default_factory=lambda: [])  # Other possible judgments
    evidence_basis: List[str] = field(default_factory=lambda: [])
    decision_context: Optional[str] = None


@dataclass
class ProblemSolvingSequence(Node):
    """Represents Hayden's structured problem-solving approach."""
    
    problem_definition: str = ""  # Added default to fix dataclass ordering
    current_stage: ProblemSolvingStage = ProblemSolvingStage.IDENTIFICATION
    status_quo_analysis: Optional[str] = None
    alternative_solutions: List[uuid.UUID] = field(default_factory=lambda: [])
    evaluation_criteria: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to SFMCriteria
    stakeholder_analysis: Dict[str, Any] = field(default_factory=lambda: {})
    implementation_barriers: List[str] = field(default_factory=lambda: [])
    selected_solution: Optional[uuid.UUID] = None
    implementation_plan: Optional[str] = None
    evaluation_results: Dict[str, Any] = field(default_factory=lambda: {})
    problem_urgency: Optional[float] = None  # How urgent is the problem (0-1)
    resource_requirements: Dict[str, float] = field(default_factory=lambda: {})  # Required resources
    timeline: Optional[str] = None  # Expected timeline for solution
    
    def __post_init__(self) -> None:
        """Validate that problem definition is provided."""
        if not self.problem_definition.strip():
            raise ValueError("problem_definition is required for ProblemSolvingSequence")


@dataclass
class CrossImpactAnalysis(Node):
    """Analyzes how changes in matrix cells affect other cells."""
    
    primary_cell_id: Optional[uuid.UUID] = None  # The cell being changed - made optional for dataclass ordering
    impacted_cells: Dict[str, float] = field(default_factory=lambda: {})  # Cell ID string -> impact strength
    impact_type: CrossImpactType = CrossImpactType.DIRECT
    impact_mechanism: Optional[str] = None  # How the impact occurs
    time_delay: Optional[float] = None  # Lag time for impact
    confidence_level: Optional[float] = None  # Confidence in impact assessment
    feedback_loops: List[uuid.UUID] = field(default_factory=lambda: [])  # Related feedback loops
    institutional_mediators: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions that mediate impact
    mitigation_strategies: List[str] = field(default_factory=lambda: [])  # Ways to reduce negative impacts
    amplification_strategies: List[str] = field(default_factory=lambda: [])  # Ways to enhance positive impacts
    
    def __post_init__(self) -> None:
        """Validate that primary cell is provided."""
        if self.primary_cell_id is None:
            raise ValueError("primary_cell_id is required for CrossImpactAnalysis")


@dataclass
class SFMMatrix(Node):
    """Represents a complete Social Fabric Matrix configuration."""
    
    institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Row headers
    criteria: List[uuid.UUID] = field(default_factory=lambda: [])  # Column headers
    matrix_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # All cells in matrix
    matrix_purpose: str = "General institutional analysis"
    analysis_context: Optional[uuid.UUID] = None  # Link to AnalyticalContext
    completeness_score: Optional[float] = None  # How complete is the matrix (0-1)
    consistency_score: Optional[float] = None  # Internal consistency (0-1)
    last_validation: Optional[datetime] = None
    validation_results: Dict[str, Any] = field(default_factory=lambda: {})
    digraph_analysis: Optional[uuid.UUID] = None  # Link to DigraphAnalysis
    cross_impact_analyses: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to CrossImpactAnalysis
    problem_solving_sequences: List[uuid.UUID] = field(default_factory=lambda: [])  # Related problem-solving
    matrix_version: int = 1  # Version number for tracking changes
    sensitivity_analysis: Dict[str, Any] = field(default_factory=lambda: {})  # Sensitivity analysis results


@dataclass
class InstitutionalStructure(Node):
    """Represents institutional arrangements and their structural properties."""
    
    structure_type: str = "formal"  # Made string with default instead of enum to fix ordering
    governance_mechanism: GovernanceMechanism = GovernanceMechanism.HIERARCHICAL
    decision_making_process: DecisionMakingType = DecisionMakingType.AUTOCRATIC
    enforcement_mechanism: EnforcementType = EnforcementType.LEGAL
    scope: InstitutionalScope = InstitutionalScope.LOCAL
    legitimacy_source: Optional[str] = None  # What gives this institution legitimacy
    power_distribution: Dict[str, float] = field(default_factory=lambda: {})  # How power is distributed
    accountability_mechanisms: List[str] = field(default_factory=lambda: [])  # How institution is held accountable
    change_mechanisms: List[str] = field(default_factory=lambda: [])  # How institution can change
    formal_rules: List[str] = field(default_factory=lambda: [])  # Written rules
    informal_norms: List[str] = field(default_factory=lambda: [])  # Unwritten conventions
    sanctions: List[str] = field(default_factory=lambda: [])  # Penalties for non-compliance
    institutional_memory: Optional[str] = None  # How knowledge is preserved


@dataclass
class TransactionCost(Node):
    """Analysis of costs associated with institutional transactions."""
    
    cost_type: TransactionCostType = TransactionCostType.SEARCH_INFORMATION
    cost_amount: Optional[float] = None  # Monetary cost if quantifiable
    time_cost: Optional[float] = None    # Time required
    uncertainty_cost: Optional[float] = None  # Risk/uncertainty costs
    bargaining_cost: Optional[float] = None   # Negotiation costs
    enforcement_cost: Optional[float] = None  # Monitoring/compliance costs
    affected_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    reduction_strategies: List[str] = field(default_factory=lambda: [])
    measurement_approach: str = "qualitative"
    cost_drivers: List[str] = field(default_factory=lambda: [])  # What causes these costs
    cost_beneficiaries: List[uuid.UUID] = field(default_factory=lambda: [])  # Who benefits from these costs


@dataclass
class CoordinationMechanism(Node):
    """Represents mechanisms for coordinating economic activity."""
    
    mechanism_type: CoordinationMechanismType = CoordinationMechanismType.PRICE_SYSTEM
    coordination_scope: CoordinationScope = CoordinationScope.LOCAL
    effectiveness_measure: Optional[float] = None  # How well it coordinates (0-1)
    participating_actors: List[uuid.UUID] = field(default_factory=lambda: [])
    coordinated_activities: List[uuid.UUID] = field(default_factory=lambda: [])
    information_requirements: List[str] = field(default_factory=lambda: [])
    failure_modes: List[str] = field(default_factory=lambda: [])
    backup_mechanisms: List[uuid.UUID] = field(default_factory=lambda: [])
    coordination_costs: Optional[float] = None  # Cost of coordination
    response_time: Optional[float] = None  # How quickly mechanism responds
    adaptability: Optional[float] = None  # Ability to adapt to change (0-1)


@dataclass
class SocialValueAssessment(Node):
    """Assessment of social value using Hayden's framework."""
    
    assessed_entity_id: Optional[uuid.UUID] = None  # What is being assessed - made optional for dataclass ordering
    life_process_impact: Optional[float] = None      # Impact on life processes
    community_continuity: Optional[float] = None     # Effect on community
    environmental_integration: Optional[float] = None # Environmental harmony
    cultural_development: Optional[float] = None     # Cultural advancement
    instrumental_efficiency: Optional[float] = None  # Problem-solving effectiveness
    
    ceremonial_elements: List[str] = field(default_factory=lambda: [])  # Status quo aspects
    instrumental_elements: List[str] = field(default_factory=lambda: [])  # Problem-solving aspects
    value_conflicts: List[str] = field(default_factory=lambda: [])     # Conflicting values
    assessment_methodology: str = "holistic"
    social_value_dimensions: List[SocialValueDimension] = field(default_factory=lambda: [])
    stakeholder_perspectives: Dict[str, float] = field(default_factory=lambda: {})  # Different stakeholder views
    
    def __post_init__(self) -> None:
        """Validate that assessed entity is provided."""
        if self.assessed_entity_id is None:
            raise ValueError("assessed_entity_id is required for SocialValueAssessment")


@dataclass
class PathDependencyAnalysis(Node):
    """Analysis of path-dependent institutional development."""
    
    analyzed_institution_id: Optional[uuid.UUID] = None  # Made optional for dataclass ordering
    dependency_strength: PathDependencyType = PathDependencyType.MODERATE
    critical_junctures: List[str] = field(default_factory=lambda: [])  # Key historical moments
    lock_in_mechanisms: List[str] = field(default_factory=lambda: [])  # What creates lock-in
    switching_costs: Dict[str, float] = field(default_factory=lambda: {})  # Costs of change
    alternative_paths: List[str] = field(default_factory=lambda: [])   # What could have been
    intervention_points: List[str] = field(default_factory=lambda: [])  # Where change is possible
    historical_trajectory: List[str] = field(default_factory=lambda: [])  # Development path
    path_efficiency: Optional[float] = None  # How efficient is current path (0-1)
    exit_barriers: List[str] = field(default_factory=lambda: [])  # Barriers to changing path
    network_effects: Optional[float] = None  # Strength of network effects (0-1)
    
    def __post_init__(self) -> None:
        """Validate that analyzed institution is provided."""
        if self.analyzed_institution_id is None:
            raise ValueError("analyzed_institution_id is required for PathDependencyAnalysis")


@dataclass
class CommonsGovernance(Node):
    """Analysis of common pool resource governance."""
    
    resource_id: Optional[uuid.UUID] = None  # The commons resource - made optional for dataclass ordering
    governance_type: CommonsGovernanceType = CommonsGovernanceType.COMMUNITY_MANAGED
    design_principles: List[str] = field(default_factory=lambda: [])  # Ostrom's principles
    user_community: List[uuid.UUID] = field(default_factory=lambda: [])  # Resource users
    monitoring_mechanisms: List[str] = field(default_factory=lambda: [])
    conflict_resolution: List[str] = field(default_factory=lambda: [])
    sustainability_indicators: List[uuid.UUID] = field(default_factory=lambda: [])
    threat_level: Optional[float] = None  # Threat to commons (0-1)
    governance_effectiveness: Optional[float] = None  # How well governance works (0-1)
    resource_condition: Optional[float] = None  # Current state of resource (0-1)
    collective_action_capacity: Optional[float] = None  # Community's ability to act collectively (0-1)
    external_pressures: List[str] = field(default_factory=lambda: [])  # External threats
    
    def __post_init__(self) -> None:
        """Validate that resource is provided."""
        if self.resource_id is None:
            raise ValueError("resource_id is required for CommonsGovernance")


@dataclass
class SystemLevelAnalysis(Node):
    """Comprehensive system-level analysis of SFM."""
    
    analyzed_system_boundary: str = "undefined"
    institutions_analyzed: List[uuid.UUID] = field(default_factory=lambda: [])
    actors_analyzed: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # System properties
    system_coherence: Optional[float] = None      # How well-integrated (0-1)
    system_resilience: Optional[float] = None     # Ability to handle shocks (0-1)  
    system_adaptability: Optional[float] = None   # Capacity for change (0-1)
    system_efficiency: Optional[float] = None     # Resource utilization (0-1)
    system_sustainability: Optional[float] = None # Long-term viability (0-1)
    
    # Bottlenecks and leverage points
    system_bottlenecks: List[uuid.UUID] = field(default_factory=lambda: [])
    leverage_points: List[uuid.UUID] = field(default_factory=lambda: [])
    critical_dependencies: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # System dynamics
    dominant_feedback_loops: List[uuid.UUID] = field(default_factory=lambda: [])
    system_archetypes: List[SystemArchetype] = field(default_factory=lambda: [])
    change_capacity: Optional[float] = None
    emergence_patterns: List[str] = field(default_factory=lambda: [])  # Emergent behaviors
    system_learning_rate: Optional[float] = None  # How fast system learns (0-1)
    
    def __post_init__(self) -> None:
        """Validate that system boundary is defined."""
        if not self.analyzed_system_boundary or self.analyzed_system_boundary == "undefined":
            raise ValueError("analyzed_system_boundary must be defined for SystemLevelAnalysis")


@dataclass
class ValueSystem(Node):
    """Represents cultural value systems in Hayden's framework."""
    
    system_type: ValueSystemType = ValueSystemType.CULTURAL_DOMINANT
    core_values: List[str] = field(default_factory=lambda: [])  # Primary values in this system
    value_hierarchy: Dict[str, float] = field(default_factory=lambda: {})  # Value priorities (0-1)
    cultural_embedding: Optional[float] = None  # How deeply embedded (0-1)
    transmission_mechanisms: List[str] = field(default_factory=lambda: [])  # How values are transmitted
    
    # SFM-specific properties
    ceremonial_elements: List[str] = field(default_factory=lambda: [])  # Status quo reinforcing aspects
    instrumental_elements: List[str] = field(default_factory=lambda: [])  # Problem-solving aspects
    value_conflicts: List[uuid.UUID] = field(default_factory=lambda: [])  # Conflicting value systems
    institutional_support: List[uuid.UUID] = field(default_factory=lambda: [])  # Supporting institutions
    change_resistance: Optional[float] = None  # Resistance to value change (0-1)
    adaptive_capacity: Optional[float] = None  # Capacity for value evolution (0-1)
    
    # Integration with matrix
    influenced_matrix_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # Matrix cells influenced
    legitimacy_source: Optional[str] = None  # What legitimizes this value system
    stakeholder_alignment: Dict[str, float] = field(default_factory=lambda: {})  # Stakeholder agreement
    
    def calculate_coherence_score(self) -> float:
        """Calculate internal coherence of value system."""
        if not self.value_hierarchy:
            return 0.0
        
        # More balanced hierarchy = higher coherence
        values = list(self.value_hierarchy.values())
        if not values:
            return 0.0
            
        # Calculate variance - lower variance = higher coherence
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return max(0.0, 1.0 - variance)  # Inverse relationship
    
    def assess_institutional_alignment(self) -> float:
        """Assess alignment between values and supporting institutions."""
        if not self.institutional_support:
            return 0.0
        
        # More institutional support = better alignment
        support_count = len(self.institutional_support)
        return min(1.0, support_count * 0.2)  # Cap at 1.0


@dataclass
class InstitutionalHolarchy(Node):
    """Represents nested levels of institutional arrangements per Hayden's framework."""
    
    institutional_levels: Dict[InstitutionalLevel, List[uuid.UUID]] = field(default_factory=lambda: {})
    level_interactions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Inter-level relationships
    authority_flows: Dict[str, List[uuid.UUID]] = field(default_factory=lambda: {})  # How authority flows
    
    # Holarchy properties
    emergence_patterns: List[str] = field(default_factory=lambda: [])  # How higher levels emerge
    constraint_flows: Dict[str, List[str]] = field(default_factory=lambda: {})  # Top-down constraints
    innovation_sources: Dict[InstitutionalLevel, List[str]] = field(default_factory=lambda: {})  # Where innovation occurs
    
    # SFM integration
    matrix_cell_mapping: Dict[InstitutionalLevel, List[uuid.UUID]] = field(default_factory=lambda: {})  # Cells by level
    cross_level_dependencies: List[uuid.UUID] = field(default_factory=lambda: [])  # Dependencies across levels
    power_concentration: Dict[InstitutionalLevel, float] = field(default_factory=lambda: {})  # Power distribution
    
    # System properties
    hierarchical_coherence: Optional[float] = None  # How well levels work together (0-1)
    adaptive_capacity_by_level: Dict[InstitutionalLevel, float] = field(default_factory=lambda: {})
    bottleneck_levels: List[InstitutionalLevel] = field(default_factory=lambda: [])  # Constraint points
    
    def calculate_system_coherence(self) -> float:
        """Calculate overall coherence of the institutional holarchy."""
        if not self.level_interactions:
            return 0.0
        
        # Sum positive interactions, penalize negative ones
        total_interactions = 0
        positive_interactions = 0
        
        for level_interactions in self.level_interactions.values():
            for interaction_strength in level_interactions.values():
                total_interactions += 1
                if interaction_strength > 0:
                    positive_interactions += interaction_strength
        
        return positive_interactions / total_interactions if total_interactions > 0 else 0.0
    
    def identify_leverage_points(self) -> List[InstitutionalLevel]:
        """Identify levels with highest leverage for system change."""
        leverage_scores: Dict[InstitutionalLevel, float] = {}
        
        for level in InstitutionalLevel:
            # Count connections and power concentration
            connections = len(self.institutional_levels.get(level, []))
            power = self.power_concentration.get(level, 0.0)
            leverage_scores[level] = connections * power
        
        # Return top 3 leverage points
        sorted_levels = sorted(leverage_scores.items(), key=lambda x: x[1], reverse=True)
        return [level for level, _ in sorted_levels[:3]]


@dataclass
class SocialFabricIndicator(Indicator):
    """Specialized indicator for measuring social fabric health per Hayden's framework."""
    
    indicator_type: SocialFabricIndicatorType = SocialFabricIndicatorType.INSTITUTIONAL_COHERENCE
    baseline_value: Optional[float] = None  # Historical baseline for comparison
    target_threshold: Optional[float] = None  # Desired threshold value
    warning_threshold: Optional[float] = None  # Warning level threshold
    
    # SFM-specific properties
    affected_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions measured
    measurement_methodology: str = "qualitative_assessment"
    stakeholder_perspectives: Dict[str, float] = field(default_factory=lambda: {})  # Different viewpoints
    
    # Integration with matrix
    related_matrix_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # Relevant matrix cells
    normative_framework: Optional[NormativeFramework] = None  # Evaluation framework
    ceremonial_biases: List[str] = field(default_factory=lambda: [])  # Potential ceremonial distortions
    
    # Trend analysis
    trend_direction: Optional[str] = None  # "improving", "declining", "stable"
    trend_strength: Optional[float] = None  # Strength of trend (0-1)
    intervention_responsiveness: Optional[float] = None  # How responsive to interventions (0-1)
    
    def assess_fabric_health(self) -> Dict[str, Any]:
        """Comprehensive assessment of social fabric health."""
        assessment: Dict[str, Any] = {
            "overall_health": "unknown",
            "trend": self.trend_direction or "unknown",
            "stakeholder_consensus": 0.0,
            "intervention_urgency": 0.0
        }
        
        # Determine overall health
        if self.current_value is not None:
            if self.warning_threshold and self.current_value < self.warning_threshold:
                assessment["overall_health"] = "poor"
                assessment["intervention_urgency"] = 0.8
            elif self.target_threshold and self.current_value >= self.target_threshold:
                assessment["overall_health"] = "good"
                assessment["intervention_urgency"] = 0.2
            else:
                assessment["overall_health"] = "moderate"
                assessment["intervention_urgency"] = 0.5
        
        # Calculate stakeholder consensus
        if self.stakeholder_perspectives:
            values = list(self.stakeholder_perspectives.values())
            if values:
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                assessment["stakeholder_consensus"] = max(0.0, 1.0 - variance)
        
        return assessment


@dataclass
class SocialCost(Node):
    """Represents social costs per Kapp's theory integrated with Hayden's SFM."""
    
    cost_type: SocialCostType = SocialCostType.ENVIRONMENTAL_DEGRADATION
    estimated_cost: Optional[float] = None  # Monetary estimate if possible
    cost_unit: Optional[str] = None  # Unit of measurement
    
    # Cost characteristics
    cost_bearers: List[uuid.UUID] = field(default_factory=lambda: [])  # Who bears the cost
    cost_creators: List[uuid.UUID] = field(default_factory=lambda: [])  # Who creates the cost
    externalization_mechanism: Optional[str] = None  # How cost is externalized
    
    # Temporal aspects
    cost_trajectory: Optional[str] = None  # "increasing", "decreasing", "stable"
    time_lag: Optional[float] = None  # Delay between cause and cost manifestation
    irreversibility: Optional[float] = None  # How reversible the cost is (0-1)
    
    # SFM integration
    institutional_sources: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions creating cost
    matrix_cell_impacts: List[uuid.UUID] = field(default_factory=lambda: [])  # Affected matrix cells
    ceremonial_enablers: List[str] = field(default_factory=lambda: [])  # Ceremonial factors enabling cost
    
    # Measurement and evidence
    measurement_challenges: List[str] = field(default_factory=lambda: [])  # Difficulties in measurement
    evidence_sources: List[str] = field(default_factory=lambda: [])  # Evidence for cost existence
    uncertainty_level: Optional[float] = None  # Uncertainty in cost estimate (0-1)
    
    # Policy implications
    internalization_mechanisms: List[str] = field(default_factory=lambda: [])  # Ways to internalize cost
    prevention_strategies: List[str] = field(default_factory=lambda: [])  # Prevention approaches
    compensation_mechanisms: List[str] = field(default_factory=lambda: [])  # Compensation approaches
    
    def calculate_social_burden(self) -> Dict[str, Any]:
        """Calculate overall social burden of this cost."""
        burden: Dict[str, Any] = {
            "severity": "unknown",
            "distribution_equity": 0.0,
            "prevention_potential": 0.0,
            "policy_urgency": 0.0
        }
        
        # Assess severity
        if self.estimated_cost and self.irreversibility:
            severity_score = self.estimated_cost * self.irreversibility
            if severity_score > 1000000:  # Arbitrary threshold
                burden["severity"] = "high"
                burden["policy_urgency"] = 0.9
            elif severity_score > 100000:
                burden["severity"] = "moderate"
                burden["policy_urgency"] = 0.6
            else:
                burden["severity"] = "low"
                burden["policy_urgency"] = 0.3
        
        # Assess distribution equity (more bearers vs creators = less equitable)
        bearers_count = len(self.cost_bearers)
        creators_count = len(self.cost_creators)
        if creators_count > 0:
            burden["distribution_equity"] = min(1.0, creators_count / bearers_count)
        
        # Assess prevention potential
        prevention_count = len(self.prevention_strategies)
        burden["prevention_potential"] = min(1.0, prevention_count * 0.2)
        
        return burden


@dataclass
class CellDependencyNetwork(Node):
    """Systematic mapping of interdependencies between matrix cells."""
    
    dependency_matrix: Dict[str, Dict[str, DependencyStrength]] = field(default_factory=lambda: {})
    network_properties: Dict[str, float] = field(default_factory=lambda: {})  # Network metrics
    
    # Network analysis
    critical_paths: List[List[uuid.UUID]] = field(default_factory=lambda: [])  # Critical dependency paths
    bottleneck_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # High-dependency cells
    isolated_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # Low-dependency cells
    
    # Dynamic properties
    dependency_changes: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # How dependencies change
    cascade_potential: Dict[str, float] = field(default_factory=lambda: {})  # Potential for cascading effects
    resilience_factors: Dict[str, List[str]] = field(default_factory=lambda: {})  # What makes cells resilient
    
    # SFM integration
    institutional_mediators: Dict[str, List[uuid.UUID]] = field(default_factory=lambda: {})  # Institutions mediating dependencies
    value_system_influences: Dict[str, List[uuid.UUID]] = field(default_factory=lambda: {})  # Value systems affecting dependencies
    temporal_evolution: Dict[str, EvolutionaryStage] = field(default_factory=lambda: {})  # How dependencies evolve
    
    def calculate_network_resilience(self) -> float:
        """Calculate overall resilience of the dependency network."""
        if not self.dependency_matrix:
            return 0.0
        
        # Count critical vs non-critical dependencies
        total_deps = 0
        critical_deps = 0
        
        for cell_deps in self.dependency_matrix.values():
            for strength in cell_deps.values():
                total_deps += 1
                if strength == DependencyStrength.CRITICAL:
                    critical_deps += 1
        
        # Lower ratio of critical dependencies = higher resilience
        if total_deps == 0:
            return 0.0
        
        critical_ratio = critical_deps / total_deps
        return max(0.0, 1.0 - critical_ratio)
    
    def identify_intervention_points(self) -> List[uuid.UUID]:
        """Identify cells where interventions would have maximum system impact."""
        impact_scores: Dict[uuid.UUID, float] = {}
        
        # Score cells based on number and strength of dependencies
        for cell_id, dependencies in self.dependency_matrix.items():
            score = 0
            for strength in dependencies.values():
                if strength == DependencyStrength.CRITICAL:
                    score += 3
                elif strength == DependencyStrength.STRONG:
                    score += 2
                elif strength == DependencyStrength.MODERATE:
                    score += 1
            
            try:
                impact_scores[uuid.UUID(cell_id)] = score
            except ValueError:
                continue  # Skip invalid UUIDs
        
        # Return top intervention points
        sorted_cells = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)
        return [cell_id for cell_id, _ in sorted_cells[:5]]


@dataclass
class InstitutionalEvolution(Node):
    """Enhanced tracking of institutional development over time."""
    
    institution_id: Optional[uuid.UUID] = None  # Institution being tracked
    evolutionary_stage: EvolutionaryStage = EvolutionaryStage.ESTABLISHMENT
    
    # Historical trajectory
    development_history: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Historical stages
    critical_junctures: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Key decision points
    path_dependencies: List[str] = field(default_factory=lambda: [])  # Path-dependent elements
    
    # Change dynamics
    change_drivers: List[str] = field(default_factory=lambda: [])  # What drives change
    change_resistances: List[str] = field(default_factory=lambda: [])  # What resists change
    adaptation_mechanisms: List[str] = field(default_factory=lambda: [])  # How institution adapts
    
    # SFM integration
    matrix_cell_evolution: Dict[str, List[CorrelationType]] = field(default_factory=lambda: {})  # How cell relationships evolve
    value_system_shifts: List[uuid.UUID] = field(default_factory=lambda: [])  # Associated value changes
    ceremonial_drift: Optional[float] = None  # Tendency toward ceremonial behavior (0-1)
    instrumental_capacity: Optional[float] = None  # Problem-solving capacity over time (0-1)
    
    # Predictive elements
    evolution_trajectory: Optional[str] = None  # Predicted future development
    intervention_opportunities: List[str] = field(default_factory=lambda: [])  # Windows for change
    lock_in_risks: List[str] = field(default_factory=lambda: [])  # Risks of getting locked in
    
    def __post_init__(self) -> None:
        """Validate that institution is provided."""
        if self.institution_id is None:
            raise ValueError("institution_id is required for InstitutionalEvolution")
    
    def assess_adaptive_capacity(self) -> float:
        """Assess the institution's capacity for adaptive change."""
        if self.instrumental_capacity is None:
            return 0.0
        
        # Consider change mechanisms and resistances
        change_factors = len(self.adaptation_mechanisms) - len(self.change_resistances)
        change_score = max(0.0, min(1.0, change_factors * 0.2))
        
        # Combine with instrumental capacity
        return (self.instrumental_capacity + change_score) / 2
    
    def predict_evolution_direction(self) -> str:
        """Predict likely direction of institutional evolution."""
        if self.ceremonial_drift and self.instrumental_capacity:
            if self.ceremonial_drift > self.instrumental_capacity:
                return "ceremonial_dominance"
            elif self.instrumental_capacity > self.ceremonial_drift:
                return "instrumental_enhancement"
            else:
                return "balanced_development"
        
        return "uncertain"


@dataclass
class NormativeInstitutionalFramework(Node):
    """Framework for evaluating institutional quality per Hayden's normative approach."""
    
    framework_type: NormativeFramework = NormativeFramework.LIFE_PROCESS_ENHANCEMENT
    evaluation_criteria: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to SFMCriteria
    
    # Normative standards
    life_process_standards: Dict[str, float] = field(default_factory=lambda: {})  # Life process criteria
    community_standards: Dict[str, float] = field(default_factory=lambda: {})  # Community continuity criteria
    environmental_standards: Dict[str, float] = field(default_factory=lambda: {})  # Environmental criteria
    democratic_standards: Dict[str, float] = field(default_factory=lambda: {})  # Democratic participation criteria
    
    # Evaluation methodology
    assessment_approach: str = "holistic_integration"
    weighting_scheme: Dict[str, float] = field(default_factory=lambda: {})  # Criteria weights
    threshold_values: Dict[str, float] = field(default_factory=lambda: {})  # Acceptable thresholds
    
    # SFM integration
    matrix_cell_evaluations: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Cell-by-cell evaluations
    institutional_rankings: Dict[str, float] = field(default_factory=lambda: {})  # Institution quality scores
    system_level_assessment: Optional[float] = None  # Overall system quality (0-1)
    
    # Policy implications
    improvement_recommendations: List[str] = field(default_factory=lambda: [])  # How to improve
    priority_interventions: List[str] = field(default_factory=lambda: [])  # High-priority changes
    resource_requirements: Dict[str, float] = field(default_factory=lambda: {})  # Resources needed
    
    def evaluate_institution(self, _institution_id: uuid.UUID, _matrix_cells: List[uuid.UUID]) -> Dict[str, Any]:
        """Evaluate an institution using the normative framework."""
        evaluation: Dict[str, Any] = {
            "overall_score": 0.0,
            "dimension_scores": {},
            "recommendations": [],
            "priority_level": "low"
        }
        
        # Calculate dimension scores
        dimensions = ["life_process", "community", "environmental", "democratic"]
        total_score = 0.0
        scored_dimensions = 0
        
        for dimension in dimensions:
            standards = getattr(self, f"{dimension}_standards", {})
            if standards:
                # Simplified scoring based on standards
                dimension_score = sum(standards.values()) / len(standards)
                evaluation["dimension_scores"][dimension] = dimension_score
                total_score += dimension_score
                scored_dimensions += 1
        
        if scored_dimensions > 0:
            evaluation["overall_score"] = total_score / scored_dimensions
        
        # Determine priority level
        if evaluation["overall_score"] < 0.3:
            evaluation["priority_level"] = "high"
            evaluation["recommendations"].append("Immediate intervention required")
        elif evaluation["overall_score"] < 0.6:
            evaluation["priority_level"] = "medium"
            evaluation["recommendations"].append("Systematic improvement needed")
        else:
            evaluation["priority_level"] = "low"
            evaluation["recommendations"].append("Maintain current performance")
        
        return evaluation
    
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system-level normative assessment."""
        report: Dict[str, Any] = {
            "system_health": "unknown",
            "key_strengths": [],
            "critical_weaknesses": [],
            "priority_interventions": self.priority_interventions.copy(),
            "resource_needs": self.resource_requirements.copy()
        }
        
        if self.system_level_assessment:
            if self.system_level_assessment >= 0.7:
                report["system_health"] = "good"
            elif self.system_level_assessment >= 0.4:
                report["system_health"] = "moderate"
            else:
                report["system_health"] = "poor"
        
        # Identify strengths and weaknesses from institutional rankings
        if self.institutional_rankings:
            sorted_institutions = sorted(self.institutional_rankings.items(), key=lambda x: x[1], reverse=True)
            
            # Top performers are strengths
            top_performers = [inst for inst, score in sorted_institutions[:3] if score >= 0.6]
            report["key_strengths"] = top_performers
            
            # Bottom performers are weaknesses
            bottom_performers = [inst for inst, score in sorted_institutions[-3:] if score < 0.4]
            report["critical_weaknesses"] = bottom_performers
        
        return report


@dataclass
class SystemBoundary(Node):
    """Defines the boundaries and scope of SFM analysis per Hayden's methodology."""
    
    boundary_types: List[BoundaryType] = field(default_factory=lambda: [])
    
    # Geographic boundaries
    geographic_scope: Optional[str] = None  # Geographic area covered
    spatial_units: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to SpatialUnit
    
    # Institutional boundaries
    included_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Included institutions
    excluded_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Explicitly excluded
    institutional_criteria: List[str] = field(default_factory=lambda: [])  # Inclusion criteria
    
    # Temporal boundaries
    time_period: Optional[str] = None  # Time period of analysis
    temporal_focus: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to TimeSlice
    
    # Functional boundaries
    functional_domains: List[str] = field(default_factory=lambda: [])  # Functional areas included
    sectoral_scope: List[str] = field(default_factory=lambda: [])  # Economic sectors
    
    # Analytical boundaries
    analytical_purpose: str = "institutional_analysis"  # Purpose of analysis
    theoretical_framework: List[str] = field(default_factory=lambda: [])  # Theoretical approaches used
    methodology_constraints: List[str] = field(default_factory=lambda: [])  # Methodological limits
    
    # Boundary justification
    inclusion_rationale: Dict[str, str] = field(default_factory=lambda: {})  # Why elements are included
    exclusion_rationale: Dict[str, str] = field(default_factory=lambda: {})  # Why elements are excluded
    boundary_sensitivity: Optional[float] = None  # How sensitive results are to boundary choices (0-1)
    
    # Integration with matrix
    affected_matrix_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # Matrix cells within boundary
    boundary_spanning_relationships: List[uuid.UUID] = field(default_factory=lambda: [])  # Cross-boundary relationships
    external_influences: Dict[str, float] = field(default_factory=lambda: {})  # External factors and influence strength
    
    def assess_boundary_completeness(self) -> Dict[str, Any]:
        """Assess how complete and appropriate the boundary definition is."""
        assessment: Dict[str, Any] = {
            "completeness": "partial",
            "consistency": "consistent",  
            "sensitivity": "low",
            "justification_quality": "adequate"
        }
        
        # Check completeness
        defined_boundaries = len([bt for bt in self.boundary_types if bt])
        if defined_boundaries >= 4:
            assessment["completeness"] = "comprehensive"
        elif defined_boundaries >= 2:
            assessment["completeness"] = "adequate"
        
        # Check consistency
        if self.included_institutions and self.excluded_institutions:
            # Check for overlaps
            included_set = set(self.included_institutions)
            excluded_set = set(self.excluded_institutions)
            if included_set.intersection(excluded_set):
                assessment["consistency"] = "inconsistent"
        
        # Check sensitivity
        if self.boundary_sensitivity:
            if self.boundary_sensitivity > 0.7:
                assessment["sensitivity"] = "high"
            elif self.boundary_sensitivity > 0.3:
                assessment["sensitivity"] = "moderate"
        
        # Check justification quality
        total_rationales = len(self.inclusion_rationale) + len(self.exclusion_rationale)
        total_elements = len(self.included_institutions) + len(self.excluded_institutions)
        if total_elements > 0:
            rationale_ratio = total_rationales / total_elements
            if rationale_ratio >= 0.8:
                assessment["justification_quality"] = "strong"
            elif rationale_ratio < 0.3:
                assessment["justification_quality"] = "weak"
        
        return assessment
    
    def identify_boundary_risks(self) -> List[str]:
        """Identify potential risks from boundary choices."""
        risks: List[str] = []
        
        if not self.boundary_types:
            risks.append("No explicit boundary types defined")
        
        if self.external_influences:
            high_influence_externals = [k for k, v in self.external_influences.items() if v > 0.5]
            if high_influence_externals:
                risks.append(f"High external influences: {', '.join(high_influence_externals)}")
        
        if not self.inclusion_rationale and self.included_institutions:
            risks.append("Included institutions lack explicit rationale")
        
        if self.boundary_sensitivity and self.boundary_sensitivity > 0.6:
            risks.append("Results highly sensitive to boundary choices")
        
        return risks


@dataclass
class ProvisioningProcess(Node):
    """Models the societal provisioning process in Hayden's SFM framework."""
    
    provisioning_stages: Dict[ProvisioningStage, List[uuid.UUID]] = field(default_factory=lambda: {})  # Stages and involved institutions
    
    # Process characteristics
    process_efficiency: Optional[float] = None  # Overall efficiency (0-1)
    environmental_sustainability: Optional[float] = None  # Environmental sustainability (0-1)
    social_equity: Optional[float] = None  # Equity in distribution (0-1)
    institutional_coherence: Optional[float] = None  # How well institutions work together (0-1)
    
    # Resource flows
    resource_inputs: List[uuid.UUID] = field(default_factory=lambda: [])  # Input resources
    resource_outputs: List[uuid.UUID] = field(default_factory=lambda: [])  # Output products/services
    waste_outputs: List[uuid.UUID] = field(default_factory=lambda: [])  # Waste and byproducts
    
    # Institutional integration
    coordinating_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Coordinating institutions
    supporting_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Supporting institutions
    regulating_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Regulatory institutions
    
    # SFM integration
    matrix_cell_dependencies: List[uuid.UUID] = field(default_factory=lambda: [])  # Dependent matrix cells
    value_system_alignment: Dict[str, float] = field(default_factory=lambda: {})  # Alignment with value systems
    ceremonial_constraints: List[str] = field(default_factory=lambda: [])  # Ceremonial barriers
    instrumental_enablers: List[str] = field(default_factory=lambda: [])  # Instrumental facilitators
    
    # Performance measures
    provision_adequacy: Optional[float] = None  # How well needs are met (0-1)
    access_equity: Optional[float] = None  # Equity of access (0-1)
    long_term_viability: Optional[float] = None  # Long-term sustainability (0-1)
    adaptive_capacity: Optional[float] = None  # Ability to adapt to change (0-1)
    
    # Problem identification
    bottlenecks: List[str] = field(default_factory=lambda: [])  # Process bottlenecks
    inefficiencies: List[str] = field(default_factory=lambda: [])  # Identified inefficiencies
    social_costs: List[uuid.UUID] = field(default_factory=lambda: [])  # Associated social costs
    improvement_opportunities: List[str] = field(default_factory=lambda: [])  # Improvement possibilities
    
    def assess_provisioning_performance(self) -> Dict[str, Any]:
        """Comprehensive assessment of provisioning process performance."""
        assessment: Dict[str, Any] = {
            "overall_performance": "unknown",
            "key_strengths": [],
            "critical_weaknesses": [],
            "priority_improvements": [],
            "sustainability_outlook": "uncertain"
        }
        
        # Calculate overall performance
        performance_indicators = [
            self.process_efficiency,
            self.environmental_sustainability,
            self.social_equity,
            self.institutional_coherence,
            self.provision_adequacy,
            self.access_equity,
            self.long_term_viability
        ]
        
        valid_indicators = [p for p in performance_indicators if p is not None]
        if valid_indicators:
            avg_performance = sum(valid_indicators) / len(valid_indicators)
            if avg_performance >= 0.7:
                assessment["overall_performance"] = "good"
            elif avg_performance >= 0.5:
                assessment["overall_performance"] = "moderate"
            else:
                assessment["overall_performance"] = "poor"
        
        # Identify strengths
        if self.process_efficiency and self.process_efficiency > 0.7:
            assessment["key_strengths"].append("High process efficiency")
        if self.social_equity and self.social_equity > 0.7:
            assessment["key_strengths"].append("Good social equity")
        if self.environmental_sustainability and self.environmental_sustainability > 0.7:
            assessment["key_strengths"].append("Environmental sustainability")
        
        # Identify weaknesses
        if self.bottlenecks:
            assessment["critical_weaknesses"].extend(self.bottlenecks)
        if self.inefficiencies:
            assessment["critical_weaknesses"].extend(self.inefficiencies)
        
        # Priority improvements
        if self.improvement_opportunities:
            assessment["priority_improvements"] = self.improvement_opportunities[:3]  # Top 3
        
        # Sustainability outlook
        if self.long_term_viability:
            if self.long_term_viability > 0.6:
                assessment["sustainability_outlook"] = "positive"
            elif self.long_term_viability < 0.4:
                assessment["sustainability_outlook"] = "concerning"
        
        return assessment
    
    def identify_institutional_gaps(self) -> List[str]:
        """Identify gaps in institutional coverage of provisioning stages."""
        gaps: List[str] = []
        
        for stage in ProvisioningStage:
            if stage not in self.provisioning_stages or not self.provisioning_stages[stage]:
                gaps.append(f"No institutions identified for {stage.name.lower().replace('_', ' ')}")
        
        # Check for coordination gaps
        if not self.coordinating_institutions:
            gaps.append("No coordinating institutions identified")
        
        # Check for regulatory gaps
        if not self.regulating_institutions:
            gaps.append("No regulatory institutions identified")
        
        return gaps


@dataclass
class PolicyEvaluation(Node):
    """Integration of policy analysis with SFM matrix transformations."""
    
    evaluated_policy_id: Optional[uuid.UUID] = None  # Policy being evaluated
    
    # Matrix transformation analysis
    pre_policy_matrix: Optional[uuid.UUID] = None  # Matrix state before policy
    post_policy_matrix: Optional[uuid.UUID] = None  # Matrix state after policy
    predicted_matrix: Optional[uuid.UUID] = None  # Predicted matrix state
    
    # Cell-level changes
    affected_cells: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Cell ID -> {attribute: change}
    correlation_changes: Dict[str, CorrelationScale] = field(default_factory=lambda: {})  # Cell ID -> new correlation
    
    # Evaluation results
    effectiveness_score: Optional[float] = None  # Overall policy effectiveness (0-1)
    efficiency_score: Optional[float] = None  # Resource efficiency (0-1)
    equity_impact: Optional[float] = None  # Impact on equity (-1 to +1)
    sustainability_impact: Optional[float] = None  # Sustainability impact (-1 to +1)
    
    # Hayden's framework integration
    ceremonial_resistance: Optional[float] = None  # Ceremonial resistance to policy (0-1)
    instrumental_support: Optional[float] = None  # Instrumental support for policy (0-1)
    life_process_enhancement: Optional[float] = None  # Enhancement of life processes (-1 to +1)
    
    # Unintended consequences
    positive_spillovers: List[str] = field(default_factory=lambda: [])  # Unexpected positive effects
    negative_spillovers: List[str] = field(default_factory=lambda: [])  # Unexpected negative effects
    cascade_effects: List[uuid.UUID] = field(default_factory=lambda: [])  # Cascading matrix changes
    
    # Implementation analysis
    implementation_feasibility: Optional[float] = None  # Feasibility of implementation (0-1)
    institutional_readiness: Optional[float] = None  # Institutional capacity (0-1)
    resource_requirements: Dict[str, float] = field(default_factory=lambda: {})  # Required resources
    timeline_analysis: Dict[str, str] = field(default_factory=lambda: {})  # Implementation timeline
    
    # Stakeholder impacts
    stakeholder_effects: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Stakeholder -> effects
    distributional_analysis: Dict[str, float] = field(default_factory=lambda: {})  # Distribution of costs/benefits
    
    def __post_init__(self) -> None:
        """Validate that policy is provided."""
        if self.evaluated_policy_id is None:
            raise ValueError("evaluated_policy_id is required for PolicyEvaluation")
    
    def calculate_net_impact(self) -> Dict[str, float]:
        """Calculate net impact across key dimensions."""
        net_impact = {
            "economic": 0.0,
            "social": 0.0,
            "environmental": 0.0,
            "institutional": 0.0
        }
        
        # Aggregate cell changes by dimension
        for _, changes in self.affected_cells.items():
            for attribute, change in changes.items():
                if "economic" in attribute.lower():
                    net_impact["economic"] += change
                elif "social" in attribute.lower():
                    net_impact["social"] += change
                elif "environmental" in attribute.lower():
                    net_impact["environmental"] += change
                elif "institutional" in attribute.lower():
                    net_impact["institutional"] += change
        
        # Normalize by number of affected cells
        num_cells = len(self.affected_cells) if self.affected_cells else 1
        for dimension in net_impact:
            net_impact[dimension] /= num_cells
        
        return net_impact
    
    def assess_implementation_risks(self) -> List[str]:
        """Assess risks to successful policy implementation."""
        risks: List[str] = []
        
        if self.ceremonial_resistance and self.ceremonial_resistance > 0.6:
            risks.append("High ceremonial resistance to change")
        
        if self.institutional_readiness and self.institutional_readiness < 0.4:
            risks.append("Low institutional readiness")
        
        if self.implementation_feasibility and self.implementation_feasibility < 0.5:
            risks.append("Low implementation feasibility")
        
        if len(self.negative_spillovers) > len(self.positive_spillovers):
            risks.append("More negative than positive spillover effects expected")
        
        if not self.resource_requirements:
            risks.append("Resource requirements not clearly identified")
        
        return risks
    
    def generate_policy_recommendation(self) -> Dict[str, Any]:
        """Generate policy recommendation based on evaluation."""
        recommendation: Dict[str, Any] = {
            "overall_recommendation": "neutral",
            "confidence_level": "medium",
            "key_benefits": [],
            "key_risks": [],
            "implementation_priority": "medium",
            "required_modifications": []
        }
        
        # Overall recommendation based on multiple factors
        positive_indicators = sum([
            1 if self.effectiveness_score and self.effectiveness_score > 0.6 else 0,
            1 if self.life_process_enhancement and self.life_process_enhancement > 0.3 else 0,
            1 if self.sustainability_impact and self.sustainability_impact > 0.3 else 0,
            1 if len(self.positive_spillovers) > len(self.negative_spillovers) else 0
        ])
        
        if positive_indicators >= 3:
            recommendation["overall_recommendation"] = "strongly_support"
            recommendation["implementation_priority"] = "high"
        elif positive_indicators >= 2:
            recommendation["overall_recommendation"] = "support_with_modifications"
            recommendation["implementation_priority"] = "medium"
        elif positive_indicators <= 1:
            recommendation["overall_recommendation"] = "oppose"
            recommendation["implementation_priority"] = "low"
        
        # Add specific benefits and risks
        recommendation["key_benefits"] = self.positive_spillovers[:3]
        recommendation["key_risks"] = self.assess_implementation_risks()[:3]
        
        return recommendation


@dataclass
class SocialAccountingMatrix(Node):
    """Integration of Social Accounting Matrix with SFM for quantitative flow analysis."""
    
    # Matrix structure
    accounts: List[str] = field(default_factory=lambda: [])  # SAM accounts (sectors, institutions, factors)
    flow_matrix: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # From account -> To account -> flow value
    
    # SFM integration
    sfm_matrix_link: Optional[uuid.UUID] = None  # Link to associated SFM matrix
    institution_account_mapping: Dict[str, List[str]] = field(default_factory=lambda: {})  # Institution -> SAM accounts
    matrix_cell_flows: Dict[str, List[str]] = field(default_factory=lambda: {})  # Matrix cell -> associated SAM flows
    
    # Flow characteristics
    monetary_flows: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Monetary transactions
    real_flows: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Real resource flows
    information_flows: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Information flows
    
    # Institutional analysis
    institutional_power_flows: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Power relationships
    decision_flows: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Decision-making flows
    influence_networks: Dict[str, List[str]] = field(default_factory=lambda: {})  # Influence relationships
    
    # System properties
    flow_balance: Dict[str, float] = field(default_factory=lambda: {})  # Account balances
    multiplier_effects: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Economic multipliers
    leakages: Dict[str, float] = field(default_factory=lambda: {})  # System leakages
    
    # Dynamic analysis
    flow_changes: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=lambda: {})  # Time period -> flows
    structural_change: Dict[str, float] = field(default_factory=lambda: {})  # Structural change indicators
    stability_measures: Dict[str, float] = field(default_factory=lambda: {})  # System stability metrics
    
    def calculate_total_flows(self) -> Dict[str, Dict[str, float]]:
        """Calculate total inflows and outflows for each account."""
        total_flows: Dict[str, Dict[str, float]] = {}
        
        for from_account, to_flows in self.flow_matrix.items():
            if from_account not in total_flows:
                total_flows[from_account] = {"inflow": 0.0, "outflow": 0.0}
            
            # Calculate outflows
            total_flows[from_account]["outflow"] = sum(to_flows.values())
            
            # Calculate inflows
            for to_account, flow_value in to_flows.items():
                if to_account not in total_flows:
                    total_flows[to_account] = {"inflow": 0.0, "outflow": 0.0}
                total_flows[to_account]["inflow"] += flow_value
        
        return total_flows
    
    def identify_key_flows(self, threshold: float = 0.05) -> List[Dict[str, Any]]:
        """Identify flows above threshold percentage of total system flows."""
        total_system_flow = sum(
            sum(to_flows.values()) 
            for to_flows in self.flow_matrix.values()
        )
        
        key_flows: List[Dict[str, Any]] = []
        for from_account, to_flows in self.flow_matrix.items():
            for to_account, flow_value in to_flows.items():
                if flow_value / total_system_flow > threshold:
                    key_flows.append({
                        "from": from_account,
                        "to": to_account,
                        "value": flow_value,
                        "percentage": (flow_value / total_system_flow) * 100
                    })
        
        return sorted(key_flows, key=lambda x: x["value"], reverse=True)
    
    def analyze_institutional_influence(self) -> Dict[str, float]:
        """Analyze relative institutional influence based on flow patterns."""
        influence_scores: Dict[str, float] = {}
        
        for institution, accounts in self.institution_account_mapping.items():
            total_flows = 0.0
            controlled_flows = 0.0
            
            for account in accounts:
                if account in self.flow_matrix:
                    account_outflows = sum(self.flow_matrix[account].values())
                    total_flows += account_outflows
                    
                    # Flows to other accounts controlled by same institution count less
                    other_institution_accounts = [a for a in accounts if a != account]
                    external_flows = sum(
                        flow for to_account, flow in self.flow_matrix[account].items()
                        if to_account not in other_institution_accounts
                    )
                    controlled_flows += external_flows
            
            if total_flows > 0:
                influence_scores[institution] = controlled_flows / total_flows
            else:
                influence_scores[institution] = 0.0
        
        return influence_scores


@dataclass
class ConflictDetection(Node):
    """System for detecting contradictory relationships and institutional conflicts per Hayden's SFM methodology."""
    
    analyzed_system_id: Optional[uuid.UUID] = None  # System being analyzed
    conflict_type: ConflictType = ConflictType.VALUE_CONFLICT
    
    # Detected conflicts
    direct_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Direct contradictions
    indirect_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Indirect conflicts
    potential_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Potential future conflicts
    
    # Conflict characteristics
    conflict_intensity: Dict[str, float] = field(default_factory=lambda: {})  # Conflict ID -> intensity (0-1)
    affected_stakeholders: Dict[str, List[uuid.UUID]] = field(default_factory=lambda: {})  # Conflict -> stakeholders
    resolution_difficulty: Dict[str, float] = field(default_factory=lambda: {})  # Conflict -> difficulty (0-1)
    
    # Enhanced SFM integration
    conflicting_matrix_cells: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=lambda: [])  # Conflicting cell pairs  
    institutional_contradictions: List[uuid.UUID] = field(default_factory=lambda: [])  # Contradictory institutions
    value_system_conflicts: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=lambda: [])  # Conflicting value systems
    
    # Delivery system conflicts - Hayden's emphasis on deliveries
    delivery_contradictions: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {})  # Conflicting deliveries
    delivery_failures: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Failed delivery relationships
    
    # Belief/Value/Attitude conflicts - Hayden's cultural analysis
    belief_value_contradictions: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Belief-value conflicts
    attitude_belief_misalignments: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Attitude-belief conflicts
    cultural_institutional_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Culture-institution conflicts
    
    # Ceremonial vs Instrumental conflicts - Core to Hayden's framework
    ceremonial_instrumental_tensions: List[Dict[str, Any]] = field(default_factory=lambda: [])
    ceremonial_dominance_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])  # When ceremonial blocks instrumental
    instrumental_disruption_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])  # When instrumental disrupts ceremonial
    
    # Technology-Institution conflicts
    technology_institution_mismatches: List[Dict[str, Any]] = field(default_factory=lambda: [])
    technological_ceremonial_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])
    
    # Ecological system conflicts - Hayden includes ecological systems
    ecological_institutional_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])
    ecological_technology_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])
    
    # Resolution approaches
    mediation_mechanisms: Dict[str, List[str]] = field(default_factory=lambda: {})  # Conflict -> mechanisms
    structural_solutions: Dict[str, List[str]] = field(default_factory=lambda: {})  # Conflict -> structural changes
    compromise_possibilities: Dict[str, List[str]] = field(default_factory=lambda: {})  # Conflict -> compromises
    
    # Temporal aspects
    conflict_trajectory: Dict[str, str] = field(default_factory=lambda: {})  # Conflict -> "escalating"/"stable"/"declining"
    historical_precedents: Dict[str, List[str]] = field(default_factory=lambda: {})  # Similar past conflicts
    urgency_levels: Dict[str, float] = field(default_factory=lambda: {})  # Conflict -> urgency (0-1)
    
    def __post_init__(self) -> None:
        """Validate that system is provided."""
        if self.analyzed_system_id is None:
            raise ValueError("analyzed_system_id is required for ConflictDetection")
    
    def detect_matrix_contradictions(self, matrix_cells: List[uuid.UUID]) -> List[Dict[str, Any]]:
        """Detect contradictions between matrix cell correlations."""
        contradictions: List[Dict[str, Any]] = []
        
        # This would need access to actual matrix cell data to implement fully
        # For now, return structure for potential contradictions
        
        for i, cell1 in enumerate(matrix_cells):
            for cell2 in matrix_cells[i+1:]:
                # Check for logical contradictions
                # Example: Institution A enhances Criterion X (+3) 
                # but Institution A conflicts with Institution B (-2)
                # and Institution B also enhances Criterion X (+3)
                # This suggests a contradiction that needs investigation
                
                contradiction: Dict[str, Any] = {
                    "cell_pair": (cell1, cell2),
                    "contradiction_type": "logical_inconsistency",
                    "severity": "moderate",
                    "description": "Potential logical inconsistency detected",
                    "investigation_needed": True
                }
                contradictions.append(contradiction)
        
        return contradictions
    
    def assess_conflict_priority(self) -> List[Dict[str, Any]]:
        """Prioritize conflicts for resolution efforts."""
        all_conflicts = self.direct_conflicts + self.indirect_conflicts
        
        prioritized: List[Dict[str, Any]] = []
        for conflict in all_conflicts:
            conflict_id = conflict.get("id", "unknown")
            
            priority_score = 0.0
            
            # Factor in intensity
            intensity = self.conflict_intensity.get(conflict_id, 0.5)
            priority_score += intensity * 0.4
            
            # Factor in urgency
            urgency = self.urgency_levels.get(conflict_id, 0.5)
            priority_score += urgency * 0.3
            
            # Factor in number of affected stakeholders
            stakeholders = len(self.affected_stakeholders.get(conflict_id, []))
            stakeholder_score = min(1.0, stakeholders / 10.0)  # Normalize
            priority_score += stakeholder_score * 0.2
            
            # Factor in resolution difficulty (inverse - easier to resolve gets higher priority)
            difficulty = self.resolution_difficulty.get(conflict_id, 0.5)
            priority_score += (1.0 - difficulty) * 0.1
            
            prioritized.append({
                "conflict": conflict,
                "priority_score": priority_score,
                "recommended_action": "immediate" if priority_score > 0.7 else "planned" if priority_score > 0.4 else "monitor"
            })
        
        return sorted(prioritized, key=lambda x: x["priority_score"], reverse=True)
    
    def generate_conflict_report(self) -> Dict[str, Any]:
        """Generate comprehensive conflict analysis report."""
        report: Dict[str, Any] = {
            "conflict_summary": {
                "total_conflicts": len(self.direct_conflicts) + len(self.indirect_conflicts),
                "direct_conflicts": len(self.direct_conflicts),
                "indirect_conflicts": len(self.indirect_conflicts),
                "high_priority_conflicts": 0
            },
            "conflict_types": {},
            "affected_areas": [],
            "resolution_recommendations": [],
            "monitoring_requirements": []
        }
        
        # Count high priority conflicts
        priority_analysis = self.assess_conflict_priority()
        report["conflict_summary"]["high_priority_conflicts"] = len([
            c for c in priority_analysis if c["priority_score"] > 0.7
        ])
        
        # Analyze conflict types
        all_conflicts = self.direct_conflicts + self.indirect_conflicts
        for conflict in all_conflicts:
            conflict_type = conflict.get("type", "unknown")
            if conflict_type not in report["conflict_types"]:
                report["conflict_types"][conflict_type] = 0
            report["conflict_types"][conflict_type] += 1
        
        # Identify affected areas
        if self.conflicting_matrix_cells:
            report["affected_areas"].append("Matrix cell relationships")
        if self.institutional_contradictions:
            report["affected_areas"].append("Institutional arrangements")
        if self.value_system_conflicts:
            report["affected_areas"].append("Value system alignment")
        
        # Generate recommendations
        for conflict_id, mechanisms in self.mediation_mechanisms.items():
            if mechanisms:
                report["resolution_recommendations"].append(f"Conflict {conflict_id}: {mechanisms[0]}")
        
        return report
    
    def analyze_delivery_system_conflicts(self) -> Dict[str, Any]:
        """Analyze conflicts in delivery relationships - core to Hayden's SFM."""
        delivery_analysis: Dict[str, Any] = {
            "total_delivery_conflicts": len(self.delivery_contradictions),
            "failed_deliveries": len(self.delivery_failures),
            "delivery_conflict_patterns": [],
            "critical_delivery_failures": []
        }
        
        # Analyze patterns in delivery conflicts
        for delivery_type, conflicts in self.delivery_contradictions.items():
            if len(conflicts) > 2:  # Pattern threshold
                delivery_analysis["delivery_conflict_patterns"].append({
                    "delivery_type": delivery_type,
                    "conflict_count": len(conflicts),
                    "severity": "high" if len(conflicts) > 5 else "moderate"
                })
        
        # Identify critical delivery failures
        for failure in self.delivery_failures:
            if failure.get("criticality", 0.0) > 0.7:
                delivery_analysis["critical_delivery_failures"].append(failure)
        
        return delivery_analysis
    
    def analyze_ceremonial_instrumental_conflicts(self) -> Dict[str, Any]:
        """Analyze ceremonial vs instrumental conflicts - central to Hayden's theory."""
        ceremonial_analysis: Dict[str, Any] = {
            "total_tensions": len(self.ceremonial_instrumental_tensions),
            "ceremonial_dominance_cases": len(self.ceremonial_dominance_conflicts),
            "instrumental_disruption_cases": len(self.instrumental_disruption_conflicts),
            "balance_assessment": "unknown",
            "intervention_priorities": []
        }
        
        # Assess overall balance
        ceremonial_dominance = len(self.ceremonial_dominance_conflicts)
        instrumental_disruption = len(self.instrumental_disruption_conflicts)
        
        if ceremonial_dominance > instrumental_disruption * 2:
            ceremonial_analysis["balance_assessment"] = "ceremonial_dominated"
            ceremonial_analysis["intervention_priorities"].append("Reduce ceremonial barriers")
        elif instrumental_disruption > ceremonial_dominance * 2:
            ceremonial_analysis["balance_assessment"] = "excessive_instrumental_change"
            ceremonial_analysis["intervention_priorities"].append("Support ceremonial stabilization")
        else:
            ceremonial_analysis["balance_assessment"] = "relatively_balanced"
        
        return ceremonial_analysis
    
    def analyze_cultural_institutional_misalignments(self) -> Dict[str, Any]:
        """Analyze conflicts between cultural elements and institutions."""
        cultural_analysis: Dict[str, Any] = {
            "belief_value_conflicts": len(self.belief_value_contradictions),
            "attitude_misalignments": len(self.attitude_belief_misalignments),
            "culture_institution_conflicts": len(self.cultural_institutional_conflicts),
            "alignment_score": 0.0,
            "cultural_intervention_needs": []
        }
        
        # Calculate overall cultural alignment score
        total_cultural_conflicts = (
            len(self.belief_value_contradictions) +
            len(self.attitude_belief_misalignments) +
            len(self.cultural_institutional_conflicts)
        )
        
        # Inverse relationship - more conflicts = lower alignment
        if total_cultural_conflicts == 0:
            cultural_analysis["alignment_score"] = 1.0
        else:
            cultural_analysis["alignment_score"] = max(0.0, 1.0 - (total_cultural_conflicts / 20.0))
        
        # Identify intervention needs
        if len(self.belief_value_contradictions) > 3:
            cultural_analysis["cultural_intervention_needs"].append("Belief-value reconciliation")
        if len(self.attitude_belief_misalignments) > 3:
            cultural_analysis["cultural_intervention_needs"].append("Attitude adjustment programs")
        if len(self.cultural_institutional_conflicts) > 3:
            cultural_analysis["cultural_intervention_needs"].append("Institutional cultural adaptation")
        
        return cultural_analysis
    
    def analyze_ecological_system_conflicts(self) -> Dict[str, Any]:
        """Analyze conflicts involving ecological systems - part of Hayden's integrated approach."""
        ecological_analysis: Dict[str, Any] = {
            "ecological_institutional_conflicts": len(self.ecological_institutional_conflicts),
            "ecological_technology_conflicts": len(self.ecological_technology_conflicts),
            "ecological_sustainability_score": 0.0,
            "environmental_intervention_priorities": []
        }
        
        # Calculate sustainability score
        total_ecological_conflicts = (
            len(self.ecological_institutional_conflicts) +
            len(self.ecological_technology_conflicts)
        )
        
        if total_ecological_conflicts == 0:
            ecological_analysis["ecological_sustainability_score"] = 1.0
        else:
            ecological_analysis["ecological_sustainability_score"] = max(0.0, 1.0 - (total_ecological_conflicts / 15.0))
        
        # Identify intervention priorities
        if len(self.ecological_institutional_conflicts) > 2:
            ecological_analysis["environmental_intervention_priorities"].append("Institutional ecological adaptation")
        if len(self.ecological_technology_conflicts) > 2:
            ecological_analysis["environmental_intervention_priorities"].append("Green technology transition")
        
        return ecological_analysis


# Missing Core Classes for Enhanced SFM Framework

@dataclass
class EcologicalSystem(Node):
    """Environmental component integration - part of Hayden's comprehensive SFM approach."""
    
    ecosystem_type: Optional[str] = None  # "forest", "watershed", "urban", etc.
    environmental_health: Optional[float] = None  # Overall health score (0-1)
    biodiversity_index: Optional[float] = None  # Biodiversity measure (0-1)
    carrying_capacity: Optional[float] = None  # Ecosystem carrying capacity
    resource_stocks: Dict[str, float] = field(default_factory=lambda: {})  # Resource type -> quantity
    
    # Institutional relationships
    governing_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions that govern this ecosystem
    impacting_technologies: List[uuid.UUID] = field(default_factory=lambda: [])  # Technologies affecting ecosystem
    dependent_communities: List[uuid.UUID] = field(default_factory=lambda: [])  # Communities dependent on ecosystem
    
    # Environmental indicators
    pollution_levels: Dict[str, float] = field(default_factory=lambda: {})  # Pollutant type -> level
    regeneration_capacity: Optional[float] = None  # Ability to regenerate (0-1)
    resilience_factors: List[str] = field(default_factory=lambda: [])  # Factors supporting resilience
    vulnerability_factors: List[str] = field(default_factory=lambda: [])  # Factors creating vulnerability
    
    # Matrix integration
    matrix_deliveries: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Ecosystem services delivered
    matrix_requirements: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Requirements from other matrix elements
    ecological_constraints: Dict[str, List[str]] = field(default_factory=lambda: {})  # Constraints on other activities
    
    def calculate_sustainability_score(self) -> float:
        """Calculate overall ecosystem sustainability score."""
        scores: List[float] = []
        
        if self.environmental_health is not None:
            scores.append(self.environmental_health * 0.3)
        if self.biodiversity_index is not None:
            scores.append(self.biodiversity_index * 0.25)
        if self.regeneration_capacity is not None:
            scores.append(self.regeneration_capacity * 0.25)
        
        # Factor in pollution (inverse relationship)
        if self.pollution_levels:
            avg_pollution = sum(self.pollution_levels.values()) / len(self.pollution_levels)
            scores.append((1.0 - min(1.0, avg_pollution)) * 0.2)
        
        return sum(scores) if scores else 0.0


@dataclass
class SocialBelief(Node):
    """Social beliefs distinct from values and attitudes - core to Hayden's cultural analysis."""
    
    belief_type: Optional[str] = None  # "factual", "normative", "existential", etc.
    belief_strength: Optional[float] = None  # Strength of belief (0-1)
    cultural_domain: Optional[str] = None  # Domain where belief operates
    legitimacy_source: Optional[str] = None  # Source of belief legitimacy
    
    # Belief relationships
    supporting_values: List[uuid.UUID] = field(default_factory=lambda: [])  # Values that support this belief
    conflicting_beliefs: List[uuid.UUID] = field(default_factory=lambda: [])  # Contradictory beliefs
    institutional_embedment: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions embedding this belief
    
    # Belief characteristics
    evidence_basis: List[str] = field(default_factory=lambda: [])  # Evidence supporting belief
    cultural_transmission: Optional[str] = None  # How belief is transmitted
    change_resistance: Optional[float] = None  # Resistance to change (0-1)
    social_reinforcement: Optional[float] = None  # Social reinforcement level (0-1)
    
    # Matrix integration
    institutional_influence: Dict[uuid.UUID, float] = field(default_factory=lambda: {})  # Institution -> influence level
    attitude_mediation_effects: Dict[uuid.UUID, float] = field(default_factory=lambda: {})  # Attitude -> mediation effect
    belief_coherence_score: Optional[float] = None  # Internal coherence (0-1)
    
    def assess_belief_stability(self) -> Dict[str, Any]:
        """Assess stability and change potential of belief."""
        stability_assessment: Dict[str, Any] = {
            "stability_level": "unknown",
            "change_potential": 0.0,
            "reinforcement_factors": len(self.supporting_values),
            "challenge_factors": len(self.conflicting_beliefs)
        }
        
        if self.change_resistance is not None and self.social_reinforcement is not None:
            stability_score = (self.change_resistance + self.social_reinforcement) / 2
            if stability_score >= 0.7:
                stability_assessment["stability_level"] = "high"
            elif stability_score >= 0.4:
                stability_assessment["stability_level"] = "moderate"
            else:
                stability_assessment["stability_level"] = "low"
            
            stability_assessment["change_potential"] = 1.0 - stability_score
        
        return stability_assessment


@dataclass
class CulturalAttitude(Node):
    """Attitudes that mediate between beliefs and institutions - Hayden's cultural framework."""
    
    attitude_type: Optional[str] = None  # "supportive", "resistant", "neutral", etc.
    attitude_strength: Optional[float] = None  # Strength of attitude (0-1)
    emotional_component: Optional[float] = None  # Emotional intensity (0-1)
    behavioral_tendency: Optional[str] = None  # Behavioral predisposition
    
    # Attitude relationships
    related_beliefs: List[uuid.UUID] = field(default_factory=lambda: [])  # Beliefs this attitude relates to
    influenced_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions influenced by attitude
    attitude_objects: List[uuid.UUID] = field(default_factory=lambda: [])  # Objects of the attitude
    
    # Attitude characteristics
    formation_context: Optional[str] = None  # Context where attitude formed
    stability_factors: List[str] = field(default_factory=lambda: [])  # Factors supporting attitude stability
    change_triggers: List[str] = field(default_factory=lambda: [])  # Potential change triggers
    social_desirability: Optional[float] = None  # Social acceptability (0-1)
    
    # Matrix integration
    institutional_mediation_effects: Dict[uuid.UUID, float] = field(default_factory=lambda: {})  # Institution -> mediation
    belief_attitude_coherence: Optional[float] = None  # Coherence with beliefs (0-1)
    behavioral_predictability: Optional[float] = None  # Predictability of behavior (0-1)
    
    def analyze_mediation_capacity(self) -> Dict[str, Any]:
        """Analyze attitude's capacity to mediate between beliefs and institutions."""
        mediation_analysis: Dict[str, Any] = {
            "mediation_strength": 0.0,
            "coherence_level": "unknown",
            "influence_scope": len(self.influenced_institutions),
            "stability_rating": "unknown"
        }
        
        # Calculate mediation strength
        if self.attitude_strength is not None and self.behavioral_predictability is not None:
            mediation_analysis["mediation_strength"] = (self.attitude_strength + self.behavioral_predictability) / 2
        
        # Assess coherence
        if self.belief_attitude_coherence is not None:
            if self.belief_attitude_coherence >= 0.7:
                mediation_analysis["coherence_level"] = "high"
            elif self.belief_attitude_coherence >= 0.4:
                mediation_analysis["coherence_level"] = "moderate"
            else:
                mediation_analysis["coherence_level"] = "low"
        
        # Assess stability
        stability_score = len(self.stability_factors) / max(1, len(self.change_triggers))
        if stability_score >= 2.0:
            mediation_analysis["stability_rating"] = "high"
        elif stability_score >= 1.0:
            mediation_analysis["stability_rating"] = "moderate"
        else:
            mediation_analysis["stability_rating"] = "low"
        
        return mediation_analysis


@dataclass
class DeliveryRelationship(Node):
    """Models how system components make deliveries to each other - core to Hayden's SFM."""
    
    source_component_id: uuid.UUID
    target_component_id: uuid.UUID
    delivery_type: Optional[str] = None  # "service", "resource", "information", "value", etc.
    delivery_content: Optional[str] = None  # What is being delivered
    delivery_mechanism: Optional[str] = None  # How delivery is made
    
    # Delivery characteristics
    delivery_quality: Optional[float] = None  # Quality of delivery (0-1)
    delivery_reliability: Optional[float] = None  # Reliability (0-1)
    delivery_frequency: Optional[str] = None  # How often delivery occurs
    delivery_capacity: Optional[float] = None  # Maximum capacity (0-1)
    delivery_efficiency: Optional[float] = None  # Efficiency of delivery (0-1)
    
    # Relationship dynamics
    reciprocity_level: Optional[float] = None  # Level of reciprocity (0-1)
    dependency_strength: Optional[float] = None  # How dependent target is on delivery (0-1)
    substitutability: Optional[float] = None  # Availability of substitutes (0-1)
    criticality: Optional[float] = None  # Criticality to system functioning (0-1)
    
    # Matrix integration
    institutional_mediation: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions mediating delivery
    technological_requirements: List[uuid.UUID] = field(default_factory=lambda: [])  # Required technologies
    cultural_factors: Dict[str, float] = field(default_factory=lambda: {})  # Cultural influences on delivery
    ecological_constraints: List[str] = field(default_factory=lambda: [])  # Environmental constraints
    
    # Performance metrics
    delivery_failures: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Record of failures
    improvement_opportunities: List[str] = field(default_factory=lambda: [])  # Potential improvements
    monitoring_indicators: Dict[str, float] = field(default_factory=lambda: {})  # Performance indicators
    
    def assess_delivery_performance(self) -> Dict[str, Any]:
        """Assess overall delivery relationship performance."""
        performance_assessment: Dict[str, Any] = {
            "overall_performance": 0.0,
            "reliability_rating": "unknown",
            "efficiency_rating": "unknown",
            "criticality_level": "unknown",
            "improvement_potential": 0.0
        }
        
        # Calculate overall performance
        performance_factors: List[float] = []
        if self.delivery_quality is not None:
            performance_factors.append(self.delivery_quality * 0.3)
        if self.delivery_reliability is not None:
            performance_factors.append(self.delivery_reliability * 0.3)
        if self.delivery_efficiency is not None:
            performance_factors.append(self.delivery_efficiency * 0.4)
        
        if performance_factors:
            performance_assessment["overall_performance"] = sum(performance_factors)
        
        # Assess reliability rating
        if self.delivery_reliability is not None:
            if self.delivery_reliability >= 0.8:
                performance_assessment["reliability_rating"] = "high"
            elif self.delivery_reliability >= 0.6:
                performance_assessment["reliability_rating"] = "moderate"
            else:
                performance_assessment["reliability_rating"] = "low"
        
        # Assess efficiency rating
        if self.delivery_efficiency is not None:
            if self.delivery_efficiency >= 0.8:
                performance_assessment["efficiency_rating"] = "high"
            elif self.delivery_efficiency >= 0.6:
                performance_assessment["efficiency_rating"] = "moderate"
            else:
                performance_assessment["efficiency_rating"] = "low"
        
        # Assess criticality level
        if self.criticality is not None:
            if self.criticality >= 0.7:
                performance_assessment["criticality_level"] = "high"
            elif self.criticality >= 0.4:
                performance_assessment["criticality_level"] = "moderate"
            else:
                performance_assessment["criticality_level"] = "low"
        
        # Calculate improvement potential
        current_performance = performance_assessment["overall_performance"]
        performance_assessment["improvement_potential"] = max(0.0, 1.0 - current_performance)
        
        return performance_assessment


@dataclass
class SocialIndicatorSystem(Node):
    """Systematic social indicator development and management - key to Hayden's methodology."""
    
    indicator_category: Optional[str] = None  # "economic", "social", "environmental", "institutional"
    measurement_framework: Optional[str] = None  # Framework used for measurement
    data_collection_method: Optional[str] = None  # How data is collected
    update_frequency: Optional[str] = None  # How often indicators are updated
    
    # Indicator components
    primary_indicators: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})  # Core indicators
    secondary_indicators: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})  # Supporting indicators
    composite_indicators: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})  # Composite measures
    
    # System characteristics
    indicator_validity: Optional[float] = None  # Validity of indicator system (0-1)
    indicator_reliability: Optional[float] = None  # Reliability of measurements (0-1)
    stakeholder_acceptance: Optional[float] = None  # Stakeholder acceptance level (0-1)
    policy_relevance: Optional[float] = None  # Relevance to policy decisions (0-1)
    
    # Matrix integration
    matrix_cell_linkages: Dict[uuid.UUID, List[str]] = field(default_factory=lambda: {})  # Cell -> indicators
    institutional_monitoring: Dict[uuid.UUID, List[str]] = field(default_factory=lambda: {})  # Institution -> responsibilities
    feedback_mechanisms: List[Dict[str, Any]] = field(default_factory=lambda: [])  # How indicators feed back to system
    
    # Database integration
    data_sources: List[str] = field(default_factory=lambda: [])  # Data sources
    database_schema: Optional[str] = None  # Database structure
    statistical_methods: List[str] = field(default_factory=lambda: [])  # Statistical approaches used
    quality_controls: List[str] = field(default_factory=lambda: [])  # Quality control measures
    
    def assess_indicator_system_quality(self) -> Dict[str, Any]:
        """Assess quality and effectiveness of indicator system."""
        quality_assessment: Dict[str, Any] = {
            "overall_quality": 0.0,
            "coverage_adequacy": "unknown",
            "measurement_quality": "unknown",
            "policy_utility": "unknown",
            "improvement_needs": []
        }
        
        # Calculate overall quality
        quality_factors: List[float] = []
        if self.indicator_validity is not None:
            quality_factors.append(self.indicator_validity * 0.3)
        if self.indicator_reliability is not None:
            quality_factors.append(self.indicator_reliability * 0.3)
        if self.policy_relevance is not None:
            quality_factors.append(self.policy_relevance * 0.4)
        
        if quality_factors:
            quality_assessment["overall_quality"] = sum(quality_factors)
        
        # Assess coverage adequacy
        total_indicators = len(self.primary_indicators) + len(self.secondary_indicators)
        if total_indicators >= 10:
            quality_assessment["coverage_adequacy"] = "comprehensive"
        elif total_indicators >= 5:
            quality_assessment["coverage_adequacy"] = "adequate"
        else:
            quality_assessment["coverage_adequacy"] = "limited"
        
        # Assess measurement quality
        if self.indicator_reliability is not None:
            if self.indicator_reliability >= 0.8:
                quality_assessment["measurement_quality"] = "high"
            elif self.indicator_reliability >= 0.6:
                quality_assessment["measurement_quality"] = "moderate"
            else:
                quality_assessment["measurement_quality"] = "low"
                quality_assessment["improvement_needs"].append("Improve measurement reliability")
        
        # Assess policy utility
        if self.policy_relevance is not None:
            if self.policy_relevance >= 0.8:
                quality_assessment["policy_utility"] = "high"
            elif self.policy_relevance >= 0.6:
                quality_assessment["policy_utility"] = "moderate"
            else:
                quality_assessment["policy_utility"] = "low"
                quality_assessment["improvement_needs"].append("Enhance policy relevance")
        
        return quality_assessment


@dataclass
class CircularCausationProcess(Node):
    """Models Veblen's circular and cumulative causation processes - foundational to Hayden's SFM."""
    
    process_type: Optional[str] = None  # "virtuous", "vicious", "neutral"
    causation_strength: Optional[float] = None  # Strength of causal process (0-1)
    feedback_polarity: Optional[str] = None  # "positive", "negative", "mixed"
    time_scale: Optional[str] = None  # "short-term", "medium-term", "long-term"
    
    # Process components
    causal_elements: List[uuid.UUID] = field(default_factory=lambda: [])  # Elements in causal chain
    feedback_loops: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Feedback mechanisms
    reinforcement_mechanisms: List[str] = field(default_factory=lambda: [])  # What reinforces the process
    disruption_factors: List[str] = field(default_factory=lambda: [])  # What can disrupt the process
    
    # Process dynamics
    momentum_level: Optional[float] = None  # Process momentum (0-1)
    stability_tendency: Optional[float] = None  # Tendency toward stability (0-1)
    change_acceleration: Optional[float] = None  # Rate of change acceleration
    threshold_effects: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Critical thresholds
    
    # Matrix integration
    institutional_embedment: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions embedding process
    technological_enablers: List[uuid.UUID] = field(default_factory=lambda: [])  # Technologies enabling process
    cultural_reinforcement: Dict[str, float] = field(default_factory=lambda: {})  # Cultural reinforcement factors
    ecological_limits: List[str] = field(default_factory=lambda: [])  # Environmental constraints
    
    # Intervention points
    intervention_opportunities: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Where to intervene
    policy_leverage_points: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Policy intervention points
    
    def analyze_causation_dynamics(self) -> Dict[str, Any]:
        """Analyze the dynamics of the circular causation process."""
        dynamics_analysis: Dict[str, Any] = {
            "process_strength": self.causation_strength or 0.0,
            "process_direction": "unknown",
            "stability_assessment": "unknown",
            "intervention_potential": 0.0,
            "system_impact": "unknown"
        }
        
        # Determine process direction
        if self.process_type == "virtuous":
            dynamics_analysis["process_direction"] = "beneficial"
        elif self.process_type == "vicious":
            dynamics_analysis["process_direction"] = "harmful"
        else:
            dynamics_analysis["process_direction"] = "neutral"
        
        # Assess stability
        if self.stability_tendency is not None:
            if self.stability_tendency >= 0.7:
                dynamics_analysis["stability_assessment"] = "highly_stable"
            elif self.stability_tendency >= 0.4:
                dynamics_analysis["stability_assessment"] = "moderately_stable"
            else:
                dynamics_analysis["stability_assessment"] = "unstable"
        
        # Calculate intervention potential
        intervention_factors = len(self.intervention_opportunities) + len(self.policy_leverage_points)
        dynamics_analysis["intervention_potential"] = min(1.0, intervention_factors / 10.0)
        
        # Assess system impact
        if self.momentum_level is not None and self.causation_strength is not None:
            impact_score = (self.momentum_level + self.causation_strength) / 2
            if impact_score >= 0.7:
                dynamics_analysis["system_impact"] = "high"
            elif impact_score >= 0.4:
                dynamics_analysis["system_impact"] = "moderate"
            else:
                dynamics_analysis["system_impact"] = "low"
        
        return dynamics_analysis


@dataclass
class MatrixDeliveryNetwork(Node):
    """Network of deliveries between matrix cells - central to Hayden's SFM methodology."""
    
    network_scope: Optional[str] = None  # "local", "regional", "national", "global"
    network_density: Optional[float] = None  # Density of delivery relationships (0-1)
    network_centralization: Optional[float] = None  # Centralization level (0-1)
    network_efficiency: Optional[float] = None  # Overall efficiency (0-1)
    
    # Network components
    delivery_relationships: List[uuid.UUID] = field(default_factory=lambda: [])  # Individual delivery relationships
    hub_components: List[uuid.UUID] = field(default_factory=lambda: [])  # Network hubs
    peripheral_components: List[uuid.UUID] = field(default_factory=lambda: [])  # Peripheral components
    bridge_components: List[uuid.UUID] = field(default_factory=lambda: [])  # Bridge components
    
    # Network characteristics
    redundancy_level: Optional[float] = None  # Network redundancy (0-1)
    vulnerability_points: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Network vulnerabilities
    resilience_factors: List[str] = field(default_factory=lambda: [])  # Factors supporting resilience
    adaptation_capacity: Optional[float] = None  # Network adaptation capacity (0-1)
    
    # Network flows
    flow_patterns: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {})  # Flow type -> patterns
    bottlenecks: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Network bottlenecks
    capacity_constraints: Dict[str, float] = field(default_factory=lambda: {})  # Constraint type -> severity
    
    # Performance metrics
    delivery_success_rate: Optional[float] = None  # Overall success rate (0-1)
    network_responsiveness: Optional[float] = None  # Response to changes (0-1)
    coordination_effectiveness: Optional[float] = None  # Coordination quality (0-1)
    
    def analyze_network_performance(self) -> Dict[str, Any]:
        """Analyze overall network performance and health."""
        network_analysis: Dict[str, Any] = {
            "overall_performance": 0.0,
            "network_health": "unknown",
            "critical_vulnerabilities": len(self.vulnerability_points),
            "improvement_priorities": [],
            "network_type": "unknown"
        }
        
        # Calculate overall performance
        performance_factors: List[float] = []
        if self.network_efficiency is not None:
            performance_factors.append(self.network_efficiency * 0.3)
        if self.delivery_success_rate is not None:
            performance_factors.append(self.delivery_success_rate * 0.3)
        if self.coordination_effectiveness is not None:
            performance_factors.append(self.coordination_effectiveness * 0.4)
        
        if performance_factors:
            network_analysis["overall_performance"] = sum(performance_factors)
        
        # Assess network health
        health_score = network_analysis["overall_performance"]
        if health_score >= 0.8:
            network_analysis["network_health"] = "excellent"
        elif health_score >= 0.6:
            network_analysis["network_health"] = "good"
        elif health_score >= 0.4:
            network_analysis["network_health"] = "fair"
        else:
            network_analysis["network_health"] = "poor"
        
        # Determine network type
        if self.network_centralization is not None:
            if self.network_centralization >= 0.7:
                network_analysis["network_type"] = "centralized"
            elif self.network_centralization <= 0.3:
                network_analysis["network_type"] = "decentralized"
            else:
                network_analysis["network_type"] = "distributed"
        
        # Identify improvement priorities
        if len(self.bottlenecks) > 2:
            network_analysis["improvement_priorities"].append("Address network bottlenecks")
        if len(self.vulnerability_points) > 3:
            network_analysis["improvement_priorities"].append("Strengthen network resilience")
        if self.redundancy_level is not None and self.redundancy_level < 0.5:
            network_analysis["improvement_priorities"].append("Increase network redundancy")
        
        return network_analysis


# Methodological Framework Classes for Hayden's SFM Approach

@dataclass
class InstrumentalistInquiryFramework(Node):
    """Represents Hayden's instrumentalist approach to inquiry - methodological foundation of SFM."""
    
    inquiry_purpose: Optional[str] = None  # Purpose of the inquiry
    problem_context: Optional[str] = None  # Context of the problem being investigated
    normative_orientation: Optional[str] = None  # Normative stance of the inquiry
    embedded_values: List[str] = field(default_factory=lambda: [])  # Values embedded in the inquiry
    
    # Instrumentalist characteristics
    problem_solving_focus: Optional[float] = None  # Focus on problem-solving (0-1)
    contextual_sensitivity: Optional[float] = None  # Sensitivity to context (0-1)
    evolutionary_perspective: Optional[float] = None  # Evolutionary approach level (0-1)
    holistic_approach: Optional[float] = None  # Holistic thinking level (0-1)
    
    # Inquiry process
    inquiry_stages: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Stages of inquiry
    knowledge_integration: Dict[str, List[str]] = field(default_factory=lambda: {})  # How knowledge is integrated
    stakeholder_involvement: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Stakeholder participation
    
    # Methodological principles
    fallibilistic_approach: Optional[bool] = None  # Recognition of fallibility
    democratic_participation: Optional[float] = None  # Level of democratic participation (0-1)
    transparent_process: Optional[float] = None  # Process transparency (0-1)
    adaptability: Optional[float] = None  # Adaptability to changing contexts (0-1)
    
    def assess_inquiry_quality(self) -> Dict[str, Any]:
        """Assess quality of instrumentalist inquiry framework."""
        quality_assessment: Dict[str, Any] = {
            "overall_quality": 0.0,
            "methodological_rigor": "unknown",
            "stakeholder_engagement": "unknown",
            "problem_relevance": "unknown",
            "improvement_areas": []
        }
        
        # Calculate overall quality
        quality_factors: List[float] = []
        if self.problem_solving_focus is not None:
            quality_factors.append(self.problem_solving_focus * 0.25)
        if self.contextual_sensitivity is not None:
            quality_factors.append(self.contextual_sensitivity * 0.25)
        if self.holistic_approach is not None:
            quality_factors.append(self.holistic_approach * 0.25)
        if self.democratic_participation is not None:
            quality_factors.append(self.democratic_participation * 0.25)
        
        if quality_factors:
            quality_assessment["overall_quality"] = sum(quality_factors)
        
        # Assess methodological rigor
        rigor_score = 0.0
        if self.transparent_process is not None:
            rigor_score += self.transparent_process * 0.5
        if self.adaptability is not None:
            rigor_score += self.adaptability * 0.5
        
        if rigor_score >= 0.7:
            quality_assessment["methodological_rigor"] = "high"
        elif rigor_score >= 0.4:
            quality_assessment["methodological_rigor"] = "moderate"
        else:
            quality_assessment["methodological_rigor"] = "low"
            quality_assessment["improvement_areas"].append("Enhance methodological rigor")
        
        # Assess stakeholder engagement
        if len(self.stakeholder_involvement) >= 3:
            quality_assessment["stakeholder_engagement"] = "comprehensive"
        elif len(self.stakeholder_involvement) >= 1:
            quality_assessment["stakeholder_engagement"] = "limited"
        else:
            quality_assessment["stakeholder_engagement"] = "minimal"
            quality_assessment["improvement_areas"].append("Increase stakeholder engagement")
        
        # Assess problem relevance
        if self.problem_solving_focus is not None:
            if self.problem_solving_focus >= 0.8:
                quality_assessment["problem_relevance"] = "high"
            elif self.problem_solving_focus >= 0.6:
                quality_assessment["problem_relevance"] = "moderate"
            else:
                quality_assessment["problem_relevance"] = "low"
                quality_assessment["improvement_areas"].append("Strengthen problem relevance")
        
        return quality_assessment


@dataclass
class NormativeSystemsAnalysis(Node):
    """Hayden's normative systems analysis framework for SFM evaluation."""
    
    normative_criteria: List[str] = field(default_factory=lambda: [])  # Normative evaluation criteria
    value_hierarchy: Dict[str, float] = field(default_factory=lambda: {})  # Value priorities
    ethical_framework: Optional[str] = None  # Underlying ethical framework
    social_welfare_measure: Optional[str] = None  # How social welfare is measured
    
    # Analysis components
    system_evaluation: Dict[str, float] = field(default_factory=lambda: {})  # System component evaluations
    alternative_assessments: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Alternative system assessments
    policy_recommendations: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Policy recommendations
    
    # Normative principles
    life_process_enhancement: Optional[float] = None  # Focus on life process (0-1)
    democratic_values: Optional[float] = None  # Democratic value emphasis (0-1)
    sustainability_priority: Optional[float] = None  # Sustainability priority (0-1)
    equity_considerations: Optional[float] = None  # Equity emphasis (0-1)
    
    # Evaluation process
    stakeholder_values: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})  # Stakeholder value sets
    value_conflicts: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Identified value conflicts
    consensus_mechanisms: List[str] = field(default_factory=lambda: [])  # Consensus-building approaches
    
    def conduct_normative_evaluation(self) -> Dict[str, Any]:
        """Conduct comprehensive normative evaluation of system."""
        evaluation_results: Dict[str, Any] = {
            "overall_system_score": 0.0,
            "value_alignment": "unknown",
            "policy_priority_areas": [],
            "stakeholder_consensus_level": 0.0,
            "normative_recommendations": []
        }
        
        # Calculate overall system score
        if self.system_evaluation:
            weighted_scores: List[float] = []
            for component, score in self.system_evaluation.items():
                weight = self.value_hierarchy.get(component, 1.0)
                weighted_scores.append(score * weight)
            
            if weighted_scores:
                evaluation_results["overall_system_score"] = sum(weighted_scores) / len(weighted_scores)
        
        # Assess value alignment
        alignment_factors: List[float] = []
        if self.life_process_enhancement is not None:
            alignment_factors.append(self.life_process_enhancement)
        if self.democratic_values is not None:
            alignment_factors.append(self.democratic_values)
        if self.sustainability_priority is not None:
            alignment_factors.append(self.sustainability_priority)
        if self.equity_considerations is not None:
            alignment_factors.append(self.equity_considerations)
        
        if alignment_factors:
            avg_alignment = sum(alignment_factors) / len(alignment_factors)
            if avg_alignment >= 0.8:
                evaluation_results["value_alignment"] = "strong"
            elif avg_alignment >= 0.6:
                evaluation_results["value_alignment"] = "moderate"
            else:
                evaluation_results["value_alignment"] = "weak"
        
        # Identify policy priority areas
        for component, score in self.system_evaluation.items():
            if score < 0.6:  # Below threshold
                evaluation_results["policy_priority_areas"].append(component)
        
        # Calculate stakeholder consensus level
        if self.stakeholder_values:
            consensus_scores: List[float] = []
            for value_type in self.value_hierarchy.keys():
                stakeholder_scores = [
                    stakeholder_vals.get(value_type, 0.0) 
                    for stakeholder_vals in self.stakeholder_values.values()
                ]
                if len(stakeholder_scores) > 1:
                    # Calculate variance as inverse measure of consensus
                    mean_score = sum(stakeholder_scores) / len(stakeholder_scores)
                    variance = sum((s - mean_score)**2 for s in stakeholder_scores) / len(stakeholder_scores)
                    consensus_scores.append(max(0.0, 1.0 - variance))
            
            if consensus_scores:
                evaluation_results["stakeholder_consensus_level"] = sum(consensus_scores) / len(consensus_scores)
        
        # Generate normative recommendations
        if evaluation_results["overall_system_score"] < 0.7:
            evaluation_results["normative_recommendations"].append("System requires significant normative improvements")
        if evaluation_results["value_alignment"] == "weak":
            evaluation_results["normative_recommendations"].append("Strengthen alignment with core values")
        if evaluation_results["stakeholder_consensus_level"] < 0.6:
            evaluation_results["normative_recommendations"].append("Build greater stakeholder consensus")
        
        return evaluation_results


@dataclass
class PolicyRelevanceIntegration(Node):
    """Integration framework connecting SFM analysis to political action - key to Hayden's approach."""
    
    policy_context: Optional[str] = None  # Policy environment context
    political_feasibility: Optional[float] = None  # Political feasibility (0-1)
    implementation_capacity: Optional[float] = None  # Implementation capacity (0-1)
    stakeholder_support: Optional[float] = None  # Stakeholder support level (0-1)
    
    # Political action integration
    lobbying_strategies: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Lobbying approaches
    budgetary_processes: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Budget integration
    administrative_implementation: Dict[str, List[str]] = field(default_factory=lambda: {})  # Implementation pathways
    legislative_pathways: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Legislative routes
    
    # Policy tools
    policy_instruments: List[uuid.UUID] = field(default_factory=lambda: [])  # Available policy tools
    institutional_leverage_points: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Leverage points
    coalition_building_opportunities: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Coalition opportunities
    
    # Implementation strategy
    short_term_actions: List[str] = field(default_factory=lambda: [])  # Immediate actions
    medium_term_goals: List[str] = field(default_factory=lambda: [])  # Medium-term objectives
    long_term_vision: Optional[str] = None  # Long-term vision
    success_indicators: Dict[str, str] = field(default_factory=lambda: {})  # Success measures
    
    # Monitoring and feedback
    policy_monitoring_system: Optional[str] = None  # How policy is monitored
    feedback_mechanisms: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Feedback systems
    adaptive_management: Optional[float] = None  # Adaptive management capacity (0-1)
    
    def assess_policy_integration_capacity(self) -> Dict[str, Any]:
        """Assess capacity for integrating analysis with policy action."""
        integration_assessment: Dict[str, Any] = {
            "overall_integration_capacity": 0.0,
            "political_viability": "unknown",
            "implementation_readiness": "unknown",
            "stakeholder_alignment": "unknown",
            "strategic_recommendations": []
        }
        
        # Calculate overall integration capacity
        capacity_factors: List[float] = []
        if self.political_feasibility is not None:
            capacity_factors.append(self.political_feasibility * 0.3)
        if self.implementation_capacity is not None:
            capacity_factors.append(self.implementation_capacity * 0.3)
        if self.stakeholder_support is not None:
            capacity_factors.append(self.stakeholder_support * 0.4)
        
        if capacity_factors:
            integration_assessment["overall_integration_capacity"] = sum(capacity_factors)
        
        # Assess political viability
        if self.political_feasibility is not None:
            if self.political_feasibility >= 0.7:
                integration_assessment["political_viability"] = "high"
            elif self.political_feasibility >= 0.4:
                integration_assessment["political_viability"] = "moderate"
            else:
                integration_assessment["political_viability"] = "low"
                integration_assessment["strategic_recommendations"].append("Build political support")
        
        # Assess implementation readiness
        if self.implementation_capacity is not None:
            if self.implementation_capacity >= 0.7:
                integration_assessment["implementation_readiness"] = "high"
            elif self.implementation_capacity >= 0.4:
                integration_assessment["implementation_readiness"] = "moderate"
            else:
                integration_assessment["implementation_readiness"] = "low"
                integration_assessment["strategic_recommendations"].append("Strengthen implementation capacity")
        
        # Assess stakeholder alignment
        if self.stakeholder_support is not None:
            if self.stakeholder_support >= 0.7:
                integration_assessment["stakeholder_alignment"] = "strong"
            elif self.stakeholder_support >= 0.4:
                integration_assessment["stakeholder_alignment"] = "moderate"
            else:
                integration_assessment["stakeholder_alignment"] = "weak"
                integration_assessment["strategic_recommendations"].append("Build stakeholder coalitions")
        
        # Additional strategic recommendations
        if len(self.policy_instruments) < 3:
            integration_assessment["strategic_recommendations"].append("Expand policy instrument toolkit")
        if len(self.coalition_building_opportunities) < 2:
            integration_assessment["strategic_recommendations"].append("Identify coalition opportunities")
        if self.adaptive_management is not None and self.adaptive_management < 0.6:
            integration_assessment["strategic_recommendations"].append("Enhance adaptive management capacity")
        
        return integration_assessment


@dataclass
class DatabaseIntegrationCapability(Node):
    """Database integration for statistical analysis support - part of Hayden's methodology."""
    
    database_type: Optional[str] = None  # Type of database system
    data_architecture: Optional[str] = None  # Data architecture approach
    integration_level: Optional[float] = None  # Level of integration (0-1)
    data_quality_standards: List[str] = field(default_factory=lambda: [])  # Quality standards
    
    # Data management
    data_collection_protocols: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Collection protocols
    data_validation_rules: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Validation rules
    data_storage_systems: Dict[str, str] = field(default_factory=lambda: {})  # Storage systems
    data_access_controls: List[str] = field(default_factory=lambda: [])  # Access controls
    
    # Statistical analysis integration
    statistical_packages: List[str] = field(default_factory=lambda: [])  # Statistical software
    analysis_workflows: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Analysis workflows
    visualization_capabilities: Dict[str, List[str]] = field(default_factory=lambda: {})  # Visualization tools
    reporting_systems: List[Dict[str, Any]] = field(default_factory=lambda: [])  # Reporting capabilities
    
    # Matrix integration
    matrix_data_mapping: Dict[str, str] = field(default_factory=lambda: {})  # Matrix -> database mapping
    indicator_calculation_rules: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})  # Calculation rules
    automated_updates: List[str] = field(default_factory=lambda: [])  # Automated update processes
    
    # Performance metrics
    data_processing_speed: Optional[float] = None  # Processing speed rating (0-1)
    system_reliability: Optional[float] = None  # System reliability (0-1)
    user_satisfaction: Optional[float] = None  # User satisfaction (0-1)
    
    def evaluate_database_capability(self) -> Dict[str, Any]:
        """Evaluate database integration and analytical capability."""
        capability_evaluation: Dict[str, Any] = {
            "overall_capability": 0.0,
            "data_management_quality": "unknown",
            "analytical_power": "unknown",
            "matrix_integration_level": "unknown",
            "improvement_priorities": []
        }
        
        # Calculate overall capability
        capability_factors: List[float] = []
        if self.integration_level is not None:
            capability_factors.append(self.integration_level * 0.3)
        if self.data_processing_speed is not None:
            capability_factors.append(self.data_processing_speed * 0.3)
        if self.system_reliability is not None:
            capability_factors.append(self.system_reliability * 0.4)
        
        if capability_factors:
            capability_evaluation["overall_capability"] = sum(capability_factors)
        
        # Assess data management quality
        quality_indicators = len(self.data_quality_standards) + len(self.data_validation_rules)
        if quality_indicators >= 5:
            capability_evaluation["data_management_quality"] = "high"
        elif quality_indicators >= 3:
            capability_evaluation["data_management_quality"] = "moderate"
        else:
            capability_evaluation["data_management_quality"] = "low"
            capability_evaluation["improvement_priorities"].append("Strengthen data management")
        
        # Assess analytical power
        analysis_capabilities = len(self.statistical_packages) + len(self.analysis_workflows)
        if analysis_capabilities >= 5:
            capability_evaluation["analytical_power"] = "strong"
        elif analysis_capabilities >= 3:
            capability_evaluation["analytical_power"] = "moderate"
        else:
            capability_evaluation["analytical_power"] = "limited"
            capability_evaluation["improvement_priorities"].append("Expand analytical capabilities")
        
        # Assess matrix integration level
        integration_indicators = len(self.matrix_data_mapping) + len(self.indicator_calculation_rules)
        if integration_indicators >= 10:
            capability_evaluation["matrix_integration_level"] = "comprehensive"
        elif integration_indicators >= 5:
            capability_evaluation["matrix_integration_level"] = "partial"
        else:
            capability_evaluation["matrix_integration_level"] = "minimal"
            capability_evaluation["improvement_priorities"].append("Enhance matrix integration")
        
        # Additional improvement priorities
        if self.user_satisfaction is not None and self.user_satisfaction < 0.7:
            capability_evaluation["improvement_priorities"].append("Improve user experience")
        if len(self.automated_updates) < 3:
            capability_evaluation["improvement_priorities"].append("Increase automation")
        
        return capability_evaluation
