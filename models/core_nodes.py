"""
Core SFM nodes representing primary entities.

This module defines the core Social Fabric Matrix entities including actors,
institutions, resources, processes, and flows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.metadata_models import TemporalDynamics
from models.sfm_enums import (
    ResourceType,
    FlowNature,
    FlowType,
    EnumValidator,
)


@dataclass
class Actor(Node):
    """Individuals, firms, agencies, communities."""

    legal_form: Optional[str] = None  # e.g. "Corporation", "Household"
    sector: Optional[str] = None  # NAICS or custom taxonomy

    # Enhanced SFM power analysis fields
    power_resources: Dict[str, float] = field(default_factory=lambda: {})
    decision_making_capacity: Optional[float] = None
    institutional_affiliations: List[uuid.UUID] = field(default_factory=lambda: [])
    cognitive_frameworks: List[uuid.UUID] = field(default_factory=lambda: [])
    behavioral_patterns: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # Advanced stakeholder power analysis
    network_centrality: Optional[float] = None  # Network position centrality (0-1)
    coalition_memberships: List[uuid.UUID] = field(default_factory=lambda: [])
    influence_relationships: Dict[uuid.UUID, float] = field(default_factory=lambda: {})  # Actor -> influence level
    resource_dependencies: Dict[uuid.UUID, str] = field(default_factory=lambda: {})  # Actor -> dependency type
    bargaining_power: Optional[float] = None  # Relative bargaining strength (0-1)
    veto_power: List[str] = field(default_factory=lambda: [])  # Areas where actor has veto power
    agenda_setting_power: Optional[float] = None  # Ability to set agendas (0-1)
    
    # Power evolution and dynamics
    power_trajectory: List[Dict[str, float]] = field(default_factory=lambda: [])  # Historical power changes
    power_consolidation_strategies: List[str] = field(default_factory=lambda: [])
    power_distribution_preferences: Dict[str, float] = field(default_factory=lambda: {})
    
    # Legitimacy and authority
    legitimacy_sources: List[str] = field(default_factory=lambda: [])  # Sources of legitimacy
    authority_scope: List[str] = field(default_factory=lambda: [])  # Areas of recognized authority
    legitimacy_challenges: List[str] = field(default_factory=lambda: [])  # Threats to legitimacy
    
    def calculate_power_index(self) -> float:
        """Calculate overall power index based on power resources."""
        if not self.power_resources:
            return 0.0
        
        # Weight different power types
        weights = {
            'institutional_authority': 0.3,
            'economic_control': 0.25,
            'information_access': 0.2,
            'network_position': 0.15,
            'cultural_legitimacy': 0.1
        }
        
        weighted_sum = sum(
            self.power_resources.get(power_type, 0.0) * weight
            for power_type, weight in weights.items()
        )
        
        return min(weighted_sum, 1.0)  # Cap at 1.0
    
    def get_dominant_power_resource(self) -> Optional[str]:
        """Get the dominant power resource type for this actor."""
        if not self.power_resources:
            return None
        return max(self.power_resources.keys(), key=lambda x: self.power_resources[x])
    
    def assess_institutional_embeddedness(self) -> float:
        """Assess how embedded this actor is in institutional structures."""
        if not self.institutional_affiliations:
            return 0.0
        
        # More affiliations = higher embeddedness, but with diminishing returns
        affiliation_count = len(self.institutional_affiliations)
        return min(1.0, affiliation_count * 0.2)  # Cap at 1.0
    
    def analyze_actor_ci_orientation(self) -> Dict[str, Any]:
        """Analyze actor's ceremonial-instrumental orientation."""
        ci_orientation = {
            'ceremonial_tendencies': {},
            'instrumental_tendencies': {},
            'orientation_balance': 0.0,
            'influence_on_institutions': {},
            'transformation_role': {}
        }
        
        # Assess ceremonial tendencies based on power resources and behaviors
        ceremonial_indicators = 0.0
        
        # High status preservation tendencies
        if 'cultural_legitimacy' in self.power_resources and self.power_resources['cultural_legitimacy'] > 0.7:
            ceremonial_indicators += 0.3
        
        # Institutional authority seeking
        if 'institutional_authority' in self.power_resources and self.power_resources['institutional_authority'] > 0.8:
            ceremonial_indicators += 0.2
        
        # Power consolidation strategies
        if len(self.power_consolidation_strategies) > 3:
            ceremonial_indicators += 0.2
        
        # Assess instrumental tendencies
        instrumental_indicators = 0.0
        
        # Problem-solving orientation
        if 'information_access' in self.power_resources and self.power_resources['information_access'] > 0.6:
            instrumental_indicators += 0.3
        
        # Network-based influence (collaborative)
        if self.network_centrality and self.network_centrality > 0.7:
            instrumental_indicators += 0.3
        
        # Coalition participation (collaborative problem-solving)
        if len(self.coalition_memberships) > 2:
            instrumental_indicators += 0.2
        
        # Calculate orientation balance
        ci_orientation['ceremonial_tendencies'] = {
            'ceremonial_score': min(ceremonial_indicators, 1.0),
            'status_preservation_focus': ceremonial_indicators > 0.5,
            'hierarchy_maintenance': len(self.power_consolidation_strategies) > 2
        }
        
        ci_orientation['instrumental_tendencies'] = {
            'instrumental_score': min(instrumental_indicators, 1.0),
            'problem_solving_focus': instrumental_indicators > 0.5,
            'collaborative_orientation': len(self.coalition_memberships) > 1
        }
        
        ci_orientation['orientation_balance'] = instrumental_indicators - ceremonial_indicators
        
        return ci_orientation
    
    def assess_transformation_influence_capacity(self) -> Dict[str, Any]:
        """Assess actor's capacity to influence institutional transformation."""
        influence_capacity = {
            'transformation_enabler_potential': 0.0,
            'transformation_barrier_potential': 0.0,
            'influence_mechanisms': [],
            'transformation_role_classification': ''
        }
        
        # Calculate enabler potential
        enabler_factors = []
        
        if self.network_centrality and self.network_centrality > 0.6:
            enabler_factors.append(0.3)  # High network position
            influence_capacity['influence_mechanisms'].append("Network position leverage")
        
        if len(self.coalition_memberships) > 2:
            enabler_factors.append(0.2)  # Coalition building capacity
            influence_capacity['influence_mechanisms'].append("Coalition mobilization")
        
        if self.agenda_setting_power and self.agenda_setting_power > 0.6:
            enabler_factors.append(0.3)  # Agenda setting ability
            influence_capacity['influence_mechanisms'].append("Agenda setting")
        
        if self.legitimacy_sources and len(self.legitimacy_sources) > 2:
            enabler_factors.append(0.2)  # Multiple legitimacy sources
            influence_capacity['influence_mechanisms'].append("Legitimacy mobilization")
        
        influence_capacity['transformation_enabler_potential'] = min(sum(enabler_factors), 1.0)
        
        # Calculate barrier potential
        barrier_factors = []
        
        if len(self.veto_power) > 1:
            barrier_factors.append(0.4)  # Veto power can block change
        
        if self.bargaining_power and self.bargaining_power > 0.7:
            barrier_factors.append(0.3)  # Strong bargaining position
        
        if len(self.power_consolidation_strategies) > 3:
            barrier_factors.append(0.3)  # Strong consolidation focus
        
        influence_capacity['transformation_barrier_potential'] = min(sum(barrier_factors), 1.0)
        
        # Classify transformation role
        enabler_score = influence_capacity['transformation_enabler_potential']
        barrier_score = influence_capacity['transformation_barrier_potential']
        
        if enabler_score > 0.7 and barrier_score < 0.3:
            influence_capacity['transformation_role_classification'] = "Transformation Champion"
        elif barrier_score > 0.7 and enabler_score < 0.3:
            influence_capacity['transformation_role_classification'] = "Transformation Resistor"
        elif enabler_score > 0.5 and barrier_score > 0.5:
            influence_capacity['transformation_role_classification'] = "Pivotal Actor"
        elif enabler_score < 0.3 and barrier_score < 0.3:
            influence_capacity['transformation_role_classification'] = "Peripheral Actor"
        else:
            influence_capacity['transformation_role_classification'] = "Moderate Influencer"
        
        return influence_capacity
    
    def generate_actor_ci_engagement_strategy(self) -> Dict[str, List[str]]:
        """Generate strategy for engaging actor in CI transformation."""
        ci_orientation = self.analyze_actor_ci_orientation()
        influence_capacity = self.assess_transformation_influence_capacity()
        
        engagement_strategy = {
            'engagement_approach': [],
            'leverage_points': [],
            'risk_mitigation': [],
            'collaboration_opportunities': []
        }
        
        # Determine engagement approach based on CI orientation
        orientation_balance = ci_orientation['orientation_balance']
        
        if orientation_balance > 0.3:  # Instrumentally oriented
            engagement_strategy['engagement_approach'] = [
                "Leverage instrumental orientation for transformation leadership",
                "Provide platforms for problem-solving initiatives",
                "Support collaborative solution development"
            ]
        elif orientation_balance < -0.3:  # Ceremonially oriented
            engagement_strategy['engagement_approach'] = [
                "Address status and legitimacy concerns",
                "Demonstrate transformation benefits for institutional position",
                "Provide gradual adaptation pathways"
            ]
        else:  # Mixed orientation
            engagement_strategy['engagement_approach'] = [
                "Appeal to both instrumental and ceremonial motivations",
                "Create win-win transformation scenarios",
                "Build on existing collaborative relationships"
            ]
        
        # Identify leverage points based on power resources
        dominant_power = self.get_dominant_power_resource()
        if dominant_power:
            engagement_strategy['leverage_points'].append(f"Leverage {dominant_power} for transformation influence")
        
        if self.network_centrality and self.network_centrality > 0.6:
            engagement_strategy['leverage_points'].append("Utilize network position for transformation diffusion")
        
        # Risk mitigation based on transformation role
        transformation_role = influence_capacity['transformation_role_classification']
        if transformation_role == "Transformation Resistor":
            engagement_strategy['risk_mitigation'] = [
                "Address concerns about status and power impacts",
                "Provide guarantees for legitimate interests",
                "Create face-saving adaptation mechanisms"
            ]
        elif transformation_role == "Pivotal Actor":
            engagement_strategy['risk_mitigation'] = [
                "Carefully balance competing interests",
                "Provide clear incentives for transformation support",
                "Monitor for potential alliance shifts"
            ]
        
        return engagement_strategy


@dataclass
class Institution(Node):
    """Formal rules, informal norms, cultural practices."""

    formality_level: Optional[str] = None  # "formal" | "informal" | "mixed"
    scope: Optional[str] = None
    enforcement_mechanism: Optional[str] = None

    # Additional SFM-relevant fields
    rule_types: List[str] = field(default_factory=lambda: [])
    enforcement_strength: Optional[float] = None
    legitimacy_score: Optional[float] = None
    change_frequency: Optional[float] = None  # How often this institution changes
    institutional_complementarity: List[uuid.UUID] = field(default_factory=lambda: [])
    ceremonial_instrumental_balance: Optional[float] = None  # -1 (ceremonial) to +1 (instrumental)
    
    def calculate_institutional_effectiveness(self) -> float:
        """Calculate overall institutional effectiveness."""
        components: List[float] = []
        
        if self.enforcement_strength is not None:
            components.append(self.enforcement_strength * 0.4)
        
        if self.legitimacy_score is not None:
            components.append(self.legitimacy_score * 0.4)
        
        # Stability (inverse of change frequency)
        if self.change_frequency is not None:
            stability = max(0.0, 1.0 - self.change_frequency)
            components.append(stability * 0.2)
        
        return sum(components) / len(components) if components else 0.0
    
    def get_institutional_type_classification(self) -> str:
        """Classify institution based on ceremonial-instrumental balance."""
        if self.ceremonial_instrumental_balance is None:
            return "unclassified"
        
        if self.ceremonial_instrumental_balance < -0.5:
            return "predominantly_ceremonial"
        elif self.ceremonial_instrumental_balance > 0.5:
            return "predominantly_instrumental"
        else:
            return "mixed_ceremonial_instrumental"
    
    def assess_complementarity_strength(self) -> float:
        """Assess strength of institutional complementarity."""
        if not self.institutional_complementarity:
            return 0.0
        
        # More complementary institutions = stronger institutional framework
        complementarity_count = len(self.institutional_complementarity)
        return min(1.0, complementarity_count * 0.15)  # Cap at 1.0
    
    def conduct_integrated_ci_analysis(self) -> Dict[str, Any]:
        """Conduct integrated ceremonial-instrumental analysis of this institution."""
        from models.ceremonial_instrumental import CeremonialInstrumentalAnalysis, CIMeasurementFramework
        
        # Create CI analysis instance
        ci_analysis = CeremonialInstrumentalAnalysis(
            label=f"CI Analysis - {self.label}",
            analyzed_entity_id=self.id,
            ceremonial_score=max(0.0, -(self.ceremonial_instrumental_balance or 0.0)),
            instrumental_score=max(0.0, self.ceremonial_instrumental_balance or 0.0),
            dichotomy_balance=self.ceremonial_instrumental_balance
        )
        
        # Conduct systematic analysis
        systematic_results = ci_analysis.conduct_systematic_ci_analysis()
        
        # Create measurement framework for detailed metrics
        measurement_framework = CIMeasurementFramework(
            label=f"CI Measurement - {self.label}",
            measurement_scope="institution"
        )
        
        # Conduct comprehensive measurement
        measurement_results = measurement_framework.conduct_comprehensive_ci_measurement(self.id)
        
        # Integrate results
        integrated_results = {
            'institution_id': self.id,
            'institution_label': self.label,
            'systematic_ci_analysis': systematic_results,
            'detailed_measurements': measurement_results,
            'institutional_context': {
                'formality_level': self.formality_level,
                'enforcement_strength': self.enforcement_strength,
                'legitimacy_score': self.legitimacy_score,
                'change_frequency': self.change_frequency
            },
            'ci_integration_recommendations': self._generate_ci_integration_recommendations(systematic_results)
        }
        
        return integrated_results
    
    def assess_ci_transformation_readiness(self) -> Dict[str, Any]:
        """Assess readiness for ceremonial-instrumental transformation."""
        readiness_assessment = {
            'current_ci_position': self.get_institutional_type_classification(),
            'transformation_enablers': [],
            'transformation_barriers': [],
            'readiness_score': 0.0,
            'recommended_interventions': []
        }
        
        # Assess enablers based on institutional characteristics
        if self.legitimacy_score and self.legitimacy_score > 0.7:
            readiness_assessment['transformation_enablers'].append("High institutional legitimacy")
        
        if self.enforcement_strength and self.enforcement_strength > 0.6:
            readiness_assessment['transformation_enablers'].append("Strong enforcement mechanisms")
        
        if self.change_frequency and self.change_frequency > 0.3:
            readiness_assessment['transformation_enablers'].append("Adaptability to change")
        
        # Assess barriers
        if self.ceremonial_instrumental_balance and self.ceremonial_instrumental_balance < -0.5:
            readiness_assessment['transformation_barriers'].append("Strong ceremonial orientation")
        
        if self.change_frequency and self.change_frequency < 0.1:
            readiness_assessment['transformation_barriers'].append("Institutional rigidity")
        
        # Calculate readiness score
        enabler_score = len(readiness_assessment['transformation_enablers']) * 0.25
        barrier_penalty = len(readiness_assessment['transformation_barriers']) * 0.2
        base_readiness = (self.ceremonial_instrumental_balance or 0.0) + 0.5  # Normalize to 0-1
        
        readiness_assessment['readiness_score'] = max(0.0, min(1.0, base_readiness + enabler_score - barrier_penalty))
        
        # Generate recommendations
        if readiness_assessment['readiness_score'] < 0.4:
            readiness_assessment['recommended_interventions'] = [
                "Address ceremonial barriers through stakeholder engagement",
                "Build instrumental capacity gradually",
                "Create pilot programs to demonstrate instrumental effectiveness"
            ]
        elif readiness_assessment['readiness_score'] > 0.7:
            readiness_assessment['recommended_interventions'] = [
                "Accelerate instrumental transformation initiatives",
                "Leverage high readiness for system-wide change",
                "Share transformation successes with other institutions"
            ]
        
        return readiness_assessment
    
    def integrate_with_matrix_ci_analysis(self, matrix_ci_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate institution's CI analysis with broader matrix CI patterns."""
        integration_analysis = {
            'institutional_ci_contribution': {},
            'matrix_alignment_assessment': {},
            'spillover_effects': {},
            'coordination_opportunities': {}
        }
        
        # Assess contribution to matrix CI patterns
        institution_ci_balance = self.ceremonial_instrumental_balance or 0.0
        matrix_avg_ci = matrix_ci_data.get('average_ci_balance', 0.0)
        
        integration_analysis['institutional_ci_contribution'] = {
            'relative_to_matrix_average': institution_ci_balance - matrix_avg_ci,
            'influence_direction': 'instrumental' if institution_ci_balance > matrix_avg_ci else 'ceremonial',
            'influence_magnitude': abs(institution_ci_balance - matrix_avg_ci)
        }
        
        # Matrix alignment assessment
        alignment_score = 1.0 - abs(institution_ci_balance - matrix_avg_ci)
        integration_analysis['matrix_alignment_assessment'] = {
            'alignment_score': alignment_score,
            'alignment_quality': 'High' if alignment_score > 0.8 else 'Moderate' if alignment_score > 0.5 else 'Low',
            'requires_coordination': alignment_score < 0.5
        }
        
        # Spillover effects analysis
        integration_analysis['spillover_effects'] = {
            'potential_positive_spillovers': self._identify_positive_spillovers(matrix_ci_data),
            'potential_negative_spillovers': self._identify_negative_spillovers(matrix_ci_data),
            'spillover_management_strategies': self._suggest_spillover_management()
        }
        
        return integration_analysis
    
    def _generate_ci_integration_recommendations(self, systematic_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for CI integration."""
        recommendations = []
        
        # Based on transformation readiness
        transformation_readiness = systematic_results.get('dichotomy_assessment', {}).get('transformation_readiness', 0.5)
        
        if transformation_readiness < 0.4:
            recommendations.extend([
                "Develop instrumental capacity building programs",
                "Address ceremonial resistance through stakeholder engagement",
                "Create demonstration projects to show instrumental effectiveness"
            ])
        elif transformation_readiness > 0.7:
            recommendations.extend([
                "Accelerate institutional transformation initiatives",
                "Leverage transformation capacity for system-wide change",
                "Develop institutional transformation leadership"
            ])
        
        # Based on institutional characteristics
        if self.legitimacy_score and self.legitimacy_score < 0.5:
            recommendations.append("Build institutional legitimacy through transparent processes")
        
        if self.enforcement_strength and self.enforcement_strength < 0.4:
            recommendations.append("Strengthen enforcement mechanisms for institutional effectiveness")
        
        return recommendations
    
    def _identify_positive_spillovers(self, matrix_ci_data: Dict[str, Any]) -> List[str]:
        """Identify potential positive CI spillovers to other institutions."""
        spillovers = []
        
        if self.ceremonial_instrumental_balance and self.ceremonial_instrumental_balance > 0.5:
            spillovers.extend([
                "Instrumental innovation diffusion to connected institutions",
                "Problem-solving methodology transfer",
                "Efficiency enhancement spillovers"
            ])
        
        if self.legitimacy_score and self.legitimacy_score > 0.7:
            spillovers.append("Legitimacy enhancement for institutional transformation")
        
        return spillovers
    
    def _identify_negative_spillovers(self, matrix_ci_data: Dict[str, Any]) -> List[str]:
        """Identify potential negative CI spillovers."""
        spillovers = []
        
        if self.ceremonial_instrumental_balance and self.ceremonial_instrumental_balance < -0.5:
            spillovers.extend([
                "Ceremonial resistance spreading to other institutions",
                "Status quo preservation pressure",
                "Innovation resistance diffusion"
            ])
        
        return spillovers
    
    def _suggest_spillover_management(self) -> List[str]:
        """Suggest strategies for managing CI spillovers."""
        return [
            "Create institutional coordination mechanisms",
            "Develop cross-institutional learning processes",
            "Establish CI transformation monitoring systems",
            "Build institutional change leadership networks"
        ]


@dataclass
class Policy(Institution):
    """Specific policy intervention or regulatory framework."""

    authority: Optional[str] = None  # Implementing body
    enforcement: Optional[float] = 0.0  # Strength of enforcement (0-1)
    target_sectors: List[str] = field(default_factory=lambda: [])
    
    # SFM integration additions:
    target_matrix_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # Cells policy aims to improve
    effectiveness_evidence: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to evidence
    unintended_consequences: Dict[str, str] = field(default_factory=lambda: {})  # Unexpected matrix effects
    ceremonial_aspects: Optional[float] = None  # How much of policy is ceremonial (0-1)
    problem_solving_sequence: Optional[uuid.UUID] = None  # Link to ProblemSolvingSequence
    policy_instruments: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to PolicyInstrument
    
    def evaluate_policy_ci_characteristics(self) -> Dict[str, Any]:
        """Evaluate ceremonial-instrumental characteristics of this policy."""
        policy_ci_evaluation = {
            'ceremonial_elements': {},
            'instrumental_elements': {},
            'policy_ci_balance': 0.0,
            'effectiveness_assessment': {},
            'transformation_potential': {}
        }
        
        # Assess ceremonial elements
        ceremonial_score = self.ceremonial_aspects or 0.0
        policy_ci_evaluation['ceremonial_elements'] = {
            'ceremonial_proportion': ceremonial_score,
            'symbolic_aspects': ceremonial_score > 0.3,
            'status_quo_reinforcement': ceremonial_score > 0.5,
            'ritual_compliance_focus': ceremonial_score > 0.4
        }
        
        # Assess instrumental elements
        instrumental_score = 1.0 - ceremonial_score
        policy_ci_evaluation['instrumental_elements'] = {
            'instrumental_proportion': instrumental_score,
            'problem_solving_focus': len(self.target_matrix_cells) > 0,
            'evidence_based_design': len(self.effectiveness_evidence) > 0,
            'adaptive_mechanisms': self.change_frequency and self.change_frequency > 0.2
        }
        
        # Calculate policy CI balance
        policy_ci_evaluation['policy_ci_balance'] = instrumental_score - ceremonial_score
        
        # Effectiveness assessment
        policy_ci_evaluation['effectiveness_assessment'] = {
            'enforcement_strength': self.enforcement or 0.0,
            'target_specificity': len(self.target_matrix_cells) / 10.0,  # Normalize
            'evidence_support': len(self.effectiveness_evidence) / 5.0,  # Normalize
            'unintended_consequences_managed': len(self.unintended_consequences) < 3
        }
        
        return policy_ci_evaluation
    
    def assess_policy_transformation_impact(self) -> Dict[str, Any]:
        """Assess policy's impact on institutional transformation."""
        transformation_impact = {
            'direct_transformation_effects': {},
            'indirect_spillover_effects': {},
            'resistance_mitigation': {},
            'enabler_enhancement': {}
        }
        
        # Direct transformation effects
        ci_balance = (1.0 - (self.ceremonial_aspects or 0.0)) - (self.ceremonial_aspects or 0.0)
        transformation_impact['direct_transformation_effects'] = {
            'instrumental_promotion': ci_balance > 0.3,
            'ceremonial_reduction': ci_balance > 0.5,
            'problem_solving_enhancement': len(self.target_matrix_cells) > 2,
            'innovation_support': self.enforcement and self.enforcement > 0.6
        }
        
        # Policy effectiveness for transformation
        effectiveness_score = (
            (self.enforcement or 0.0) * 0.4 +
            (self.legitimacy_score or 0.5) * 0.3 +
            (len(self.effectiveness_evidence) / 5.0) * 0.3
        )
        
        transformation_impact['transformation_effectiveness'] = {
            'effectiveness_score': min(effectiveness_score, 1.0),
            'high_impact_potential': effectiveness_score > 0.7,
            'moderate_impact_potential': 0.4 < effectiveness_score <= 0.7,
            'limited_impact_potential': effectiveness_score <= 0.4
        }
        
        return transformation_impact
    
    def generate_policy_ci_recommendations(self) -> Dict[str, List[str]]:
        """Generate CI-based policy improvement recommendations."""
        recommendations = {
            'ceremonial_reduction': [],
            'instrumental_enhancement': [],
            'implementation_improvements': [],
            'monitoring_enhancements': []
        }
        
        # Ceremonial reduction recommendations
        if self.ceremonial_aspects and self.ceremonial_aspects > 0.5:
            recommendations['ceremonial_reduction'] = [
                "Reduce symbolic elements in favor of substantive action",
                "Focus on outcomes rather than compliance procedures",
                "Eliminate redundant reporting requirements",
                "Streamline bureaucratic processes"
            ]
        
        # Instrumental enhancement recommendations
        if len(self.effectiveness_evidence) < 3:
            recommendations['instrumental_enhancement'].append(
                "Strengthen evidence base for policy effectiveness"
            )
        
        if len(self.target_matrix_cells) < 2:
            recommendations['instrumental_enhancement'].append(
                "Specify clearer target outcomes and matrix effects"
            )
        
        # Implementation improvements
        if self.enforcement and self.enforcement < 0.5:
            recommendations['implementation_improvements'] = [
                "Strengthen enforcement mechanisms",
                "Clarify implementation responsibilities",
                "Improve resource allocation for implementation"
            ]
        
        # Monitoring enhancements
        recommendations['monitoring_enhancements'] = [
            "Establish systematic CI monitoring processes",
            "Track both intended and unintended consequences",
            "Create feedback loops for policy adaptation",
            "Develop CI transformation indicators"
        ]
        
        return recommendations


@dataclass
class Resource(Node):
    """Stock or asset available for use or transformation."""

    rtype: ResourceType = ResourceType.NATURAL
    unit: Optional[str] = None  # e.g. "tonnes", "person-hours"


@dataclass
class Process(Node):
    """
    Transformation activity that converts inputs to outputs
    (production, consumption, disposal).
    """

    technology: Optional[str] = None  # e.g. "EAF-Steel-2024"
    responsible_actor_id: Optional[str] = None  # Actor that controls the process


@dataclass
class Flow(Node):  # pylint: disable=too-many-instance-attributes
    """Edge-like node representing an actual quantified transfer of resources or value."""

    nature: FlowNature = FlowNature.TRANSFER
    quantity: Optional[float] = None
    unit: Optional[str] = None
    time: Optional[TimeSlice] = None
    space: Optional[SpatialUnit] = None
    scenario: Optional[Scenario] = None

    # Additional SFM-specific fields
    flow_type: FlowType = FlowType.MATERIAL  # material, energy, information, financial, social
    source_process_id: Optional[uuid.UUID] = None
    target_process_id: Optional[uuid.UUID] = None
    transformation_coefficient: Optional[float] = None
    loss_factor: Optional[float] = None  # inefficiencies, waste

    # Hayden's value theory integration
    ceremonial_component: Optional[float] = None
    instrumental_component: Optional[float] = None
    temporal_dynamics: Optional[TemporalDynamics] = None  # Change over time
    
    # SFM Matrix integration additions:
    affecting_matrix_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # Which cells this flow affects
    institutional_constraints: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions that constrain flow
    technology_dependencies: List[uuid.UUID] = field(default_factory=lambda: [])  # Required technologies
    ceremonial_barriers: List[str] = field(default_factory=lambda: [])  # Ceremonial obstacles to flow
    instrumental_enablers: List[str] = field(default_factory=lambda: [])  # Instrumental flow facilitators

    def __post_init__(self) -> None:
        """Validate flow nature and type combination after initialization."""
        # Validate flow nature and type combination
        EnumValidator.validate_flow_combination(self.nature, self.flow_type)


@dataclass
class ValueFlow(Flow):
    """Tracks value creation, capture, and distribution."""

    value_created: Optional[float] = None
    value_captured: Optional[float] = None
    beneficiary_actors: List[uuid.UUID] = field(default_factory=lambda: [])
    distributional_impact: Dict[str, float] = field(default_factory=lambda: {})


@dataclass
class GovernanceStructure(Institution):
    """Formal and informal governance arrangements."""

    decision_making_process: Optional[str] = None
    power_distribution: Dict[str, float] = field(default_factory=lambda: {})
    accountability_mechanisms: List[str] = field(default_factory=lambda: [])
    
    # Enhanced SFM integration:
    governance_effectiveness: Optional[float] = None  # How well governance works (0-1)
    participatory_mechanisms: List[str] = field(default_factory=lambda: [])  # How stakeholders participate
    transparency_level: Optional[float] = None  # Level of transparency (0-1)
    conflict_resolution: List[str] = field(default_factory=lambda: [])  # How conflicts are resolved
    adaptive_capacity: Optional[float] = None  # Ability to adapt to change (0-1)
