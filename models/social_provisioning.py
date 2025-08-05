"""
Social Provisioning Process for comprehensive lifecycle analysis in SFM.

This module implements Hayden's concept of social provisioning as the central
organizing principle for institutional analysis. It models the complete lifecycle
of social provisioning from need identification through delivery and evaluation,
integrating with the broader SFM framework.

Key Components:
- ProvisioningProcess: Complete social provisioning lifecycle
- ProvisioningStage: Individual stages in the provisioning process
- ProvisioningNetwork: Networks of interconnected provisioning processes
- ProvisioningEffectiveness: Assessment of provisioning outcomes
- ProvisioningCoordination: Cross-institutional coordination mechanisms
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.sfm_enums import (
    ProvisioningStage,
    ValueCategory,
    ResourceType,
    FlowType,
    SystemLevel,
    InstitutionalScope,
    DeliveryQuantificationMethod,
)


class ProvisioningType(Enum):
    """Types of social provisioning processes."""
    
    BASIC_NEEDS = auto()          # Essential human needs
    PUBLIC_SERVICES = auto()      # Government-provided services
    MARKET_GOODS = auto()         # Market-based provisioning
    COMMUNITY_SERVICES = auto()   # Community-based provisioning
    INFRASTRUCTURE = auto()       # Infrastructure provisioning
    KNOWLEDGE_SERVICES = auto()   # Education and information
    CARE_SERVICES = auto()        # Health and social care
    ENVIRONMENTAL_SERVICES = auto() # Environmental goods/services


class ProvisioningMode(Enum):
    """Modes of social provisioning."""
    
    INSTITUTIONAL = auto()        # Through formal institutions
    MARKET = auto()              # Through market mechanisms
    COMMUNITY = auto()           # Through community networks
    HYBRID = auto()              # Mixed approaches
    SELF_PROVISIONING = auto()   # Individual/household provision


class ProvisioningQuality(Enum):
    """Quality levels of provisioning outcomes."""
    
    EXCELLENT = auto()           # Exceeds standards
    GOOD = auto()               # Meets standards well
    ADEQUATE = auto()           # Basic standards met
    INADEQUATE = auto()         # Below standards
    POOR = auto()               # Significantly deficient


class CoordinationMechanism(Enum):
    """Mechanisms for provisioning coordination."""
    
    HIERARCHICAL = auto()        # Top-down coordination
    NETWORK = auto()            # Network-based coordination
    MARKET = auto()             # Market-based coordination
    COLLABORATIVE = auto()      # Collaborative coordination
    EMERGENT = auto()           # Self-organizing coordination


@dataclass
class ProvisioningNeed(Node):
    """Identified need requiring social provisioning."""
    
    need_type: Optional[str] = None
    need_category: ValueCategory = ValueCategory.SOCIAL
    
    # Need characterization
    need_description: Optional[str] = None
    target_population: Optional[str] = None
    need_intensity: Optional[float] = None  # 0-1 scale
    need_urgency: Optional[float] = None   # 0-1 scale
    
    # Need scope
    affected_population_size: Optional[int] = None
    geographic_scope: Optional[SpatialUnit] = None
    temporal_scope: Optional[TimeSlice] = None
    
    # Need assessment
    need_evidence: List[str] = field(default_factory=list)
    need_measurement: Dict[str, float] = field(default_factory=dict)
    need_validation: List[str] = field(default_factory=list)
    
    # Stakeholder perspectives
    stakeholder_need_assessment: Dict[uuid.UUID, float] = field(default_factory=dict)
    need_priority_ranking: Dict[uuid.UUID, int] = field(default_factory=dict)
    
    # Need relationships
    related_needs: List[uuid.UUID] = field(default_factory=list)
    prerequisite_needs: List[uuid.UUID] = field(default_factory=list)
    competing_needs: List[uuid.UUID] = field(default_factory=list)
    
    # Current provisioning status
    current_provisioning_level: Optional[float] = None  # 0-1 scale
    provisioning_gap: Optional[float] = None           # Unmet need level
    existing_provisioning_sources: List[uuid.UUID] = field(default_factory=list)
    
    def calculate_provisioning_gap(self) -> Optional[float]:
        """Calculate the gap between need and current provisioning."""
        if (self.need_intensity is not None and 
            self.current_provisioning_level is not None):
            gap = self.need_intensity - self.current_provisioning_level
            self.provisioning_gap = max(0.0, gap)
            return self.provisioning_gap
        return None
    
    def assess_need_priority(self) -> Dict[str, Any]:
        """Assess priority of this need for provisioning attention."""
        priority_assessment = {
            'urgency_score': self.need_urgency or 0.5,
            'intensity_score': self.need_intensity or 0.5,
            'gap_score': self.provisioning_gap or 0.5,
            'population_impact': 0.0,
            'overall_priority': 0.0
        }
        
        # Population impact assessment
        if self.affected_population_size:
            # Normalize population impact (simplified)
            population_impact = min(self.affected_population_size / 10000, 1.0)
            priority_assessment['population_impact'] = population_impact
        
        # Overall priority calculation
        priority_factors = [
            priority_assessment['urgency_score'] * 0.3,
            priority_assessment['intensity_score'] * 0.25,
            priority_assessment['gap_score'] * 0.25,
            priority_assessment['population_impact'] * 0.2
        ]
        
        priority_assessment['overall_priority'] = sum(priority_factors)
        
        return priority_assessment


@dataclass
class ProvisioningStageImplementation(Node):
    """Implementation of a specific stage in the provisioning process."""
    
    stage_type: ProvisioningStage = ProvisioningStage.PRODUCTION
    stage_sequence: Optional[int] = None  # Order in process
    
    # Stage implementation
    responsible_institutions: List[uuid.UUID] = field(default_factory=list)
    stage_activities: List[str] = field(default_factory=list)
    stage_inputs: List[uuid.UUID] = field(default_factory=list)  # Required resources/flows
    stage_outputs: List[uuid.UUID] = field(default_factory=list) # Produced resources/flows
    
    # Stage performance
    stage_duration: Optional[timedelta] = None
    stage_cost: Optional[float] = None
    stage_effectiveness: Optional[float] = None  # 0-1 scale
    stage_efficiency: Optional[float] = None    # 0-1 scale
    
    # Stage coordination
    coordination_mechanisms: List[CoordinationMechanism] = field(default_factory=list)
    coordination_challenges: List[str] = field(default_factory=list)
    coordination_effectiveness: Optional[float] = None  # 0-1 scale
    
    # Stage dependencies
    prerequisite_stages: List[uuid.UUID] = field(default_factory=list)
    dependent_stages: List[uuid.UUID] = field(default_factory=list)
    parallel_stages: List[uuid.UUID] = field(default_factory=list)
    
    # Stage quality
    quality_standards: Dict[str, Any] = field(default_factory=dict)
    quality_assessment: Dict[str, float] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)
    
    # Stakeholder involvement
    stage_stakeholders: List[uuid.UUID] = field(default_factory=list)
    stakeholder_roles: Dict[uuid.UUID, str] = field(default_factory=dict)
    stakeholder_satisfaction: Dict[uuid.UUID, float] = field(default_factory=dict)
    
    def assess_stage_performance(self) -> Dict[str, Any]:
        """Assess performance of this provisioning stage."""
        performance_assessment = {
            'effectiveness_score': self.stage_effectiveness or 0.5,
            'efficiency_score': self.stage_efficiency or 0.5,
            'coordination_score': self.coordination_effectiveness or 0.5,
            'quality_score': 0.0,
            'stakeholder_score': 0.0,
            'overall_performance': 0.0,
            'performance_issues': [],
            'improvement_opportunities': []
        }
        
        # Quality score calculation
        if self.quality_assessment:
            quality_scores = list(self.quality_assessment.values())
            performance_assessment['quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # Stakeholder satisfaction score
        if self.stakeholder_satisfaction:
            satisfaction_scores = list(self.stakeholder_satisfaction.values())
            performance_assessment['stakeholder_score'] = sum(satisfaction_scores) / len(satisfaction_scores)
        
        # Overall performance
        performance_factors = [
            performance_assessment['effectiveness_score'] * 0.3,
            performance_assessment['efficiency_score'] * 0.2,
            performance_assessment['coordination_score'] * 0.2,
            performance_assessment['quality_score'] * 0.15,
            performance_assessment['stakeholder_score'] * 0.15
        ]
        
        performance_assessment['overall_performance'] = sum(performance_factors)
        
        # Identify performance issues
        if performance_assessment['effectiveness_score'] < 0.6:
            performance_assessment['performance_issues'].append('Low effectiveness')
        if performance_assessment['coordination_score'] < 0.5:
            performance_assessment['performance_issues'].append('Coordination problems')
        if self.quality_issues:
            performance_assessment['performance_issues'].extend(self.quality_issues)
        
        # Generate improvement opportunities
        if performance_assessment['efficiency_score'] < 0.7:
            performance_assessment['improvement_opportunities'].append('Efficiency improvements needed')
        if performance_assessment['stakeholder_score'] < 0.6:
            performance_assessment['improvement_opportunities'].append('Enhanced stakeholder engagement')
        
        return performance_assessment


@dataclass
class ProvisioningProcess(Node):
    """Complete social provisioning process from need to outcome."""
    
    provisioning_type: ProvisioningType = ProvisioningType.PUBLIC_SERVICES
    provisioning_mode: ProvisioningMode = ProvisioningMode.INSTITUTIONAL
    
    # Process scope
    target_need: Optional[uuid.UUID] = None  # ProvisioningNeed being addressed
    target_population: Optional[str] = None
    geographic_scope: Optional[SpatialUnit] = None
    temporal_scope: Optional[TimeSlice] = None
    
    # Process structure
    provisioning_stages: List[uuid.UUID] = field(default_factory=list)  # ProvisioningStageImplementation IDs
    stage_sequence: List[Tuple[uuid.UUID, int]] = field(default_factory=list)  # Stage ID, sequence number
    process_flows: List[uuid.UUID] = field(default_factory=list)  # Resource/value flows
    
    # Process participants
    lead_institution: Optional[uuid.UUID] = None
    participating_institutions: List[uuid.UUID] = field(default_factory=list)
    process_stakeholders: List[uuid.UUID] = field(default_factory=list)
    beneficiary_groups: List[str] = field(default_factory=list)
    
    # Process resources
    required_resources: Dict[ResourceType, float] = field(default_factory=dict)
    resource_sources: Dict[uuid.UUID, str] = field(default_factory=dict)  # Institution -> resource type
    resource_constraints: List[str] = field(default_factory=list)
    
    # Process performance
    process_effectiveness: Optional[float] = None  # 0-1 scale
    process_efficiency: Optional[float] = None     # 0-1 scale
    process_equity: Optional[float] = None         # Fairness of access/outcomes
    process_sustainability: Optional[float] = None # Long-term viability
    
    # Process outcomes
    provisioning_outcomes: Dict[str, Any] = field(default_factory=dict)
    outcome_indicators: List[str] = field(default_factory=list)
    unintended_consequences: List[str] = field(default_factory=list)
    
    # Process quality
    quality_level: ProvisioningQuality = ProvisioningQuality.ADEQUATE
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    quality_assurance_mechanisms: List[str] = field(default_factory=list)
    
    # Process coordination
    coordination_mechanisms: List[CoordinationMechanism] = field(default_factory=list)
    coordination_challenges: List[str] = field(default_factory=list)
    coordination_effectiveness: Optional[float] = None
    
    # Process learning and adaptation
    process_feedback_loops: List[str] = field(default_factory=list)
    adaptation_mechanisms: List[str] = field(default_factory=list)
    learning_outcomes: List[str] = field(default_factory=list)
    
    def assess_process_effectiveness(self) -> Dict[str, Any]:
        """Assess overall effectiveness of the provisioning process."""
        effectiveness_assessment = {
            'need_fulfillment': 0.0,
            'stakeholder_satisfaction': 0.0,
            'resource_efficiency': 0.0,
            'outcome_quality': 0.0,
            'coordination_effectiveness': self.coordination_effectiveness or 0.5,
            'overall_effectiveness': 0.0,
            'effectiveness_drivers': [],
            'effectiveness_barriers': []
        }
        
        # Need fulfillment assessment
        # Would need to compare against target need - simplified here
        if self.process_effectiveness:
            effectiveness_assessment['need_fulfillment'] = self.process_effectiveness
        
        # Resource efficiency
        if self.process_efficiency:
            effectiveness_assessment['resource_efficiency'] = self.process_efficiency
        
        # Outcome quality based on quality metrics
        if self.quality_metrics:
            quality_scores = list(self.quality_metrics.values())
            effectiveness_assessment['outcome_quality'] = sum(quality_scores) / len(quality_scores)
        
        # Overall effectiveness calculation
        effectiveness_factors = [
            effectiveness_assessment['need_fulfillment'] * 0.3,
            effectiveness_assessment['resource_efficiency'] * 0.2,
            effectiveness_assessment['outcome_quality'] * 0.25,
            effectiveness_assessment['coordination_effectiveness'] * 0.25
        ]
        
        effectiveness_assessment['overall_effectiveness'] = sum(effectiveness_factors)
        self.process_effectiveness = effectiveness_assessment['overall_effectiveness']
        
        # Identify drivers and barriers
        if effectiveness_assessment['coordination_effectiveness'] > 0.7:
            effectiveness_assessment['effectiveness_drivers'].append('Strong coordination')
        if effectiveness_assessment['resource_efficiency'] < 0.5:
            effectiveness_assessment['effectiveness_barriers'].append('Resource inefficiency')
        if self.coordination_challenges:
            effectiveness_assessment['effectiveness_barriers'].extend(self.coordination_challenges)
        
        return effectiveness_assessment
    
    def analyze_process_bottlenecks(self) -> Dict[str, Any]:
        """Analyze bottlenecks and constraints in the provisioning process."""
        bottleneck_analysis = {
            'stage_bottlenecks': [],
            'resource_bottlenecks': [],
            'coordination_bottlenecks': [],
            'capacity_bottlenecks': [],
            'critical_path_stages': [],
            'bottleneck_impact': {},
            'resolution_priorities': []
        }
        
        # Resource bottlenecks
        for resource_type, required_amount in self.required_resources.items():
            if required_amount > 1000:  # Simplified threshold
                bottleneck_analysis['resource_bottlenecks'].append({
                    'resource_type': resource_type.name,
                    'required_amount': required_amount,
                    'bottleneck_severity': 'high' if required_amount > 5000 else 'moderate'
                })
        
        # Coordination bottlenecks from challenges
        if self.coordination_challenges:
            bottleneck_analysis['coordination_bottlenecks'] = self.coordination_challenges.copy()
        
        # Generate resolution priorities
        if bottleneck_analysis['resource_bottlenecks']:
            bottleneck_analysis['resolution_priorities'].append('Address resource constraints')
        if bottleneck_analysis['coordination_bottlenecks']:
            bottleneck_analysis['resolution_priorities'].append('Improve coordination mechanisms')
        
        return bottleneck_analysis
    
    def generate_improvement_plan(self) -> Dict[str, Any]:
        """Generate comprehensive improvement plan for the provisioning process."""
        improvement_plan = {
            'improvement_areas': [],
            'specific_actions': [],
            'resource_requirements': [],
            'implementation_timeline': {},
            'success_indicators': [],
            'risk_mitigation': []
        }
        
        # Identify improvement areas based on performance
        if self.process_effectiveness and self.process_effectiveness < 0.7:
            improvement_plan['improvement_areas'].append('Process effectiveness')
        if self.process_efficiency and self.process_efficiency < 0.6:
            improvement_plan['improvement_areas'].append('Resource efficiency')
        if self.coordination_effectiveness and self.coordination_effectiveness < 0.6:
            improvement_plan['improvement_areas'].append('Coordination mechanisms')
        
        # Generate specific actions
        for area in improvement_plan['improvement_areas']:
            if area == 'Process effectiveness':
                improvement_plan['specific_actions'].extend([
                    'Review and redesign underperforming stages',
                    'Strengthen outcome measurement and feedback',
                    'Enhance stakeholder engagement processes'
                ])
            elif area == 'Resource efficiency':
                improvement_plan['specific_actions'].extend([
                    'Conduct resource utilization analysis',
                    'Implement resource optimization measures',
                    'Explore resource sharing opportunities'
                ])
            elif area == 'Coordination mechanisms':
                improvement_plan['specific_actions'].extend([
                    'Strengthen inter-institutional coordination',
                    'Implement coordination protocols',
                    'Establish regular coordination meetings'
                ])
        
        # Success indicators
        improvement_plan['success_indicators'] = [
            'Increased process effectiveness score',
            'Improved resource efficiency metrics',
            'Enhanced stakeholder satisfaction',
            'Reduced coordination challenges'
        ]
        
        return improvement_plan


@dataclass
class ProvisioningNetwork(Node):
    """Network of interconnected provisioning processes."""
    
    network_scope: Optional[str] = None
    network_type: Optional[str] = None  # Sectoral, geographic, functional
    
    # Network structure
    provisioning_processes: List[uuid.UUID] = field(default_factory=list)
    process_relationships: Dict[Tuple[uuid.UUID, uuid.UUID], str] = field(default_factory=dict)
    network_flows: List[uuid.UUID] = field(default_factory=list)
    
    # Network properties
    network_density: Optional[float] = None        # Interconnection density
    network_centralization: Optional[float] = None # Centralization degree
    network_efficiency: Optional[float] = None     # Overall efficiency
    network_resilience: Optional[float] = None     # Resilience to disruption
    
    # Network coordination
    network_governance: List[str] = field(default_factory=list)
    coordination_institutions: List[uuid.UUID] = field(default_factory=list)
    network_standards: Dict[str, Any] = field(default_factory=dict)
    
    # Network performance
    aggregate_outcomes: Dict[str, float] = field(default_factory=dict)
    network_synergies: List[str] = field(default_factory=list)
    network_conflicts: List[str] = field(default_factory=list)
    
    # Network evolution
    network_adaptation_capacity: Optional[float] = None
    network_learning_mechanisms: List[str] = field(default_factory=list)
    network_innovation_capacity: Optional[float] = None
    
    def analyze_network_performance(self) -> Dict[str, Any]:
        """Analyze overall performance of the provisioning network."""
        network_analysis = {
            'performance_overview': {},
            'synergy_analysis': {},
            'conflict_analysis': {},
            'coordination_assessment': {},
            'improvement_opportunities': []
        }
        
        # Performance overview
        if self.aggregate_outcomes:
            avg_performance = sum(self.aggregate_outcomes.values()) / len(self.aggregate_outcomes)
            network_analysis['performance_overview'] = {
                'average_performance': avg_performance,
                'performance_range': (min(self.aggregate_outcomes.values()), 
                                    max(self.aggregate_outcomes.values())),
                'performance_consistency': 1.0 - (max(self.aggregate_outcomes.values()) - 
                                                min(self.aggregate_outcomes.values()))
            }
        
        # Synergy analysis
        if self.network_synergies:
            network_analysis['synergy_analysis'] = {
                'synergy_count': len(self.network_synergies),
                'synergy_types': self.network_synergies.copy(),
                'synergy_strength': 'high' if len(self.network_synergies) > 5 else 'moderate'
            }
        
        # Conflict analysis
        if self.network_conflicts:
            network_analysis['conflict_analysis'] = {
                'conflict_count': len(self.network_conflicts),
                'conflict_types': self.network_conflicts.copy(),
                'conflict_severity': 'high' if len(self.network_conflicts) > 3 else 'moderate'
            }
        
        # Generate improvement opportunities
        if network_analysis.get('conflict_analysis', {}).get('conflict_count', 0) > 2:
            network_analysis['improvement_opportunities'].append('Conflict resolution mechanisms needed')
        if self.network_efficiency and self.network_efficiency < 0.6:
            network_analysis['improvement_opportunities'].append('Network efficiency improvements')
        
        return network_analysis
    
    def assess_network_resilience(self) -> Dict[str, Any]:
        """Assess resilience of the provisioning network."""
        resilience_assessment = {
            'structural_resilience': 0.0,
            'functional_resilience': 0.0,
            'adaptive_resilience': 0.0,
            'overall_resilience': 0.0,
            'vulnerability_points': [],
            'resilience_strategies': []
        }
        
        # Structural resilience based on network properties
        if self.network_density:
            resilience_assessment['structural_resilience'] = self.network_density
        
        # Adaptive resilience
        if self.network_adaptation_capacity:
            resilience_assessment['adaptive_resilience'] = self.network_adaptation_capacity
        
        # Overall resilience
        resilience_factors = [
            resilience_assessment['structural_resilience'],
            resilience_assessment['adaptive_resilience']
        ]
        valid_factors = [f for f in resilience_factors if f > 0]
        if valid_factors:
            resilience_assessment['overall_resilience'] = sum(valid_factors) / len(valid_factors)
            self.network_resilience = resilience_assessment['overall_resilience']
        
        # Identify vulnerability points
        if resilience_assessment['structural_resilience'] < 0.5:
            resilience_assessment['vulnerability_points'].append('Low network connectivity')
        if not self.network_learning_mechanisms:
            resilience_assessment['vulnerability_points'].append('Limited learning mechanisms')
        
        # Generate resilience strategies
        resilience_assessment['resilience_strategies'] = [
            'Strengthen network redundancy',
            'Enhance coordination mechanisms',
            'Build adaptive capacity',
            'Develop contingency plans'
        ]
        
        return resilience_assessment


@dataclass
class ProvisioningEffectiveness(Node):
    """Assessment of provisioning process effectiveness and outcomes."""
    
    assessment_scope: Optional[str] = None
    assessment_timeframe: Optional[TimeSlice] = None
    
    # Assessment targets
    assessed_processes: List[uuid.UUID] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)
    assessment_stakeholders: List[uuid.UUID] = field(default_factory=list)
    
    # Effectiveness metrics
    outcome_achievement: Dict[str, float] = field(default_factory=dict)  # Outcome -> achievement level
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)   # Efficiency measures
    equity_metrics: Dict[str, float] = field(default_factory=dict)       # Equity measures
    sustainability_metrics: Dict[str, float] = field(default_factory=dict) # Sustainability measures
    
    # Stakeholder assessment
    stakeholder_satisfaction: Dict[uuid.UUID, float] = field(default_factory=dict)
    beneficiary_outcomes: Dict[str, Any] = field(default_factory=dict)
    provider_assessment: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    
    # Comparative analysis
    benchmark_comparisons: Dict[str, float] = field(default_factory=dict)
    best_practice_identification: List[str] = field(default_factory=list)
    performance_gaps: Dict[str, float] = field(default_factory=dict)
    
    # Impact assessment
    direct_impacts: Dict[str, Any] = field(default_factory=dict)
    indirect_impacts: Dict[str, Any] = field(default_factory=dict)
    long_term_impacts: Dict[str, Any] = field(default_factory=dict)
    unintended_impacts: List[str] = field(default_factory=list)
    
    # Assessment quality
    assessment_reliability: Optional[float] = None
    assessment_validity: Optional[float] = None
    assessment_comprehensiveness: Optional[float] = None
    
    def calculate_overall_effectiveness(self) -> Dict[str, float]:
        """Calculate overall effectiveness across all dimensions."""
        effectiveness_scores = {}
        
        # Outcome achievement score
        if self.outcome_achievement:
            effectiveness_scores['outcome_score'] = sum(self.outcome_achievement.values()) / len(self.outcome_achievement)
        
        # Efficiency score
        if self.efficiency_metrics:
            effectiveness_scores['efficiency_score'] = sum(self.efficiency_metrics.values()) / len(self.efficiency_metrics)
        
        # Equity score
        if self.equity_metrics:
            effectiveness_scores['equity_score'] = sum(self.equity_metrics.values()) / len(self.equity_metrics)
        
        # Sustainability score
        if self.sustainability_metrics:
            effectiveness_scores['sustainability_score'] = sum(self.sustainability_metrics.values()) / len(self.sustainability_metrics)
        
        # Stakeholder satisfaction score
        if self.stakeholder_satisfaction:
            effectiveness_scores['satisfaction_score'] = sum(self.stakeholder_satisfaction.values()) / len(self.stakeholder_satisfaction)
        
        # Overall effectiveness
        if effectiveness_scores:
            effectiveness_scores['overall_effectiveness'] = sum(effectiveness_scores.values()) / len(effectiveness_scores)
        
        return effectiveness_scores
    
    def generate_effectiveness_report(self) -> Dict[str, Any]:
        """Generate comprehensive effectiveness report."""
        effectiveness_report = {
            'executive_summary': {},
            'detailed_findings': {},
            'stakeholder_perspectives': {},
            'recommendations': [],
            'next_steps': []
        }
        
        # Executive summary
        overall_scores = self.calculate_overall_effectiveness()
        effectiveness_report['executive_summary'] = {
            'overall_effectiveness': overall_scores.get('overall_effectiveness', 0.0),
            'key_strengths': [],
            'key_challenges': [],
            'priority_recommendations': []
        }
        
        # Identify strengths and challenges
        for metric, score in overall_scores.items():
            if score > 0.8:
                effectiveness_report['executive_summary']['key_strengths'].append(metric)
            elif score < 0.5:
                effectiveness_report['executive_summary']['key_challenges'].append(metric)
        
        # Generate recommendations
        if overall_scores.get('efficiency_score', 1.0) < 0.6:
            effectiveness_report['recommendations'].append('Improve resource efficiency in provisioning processes')
        if overall_scores.get('equity_score', 1.0) < 0.5:
            effectiveness_report['recommendations'].append('Address equity gaps in service access and outcomes')
        if overall_scores.get('sustainability_score', 1.0) < 0.6:
            effectiveness_report['recommendations'].append('Strengthen sustainability of provisioning systems')
        
        return effectiveness_report