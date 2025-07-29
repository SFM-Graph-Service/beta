"""
Tool-Skill-Technology Complex modeling for Social Fabric Matrix analysis.

This module implements comprehensive modeling of Tool-Skill-Technology (TST) 
complexes as integrated systems within Hayden's framework. TST complexes are
fundamental to understanding how technological change affects institutional
arrangements and social provisioning processes.

Key Components:
- ToolSkillTechnologyComplex: Integrated TST system modeling
- TechnologicalCapability: Individual technology capabilities
- SkillRequirement: Skill requirements and development
- ToolSystem: Tool and equipment systems
- TechnologyTransition: Technology adoption and change processes
- TST_Integration: Integration analysis across complexes
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum, auto
import statistics
import math

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit
from models.sfm_enums import (
    ToolSkillTechnologyType,
    TechnologyMaturityLevel,
    SkillLevel,
    SystemLevel,
    ChangeType,
)


class TSTIntegrationLevel(Enum):
    """Level of integration within Tool-Skill-Technology complex."""
    
    FRAGMENTED = auto()          # Disconnected tools, skills, technologies
    LOOSELY_COUPLED = auto()     # Some connections but limited integration
    INTEGRATED = auto()          # Well-integrated TST components
    HIGHLY_INTEGRATED = auto()   # Seamless integration across all components
    SYSTEMS_INTEGRATED = auto()  # Integration extends to related systems


class SkillType(Enum):
    """Types of skills within TST complexes."""
    
    TECHNICAL_SKILL = auto()     # Technical/engineering skills
    OPERATIONAL_SKILL = auto()   # Operational and process skills
    COGNITIVE_SKILL = auto()     # Analytical and problem-solving skills
    SOCIAL_SKILL = auto()        # Communication and collaboration skills
    MANAGERIAL_SKILL = auto()    # Management and coordination skills
    CREATIVE_SKILL = auto()      # Innovation and creativity skills
    ADAPTIVE_SKILL = auto()      # Learning and adaptation skills


class TechnologyDiffusionStage(Enum):
    """Stages of technology diffusion and adoption."""
    
    INNOVATION = auto()          # Initial innovation/invention
    EARLY_ADOPTION = auto()      # Early adopters
    RAPID_DIFFUSION = auto()     # Mainstream adoption
    MATURITY = auto()            # Mature, widespread use
    DECLINE = auto()             # Declining use/obsolescence


class TST_Compatibility(Enum):
    """Compatibility between different TST components."""
    
    INCOMPATIBLE = auto()        # Cannot work together
    CONFLICTING = auto()         # Work against each other
    NEUTRAL = auto()             # No significant interaction
    COMPLEMENTARY = auto()       # Enhance each other
    SYNERGISTIC = auto()         # Create emergent capabilities


@dataclass
class TechnologicalCapability(Node):
    """Individual technology capability within TST complex."""
    
    technology_type: ToolSkillTechnologyType = ToolSkillTechnologyType.PHYSICAL_TECHNOLOGY
    maturity_level: TechnologyMaturityLevel = TechnologyMaturityLevel.EMERGING
    
    # Capability characteristics
    capability_description: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=lambda: {})
    capacity_limits: Dict[str, float] = field(default_factory=lambda: {})
    reliability_measures: Dict[str, float] = field(default_factory=lambda: {})
    
    # Resource requirements
    resource_requirements: Dict[str, float] = field(default_factory=lambda: {})
    maintenance_requirements: Dict[str, float] = field(default_factory=lambda: {})
    energy_consumption: Optional[float] = None
    environmental_impact: Dict[str, float] = field(default_factory=lambda: {})
    
    # Integration aspects
    required_complementary_technologies: List[uuid.UUID] = field(default_factory=lambda: [])
    conflicting_technologies: List[uuid.UUID] = field(default_factory=lambda: [])
    supporting_infrastructure: List[str] = field(default_factory=lambda: [])
    
    # Evolution and improvement
    improvement_trajectory: List[Dict[str, Any]] = field(default_factory=lambda: [])
    obsolescence_risk: Optional[float] = None
    upgrade_pathway: List[str] = field(default_factory=lambda: [])
    
    # Adoption characteristics
    adoption_barriers: List[str] = field(default_factory=lambda: [])
    adoption_drivers: List[str] = field(default_factory=lambda: [])
    diffusion_stage: TechnologyDiffusionStage = TechnologyDiffusionStage.INNOVATION
    
    def assess_technology_readiness(self) -> Dict[str, float]:
        """Assess readiness level of this technology capability."""
        readiness_assessment = {}
        
        # Technical readiness
        if self.performance_metrics:
            avg_performance = sum(self.performance_metrics.values()) / len(self.performance_metrics)
            readiness_assessment['technical_readiness'] = avg_performance
        
        # Reliability readiness
        if self.reliability_measures:
            avg_reliability = sum(self.reliability_measures.values()) / len(self.reliability_measures)
            readiness_assessment['reliability_readiness'] = avg_reliability
        
        # Resource readiness (inverse of requirements - lower requirements = higher readiness)
        if self.resource_requirements:
            # Normalize resource requirements (simplified)
            total_requirements = sum(self.resource_requirements.values())
            normalized_requirements = min(total_requirements / 100.0, 1.0)  # Assume 100 as high requirement
            readiness_assessment['resource_readiness'] = 1.0 - normalized_requirements
        
        # Integration readiness
        required_count = len(self.required_complementary_technologies)
        if required_count == 0:
            readiness_assessment['integration_readiness'] = 1.0
        else:
            # Simplified - in practice would check actual availability
            readiness_assessment['integration_readiness'] = max(0.0, 1.0 - required_count * 0.2)
        
        # Market readiness (based on diffusion stage)
        diffusion_readiness = {
            TechnologyDiffusionStage.INNOVATION: 0.2,
            TechnologyDiffusionStage.EARLY_ADOPTION: 0.4,
            TechnologyDiffusionStage.RAPID_DIFFUSION: 0.8,
            TechnologyDiffusionStage.MATURITY: 1.0,
            TechnologyDiffusionStage.DECLINE: 0.3
        }
        readiness_assessment['market_readiness'] = diffusion_readiness[self.diffusion_stage]
        
        # Overall readiness
        if readiness_assessment:
            readiness_assessment['overall_readiness'] = sum(readiness_assessment.values()) / len(readiness_assessment)
        
        return readiness_assessment
    
    def identify_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for technology improvement."""
        opportunities = []
        
        readiness = self.assess_technology_readiness()
        
        # Performance improvement opportunities
        if readiness.get('technical_readiness', 1.0) < 0.7:
            opportunities.append({
                'type': 'performance_enhancement',
                'description': 'Improve technical performance metrics',
                'priority': 'high',
                'estimated_impact': 0.3
            })
        
        # Reliability improvement
        if readiness.get('reliability_readiness', 1.0) < 0.6:
            opportunities.append({
                'type': 'reliability_improvement',
                'description': 'Enhance reliability and reduce failure rates',
                'priority': 'high',
                'estimated_impact': 0.4
            })
        
        # Resource efficiency
        if readiness.get('resource_readiness', 1.0) < 0.5:
            opportunities.append({
                'type': 'resource_optimization',
                'description': 'Reduce resource requirements and costs',
                'priority': 'medium',
                'estimated_impact': 0.3
            })
        
        # Integration enhancement
        if readiness.get('integration_readiness', 1.0) < 0.6:
            opportunities.append({
                'type': 'integration_improvement',
                'description': 'Improve compatibility and reduce dependencies',
                'priority': 'medium',
                'estimated_impact': 0.2
            })
        
        # Market adoption
        if readiness.get('market_readiness', 1.0) < 0.5:
            opportunities.append({
                'type': 'adoption_acceleration',
                'description': 'Address adoption barriers and accelerate diffusion',
                'priority': 'low',
                'estimated_impact': 0.4
            })
        
        return opportunities


@dataclass
class SkillRequirement(Node):
    """Skill requirements and development within TST complex."""
    
    skill_type: SkillType = SkillType.TECHNICAL_SKILL
    required_level: SkillLevel = SkillLevel.INTERMEDIATE
    
    # Skill characteristics
    skill_description: str = ""
    competency_areas: List[str] = field(default_factory=lambda: [])
    performance_standards: Dict[str, float] = field(default_factory=lambda: {})
    assessment_criteria: List[str] = field(default_factory=lambda: [])
    
    # Development requirements
    training_requirements: Dict[str, float] = field(default_factory=lambda: {})  # Hours, costs, etc.
    prerequisite_skills: List[uuid.UUID] = field(default_factory=lambda: [])
    learning_pathways: List[str] = field(default_factory=lambda: [])
    certification_requirements: List[str] = field(default_factory=lambda: [])
    
    # Skill evolution
    skill_half_life: Optional[timedelta] = None  # How quickly skill becomes obsolete
    continuous_learning_needs: List[str] = field(default_factory=lambda: [])
    skill_update_frequency: Optional[timedelta] = None
    
    # Availability and gaps
    current_availability: Optional[float] = None  # Proportion of required skills available
    skill_gaps: List[str] = field(default_factory=lambda: [])
    development_barriers: List[str] = field(default_factory=lambda: [])
    
    # Technology relationships
    supporting_technologies: List[uuid.UUID] = field(default_factory=lambda: [])
    technology_dependencies: List[uuid.UUID] = field(default_factory=lambda: [])
    
    def assess_skill_development_needs(self) -> Dict[str, Any]:
        """Assess skill development needs and gaps."""
        development_assessment = {
            'gap_analysis': {},
            'development_priorities': [],
            'resource_requirements': {},
            'timeline_estimates': {}
        }
        
        # Gap analysis
        if self.current_availability is not None:
            gap_size = max(0.0, 1.0 - self.current_availability)
            development_assessment['gap_analysis'] = {
                'current_availability': self.current_availability,
                'gap_size': gap_size,
                'gap_severity': 'high' if gap_size > 0.5 else 'medium' if gap_size > 0.2 else 'low'
            }
        
        # Development priorities
        if self.skill_gaps:
            for gap in self.skill_gaps:
                priority_level = 'high' if gap in ['critical shortage', 'urgent need'] else 'medium'
                development_assessment['development_priorities'].append({
                    'gap': gap,
                    'priority': priority_level
                })
        
        # Resource requirements
        if self.training_requirements:
            total_training_hours = self.training_requirements.get('hours', 0)
            total_training_cost = self.training_requirements.get('cost', 0)
            
            development_assessment['resource_requirements'] = {
                'training_hours': total_training_hours,
                'training_cost': total_training_cost,
                'infrastructure_needs': len(self.learning_pathways)
            }
        
        # Timeline estimates
        if self.training_requirements and 'hours' in self.training_requirements:
            training_hours = self.training_requirements['hours']
            # Assume 20 hours per week training capacity
            estimated_weeks = training_hours / 20.0
            development_assessment['timeline_estimates'] = {
                'training_duration_weeks': estimated_weeks,
                'full_competency_timeline': estimated_weeks * 1.5  # Add practice time
            }
        
        return development_assessment
    
    def calculate_skill_criticality(self, tst_context: Dict[str, Any]) -> float:
        """Calculate criticality of this skill within TST context."""
        criticality_factors = []
        
        # Dependency factor - how many other components depend on this skill
        dependency_count = len(self.technology_dependencies)
        if dependency_count > 0:
            dependency_factor = min(dependency_count / 5.0, 1.0)  # Normalize to 5 dependencies
            criticality_factors.append(dependency_factor * 0.3)
        
        # Scarcity factor - how available is this skill
        if self.current_availability is not None:
            scarcity_factor = 1.0 - self.current_availability
            criticality_factors.append(scarcity_factor * 0.3)
        
        # Complexity factor - based on required level
        complexity_weights = {
            SkillLevel.BASIC: 0.2,
            SkillLevel.INTERMEDIATE: 0.5,
            SkillLevel.ADVANCED: 0.8,
            SkillLevel.EXPERT: 1.0
        }
        complexity_factor = complexity_weights[self.required_level]
        criticality_factors.append(complexity_factor * 0.2)
        
        # Development difficulty factor
        if self.training_requirements:
            training_hours = self.training_requirements.get('hours', 40)
            difficulty_factor = min(training_hours / 200.0, 1.0)  # Normalize to 200 hours
            criticality_factors.append(difficulty_factor * 0.2)
        
        return sum(criticality_factors) if criticality_factors else 0.5


@dataclass
class ToolSystem(Node):
    """Tool and equipment systems within TST complex."""
    
    tool_category: str = ""
    system_type: str = ""  # "hardware", "software", "hybrid", "organizational"
    
    # System characteristics
    system_components: List[str] = field(default_factory=lambda: [])
    operational_parameters: Dict[str, float] = field(default_factory=lambda: {})
    performance_specifications: Dict[str, Any] = field(default_factory=lambda: {})
    
    # Integration and compatibility
    interface_standards: List[str] = field(default_factory=lambda: [])
    compatibility_matrix: Dict[str, TST_Compatibility] = field(default_factory=lambda: {})
    interoperability_requirements: List[str] = field(default_factory=lambda: [])
    
    # Lifecycle management
    acquisition_cost: Optional[float] = None
    operational_cost_per_period: Optional[float] = None
    maintenance_schedule: Dict[str, timedelta] = field(default_factory=lambda: {})
    expected_lifespan: Optional[timedelta] = None
    replacement_planning: Dict[str, Any] = field(default_factory=lambda: {})
    
    # Usage and utilization
    current_utilization_rate: Optional[float] = None
    capacity_constraints: List[str] = field(default_factory=lambda: [])
    usage_patterns: Dict[str, Any] = field(default_factory=lambda: {})
    
    # Quality and reliability
    reliability_metrics: Dict[str, float] = field(default_factory=lambda: {})
    failure_modes: List[str] = field(default_factory=lambda: [])
    quality_standards: List[str] = field(default_factory=lambda: [])
    
    def assess_tool_system_effectiveness(self) -> Dict[str, float]:
        """Assess effectiveness of the tool system."""
        effectiveness_metrics = {}
        
        # Performance effectiveness
        if self.operational_parameters:
            # Simplified performance assessment
            avg_performance = sum(self.operational_parameters.values()) / len(self.operational_parameters)
            effectiveness_metrics['performance_effectiveness'] = avg_performance
        
        # Utilization effectiveness
        if self.current_utilization_rate is not None:
            # Optimal utilization around 80%
            utilization_effectiveness = 1.0 - abs(self.current_utilization_rate - 0.8)
            effectiveness_metrics['utilization_effectiveness'] = max(0.0, utilization_effectiveness)
        
        # Reliability effectiveness
        if self.reliability_metrics:
            avg_reliability = sum(self.reliability_metrics.values()) / len(self.reliability_metrics)
            effectiveness_metrics['reliability_effectiveness'] = avg_reliability
        
        # Cost effectiveness
        if self.acquisition_cost is not None and self.expected_lifespan is not None:
            # Simplified cost-effectiveness (lower cost per time unit = higher effectiveness)
            cost_per_day = self.acquisition_cost / max(self.expected_lifespan.days, 1)
            # Normalize (assuming $100/day as benchmark)
            cost_effectiveness = max(0.0, 1.0 - (cost_per_day / 100.0))
            effectiveness_metrics['cost_effectiveness'] = min(cost_effectiveness, 1.0)
        
        # Integration effectiveness
        compatible_systems = sum(1 for comp in self.compatibility_matrix.values() 
                               if comp in [TST_Compatibility.COMPLEMENTARY, TST_Compatibility.SYNERGISTIC])
        total_systems = len(self.compatibility_matrix)
        if total_systems > 0:
            effectiveness_metrics['integration_effectiveness'] = compatible_systems / total_systems
        
        # Overall effectiveness
        if effectiveness_metrics:
            effectiveness_metrics['overall_effectiveness'] = sum(effectiveness_metrics.values()) / len(effectiveness_metrics)
        
        return effectiveness_metrics
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for tool system optimization."""
        opportunities = []
        
        effectiveness = self.assess_tool_system_effectiveness()
        
        # Performance optimization
        if effectiveness.get('performance_effectiveness', 1.0) < 0.7:
            opportunities.append({
                'type': 'performance_upgrade',
                'description': 'Upgrade system components for better performance',
                'estimated_benefit': 0.3,
                'implementation_effort': 'medium'
            })
        
        # Utilization optimization
        if self.current_utilization_rate is not None:
            if self.current_utilization_rate < 0.5:
                opportunities.append({
                    'type': 'utilization_improvement',
                    'description': 'Increase system utilization through better scheduling or capacity sharing',
                    'estimated_benefit': 0.4,
                    'implementation_effort': 'low'
                })
            elif self.current_utilization_rate > 0.9:
                opportunities.append({
                    'type': 'capacity_expansion',
                    'description': 'Expand capacity to relieve bottlenecks',
                    'estimated_benefit': 0.2,
                    'implementation_effort': 'high'
                })
        
        # Reliability improvement
        if effectiveness.get('reliability_effectiveness', 1.0) < 0.6:
            opportunities.append({
                'type': 'reliability_enhancement',
                'description': 'Improve maintenance procedures and system reliability',
                'estimated_benefit': 0.3,
                'implementation_effort': 'medium'
            })
        
        # Integration improvement
        if effectiveness.get('integration_effectiveness', 1.0) < 0.5:
            opportunities.append({
                'type': 'integration_enhancement',
                'description': 'Improve compatibility and interoperability with other systems',
                'estimated_benefit': 0.4,
                'implementation_effort': 'high'
            })
        
        return opportunities


@dataclass
class TechnologyTransition(Node):
    """Technology adoption and change processes within TST complex."""
    
    transition_type: ChangeType = ChangeType.INCREMENTAL_CHANGE
    from_technology_id: Optional[uuid.UUID] = None
    to_technology_id: uuid.UUID
    
    # Transition characteristics
    transition_drivers: List[str] = field(default_factory=lambda: [])
    transition_barriers: List[str] = field(default_factory=lambda: [])
    transition_timeline: Optional[timedelta] = None
    transition_stages: List[str] = field(default_factory=lambda: [])
    
    # Impact analysis
    affected_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    affected_processes: List[uuid.UUID] = field(default_factory=lambda: [])
    skill_transition_requirements: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # Resource requirements
    transition_costs: Dict[str, float] = field(default_factory=lambda: {})
    resource_mobilization_plan: Dict[str, Any] = field(default_factory=lambda: {})
    infrastructure_changes: List[str] = field(default_factory=lambda: [])
    
    # Risk and mitigation
    transition_risks: List[str] = field(default_factory=lambda: [])
    risk_mitigation_strategies: Dict[str, str] = field(default_factory=lambda: {})
    contingency_plans: List[str] = field(default_factory=lambda: [])
    
    # Progress tracking
    transition_milestones: List[Dict[str, Any]] = field(default_factory=lambda: [])
    current_progress: Optional[float] = None  # 0-1 scale
    success_metrics: List[str] = field(default_factory=lambda: [])
    
    def assess_transition_feasibility(self) -> Dict[str, float]:
        """Assess feasibility of the technology transition."""
        feasibility_assessment = {}
        
        # Technical feasibility
        if self.to_technology_id:
            # Simplified - would assess actual technology readiness
            feasibility_assessment['technical_feasibility'] = 0.7  # Placeholder
        
        # Resource feasibility
        if self.transition_costs:
            total_cost = sum(self.transition_costs.values())
            # Simplified assessment - in practice would compare to available resources
            resource_availability = 0.6  # Placeholder
            feasibility_assessment['resource_feasibility'] = resource_availability
        
        # Organizational feasibility
        affected_count = len(self.affected_institutions) + len(self.affected_processes)
        if affected_count > 10:
            feasibility_assessment['organizational_feasibility'] = 0.4  # High complexity
        elif affected_count > 5:
            feasibility_assessment['organizational_feasibility'] = 0.6  # Medium complexity
        else:
            feasibility_assessment['organizational_feasibility'] = 0.8  # Low complexity
        
        # Skill feasibility
        skill_transition_count = len(self.skill_transition_requirements)
        if skill_transition_count == 0:
            feasibility_assessment['skill_feasibility'] = 1.0
        else:
            # Assume each skill transition adds complexity
            feasibility_assessment['skill_feasibility'] = max(0.2, 1.0 - skill_transition_count * 0.1)
        
        # Timeline feasibility
        if self.transition_timeline is not None:
            # Simplified assessment based on timeline length
            timeline_days = self.transition_timeline.days
            if timeline_days > 365:  # More than 1 year
                feasibility_assessment['timeline_feasibility'] = 0.9
            elif timeline_days > 180:  # 6 months to 1 year
                feasibility_assessment['timeline_feasibility'] = 0.7
            elif timeline_days > 90:   # 3-6 months
                feasibility_assessment['timeline_feasibility'] = 0.5
            else:  # Less than 3 months
                feasibility_assessment['timeline_feasibility'] = 0.3
        
        # Overall feasibility
        if feasibility_assessment:
            feasibility_assessment['overall_feasibility'] = sum(feasibility_assessment.values()) / len(feasibility_assessment)
        
        return feasibility_assessment
    
    def develop_transition_plan(self) -> Dict[str, Any]:
        """Develop comprehensive transition plan."""
        transition_plan = {
            'transition_overview': {
                'transition_type': self.transition_type.name,
                'estimated_duration': self.transition_timeline,
                'affected_components': len(self.affected_institutions) + len(self.affected_processes)
            },
            'implementation_phases': [],
            'resource_plan': {},
            'risk_management': {},
            'success_metrics': self.success_metrics,
            'monitoring_plan': {}
        }
        
        # Implementation phases
        if self.transition_stages:
            for i, stage in enumerate(self.transition_stages):
                phase_duration = None
                if self.transition_timeline:
                    phase_duration = self.transition_timeline / len(self.transition_stages)
                
                transition_plan['implementation_phases'].append({
                    'phase': i + 1,
                    'stage_name': stage,
                    'estimated_duration': phase_duration,
                    'key_activities': [],  # Would be populated with specific activities
                    'success_criteria': []
                })
        
        # Resource plan
        if self.transition_costs:
            transition_plan['resource_plan'] = {
                'total_cost': sum(self.transition_costs.values()),
                'cost_breakdown': self.transition_costs,
                'resource_mobilization': self.resource_mobilization_plan
            }
        
        # Risk management
        transition_plan['risk_management'] = {
            'identified_risks': self.transition_risks,
            'mitigation_strategies': self.risk_mitigation_strategies,
            'contingency_plans': self.contingency_plans
        }
        
        # Monitoring plan
        if self.transition_milestones:
            transition_plan['monitoring_plan'] = {
                'milestones': self.transition_milestones,
                'progress_indicators': self.success_metrics,
                'review_frequency': 'monthly'  # Default
            }
        
        return transition_plan


@dataclass
class TST_Integration(Node):
    """Integration analysis across Tool-Skill-Technology complexes."""
    
    integrated_complexes: List[uuid.UUID] = field(default_factory=lambda: [])
    integration_level: TSTIntegrationLevel = TSTIntegrationLevel.LOOSELY_COUPLED
    
    # Integration metrics
    integration_score: Optional[float] = None
    synergy_potential: Optional[float] = None
    compatibility_assessment: Dict[str, TST_Compatibility] = field(default_factory=lambda: {})
    
    # Integration challenges
    integration_barriers: List[str] = field(default_factory=lambda: [])
    coordination_requirements: List[str] = field(default_factory=lambda: [])
    standardization_needs: List[str] = field(default_factory=lambda: [])
    
    # Integration benefits
    expected_synergies: List[str] = field(default_factory=lambda: [])
    efficiency_gains: Dict[str, float] = field(default_factory=lambda: {})
    capability_enhancements: List[str] = field(default_factory=lambda: [])
    
    def analyze_integration_opportunities(self, tst_complexes: List['ToolSkillTechnologyComplex']) -> Dict[str, Any]:
        """Analyze opportunities for TST integration."""
        integration_analysis = {
            'compatibility_matrix': {},
            'synergy_opportunities': [],
            'integration_challenges': [],
            'recommended_integration_sequence': []
        }
        
        # Analyze pairwise compatibility
        for i, complex1 in enumerate(tst_complexes):
            for j, complex2 in enumerate(tst_complexes[i+1:], i+1):
                compatibility = self._assess_tst_compatibility(complex1, complex2)
                integration_analysis['compatibility_matrix'][f"{complex1.id}-{complex2.id}"] = compatibility
                
                # Identify synergy opportunities
                if compatibility['overall_compatibility'] > 0.6:
                    integration_analysis['synergy_opportunities'].append({
                        'complex1_id': complex1.id,
                        'complex2_id': complex2.id,
                        'synergy_type': self._identify_synergy_type(complex1, complex2),
                        'potential_benefit': compatibility['overall_compatibility']
                    })
        
        # Identify integration challenges
        common_challenges = ['standardization_conflicts', 'resource_competition', 'coordination_complexity']
        integration_analysis['integration_challenges'] = common_challenges
        
        # Recommend integration sequence
        # Sort by compatibility and benefit potential
        synergies = integration_analysis['synergy_opportunities']
        sorted_synergies = sorted(synergies, key=lambda x: x['potential_benefit'], reverse=True)
        integration_analysis['recommended_integration_sequence'] = sorted_synergies[:5]  # Top 5
        
        return integration_analysis
    
    def _assess_tst_compatibility(self, complex1: 'ToolSkillTechnologyComplex', 
                                complex2: 'ToolSkillTechnologyComplex') -> Dict[str, float]:
        """Assess compatibility between two TST complexes."""
        compatibility_assessment = {}
        
        # Technology compatibility (simplified)
        # In practice would analyze actual technology compatibility
        compatibility_assessment['technology_compatibility'] = 0.6  # Placeholder
        
        # Skill compatibility (simplified)
        # Would analyze skill overlap and complementarity
        compatibility_assessment['skill_compatibility'] = 0.7  # Placeholder
        
        # Tool compatibility (simplified)
        # Would analyze tool interoperability
        compatibility_assessment['tool_compatibility'] = 0.5  # Placeholder
        
        # Resource compatibility
        # Would analyze resource sharing potential
        compatibility_assessment['resource_compatibility'] = 0.6  # Placeholder
        
        # Overall compatibility
        compatibility_assessment['overall_compatibility'] = sum(compatibility_assessment.values()) / len(compatibility_assessment)
        
        return compatibility_assessment
    
    def _identify_synergy_type(self, complex1: 'ToolSkillTechnologyComplex', 
                             complex2: 'ToolSkillTechnologyComplex') -> str:
        """Identify type of synergy between TST complexes."""
        # Simplified synergy type identification
        synergy_types = [
            'complementary_capabilities',
            'shared_infrastructure',
            'skill_cross_training',
            'technology_integration',
            'resource_optimization'
        ]
        return synergy_types[0]  # Placeholder - would use actual analysis


@dataclass
class ToolSkillTechnologyComplex(Node):
    """Integrated Tool-Skill-Technology complex system."""
    
    complex_type: ToolSkillTechnologyType = ToolSkillTechnologyType.INTEGRATED_SYSTEM
    integration_level: TSTIntegrationLevel = TSTIntegrationLevel.INTEGRATED
    
    # Component systems
    technology_capabilities: List[uuid.UUID] = field(default_factory=lambda: [])
    skill_requirements: List[uuid.UUID] = field(default_factory=lambda: [])
    tool_systems: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # Complex characteristics
    system_purpose: str = ""
    primary_functions: List[str] = field(default_factory=lambda: [])
    performance_objectives: Dict[str, float] = field(default_factory=lambda: {})
    
    # Integration properties
    internal_dependencies: Dict[str, List[str]] = field(default_factory=lambda: {})
    integration_mechanisms: List[str] = field(default_factory=lambda: [])
    coordination_processes: List[str] = field(default_factory=lambda: [])
    
    # Complex evolution
    development_trajectory: List[Dict[str, Any]] = field(default_factory=lambda: [])
    adaptation_capacity: Optional[float] = None
    innovation_potential: Optional[float] = None
    
    # Environmental context
    operating_environment: Dict[str, Any] = field(default_factory=lambda: {})
    external_dependencies: List[str] = field(default_factory=lambda: [])
    environmental_constraints: List[str] = field(default_factory=lambda: [])
    
    # Performance and effectiveness
    complex_performance_metrics: Dict[str, float] = field(default_factory=lambda: {})
    effectiveness_indicators: List[str] = field(default_factory=lambda: [])
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=lambda: [])
    
    def assess_complex_integration(self) -> Dict[str, float]:
        """Assess integration level within the TST complex."""
        integration_assessment = {}
        
        # Component integration
        total_components = (len(self.technology_capabilities) + 
                          len(self.skill_requirements) + 
                          len(self.tool_systems))
        
        if total_components > 0:
            # Integration mechanisms per component
            mechanism_density = len(self.integration_mechanisms) / total_components
            integration_assessment['mechanism_integration'] = min(mechanism_density, 1.0)
        
        # Dependency integration
        if self.internal_dependencies:
            total_dependencies = sum(len(deps) for deps in self.internal_dependencies.values())
            dependency_density = total_dependencies / max(total_components, 1)
            integration_assessment['dependency_integration'] = min(dependency_density / 2.0, 1.0)
        
        # Coordination integration
        if self.coordination_processes:
            coordination_score = min(len(self.coordination_processes) / 3.0, 1.0)
            integration_assessment['coordination_integration'] = coordination_score
        
        # Performance integration (how well components work together)
        if self.complex_performance_metrics:
            avg_performance = sum(self.complex_performance_metrics.values()) / len(self.complex_performance_metrics)
            integration_assessment['performance_integration'] = avg_performance
        
        # Overall integration
        if integration_assessment:
            overall_integration = sum(integration_assessment.values()) / len(integration_assessment)
            integration_assessment['overall_integration'] = overall_integration
            
            # Update integration level based on score
            if overall_integration > 0.8:
                self.integration_level = TSTIntegrationLevel.HIGHLY_INTEGRATED
            elif overall_integration > 0.6:
                self.integration_level = TSTIntegrationLevel.INTEGRATED
            elif overall_integration > 0.4:
                self.integration_level = TSTIntegrationLevel.LOOSELY_COUPLED
            else:
                self.integration_level = TSTIntegrationLevel.FRAGMENTED
        
        return integration_assessment
    
    def analyze_complex_capabilities(self) -> Dict[str, Any]:
        """Analyze capabilities of the integrated TST complex."""
        capability_analysis = {
            'core_capabilities': [],
            'capability_gaps': [],
            'enhancement_opportunities': [],
            'competitive_advantages': []
        }
        
        # Core capabilities (simplified analysis)
        for function in self.primary_functions:
            capability_analysis['core_capabilities'].append({
                'function': function,
                'strength_level': 'high',  # Placeholder
                'supporting_components': []  # Would map to actual components
            })
        
        # Capability gaps
        if len(self.technology_capabilities) < 3:
            capability_analysis['capability_gaps'].append('Limited technology diversity')
        
        if len(self.skill_requirements) < 5:
            capability_analysis['capability_gaps'].append('Narrow skill base')
        
        # Enhancement opportunities
        integration_assessment = self.assess_complex_integration()
        if integration_assessment.get('overall_integration', 0) < 0.7:
            capability_analysis['enhancement_opportunities'].append({
                'type': 'integration_enhancement',
                'description': 'Improve integration between TST components',
                'priority': 'high'
            })
        
        if self.adaptation_capacity is not None and self.adaptation_capacity < 0.6:
            capability_analysis['enhancement_opportunities'].append({
                'type': 'adaptation_improvement',
                'description': 'Enhance adaptive capacity for changing conditions',
                'priority': 'medium'
            })
        
        # Competitive advantages
        if self.integration_level in [TSTIntegrationLevel.HIGHLY_INTEGRATED, TSTIntegrationLevel.SYSTEMS_INTEGRATED]:
            capability_analysis['competitive_advantages'].append('Superior integration capabilities')
        
        if self.innovation_potential is not None and self.innovation_potential > 0.7:
            capability_analysis['competitive_advantages'].append('High innovation potential')
        
        return capability_analysis
    
    def generate_complex_optimization_plan(self) -> Dict[str, Any]:
        """Generate optimization plan for the TST complex."""
        optimization_plan = {
            'current_state_assessment': {},
            'optimization_objectives': [],
            'improvement_initiatives': [],
            'implementation_roadmap': {},
            'success_metrics': []
        }
        
        # Current state assessment
        integration_assessment = self.assess_complex_integration()
        capability_analysis = self.analyze_complex_capabilities()
        
        optimization_plan['current_state_assessment'] = {
            'integration_level': self.integration_level.name,
            'integration_score': integration_assessment.get('overall_integration', 0),
            'capability_gaps': len(capability_analysis['capability_gaps']),
            'enhancement_opportunities': len(capability_analysis['enhancement_opportunities'])
        }
        
        # Optimization objectives
        if integration_assessment.get('overall_integration', 0) < 0.8:
            optimization_plan['optimization_objectives'].append('Achieve higher integration level')
        
        if capability_analysis['capability_gaps']:
            optimization_plan['optimization_objectives'].append('Address identified capability gaps')
        
        # Improvement initiatives
        for opportunity in capability_analysis['enhancement_opportunities']:
            optimization_plan['improvement_initiatives'].append({
                'initiative': opportunity['description'],
                'type': opportunity['type'],
                'priority': opportunity['priority'],
                'estimated_benefit': 'medium'  # Placeholder
            })
        
        # Success metrics
        optimization_plan['success_metrics'] = [
            'Integration score improvement',
            'Capability gap reduction',
            'Performance metric enhancement',
            'Stakeholder satisfaction increase'
        ]
        
        return optimization_plan