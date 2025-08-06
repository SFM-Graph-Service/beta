"""
Enhanced Stakeholder Engagement Framework for Social Fabric Matrix analysis.

This module implements comprehensive stakeholder engagement methodology following
Hayden's participatory approach to institutional economics. It provides systematic
tools for stakeholder identification, engagement planning, relationship management,
and collaborative capacity building within the SFM framework.

Key Components:
- StakeholderEngagementPlan: Comprehensive engagement strategy
- EngagementActivity: Individual engagement activities
- StakeholderRelationshipManager: Relationship management system
- CapacityBuildingProgram: Stakeholder capacity development
- EngagementOutcomeAssessment: Evaluation of engagement effectiveness
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
from models.meta_entities import TimeSlice
from models.sfm_enums import (
    StakeholderType,
    ParticipationLevel,
    InfluenceStrategy,
    CommunicationChannel,
)


class EngagementStrategy(Enum):
    """Types of stakeholder engagement strategies."""
    
    INFORMATIONAL = auto()       # One-way information sharing
    CONSULTATIVE = auto()        # Seeking input and feedback
    PARTICIPATORY = auto()       # Joint problem-solving
    COLLABORATIVE = auto()       # Shared decision-making
    EMPOWERMENT = auto()         # Stakeholder-led initiatives


class EngagementPhase(Enum):
    """Phases of stakeholder engagement process."""
    
    IDENTIFICATION = auto()      # Stakeholder identification
    MAPPING = auto()            # Stakeholder analysis and mapping
    PLANNING = auto()           # Engagement planning
    IMPLEMENTATION = auto()     # Active engagement
    EVALUATION = auto()         # Assessment and learning
    ADAPTATION = auto()         # Process improvement


class EngagementFrequency(Enum):
    """Frequency of stakeholder engagement activities."""
    
    ONE_TIME = auto()           # Single engagement event
    PERIODIC = auto()           # Regular scheduled engagement
    ONGOING = auto()            # Continuous engagement
    AS_NEEDED = auto()          # Triggered by specific needs
    MILESTONE_BASED = auto()    # Tied to specific milestones


class RelationshipQuality(Enum):
    """Quality levels of stakeholder relationships."""
    
    EXCELLENT = auto()          # High trust, strong collaboration
    GOOD = auto()              # Positive, productive relationship
    FAIR = auto()              # Neutral, functional relationship
    POOR = auto()              # Tensions, limited cooperation
    ADVERSARIAL = auto()       # Conflictual, opposing interests


class CapacityGap(Enum):
    """Types of stakeholder capacity gaps."""
    
    TECHNICAL_KNOWLEDGE = auto()    # Technical/subject matter expertise
    PROCESS_SKILLS = auto()         # Participation and engagement skills
    RESOURCES = auto()              # Financial, time, or material resources
    ACCESS = auto()                 # Access to information or processes
    REPRESENTATION = auto()         # Ability to represent constituency
    COMMUNICATION = auto()          # Communication and language skills


@dataclass
class StakeholderProfile(Node):
    """Comprehensive profile of individual stakeholder for engagement."""
    
    stakeholder_type: StakeholderType = StakeholderType.COMMUNITY_GROUP
    stakeholder_category: Optional[str] = None
    
    # Basic information
    stakeholder_name: Optional[str] = None
    organization_affiliation: Optional[str] = None
    role_description: Optional[str] = None
    
    # Contact and accessibility
    contact_information: Dict[str, str] = field(default_factory=dict)
    preferred_communication_channels: List[CommunicationChannel] = field(default_factory=list)
    accessibility_requirements: List[str] = field(default_factory=list)
    
    # Interests and concerns
    primary_interests: List[str] = field(default_factory=list)
    key_concerns: List[str] = field(default_factory=list)
    value_priorities: List[str] = field(default_factory=list)
    
    # Capacity assessment
    engagement_capacity: Optional[float] = None  # Available capacity (0-1)
    technical_expertise: Dict[str, float] = field(default_factory=dict)
    participation_experience: Optional[float] = None  # 0-1 scale
    
    # Relationship dynamics
    current_relationship_quality: RelationshipQuality = RelationshipQuality.FAIR
    trust_level: Optional[float] = None  # 0-1 scale
    collaboration_history: List[str] = field(default_factory=list)
    
    # Influence and networks
    influence_networks: List[uuid.UUID] = field(default_factory=list)
    constituency_representation: Optional[str] = None
    decision_making_authority: Optional[float] = None  # 0-1 scale
    
    # Engagement preferences
    preferred_engagement_style: Optional[str] = None
    availability_constraints: List[str] = field(default_factory=list)
    engagement_motivations: List[str] = field(default_factory=list)
    
    # Capacity building needs
    identified_capacity_gaps: List[CapacityGap] = field(default_factory=list)
    learning_preferences: List[str] = field(default_factory=list)
    support_requirements: List[str] = field(default_factory=list)
    
    # SFM context
    matrix_interests: List[uuid.UUID] = field(default_factory=list)
    institutional_connections: List[uuid.UUID] = field(default_factory=list)
    delivery_system_involvement: List[uuid.UUID] = field(default_factory=list)
    
    def assess_engagement_readiness(self) -> Dict[str, Any]:
        """Assess stakeholder's readiness for engagement."""
        readiness_assessment = {
            'readiness_score': 0.0,
            'readiness_factors': {},
            'enabling_factors': [],
            'barriers': [],
            'preparation_needs': []
        }
        
        # Readiness factors
        factors = {
            'capacity': self.engagement_capacity or 0.5,
            'trust': self.trust_level or 0.5,
            'experience': self.participation_experience or 0.5,
            'motivation': len(self.engagement_motivations) / 5.0 if self.engagement_motivations else 0.3
        }
        
        readiness_assessment['readiness_factors'] = factors
        readiness_assessment['readiness_score'] = sum(factors.values()) / len(factors)
        
        # Identify enabling factors and barriers
        if factors['trust'] > 0.7:
            readiness_assessment['enabling_factors'].append('High trust level')
        elif factors['trust'] < 0.4:
            readiness_assessment['barriers'].append('Low trust level')
        
        if factors['capacity'] > 0.7:
            readiness_assessment['enabling_factors'].append('Good engagement capacity')
        elif factors['capacity'] < 0.4:
            readiness_assessment['barriers'].append('Limited engagement capacity')
        
        # Preparation needs
        if self.identified_capacity_gaps:
            readiness_assessment['preparation_needs'].append('Address capacity gaps')
        
        if self.current_relationship_quality in [RelationshipQuality.POOR, RelationshipQuality.ADVERSARIAL]:
            readiness_assessment['preparation_needs'].append('Relationship building required')
        
        return readiness_assessment


@dataclass
class EngagementActivity(Node):
    """Individual stakeholder engagement activity."""
    
    activity_type: Optional[str] = None
    engagement_strategy: EngagementStrategy = EngagementStrategy.CONSULTATIVE
    activity_phase: EngagementPhase = EngagementPhase.IMPLEMENTATION
    
    # Activity design
    activity_objectives: List[str] = field(default_factory=list)
    target_participants: List[uuid.UUID] = field(default_factory=list)
    activity_format: Optional[str] = None  # "workshop", "survey", "interview", etc.
    
    # Logistics
    scheduled_date: Optional[datetime] = None
    duration: Optional[timedelta] = None
    location: Optional[str] = None
    communication_channels: List[CommunicationChannel] = field(default_factory=list)
    
    # Content and materials
    agenda_items: List[str] = field(default_factory=list)
    information_materials: List[str] = field(default_factory=list)
    facilitation_approach: Optional[str] = None
    
    # Accessibility
    accessibility_provisions: List[str] = field(default_factory=list)
    language_support: List[str] = field(default_factory=list)
    accommodation_measures: List[str] = field(default_factory=list)
    
    # Outcomes
    actual_participants: List[uuid.UUID] = field(default_factory=list)
    participation_rate: Optional[float] = None  # Actual/planned participation
    activity_outputs: List[str] = field(default_factory=list)
    
    # Feedback and evaluation
    participant_feedback: Dict[uuid.UUID, str] = field(default_factory=dict)
    satisfaction_scores: Dict[uuid.UUID, float] = field(default_factory=dict)
    effectiveness_assessment: Optional[float] = None  # 0-1 scale
    
    # Follow-up
    follow_up_actions: List[str] = field(default_factory=list)
    information_sharing_commitments: List[str] = field(default_factory=list)
    next_engagement_opportunities: List[str] = field(default_factory=list)
    
    # Learning and adaptation
    lessons_learned: List[str] = field(default_factory=list)
    process_improvements: List[str] = field(default_factory=list)
    replication_potential: Optional[str] = None
    
    def evaluate_activity_success(self) -> Dict[str, Any]:
        """Evaluate success of engagement activity."""
        success_evaluation = {
            'success_score': 0.0,
            'success_dimensions': {},
            'achievements': [],
            'challenges': [],
            'improvement_opportunities': []
        }
        
        # Success dimensions
        dimensions = {}
        
        # Participation success
        if self.participation_rate:
            dimensions['participation'] = self.participation_rate
        
        # Satisfaction success
        if self.satisfaction_scores:
            avg_satisfaction = sum(self.satisfaction_scores.values()) / len(self.satisfaction_scores)
            dimensions['satisfaction'] = avg_satisfaction
        
        # Effectiveness success
        if self.effectiveness_assessment:
            dimensions['effectiveness'] = self.effectiveness_assessment
        
        # Output success
        expected_outputs = len(self.activity_objectives)
        actual_outputs = len(self.activity_outputs)
        if expected_outputs > 0:
            dimensions['output_achievement'] = min(1.0, actual_outputs / expected_outputs)
        
        success_evaluation['success_dimensions'] = dimensions
        
        # Calculate overall success
        if dimensions:
            success_evaluation['success_score'] = sum(dimensions.values()) / len(dimensions)
        
        # Identify achievements and challenges
        for dimension, score in dimensions.items():
            if score >= 0.8:
                success_evaluation['achievements'].append(f"Strong {dimension}")
            elif score <= 0.4:
                success_evaluation['challenges'].append(f"Weak {dimension}")
        
        # Generate improvement opportunities
        if success_evaluation['success_score'] < 0.6:
            success_evaluation['improvement_opportunities'].extend([
                'Enhance activity design',
                'Improve participant preparation',
                'Strengthen facilitation approach',
                'Better follow-up processes'
            ])
        
        return success_evaluation


@dataclass
class StakeholderEngagementPlan(Node):
    """Comprehensive stakeholder engagement plan."""
    
    plan_scope: Optional[str] = None
    planning_timeframe: Optional[TimeSlice] = None
    
    # Strategic framework
    engagement_vision: Optional[str] = None
    engagement_principles: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Stakeholder universe
    identified_stakeholders: List[uuid.UUID] = field(default_factory=list)  # StakeholderProfile IDs
    stakeholder_prioritization: Dict[uuid.UUID, float] = field(default_factory=dict)
    engagement_levels: Dict[uuid.UUID, ParticipationLevel] = field(default_factory=dict)
    
    # Engagement strategy
    overall_strategy: EngagementStrategy = EngagementStrategy.PARTICIPATORY
    phase_strategies: Dict[EngagementPhase, str] = field(default_factory=dict)
    differentiated_approaches: Dict[str, str] = field(default_factory=dict)
    
    # Activity planning
    planned_activities: List[uuid.UUID] = field(default_factory=list)  # EngagementActivity IDs
    activity_sequence: List[uuid.UUID] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Communication strategy
    communication_plan: Dict[str, str] = field(default_factory=dict)
    information_sharing_protocols: List[str] = field(default_factory=list)
    feedback_mechanisms: List[str] = field(default_factory=list)
    
    # Capacity building
    capacity_building_strategy: Optional[str] = None
    training_programs: List[uuid.UUID] = field(default_factory=list)
    resource_support_plans: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    
    # Risk management
    engagement_risks: List[str] = field(default_factory=list)
    mitigation_strategies: Dict[str, List[str]] = field(default_factory=dict)
    contingency_plans: List[str] = field(default_factory=list)
    
    # Monitoring and evaluation
    monitoring_framework: List[str] = field(default_factory=list)
    evaluation_indicators: Dict[str, str] = field(default_factory=dict)
    learning_integration_process: Optional[str] = None
    
    # SFM integration
    matrix_engagement_priorities: List[uuid.UUID] = field(default_factory=list)
    institutional_engagement_implications: List[uuid.UUID] = field(default_factory=list)
    delivery_system_engagement_connections: List[uuid.UUID] = field(default_factory=list)
    
    def develop_engagement_strategy(self) -> Dict[str, Any]:
        """Develop comprehensive engagement strategy."""
        strategy_development = {
            'strategy_framework': {},
            'stakeholder_segmentation': {},
            'engagement_pathways': {},
            'resource_allocation': {},
            'timeline': []
        }
        
        # Strategy framework based on stakeholder analysis
        high_priority = [s for s, priority in self.stakeholder_prioritization.items() if priority > 0.8]
        medium_priority = [s for s, priority in self.stakeholder_prioritization.items() if 0.5 <= priority <= 0.8]
        low_priority = [s for s, priority in self.stakeholder_prioritization.items() if priority < 0.5]
        
        strategy_development['stakeholder_segmentation'] = {
            'high_priority': len(high_priority),
            'medium_priority': len(medium_priority),
            'low_priority': len(low_priority)
        }
        
        # Engagement pathways by priority
        strategy_development['engagement_pathways'] = {
            'high_priority': 'Collaborative partnership',
            'medium_priority': 'Active consultation',
            'low_priority': 'Information sharing'
        }
        
        # Resource allocation
        total_stakeholders = len(self.identified_stakeholders)
        if total_stakeholders > 0:
            strategy_development['resource_allocation'] = {
                'high_priority_allocation': 0.6,
                'medium_priority_allocation': 0.3,
                'low_priority_allocation': 0.1
            }
        
        # Timeline development
        strategy_development['timeline'] = [
            'Phase 1: Stakeholder preparation and relationship building',
            'Phase 2: Active engagement and collaboration',
            'Phase 3: Decision-making and agreement',
            'Phase 4: Implementation and monitoring'
        ]
        
        return strategy_development
    
    def assess_plan_feasibility(self) -> Dict[str, Any]:
        """Assess feasibility of engagement plan."""
        feasibility_assessment = {
            'feasibility_score': 0.0,
            'feasibility_dimensions': {},
            'implementation_challenges': [],
            'success_enablers': [],
            'recommendations': []
        }
        
        # Feasibility dimensions
        dimensions = {}
        
        # Resource feasibility
        total_activities = len(self.planned_activities)
        total_resources = sum(self.resource_requirements.values()) if self.resource_requirements else 0
        if total_activities > 0:
            resource_per_activity = total_resources / total_activities if total_resources > 0 else 0.5
            dimensions['resource_feasibility'] = min(1.0, resource_per_activity / 10)  # Normalized
        
        # Stakeholder readiness
        total_stakeholders = len(self.identified_stakeholders)
        if total_stakeholders > 0:
            # Simplified readiness assessment
            dimensions['stakeholder_readiness'] = 0.7  # Placeholder
        
        # Timeline feasibility
        if self.planning_timeframe:
            dimensions['timeline_feasibility'] = 0.8  # Placeholder
        
        feasibility_assessment['feasibility_dimensions'] = dimensions
        
        # Calculate overall feasibility
        if dimensions:
            feasibility_assessment['feasibility_score'] = sum(dimensions.values()) / len(dimensions)
        
        # Identify challenges and enablers
        if feasibility_assessment['feasibility_score'] < 0.6:
            feasibility_assessment['implementation_challenges'].extend([
                'Resource constraints may limit effectiveness',
                'Stakeholder readiness concerns',
                'Timeline pressures'
            ])
        else:
            feasibility_assessment['success_enablers'].extend([
                'Adequate resource allocation',
                'Good stakeholder readiness',
                'Realistic timeline'
            ])
        
        # Generate recommendations
        feasibility_assessment['recommendations'] = [
            'Regular feasibility monitoring',
            'Adaptive plan management',
            'Stakeholder feedback integration'
        ]
        
        return feasibility_assessment


@dataclass 
class StakeholderRelationshipManager(Node):
    """System for managing stakeholder relationships over time."""
    
    relationship_scope: Optional[str] = None
    management_approach: Optional[str] = None
    
    # Relationship portfolio
    managed_relationships: List[uuid.UUID] = field(default_factory=list)  # StakeholderProfile IDs
    relationship_categories: Dict[uuid.UUID, str] = field(default_factory=dict)
    relationship_priorities: Dict[uuid.UUID, float] = field(default_factory=dict)
    
    # Relationship tracking
    relationship_history: Dict[uuid.UUID, List[Dict[str, Any]]] = field(default_factory=dict)
    interaction_records: List[Dict[str, Any]] = field(default_factory=list)
    relationship_trends: Dict[uuid.UUID, str] = field(default_factory=dict)
    
    # Relationship maintenance
    maintenance_strategies: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    communication_schedules: Dict[uuid.UUID, str] = field(default_factory=dict)
    relationship_investments: Dict[uuid.UUID, float] = field(default_factory=dict)
    
    # Conflict management
    relationship_conflicts: List[uuid.UUID] = field(default_factory=list)
    conflict_resolution_approaches: Dict[uuid.UUID, str] = field(default_factory=dict)
    mediation_processes: List[uuid.UUID] = field(default_factory=list)
    
    # Performance monitoring
    relationship_health_scores: Dict[uuid.UUID, float] = field(default_factory=dict)
    satisfaction_tracking: Dict[uuid.UUID, List[float]] = field(default_factory=dict)
    mutual_benefit_assessment: Dict[uuid.UUID, float] = field(default_factory=dict)
    
    # Capacity building
    relationship_capacity_building: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    collaborative_learning_initiatives: List[str] = field(default_factory=list)
    knowledge_sharing_mechanisms: List[str] = field(default_factory=list)
    
    def evaluate_relationship_portfolio(self) -> Dict[str, Any]:
        """Evaluate overall stakeholder relationship portfolio."""
        portfolio_evaluation = {
            'portfolio_health': 0.0,
            'relationship_distribution': {},
            'high_performing_relationships': [],
            'at_risk_relationships': [],
            'investment_priorities': []
        }
        
        # Portfolio health calculation
        if self.relationship_health_scores:
            portfolio_health = sum(self.relationship_health_scores.values()) / len(self.relationship_health_scores)
            portfolio_evaluation['portfolio_health'] = portfolio_health
        
        # Relationship distribution
        quality_distribution = {}
        for stakeholder_id, health_score in self.relationship_health_scores.items():
            if health_score >= 0.8:
                quality_distribution['excellent'] = quality_distribution.get('excellent', 0) + 1
                portfolio_evaluation['high_performing_relationships'].append(stakeholder_id)
            elif health_score >= 0.6:
                quality_distribution['good'] = quality_distribution.get('good', 0) + 1
            elif health_score >= 0.4:
                quality_distribution['fair'] = quality_distribution.get('fair', 0) + 1
            else:
                quality_distribution['poor'] = quality_distribution.get('poor', 0) + 1
                portfolio_evaluation['at_risk_relationships'].append(stakeholder_id)
        
        portfolio_evaluation['relationship_distribution'] = quality_distribution
        
        # Investment priorities
        for stakeholder_id in portfolio_evaluation['at_risk_relationships']:
            portfolio_evaluation['investment_priorities'].append(f"Strengthen relationship with {stakeholder_id}")
        
        return portfolio_evaluation


@dataclass
class CapacityBuildingProgram(Node):
    """Stakeholder capacity building program."""
    
    program_scope: Optional[str] = None
    target_capacities: List[str] = field(default_factory=list)
    
    # Program design
    program_objectives: List[str] = field(default_factory=list)
    target_participants: List[uuid.UUID] = field(default_factory=list)
    capacity_assessment_results: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    
    # Learning approach
    learning_methods: List[str] = field(default_factory=list)
    training_modules: List[str] = field(default_factory=list)
    practical_application_opportunities: List[str] = field(default_factory=list)
    
    # Resource provision
    material_resources: List[str] = field(default_factory=list)
    technical_assistance: List[str] = field(default_factory=list)
    funding_support: Dict[uuid.UUID, float] = field(default_factory=dict)
    
    # Program delivery
    delivery_schedule: List[datetime] = field(default_factory=list)
    facilitators: List[uuid.UUID] = field(default_factory=list)
    delivery_locations: List[str] = field(default_factory=list)
    
    # Outcomes tracking
    capacity_improvements: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    skill_development_progress: Dict[uuid.UUID, List[str]] = field(default_factory=list)
    application_success_stories: List[str] = field(default_factory=list)
    
    # Sustainability
    sustainability_mechanisms: List[str] = field(default_factory=list)
    peer_learning_networks: List[uuid.UUID] = field(default_factory=list)
    ongoing_support_systems: List[str] = field(default_factory=list)
    
    def assess_capacity_building_effectiveness(self) -> Dict[str, Any]:
        """Assess effectiveness of capacity building program."""
        effectiveness_assessment = {
            'effectiveness_score': 0.0,
            'improvement_achievements': {},
            'success_factors': [],
            'challenge_areas': [],
            'sustainability_indicators': []
        }
        
        # Calculate improvement achievements
        total_improvements = 0
        significant_improvements = 0
        
        for participant_id, improvements in self.capacity_improvements.items():
            for capacity, improvement in improvements.items():
                total_improvements += 1
                if improvement > 0.3:  # Significant improvement threshold
                    significant_improvements += 1
        
        if total_improvements > 0:
            improvement_rate = significant_improvements / total_improvements
            effectiveness_assessment['effectiveness_score'] = improvement_rate
            effectiveness_assessment['improvement_achievements'] = {
                'total_improvements': total_improvements,
                'significant_improvements': significant_improvements,
                'improvement_rate': improvement_rate
            }
        
        # Success factors
        if len(self.application_success_stories) > 0:
            effectiveness_assessment['success_factors'].append('Practical application success')
        
        if len(self.peer_learning_networks) > 0:
            effectiveness_assessment['success_factors'].append('Strong peer learning networks')
        
        # Sustainability indicators
        if self.sustainability_mechanisms:
            effectiveness_assessment['sustainability_indicators'].extend(self.sustainability_mechanisms)
        
        return effectiveness_assessment


@dataclass
class EngagementOutcomeAssessment(Node):
    """Assessment of stakeholder engagement outcomes and impacts."""
    
    assessment_scope: Optional[str] = None
    assessment_timeframe: Optional[TimeSlice] = None
    
    # Assessment framework
    evaluation_criteria: List[str] = field(default_factory=list)
    success_indicators: Dict[str, str] = field(default_factory=dict)
    measurement_methods: List[str] = field(default_factory=list)
    
    # Outcome measurement
    engagement_reach: Optional[float] = None  # Proportion of stakeholders engaged
    participation_quality: Optional[float] = None  # Quality of participation (0-1)
    stakeholder_satisfaction: Optional[float] = None  # Overall satisfaction (0-1)
    
    # Impact assessment
    relationship_improvements: Dict[uuid.UUID, float] = field(default_factory=dict)
    capacity_enhancements: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    collaborative_outcomes: List[str] = field(default_factory=list)
    
    # System-level impacts
    institutional_influence: Optional[float] = None  # Influence on institutions (0-1)
    policy_impacts: List[str] = field(default_factory=list)
    decision_making_improvements: List[str] = field(default_factory=list)
    
    # Learning and adaptation
    lessons_learned: List[str] = field(default_factory=list)
    best_practices_identified: List[str] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)
    
    # Future engagement
    sustained_engagement_indicators: List[str] = field(default_factory=list)
    engagement_momentum: Optional[float] = None  # Momentum for continued engagement (0-1)
    future_engagement_opportunities: List[str] = field(default_factory=list)
    
    def conduct_comprehensive_outcome_assessment(self) -> Dict[str, Any]:
        """Conduct comprehensive assessment of engagement outcomes."""
        outcome_assessment = {
            'overall_success_score': 0.0,
            'outcome_dimensions': {},
            'key_achievements': [],
            'improvement_areas': [],
            'strategic_recommendations': []
        }
        
        # Outcome dimensions
        dimensions = {
            'reach': self.engagement_reach or 0.5,
            'quality': self.participation_quality or 0.5,
            'satisfaction': self.stakeholder_satisfaction or 0.5,
            'institutional_influence': self.institutional_influence or 0.5,
            'momentum': self.engagement_momentum or 0.5
        }
        
        outcome_assessment['outcome_dimensions'] = dimensions
        
        # Calculate overall success
        overall_success = sum(dimensions.values()) / len(dimensions)
        outcome_assessment['overall_success_score'] = overall_success
        
        # Identify achievements and improvement areas
        for dimension, score in dimensions.items():
            if score >= 0.8:
                outcome_assessment['key_achievements'].append(f"Strong {dimension}")
            elif score <= 0.4:
                outcome_assessment['improvement_areas'].append(f"Weak {dimension}")
        
        # Strategic recommendations
        if overall_success >= 0.8:
            outcome_assessment['strategic_recommendations'] = [
                'Sustain current approach',
                'Scale successful practices',
                'Share lessons learned'
            ]
        elif overall_success >= 0.6:
            outcome_assessment['strategic_recommendations'] = [
                'Build on current strengths',
                'Address identified weaknesses',
                'Enhance stakeholder capacity'
            ]
        else:
            outcome_assessment['strategic_recommendations'] = [
                'Fundamental review of engagement approach',
                'Strengthen stakeholder relationships',
                'Invest in capacity building'
            ]
        
        return outcome_assessment