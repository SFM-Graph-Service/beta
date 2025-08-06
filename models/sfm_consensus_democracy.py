"""
Consensus Building and Democratic Participation Framework for Social Fabric Matrix analysis.

This module implements Hayden's democratic methodology for consensus building,
participatory decision-making, and collaborative problem-solving within the
Social Fabric Matrix framework. Essential for ensuring legitimate and effective
institutional outcomes through inclusive stakeholder engagement.

Key Components:
- ConsensusProcess: Structured consensus building methodology
- DeliberationProcess: Democratic deliberation and dialogue
- ParticipationFramework: Comprehensive stakeholder participation
- DemocraticDecisionMaking: Democratic decision processes
- ConflictMediation: Mediation and conflict resolution
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
    DecisionMakingApproach,
    ConflictResolutionMethod,
)


class ConsensusType(Enum):
    """Types of consensus building approaches."""
    
    UNANIMOUS_CONSENSUS = auto()    # Full agreement required
    SUBSTANTIAL_CONSENSUS = auto()  # Majority with no strong objections
    WORKING_CONSENSUS = auto()      # Practical agreement to proceed
    FACILITATED_CONSENSUS = auto()  # Mediator-assisted consensus
    MODIFIED_CONSENSUS = auto()     # Consensus with recorded dissent


class DeliberationQuality(Enum):
    """Quality levels of democratic deliberation."""
    
    EXCELLENT = auto()      # High-quality inclusive deliberation
    GOOD = auto()          # Adequate deliberation process
    FAIR = auto()          # Basic deliberation with some issues
    POOR = auto()          # Inadequate deliberation
    INADEQUATE = auto()    # Severely deficient deliberation


class ParticipationBarrier(Enum):
    """Types of barriers to stakeholder participation."""
    
    ACCESS_BARRIERS = auto()        # Physical/technological access
    RESOURCE_CONSTRAINTS = auto()   # Time, money, capacity limits
    INFORMATION_GAPS = auto()       # Lack of relevant information
    POWER_IMBALANCES = auto()      # Unequal power relationships
    CULTURAL_BARRIERS = auto()      # Cultural or language barriers
    PROCEDURAL_BARRIERS = auto()    # Complex or unclear processes
    TRUST_DEFICITS = auto()        # Low trust in process/institutions


class MediationOutcome(Enum):
    """Outcomes of mediation processes."""
    
    FULL_AGREEMENT = auto()        # Complete resolution achieved
    PARTIAL_AGREEMENT = auto()     # Some issues resolved
    IMPROVED_UNDERSTANDING = auto() # Better mutual understanding
    STRUCTURED_DISAGREEMENT = auto() # Clear disagreement framework
    MEDIATION_FAILED = auto()      # No progress achieved


@dataclass
class ConsensusProcess(Node):
    """Structured consensus building process for stakeholder agreement."""
    
    consensus_type: ConsensusType = ConsensusType.SUBSTANTIAL_CONSENSUS
    process_scope: Optional[str] = None
    target_decision: Optional[str] = None
    
    # Process design
    consensus_threshold: Optional[float] = None  # Required agreement level (0-1)
    participation_requirements: List[str] = field(default_factory=list)
    decision_criteria: List[str] = field(default_factory=list)
    
    # Stakeholder involvement
    participating_stakeholders: List[uuid.UUID] = field(default_factory=list)
    stakeholder_roles: Dict[uuid.UUID, str] = field(default_factory=dict)
    representation_adequacy: Optional[float] = None  # 0-1 scale
    
    # Process stages
    information_sharing_stage: Optional[uuid.UUID] = None
    deliberation_stage: Optional[uuid.UUID] = None
    consensus_building_stage: Optional[uuid.UUID] = None
    decision_validation_stage: Optional[uuid.UUID] = None
    
    # Facilitation
    facilitator_id: Optional[uuid.UUID] = None
    facilitation_methods: List[str] = field(default_factory=list)
    ground_rules: List[str] = field(default_factory=list)
    
    # Information support
    background_information: List[str] = field(default_factory=list)
    expert_inputs: List[uuid.UUID] = field(default_factory=list)
    analytical_support: List[str] = field(default_factory=list)
    
    # Process dynamics
    participation_quality: Optional[float] = None  # 0-1 scale
    information_quality: Optional[float] = None    # 0-1 scale
    deliberation_depth: Optional[float] = None     # 0-1 scale
    
    # Consensus development
    initial_positions: Dict[uuid.UUID, str] = field(default_factory=dict)
    position_evolution: List[Dict[str, Any]] = field(default_factory=list)
    convergence_indicators: List[str] = field(default_factory=list)
    
    # Barriers and challenges
    participation_barriers: List[ParticipationBarrier] = field(default_factory=list)
    process_challenges: List[str] = field(default_factory=list)
    barrier_mitigation_strategies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Outcomes
    consensus_achieved: Optional[bool] = None
    consensus_level: Optional[float] = None  # Actual agreement level achieved
    dissenting_views: List[str] = field(default_factory=list)
    minority_positions: List[str] = field(default_factory=list)
    
    # Implementation support
    implementation_commitments: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    monitoring_agreements: List[str] = field(default_factory=list)
    review_mechanisms: List[str] = field(default_factory=list)
    
    # SFM integration
    matrix_consensus_implications: List[uuid.UUID] = field(default_factory=list)
    institutional_consensus_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_system_consensus_impact: List[uuid.UUID] = field(default_factory=list)
    
    def assess_consensus_feasibility(self) -> Dict[str, Any]:
        """Assess feasibility of achieving consensus."""
        feasibility_assessment = {
            'feasibility_score': 0.0,
            'supporting_factors': [],
            'hindering_factors': [],
            'recommendations': []
        }
        
        # Supporting factors
        supporting_score = 0.0
        if self.representation_adequacy and self.representation_adequacy > 0.8:
            feasibility_assessment['supporting_factors'].append('High stakeholder representation')
            supporting_score += 0.3
        
        if self.information_quality and self.information_quality > 0.7:
            feasibility_assessment['supporting_factors'].append('Good information quality')
            supporting_score += 0.2
        
        if self.facilitator_id:
            feasibility_assessment['supporting_factors'].append('Professional facilitation available')
            supporting_score += 0.2
        
        # Hindering factors
        hindering_score = 0.0
        if ParticipationBarrier.POWER_IMBALANCES in self.participation_barriers:
            feasibility_assessment['hindering_factors'].append('Significant power imbalances')
            hindering_score += 0.3
        
        if ParticipationBarrier.TRUST_DEFICITS in self.participation_barriers:
            feasibility_assessment['hindering_factors'].append('Low trust levels')
            hindering_score += 0.2
        
        if len(self.process_challenges) > 3:
            feasibility_assessment['hindering_factors'].append('Multiple process challenges')
            hindering_score += 0.2
        
        # Calculate overall feasibility
        feasibility_assessment['feasibility_score'] = max(0.0, 0.5 + supporting_score - hindering_score)
        
        # Generate recommendations
        if feasibility_assessment['feasibility_score'] < 0.5:
            feasibility_assessment['recommendations'].extend([
                'Address major participation barriers before proceeding',
                'Consider alternative decision-making approaches',
                'Invest in trust-building activities'
            ])
        elif feasibility_assessment['feasibility_score'] < 0.7:
            feasibility_assessment['recommendations'].extend([
                'Strengthen facilitation support',
                'Improve information provision',
                'Address identified barriers'
            ])
        
        return feasibility_assessment
    
    def track_consensus_development(self) -> Dict[str, Any]:
        """Track the development of consensus over time."""
        development_tracking = {
            'consensus_trajectory': 'unknown',
            'convergence_rate': 0.0,
            'stability_indicators': [],
            'breakthrough_moments': [],
            'next_steps': []
        }
        
        # Analyze position evolution
        if len(self.position_evolution) >= 2:
            # Calculate convergence rate (simplified)
            initial_diversity = len(set(pos for pos in self.initial_positions.values()))
            recent_positions = self.position_evolution[-1].get('positions', {})
            current_diversity = len(set(pos for pos in recent_positions.values()))
            
            if current_diversity < initial_diversity:
                development_tracking['consensus_trajectory'] = 'converging'
                development_tracking['convergence_rate'] = (initial_diversity - current_diversity) / initial_diversity
            elif current_diversity == initial_diversity:
                development_tracking['consensus_trajectory'] = 'stable'
            else:
                development_tracking['consensus_trajectory'] = 'diverging'
        
        # Identify stability indicators
        if self.consensus_level and self.consensus_level > 0.8:
            development_tracking['stability_indicators'].append('High consensus level achieved')
        
        if len(self.dissenting_views) <= 1:
            development_tracking['stability_indicators'].append('Few dissenting views')
        
        # Generate next steps
        if development_tracking['consensus_trajectory'] == 'converging':
            development_tracking['next_steps'] = [
                'Continue current process',
                'Focus on remaining disagreements',
                'Prepare for decision validation'
            ]
        elif development_tracking['consensus_trajectory'] == 'stable':
            development_tracking['next_steps'] = [
                'Identify breakthrough opportunities',
                'Consider modified consensus approach',
                'Explore creative alternatives'
            ]
        else:  # diverging
            development_tracking['next_steps'] = [
                'Reassess process design',
                'Address fundamental disagreements',
                'Consider alternative approaches'
            ]
        
        return development_tracking


@dataclass
class DeliberationProcess(Node):
    """Democratic deliberation and dialogue process."""
    
    deliberation_purpose: Optional[str] = None
    deliberation_scope: Optional[str] = None
    
    # Process design
    deliberation_format: Optional[str] = None  # "town_hall", "focus_groups", "panels", etc.
    duration: Optional[timedelta] = None
    session_structure: List[str] = field(default_factory=list)
    
    # Participant composition
    deliberation_participants: List[uuid.UUID] = field(default_factory=list)
    participant_selection_method: Optional[str] = None
    demographic_representation: Dict[str, float] = field(default_factory=dict)
    
    # Information provision
    briefing_materials: List[str] = field(default_factory=list)
    expert_presentations: List[uuid.UUID] = field(default_factory=list)
    information_balance: Optional[float] = None  # Balanced presentation (0-1)
    
    # Deliberation quality
    deliberation_quality: DeliberationQuality = DeliberationQuality.FAIR
    participation_equality: Optional[float] = None  # Equal participation (0-1)
    argument_quality: Optional[float] = None       # Quality of arguments (0-1)
    listening_quality: Optional[float] = None      # Quality of listening (0-1)
    
    # Process dynamics
    dialogue_patterns: List[str] = field(default_factory=list)
    perspective_sharing: Optional[float] = None  # Diversity of views shared (0-1)
    mutual_learning: Optional[float] = None      # Evidence of learning (0-1)
    
    # Outcomes
    shared_understandings: List[str] = field(default_factory=list)
    identified_common_ground: List[str] = field(default_factory=list)
    clarified_differences: List[str] = field(default_factory=list)
    generated_options: List[str] = field(default_factory=list)
    
    # Follow-up
    deliberation_outputs: List[str] = field(default_factory=list)
    participant_feedback: Dict[uuid.UUID, str] = field(default_factory=dict)
    implementation_recommendations: List[str] = field(default_factory=list)
    
    # SFM context
    matrix_deliberation_focus: List[uuid.UUID] = field(default_factory=list)
    institutional_deliberation_implications: List[uuid.UUID] = field(default_factory=list)
    delivery_deliberation_connections: List[uuid.UUID] = field(default_factory=list)
    
    def evaluate_deliberation_quality(self) -> Dict[str, Any]:
        """Evaluate the quality of deliberative process."""
        quality_evaluation = {
            'overall_quality_score': 0.0,
            'quality_dimensions': {},
            'strengths': [],
            'weaknesses': [],
            'improvement_recommendations': []
        }
        
        # Quality dimensions
        dimensions = {
            'participation_equality': self.participation_equality or 0.5,
            'argument_quality': self.argument_quality or 0.5,
            'listening_quality': self.listening_quality or 0.5,
            'information_balance': self.information_balance or 0.5,
            'perspective_sharing': self.perspective_sharing or 0.5,
            'mutual_learning': self.mutual_learning or 0.5
        }
        
        quality_evaluation['quality_dimensions'] = dimensions
        
        # Calculate overall quality
        overall_quality = sum(dimensions.values()) / len(dimensions)
        quality_evaluation['overall_quality_score'] = overall_quality
        
        # Identify strengths and weaknesses
        for dimension, score in dimensions.items():
            if score >= 0.8:
                quality_evaluation['strengths'].append(f"Strong {dimension.replace('_', ' ')}")
            elif score <= 0.4:
                quality_evaluation['weaknesses'].append(f"Weak {dimension.replace('_', ' ')}")
        
        # Generate improvement recommendations
        if overall_quality < 0.6:
            quality_evaluation['improvement_recommendations'].extend([
                'Enhance facilitation training',
                'Improve participant preparation',
                'Balance information presentation',
                'Create more inclusive dialogue structures'
            ])
        
        return quality_evaluation


@dataclass
class ParticipationFramework(Node):
    """Comprehensive framework for stakeholder participation."""
    
    participation_scope: Optional[str] = None
    participation_objectives: List[str] = field(default_factory=list)
    
    # Stakeholder mapping
    stakeholder_universe: List[uuid.UUID] = field(default_factory=list)
    participation_levels: Dict[uuid.UUID, ParticipationLevel] = field(default_factory=dict)
    stakeholder_interests: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    
    # Participation design
    participation_methods: List[str] = field(default_factory=list)
    engagement_channels: List[str] = field(default_factory=list)
    accessibility_provisions: List[str] = field(default_factory=list)
    
    # Barrier analysis
    identified_barriers: List[ParticipationBarrier] = field(default_factory=list)
    barrier_impacts: Dict[ParticipationBarrier, str] = field(default_factory=dict)
    mitigation_strategies: Dict[ParticipationBarrier, List[str]] = field(default_factory=dict)
    
    # Capacity building
    capacity_building_needs: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    training_programs: List[str] = field(default_factory=list)
    resource_support: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    
    # Quality assurance
    participation_standards: List[str] = field(default_factory=list)
    quality_indicators: Dict[str, float] = field(default_factory=dict)
    monitoring_mechanisms: List[str] = field(default_factory=list)
    
    # Outcomes
    participation_effectiveness: Optional[float] = None  # 0-1 scale
    stakeholder_satisfaction: Dict[uuid.UUID, float] = field(default_factory=dict)
    influence_on_decisions: Optional[float] = None  # 0-1 scale
    
    # Continuous improvement
    feedback_mechanisms: List[str] = field(default_factory=list)
    adaptation_strategies: List[str] = field(default_factory=list)
    learning_integration: List[str] = field(default_factory=list)
    
    def assess_participation_effectiveness(self) -> Dict[str, Any]:
        """Assess effectiveness of participation framework."""
        effectiveness_assessment = {
            'effectiveness_score': 0.0,
            'high_performing_areas': [],
            'improvement_areas': [],
            'stakeholder_feedback_summary': {},
            'recommendations': []
        }
        
        # Calculate effectiveness dimensions
        dimensions = {}
        
        # Inclusiveness
        if self.stakeholder_universe and self.participation_levels:
            participating_stakeholders = len([s for s, level in self.participation_levels.items() 
                                            if level != ParticipationLevel.INFORMED])
            inclusiveness = participating_stakeholders / len(self.stakeholder_universe)
            dimensions['inclusiveness'] = inclusiveness
        
        # Satisfaction
        if self.stakeholder_satisfaction:
            avg_satisfaction = sum(self.stakeholder_satisfaction.values()) / len(self.stakeholder_satisfaction)
            dimensions['satisfaction'] = avg_satisfaction
        
        # Influence
        if self.influence_on_decisions:
            dimensions['influence'] = self.influence_on_decisions
        
        # Overall effectiveness
        if dimensions:
            effectiveness_assessment['effectiveness_score'] = sum(dimensions.values()) / len(dimensions)
            self.participation_effectiveness = effectiveness_assessment['effectiveness_score']
        
        # Identify high-performing and improvement areas
        for dimension, score in dimensions.items():
            if score >= 0.8:
                effectiveness_assessment['high_performing_areas'].append(dimension)
            elif score <= 0.5:
                effectiveness_assessment['improvement_areas'].append(dimension)
        
        # Generate recommendations
        if 'inclusiveness' in effectiveness_assessment['improvement_areas']:
            effectiveness_assessment['recommendations'].append('Expand stakeholder outreach and engagement')
        
        if 'satisfaction' in effectiveness_assessment['improvement_areas']:
            effectiveness_assessment['recommendations'].append('Address stakeholder concerns and improve process design')
        
        if 'influence' in effectiveness_assessment['improvement_areas']:
            effectiveness_assessment['recommendations'].append('Strengthen connection between participation and decision-making')
        
        return effectiveness_assessment


@dataclass
class DemocraticDecisionMaking(Node):
    """Democratic decision-making process within SFM framework."""
    
    decision_scope: Optional[str] = None
    decision_authority: Optional[uuid.UUID] = None
    decision_timeline: Optional[TimeSlice] = None
    
    # Decision process design
    decision_approach: DecisionMakingApproach = DecisionMakingApproach.CONSENSUS_BUILDING
    voting_mechanisms: List[str] = field(default_factory=list)
    decision_criteria: List[str] = field(default_factory=list)
    
    # Stakeholder involvement
    decision_participants: List[uuid.UUID] = field(default_factory=list)
    voting_weights: Dict[uuid.UUID, float] = field(default_factory=dict)
    representation_model: Optional[str] = None
    
    # Information base
    decision_information: List[str] = field(default_factory=list)
    analytical_inputs: List[uuid.UUID] = field(default_factory=list)
    expert_advice: List[uuid.UUID] = field(default_factory=list)
    
    # Democratic principles
    transparency_level: Optional[float] = None      # 0-1 scale
    accountability_mechanisms: List[str] = field(default_factory=list)
    legitimacy_sources: List[str] = field(default_factory=list)
    
    # Process outcomes
    decision_outcome: Optional[str] = None
    support_level: Optional[float] = None  # Level of support (0-1)
    implementation_feasibility: Optional[float] = None  # 0-1 scale
    
    # Democratic quality
    procedural_fairness: Optional[float] = None     # 0-1 scale
    substantive_equality: Optional[float] = None    # 0-1 scale
    minority_protection: Optional[float] = None     # 0-1 scale
    
    # Implementation
    implementation_plan: List[str] = field(default_factory=list)
    accountability_measures: List[str] = field(default_factory=list)
    review_mechanisms: List[str] = field(default_factory=list)
    
    def evaluate_democratic_quality(self) -> Dict[str, Any]:
        """Evaluate democratic quality of decision-making process."""
        democratic_evaluation = {
            'overall_democratic_score': 0.0,
            'democratic_dimensions': {},
            'democratic_strengths': [],
            'democratic_deficits': [],
            'enhancement_recommendations': []
        }
        
        # Democratic dimensions
        dimensions = {
            'transparency': self.transparency_level or 0.5,
            'procedural_fairness': self.procedural_fairness or 0.5,
            'substantive_equality': self.substantive_equality or 0.5,
            'minority_protection': self.minority_protection or 0.5
        }
        
        democratic_evaluation['democratic_dimensions'] = dimensions
        
        # Calculate overall democratic score
        overall_score = sum(dimensions.values()) / len(dimensions)
        democratic_evaluation['overall_democratic_score'] = overall_score
        
        # Identify strengths and deficits
        for dimension, score in dimensions.items():
            if score >= 0.8:
                democratic_evaluation['democratic_strengths'].append(f"Strong {dimension}")
            elif score <= 0.4:
                democratic_evaluation['democratic_deficits'].append(f"Weak {dimension}")
        
        # Generate enhancement recommendations
        if overall_score < 0.6:
            democratic_evaluation['enhancement_recommendations'].extend([
                'Strengthen transparency measures',
                'Improve procedural fairness',
                'Enhance minority voice protection',
                'Increase substantive equality'
            ])
        
        return democratic_evaluation


@dataclass
class ConflictMediation(Node):
    """Mediation process for resolving conflicts in democratic processes."""
    
    conflict_description: Optional[str] = None
    mediation_scope: Optional[str] = None
    
    # Conflict parties
    conflict_parties: List[uuid.UUID] = field(default_factory=list)
    mediator_id: Optional[uuid.UUID] = None
    neutral_observers: List[uuid.UUID] = field(default_factory=list)
    
    # Mediation design
    mediation_approach: ConflictResolutionMethod = ConflictResolutionMethod.COLLABORATIVE_PROBLEM_SOLVING
    mediation_phases: List[str] = field(default_factory=list)
    ground_rules: List[str] = field(default_factory=list)
    
    # Process support
    information_sharing: List[str] = field(default_factory=list)
    expert_facilitation: List[str] = field(default_factory=list)
    creative_techniques: List[str] = field(default_factory=list)
    
    # Mediation dynamics
    trust_building_progress: Optional[float] = None  # 0-1 scale
    communication_quality: Optional[float] = None   # 0-1 scale
    mutual_understanding: Optional[float] = None    # 0-1 scale
    
    # Outcomes
    mediation_outcome: MediationOutcome = MediationOutcome.MEDIATION_FAILED
    agreement_elements: List[str] = field(default_factory=list)
    remaining_disagreements: List[str] = field(default_factory=list)
    
    # Implementation
    agreement_implementation: List[str] = field(default_factory=list)
    monitoring_arrangements: List[str] = field(default_factory=list)
    review_schedules: List[datetime] = field(default_factory=list)
    
    def assess_mediation_success(self) -> Dict[str, Any]:
        """Assess success of mediation process."""
        success_assessment = {
            'success_score': 0.0,
            'success_factors': [],
            'limiting_factors': [],
            'sustainability_indicators': [],
            'follow_up_needs': []
        }
        
        # Calculate success score based on outcome
        outcome_scores = {
            MediationOutcome.FULL_AGREEMENT: 1.0,
            MediationOutcome.PARTIAL_AGREEMENT: 0.7,
            MediationOutcome.IMPROVED_UNDERSTANDING: 0.5,
            MediationOutcome.STRUCTURED_DISAGREEMENT: 0.3,
            MediationOutcome.MEDIATION_FAILED: 0.0
        }
        
        base_score = outcome_scores.get(self.mediation_outcome, 0.0)
        
        # Adjust based on process quality
        process_quality_factors = [
            self.trust_building_progress or 0.5,
            self.communication_quality or 0.5,
            self.mutual_understanding or 0.5
        ]
        process_adjustment = (sum(process_quality_factors) / len(process_quality_factors) - 0.5) * 0.3
        
        success_assessment['success_score'] = min(1.0, max(0.0, base_score + process_adjustment))
        
        # Identify success factors
        if self.trust_building_progress and self.trust_building_progress > 0.7:
            success_assessment['success_factors'].append('Strong trust building')
        
        if self.mediator_id:
            success_assessment['success_factors'].append('Professional mediation')
        
        if len(self.agreement_elements) > 0:
            success_assessment['success_factors'].append('Concrete agreements achieved')
        
        # Identify limiting factors
        if self.communication_quality and self.communication_quality < 0.5:
            success_assessment['limiting_factors'].append('Poor communication quality')
        
        if len(self.remaining_disagreements) > len(self.agreement_elements):
            success_assessment['limiting_factors'].append('More disagreements than agreements')
        
        # Sustainability indicators
        if self.monitoring_arrangements:
            success_assessment['sustainability_indicators'].append('Monitoring mechanisms established')
        
        if self.review_schedules:
            success_assessment['sustainability_indicators'].append('Regular review scheduled')
        
        # Follow-up needs
        if self.mediation_outcome in [MediationOutcome.PARTIAL_AGREEMENT, MediationOutcome.IMPROVED_UNDERSTANDING]:
            success_assessment['follow_up_needs'].extend([
                'Continue working on remaining issues',
                'Strengthen implementation support',
                'Monitor progress regularly'
            ])
        
        return success_assessment