"""
Democratic Participation Framework for Social Fabric Matrix analysis.

This module implements democratic participation and governance structures
essential to Hayden's SFM framework, focusing on citizen engagement,
participatory decision-making, and democratic legitimacy in institutional analysis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
from models.sfm_enums import (
    DecisionMakingType,
    GovernanceMechanism
)
# Note: ParticipationLevel, DemocraticProcessType, and StakeholderRole are defined locally in this file rather than imported from models.sfm_enums

class ParticipationLevel(Enum):
    """Levels of democratic participation in SFM analysis."""

    INFORMATION = auto()        # Information sharing only
    CONSULTATION = auto()       # Feedback collection
    INVOLVEMENT = auto()        # Two-way communication
    COLLABORATION = auto()      # Joint decision-making
    EMPOWERMENT = auto()        # Citizen-led decisions

class DemocraticProcessType(Enum):
    """Types of democratic processes in institutional analysis."""

    DELIBERATIVE = auto()       # Deliberative democracy
    PARTICIPATORY = auto()      # Participatory democracy
    REPRESENTATIVE = auto()     # Representative democracy
    DIRECT = auto()            # Direct democracy
    COLLABORATIVE = auto()      # Collaborative governance
    CONSULTATIVE = auto()       # Consultative processes

class StakeholderRole(Enum):
    """Roles stakeholders can play in democratic processes."""

    INITIATOR = auto()         # Process initiator
    PARTICIPANT = auto()       # Active participant
    OBSERVER = auto()          # Observer/monitor
    FACILITATOR = auto()       # Process facilitator
    DECISION_MAKER = auto()    # Final decision authority
    IMPLEMENTER = auto()       # Implementation responsibility

@dataclass
class DemocraticProcess(Node):
    """Models democratic participation processes within SFM analysis."""

    process_type: Optional[DemocraticProcessType] = None
    participation_level: Optional[ParticipationLevel] = None
    governance_mechanism: Optional[GovernanceMechanism] = None

    # Stakeholder engagement
    stakeholder_roles: Dict[uuid.UUID, StakeholderRole] = field(default_factory=dict)
    participation_barriers: List[str] = field(default_factory=list)
    inclusion_mechanisms: List[str] = field(default_factory=list)

    # Process characteristics
    duration: Optional[float] = None  # Process duration in days
    frequency: Optional[str] = None   # How often process occurs
    resource_requirements: Dict[str, float] = field(default_factory=dict)

    # Legitimacy and effectiveness
    legitimacy_score: Optional[float] = None  # 0-1 scale
    effectiveness_score: Optional[float] = None  # 0-1 scale
    satisfaction_scores: Dict[uuid.UUID, float] = field(default_factory=dict)

    # SFM integration
    matrix_cells_influenced: List[uuid.UUID] = field(default_factory=list)
    institutional_impact: List[uuid.UUID] = field(default_factory=list)
    policy_outcomes: List[uuid.UUID] = field(default_factory=list)

@dataclass
class CitizenEngagement(Node):
    """Models citizen engagement mechanisms in SFM institutional analysis."""

    engagement_type: Optional[str] = None  # e.g., "Public Hearing", "Citizen Panel"
    target_demographics: List[str] = field(default_factory=list)
    reach: Optional[int] = None  # Number of people reached
    response_rate: Optional[float] = None  # Participation rate

    # Engagement quality
    representation_quality: Optional[float] = None  # 0-1 scale
    diversity_index: Optional[float] = None  # Demographic diversity
    knowledge_level: Optional[float] = None  # Participant knowledge level

    # Outcomes
    input_quality: Optional[float] = None  # Quality of citizen input
    influence_on_decisions: Optional[float] = None  # Actual influence
    satisfaction_level: Optional[float] = None  # Participant satisfaction

    # SFM context
    ceremonial_barriers: List[str] = field(default_factory=list)
    instrumental_opportunities: List[str] = field(default_factory=list)
    power_imbalances: Dict[str, float] = field(default_factory=dict)

@dataclass
class StakeholderConsultation(Node):
    """Models systematic stakeholder consultation processes."""

    consultation_stage: Optional[str] = None  # e.g., "Problem Definition", "Solution Design"
    stakeholder_mapping: Dict[uuid.UUID, str] = field(default_factory=dict)  # ID -> role
    consultation_methods: List[str] = field(default_factory=list)

    # Process design
    structured_questions: List[str] = field(default_factory=list)
    feedback_mechanisms: List[str] = field(default_factory=list)
    consensus_building_methods: List[str] = field(default_factory=list)

    # Results
    key_findings: List[str] = field(default_factory=list)
    consensus_areas: List[str] = field(default_factory=list)
    conflict_areas: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Implementation tracking
    recommendation_adoption_rate: Optional[float] = None
    stakeholder_satisfaction: Dict[uuid.UUID, float] = field(default_factory=dict)
    follow_up_required: bool = False

@dataclass
class ParticipatorBudgeting(Node):
    """Models participatory budgeting processes within SFM framework."""

    budget_amount: Optional[float] = None
    allocation_categories: List[str] = field(default_factory=list)
    participant_count: Optional[int] = None

    # Process structure
    proposal_submission_phase: Optional[datetime] = None
    deliberation_phase: Optional[datetime] = None
    voting_phase: Optional[datetime] = None

    # Proposals and voting
    proposals: List[Dict[str, any]] = field(default_factory=list)
    voting_results: Dict[str, float] = field(default_factory=dict)
    funded_projects: List[str] = field(default_factory=list)

    # Impact assessment
    community_impact: Optional[float] = None  # 0-1 scale
    democratic_learning: Optional[float] = None  # Civic capacity building
    implementation_success: Optional[float] = None  # Project completion rate

    # SFM analysis
    institutional_relationships: List[uuid.UUID] = field(default_factory=list)
    delivery_impacts: Dict[uuid.UUID, float] = field(default_factory=dict)
    matrix_cell_effects: List[uuid.UUID] = field(default_factory=list)

@dataclass
class DeliberativeProcess(Node):
    """Models deliberative democracy processes for SFM policy analysis."""

    deliberation_topic: Optional[str] = None
    participant_selection_method: Optional[str] = None
    participant_count: Optional[int] = None
    session_count: Optional[int] = None

    # Process design
    information_materials: List[str] = field(default_factory=list)
    expert_inputs: List[uuid.UUID] = field(default_factory=list)
    facilitation_methods: List[str] = field(default_factory=list)

    # Outcomes
    initial_opinions: Dict[str, float] = field(default_factory=dict)
    final_opinions: Dict[str, float] = field(default_factory=dict)
    opinion_change_magnitude: Optional[float] = None
    consensus_level: Optional[float] = None  # 0-1 scale

    # Quality measures
    information_quality: Optional[float] = None
    deliberation_quality: Optional[float] = None
    representativeness: Optional[float] = None

    # SFM integration
    policy_influence: Optional[float] = None
    institutional_legitimacy_impact: Optional[float] = None
    matrix_implications: List[str] = field(default_factory=list)

@dataclass
class GovernanceNetwork(Node):
    """Models multi-stakeholder governance networks in SFM analysis."""

    network_type: Optional[str] = None  # e.g., "Policy Network", "Issue Network"
    governance_structure: Optional[GovernanceMechanism] = None
    coordination_mechanisms: List[str] = field(default_factory=list)

    # Network composition
    member_institutions: List[uuid.UUID] = field(default_factory=list)
    member_roles: Dict[uuid.UUID, str] = field(default_factory=dict)
    power_distribution: Dict[uuid.UUID, float] = field(default_factory=dict)

    # Network dynamics
    collaboration_intensity: Optional[float] = None  # 0-1 scale
    trust_level: Optional[float] = None  # 0-1 scale
    information_sharing: Optional[float] = None  # 0-1 scale
    resource_sharing: Optional[float] = None  # 0-1 scale

    # Performance
    collective_action_capacity: Optional[float] = None
    problem_solving_effectiveness: Optional[float] = None
    democratic_accountability: Optional[float] = None

    # SFM context
    delivery_coordination: Dict[str, float] = field(default_factory=dict)
    institutional_adjustment_capacity: Optional[float] = None
    matrix_integration_level: Optional[float] = None

@dataclass
class DemocraticLegitimacy(Node):
    """Models democratic legitimacy within SFM institutional analysis."""

    legitimacy_type: Optional[str] = None  # e.g., "Input", "Output", "Throughput"
    source_of_legitimacy: List[str] = field(default_factory=list)

    # Legitimacy dimensions
    procedural_legitimacy: Optional[float] = None  # 0-1 scale
    substantive_legitimacy: Optional[float] = None  # 0-1 scale
    democratic_legitimacy: Optional[float] = None  # 0-1 scale

    # Assessment factors
    participation_quality: Optional[float] = None
    transparency_level: Optional[float] = None
    accountability_mechanisms: List[str] = field(default_factory=list)
    responsiveness_score: Optional[float] = None

    # Stakeholder perceptions
    citizen_trust: Optional[float] = None  # 0-1 scale
    stakeholder_acceptance: Dict[uuid.UUID, float] = field(default_factory=dict)
    media_coverage_tone: Optional[float] = None  # -1 to 1 scale

    # SFM implications
    institutional_stability: Optional[float] = None
    policy_implementation_capacity: Optional[float] = None
    matrix_relationship_strength: Optional[float] = None

@dataclass
class CivicCapacity(Node):
    """Models civic capacity development within SFM framework."""

    capacity_type: Optional[str] = None  # e.g., "Individual", "Organizational", "Community"
    skill_areas: List[str] = field(default_factory=list)
    development_stage: Optional[str] = None

    # Capacity dimensions
    knowledge_level: Optional[float] = None  # 0-1 scale
    skill_level: Optional[float] = None  # 0-1 scale
    motivation_level: Optional[float] = None  # 0-1 scale
    resource_access: Optional[float] = None  # 0-1 scale

    # Development mechanisms
    training_programs: List[str] = field(default_factory=list)
    mentorship_relationships: List[uuid.UUID] = field(default_factory=list)
    practice_opportunities: List[str] = field(default_factory=list)

    # Outcomes
    participation_frequency: Optional[float] = None
    leadership_roles_taken: Optional[int] = None
    civic_engagement_level: Optional[float] = None

    # SFM integration
    institutional_contribution: List[uuid.UUID] = field(default_factory=list)
    delivery_system_participation: List[uuid.UUID] = field(default_factory=list)
    democratic_process_leadership: List[uuid.UUID] = field(default_factory=list)
