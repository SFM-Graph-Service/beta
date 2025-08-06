"""
Conflict Resolution Framework for Social Fabric Matrix analysis.

This module models conflict analysis, resolution processes, mediation mechanisms,
and dispute resolution within institutional systems. Essential for understanding
how conflicts arise and are resolved in complex socio-economic systems following
Hayden's institutional economics approach.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
# Note: ConflictType, ConflictIntensity, ResolutionMechanism, and MediationStyle
# are defined locally in this file rather than imported from models.sfm_enums

class ConflictType(Enum):
    """Types of conflicts in institutional systems."""

    RESOURCE_CONFLICT = auto()       # Conflicts over resource allocation
    VALUE_CONFLICT = auto()          # Conflicts over values and beliefs
    INTEREST_CONFLICT = auto()       # Conflicts over competing interests
    PROCEDURAL_CONFLICT = auto()     # Conflicts over procedures and processes
    POWER_CONFLICT = auto()          # Conflicts over power and authority
    IDENTITY_CONFLICT = auto()       # Conflicts over identity and recognition
    STRUCTURAL_CONFLICT = auto()     # Conflicts due to structural inequalities

class ConflictIntensity(Enum):
    """Levels of conflict intensity."""

    LATENT = auto()                 # Hidden or potential conflict
    MANIFEST = auto()               # Open, visible conflict
    LOW_INTENSITY = auto()          # Low-level ongoing conflict
    MODERATE_INTENSITY = auto()     # Moderate conflict with periodic escalation
    HIGH_INTENSITY = auto()         # High-intensity, disruptive conflict
    CRISIS = auto()                 # Crisis-level conflict threatening system

class ResolutionMechanism(Enum):
    """Mechanisms for resolving conflicts."""

    NEGOTIATION = auto()            # Direct negotiation between parties
    MEDIATION = auto()              # Third-party mediated resolution
    ARBITRATION = auto()            # Binding third-party decision
    ADJUDICATION = auto()           # Formal legal/judicial resolution
    COLLABORATIVE_PROBLEM_SOLVING = auto()  # Joint problem-solving approach
    POWER_SETTLEMENT = auto()       # Resolution through power dynamics
    AVOIDANCE = auto()              # Conflict avoidance strategies

class MediationStyle(Enum):
    """Styles of mediation in conflict resolution."""

    FACILITATIVE = auto()           # Facilitating communication and understanding
    EVALUATIVE = auto()             # Evaluating merits and providing opinions
    TRANSFORMATIVE = auto()         # Transforming relationships and perspectives
    DIRECTIVE = auto()              # Directive approach with specific guidance
    NON_DIRECTIVE = auto()          # Non-directive, party-led process

class ConflictOutcome(Enum):
    """Possible outcomes of conflict resolution processes."""

    WIN_WIN = auto()                # Mutual benefit outcome
    WIN_LOSE = auto()               # One party wins, other loses
    LOSE_LOSE = auto()              # Both parties lose
    COMPROMISE = auto()             # Mutual concessions
    ACCOMMODATION = auto()          # One party accommodates the other
    STALEMATE = auto()              # No resolution achieved
    ESCALATION = auto()             # Conflict escalates further

@dataclass
class Conflict(Node):
    """Models individual conflicts within institutional systems."""

    conflict_type: Optional[ConflictType] = None
    conflict_intensity: Optional[ConflictIntensity] = None
    conflict_duration: Optional[timedelta] = None

    # Conflict parties
    primary_parties: List[uuid.UUID] = field(default_factory=list)
    secondary_parties: List[uuid.UUID] = field(default_factory=list)
    affected_stakeholders: List[uuid.UUID] = field(default_factory=list)

    # Conflict issues
    core_issues: List[str] = field(default_factory=list)
    underlying_interests: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    stated_positions: Dict[uuid.UUID, List[str]] = field(default_factory=dict)

    # Conflict dynamics
    escalation_factors: List[str] = field(default_factory=list)
    de_escalation_factors: List[str] = field(default_factory=list)
    conflict_triggers: List[str] = field(default_factory=list)
    power_imbalances: Dict[str, float] = field(default_factory=dict)

    # Conflict context
    institutional_context: List[uuid.UUID] = field(default_factory=list)
    historical_background: Optional[str] = None
    cultural_factors: List[str] = field(default_factory=list)
    environmental_pressures: List[str] = field(default_factory=list)

    # Conflict costs
    direct_costs: Dict[str, float] = field(default_factory=dict)
    opportunity_costs: Dict[str, float] = field(default_factory=dict)
    relationship_costs: Optional[float] = None  # 0-1 scale
    system_costs: Optional[float] = None  # Impact on broader system

    # Resolution attempts
    resolution_attempts: List[uuid.UUID] = field(default_factory=list)
    current_resolution_status: Optional[str] = None
    resolution_barriers: List[str] = field(default_factory=list)

    # SFM integration
    matrix_conflict_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_system_disruptions: List[uuid.UUID] = field(default_factory=list)
    institutional_stability_impact: Optional[float] = None  # 0-1 scale

@dataclass
class ConflictAnalysis(Node):
    """Systematic analysis of conflict situations."""

    conflict_id: Optional[uuid.UUID] = None
    analysis_date: Optional[datetime] = None
    analyst: Optional[uuid.UUID] = None

    # Stakeholder analysis
    stakeholder_mapping: Dict[uuid.UUID, str] = field(default_factory=dict)  # ID -> role
    stakeholder_interests: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    stakeholder_positions: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    stakeholder_power: Dict[uuid.UUID, float] = field(default_factory=dict)

    # Issue analysis
    issue_hierarchy: List[str] = field(default_factory=list)  # From most to least important
    issue_relationships: Dict[str, List[str]] = field(default_factory=dict)
    negotiable_issues: List[str] = field(default_factory=list)
    non_negotiable_issues: List[str] = field(default_factory=list)

    # Conflict assessment
    conflict_complexity: Optional[float] = None  # 0-1 scale
    resolution_difficulty: Optional[float] = None  # 0-1 scale
    time_pressure: Optional[float] = None  # 0-1 scale
    resource_availability: Dict[str, float] = field(default_factory=dict)

    # System impact analysis
    system_disruption_level: Optional[float] = None  # 0-1 scale
    institutional_impact: Dict[uuid.UUID, float] = field(default_factory=dict)
    network_effects: List[str] = field(default_factory=list)

    # Resolution prospects
    resolution_potential: Optional[float] = None  # 0-1 scale
    recommended_approaches: List[ResolutionMechanism] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

    # SFM context
    matrix_analysis_implications: List[str] = field(default_factory=list)
    delivery_system_vulnerability: Dict[uuid.UUID, float] = field(default_factory=dict)
    institutional_adjustment_needs: List[str] = field(default_factory=list)

@dataclass
class ResolutionProcess(Node):
    """Models conflict resolution processes and interventions."""

    conflict_id: Optional[uuid.UUID] = None
    resolution_mechanism: Optional[ResolutionMechanism] = None
    process_facilitator: Optional[uuid.UUID] = None

    # Process design
    process_structure: List[str] = field(default_factory=list)
    participation_rules: List[str] = field(default_factory=list)
    communication_protocols: List[str] = field(default_factory=list)
    decision_making_rules: List[str] = field(default_factory=list)

    # Process participants
    direct_participants: List[uuid.UUID] = field(default_factory=list)
    observers: List[uuid.UUID] = field(default_factory=list)
    advisors: List[uuid.UUID] = field(default_factory=list)

    # Process stages
    preparation_phase: Dict[str, any] = field(default_factory=dict)
    exploration_phase: Dict[str, any] = field(default_factory=dict)
    negotiation_phase: Dict[str, any] = field(default_factory=dict)
    agreement_phase: Dict[str, any] = field(default_factory=dict)

    # Process characteristics
    process_duration: Optional[timedelta] = None
    process_costs: Dict[str, float] = field(default_factory=dict)
    process_transparency: Optional[float] = None  # 0-1 scale
    process_legitimacy: Optional[float] = None  # 0-1 scale

    # Process outcomes
    process_outcome: Optional[ConflictOutcome] = None
    agreement_terms: List[str] = field(default_factory=list)
    implementation_plan: List[str] = field(default_factory=list)
    monitoring_mechanisms: List[str] = field(default_factory=list)

    # Process evaluation
    participant_satisfaction: Dict[uuid.UUID, float] = field(default_factory=dict)
    process_effectiveness: Optional[float] = None  # 0-1 scale
    learning_outcomes: List[str] = field(default_factory=list)

    # SFM integration
    matrix_process_integration: Optional[float] = None
    delivery_system_process_effects: List[uuid.UUID] = field(default_factory=list)
    institutional_process_alignment: Dict[uuid.UUID, float] = field(default_factory=dict)

@dataclass
class Mediation(ResolutionProcess):
    """Specialized mediation process for conflict resolution."""

    def __post_init__(self):
        self.resolution_mechanism = ResolutionMechanism.MEDIATION

    mediation_style: Optional[MediationStyle] = None
    mediator_qualifications: List[str] = field(default_factory=list)
    mediator_neutrality: Optional[float] = None  # 0-1 scale

    # Mediation specific elements
    caucus_sessions: List[Dict[str, any]] = field(default_factory=list)
    joint_sessions: List[Dict[str, any]] = field(default_factory=list)
    reality_testing: List[str] = field(default_factory=list)
    option_generation: List[str] = field(default_factory=list)

    # Mediation techniques
    reframing_techniques: List[str] = field(default_factory=list)
    communication_facilitation: List[str] = field(default_factory=list)
    interest_exploration: List[str] = field(default_factory=list)

    # Mediation outcomes
    settlement_rate: Optional[float] = None  # 0-1 scale
    relationship_improvement: Optional[float] = None  # 0-1 scale
    future_conflict_prevention: Optional[float] = None  # 0-1 scale

@dataclass
class CollaborativeProblemSolving(ResolutionProcess):
    """Collaborative problem-solving approach to conflict resolution."""

    def __post_init__(self):
        self.resolution_mechanism = ResolutionMechanism.COLLABORATIVE_PROBLEM_SOLVING

    # Collaborative elements
    shared_problem_definition: Optional[str] = None
    joint_fact_finding: List[str] = field(default_factory=list)
    collaborative_analysis: List[str] = field(default_factory=list)
    solution_co_creation: List[str] = field(default_factory=list)

    # Problem-solving process
    problem_framing: List[str] = field(default_factory=list)
    information_gathering: List[str] = field(default_factory=list)
    option_development: List[str] = field(default_factory=list)
    solution_evaluation: List[str] = field(default_factory=list)

    # Collaboration quality
    trust_building: Optional[float] = None  # 0-1 scale
    mutual_understanding: Optional[float] = None  # 0-1 scale
    shared_commitment: Optional[float] = None  # 0-1 scale

    # Innovation outcomes
    creative_solutions: List[str] = field(default_factory=list)
    systemic_improvements: List[str] = field(default_factory=list)
    relationship_transformation: Optional[float] = None  # 0-1 scale

@dataclass
class ConflictPreventionSystem(Node):
    """Systems and mechanisms for preventing conflicts."""

    system_name: Optional[str] = None
    system_scope: Optional[str] = None
    prevention_focus: List[ConflictType] = field(default_factory=list)

    # Prevention mechanisms
    early_warning_systems: List[str] = field(default_factory=list)
    grievance_mechanisms: List[str] = field(default_factory=list)
    dialogue_forums: List[str] = field(default_factory=list)
    capacity_building_programs: List[str] = field(default_factory=list)

    # Structural prevention
    institutional_reforms: List[str] = field(default_factory=list)
    power_sharing_arrangements: List[str] = field(default_factory=list)
    resource_distribution_mechanisms: List[str] = field(default_factory=list)

    # Cultural prevention
    peace_education: List[str] = field(default_factory=list)
    cultural_sensitivity_training: List[str] = field(default_factory=list)
    intercultural_dialogue: List[str] = field(default_factory=list)

    # Prevention effectiveness
    conflict_reduction_rate: Optional[float] = None  # 0-1 scale
    early_intervention_success: Optional[float] = None  # 0-1 scale
    system_resilience: Optional[float] = None  # 0-1 scale

    # System sustainability
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    stakeholder_support: Dict[uuid.UUID, float] = field(default_factory=dict)
    institutional_embedding: Optional[float] = None  # 0-1 scale

    # SFM integration
    matrix_prevention_integration: Optional[float] = None
    delivery_system_stability_contribution: Optional[float] = None
    institutional_resilience_enhancement: Optional[float] = None

@dataclass
class ConflictTransformation(Node):
    """Models long-term conflict transformation processes."""

    transformation_scope: Optional[str] = None
    transformation_timeline: Optional[str] = None

    # Transformation dimensions
    structural_transformation: List[str] = field(default_factory=list)
    relational_transformation: List[str] = field(default_factory=list)
    cultural_transformation: List[str] = field(default_factory=list)
    personal_transformation: List[str] = field(default_factory=list)

    # Transformation process
    transformation_stages: List[str] = field(default_factory=list)
    key_milestones: List[Dict[str, any]] = field(default_factory=list)
    transformation_catalysts: List[str] = field(default_factory=list)

    # Transformation actors
    transformation_leaders: List[uuid.UUID] = field(default_factory=list)
    change_agents: List[uuid.UUID] = field(default_factory=list)
    transformation_constituencies: List[uuid.UUID] = field(default_factory=list)

    # Transformation outcomes
    system_changes: List[str] = field(default_factory=list)
    relationship_changes: List[str] = field(default_factory=list)
    capacity_changes: List[str] = field(default_factory=list)

    # Sustainability factors
    transformation_institutionalization: Optional[float] = None  # 0-1 scale
    cultural_embedding: Optional[float] = None  # 0-1 scale
    regenerative_capacity: Optional[float] = None  # 0-1 scale

    # SFM integration
    matrix_transformation_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_system_evolution: List[uuid.UUID] = field(default_factory=list)
    institutional_transformation_alignment: Dict[uuid.UUID, float] = field(default_factory=dict)

@dataclass
class DisputeResolutionInstitution(Node):
    """Models institutions specialized in dispute resolution."""

    institution_type: Optional[str] = None  # e.g., "Court", "Tribunal", "Ombudsman"
    jurisdiction: Optional[str] = None
    specialized_domains: List[str] = field(default_factory=list)

    # Institutional characteristics
    formal_authority: Optional[float] = None  # Level of formal authority (0-1)
    binding_power: Optional[bool] = None  # Can make binding decisions
    enforcement_capacity: Optional[float] = None  # Enforcement capability (0-1)

    # Resolution procedures
    procedural_rules: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    decision_criteria: List[str] = field(default_factory=list)
    appeal_mechanisms: List[str] = field(default_factory=list)

    # Performance metrics
    case_load: Optional[int] = None  # Number of cases handled
    resolution_rate: Optional[float] = None  # Cases resolved successfully
    average_resolution_time: Optional[timedelta] = None
    user_satisfaction: Optional[float] = None  # 0-1 scale

    # Access and equity
    accessibility_barriers: List[str] = field(default_factory=list)
    cost_barriers: List[str] = field(default_factory=list)
    equity_measures: List[str] = field(default_factory=list)

    # Institutional relationships
    partner_institutions: List[uuid.UUID] = field(default_factory=list)
    referral_networks: List[uuid.UUID] = field(default_factory=list)
    coordination_mechanisms: List[str] = field(default_factory=list)

    # SFM integration
    matrix_dispute_resolution_role: Optional[str] = None
    delivery_system_dispute_handling: List[uuid.UUID] = field(default_factory=list)
    institutional_legitimacy_contribution: Optional[float] = None
