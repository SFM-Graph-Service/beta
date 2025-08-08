"""
Enhanced Stakeholder Power Analysis Framework for Social Fabric Matrix analysis.

This module provides comprehensive models for analyzing stakeholder power,
influence networks, coalition formation, and power dynamics within institutional
systems. Essential for understanding how power shapes institutional outcomes
in SFM analysis following Hayden's institutional economics approach.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
# Note: PowerType, InfluenceStrategy, CoalitionType, and PowerDynamic are defined locally in this file rather than imported from models.sfm_enums

class PowerType(Enum):
    """Types of power in stakeholder analysis."""

    COERCIVE = auto()         # Power through force or threat
    REWARD = auto()           # Power through rewards/incentives
    LEGITIMATE = auto()       # Power through formal authority
    REFERENT = auto()         # Power through respect/admiration
    EXPERT = auto()           # Power through knowledge/expertise
    INFORMATION = auto()      # Power through information control
    CONNECTION = auto()       # Power through network position
    RESOURCE = auto()         # Power through resource control

class InfluenceStrategy(Enum):
    """Strategies for exercising influence."""

    RATIONAL_PERSUASION = auto()  # Logic and evidence
    INSPIRATIONAL_APPEALS = auto() # Values and ideals
    CONSULTATION = auto()         # Seeking input and involvement
    INGRATIATION = auto()         # Flattery and impression management
    EXCHANGE = auto()             # Offering benefits/favors
    COALITION_TACTICS = auto()    # Building alliances
    LEGITIMATING_TACTICS = auto() # Appealing to rules/precedent
    PRESSURE = auto()             # Demands and threats

class CoalitionType(Enum):
    """Types of stakeholder coalitions."""

    ADVOCACY_COALITION = auto()   # Policy advocacy coalitions
    ISSUE_COALITION = auto()      # Single-issue coalitions
    STRATEGIC_ALLIANCE = auto()   # Strategic partnerships
    BLOCKING_COALITION = auto()   # Coalitions to block action
    RESOURCE_COALITION = auto()   # Resource sharing coalitions
    KNOWLEDGE_COALITION = auto()  # Knowledge sharing networks

class PowerDynamic(Enum):
    """Types of power dynamics between stakeholders."""

    DOMINANCE = auto()           # One-sided power relationship
    INTERDEPENDENCE = auto()     # Mutual dependence
    COMPETITION = auto()         # Competitive relationship
    COLLABORATION = auto()       # Collaborative relationship
    CONFLICT = auto()            # Conflictual relationship
    NEGOTIATION = auto()         # Ongoing negotiation

@dataclass
class PowerAssessment(Node):  # pylint: disable=too-many-instance-attributes
    """Comprehensive power assessment for individual stakeholders."""

    stakeholder_id: Optional[uuid.UUID] = None
    assessment_date: Optional[datetime] = None

    # Power sources
    formal_authority: Optional[float] = None  # 0-1 scale
    resource_control: Optional[float] = None  # 0-1 scale
    expertise_level: Optional[float] = None  # 0-1 scale
    network_position: Optional[float] = None  # 0-1 scale
    legitimacy_level: Optional[float] = None  # 0-1 scale

    # Power types breakdown
    power_by_type: Dict[PowerType, float] = field(default_factory=dict)
    dominant_power_type: Optional[PowerType] = None

    # Influence capabilities
    influence_reach: Optional[float] = None  # Geographic/organizational reach
    influence_intensity: Optional[float] = None  # Strength of influence
    influence_strategies: List[InfluenceStrategy] = field(default_factory=list)
    influence_success_rate: Optional[float] = None  # Historical success rate

    # Relationship-based power
    alliance_strength: Optional[float] = None  # Strength of alliances
    coalition_leadership: List[uuid.UUID] = field(default_factory=list)
    dependency_relationships: Dict[uuid.UUID, float] = field(default_factory=dict)

    # Power constraints
    power_limitations: List[str] = field(default_factory=list)
    accountability_constraints: List[str] = field(default_factory=list)
    resource_constraints: List[str] = field(default_factory=list)

    # Power dynamics
    power_trajectory: Optional[str] = None  # "Increasing", "Stable", "Declining"
    power_volatility: Optional[float] = None  # 0-1 scale
    power_sustainability: Optional[float] = None  # 0-1 scale

    # SFM integration
    matrix_influence: List[uuid.UUID] = field(default_factory=list)  # Matrix cells influenced
    delivery_control: List[uuid.UUID] = field(default_factory=list)  # Deliveries controlled
    institutional_influence: List[uuid.UUID] = field(default_factory=list)

    def calculate_overall_power(self) -> Optional[float]:
        """Calculate overall power score."""
        power_dimensions = [
            self.formal_authority, self.resource_control, self.expertise_level,
            self.network_position, self.legitimacy_level
        ]
        valid_dimensions = [p for p in power_dimensions if p is not None]

        if not valid_dimensions:
            return None

        # Weighted average with network position having higher weight
        weights = [0.25, 0.25, 0.2, 0.3, 0.2][:len(valid_dimensions)]
        return sum(p * w for p, w in zip(valid_dimensions, weights)) / sum(weights)

@dataclass
class InfluenceNetwork(Node):  # pylint: disable=too-many-instance-attributes
    """Models networks of influence relationships between stakeholders."""

    network_name: Optional[str] = None
    network_domain: Optional[str] = None  # Issue area or domain

    # Network composition
    network_members: List[uuid.UUID] = field(default_factory=list)
    influence_relationships: List[uuid.UUID] = field(default_factory=list)

    # Network structure
    network_density: Optional[float] = None  # 0-1 scale
    centralization: Optional[float] = None  # Network centralization
    clustering_coefficient: Optional[float] = None  # Local clustering
    average_path_length: Optional[float] = None  # Average shortest path

    # Network characteristics
    hierarchy_level: Optional[float] = None  # 0-1 scale
    reciprocity_level: Optional[float] = None  # Mutual influence level
    trust_level: Optional[float] = None  # Overall network trust
    information_flow_quality: Optional[float] = None  # Quality of information flow

    # Power distribution
    power_concentration: Optional[float] = None  # How concentrated power is
    core_periphery_structure: Optional[bool] = None  # Core-periphery pattern
    influential_brokers: List[uuid.UUID] = field(default_factory=list)

    # Network dynamics
    network_stability: Optional[float] = None  # Stability over time
    change_frequency: Optional[float] = None  # Rate of network change
    adaptation_capacity: Optional[float] = None  # Ability to adapt

    # Performance
    collective_action_capacity: Optional[float] = None  # 0-1 scale
    conflict_resolution_capacity: Optional[float] = None  # 0-1 scale
    innovation_generation: Optional[float] = None  # Innovation capacity

    # SFM integration
    matrix_network_influence: Optional[float] = None
    delivery_network_overlap: List[uuid.UUID] = field(default_factory=list)
    institutional_network_effects: List[str] = field(default_factory=list)

@dataclass
class StakeholderCoalition(Node):  # pylint: disable=too-many-instance-attributes
    """Models coalitions formed by stakeholders for collective action."""

    coalition_type: Optional[CoalitionType] = None
    coalition_purpose: Optional[str] = None
    formation_date: Optional[datetime] = None

    # Coalition composition
    core_members: List[uuid.UUID] = field(default_factory=list)
    supporting_members: List[uuid.UUID] = field(default_factory=list)
    coalition_leader: Optional[uuid.UUID] = None
    member_roles: Dict[uuid.UUID, str] = field(default_factory=dict)

    # Coalition characteristics
    cohesion_level: Optional[float] = None  # Internal cohesion (0-1)
    resource_pooling: Optional[float] = None  # Resource sharing level
    decision_making_process: Optional[str] = None  # How decisions are made
    communication_frequency: Optional[str] = None

    # Coalition power
    collective_power: Optional[float] = None  # Combined power (0-1)
    power_amplification: Optional[float] = None  # Power increase through coalition
    external_influence: Optional[float] = None  # Influence on external actors

    # Coalition activities
    advocacy_activities: List[str] = field(default_factory=list)
    resource_mobilization: List[str] = field(default_factory=list)
    opposition_activities: List[str] = field(default_factory=list)

    # Coalition effectiveness
    goal_achievement: Optional[float] = None  # Success in achieving goals (0-1)
    member_satisfaction: Dict[uuid.UUID, float] = field(default_factory=dict)
    external_recognition: Optional[float] = None  # External recognition level

    # Coalition challenges
    internal_conflicts: List[str] = field(default_factory=list)
    free_rider_problems: Optional[float] = None  # Extent of free-riding
    coordination_costs: Optional[float] = None  # Cost of coordination

    # Coalition evolution
    lifecycle_stage: Optional[str] = None  # Formation, Growth, Maturity, Decline
    durability_factors: List[str] = field(default_factory=list)
    dissolution_risks: List[str] = field(default_factory=list)

    # SFM integration
    matrix_coalition_impact: List[uuid.UUID] = field(default_factory=list)
    delivery_system_influence: List[uuid.UUID] = field(default_factory=list)
    institutional_change_advocacy: List[str] = field(default_factory=list)

@dataclass
class PowerRelationship(Node):  # pylint: disable=too-many-instance-attributes
    """Models individual power relationships between stakeholders."""

    power_holder: Optional[uuid.UUID] = None  # Actor with power
    power_target: Optional[uuid.UUID] = None  # Actor subject to power
    relationship_type: Optional[PowerDynamic] = None

    # Relationship characteristics
    power_asymmetry: Optional[float] = None  # Degree of asymmetry (0-1)
    dependency_level: Optional[float] = None  # Target's dependency (0-1)
    reciprocity_level: Optional[float] = None  # Mutual influence (0-1)

    # Power exercise
    influence_frequency: Optional[str] = None  # How often influence is exercised
    influence_methods: List[InfluenceStrategy] = field(default_factory=list)
    resistance_level: Optional[float] = None  # Target's resistance (0-1)
    compliance_rate: Optional[float] = None  # Compliance with influence (0-1)

    # Relationship dynamics
    trust_level: Optional[float] = None  # Trust between parties (0-1)
    conflict_intensity: Optional[float] = None  # Level of conflict (0-1)
    cooperation_level: Optional[float] = None  # Level of cooperation (0-1)

    # Resource exchange
    resource_flows: Dict[str, float] = field(default_factory=dict)
    benefit_distribution: Dict[str, float] = field(default_factory=dict)
    cost_distribution: Dict[str, float] = field(default_factory=dict)

    # Relationship outcomes
    relationship_satisfaction: Dict[uuid.UUID, float] = field(default_factory=dict)
    mutual_benefits: List[str] = field(default_factory=list)
    negative_consequences: List[str] = field(default_factory=list)

    # Evolution and stability
    relationship_duration: Optional[float] = None  # Duration in days
    relationship_stability: Optional[float] = None  # Stability over time (0-1)
    change_drivers: List[str] = field(default_factory=list)

    # SFM context
    matrix_relationship_basis: List[uuid.UUID] = field(default_factory=list)
    delivery_relationship_overlap: List[uuid.UUID] = field(default_factory=list)
    institutional_relationship_foundation: List[uuid.UUID] = field(default_factory=list)

@dataclass
class PowerMap(Node):  # pylint: disable=too-many-instance-attributes
    """Comprehensive mapping of power relationships in a system."""

    system_domain: Optional[str] = None  # Domain being mapped
    mapping_scope: Optional[str] = None  # Scope of the mapping

    # Power distribution
    stakeholder_power_levels: Dict[uuid.UUID, float] = field(default_factory=dict)
    power_concentration_index: Optional[float] = None  # Gini-like coefficient
    power_balance: Optional[str] = None  # "Balanced", "Concentrated", "Fragmented"

    # Key power holders
    dominant_stakeholders: List[uuid.UUID] = field(default_factory=list)
    veto_players: List[uuid.UUID] = field(default_factory=list)
    agenda_setters: List[uuid.UUID] = field(default_factory=list)
    influence_brokers: List[uuid.UUID] = field(default_factory=list)

    # Power dynamics
    power_shifts: List[Dict[str, Any]] = field(default_factory=list)
    emerging_power_centers: List[uuid.UUID] = field(default_factory=list)
    declining_power_centers: List[uuid.UUID] = field(default_factory=list)

    # Coalition landscape
    major_coalitions: List[uuid.UUID] = field(default_factory=list)
    coalition_competition: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)
    coalition_cooperation: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)

    # System characteristics
    system_stability: Optional[float] = None  # Overall system stability (0-1)
    change_potential: Optional[float] = None  # Potential for system change (0-1)
    democratic_quality: Optional[float] = None  # Democratic quality of power distribution

    # Intervention opportunities
    leverage_points: List[str] = field(default_factory=list)
    intervention_strategies: List[str] = field(default_factory=list)
    coalition_building_opportunities: List[str] = field(default_factory=list)

    # SFM integration
    matrix_power_integration: Optional[float] = None
    delivery_system_power_effects: List[uuid.UUID] = field(default_factory=list)
    institutional_power_foundation: List[uuid.UUID] = field(default_factory=list)

@dataclass
class PowerShift(Node):  # pylint: disable=too-many-instance-attributes
    """Models changes in power relationships over time."""

    shift_description: Optional[str] = None
    shift_direction: Optional[str] = None  # "Toward", "Away from" specific actors
    shift_magnitude: Optional[float] = None  # Magnitude of change (0-1)

    # Shift characteristics
    shift_speed: Optional[str] = None  # "Gradual", "Rapid", "Sudden"
    shift_scope: Optional[str] = None  # "Local", "Sectoral", "System-wide"
    shift_permanence: Optional[str] = None  # "Temporary", "Sustained", "Permanent"

    # Actors involved
    power_gainers: List[uuid.UUID] = field(default_factory=list)
    power_losers: List[uuid.UUID] = field(default_factory=list)
    shift_catalysts: List[uuid.UUID] = field(default_factory=list)

    # Shift drivers
    internal_drivers: List[str] = field(default_factory=list)
    external_drivers: List[str] = field(default_factory=list)
    technological_drivers: List[str] = field(default_factory=list)
    institutional_drivers: List[str] = field(default_factory=list)

    # Shift impacts
    system_impacts: List[str] = field(default_factory=list)
    stakeholder_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    institutional_impacts: List[str] = field(default_factory=list)

    # Responses and adaptations
    adaptation_strategies: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    resistance_efforts: List[str] = field(default_factory=list)
    accommodation_efforts: List[str] = field(default_factory=list)

    # Future implications
    future_power_trajectory: List[str] = field(default_factory=list)
    stabilization_mechanisms: List[str] = field(default_factory=list)
    further_change_potential: Optional[float] = None  # 0-1 scale

    # SFM integration
    matrix_power_shift_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_system_reconfigurations: List[uuid.UUID] = field(default_factory=list)
    institutional_adjustment_requirements: List[str] = field(default_factory=list)
