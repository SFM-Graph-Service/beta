"""
Multi-Scale Integration Framework for Social Fabric Matrix analysis.

This module implements multi-scale integration capabilities for SFM analysis,
enabling the modeling of interactions across local, regional, national, and
global scales. Essential for understanding how institutional arrangements
and delivery systems operate across different organizational and geographical levels.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Tuple
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
from models.sfm_enums import InstitutionalScope
# Note: SystemLevel, ScaleType, IntegrationMechanism, and ScaleInteractionType are defined locally in this file rather than imported from models.sfm_enums


class SystemLevel(Enum):
    """Hierarchical levels in multi-scale systems."""
    
    INDIVIDUAL = auto()       # Individual actors
    ORGANIZATIONAL = auto()   # Organizations/firms
    COMMUNITY = auto()        # Local communities
    REGIONAL = auto()         # Regional systems
    NATIONAL = auto()         # National systems
    INTERNATIONAL = auto()    # International systems
    GLOBAL = auto()          # Global systems


class ScaleType(Enum):
    """Types of scale in institutional analysis."""
    
    SPATIAL = auto()         # Geographic/spatial scale
    TEMPORAL = auto()        # Time scale
    ORGANIZATIONAL = auto()   # Organizational hierarchy scale
    JURISDICTIONAL = auto()  # Legal/governance jurisdiction scale
    FUNCTIONAL = auto()      # Functional domain scale
    NETWORK = auto()         # Network reach scale


class IntegrationMechanism(Enum):
    """Mechanisms for cross-scale integration."""
    
    HIERARCHICAL = auto()    # Top-down hierarchical coordination
    NETWORK = auto()         # Network-based coordination
    MARKET = auto()          # Market-based coordination
    HYBRID = auto()          # Hybrid coordination mechanisms
    FEDERATION = auto()      # Federal coordination
    POLYCENTRICISM = auto()  # Polycentric governance


class ScaleInteractionType(Enum):
    """Types of interactions between scales."""
    
    UPWARD_CAUSATION = auto()    # Lower levels affect higher levels
    DOWNWARD_CAUSATION = auto()  # Higher levels affect lower levels
    SAME_LEVEL = auto()          # Interactions within same level
    CROSS_LEVEL = auto()         # Complex cross-level interactions
    EMERGENT = auto()            # Emergent cross-scale properties
    FEEDBACK = auto()            # Feedback loops across scales


@dataclass
class ScaleLevel(Node):
    """Models individual scale levels within multi-scale systems."""
    
    system_level: Optional[SystemLevel] = None
    scale_type: Optional[ScaleType] = None
    scale_description: Optional[str] = None
    
    # Scale characteristics
    spatial_extent: Optional[str] = None  # Geographic coverage
    temporal_extent: Optional[str] = None  # Time horizon
    population_size: Optional[int] = None  # Number of actors/entities
    resource_scope: Optional[str] = None  # Resource domain
    
    # Institutional arrangements
    governance_structures: List[uuid.UUID] = field(default_factory=list)
    formal_institutions: List[uuid.UUID] = field(default_factory=list)
    informal_institutions: List[uuid.UUID] = field(default_factory=list)
    
    # Actors and organizations
    key_actors: List[uuid.UUID] = field(default_factory=list)
    institutional_actors: List[uuid.UUID] = field(default_factory=list)
    stakeholder_groups: List[uuid.UUID] = field(default_factory=list)
    
    # Performance characteristics
    coordination_capacity: Optional[float] = None  # 0-1 scale
    autonomy_level: Optional[float] = None  # 0-1 scale
    resource_availability: Dict[str, float] = field(default_factory=dict)
    problem_solving_capacity: Optional[float] = None  # 0-1 scale
    
    # SFM integration
    matrix_cells: List[uuid.UUID] = field(default_factory=list)
    delivery_systems: List[uuid.UUID] = field(default_factory=list)
    institutional_relationships: List[uuid.UUID] = field(default_factory=list)


@dataclass
class ScaleInteraction(Node):
    """Models interactions between different scale levels."""
    
    source_level: Optional[uuid.UUID] = None  # Source scale level
    target_level: Optional[uuid.UUID] = None  # Target scale level
    interaction_type: Optional[ScaleInteractionType] = None
    
    # Interaction characteristics
    interaction_strength: Optional[float] = None  # 0-1 scale
    interaction_frequency: Optional[str] = None  # How often interaction occurs
    interaction_mechanisms: List[str] = field(default_factory=list)
    
    # Content of interaction
    resource_flows: Dict[str, float] = field(default_factory=dict)
    information_flows: Dict[str, float] = field(default_factory=dict)
    authority_relationships: List[str] = field(default_factory=list)
    coordination_activities: List[str] = field(default_factory=list)
    
    # Interaction outcomes
    coordination_effectiveness: Optional[float] = None  # 0-1 scale
    conflict_level: Optional[float] = None  # 0-1 scale
    mutual_adaptation: Optional[float] = None  # 0-1 scale
    learning_exchange: Optional[float] = None  # 0-1 scale
    
    # Barriers and facilitators
    interaction_barriers: List[str] = field(default_factory=list)
    interaction_facilitators: List[str] = field(default_factory=list)
    institutional_gaps: List[str] = field(default_factory=list)
    
    # SFM context
    delivery_coordination: List[uuid.UUID] = field(default_factory=list)
    matrix_integration_effects: List[str] = field(default_factory=list)
    institutional_adjustment_needs: List[str] = field(default_factory=list)


@dataclass
class MultiScaleSystem(Node):
    """Models complete multi-scale institutional systems."""
    
    system_name: Optional[str] = None
    system_domain: Optional[str] = None  # e.g., "Environmental", "Economic"
    integration_mechanism: Optional[IntegrationMechanism] = None
    
    # System composition
    scale_levels: List[uuid.UUID] = field(default_factory=list)
    scale_interactions: List[uuid.UUID] = field(default_factory=list)
    cross_scale_institutions: List[uuid.UUID] = field(default_factory=list)
    
    # System characteristics
    system_coherence: Optional[float] = None  # 0-1 scale
    integration_level: Optional[float] = None  # 0-1 scale
    adaptability: Optional[float] = None  # 0-1 scale
    stability: Optional[float] = None  # 0-1 scale
    
    # System performance
    collective_action_capacity: Optional[float] = None  # 0-1 scale
    problem_solving_effectiveness: Optional[float] = None  # 0-1 scale
    resource_allocation_efficiency: Optional[float] = None  # 0-1 scale
    
    # System dynamics
    emergence_patterns: List[str] = field(default_factory=list)
    feedback_loops: List[uuid.UUID] = field(default_factory=list)
    system_evolution: List[str] = field(default_factory=list)
    
    # Governance and coordination
    governance_arrangements: List[str] = field(default_factory=list)
    coordination_mechanisms: List[str] = field(default_factory=list)
    conflict_resolution_mechanisms: List[str] = field(default_factory=list)
    
    # SFM integration
    matrix_system_integration: Optional[float] = None  # 0-1 scale
    delivery_system_coordination: List[uuid.UUID] = field(default_factory=list)
    institutional_complementarity: Optional[float] = None  # 0-1 scale


@dataclass
class CrossScaleInstitution(Node):
    """Models institutions that operate across multiple scales."""
    
    institution_type: Optional[str] = None
    operating_scales: List[uuid.UUID] = field(default_factory=list)
    bridging_function: Optional[str] = None
    
    # Bridging characteristics
    scale_bridging_capacity: Optional[float] = None  # 0-1 scale
    information_brokerage: Optional[float] = None  # 0-1 scale
    resource_mobilization: Optional[float] = None  # 0-1 scale
    
    # Multi-scale operations
    scale_specific_functions: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    cross_scale_coordination: List[str] = field(default_factory=list)
    vertical_integration: Optional[float] = None  # 0-1 scale
    horizontal_integration: Optional[float] = None  # 0-1 scale
    
    # Performance across scales
    effectiveness_by_scale: Dict[uuid.UUID, float] = field(default_factory=dict)
    legitimacy_by_scale: Dict[uuid.UUID, float] = field(default_factory=dict)
    resource_efficiency: Optional[float] = None  # 0-1 scale
    
    # Challenges and opportunities
    scale_mismatches: List[str] = field(default_factory=list)
    coordination_challenges: List[str] = field(default_factory=list)
    integration_opportunities: List[str] = field(default_factory=list)
    
    # SFM integration
    matrix_bridging_role: Optional[str] = None
    delivery_coordination_function: List[uuid.UUID] = field(default_factory=list)
    institutional_innovation_potential: Optional[float] = None


@dataclass
class ScaleMismatch(Node):
    """Models mismatches between scales in institutional systems."""
    
    mismatch_type: Optional[str] = None  # e.g., "Spatial", "Temporal", "Functional"
    problem_scale: Optional[uuid.UUID] = None  # Scale where problem exists
    solution_scale: Optional[uuid.UUID] = None  # Scale where solution exists
    
    # Mismatch characteristics
    mismatch_severity: Optional[float] = None  # 0-1 scale
    mismatch_persistence: Optional[str] = None  # How long mismatch has existed
    affected_stakeholders: List[uuid.UUID] = field(default_factory=list)
    
    # Impacts
    coordination_costs: Optional[float] = None  # Increased coordination costs
    effectiveness_reduction: Optional[float] = None  # Reduced effectiveness
    legitimacy_impacts: Dict[uuid.UUID, float] = field(default_factory=dict)
    
    # Resolution approaches
    mismatch_resolution_strategies: List[str] = field(default_factory=list)
    institutional_innovations: List[str] = field(default_factory=list)
    bridging_mechanisms: List[str] = field(default_factory=list)
    
    # Progress tracking
    resolution_progress: Optional[float] = None  # 0-1 scale
    remaining_challenges: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)
    
    # SFM context
    matrix_integration_challenges: List[str] = field(default_factory=list)
    delivery_system_impacts: List[uuid.UUID] = field(default_factory=list)
    institutional_adjustment_needs: List[str] = field(default_factory=list)


@dataclass
class VerticalIntegration(Node):
    """Models vertical integration across hierarchical levels."""
    
    integration_domain: Optional[str] = None  # Domain of integration
    hierarchical_levels: List[uuid.UUID] = field(default_factory=list)
    integration_mechanisms: List[str] = field(default_factory=list)
    
    # Integration characteristics
    integration_strength: Optional[float] = None  # 0-1 scale
    centralization_level: Optional[float] = None  # 0-1 scale
    autonomy_preservation: Optional[float] = None  # 0-1 scale
    
    # Coordination mechanisms
    formal_coordination: List[str] = field(default_factory=list)
    informal_coordination: List[str] = field(default_factory=list)
    hierarchical_controls: List[str] = field(default_factory=list)
    
    # Performance outcomes
    coordination_efficiency: Optional[float] = None  # 0-1 scale
    decision_speed: Optional[float] = None  # 0-1 scale
    local_responsiveness: Optional[float] = None  # 0-1 scale
    system_coherence: Optional[float] = None  # 0-1 scale
    
    # Trade-offs
    efficiency_flexibility_tradeoff: Optional[float] = None  # -1 to 1 scale
    control_autonomy_tradeoff: Optional[float] = None  # -1 to 1 scale
    standardization_adaptation_tradeoff: Optional[float] = None  # -1 to 1 scale
    
    # SFM integration
    matrix_vertical_coherence: Optional[float] = None  # 0-1 scale
    delivery_system_alignment: List[uuid.UUID] = field(default_factory=list)
    institutional_hierarchy_effectiveness: Optional[float] = None


@dataclass
class HorizontalIntegration(Node):
    """Models horizontal integration across same-level entities."""
    
    integration_domain: Optional[str] = None
    participating_entities: List[uuid.UUID] = field(default_factory=list)
    integration_mechanisms: List[str] = field(default_factory=list)
    
    # Integration characteristics
    integration_intensity: Optional[float] = None  # 0-1 scale
    coordination_formality: Optional[float] = None  # 0-1 scale
    resource_sharing_level: Optional[float] = None  # 0-1 scale
    
    # Collaboration patterns
    information_sharing: Optional[float] = None  # 0-1 scale
    joint_planning: Optional[float] = None  # 0-1 scale
    resource_pooling: Optional[float] = None  # 0-1 scale
    collective_action: Optional[float] = None  # 0-1 scale
    
    # Network characteristics
    network_density: Optional[float] = None  # Connection density
    network_centralization: Optional[float] = None  # Network centralization
    trust_level: Optional[float] = None  # 0-1 scale
    
    # Performance outcomes
    collective_problem_solving: Optional[float] = None  # 0-1 scale
    innovation_capacity: Optional[float] = None  # 0-1 scale
    resource_efficiency: Optional[float] = None  # 0-1 scale
    
    # Challenges
    coordination_costs: Optional[float] = None  # Cost of coordination
    free_rider_problems: Optional[float] = None  # Extent of free-riding
    conflict_management: Optional[float] = None  # Conflict resolution capacity
    
    # SFM integration
    matrix_horizontal_coherence: Optional[float] = None  # 0-1 scale
    delivery_network_integration: List[uuid.UUID] = field(default_factory=list)
    institutional_complementarity: Optional[float] = None


@dataclass
class ScaleTransition(Node):
    """Models transitions and transformations across scales."""
    
    transition_type: Optional[str] = None  # e.g., "Scaling Up", "Scaling Down"
    source_scale: Optional[uuid.UUID] = None
    target_scale: Optional[uuid.UUID] = None
    
    # Transition characteristics
    transition_mechanism: List[str] = field(default_factory=list)
    transition_speed: Optional[str] = None  # Speed of transition
    transition_completeness: Optional[float] = None  # 0-1 scale
    
    # Transition process
    transition_stages: List[str] = field(default_factory=list)
    key_milestones: List[Dict[str, any]] = field(default_factory=list)
    critical_junctures: List[str] = field(default_factory=list)
    
    # Transition outcomes
    institutional_changes: List[str] = field(default_factory=list)
    performance_changes: Dict[str, float] = field(default_factory=dict)
    stakeholder_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    
    # Success factors
    enabling_conditions: List[str] = field(default_factory=list)
    barrier_conditions: List[str] = field(default_factory=list)
    leadership_factors: List[str] = field(default_factory=list)
    
    # Learning and adaptation
    lessons_learned: List[str] = field(default_factory=list)
    adaptive_adjustments: List[str] = field(default_factory=list)
    knowledge_transfer: List[str] = field(default_factory=list)
    
    # SFM integration
    matrix_transition_impacts: List[str] = field(default_factory=list)
    delivery_system_adjustments: List[uuid.UUID] = field(default_factory=list)
    institutional_innovation_outcomes: List[str] = field(default_factory=list)