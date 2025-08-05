"""
Cultural Context Integration Framework for Social Fabric Matrix analysis.

This module models cultural value systems, meaning-making processes, cultural
transmission mechanisms, and cultural evolution within institutional systems.
Essential for understanding how culture shapes institutional outcomes in SFM
analysis following Hayden's institutional economics approach.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
# Note: CulturalDimension, ValueOrientation, CulturalTransmissionType, and MeaningSystemType
# are defined locally in this file rather than imported from models.sfm_enums


class CulturalDimension(Enum):
    """Dimensions of cultural variation in institutional analysis."""
    
    INDIVIDUALISM_COLLECTIVISM = auto()  # Individual vs. collective orientation
    POWER_DISTANCE = auto()              # Acceptance of hierarchy/inequality
    UNCERTAINTY_AVOIDANCE = auto()       # Tolerance for ambiguity/uncertainty
    MASCULINITY_FEMININITY = auto()      # Achievement vs. relationship orientation
    LONG_TERM_ORIENTATION = auto()       # Long-term vs. short-term focus
    INDULGENCE_RESTRAINT = auto()        # Gratification vs. restraint of desires


class ValueOrientation(Enum):
    """Basic value orientations affecting institutional choices."""
    
    TRADITIONAL = auto()          # Traditional values and practices
    SECULAR_RATIONAL = auto()     # Secular, rational approaches
    SURVIVAL = auto()             # Survival-focused values
    SELF_EXPRESSION = auto()      # Self-expression and quality of life
    MATERIALIST = auto()          # Material security and economic growth
    POST_MATERIALIST = auto()     # Quality of life and self-actualization


class CulturalTransmissionType(Enum):
    """Types of cultural transmission mechanisms."""
    
    VERTICAL = auto()             # Parent to child transmission
    HORIZONTAL = auto()           # Peer-to-peer transmission
    OBLIQUE = auto()              # Elder to younger generation transmission
    INSTITUTIONAL = auto()        # Through formal institutions
    MEDIA = auto()                # Through media and communication
    EXPERIENTIAL = auto()         # Through direct experience


class MeaningSystemType(Enum):
    """Types of meaning systems in cultural contexts."""
    
    RELIGIOUS = auto()            # Religious belief systems
    IDEOLOGICAL = auto()          # Political/ideological systems
    PROFESSIONAL = auto()         # Professional/occupational cultures
    ETHNIC = auto()               # Ethnic/tribal meaning systems
    GENERATIONAL = auto()         # Generational subcultures
    LIFESTYLE = auto()            # Lifestyle-based meaning systems


@dataclass
class CulturalValueSystem(Node):
    """Models cultural value systems within institutional contexts."""
    
    value_system_name: Optional[str] = None
    cultural_group: Optional[str] = None  # Group associated with value system
    geographic_scope: Optional[str] = None  # Geographic extent
    
    # Core values
    primary_values: List[str] = field(default_factory=list)
    secondary_values: List[str] = field(default_factory=list)
    value_hierarchies: Dict[str, int] = field(default_factory=dict)  # Value -> priority rank
    
    # Cultural dimensions
    cultural_dimensions: Dict[CulturalDimension, float] = field(default_factory=dict)
    value_orientations: List[ValueOrientation] = field(default_factory=list)
    dominant_orientation: Optional[ValueOrientation] = None
    
    # Value characteristics
    value_coherence: Optional[float] = None  # Internal consistency (0-1)
    value_stability: Optional[float] = None  # Stability over time (0-1)
    value_adaptability: Optional[float] = None  # Capacity for change (0-1)
    
    # Cultural expressions
    symbolic_expressions: List[str] = field(default_factory=list)
    ritual_practices: List[str] = field(default_factory=list)
    narrative_traditions: List[str] = field(default_factory=list)
    cultural_artifacts: List[str] = field(default_factory=list)
    
    # Value conflicts and tensions
    internal_tensions: List[str] = field(default_factory=list)
    external_conflicts: List[str] = field(default_factory=list)
    value_trade_offs: Dict[str, str] = field(default_factory=dict)
    
    # Cultural transmission
    transmission_mechanisms: List[CulturalTransmissionType] = field(default_factory=list)
    cultural_carriers: List[uuid.UUID] = field(default_factory=list)  # Institutions/actors
    transmission_effectiveness: Optional[float] = None  # 0-1 scale
    
    # SFM integration
    institutional_value_alignment: Dict[uuid.UUID, float] = field(default_factory=dict)
    matrix_cultural_influence: List[uuid.UUID] = field(default_factory=list)
    delivery_system_cultural_fit: Dict[uuid.UUID, float] = field(default_factory=dict)


@dataclass
class MeaningSystem(Node):
    """Models meaning-making systems within cultural contexts."""
    
    meaning_system_type: Optional[MeaningSystemType] = None
    system_name: Optional[str] = None
    adherent_population: Optional[int] = None
    
    # Core meaning structures
    fundamental_beliefs: List[str] = field(default_factory=list)
    worldview_assumptions: List[str] = field(default_factory=list)
    interpretive_frameworks: List[str] = field(default_factory=list)
    
    # Meaning construction
    reality_construction_processes: List[str] = field(default_factory=list)
    legitimation_mechanisms: List[str] = field(default_factory=list)
    sense_making_patterns: List[str] = field(default_factory=list)
    
    # Symbol and narrative systems
    symbolic_repertoire: List[str] = field(default_factory=list)
    master_narratives: List[str] = field(default_factory=list)
    cultural_scripts: List[str] = field(default_factory=list)
    
    # Meaning system characteristics
    coherence_level: Optional[float] = None  # Internal logical consistency (0-1)
    comprehensiveness: Optional[float] = None  # Scope of meaning provision (0-1)
    flexibility: Optional[float] = None  # Adaptability to new situations (0-1)
    emotional_resonance: Optional[float] = None  # Emotional power (0-1)
    
    # Social functions
    identity_construction: Optional[float] = None  # Identity formation capacity
    social_integration: Optional[float] = None  # Community building capacity
    moral_guidance: Optional[float] = None  # Ethical guidance provision
    existential_comfort: Optional[float] = None  # Existential anxiety reduction
    
    # Competing meaning systems
    rival_systems: List[uuid.UUID] = field(default_factory=list)
    syncretism_potential: Optional[float] = None  # Potential for blending
    conflict_intensity: Dict[uuid.UUID, float] = field(default_factory=dict)
    
    # Evolution and change
    change_mechanisms: List[str] = field(default_factory=list)
    adaptation_strategies: List[str] = field(default_factory=list)
    innovation_capacity: Optional[float] = None  # 0-1 scale
    
    # SFM integration
    institutional_meaning_provision: Dict[uuid.UUID, float] = field(default_factory=dict)
    matrix_meaning_integration: List[uuid.UUID] = field(default_factory=list)
    delivery_meaning_legitimation: Dict[uuid.UUID, float] = field(default_factory=dict)


@dataclass
class CulturalNorm(Node):
    """Models specific cultural norms within institutional systems."""
    
    norm_description: Optional[str] = None
    norm_domain: Optional[str] = None  # Domain where norm applies
    norm_strength: Optional[float] = None  # Strength of norm adherence (0-1)
    
    # Norm characteristics
    explicitness: Optional[float] = None  # How explicit the norm is (0-1)
    universality: Optional[float] = None  # How universally applied (0-1)
    enforcement_strength: Optional[float] = None  # Enforcement intensity (0-1)
    
    # Norm content
    prescribed_behaviors: List[str] = field(default_factory=list)
    proscribed_behaviors: List[str] = field(default_factory=list)
    sanctions_positive: List[str] = field(default_factory=list)
    sanctions_negative: List[str] = field(default_factory=list)
    
    # Norm compliance
    compliance_rate: Optional[float] = None  # General compliance level (0-1)
    compliance_variation: Dict[str, float] = field(default_factory=dict)  # By group
    deviation_patterns: List[str] = field(default_factory=list)
    
    # Norm enforcement
    enforcement_actors: List[uuid.UUID] = field(default_factory=list)
    enforcement_mechanisms: List[str] = field(default_factory=list)
    enforcement_consistency: Optional[float] = None  # Consistency of enforcement
    
    # Norm evolution
    norm_emergence_story: Optional[str] = None
    change_pressures: List[str] = field(default_factory=list)
    adaptation_mechanisms: List[str] = field(default_factory=list)
    
    # SFM integration
    institutional_norm_support: List[uuid.UUID] = field(default_factory=list)
    matrix_norm_influence: List[uuid.UUID] = field(default_factory=list)
    delivery_norm_constraints: Dict[uuid.UUID, str] = field(default_factory=dict)


@dataclass
class CulturalCapital(Node):
    """Models cultural capital resources within institutional contexts."""
    
    capital_type: Optional[str] = None  # Type of cultural capital
    capital_holder: Optional[uuid.UUID] = None  # Actor holding the capital
    
    # Capital forms
    embodied_capital: List[str] = field(default_factory=list)  # Skills, knowledge, dispositions
    objectified_capital: List[str] = field(default_factory=list)  # Cultural goods, artifacts
    institutionalized_capital: List[str] = field(default_factory=list)  # Credentials, qualifications
    
    # Capital characteristics
    capital_volume: Optional[float] = None  # Amount of capital (0-1)
    capital_legitimacy: Optional[float] = None  # Recognition/legitimacy (0-1)
    capital_convertibility: Optional[float] = None  # Convertibility to other forms (0-1)
    
    # Capital acquisition
    acquisition_mechanisms: List[str] = field(default_factory=list)
    acquisition_costs: Dict[str, float] = field(default_factory=dict)
    acquisition_barriers: List[str] = field(default_factory=list)
    
    # Capital deployment
    deployment_contexts: List[str] = field(default_factory=list)
    deployment_strategies: List[str] = field(default_factory=list)
    deployment_effectiveness: Optional[float] = None  # 0-1 scale
    
    # Capital returns
    economic_returns: Optional[float] = None  # Economic benefits
    social_returns: Optional[float] = None  # Social status benefits
    symbolic_returns: Optional[float] = None  # Symbolic/prestige benefits
    
    # Capital dynamics
    capital_accumulation_rate: Optional[float] = None  # Rate of accumulation
    capital_depreciation_rate: Optional[float] = None  # Rate of depreciation
    capital_transmission_potential: Optional[float] = None  # Inheritance potential
    
    # SFM integration
    institutional_capital_value: Dict[uuid.UUID, float] = field(default_factory=dict)
    matrix_capital_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_capital_requirements: Dict[uuid.UUID, float] = field(default_factory=dict)


@dataclass
class CulturalInnovation(Node):
    """Models cultural innovation processes within institutional systems."""
    
    innovation_description: Optional[str] = None
    innovation_type: Optional[str] = None  # e.g., "Symbolic", "Normative", "Practical"
    innovation_scope: Optional[str] = None  # Scope of cultural change
    
    # Innovation characteristics
    novelty_level: Optional[float] = None  # Degree of novelty (0-1)
    disruptiveness: Optional[float] = None  # How disruptive to existing culture (0-1)
    complexity: Optional[float] = None  # Complexity of innovation (0-1)
    
    # Innovation process
    innovation_triggers: List[str] = field(default_factory=list)
    innovation_actors: List[uuid.UUID] = field(default_factory=list)
    development_process: List[str] = field(default_factory=list)
    
    # Innovation content
    new_symbols: List[str] = field(default_factory=list)
    new_practices: List[str] = field(default_factory=list)
    new_meanings: List[str] = field(default_factory=list)
    new_values: List[str] = field(default_factory=list)
    
    # Adoption and diffusion
    early_adopters: List[uuid.UUID] = field(default_factory=list)
    diffusion_mechanisms: List[str] = field(default_factory=list)
    adoption_rate: Optional[float] = None  # Rate of adoption
    diffusion_barriers: List[str] = field(default_factory=list)
    
    # Innovation outcomes
    cultural_impact: Optional[float] = None  # Overall cultural impact (0-1)
    institutional_impact: Optional[float] = None  # Impact on institutions (0-1)
    behavioral_changes: List[str] = field(default_factory=list)
    
    # Innovation resistance
    resistance_sources: List[str] = field(default_factory=list)
    adaptation_mechanisms: List[str] = field(default_factory=list)
    compromise_outcomes: List[str] = field(default_factory=list)
    
    # SFM integration
    matrix_innovation_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_innovation_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    institutional_innovation_alignment: Dict[uuid.UUID, float] = field(default_factory=dict)


@dataclass
class CulturalConflict(Node):
    """Models conflicts between different cultural systems."""
    
    conflict_description: Optional[str] = None
    conflict_domain: Optional[str] = None  # Domain of conflict
    conflict_intensity: Optional[float] = None  # Intensity level (0-1)
    
    # Conflicting parties
    primary_cultures: List[uuid.UUID] = field(default_factory=list)
    cultural_representatives: List[uuid.UUID] = field(default_factory=list)
    affected_populations: List[str] = field(default_factory=list)
    
    # Conflict dimensions
    value_conflicts: List[str] = field(default_factory=list)
    normative_conflicts: List[str] = field(default_factory=list)
    symbolic_conflicts: List[str] = field(default_factory=list)
    practical_conflicts: List[str] = field(default_factory=list)
    
    # Conflict dynamics
    escalation_factors: List[str] = field(default_factory=list)
    de_escalation_factors: List[str] = field(default_factory=list)
    conflict_cycles: List[str] = field(default_factory=list)
    
    # Resolution approaches
    dialogue_mechanisms: List[str] = field(default_factory=list)
    mediation_efforts: List[str] = field(default_factory=list)
    accommodation_strategies: List[str] = field(default_factory=list)
    synthesis_attempts: List[str] = field(default_factory=list)
    
    # Conflict outcomes
    resolution_status: Optional[str] = None  # "Ongoing", "Resolved", "Contained"
    cultural_changes: List[str] = field(default_factory=list)
    institutional_adaptations: List[str] = field(default_factory=list)
    
    # Learning and evolution
    conflict_lessons: List[str] = field(default_factory=list)
    institutional_learning: List[str] = field(default_factory=list)
    cultural_evolution: List[str] = field(default_factory=list)
    
    # SFM integration
    matrix_conflict_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_conflict_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    institutional_conflict_responses: Dict[uuid.UUID, str] = field(default_factory=dict)


@dataclass
class CulturalBridge(Node):
    """Models mechanisms that bridge different cultural systems."""
    
    bridge_type: Optional[str] = None  # Type of bridging mechanism
    bridged_cultures: List[uuid.UUID] = field(default_factory=list)
    bridge_actors: List[uuid.UUID] = field(default_factory=list)
    
    # Bridging mechanisms
    translation_mechanisms: List[str] = field(default_factory=list)
    mediation_processes: List[str] = field(default_factory=list)
    hybrid_formations: List[str] = field(default_factory=list)
    
    # Bridge characteristics
    bridging_effectiveness: Optional[float] = None  # Effectiveness (0-1)
    cultural_authenticity: Optional[float] = None  # Authenticity to source cultures
    acceptance_level: Dict[uuid.UUID, float] = field(default_factory=dict)  # By culture
    
    # Bridge functions
    communication_facilitation: Optional[float] = None  # Communication improvement
    conflict_reduction: Optional[float] = None  # Conflict mitigation
    collaboration_enhancement: Optional[float] = None  # Collaboration improvement
    
    # Bridge sustainability
    sustainability_factors: List[str] = field(default_factory=list)
    maintenance_requirements: List[str] = field(default_factory=list)
    evolution_potential: Optional[float] = None  # Potential for development
    
    # SFM integration
    matrix_bridging_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_bridging_facilitation: Dict[uuid.UUID, float] = field(default_factory=dict)
    institutional_bridging_support: Dict[uuid.UUID, float] = field(default_factory=dict)