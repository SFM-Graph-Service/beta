"""
Institutional Learning and Evolution Framework for Social Fabric Matrix analysis.

This module models how institutions learn, adapt, and evolve over time within
the SFM framework. It captures learning mechanisms, institutional memory,
adaptive capacity, and evolutionary processes essential for understanding
institutional change in complex socio-economic systems.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
# Note: LearningType and AdaptationMechanism are defined locally in this file
# EvolutionaryStage, InstitutionalScope, and ChangeType are imported from models.sfm_enums
from models.sfm_enums import (
    # LearningType,  # Defined locally
    # AdaptationMechanism,  # Defined locally
    EvolutionaryStage,
    InstitutionalScope,
    ChangeType
)

class LearningType(Enum):
    """Types of institutional learning processes."""

    SINGLE_LOOP = auto()      # Error correction within existing frameworks
    DOUBLE_LOOP = auto()      # Questioning underlying assumptions
    TRIPLE_LOOP = auto()      # Learning about learning processes
    EXPERIENTIAL = auto()     # Learning through direct experience
    VICARIOUS = auto()        # Learning from others' experiences
    EXPERIMENTAL = auto()     # Learning through deliberate experimentation

class AdaptationMechanism(Enum):
    """Mechanisms through which institutions adapt."""

    INCREMENTAL_CHANGE = auto()  # Small, gradual adjustments
    PUNCTUATED_CHANGE = auto()   # Rapid, significant changes
    MORPHOGENESIS = auto()       # Structural transformation
    DRIFT = auto()              # Gradual, unintended change
    CONVERSION = auto()         # Redirection of existing institutions
    LAYERING = auto()           # Adding new elements to existing structures

class InstitutionalMemory(Enum):
    """Types of institutional memory storage."""

    FORMAL_RECORDS = auto()     # Written documentation, databases
    INFORMAL_KNOWLEDGE = auto() # Tacit knowledge, cultural memory
    PROCEDURAL_MEMORY = auto()  # Embedded in routines and practices
    STRUCTURAL_MEMORY = auto()  # Embedded in organizational structure
    RELATIONAL_MEMORY = auto()  # Stored in network relationships
    ARTIFACT_MEMORY = auto()    # Embedded in physical artifacts

class KnowledgeType(Enum):
    """Types of knowledge in institutional learning."""

    TACIT = auto()             # Implicit, experiential knowledge
    EXPLICIT = auto()          # Codified, articulated knowledge
    PROCEDURAL = auto()        # How-to knowledge
    DECLARATIVE = auto()       # What knowledge
    CONTEXTUAL = auto()        # Situational knowledge
    STRATEGIC = auto()         # Goal-oriented knowledge

@dataclass
class InstitutionalLearning(Node):
    """Models learning processes within institutions in SFM analysis."""

    learning_type: Optional[LearningType] = None
    learning_trigger: Optional[str] = None  # What triggered the learning
    learning_scope: Optional[InstitutionalScope] = None

    # Learning process
    problem_identification: Optional[str] = None
    information_gathering: List[str] = field(default_factory=list)
    analysis_methods: List[str] = field(default_factory=list)
    solution_development: List[str] = field(default_factory=list)

    # Knowledge involved
    prior_knowledge: List[str] = field(default_factory=list)
    new_knowledge_acquired: List[str] = field(default_factory=list)
    knowledge_sources: List[uuid.UUID] = field(default_factory=list)
    knowledge_type: Optional[KnowledgeType] = None

    # Learning outcomes
    behavioral_changes: List[str] = field(default_factory=list)
    structural_changes: List[str] = field(default_factory=list)
    performance_improvements: Dict[str, float] = field(default_factory=dict)
    new_capabilities: List[str] = field(default_factory=list)

    # Learning effectiveness
    learning_speed: Optional[float] = None  # Rate of learning
    learning_depth: Optional[float] = None  # Depth of understanding
    knowledge_retention: Optional[float] = None  # Memory retention rate
    knowledge_transfer: Optional[float] = None  # Transfer to other contexts

    # Barriers and facilitators
    learning_barriers: List[str] = field(default_factory=list)
    learning_facilitators: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)

    # SFM integration
    matrix_implications: List[str] = field(default_factory=list)
    delivery_improvements: List[uuid.UUID] = field(default_factory=list)
    institutional_relationships_affected: List[uuid.UUID] = field(default_factory=list)

@dataclass
class AdaptiveCapacity(Node):
    """Models the adaptive capacity of institutions within SFM framework."""

    capacity_level: Optional[float] = None  # 0-1 scale
    adaptation_mechanisms: List[AdaptationMechanism] = field(default_factory=list)

    # Capacity dimensions
    sensing_capacity: Optional[float] = None  # Ability to detect changes
    learning_capacity: Optional[float] = None  # Ability to learn from experience
    innovation_capacity: Optional[float] = None  # Ability to generate new solutions
    implementation_capacity: Optional[float] = None  # Ability to execute changes

    # Organizational factors
    leadership_quality: Optional[float] = None  # Quality of adaptive leadership
    organizational_culture: Optional[str] = None  # Culture supporting adaptation
    resource_availability: Dict[str, float] = field(default_factory=dict)
    network_connections: List[uuid.UUID] = field(default_factory=list)

    # Environmental factors
    environmental_complexity: Optional[float] = None  # Complexity of environment
    change_frequency: Optional[float] = None  # Frequency of environmental changes
    uncertainty_level: Optional[float] = None  # Level of uncertainty faced
    competitive_pressure: Optional[float] = None  # Pressure for adaptation

    # Adaptation history
    past_adaptations: List[uuid.UUID] = field(default_factory=list)
    adaptation_success_rate: Optional[float] = None  # Historical success rate
    learning_from_failures: List[str] = field(default_factory=list)

    # Constraints and enablers
    path_dependencies: List[str] = field(default_factory=list)
    resource_constraints: List[str] = field(default_factory=list)
    regulatory_constraints: List[str] = field(default_factory=list)
    cultural_constraints: List[str] = field(default_factory=list)

    # SFM context
    ceremonial_resistance: Optional[float] = None  # Resistance to change
    instrumental_motivation: Optional[float] = None  # Motivation for problem-solving
    matrix_adaptation_potential: Optional[float] = None  # Potential for matrix evolution

@dataclass
class OrganizationalMemory(Node):
    """Models institutional memory systems within SFM analysis."""

    memory_type: Optional[InstitutionalMemory] = None
    storage_mechanisms: List[str] = field(default_factory=list)

    # Memory content
    historical_experiences: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    failure_analyses: List[str] = field(default_factory=list)

    # Memory characteristics
    accessibility: Optional[float] = None  # Ease of access to memory
    accuracy: Optional[float] = None  # Accuracy of stored information
    completeness: Optional[float] = None  # Completeness of memory
    currency: Optional[float] = None  # How up-to-date the memory is

    # Memory maintenance
    updating_mechanisms: List[str] = field(default_factory=list)
    validation_processes: List[str] = field(default_factory=list)
    knowledge_custodians: List[uuid.UUID] = field(default_factory=list)

    # Memory utilization
    retrieval_frequency: Optional[float] = None  # How often memory is accessed
    application_rate: Optional[float] = None  # How often memory influences decisions
    knowledge_sharing: Optional[float] = None  # Extent of knowledge sharing

    # Memory evolution
    memory_decay: Optional[float] = None  # Rate of memory loss
    selective_retention: List[str] = field(default_factory=list)
    memory_reconstruction: List[str] = field(default_factory=list)

    # SFM integration
    matrix_knowledge_base: List[str] = field(default_factory=list)
    delivery_relationship_history: List[uuid.UUID] = field(default_factory=list)
    institutional_evolution_record: List[uuid.UUID] = field(default_factory=list)

@dataclass
class KnowledgeManagement(Node):
    """Models knowledge management systems within institutions."""

    knowledge_strategy: Optional[str] = None
    management_systems: List[str] = field(default_factory=list)

    # Knowledge processes
    knowledge_creation: List[str] = field(default_factory=list)
    knowledge_capture: List[str] = field(default_factory=list)
    knowledge_storage: List[str] = field(default_factory=list)
    knowledge_sharing: List[str] = field(default_factory=list)
    knowledge_application: List[str] = field(default_factory=list)

    # Knowledge assets
    explicit_knowledge: Dict[str, str] = field(default_factory=dict)
    tacit_knowledge: List[str] = field(default_factory=list)
    knowledge_repositories: List[str] = field(default_factory=list)
    expert_networks: List[uuid.UUID] = field(default_factory=list)

    # Knowledge quality
    knowledge_accuracy: Optional[float] = None  # 0-1 scale
    knowledge_relevance: Optional[float] = None  # 0-1 scale
    knowledge_timeliness: Optional[float] = None  # 0-1 scale
    knowledge_completeness: Optional[float] = None  # 0-1 scale

    # Knowledge flows
    internal_knowledge_flows: Dict[str, float] = field(default_factory=dict)
    external_knowledge_flows: Dict[str, float] = field(default_factory=dict)
    knowledge_brokers: List[uuid.UUID] = field(default_factory=list)

    # Performance metrics
    knowledge_utilization_rate: Optional[float] = None
    innovation_rate: Optional[float] = None
    learning_effectiveness: Optional[float] = None
    knowledge_value_creation: Optional[float] = None

    # SFM context
    matrix_knowledge_integration: Optional[float] = None
    delivery_system_intelligence: Optional[float] = None
    institutional_wisdom: Optional[float] = None

@dataclass
class InstitutionalEvolution(Node):
    """Models long-term evolutionary processes of institutions."""

    evolution_stage: Optional[EvolutionaryStage] = None
    evolution_drivers: List[str] = field(default_factory=list)
    evolution_trajectory: Optional[str] = None

    # Evolutionary characteristics
    variation_mechanisms: List[str] = field(default_factory=list)
    selection_pressures: List[str] = field(default_factory=list)
    retention_mechanisms: List[str] = field(default_factory=list)

    # Historical development
    founding_conditions: Dict[str, str] = field(default_factory=dict)
    key_milestones: List[Dict[str, any]] = field(default_factory=list)
    critical_junctures: List[Dict[str, any]] = field(default_factory=list)
    path_dependencies: List[str] = field(default_factory=list)

    # Evolutionary patterns
    growth_phases: List[str] = field(default_factory=list)
    decline_phases: List[str] = field(default_factory=list)
    transformation_phases: List[str] = field(default_factory=list)
    stability_phases: List[str] = field(default_factory=list)

    # Co-evolution relationships
    co_evolving_institutions: List[uuid.UUID] = field(default_factory=list)
    environmental_co_evolution: List[str] = field(default_factory=list)
    technological_co_evolution: List[str] = field(default_factory=list)

    # Future trajectory
    evolutionary_potential: Optional[float] = None  # 0-1 scale
    adaptation_scenarios: List[str] = field(default_factory=list)
    extinction_risks: List[str] = field(default_factory=list)

    # SFM integration
    matrix_evolution_contribution: Optional[float] = None
    system_level_implications: List[str] = field(default_factory=list)
    delivery_system_co_evolution: List[uuid.UUID] = field(default_factory=list)

@dataclass
class LearningNetwork(Node):
    """Models networks through which institutional learning occurs."""

    network_type: Optional[str] = None  # e.g., "Professional", "Policy", "Research"
    network_members: List[uuid.UUID] = field(default_factory=list)

    # Network structure
    network_density: Optional[float] = None  # Connection density
    centralization: Optional[float] = None  # Network centralization
    clustering: Optional[float] = None  # Local clustering coefficient

    # Learning processes
    knowledge_sharing_patterns: Dict[str, float] = field(default_factory=dict)
    learning_partnerships: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)
    best_practice_diffusion: List[str] = field(default_factory=list)
    collaborative_learning: List[str] = field(default_factory=list)

    # Network dynamics
    membership_stability: Optional[float] = None  # Member retention rate
    knowledge_flow_intensity: Optional[float] = None  # Intensity of knowledge exchange
    network_learning_capacity: Optional[float] = None  # Collective learning capacity

    # Learning outcomes
    network_innovations: List[str] = field(default_factory=list)
    collective_problem_solving: List[str] = field(default_factory=list)
    knowledge_creation: List[str] = field(default_factory=list)

    # SFM context
    matrix_learning_contribution: Optional[float] = None
    institutional_development_impact: List[uuid.UUID] = field(default_factory=list)
    system_intelligence_enhancement: Optional[float] = None

@dataclass
class InstitutionalInnovation(Node):
    """Models innovation processes within institutional contexts."""

    innovation_type: Optional[str] = None  # e.g., "Process", "Product", "Organizational"
    innovation_scope: Optional[InstitutionalScope] = None

    # Innovation process
    problem_identification: Optional[str] = None
    idea_generation: List[str] = field(default_factory=list)
    concept_development: List[str] = field(default_factory=list)
    experimentation: List[str] = field(default_factory=list)
    implementation: List[str] = field(default_factory=list)

    # Innovation characteristics
    novelty_level: Optional[float] = None  # 0-1 scale
    complexity_level: Optional[float] = None  # 0-1 scale
    risk_level: Optional[float] = None  # 0-1 scale
    resource_requirements: Dict[str, float] = field(default_factory=dict)

    # Innovation actors
    innovation_champions: List[uuid.UUID] = field(default_factory=list)
    innovation_teams: List[uuid.UUID] = field(default_factory=list)
    external_collaborators: List[uuid.UUID] = field(default_factory=list)

    # Innovation outcomes
    performance_improvements: Dict[str, float] = field(default_factory=dict)
    new_capabilities: List[str] = field(default_factory=list)
    competitive_advantages: List[str] = field(default_factory=list)

    # Diffusion and adoption
    adoption_rate: Optional[float] = None  # Rate of innovation adoption
    diffusion_mechanisms: List[str] = field(default_factory=list)
    adoption_barriers: List[str] = field(default_factory=list)

    # SFM integration
    matrix_innovation_impact: List[str] = field(default_factory=list)
    delivery_system_improvements: List[uuid.UUID] = field(default_factory=list)
    institutional_relationship_changes: List[uuid.UUID] = field(default_factory=list)
