"""
Core SFM nodes representing primary entities.

This module defines the core Social Fabric Matrix entities including actors,
institutions, resources, processes, and flows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.metadata_models import TemporalDynamics
from models.sfm_enums import (
    ResourceType,
    FlowNature,
    FlowType,
    EnumValidator,
)


@dataclass
class Actor(Node):
    """Individuals, firms, agencies, communities."""

    legal_form: Optional[str] = None  # e.g. "Corporation", "Household"
    sector: Optional[str] = None  # NAICS or custom taxonomy

    # Additional SFM-relevant fields
    power_resources: Dict[str, float] = field(default_factory=lambda: {})
    decision_making_capacity: Optional[float] = None
    institutional_affiliations: List[uuid.UUID] = field(default_factory=lambda: [])
    cognitive_frameworks: List[uuid.UUID] = field(default_factory=lambda: [])
    behavioral_patterns: List[uuid.UUID] = field(default_factory=lambda: [])
    
    def calculate_power_index(self) -> float:
        """Calculate overall power index based on power resources."""
        if not self.power_resources:
            return 0.0
        
        # Weight different power types
        weights = {
            'institutional_authority': 0.3,
            'economic_control': 0.25,
            'information_access': 0.2,
            'network_position': 0.15,
            'cultural_legitimacy': 0.1
        }
        
        weighted_sum = sum(
            self.power_resources.get(power_type, 0.0) * weight
            for power_type, weight in weights.items()
        )
        
        return min(weighted_sum, 1.0)  # Cap at 1.0
    
    def get_dominant_power_resource(self) -> Optional[str]:
        """Get the dominant power resource type for this actor."""
        if not self.power_resources:
            return None
        return max(self.power_resources.keys(), key=lambda x: self.power_resources[x])
    
    def assess_institutional_embeddedness(self) -> float:
        """Assess how embedded this actor is in institutional structures."""
        if not self.institutional_affiliations:
            return 0.0
        
        # More affiliations = higher embeddedness, but with diminishing returns
        affiliation_count = len(self.institutional_affiliations)
        return min(1.0, affiliation_count * 0.2)  # Cap at 1.0


@dataclass
class Institution(Node):
    """Formal rules, informal norms, cultural practices."""

    formality_level: Optional[str] = None  # "formal" | "informal" | "mixed"
    scope: Optional[str] = None
    enforcement_mechanism: Optional[str] = None

    # Additional SFM-relevant fields
    rule_types: List[str] = field(default_factory=lambda: [])
    enforcement_strength: Optional[float] = None
    legitimacy_score: Optional[float] = None
    change_frequency: Optional[float] = None  # How often this institution changes
    institutional_complementarity: List[uuid.UUID] = field(default_factory=lambda: [])
    ceremonial_instrumental_balance: Optional[float] = None  # -1 (ceremonial) to +1 (instrumental)
    
    def calculate_institutional_effectiveness(self) -> float:
        """Calculate overall institutional effectiveness."""
        components: List[float] = []
        
        if self.enforcement_strength is not None:
            components.append(self.enforcement_strength * 0.4)
        
        if self.legitimacy_score is not None:
            components.append(self.legitimacy_score * 0.4)
        
        # Stability (inverse of change frequency)
        if self.change_frequency is not None:
            stability = max(0.0, 1.0 - self.change_frequency)
            components.append(stability * 0.2)
        
        return sum(components) / len(components) if components else 0.0
    
    def get_institutional_type_classification(self) -> str:
        """Classify institution based on ceremonial-instrumental balance."""
        if self.ceremonial_instrumental_balance is None:
            return "unclassified"
        
        if self.ceremonial_instrumental_balance < -0.5:
            return "predominantly_ceremonial"
        elif self.ceremonial_instrumental_balance > 0.5:
            return "predominantly_instrumental"
        else:
            return "mixed_ceremonial_instrumental"
    
    def assess_complementarity_strength(self) -> float:
        """Assess strength of institutional complementarity."""
        if not self.institutional_complementarity:
            return 0.0
        
        # More complementary institutions = stronger institutional framework
        complementarity_count = len(self.institutional_complementarity)
        return min(1.0, complementarity_count * 0.15)  # Cap at 1.0


@dataclass
class Policy(Institution):
    """Specific policy intervention or regulatory framework."""

    authority: Optional[str] = None  # Implementing body
    enforcement: Optional[float] = 0.0  # Strength of enforcement (0-1)
    target_sectors: List[str] = field(default_factory=lambda: [])
    
    # SFM integration additions:
    target_matrix_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # Cells policy aims to improve
    effectiveness_evidence: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to evidence
    unintended_consequences: Dict[str, str] = field(default_factory=lambda: {})  # Unexpected matrix effects
    ceremonial_aspects: Optional[float] = None  # How much of policy is ceremonial (0-1)
    problem_solving_sequence: Optional[uuid.UUID] = None  # Link to ProblemSolvingSequence
    policy_instruments: List[uuid.UUID] = field(default_factory=lambda: [])  # Links to PolicyInstrument


@dataclass
class Resource(Node):
    """Stock or asset available for use or transformation."""

    rtype: ResourceType = ResourceType.NATURAL
    unit: Optional[str] = None  # e.g. "tonnes", "person-hours"


@dataclass
class Process(Node):
    """
    Transformation activity that converts inputs to outputs
    (production, consumption, disposal).
    """

    technology: Optional[str] = None  # e.g. "EAF-Steel-2024"
    responsible_actor_id: Optional[str] = None  # Actor that controls the process


@dataclass
class Flow(Node):  # pylint: disable=too-many-instance-attributes
    """Edge-like node representing an actual quantified transfer of resources or value."""

    nature: FlowNature = FlowNature.TRANSFER
    quantity: Optional[float] = None
    unit: Optional[str] = None
    time: Optional[TimeSlice] = None
    space: Optional[SpatialUnit] = None
    scenario: Optional[Scenario] = None

    # Additional SFM-specific fields
    flow_type: FlowType = FlowType.MATERIAL  # material, energy, information, financial, social
    source_process_id: Optional[uuid.UUID] = None
    target_process_id: Optional[uuid.UUID] = None
    transformation_coefficient: Optional[float] = None
    loss_factor: Optional[float] = None  # inefficiencies, waste

    # Hayden's value theory integration
    ceremonial_component: Optional[float] = None
    instrumental_component: Optional[float] = None
    temporal_dynamics: Optional[TemporalDynamics] = None  # Change over time
    
    # SFM Matrix integration additions:
    affecting_matrix_cells: List[uuid.UUID] = field(default_factory=lambda: [])  # Which cells this flow affects
    institutional_constraints: List[uuid.UUID] = field(default_factory=lambda: [])  # Institutions that constrain flow
    technology_dependencies: List[uuid.UUID] = field(default_factory=lambda: [])  # Required technologies
    ceremonial_barriers: List[str] = field(default_factory=lambda: [])  # Ceremonial obstacles to flow
    instrumental_enablers: List[str] = field(default_factory=lambda: [])  # Instrumental flow facilitators

    def __post_init__(self) -> None:
        """Validate flow nature and type combination after initialization."""
        # Validate flow nature and type combination
        EnumValidator.validate_flow_combination(self.nature, self.flow_type)


@dataclass
class ValueFlow(Flow):
    """Tracks value creation, capture, and distribution."""

    value_created: Optional[float] = None
    value_captured: Optional[float] = None
    beneficiary_actors: List[uuid.UUID] = field(default_factory=lambda: [])
    distributional_impact: Dict[str, float] = field(default_factory=lambda: {})


@dataclass
class GovernanceStructure(Institution):
    """Formal and informal governance arrangements."""

    decision_making_process: Optional[str] = None
    power_distribution: Dict[str, float] = field(default_factory=lambda: {})
    accountability_mechanisms: List[str] = field(default_factory=lambda: [])
    
    # Enhanced SFM integration:
    governance_effectiveness: Optional[float] = None  # How well governance works (0-1)
    participatory_mechanisms: List[str] = field(default_factory=lambda: [])  # How stakeholders participate
    transparency_level: Optional[float] = None  # Level of transparency (0-1)
    conflict_resolution: List[str] = field(default_factory=lambda: [])  # How conflicts are resolved
    adaptive_capacity: Optional[float] = None  # Ability to adapt to change (0-1)
