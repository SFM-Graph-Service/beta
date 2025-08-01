"""
Ceremonial-Instrumental Dichotomy Analysis Framework for Social Fabric Matrix.

This module implements the core Veblen-Ayres ceremonial-instrumental dichotomy
that is fundamental to Hayden's SFM framework. It provides systematic analysis
tools for evaluating the ceremonial versus instrumental characteristics of
institutions, behaviors, technologies, and policies within SFM analysis.
"""

# type: ignore
# mypy: disable-error-code=misc,type-arg,attr-defined,assignment,operator,call-overload,return-value,arg-type,union-attr,var-annotated,name-defined,no-any-return,override
# pylint: disable=too-many-instance-attributes,too-many-public-methods,unnecessary-isinstance,arguments-differ
# pyright: reportGeneralTypeIssues=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
# Local enum definitions - no imports needed from sfm_enums for these


class CeremonialType(Enum):
    """Types of ceremonial behavior and institutions."""
    
    STATUS_MAINTENANCE = auto()      # Preserving existing status hierarchies
    POWER_PRESERVATION = auto()      # Maintaining existing power structures
    TRADITION_ADHERENCE = auto()     # Following tradition without question
    RITUAL_EMPHASIS = auto()         # Emphasis on ritual over substance
    PRESTIGE_SEEKING = auto()        # Seeking prestige and recognition
    RESISTANCE_TO_CHANGE = auto()    # Opposition to technological/social change
    WASTE_DISPLAY = auto()           # Conspicuous consumption/waste
    EXCLUSION_PRACTICES = auto()     # Excluding others from participation


class InstrumentalType(Enum):
    """Types of instrumental behavior and institutions."""
    
    PROBLEM_SOLVING = auto()         # Focus on solving real problems
    EFFICIENCY_SEEKING = auto()      # Seeking technological efficiency
    KNOWLEDGE_APPLICATION = auto()   # Applying scientific knowledge
    ADAPTATION_PROMOTION = auto()    # Promoting adaptive change
    INCLUSIVITY_ENHANCEMENT = auto() # Including more participants
    WASTE_REDUCTION = auto()         # Reducing waste and inefficiency
    INNOVATION_FOSTERING = auto()    # Fostering technological innovation
    COMMUNITY_ENHANCEMENT = auto()   # Enhancing community well-being


class DichotomyIndicator(Enum):
    """Indicators for measuring ceremonial vs instrumental characteristics."""
    
    TECHNOLOGY_ADOPTION = auto()     # Adoption of new technologies
    KNOWLEDGE_UTILIZATION = auto()   # Use of scientific knowledge
    CHANGE_RESISTANCE = auto()       # Resistance to change
    WASTE_GENERATION = auto()        # Generation of waste
    POWER_CONCENTRATION = auto()     # Concentration of power
    INCLUSION_LEVEL = auto()         # Level of inclusive participation
    EFFICIENCY_MEASURES = auto()     # Efficiency in resource use
    INNOVATION_RATE = auto()         # Rate of innovation


class TransformationStage(Enum):
    """Stages of ceremonial-instrumental transformation."""
    
    CEREMONIAL_DOMINANCE = auto()    # Ceremonial patterns dominate
    TENSION_EMERGENCE = auto()       # Tensions between C and I emerge
    CONFLICT_INTENSIFICATION = auto() # Conflicts intensify
    TRANSFORMATION_INITIATION = auto() # Transformation begins
    INSTRUMENTAL_ASCENDANCE = auto() # Instrumental patterns gain strength
    NEW_EQUILIBRIUM = auto()         # New C-I balance established
    CONTINUOUS_EVOLUTION = auto()    # Ongoing evolutionary change


@dataclass
class CeremonialInstrumentalAnalysis(Node):
    """Core framework for analyzing the ceremonial-instrumental dichotomy."""
    
    analyzed_entity_id: Optional[uuid.UUID] = None  # Institution, policy, or actor being analyzed
    analysis_date: Optional[datetime] = None
    analyst_id: Optional[uuid.UUID] = None
    
    # Core dichotomy scores
    ceremonial_score: Optional[float] = None  # 0-1 scale
    instrumental_score: Optional[float] = None  # 0-1 scale
    dichotomy_balance: Optional[float] = None  # -1 (ceremonial) to +1 (instrumental)
    dichotomy_intensity: Optional[float] = None  # Strength of dichotomy (0-1)
    
    # Ceremonial characteristics
    ceremonial_indicators: Dict[DichotomyIndicator, float] = field(default_factory=dict)  # type: ignore[misc]
    ceremonial_behaviors: List[CeremonialType] = field(default_factory=list)  # type: ignore[misc]
    ceremonial_manifestations: List[str] = field(default_factory=list)  # type: ignore[misc]
    ceremonial_functions: List[str] = field(default_factory=list)  # What ceremonial aspects do  # type: ignore[misc]
    
    # Instrumental characteristics
    instrumental_indicators: Dict[DichotomyIndicator, float] = field(default_factory=dict)  # type: ignore[misc]
    instrumental_behaviors: List[InstrumentalType] = field(default_factory=list)  # type: ignore[misc]
    instrumental_manifestations: List[str] = field(default_factory=list)  # type: ignore[misc]
    instrumental_functions: List[str] = field(default_factory=list)  # What instrumental aspects do  # type: ignore[misc]
    
    # Dichotomy dynamics
    tension_areas: List[str] = field(default_factory=list)  # Areas of C-I tension  # type: ignore[misc]
    conflict_points: List[str] = field(default_factory=list)  # Points of active conflict  # type: ignore[misc]
    transformation_pressures: List[str] = field(default_factory=list)  # Pressures for change  # type: ignore[misc]
    resistance_mechanisms: List[str] = field(default_factory=list)  # Mechanisms resisting change  # type: ignore[misc]
    
    # Change analysis
    transformation_stage: Optional[TransformationStage] = None
    change_trajectory: Optional[str] = None  # Direction of change
    change_velocity: Optional[float] = None  # Speed of change (0-1)
    transformation_potential: Optional[float] = None  # Potential for transformation (0-1)
    
    # Historical analysis
    ceremonial_evolution: List[Dict[str, Any]] = field(default_factory=list)  # type: ignore[misc]
    instrumental_development: List[Dict[str, Any]] = field(default_factory=list)  # type: ignore[misc]
    dichotomy_history: List[Dict[str, Any]] = field(default_factory=list)  # type: ignore[misc]
    
    # Contextual factors
    cultural_context: List[str] = field(default_factory=list)  # type: ignore[misc]
    technological_context: List[str] = field(default_factory=list)  # type: ignore[misc]
    economic_context: List[str] = field(default_factory=list)  # type: ignore[misc]
    political_context: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # SFM integration
    matrix_ci_effects: List[uuid.UUID] = field(default_factory=list)  # Matrix cells affected  # type: ignore[misc]
    delivery_ci_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)  # Delivery impacts  # type: ignore[misc]
    institutional_ci_relationships: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    
    def calculate_dichotomy_balance(self) -> Optional[float]:
        """Calculate the overall ceremonial-instrumental balance."""
        if self.ceremonial_score is None or self.instrumental_score is None:
            return None
        
        # Balance ranges from -1 (pure ceremonial) to +1 (pure instrumental)
        total_score = self.ceremonial_score + self.instrumental_score
        if total_score == 0:
            return 0.0
        
        balance = (self.instrumental_score - self.ceremonial_score) / total_score
        self.dichotomy_balance = balance
        return balance
    
    def assess_transformation_potential(self) -> Optional[float]:
        """Assess potential for ceremonial-instrumental transformation."""
        factors = []
        
        # Pressure factors
        if self.transformation_pressures:
            pressure_score = min(len(self.transformation_pressures) / 5.0, 1.0)
            factors.append(pressure_score * 0.3)
        
        # Resistance factors (inverse)
        if self.resistance_mechanisms:
            resistance_score = min(len(self.resistance_mechanisms) / 5.0, 1.0)
            factors.append((1.0 - resistance_score) * 0.2)
        
        # Current instrumental level
        if self.instrumental_score is not None:
            factors.append(self.instrumental_score * 0.3)
        
        # Change velocity
        if self.change_velocity is not None:
            factors.append(self.change_velocity * 0.2)
        
        if factors:
            potential = sum(factors) / len(factors) * 4  # Weight factors appropriately  # type: ignore[arg-type]
            self.transformation_potential = min(potential, 1.0)
            return self.transformation_potential
        
        return None
    
    def identify_transformation_barriers(self) -> List[str]:
        """Identify key barriers to instrumental transformation."""
        barriers = []
        
        # High ceremonial score indicates barriers
        if self.ceremonial_score and self.ceremonial_score > 0.7:
            barriers.append("High ceremonial entrenchment")
        
        # Specific ceremonial types that create barriers
        if CeremonialType.RESISTANCE_TO_CHANGE in self.ceremonial_behaviors:
            barriers.append("Active resistance to change")
        
        if CeremonialType.POWER_PRESERVATION in self.ceremonial_behaviors:
            barriers.append("Power structure preservation")
        
        if CeremonialType.STATUS_MAINTENANCE in self.ceremonial_behaviors:
            barriers.append("Status hierarchy maintenance")
        
        # Low transformation potential
        if self.transformation_potential and self.transformation_potential < 0.3:
            barriers.append("Low transformation potential")
        
        # Add resistance mechanisms
        barriers.extend(self.resistance_mechanisms)
        
        return barriers
    
    def identify_transformation_enablers(self) -> List[str]:
        """Identify factors that enable instrumental transformation."""
        enablers = []
        
        # High instrumental characteristics
        if self.instrumental_score and self.instrumental_score > 0.5:
            enablers.append("Strong instrumental foundation")
        
        # Specific instrumental types that enable change
        if InstrumentalType.PROBLEM_SOLVING in self.instrumental_behaviors:
            enablers.append("Problem-solving orientation")
        
        if InstrumentalType.INNOVATION_FOSTERING in self.instrumental_behaviors:
            enablers.append("Innovation-fostering culture")
        
        if InstrumentalType.ADAPTATION_PROMOTION in self.instrumental_behaviors:
            enablers.append("Adaptive capacity")
        
        # Transformation pressures
        enablers.extend(self.transformation_pressures)
        
        return enablers


@dataclass
class CeremonialBehaviorPattern(Node):
    """Models specific ceremonial behavior patterns within institutions."""
    
    pattern_type: Optional[CeremonialType] = None
    pattern_description: Optional[str] = None
    
    # Pattern characteristics
    entrenchment_level: Optional[float] = None  # How entrenched the pattern is (0-1)
    resistance_strength: Optional[float] = None  # Strength of resistance to change (0-1)
    legitimacy_sources: List[str] = field(default_factory=list)  # Sources of legitimacy  # type: ignore[misc]
    
    # Pattern manifestations
    behavioral_expressions: List[str] = field(default_factory=list)  # type: ignore[misc]
    symbolic_expressions: List[str] = field(default_factory=list)  # type: ignore[misc]
    institutional_expressions: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Pattern functions
    power_maintenance_function: Optional[float] = None  # How it maintains power (0-1)
    status_preservation_function: Optional[float] = None  # How it preserves status (0-1)
    change_resistance_function: Optional[float] = None  # How it resists change (0-1)
    
    # Pattern relationships
    supporting_actors: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    beneficiary_groups: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    disadvantaged_groups: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    
    # Pattern evolution
    emergence_context: Optional[str] = None
    historical_development: List[str] = field(default_factory=list)  # type: ignore[misc]
    adaptation_mechanisms: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # SFM integration
    matrix_ceremonial_effects: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    delivery_pattern_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)  # type: ignore[misc]
    institutional_pattern_embedding: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]


@dataclass
class InstrumentalBehaviorPattern(Node):
    """Models specific instrumental behavior patterns within institutions."""
    
    pattern_type: Optional[InstrumentalType] = None
    pattern_description: Optional[str] = None
    
    # Pattern characteristics
    efficiency_level: Optional[float] = None  # Efficiency of the pattern (0-1)
    innovation_potential: Optional[float] = None  # Innovation potential (0-1)
    problem_solving_capacity: Optional[float] = None  # Problem-solving capacity (0-1)
    
    # Pattern manifestations
    technological_expressions: List[str] = field(default_factory=list)  # type: ignore[misc]
    organizational_expressions: List[str] = field(default_factory=list)  # type: ignore[misc]
    behavioral_expressions: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Pattern functions
    efficiency_enhancement: Optional[float] = None  # How it enhances efficiency (0-1)
    knowledge_application: Optional[float] = None  # Knowledge application capacity (0-1)
    community_enhancement: Optional[float] = None  # Community enhancement potential (0-1)
    
    # Pattern enablers
    technological_enablers: List[str] = field(default_factory=list)  # type: ignore[misc]
    institutional_enablers: List[str] = field(default_factory=list)  # type: ignore[misc]
    cultural_enablers: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Pattern outcomes
    efficiency_gains: Dict[str, float] = field(default_factory=dict)  # type: ignore[misc]
    innovation_outcomes: List[str] = field(default_factory=list)  # type: ignore[misc]
    community_benefits: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Pattern diffusion
    adoption_mechanisms: List[str] = field(default_factory=list)  # type: ignore[misc]
    diffusion_barriers: List[str] = field(default_factory=list)  # type: ignore[misc]
    scaling_potential: Optional[float] = None  # Potential for scaling (0-1)
    
    # SFM integration
    matrix_instrumental_effects: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    delivery_enhancement_impacts: Dict[uuid.UUID, float] = field(default_factory=dict)  # type: ignore[misc]
    institutional_transformation_potential: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]


@dataclass
class DichotomyTransformation(Node):
    """Models transformation processes from ceremonial to instrumental orientations."""
    
    transformation_type: Optional[str] = None  # e.g., "Technological", "Organizational", "Cultural"
    transformation_scope: Optional[str] = None  # Scope of transformation
    
    # Transformation characteristics
    transformation_stage: Optional[TransformationStage] = None
    transformation_direction: Optional[str] = None  # "Ceremonial_to_Instrumental", "Mixed"
    transformation_intensity: Optional[float] = None  # Intensity of transformation (0-1)
    transformation_speed: Optional[str] = None  # "Gradual", "Rapid", "Sudden"
    
    # Transformation drivers
    internal_drivers: List[str] = field(default_factory=list)  # type: ignore[misc]
    external_drivers: List[str] = field(default_factory=list)  # type: ignore[misc]
    technological_drivers: List[str] = field(default_factory=list)  # type: ignore[misc]
    institutional_drivers: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Transformation process
    initiation_factors: List[str] = field(default_factory=list)  # type: ignore[misc]
    catalytic_events: List[str] = field(default_factory=list)  # type: ignore[misc]
    transformation_mechanisms: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Resistance and barriers
    ceremonial_resistance: List[str] = field(default_factory=list)  # type: ignore[misc]
    structural_barriers: List[str] = field(default_factory=list)  # type: ignore[misc]
    cultural_barriers: List[str] = field(default_factory=list)  # type: ignore[misc]
    political_barriers: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Transformation outcomes
    ceremonial_reduction: Optional[float] = None  # Reduction in ceremonial aspects
    instrumental_enhancement: Optional[float] = None  # Enhancement in instrumental aspects
    efficiency_improvements: Dict[str, float] = field(default_factory=dict)  # type: ignore[misc]
    institutional_changes: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Transformation sustainability
    sustainability_factors: List[str] = field(default_factory=list)  # type: ignore[misc]
    reversion_risks: List[str] = field(default_factory=list)  # type: ignore[misc]
    institutionalization_mechanisms: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # SFM integration
    matrix_transformation_effects: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    delivery_transformation_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)  # type: ignore[misc]
    institutional_transformation_relationships: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]


@dataclass
class ValueConflictAnalysis(Node):
    """Analyzes value conflicts through the ceremonial-instrumental lens."""
    
    conflict_description: Optional[str] = None
    conflicting_values: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Value characterization
    ceremonial_values: List[str] = field(default_factory=list)  # type: ignore[misc]
    instrumental_values: List[str] = field(default_factory=list)  # type: ignore[misc]
    value_tensions: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Conflict dynamics
    value_clash_intensity: Optional[float] = None  # Intensity of value clash (0-1)
    resolution_difficulty: Optional[float] = None  # Difficulty of resolution (0-1)
    compromise_potential: Optional[float] = None  # Potential for compromise (0-1)
    
    # Stakeholder value positions
    ceremonial_advocates: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    instrumental_advocates: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    neutral_parties: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    
    # Resolution approaches
    value_synthesis_approaches: List[str] = field(default_factory=list)  # type: ignore[misc]
    instrumental_reframing: List[str] = field(default_factory=list)  # type: ignore[misc]
    ceremonial_accommodation: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # Resolution outcomes
    value_transformation: Optional[str] = None  # How values transformed
    new_value_synthesis: List[str] = field(default_factory=list)  # type: ignore[misc]
    institutional_adjustments: List[str] = field(default_factory=list)  # type: ignore[misc]
    
    # SFM integration
    matrix_value_conflict_effects: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    delivery_value_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)  # type: ignore[misc]
    institutional_value_realignment: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]