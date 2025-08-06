"""
Hayden's Standardized Criteria Taxonomy for Social Fabric Matrix analysis.

This module implements F. Gregory Hayden's standardized criteria taxonomy which is
fundamental to consistent SFM analysis. It provides the core evaluation criteria
that serve as the columns in Hayden's matrix methodology, ensuring systematic and
comparable analysis across different SFM applications.

Key Components:
- HaydenStandardCriteria: Hayden's complete standardized criteria set
- CriteriaTaxonomy: Hierarchical organization of criteria
- CriteriaSpecification: Detailed specifications for each criterion
- CriteriaWeighting: Systematic weighting methodology
- CriteriaValidation: Validation and consistency checking
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum, auto

from models.base_nodes import Node
from models.sfm_enums import (
    ValueCategory,
    SystemLevel,
    ValidationMethod,
    EvidenceQuality,
)


class HaydenCriteriaCategory(Enum):
    """Hayden's primary criteria categories for SFM analysis."""
    
    # Core Life Process Enhancement Criteria
    SECURITY = auto()           # Safety, stability, predictability, protection
    EQUITY = auto()            # Fairness, justice, equal access, distribution
    LIBERTY = auto()           # Freedom, autonomy, choice, self-determination
    EFFICIENCY = auto()        # Resource optimization, productivity, effectiveness
    
    # Environmental and Sustainability Criteria  
    ENVIRONMENTAL_QUALITY = auto()    # Environmental health, sustainability
    RESOURCE_CONSERVATION = auto()    # Conservation, stewardship, prudence
    
    # Social and Cultural Criteria
    COMMUNITY_COHESION = auto()      # Social bonds, solidarity, integration
    CULTURAL_PRESERVATION = auto()   # Cultural identity, heritage, diversity
    DEMOCRATIC_PARTICIPATION = auto() # Participation, inclusion, representation
    
    # Economic and Institutional Criteria
    ECONOMIC_STABILITY = auto()      # Economic security, stability, resilience
    INSTITUTIONAL_CAPACITY = auto()  # Institutional effectiveness, governance
    TECHNOLOGICAL_APPROPRIATENESS = auto()  # Technology fit, innovation


class CriteriaImportanceLevel(Enum):
    """Importance levels for standardized criteria."""
    
    FUNDAMENTAL = auto()        # Core life process criteria
    ESSENTIAL = auto()         # Critical supporting criteria  
    IMPORTANT = auto()         # Significant contributing criteria
    SUPPORTING = auto()        # Additional supporting criteria
    CONTEXTUAL = auto()        # Context-specific criteria


class CriteriaMeasurementDimension(Enum):
    """Dimensions for criteria measurement."""
    
    MAGNITUDE = auto()         # Scale or size of impact
    SCOPE = auto()            # Breadth of impact
    DURATION = auto()         # Time dimension of impact
    QUALITY = auto()          # Qualitative aspects
    DISTRIBUTION = auto()     # How impacts are distributed
    SUSTAINABILITY = auto()   # Long-term viability


@dataclass
class HaydenStandardCriteria(Node):
    """Implementation of Hayden's standardized criteria for SFM analysis."""
    
    criteria_category: Optional[HaydenCriteriaCategory] = None
    importance_level: Optional[CriteriaImportanceLevel] = None
    
    # Criteria specification
    operational_definition: Optional[str] = None
    measurement_approach: Optional[str] = None
    measurement_dimensions: List[CriteriaMeasurementDimension] = field(default_factory=list)
    
    # Criteria characteristics
    life_process_enhancement: Optional[float] = None  # 0-1 scale
    instrumental_value: Optional[float] = None  # Instrumental value component
    ceremonial_value: Optional[float] = None   # Ceremonial value component
    
    # Measurement specifications
    quantitative_indicators: List[str] = field(default_factory=list)
    qualitative_indicators: List[str] = field(default_factory=list)
    proxy_measures: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    
    # Relationships to other criteria
    complementary_criteria: List[uuid.UUID] = field(default_factory=list)
    conflicting_criteria: List[uuid.UUID] = field(default_factory=list)
    hierarchical_relationships: Dict[uuid.UUID, str] = field(default_factory=dict)
    
    # Context and scope
    applicable_contexts: List[str] = field(default_factory=list)
    system_level_relevance: List[SystemLevel] = field(default_factory=list)
    cultural_sensitivity: Optional[float] = None  # Cultural adaptation needed (0-1)
    
    # Validation and quality
    validation_methods: List[ValidationMethod] = field(default_factory=list)
    evidence_requirements: Optional[EvidenceQuality] = None
    reliability_score: Optional[float] = None  # Reliability of measurement (0-1)
    validity_score: Optional[float] = None     # Validity of measurement (0-1)
    
    # Weighting and prioritization
    default_weight: Optional[float] = None     # Default weight in analysis (0-1)
    weight_rationale: Optional[str] = None
    stakeholder_priorities: Dict[str, float] = field(default_factory=dict)
    
    # Implementation guidance
    implementation_guidelines: List[str] = field(default_factory=list)
    common_challenges: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)


@dataclass
class CriteriaTaxonomy(Node):
    """Hierarchical organization of Hayden's standardized criteria."""
    
    taxonomy_name: Optional[str] = None
    taxonomy_version: Optional[str] = None
    
    # Hierarchical structure
    root_criteria: List[uuid.UUID] = field(default_factory=list)
    parent_child_relationships: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)
    criteria_levels: Dict[int, List[uuid.UUID]] = field(default_factory=dict)
    
    # Criteria organization
    criteria_by_category: Dict[HaydenCriteriaCategory, List[uuid.UUID]] = field(default_factory=dict)
    criteria_by_importance: Dict[CriteriaImportanceLevel, List[uuid.UUID]] = field(default_factory=dict)
    criteria_by_system_level: Dict[SystemLevel, List[uuid.UUID]] = field(default_factory=dict)
    
    # Taxonomy characteristics
    completeness_assessment: Optional[float] = None  # Coverage assessment (0-1)
    mutual_exclusivity: Optional[float] = None       # Non-overlap assessment (0-1)
    balance_assessment: Optional[float] = None       # Balance across categories (0-1)
    
    # Usage and application
    application_contexts: List[str] = field(default_factory=list)
    usage_guidelines: List[str] = field(default_factory=list)
    customization_options: List[str] = field(default_factory=list)
    
    # Validation and maintenance
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    peer_review_status: Optional[str] = None


@dataclass 
class CriteriaSpecification(Node):
    """Detailed specification for individual standardized criteria."""
    
    criteria_reference: Optional[uuid.UUID] = None
    specification_version: Optional[str] = None
    
    # Core specification
    formal_definition: Optional[str] = None
    operational_definition: Optional[str] = None
    conceptual_framework: Optional[str] = None
    theoretical_foundation: Optional[str] = None
    
    # Measurement specification
    primary_measures: List[str] = field(default_factory=list)
    secondary_measures: List[str] = field(default_factory=list)
    measurement_protocol: Optional[str] = None
    data_collection_methods: List[str] = field(default_factory=list)
    
    # Scoring methodology
    scoring_scale: Optional[str] = None  # e.g., "-3 to +3", "0 to 100"
    scoring_guidelines: List[str] = field(default_factory=list)
    scoring_examples: List[Dict[str, Any]] = field(default_factory=list)
    inter_rater_reliability: Optional[float] = None
    
    # Quality assurance
    validation_criteria: List[str] = field(default_factory=list)
    quality_control_measures: List[str] = field(default_factory=list)
    error_detection_methods: List[str] = field(default_factory=list)
    
    # Application guidance
    when_to_use: List[str] = field(default_factory=list)
    when_not_to_use: List[str] = field(default_factory=list)
    adaptation_guidelines: List[str] = field(default_factory=list)
    common_misapplications: List[str] = field(default_factory=list)
    
    # Relationships and dependencies
    prerequisite_criteria: List[uuid.UUID] = field(default_factory=list)
    complementary_specifications: List[uuid.UUID] = field(default_factory=list)
    conflicting_specifications: List[uuid.UUID] = field(default_factory=list)


@dataclass
class CriteriaWeighting(Node):
    """Systematic methodology for weighting standardized criteria."""
    
    weighting_methodology: Optional[str] = None
    weighting_context: Optional[str] = None
    
    # Weight assignments
    criteria_weights: Dict[uuid.UUID, float] = field(default_factory=dict)
    category_weights: Dict[HaydenCriteriaCategory, float] = field(default_factory=dict)
    importance_weights: Dict[CriteriaImportanceLevel, float] = field(default_factory=dict)
    
    # Weighting rationale
    weighting_principles: List[str] = field(default_factory=list)
    stakeholder_input: Dict[str, Dict[uuid.UUID, float]] = field(default_factory=dict)
    expert_judgments: Dict[str, Dict[uuid.UUID, float]] = field(default_factory=dict)
    empirical_basis: List[str] = field(default_factory=list)
    
    # Sensitivity analysis
    weight_sensitivity: Dict[uuid.UUID, float] = field(default_factory=dict)
    robustness_testing: List[Dict[str, Any]] = field(default_factory=list)
    alternative_weightings: List[Dict[uuid.UUID, float]] = field(default_factory=list)
    
    # Validation and consensus
    stakeholder_agreement: Optional[float] = None  # Level of agreement (0-1)
    expert_consensus: Optional[float] = None       # Expert consensus level (0-1)
    weight_stability: Optional[float] = None       # Stability over time (0-1)
    
    # Application guidelines
    weight_adjustment_rules: List[str] = field(default_factory=list)
    context_specific_modifications: Dict[str, Dict[uuid.UUID, float]] = field(default_factory=dict)
    update_criteria: List[str] = field(default_factory=list)


@dataclass
class CriteriaValidation(Node):
    """Validation framework for standardized criteria implementation."""
    
    validation_scope: Optional[str] = None
    validation_date: Optional[str] = None
    
    # Validation dimensions
    content_validity: Optional[float] = None       # Content coverage (0-1)
    construct_validity: Optional[float] = None     # Construct measurement (0-1)
    criterion_validity: Optional[float] = None     # Predictive validity (0-1)
    face_validity: Optional[float] = None          # Apparent validity (0-1)
    
    # Reliability measures
    internal_consistency: Optional[float] = None   # Cronbach's alpha
    test_retest_reliability: Optional[float] = None
    inter_rater_reliability: Optional[float] = None
    
    # Stakeholder validation
    stakeholder_acceptance: Dict[str, float] = field(default_factory=dict)
    expert_endorsement: Dict[str, float] = field(default_factory=dict)
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)
    
    # Empirical validation
    empirical_studies: List[str] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    statistical_validation: Dict[str, float] = field(default_factory=dict)
    
    # Cross-cultural validation
    cultural_applicability: Dict[str, float] = field(default_factory=dict)
    cross_cultural_studies: List[str] = field(default_factory=list)
    adaptation_requirements: List[str] = field(default_factory=list)
    
    # Continuous validation
    ongoing_validation_plan: List[str] = field(default_factory=list)
    validation_update_schedule: Optional[str] = None
    validation_improvement_recommendations: List[str] = field(default_factory=list)