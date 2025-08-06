"""
Comprehensive Institutional Mapping Framework for Social Fabric Matrix analysis.

This module implements systematic institutional identification and mapping following
Hayden's methodology. It provides structured approaches for identifying all relevant
institutional actors, mapping their relationships, and understanding their roles
in the SFM analysis.

Key Components:
- InstitutionalActor: Individual institutional entities
- InstitutionalMapping: Systematic mapping process
- InstitutionalNetwork: Network of institutional relationships
- InstitutionalHierarchy: Hierarchical institutional structures
- InstitutionalRoleAnalysis: Analysis of institutional roles and functions
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
from models.sfm_enums import (
    InstitutionalScope,
    SystemLevel,
    ValidationMethod,
)


class InstitutionalType(Enum):
    """Types of institutional actors in SFM analysis."""
    
    # Government and Public Sector
    FEDERAL_GOVERNMENT = auto()
    STATE_GOVERNMENT = auto()
    LOCAL_GOVERNMENT = auto()
    REGULATORY_AGENCY = auto()
    PUBLIC_ENTERPRISE = auto()
    
    # Private Sector
    CORPORATION = auto()
    SMALL_BUSINESS = auto()
    COOPERATIVE = auto()
    PARTNERSHIP = auto()
    SOLE_PROPRIETORSHIP = auto()
    
    # Civil Society
    NONPROFIT_ORGANIZATION = auto()
    COMMUNITY_ORGANIZATION = auto()
    ADVOCACY_GROUP = auto()
    LABOR_UNION = auto()
    PROFESSIONAL_ASSOCIATION = auto()
    
    # Social Institutions
    HOUSEHOLD = auto()
    FAMILY = auto()
    EDUCATIONAL_INSTITUTION = auto()
    RELIGIOUS_INSTITUTION = auto()
    CULTURAL_INSTITUTION = auto()
    
    # Economic Institutions
    FINANCIAL_INSTITUTION = auto()
    MARKET_INSTITUTION = auto()
    TRADE_ASSOCIATION = auto()
    ECONOMIC_DEVELOPMENT_AGENCY = auto()
    
    # Infrastructure and Utilities
    UTILITY_PROVIDER = auto()
    INFRASTRUCTURE_OPERATOR = auto()
    TRANSPORTATION_AUTHORITY = auto()
    COMMUNICATION_PROVIDER = auto()


class InstitutionalRole(Enum):
    """Primary roles that institutions play in SFM systems."""
    
    # Decision-Making Roles
    POLICY_MAKER = auto()          # Creates policies and rules
    REGULATOR = auto()             # Enforces rules and standards
    COORDINATOR = auto()           # Coordinates activities
    FACILITATOR = auto()           # Enables and supports
    
    # Operational Roles  
    SERVICE_PROVIDER = auto()      # Delivers services
    RESOURCE_PROVIDER = auto()     # Provides resources
    PRODUCER = auto()              # Produces goods/services
    DISTRIBUTOR = auto()           # Distributes goods/services
    
    # Support Roles
    FINANCIER = auto()             # Provides funding
    INFORMATION_PROVIDER = auto()  # Provides information
    CAPACITY_BUILDER = auto()      # Builds capacities
    ADVOCATE = auto()              # Advocates for interests
    
    # Oversight Roles
    MONITOR = auto()               # Monitors performance
    EVALUATOR = auto()             # Evaluates outcomes
    AUDITOR = auto()               # Audits compliance
    QUALITY_ASSURER = auto()       # Ensures quality


class InstitutionalPowerLevel(Enum):
    """Levels of institutional power and influence."""
    
    DOMINANT = auto()              # Dominant influence in system
    MAJOR = auto()                 # Major influence
    MODERATE = auto()              # Moderate influence  
    MINOR = auto()                 # Minor influence
    MINIMAL = auto()               # Minimal influence


class RelationshipType(Enum):
    """Types of relationships between institutional actors."""
    
    # Hierarchical Relationships
    SUBORDINATION = auto()         # Hierarchical subordination
    SUPERVISION = auto()           # Oversight and supervision
    DELEGATION = auto()            # Delegated authority
    
    # Collaborative Relationships
    PARTNERSHIP = auto()           # Equal partnership
    ALLIANCE = auto()              # Strategic alliance
    COOPERATION = auto()           # Cooperative relationship
    COORDINATION = auto()          # Coordinated activities
    
    # Exchange Relationships
    CONTRACTUAL = auto()           # Contract-based relationship
    MARKET_EXCHANGE = auto()       # Market transactions
    RESOURCE_SHARING = auto()      # Resource sharing
    INFORMATION_SHARING = auto()   # Information exchange
    
    # Competitive Relationships
    COMPETITION = auto()           # Direct competition
    RIVALRY = auto()               # Competitive rivalry
    CONFLICT = auto()              # Conflictual relationship
    
    # Support Relationships
    FUNDING = auto()               # Financial support
    TECHNICAL_ASSISTANCE = auto()  # Technical support
    ADVOCACY_SUPPORT = auto()      # Advocacy relationship


@dataclass
class InstitutionalActor(Node):
    """Individual institutional entity in SFM analysis."""
    
    institutional_type: Optional[InstitutionalType] = None
    primary_role: Optional[InstitutionalRole] = None
    secondary_roles: List[InstitutionalRole] = field(default_factory=list)
    
    # Basic characteristics
    legal_status: Optional[str] = None
    founding_date: Optional[datetime] = None
    jurisdiction: Optional[str] = None
    geographic_scope: Optional[str] = None
    
    # Organizational characteristics
    organizational_structure: Optional[str] = None
    governance_structure: Optional[str] = None
    decision_making_process: Optional[str] = None
    leadership_structure: List[str] = field(default_factory=list)
    
    # Capacity and resources
    human_resources: Optional[int] = None  # Number of staff/members
    financial_resources: Optional[float] = None  # Annual budget/revenue
    technical_capacity: Optional[float] = None  # Technical capability (0-1)
    organizational_capacity: Optional[float] = None  # Organizational effectiveness (0-1)
    
    # Power and influence
    institutional_power: Optional[InstitutionalPowerLevel] = None
    influence_scope: Optional[InstitutionalScope] = None
    system_level_influence: List[SystemLevel] = field(default_factory=list)
    veto_power_areas: List[str] = field(default_factory=list)
    
    # Functions and activities
    primary_functions: List[str] = field(default_factory=list)
    service_delivery_areas: List[str] = field(default_factory=list)
    policy_influence_areas: List[str] = field(default_factory=list)
    resource_control_areas: List[str] = field(default_factory=list)
    
    # Relationships
    superior_institutions: List[uuid.UUID] = field(default_factory=list)
    subordinate_institutions: List[uuid.UUID] = field(default_factory=list)
    partner_institutions: List[uuid.UUID] = field(default_factory=list)
    competitor_institutions: List[uuid.UUID] = field(default_factory=list)
    
    # Performance and effectiveness
    institutional_effectiveness: Optional[float] = None  # Overall effectiveness (0-1)
    service_quality: Optional[float] = None  # Quality of services (0-1)
    stakeholder_satisfaction: Optional[float] = None  # Stakeholder satisfaction (0-1)
    adaptive_capacity: Optional[float] = None  # Ability to adapt (0-1)
    
    # SFM integration
    matrix_relevance: Optional[float] = None  # Relevance to SFM analysis (0-1)
    criteria_influence: List[uuid.UUID] = field(default_factory=list)
    delivery_responsibilities: List[uuid.UUID] = field(default_factory=list)
    stakeholder_representation: List[uuid.UUID] = field(default_factory=list)


@dataclass
class InstitutionalMapping(Node):
    """Systematic process for mapping institutional actors."""
    
    mapping_scope: Optional[str] = None
    mapping_methodology: Optional[str] = None
    mapping_date: Optional[datetime] = None
    
    # Mapping process
    identification_method: Optional[str] = None
    identification_criteria: List[str] = field(default_factory=list)
    validation_methods: List[ValidationMethod] = field(default_factory=list)
    
    # Institutional inventory
    identified_institutions: List[uuid.UUID] = field(default_factory=list)
    institutions_by_type: Dict[InstitutionalType, List[uuid.UUID]] = field(default_factory=dict)
    institutions_by_role: Dict[InstitutionalRole, List[uuid.UUID]] = field(default_factory=dict)
    institutions_by_level: Dict[SystemLevel, List[uuid.UUID]] = field(default_factory=dict)
    
    # Mapping characteristics
    mapping_completeness: Optional[float] = None  # Completeness assessment (0-1)
    mapping_accuracy: Optional[float] = None     # Accuracy assessment (0-1)
    institutional_coverage: Optional[float] = None  # Coverage of relevant institutions (0-1)
    
    # Quality assurance
    verification_methods: List[str] = field(default_factory=list)
    validation_sources: List[str] = field(default_factory=list)
    expert_review_status: Optional[str] = None
    stakeholder_validation: Dict[str, bool] = field(default_factory=dict)
    
    # Mapping updates
    update_frequency: Optional[str] = None
    last_update_date: Optional[datetime] = None
    update_triggers: List[str] = field(default_factory=list)
    change_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Usage and application
    mapping_applications: List[str] = field(default_factory=list)
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)


@dataclass
class InstitutionalNetwork(Node):
    """Network of relationships between institutional actors."""
    
    network_scope: Optional[str] = None
    network_domain: Optional[str] = None
    
    # Network structure
    network_nodes: List[uuid.UUID] = field(default_factory=list)
    network_relationships: List[uuid.UUID] = field(default_factory=list)
    
    # Network metrics
    network_size: Optional[int] = None  # Number of nodes
    network_density: Optional[float] = None  # Connection density (0-1)
    average_degree: Optional[float] = None  # Average connections per node
    clustering_coefficient: Optional[float] = None  # Local clustering
    
    # Network characteristics
    centralization: Optional[float] = None  # Network centralization (0-1)
    hierarchy_level: Optional[float] = None  # Hierarchical organization (0-1)
    fragmentation: Optional[float] = None  # Network fragmentation (0-1)
    connectivity: Optional[float] = None    # Overall connectivity (0-1)
    
    # Key network positions
    central_institutions: List[uuid.UUID] = field(default_factory=list)
    bridge_institutions: List[uuid.UUID] = field(default_factory=list)
    peripheral_institutions: List[uuid.UUID] = field(default_factory=list)
    isolated_institutions: List[uuid.UUID] = field(default_factory=list)
    
    # Network dynamics
    network_stability: Optional[float] = None  # Stability over time (0-1)
    change_rate: Optional[float] = None        # Rate of network change
    evolution_pattern: Optional[str] = None    # Pattern of network evolution
    
    # Network performance
    coordination_effectiveness: Optional[float] = None  # Coordination quality (0-1)
    information_flow_quality: Optional[float] = None   # Information flow (0-1)
    collective_action_capacity: Optional[float] = None # Collective action (0-1)
    resilience: Optional[float] = None                 # Network resilience (0-1)


@dataclass
class InstitutionalHierarchy(Node):
    """Hierarchical structure of institutional relationships."""
    
    hierarchy_type: Optional[str] = None  # Type of hierarchy (legal, functional, etc.)
    hierarchy_levels: Optional[int] = None  # Number of hierarchical levels
    
    # Hierarchical structure
    root_institutions: List[uuid.UUID] = field(default_factory=list)
    level_structure: Dict[int, List[uuid.UUID]] = field(default_factory=dict)
    parent_child_relationships: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)
    
    # Authority relationships
    command_relationships: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)
    delegation_relationships: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)
    oversight_relationships: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)
    
    # Hierarchy characteristics
    span_of_control: Dict[uuid.UUID, int] = field(default_factory=dict)
    hierarchy_depth: Optional[int] = None  # Maximum depth
    hierarchy_breadth: Optional[int] = None  # Maximum breadth
    formalization_level: Optional[float] = None  # Level of formalization (0-1)
    
    # Hierarchy effectiveness
    coordination_efficiency: Optional[float] = None  # Coordination efficiency (0-1)
    decision_speed: Optional[float] = None          # Decision-making speed (0-1)
    accountability_clarity: Optional[float] = None  # Accountability clarity (0-1)
    flexibility: Optional[float] = None             # Hierarchical flexibility (0-1)
    
    # Hierarchy dynamics
    hierarchy_stability: Optional[float] = None  # Stability of hierarchy (0-1)
    restructuring_frequency: Optional[float] = None  # Frequency of changes
    adaptation_capacity: Optional[float] = None     # Capacity to adapt (0-1)


@dataclass
class InstitutionalRoleAnalysis(Node):
    """Analysis of institutional roles and functions in SFM systems."""
    
    analysis_scope: Optional[str] = None
    analysis_methodology: Optional[str] = None
    
    # Role mapping
    role_assignments: Dict[uuid.UUID, List[InstitutionalRole]] = field(default_factory=dict)
    role_conflicts: List[Tuple[uuid.UUID, InstitutionalRole]] = field(default_factory=list)
    role_gaps: List[InstitutionalRole] = field(default_factory=list)
    role_overlaps: List[Tuple[InstitutionalRole, List[uuid.UUID]]] = field(default_factory=list)
    
    # Role performance
    role_effectiveness: Dict[uuid.UUID, Dict[InstitutionalRole, float]] = field(default_factory=dict)
    role_capacity: Dict[uuid.UUID, Dict[InstitutionalRole, float]] = field(default_factory=dict)
    role_satisfaction: Dict[uuid.UUID, Dict[InstitutionalRole, float]] = field(default_factory=dict)
    
    # System role analysis
    critical_roles: List[InstitutionalRole] = field(default_factory=list)
    underperforming_roles: List[InstitutionalRole] = field(default_factory=list)
    redundant_roles: List[InstitutionalRole] = field(default_factory=list)
    missing_roles: List[InstitutionalRole] = field(default_factory=list)
    
    # Role optimization
    role_reallocation_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    capacity_building_needs: Dict[uuid.UUID, List[InstitutionalRole]] = field(default_factory=dict)
    structural_improvements: List[str] = field(default_factory=list)
    
    # Stakeholder perspectives
    stakeholder_role_expectations: Dict[str, Dict[uuid.UUID, List[InstitutionalRole]]] = field(default_factory=dict)
    role_legitimacy_assessment: Dict[uuid.UUID, Dict[InstitutionalRole, float]] = field(default_factory=dict)
    role_accountability_mechanisms: Dict[uuid.UUID, List[str]] = field(default_factory=dict)