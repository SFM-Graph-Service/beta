"""
Transactional Analysis Framework for Social Fabric Matrix analysis.

This module implements John R. Commons' transactional approach to institutional
analysis, providing the foundation for understanding how institutions coordinate
economic activity through different types of transactions. Essential for complete
SFM implementation following Hayden's institutional economics framework.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
from models.sfm_enums import (
    InstitutionalScope
)
# Note: TransactionType, PowerRelationType, AuthorityType are defined locally below

class TransactionType(Enum):
    """Commons' three fundamental transaction types in institutional analysis."""

    BARGAINING = auto()     # Market transactions between legal equals
    MANAGERIAL = auto()     # Hierarchical transactions with authority relations
    RATIONING = auto()      # Collective choice/governance transactions

class PowerRelationType(Enum):
    """Types of power relationships in transactions."""

    COERCION = auto()       # Physical or legal coercion
    ECONOMIC_DURESS = auto() # Economic pressure/necessity
    PERSUASION = auto()     # Influence through communication
    AUTHORITY = auto()      # Legitimate institutional authority
    BARGAINING_POWER = auto() # Market-based negotiating power
    COLLECTIVE_ACTION = auto() # Group-based power

class AuthorityType(Enum):
    """Types of institutional authority in transactional relationships."""

    LEGAL = auto()          # Legal/formal authority
    TRADITIONAL = auto()    # Traditional/customary authority
    CHARISMATIC = auto()    # Personal charismatic authority
    RATIONAL_LEGAL = auto() # Bureaucratic/procedural authority
    EXPERT = auto()         # Technical/professional authority
    MORAL = auto()          # Moral/ethical authority

class TransactionOutcome(Enum):
    """Possible outcomes of transactional processes."""

    MUTUAL_BENEFIT = auto()    # Win-win outcome
    ZERO_SUM = auto()         # One party gains, other loses
    NEGATIVE_SUM = auto()     # Both parties lose (inefficient)
    POSITIVE_SUM = auto()     # Both parties gain more than expected
    CONFLICT = auto()         # Unresolved conflict
    BREAKDOWN = auto()        # Transaction failure

@dataclass
class Transaction(Node):  # pylint: disable=too-many-instance-attributes
    """Models individual transactions within SFM institutional analysis."""

    transaction_type: Optional[TransactionType] = None
    participants: List[uuid.UUID] = field(default_factory=list)
    initiator: Optional[uuid.UUID] = None

    # Transaction context
    institutional_context: List[uuid.UUID] = field(default_factory=list)
    legal_framework: List[str] = field(default_factory=list)
    cultural_norms: List[str] = field(default_factory=list)

    # Power and authority analysis
    power_relationships: Dict[uuid.UUID, PowerRelationType] = field(default_factory=dict)
    authority_structures: Dict[uuid.UUID, AuthorityType] = field(default_factory=dict)
    power_asymmetries: Dict[str, float] = field(default_factory=dict)

    # Transaction content
    resources_exchanged: Dict[str, float] = field(default_factory=dict)
    services_provided: List[str] = field(default_factory=list)
    obligations_created: List[str] = field(default_factory=list)
    rights_transferred: List[str] = field(default_factory=list)

    # Process characteristics
    negotiation_duration: Optional[float] = None  # Duration in days
    complexity_level: Optional[float] = None  # 0-1 scale
    formalization_level: Optional[float] = None  # 0-1 scale
    transparency_level: Optional[float] = None  # 0-1 scale

    # Outcomes
    transaction_outcome: Optional[TransactionOutcome] = None
    satisfaction_levels: Dict[uuid.UUID, float] = field(default_factory=dict)
    efficiency_score: Optional[float] = None  # 0-1 scale
    fairness_perception: Dict[uuid.UUID, float] = field(default_factory=dict)

    # SFM integration
    matrix_cells_affected: List[uuid.UUID] = field(default_factory=list)
    delivery_relationships_created: List[uuid.UUID] = field(default_factory=list)
    institutional_adjustments: List[uuid.UUID] = field(default_factory=list)

@dataclass
class BargainingTransaction(Transaction):  # pylint: disable=too-many-instance-attributes
    """Models market-type bargaining transactions between legal equals."""

    def __post_init__(self):
        self.transaction_type = TransactionType.BARGAINING

    # Market characteristics
    market_structure: Optional[str] = None  # e.g., "Competitive", "Monopolistic"
    price_mechanism: Optional[str] = None  # e.g., "Auction", "Posted Price"
    information_symmetry: Optional[float] = None  # 0-1 scale

    # Bargaining process
    initial_offers: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    final_agreement: Dict[str, float] = field(default_factory=dict)
    bargaining_rounds: Optional[int] = None
    concessions_made: Dict[uuid.UUID, List[str]] = field(default_factory=dict)

    # Market outcomes
    price_efficiency: Optional[float] = None  # Proximity to theoretical optimum
    allocative_efficiency: Optional[float] = None  # Resource allocation quality
    transaction_costs: Optional[float] = None  # Costs of conducting transaction

    # SFM context
    ceremonial_constraints: List[str] = field(default_factory=list)
    instrumental_innovations: List[str] = field(default_factory=list)

@dataclass
class ManagerialTransaction(Transaction):  # pylint: disable=too-many-instance-attributes
    """Models hierarchical transactions with authority/subordination relationships."""

    def __post_init__(self):
        self.transaction_type = TransactionType.MANAGERIAL

    # Authority structure
    supervisor: Optional[uuid.UUID] = None
    subordinates: List[uuid.UUID] = field(default_factory=list)
    authority_scope: List[str] = field(default_factory=list)
    command_chain: List[uuid.UUID] = field(default_factory=list)

    # Management process
    directives_issued: List[str] = field(default_factory=list)
    compliance_level: Dict[uuid.UUID, float] = field(default_factory=dict)
    monitoring_mechanisms: List[str] = field(default_factory=list)
    feedback_loops: List[uuid.UUID] = field(default_factory=list)

    # Performance measurement
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    innovation_level: Optional[float] = None  # 0-1 scale

    # Coordination outcomes
    coordination_effectiveness: Optional[float] = None  # 0-1 scale
    goal_alignment: Optional[float] = None  # 0-1 scale
    organizational_learning: Optional[float] = None  # 0-1 scale

    # SFM context
    bureaucratic_constraints: List[str] = field(default_factory=list)
    adaptive_capacity: Optional[float] = None  # 0-1 scale

@dataclass
class RationingTransaction(Transaction):  # pylint: disable=too-many-instance-attributes
    """Models collective choice/rationing transactions in governance contexts."""

    def __post_init__(self):
        self.transaction_type = TransactionType.RATIONING

    # Governance context
    governing_body: Optional[uuid.UUID] = None
    affected_parties: List[uuid.UUID] = field(default_factory=list)
    decision_criteria: List[str] = field(default_factory=list)
    allocation_rules: List[str] = field(default_factory=list)

    # Rationing process
    resource_availability: Dict[str, float] = field(default_factory=dict)
    demand_assessment: Dict[uuid.UUID, float] = field(default_factory=dict)
    allocation_method: Optional[str] = None  # e.g., "Merit-based", "Need-based"
    priority_system: Dict[str, int] = field(default_factory=dict)

    # Democratic elements
    participation_level: Optional[float] = None  # 0-1 scale
    representation_quality: Optional[float] = None  # 0-1 scale
    deliberation_quality: Optional[float] = None  # 0-1 scale

    # Outcomes
    allocation_results: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    equity_score: Optional[float] = None  # 0-1 scale
    efficiency_score: Optional[float] = None  # 0-1 scale
    legitimacy_score: Optional[float] = None  # 0-1 scale

    # SFM integration
    democratic_process_link: Optional[uuid.UUID] = None
    policy_implementation_effects: List[uuid.UUID] = field(default_factory=list)

@dataclass
class TransactionCosts(Node):  # pylint: disable=too-many-instance-attributes
    """Models transaction costs associated with institutional coordination."""

    transaction_id: Optional[uuid.UUID] = None
    cost_categories: Dict[str, float] = field(default_factory=dict)

    # Specific cost types
    search_costs: Optional[float] = None  # Finding trading partners
    information_costs: Optional[float] = None  # Gathering information
    negotiation_costs: Optional[float] = None  # Bargaining process
    monitoring_costs: Optional[float] = None  # Ensuring compliance
    enforcement_costs: Optional[float] = None  # Dealing with violations

    # Institutional cost factors
    legal_complexity: Optional[float] = None  # Legal system complexity
    cultural_barriers: Optional[float] = None  # Cultural misunderstandings
    technology_costs: Optional[float] = None  # Technology requirements
    coordination_overhead: Optional[float] = None  # Administrative costs

    # Cost reduction mechanisms
    institutional_innovations: List[str] = field(default_factory=list)
    technology_solutions: List[str] = field(default_factory=list)
    standardization_effects: List[str] = field(default_factory=list)

    # SFM analysis
    ceremonial_cost_inflation: Optional[float] = None  # Ceremonial inefficiencies
    instrumental_cost_reduction: Optional[float] = None  # Efficiency improvements
    matrix_optimization_potential: Optional[float] = None  # System-wide savings

@dataclass
class InstitutionalContract(Node):  # pylint: disable=too-many-instance-attributes
    """Models formal and informal contracts governing transactional relationships."""

    contract_type: Optional[str] = None  # e.g., "Formal", "Relational", "Implicit"
    parties: List[uuid.UUID] = field(default_factory=list)

    # Contract content
    explicit_terms: List[str] = field(default_factory=list)
    implicit_understandings: List[str] = field(default_factory=list)
    performance_standards: Dict[str, float] = field(default_factory=dict)
    penalty_structures: Dict[str, float] = field(default_factory=dict)

    # Contract characteristics
    completeness_level: Optional[float] = None  # 0-1 scale
    flexibility_level: Optional[float] = None  # 0-1 scale
    enforcement_strength: Optional[float] = None  # 0-1 scale

    # Performance tracking
    compliance_rates: Dict[uuid.UUID, float] = field(default_factory=dict)
    dispute_frequency: Optional[float] = None  # Disputes per time period
    renegotiation_frequency: Optional[float] = None  # Renegotiations per time period

    # Evolution over time
    adaptation_mechanisms: List[str] = field(default_factory=list)
    learning_incorporation: List[str] = field(default_factory=list)
    relationship_development: Optional[float] = None  # Trust building over time

    # SFM context
    supporting_institutions: List[uuid.UUID] = field(default_factory=list)
    delivery_relationship_basis: List[uuid.UUID] = field(default_factory=list)
    matrix_stability_contribution: Optional[float] = None

@dataclass
class TransactionalRegime(Node):  # pylint: disable=too-many-instance-attributes
    """Models the broader transactional regime governing sets of related transactions."""

    regime_type: Optional[str] = None  # e.g., "Market", "Hierarchical", "Network"
    scope: Optional[InstitutionalScope] = None
    governing_institutions: List[uuid.UUID] = field(default_factory=list)

    # Regime characteristics
    dominant_transaction_types: List[TransactionType] = field(default_factory=list)
    coordination_mechanisms: List[str] = field(default_factory=list)
    governance_structure: Optional[str] = None

    # Performance metrics
    efficiency_level: Optional[float] = None  # 0-1 scale
    adaptability: Optional[float] = None  # 0-1 scale
    stability: Optional[float] = None  # 0-1 scale
    legitimacy: Optional[float] = None  # 0-1 scale

    # Evolution and change
    change_pressures: List[str] = field(default_factory=list)
    adaptation_mechanisms: List[str] = field(default_factory=list)
    innovation_capacity: Optional[float] = None  # 0-1 scale

    # SFM integration
    matrix_coordination_role: Optional[str] = None
    institutional_complementarities: List[uuid.UUID] = field(default_factory=list)
    system_level_effects: Dict[str, float] = field(default_factory=dict)

@dataclass
class PropertyRights(Node):  # pylint: disable=too-many-instance-attributes
    """Models property rights structures underlying transactional relationships."""

    rights_type: Optional[str] = None  # e.g., "Private", "Common", "Public"
    resource_reference: Optional[uuid.UUID] = None  # Resource being governed
    rights_holders: List[uuid.UUID] = field(default_factory=list)

    # Rights bundle analysis
    use_rights: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    control_rights: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    transfer_rights: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    exclusion_rights: Dict[uuid.UUID, List[str]] = field(default_factory=dict)

    # Rights characteristics
    clarity_level: Optional[float] = None  # 0-1 scale
    security_level: Optional[float] = None  # 0-1 scale
    transferability: Optional[float] = None  # 0-1 scale
    divisibility: Optional[float] = None  # 0-1 scale

    # Enforcement mechanisms
    formal_enforcement: List[str] = field(default_factory=list)
    informal_enforcement: List[str] = field(default_factory=list)
    enforcement_effectiveness: Optional[float] = None  # 0-1 scale

    # Performance outcomes
    resource_conservation: Optional[float] = None  # Conservation effectiveness
    investment_incentives: Optional[float] = None  # Investment promotion
    transaction_facilitation: Optional[float] = None  # Transaction cost reduction

    # SFM context
    institutional_support: List[uuid.UUID] = field(default_factory=list)
    ceremonial_legitimation: List[str] = field(default_factory=list)
    instrumental_efficiency: Optional[float] = None  # 0-1 scale
