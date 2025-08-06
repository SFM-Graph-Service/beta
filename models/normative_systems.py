"""
Normative systems framework with deontic logic for Social Fabric Matrix modeling.

This module implements Hayden's normative systems analysis based on deontic logic
categories from Polanyi and Commons. It provides formal structures for analyzing
permissions, obligations, prohibitions, and value judgments in institutional systems.

Key Components:
- NormativeRule: Individual deontic rules (permissions, obligations, prohibitions)
- NormativeSystem: Collection of rules forming a coherent normative framework
- ValueJudgment: Explicit value judgments in policy analysis
- DeontologicalAnalyzer: Tools for analyzing normative rule systems
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
from models.sfm_enums import (
    DeontologicalCategory,
    ValueJudgmentType,
    NormativeFramework,
    ValueCategory,
    InstitutionalScope,
    EnforcementType,
)

class NormativeConflictType(Enum):
    """Types of conflicts between normative rules."""

    PERMISSION_PROHIBITION = auto()  # Something both permitted and prohibited
    OBLIGATION_PROHIBITION = auto()  # Something both required and forbidden
    COMPETING_OBLIGATIONS = auto()   # Multiple conflicting obligations
    SCOPE_CONFLICT = auto()         # Different rules for same scope
    TEMPORAL_CONFLICT = auto()      # Rules conflict over time
    AUTHORITY_CONFLICT = auto()     # Different authorities specify different rules

class RuleStrength(Enum):
    """Strength or bindingness of normative rules."""

    ABSOLUTE = auto()     # Cannot be overridden
    STRONG = auto()       # High priority, rarely overridden
    MODERATE = auto()     # Standard priority
    WEAK = auto()         # Low priority, easily overridden
    ADVISORY = auto()     # Recommendation only

class LogicalOperator(Enum):
    """Logical operators for combining normative rules."""

    AND = auto()          # All conditions must be met
    OR = auto()           # Any condition can be met
    NOT = auto()          # Negation
    IF_THEN = auto()      # Conditional
    IF_AND_ONLY_IF = auto() # Biconditional

@dataclass
class NormativeCondition:
    """Condition that must be met for a normative rule to apply."""

    condition_type: str  # Type of condition (temporal, spatial, contextual, etc.)
    condition_value: Any  # Value or specification of the condition
    operator: LogicalOperator = LogicalOperator.AND

    # Metadata
    description: str = ""
    is_negated: bool = False

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate whether this condition is met in the given context."""
        context_value = context.get(self.condition_type)

        if context_value is None:
            return False

        # Simple evaluation - in practice, this would be more sophisticated
        result = False

        if self.condition_type == "temporal":
            # Check if current time meets temporal condition
            current_time = context.get("current_time", datetime.now())
            if isinstance(self.condition_value, tuple):
                start_time, end_time = self.condition_value
                result = start_time <= current_time <= end_time
            else:
                result = current_time >= self.condition_value

        elif self.condition_type == "spatial":
            # Check spatial condition
            current_location = context.get("spatial_unit")
            result = current_location == self.condition_value

        elif self.condition_type == "actor_type":
            # Check actor type condition
            actor_type = context.get("actor_type")
            result = actor_type == self.condition_value

        else:
            # Generic equality check
            result = context_value == self.condition_value

        return not result if self.is_negated else result

@dataclass
class NormativeRule(Node):
    """Individual deontic rule representing permissions, obligations, or prohibitions."""

    deontic_category: DeontologicalCategory
    subject: str  # Who the rule applies to (actor, role, etc.)
    object: str   # What the rule is about (action, resource, etc.)

    # Rule specification
    conditions: List[NormativeCondition] = field(default_factory=list)
    rule_strength: RuleStrength = RuleStrength.MODERATE
    enforcement_type: EnforcementType = EnforcementType.SOCIAL

    # Authority and legitimacy
    issuing_authority: Optional[uuid.UUID] = None  # Institution that issued the rule
    legitimacy_source: Optional[str] = None  # Basis of rule legitimacy
    authority_scope: InstitutionalScope = InstitutionalScope.LOCAL

    # Rule relationships
    prerequisite_rules: List[uuid.UUID] = field(default_factory=list)
    conflicting_rules: List[uuid.UUID] = field(default_factory=list)
    derivative_rules: List[uuid.UUID] = field(default_factory=list)  # Rules derived from this one

    # Effectiveness and compliance
    compliance_rate: Optional[float] = None  # Observed compliance (0-1)
    enforcement_effectiveness: Optional[float] = None  # How well enforced (0-1)
    violation_penalty: Optional[str] = None  # Consequences of violation

    # SFM integration
    affected_matrix_cells: List[uuid.UUID] = field(default_factory=list)
    ceremonial_component: Optional[float] = None  # Ceremonial aspect (0-1)
    instrumental_component: Optional[float] = None  # Instrumental aspect (0-1)

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this rule applies in the given context."""
        if not self.conditions:
            return True

        # Evaluate all conditions
        condition_results = []
        current_operator = LogicalOperator.AND

        for condition in self.conditions:
            result = condition.evaluate(context)
            condition_results.append((result, condition.operator))

        # Combine condition results (simplified logic)
        if not condition_results:
            return True

        # Start with first condition
        overall_result = condition_results[0][0]

        for i in range(1, len(condition_results)):
            result, operator = condition_results[i]

            if operator == LogicalOperator.AND:
                overall_result = overall_result and result
            elif operator == LogicalOperator.OR:
                overall_result = overall_result or result
            # Add more operators as needed

        return overall_result

    def get_deontic_status(self, action: str, context: Dict[str, Any]) -> Optional[str]:
        """Get the deontic status of an action under this rule."""
        if not self.is_applicable(context) or self.object != action:
            return None

        if self.deontic_category == DeontologicalCategory.PERMISSION:
            return "permitted"
        elif self.deontic_category == DeontologicalCategory.OBLIGATION:
            return "obligatory"
        elif self.deontic_category == DeontologicalCategory.PROHIBITION:
            return "prohibited"
        elif self.deontic_category == DeontologicalCategory.RIGHT:
            return "right"
        elif self.deontic_category == DeontologicalCategory.DUTY:
            return "duty"

        return "unspecified"

    def calculate_rule_effectiveness(self) -> float:
        """Calculate overall effectiveness of this rule."""
        effectiveness_factors = []

        if self.compliance_rate is not None:
            effectiveness_factors.append(self.compliance_rate * 0.4)

        if self.enforcement_effectiveness is not None:
            effectiveness_factors.append(self.enforcement_effectiveness * 0.4)

        # Rule strength contribution
        strength_values = {
            RuleStrength.ABSOLUTE: 1.0,
            RuleStrength.STRONG: 0.8,
            RuleStrength.MODERATE: 0.6,
            RuleStrength.WEAK: 0.4,
            RuleStrength.ADVISORY: 0.2
        }
        effectiveness_factors.append(strength_values[self.rule_strength] * 0.2)

        return sum(effectiveness_factors) / len(effectiveness_factors) if effectiveness_factors else 0.0

    def calculate_ceremonial_instrumental_balance(self) -> Optional[float]:
        """Calculate balance between ceremonial and instrumental aspects."""
        if self.ceremonial_component is None or self.instrumental_component is None:
            return None

        total = self.ceremonial_component + self.instrumental_component
        if total == 0:
            return 0.0

        # Return value from -1 (purely ceremonial) to +1 (purely instrumental)
        return (self.instrumental_component - self.ceremonial_component) / total

@dataclass
class ValueJudgment(Node):
    """Explicit value judgment used in policy analysis."""

    judgment_type: ValueJudgmentType
    value_category: ValueCategory
    judgment_statement: str  # The actual value judgment being made

    # Judgment context
    policy_context: Optional[uuid.UUID] = None  # Related policy
    decision_context: str = ""  # Context in which judgment is made
    stakeholder_perspectives: Dict[str, str] = field(default_factory=dict)

    # Justification and support
    ethical_framework: Optional[str] = None  # Underlying ethical framework
    supporting_evidence: List[str] = field(default_factory=list)
    counter_arguments: List[str] = field(default_factory=list)

    # Judgment properties
    confidence_level: float = 0.5  # Confidence in the judgment (0-1)
    consensus_level: Optional[float] = None  # Level of stakeholder consensus (0-1)
    controversy_level: Optional[float] = None  # How controversial the judgment is (0-1)

    # Authority and legitimacy
    making_authority: Optional[uuid.UUID] = None  # Who made the judgment
    authority_legitimacy: Optional[float] = None  # Legitimacy of the authority (0-1)

    # SFM integration
    affected_institutions: List[uuid.UUID] = field(default_factory=list)
    value_trade_offs: Dict[str, float] = field(default_factory=dict)  # Trade-offs with other values

    def assess_judgment_quality(self) -> Dict[str, float]:
        """Assess the quality of this value judgment."""
        quality_factors = {}

        # Evidence support
        evidence_quality = min(
            len(self.supporting_evidence) / 3.0,
            1.0)  # Up to 3 pieces of evidence
        quality_factors['evidence_support'] = evidence_quality

        # Consideration of counter-arguments
        balance_quality = min(len(self.counter_arguments) / 2.0, 1.0)  # Up to 2 counter-arguments
        quality_factors['argument_balance'] = balance_quality

        # Stakeholder consideration
        stakeholder_quality = min(
            len(self.stakeholder_perspectives) / 4.0,
            1.0)  # Up to 4 perspectives
        quality_factors['stakeholder_inclusion'] = stakeholder_quality

        # Confidence and consensus
        if self.confidence_level is not None:
            quality_factors['confidence'] = self.confidence_level

        if self.consensus_level is not None:
            quality_factors['consensus'] = self.consensus_level

        # Authority legitimacy
        if self.authority_legitimacy is not None:
            quality_factors['authority'] = self.authority_legitimacy

        # Overall quality score
        overall_quality = sum(quality_factors.values()) / len(quality_factors) if quality_factors else 0.0
        quality_factors['overall'] = overall_quality

        return quality_factors

    def identify_value_conflicts(
        self,
        other_judgments: List[ValueJudgment]) -> List[Dict[str, Any]]:
        """Identify conflicts with other value judgments."""
        conflicts = []

        for other in other_judgments:
            if other.id == self.id:
                continue

            conflict_indicators = []

            # Check for direct contradiction in same category
            if (self.value_category == other.value_category and
                self.judgment_statement.lower() != other.judgment_statement.lower()):
                conflict_indicators.append("category_contradiction")

            # Check for trade-off conflicts
            for value, trade_off in self.value_trade_offs.items():
                if value in other.value_trade_offs:
                    other_trade_off = other.value_trade_offs[value]
                    if abs(trade_off - other_trade_off) > 0.5:  # Significant difference
                        conflict_indicators.append("trade_off_conflict")

            # Check for authority conflicts
            if (self.making_authority and other.making_authority and
                self.making_authority != other.making_authority):
                conflict_indicators.append("authority_conflict")

            if conflict_indicators:
                conflicts.append({
                    'conflicting_judgment': other.id,
                    'conflict_types': conflict_indicators,
                    'severity': len(conflict_indicators) / 3.0  # Normalize by max possible conflicts
                })

        return conflicts

@dataclass
class NormativeSystem(Node):
    """Collection of normative rules forming a coherent framework."""

    framework_type: NormativeFramework
    rules: Dict[uuid.UUID, NormativeRule] = field(default_factory=dict)
    value_judgments: Dict[uuid.UUID, ValueJudgment] = field(default_factory=dict)

    # System properties
    internal_consistency: Optional[float] = None  # How consistent rules are (0-1)
    completeness_score: Optional[float] = None    # How complete the system is (0-1)
    coherence_level: Optional[float] = None       # Overall system coherence (0-1)

    # System relationships
    hierarchical_structure: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)  # Parent -> Children
    rule_priorities: Dict[uuid.UUID, float] = field(default_factory=dict)  # Rule priority scores

    # Institutional context
    governing_institutions: List[uuid.UUID] = field(default_factory=list)
    enforcement_mechanisms: Dict[EnforcementType, List[uuid.UUID]] = field(default_factory=dict)

    # System evolution
    historical_versions: List[datetime] = field(default_factory=list)
    change_frequency: Optional[float] = None  # How often system changes
    adaptation_capacity: Optional[float] = None  # Ability to adapt (0-1)

    def add_rule(self, rule: NormativeRule, priority: Optional[float] = None) -> None:
        """Add a rule to the normative system."""
        self.rules[rule.id] = rule

        if priority is not None:
            self.rule_priorities[rule.id] = priority

        # Update system consistency
        self._update_consistency_metrics()

    def add_value_judgment(self, judgment: ValueJudgment) -> None:
        """Add a value judgment to the system."""
        self.value_judgments[judgment.id] = judgment

    def detect_rule_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts between rules in the system."""
        conflicts = []

        rules_list = list(self.rules.values())

        for i, rule1 in enumerate(rules_list):
            for rule2 in rules_list[i+1:]:
                conflict_types = self._analyze_rule_conflict(rule1, rule2)

                if conflict_types:
                    conflicts.append({
                        'rule1_id': rule1.id,
                        'rule2_id': rule2.id,
                        'conflict_types': conflict_types,
                        'severity': self._calculate_conflict_severity(rule1, rule2, conflict_types)
                    })

        return conflicts

    def _analyze_rule_conflict(
        self,
        rule1: NormativeRule,
        rule2: NormativeRule) -> List[NormativeConflictType]:
        """Analyze potential conflicts between two rules."""
        conflicts = []

        # Check if rules apply to same subject and object
        if rule1.subject == rule2.subject and rule1.object == rule2.object:

            # Check for permission-prohibition conflicts
            if ((rule1.deontic_category == DeontologicalCategory.PERMISSION and
                 rule2.deontic_category == DeontologicalCategory.PROHIBITION) or
                (rule1.deontic_category == DeontologicalCategory.PROHIBITION and
                 rule2.deontic_category == DeontologicalCategory.PERMISSION)):
                conflicts.append(NormativeConflictType.PERMISSION_PROHIBITION)

            # Check for obligation-prohibition conflicts
            elif ((rule1.deontic_category == DeontologicalCategory.OBLIGATION and
                   rule2.deontic_category == DeontologicalCategory.PROHIBITION) or
                  (rule1.deontic_category == DeontologicalCategory.PROHIBITION and
                   rule2.deontic_category == DeontologicalCategory.OBLIGATION)):
                conflicts.append(NormativeConflictType.OBLIGATION_PROHIBITION)

            # Check for competing obligations
            elif (rule1.deontic_category == DeontologicalCategory.OBLIGATION and
                  rule2.deontic_category == DeontologicalCategory.OBLIGATION):
                # If different conditions or authorities, might be competing
                if (rule1.issuing_authority != rule2.issuing_authority or
                    rule1.conditions != rule2.conditions):
                    conflicts.append(NormativeConflictType.COMPETING_OBLIGATIONS)

        # Check for authority conflicts
        if (rule1.issuing_authority != rule2.issuing_authority and
            rule1.authority_scope == rule2.authority_scope):
            conflicts.append(NormativeConflictType.AUTHORITY_CONFLICT)

        return conflicts

    def _calculate_conflict_severity(self, rule1: NormativeRule, rule2: NormativeRule,
                                   conflict_types: List[NormativeConflictType]) -> float:
        """Calculate severity of conflict between two rules."""
        base_severity = len(conflict_types) * 0.3

        # Adjust based on rule strengths
        strength_values = {
            RuleStrength.ABSOLUTE: 1.0,
            RuleStrength.STRONG: 0.8,
            RuleStrength.MODERATE: 0.6,
            RuleStrength.WEAK: 0.4,
            RuleStrength.ADVISORY: 0.2
        }

        rule1_strength = strength_values[rule1.rule_strength]
        rule2_strength = strength_values[rule2.rule_strength]

        # Higher severity if both rules are strong
        strength_factor = (rule1_strength + rule2_strength) / 2.0

        return min(base_severity * (1 + strength_factor), 1.0)

    def resolve_conflicts(
        self,
        conflicts: Optional[List[Dict[str,
        Any]]] = None) -> List[Dict[str, Any]]:
        """Attempt to resolve conflicts between rules."""
        if conflicts is None:
            conflicts = self.detect_rule_conflicts()

        resolutions = []

        for conflict in conflicts:
            rule1_id = conflict['rule1_id']
            rule2_id = conflict['rule2_id']

            rule1 = self.rules.get(rule1_id)
            rule2 = self.rules.get(rule2_id)

            if not rule1 or not rule2:
                continue

            resolution_strategy = self._determine_resolution_strategy(rule1, rule2, conflict)

            if resolution_strategy:
                resolutions.append({
                    'conflict': conflict,
                    'resolution_strategy': resolution_strategy,
                    'resolved': self._apply_resolution(rule1, rule2, resolution_strategy)
                })

        return resolutions

    def _determine_resolution_strategy(self, rule1: NormativeRule, rule2: NormativeRule,
                                     conflict: Dict[str, Any]) -> Optional[str]:
        """Determine strategy for resolving a conflict."""
        # Priority-based resolution
        priority1 = self.rule_priorities.get(rule1.id, 0.5)
        priority2 = self.rule_priorities.get(rule2.id, 0.5)

        if abs(priority1 - priority2) > 0.2:
            return "priority_override"

        # Strength-based resolution
        strength_values = {
            RuleStrength.ABSOLUTE: 5,
            RuleStrength.STRONG: 4,
            RuleStrength.MODERATE: 3,
            RuleStrength.WEAK: 2,
            RuleStrength.ADVISORY: 1
        }

        if strength_values[rule1.rule_strength] != strength_values[rule2.rule_strength]:
            return "strength_hierarchy"

        # Scope-based resolution
        if rule1.authority_scope != rule2.authority_scope:
            return "scope_differentiation"

        # Temporal resolution (if conditions differ)
        if rule1.conditions != rule2.conditions:
            return "conditional_differentiation"

        return "requires_manual_resolution"

    def _apply_resolution(self, rule1: NormativeRule, rule2: NormativeRule,
                         strategy: str) -> bool:
        """Apply resolution strategy to conflicting rules."""
        if strategy == "priority_override":
            priority1 = self.rule_priorities.get(rule1.id, 0.5)
            priority2 = self.rule_priorities.get(rule2.id, 0.5)

            if priority1 > priority2:
                rule2.conflicting_rules.append(rule1.id)
            else:
                rule1.conflicting_rules.append(rule2.id)

            return True

        elif strategy == "strength_hierarchy":
            # Mark the weaker rule as subordinate
            strength_values = {
                RuleStrength.ABSOLUTE: 5,
                RuleStrength.STRONG: 4,
                RuleStrength.MODERATE: 3,
                RuleStrength.WEAK: 2,
                RuleStrength.ADVISORY: 1
            }

            if strength_values[rule1.rule_strength] > strength_values[rule2.rule_strength]:
                rule2.prerequisite_rules.append(rule1.id)
            else:
                rule1.prerequisite_rules.append(rule2.id)

            return True

        # Other strategies would be implemented here
        return False

    def _update_consistency_metrics(self) -> None:
        """Update internal consistency metrics for the system."""
        if not self.rules:
            self.internal_consistency = 1.0
            return

        conflicts = self.detect_rule_conflicts()
        total_possible_conflicts = len(self.rules) * (len(self.rules) - 1) / 2

        if total_possible_conflicts == 0:
            self.internal_consistency = 1.0
        else:
            conflict_ratio = len(conflicts) / total_possible_conflicts
            self.internal_consistency = max(0.0, 1.0 - conflict_ratio)

    def evaluate_system_coherence(self) -> Dict[str, float]:
        """Evaluate overall coherence of the normative system."""
        coherence_metrics = {}

        # Internal consistency
        if self.internal_consistency is not None:
            coherence_metrics['consistency'] = self.internal_consistency

        # Rule coverage (simplified)
        deontic_categories = set(rule.deontic_category for rule in self.rules.values())
        coverage_score = len(deontic_categories) / len(DeontologicalCategory)
        coherence_metrics['coverage'] = coverage_score

        # Authority consistency
        authorities = set(rule.issuing_authority for rule in self.rules.values()
                         if rule.issuing_authority is not None)
        authority_diversity = len(authorities) / max(len(self.rules), 1)
        coherence_metrics['authority_consistency'] = max(0.0, 1.0 - authority_diversity)

        # Overall coherence
        overall_coherence = sum(coherence_metrics.values()) / len(coherence_metrics)
        coherence_metrics['overall'] = overall_coherence
        self.coherence_level = overall_coherence

        return coherence_metrics

    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of the normative system."""
        conflicts = self.detect_rule_conflicts()
        coherence = self.evaluate_system_coherence()

        # Value judgment analysis
        value_conflicts = []
        judgments_list = list(self.value_judgments.values())
        for judgment in judgments_list:
            judgment_conflicts = judgment.identify_value_conflicts(judgments_list)
            value_conflicts.extend(judgment_conflicts)

        return {
            'system_overview': {
                'framework_type': self.framework_type.name,
                'total_rules': len(self.rules),
                'total_value_judgments': len(self.value_judgments),
                'last_updated': self.modified_at or self.created_at
            },
            'rule_analysis': {
                'deontic_distribution': self._analyze_deontic_distribution(),
                'authority_distribution': self._analyze_authority_distribution(),
                'strength_distribution': self._analyze_strength_distribution()
            },
            'conflict_analysis': {
                'total_conflicts': len(conflicts),
                'conflict_types': self._summarize_conflict_types(conflicts),
                'average_severity': sum(c['severity'] for c in conflicts) / len(conflicts) if conflicts else 0.0
            },
            'coherence_analysis': coherence,
            'value_judgment_analysis': {
                'total_judgments': len(self.value_judgments),
                'value_conflicts': len(value_conflicts),
                'average_quality': self._calculate_average_judgment_quality()
            }
        }

    def _analyze_deontic_distribution(self) -> Dict[str, int]:
        """Analyze distribution of deontic categories."""
        distribution = {}
        for rule in self.rules.values():
            category = rule.deontic_category.name
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def _analyze_authority_distribution(self) -> Dict[str, int]:
        """Analyze distribution of issuing authorities."""
        distribution = {}
        for rule in self.rules.values():
            authority = str(rule.issuing_authority) if rule.issuing_authority else "unknown"
            distribution[authority] = distribution.get(authority, 0) + 1
        return distribution

    def _analyze_strength_distribution(self) -> Dict[str, int]:
        """Analyze distribution of rule strengths."""
        distribution = {}
        for rule in self.rules.values():
            strength = rule.rule_strength.name
            distribution[strength] = distribution.get(strength, 0) + 1
        return distribution

    def _summarize_conflict_types(self, conflicts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize types of conflicts found."""
        conflict_summary = {}
        for conflict in conflicts:
            for conflict_type in conflict['conflict_types']:
                type_name = conflict_type.name
                conflict_summary[type_name] = conflict_summary.get(type_name, 0) + 1
        return conflict_summary

    def _calculate_average_judgment_quality(self) -> float:
        """Calculate average quality of value judgments."""
        if not self.value_judgments:
            return 0.0

        quality_scores = []
        for judgment in self.value_judgments.values():
            quality_assessment = judgment.assess_judgment_quality()
            quality_scores.append(quality_assessment.get('overall', 0.0))

        return sum(quality_scores) / len(quality_scores)

@dataclass
class DeontologicalAnalyzer:
    """Tools for analyzing normative rule systems and deontic logic."""

    normative_system: NormativeSystem

    def analyze_deontic_completeness(self, domain: str) -> Dict[str, Any]:
        """Analyze completeness of deontic coverage for a specific domain."""
        domain_rules = [rule for rule in self.normative_system.rules.values()
                       if domain.lower() in rule.object.lower()]

        if not domain_rules:
            return {'completeness': 0.0, 'gaps': ['No rules found for domain']}

        # Check coverage of basic deontic categories
        covered_categories = set(rule.deontic_category for rule in domain_rules)
        essential_categories = {
            DeontologicalCategory.PERMISSION,
            DeontologicalCategory.OBLIGATION,
            DeontologicalCategory.PROHIBITION
        }

        coverage = len(covered_categories.intersection(essential_categories)) / len(essential_categories)

        gaps = []
        missing_categories = essential_categories - covered_categories
        for category in missing_categories:
            gaps.append(f"Missing {category.name.lower()} rules")

        return {
            'domain': domain,
            'completeness': coverage,
            'covered_categories': [cat.name for cat in covered_categories],
            'gaps': gaps,
            'rule_count': len(domain_rules)
        }

    def trace_rule_derivations(self, base_rule_id: uuid.UUID) -> Dict[str, Any]:
        """Trace derivations from a base rule."""
        if base_rule_id not in self.normative_system.rules:
            return {'error': 'Base rule not found'}

        base_rule = self.normative_system.rules[base_rule_id]

        # Build derivation tree
        derivation_tree = {
            'base_rule': {
                'id': str(base_rule.id),
                'description': base_rule.description,
                'deontic_category': base_rule.deontic_category.name
            },
            'direct_derivatives': [],
            'all_derivatives': set(),
            'derivation_depth': 0
        }

        # Find direct derivatives
        for rule_id in base_rule.derivative_rules:
            if rule_id in self.normative_system.rules:
                derivative = self.normative_system.rules[rule_id]
                derivation_tree['direct_derivatives'].append({
                    'id': str(derivative.id),
                    'description': derivative.description,
                    'deontic_category': derivative.deontic_category.name
                })
                derivation_tree['all_derivatives'].add(rule_id)

        # Trace deeper derivatives (simplified - avoid infinite recursion)
        max_depth = 5
        current_level = base_rule.derivative_rules.copy()
        depth = 1

        while current_level and depth < max_depth:
            next_level = set()
            for rule_id in current_level:
                if rule_id in self.normative_system.rules:
                    rule = self.normative_system.rules[rule_id]
                    for derivative_id in rule.derivative_rules:
                        if derivative_id not in derivation_tree['all_derivatives']:
                            next_level.add(derivative_id)
                            derivation_tree['all_derivatives'].add(derivative_id)

            current_level = next_level
            depth += 1

        derivation_tree['derivation_depth'] = depth - 1
        derivation_tree['total_derivatives'] = len(derivation_tree['all_derivatives'])

        return derivation_tree

    def analyze_enforcement_gaps(self) -> List[Dict[str, Any]]:
        """Analyze gaps in enforcement mechanisms."""
        gaps = []

        for rule in self.normative_system.rules.values():
            gap_indicators = []

            # Check for enforcement mechanism
            if rule.enforcement_type == EnforcementType.SOCIAL and not rule.violation_penalty:
                gap_indicators.append("No penalty specified for social enforcement")

            # Check compliance rate
            if rule.compliance_rate is not None and rule.compliance_rate < 0.5:
                gap_indicators.append(f"Low compliance rate: {rule.compliance_rate:.2f}")

            # Check enforcement effectiveness
            if rule.enforcement_effectiveness is not None and rule.enforcement_effectiveness < 0.4:
                gap_indicators.append(f"Low enforcement effectiveness: {rule.enforcement_effectiveness:.2f}")

            # Check for conflicting rules without resolution
            if rule.conflicting_rules and not rule.prerequisite_rules:
                gap_indicators.append("Unresolved rule conflicts")

            if gap_indicators:
                gaps.append({
                    'rule_id': rule.id,
                    'rule_description': rule.description,
                    'gaps': gap_indicators,
                    'severity': len(gap_indicators) / 4.0  # Normalize by max possible gaps
                })

        return sorted(gaps, key=lambda x: x['severity'], reverse=True)
