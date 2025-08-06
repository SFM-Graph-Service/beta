"""
System Boundary and Problem Definition for Social Fabric Matrix analysis.

This module implements Hayden's systematic approach to problem definition and
system boundary identification, which is the foundational step in SFM analysis.
It provides structured tools for defining problems, setting analysis boundaries,
and establishing the scope for institutional analysis.

Key Components:
- SystemBoundary: Defines boundaries for SFM analysis
- ProblemDefinition: Structures problem identification and scoping
- AnalysisScope: Manages temporal, spatial, and institutional scope
- BoundaryValidator: Validates boundary completeness and consistency
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.sfm_enums import (
    SystemLevel,
    BoundaryType,
    SystemBoundaryType,
    ProblemSolvingStage,
    InstitutionalScope,
    ValueCategory,
)

class ProblemType(Enum):
    """Types of problems in SFM analysis."""

    INSTITUTIONAL_COORDINATION = auto()  # Coordination between institutions
    DELIVERY_EFFECTIVENESS = auto()      # Delivery system problems
    VALUE_CONFLICT = auto()              # Conflicting values and priorities
    RESOURCE_ALLOCATION = auto()         # Resource distribution issues
    GOVERNANCE_FAILURE = auto()          # Governance and accountability problems
    SUSTAINABILITY_CHALLENGE = auto()    # Environmental/social sustainability
    INNOVATION_ADOPTION = auto()         # Technology and innovation barriers
    SOCIAL_EQUITY = auto()              # Equity and fairness issues

class ProblemUrgency(Enum):
    """Urgency levels for problem addressing."""

    CRITICAL = auto()      # Immediate attention required
    HIGH = auto()          # Important, needs addressing soon
    MODERATE = auto()      # Standard priority
    LOW = auto()           # Can be deferred
    MONITORING = auto()    # Watch for changes

class ScopeComplexity(Enum):
    """Complexity levels for analysis scope."""

    SIMPLE = auto()        # Single institution, clear boundaries
    MODERATE = auto()      # Few institutions, some interactions
    COMPLEX = auto()       # Multiple institutions, many interactions
    HIGHLY_COMPLEX = auto() # Many institutions, complex interactions

@dataclass
class SystemBoundary(Node):
    """Defines the boundaries for SFM analysis following Hayden's methodology."""

    boundary_type: SystemBoundaryType = SystemBoundaryType.CONCEPTUAL_BOUNDARY
    boundary_purpose: Optional[str] = None

    # Institutional boundaries
    included_institutions: List[uuid.UUID] = field(default_factory=list)
    excluded_institutions: List[uuid.UUID] = field(default_factory=list)
    boundary_institutions: List[uuid.UUID] = field(default_factory=list)  # On the boundary

    # Temporal boundaries
    analysis_time_frame: Optional[TimeSlice] = None
    historical_context_period: Optional[TimeSlice] = None
    future_projection_period: Optional[TimeSlice] = None

    # Spatial boundaries
    geographic_scope: Optional[SpatialUnit] = None
    spatial_hierarchy_levels: List[str] = field(default_factory=list)
    cross_boundary_flows: List[uuid.UUID] = field(default_factory=list)

    # System level boundaries
    system_levels: List[SystemLevel] = field(default_factory=list)
    hierarchical_relationships: Dict[SystemLevel, List[uuid.UUID]] = field(default_factory=dict)

    # Value boundaries
    included_value_categories: List[ValueCategory] = field(default_factory=list)
    value_measurement_scope: Dict[ValueCategory, str] = field(default_factory=dict)

    # Resource boundaries
    resource_types_in_scope: List[str] = field(default_factory=list)
    resource_flows_tracked: List[uuid.UUID] = field(default_factory=list)

    # Analytical boundaries
    analysis_methods_scope: List[str] = field(default_factory=list)
    data_collection_boundaries: Dict[str, str] = field(default_factory=dict)

    # Boundary validation
    boundary_completeness: Optional[float] = None  # 0-1 scale
    boundary_consistency: Optional[float] = None   # 0-1 scale
    stakeholder_boundary_agreement: Optional[float] = None  # 0-1 scale

    # Boundary management
    boundary_rationale: Dict[str, str] = field(default_factory=dict)
    boundary_assumptions: List[str] = field(default_factory=list)
    boundary_limitations: List[str] = field(default_factory=list)

    # Dynamic boundary considerations
    boundary_sensitivity: Dict[str, float] = field(default_factory=dict)
    boundary_evolution: List[Dict[str, Any]] = field(default_factory=list)

    def validate_boundary_completeness(self) -> Dict[str, Any]:
        """Validate completeness of system boundary definition."""
        validation_results = {
            'institutional_completeness': 0.0,
            'temporal_completeness': 0.0,
            'spatial_completeness': 0.0,
            'value_completeness': 0.0,
            'overall_completeness': 0.0,
            'missing_components': [],
            'completeness_recommendations': []
        }

        # Institutional completeness
        if self.included_institutions:
            institutional_score = 1.0
            if not self.boundary_institutions:
                validation_results['missing_components'].append('boundary_institutions')
                institutional_score *= 0.8
            if not self.excluded_institutions:
                validation_results['missing_components'].append('excluded_institutions')
                institutional_score *= 0.9
        else:
            institutional_score = 0.0
            validation_results['missing_components'].append('included_institutions')

        validation_results['institutional_completeness'] = institutional_score

        # Temporal completeness
        temporal_score = 0.0
        if self.analysis_time_frame:
            temporal_score += 0.7
        if self.historical_context_period:
            temporal_score += 0.2
        if self.future_projection_period:
            temporal_score += 0.1

        validation_results['temporal_completeness'] = min(temporal_score, 1.0)

        # Spatial completeness
        spatial_score = 0.0
        if self.geographic_scope:
            spatial_score += 0.6
        if self.spatial_hierarchy_levels:
            spatial_score += 0.4

        validation_results['spatial_completeness'] = min(spatial_score, 1.0)

        # Value completeness
        value_score = 0.0
        if self.included_value_categories:
            value_score += 0.7
        if self.value_measurement_scope:
            value_score += 0.3

        validation_results['value_completeness'] = min(value_score, 1.0)

        # Overall completeness
        completeness_scores = [
            validation_results['institutional_completeness'],
            validation_results['temporal_completeness'],
            validation_results['spatial_completeness'],
            validation_results['value_completeness']
        ]
        validation_results['overall_completeness'] = sum(completeness_scores) / len(completeness_scores)
        self.boundary_completeness = validation_results['overall_completeness']

        # Generate recommendations
        if validation_results['overall_completeness'] < 0.7:
            validation_results['completeness_recommendations'].append(
                'Boundary definition needs strengthening in multiple areas'
            )

        if validation_results['institutional_completeness'] < 0.8:
            validation_results['completeness_recommendations'].append(
                'Clarify institutional boundaries and exclusions'
            )

        return validation_results

    def assess_boundary_sensitivity(self) -> Dict[str, float]:
        """Assess sensitivity of analysis to boundary choices."""
        sensitivity_assessment = {}

        # Institutional boundary sensitivity
        boundary_institutions_count = len(self.boundary_institutions)
        if boundary_institutions_count > 0:
            institutional_sensitivity = min(boundary_institutions_count / 5.0, 1.0)
            sensitivity_assessment['institutional_sensitivity'] = institutional_sensitivity

        # Temporal boundary sensitivity
        if self.analysis_time_frame and self.analysis_time_frame.duration:
            # Longer time frames may be more sensitive to temporal boundaries
            duration_years = self.analysis_time_frame.duration.days / 365.25
            temporal_sensitivity = min(duration_years / 10.0, 1.0)
            sensitivity_assessment['temporal_sensitivity'] = temporal_sensitivity

        # Cross-boundary flow sensitivity
        if self.cross_boundary_flows:
            flow_sensitivity = min(len(self.cross_boundary_flows) / 10.0, 1.0)
            sensitivity_assessment['flow_sensitivity'] = flow_sensitivity

        # Overall sensitivity
        if sensitivity_assessment:
            overall_sensitivity = sum(sensitivity_assessment.values()) / len(sensitivity_assessment)
            sensitivity_assessment['overall_sensitivity'] = overall_sensitivity
            self.boundary_sensitivity['overall'] = overall_sensitivity

        return sensitivity_assessment

@dataclass
class ProblemDefinition(Node):
    """Structured problem definition for SFM analysis."""

    problem_type: ProblemType = ProblemType.INSTITUTIONAL_COORDINATION
    problem_urgency: ProblemUrgency = ProblemUrgency.MODERATE

    # Problem characterization
    problem_statement: Optional[str] = None
    problem_context: Optional[str] = None
    problem_stakeholders: List[uuid.UUID] = field(default_factory=list)

    # Problem scope
    affected_institutions: List[uuid.UUID] = field(default_factory=list)
    affected_value_categories: List[ValueCategory] = field(default_factory=list)
    problem_geographic_scope: Optional[SpatialUnit] = None
    problem_time_horizon: Optional[TimeSlice] = None

    # Problem analysis
    root_causes: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    problem_symptoms: List[str] = field(default_factory=list)

    # Problem relationships
    related_problems: List[uuid.UUID] = field(default_factory=list)
    problem_dependencies: List[uuid.UUID] = field(default_factory=list)
    upstream_problems: List[uuid.UUID] = field(default_factory=list)
    downstream_impacts: List[uuid.UUID] = field(default_factory=list)

    # Problem assessment
    problem_severity: Optional[float] = None  # 0-1 scale
    problem_complexity: Optional[ScopeComplexity] = None
    solution_feasibility: Optional[float] = None  # 0-1 scale

    # Stakeholder perspectives
    stakeholder_problem_views: Dict[uuid.UUID, str] = field(default_factory=dict)
    stakeholder_priorities: Dict[uuid.UUID, float] = field(default_factory=dict)
    consensus_level: Optional[float] = None  # Agreement on problem definition

    # Problem evolution
    problem_history: List[Dict[str, Any]] = field(default_factory=list)
    problem_trends: List[str] = field(default_factory=list)
    future_problem_projections: List[str] = field(default_factory=list)

    # SFM integration
    matrix_problem_cells: List[uuid.UUID] = field(default_factory=list)
    delivery_system_problems: List[uuid.UUID] = field(default_factory=list)
    institutional_problem_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)

    def assess_problem_complexity(self) -> Dict[str, Any]:
        """Assess complexity of the problem for analysis planning."""
        complexity_assessment = {
            'stakeholder_complexity': 0.0,
            'institutional_complexity': 0.0,
            'value_complexity': 0.0,
            'temporal_complexity': 0.0,
            'causal_complexity': 0.0,
            'overall_complexity': ScopeComplexity.SIMPLE,
            'complexity_factors': []
        }

        # Stakeholder complexity
        stakeholder_count = len(self.problem_stakeholders)
        if stakeholder_count <= 3:
            complexity_assessment['stakeholder_complexity'] = 0.2
        elif stakeholder_count <= 7:
            complexity_assessment['stakeholder_complexity'] = 0.5
        elif stakeholder_count <= 15:
            complexity_assessment['stakeholder_complexity'] = 0.8
        else:
            complexity_assessment['stakeholder_complexity'] = 1.0

        if stakeholder_count > 10:
            complexity_assessment['complexity_factors'].append('High stakeholder count')

        # Institutional complexity
        institution_count = len(self.affected_institutions)
        if institution_count <= 2:
            complexity_assessment['institutional_complexity'] = 0.3
        elif institution_count <= 5:
            complexity_assessment['institutional_complexity'] = 0.6
        else:
            complexity_assessment['institutional_complexity'] = 1.0
            complexity_assessment['complexity_factors'].append('Multiple institutions involved')

        # Value complexity
        value_count = len(self.affected_value_categories)
        if value_count <= 2:
            complexity_assessment['value_complexity'] = 0.3
        elif value_count <= 4:
            complexity_assessment['value_complexity'] = 0.6
        else:
            complexity_assessment['value_complexity'] = 1.0
            complexity_assessment['complexity_factors'].append('Multiple value dimensions')

        # Causal complexity
        cause_count = len(self.root_causes) + len(self.contributing_factors)
        if cause_count <= 3:
            complexity_assessment['causal_complexity'] = 0.2
        elif cause_count <= 6:
            complexity_assessment['causal_complexity'] = 0.5
        else:
            complexity_assessment['causal_complexity'] = 1.0
            complexity_assessment['complexity_factors'].append('Complex causal structure')

        # Overall complexity classification
        complexity_scores = [
            complexity_assessment['stakeholder_complexity'],
            complexity_assessment['institutional_complexity'],
            complexity_assessment['value_complexity'],
            complexity_assessment['causal_complexity']
        ]
        average_complexity = sum(complexity_scores) / len(complexity_scores)

        if average_complexity <= 0.3:
            complexity_assessment['overall_complexity'] = ScopeComplexity.SIMPLE
        elif average_complexity <= 0.6:
            complexity_assessment['overall_complexity'] = ScopeComplexity.MODERATE
        elif average_complexity <= 0.8:
            complexity_assessment['overall_complexity'] = ScopeComplexity.COMPLEX
        else:
            complexity_assessment['overall_complexity'] = ScopeComplexity.HIGHLY_COMPLEX

        self.problem_complexity = complexity_assessment['overall_complexity']

        return complexity_assessment

    def generate_analysis_requirements(self) -> Dict[str, List[str]]:
        """Generate analysis requirements based on problem definition."""
        requirements = {
            'data_collection_needs': [],
            'analytical_methods': [],
            'stakeholder_engagement': [],
            'institutional_analysis': [],
            'value_analysis': []
        }

        # Data collection based on problem type
        if self.problem_type == ProblemType.DELIVERY_EFFECTIVENESS:
            requirements['data_collection_needs'].extend([
                'Delivery performance metrics',
                'Service quality indicators',
                'Beneficiary satisfaction data'
            ])
        elif self.problem_type == ProblemType.VALUE_CONFLICT:
            requirements['data_collection_needs'].extend([
                'Stakeholder value preferences',
                'Value trade-off data',
                'Conflict incident records'
            ])

        # Analytical methods based on complexity
        if self.problem_complexity in [ScopeComplexity.COMPLEX, ScopeComplexity.HIGHLY_COMPLEX]:
            requirements['analytical_methods'].extend([
                'Multi-criteria decision analysis',
                'Stakeholder network analysis',
                'Systems modeling'
            ])

        # Stakeholder engagement based on stakeholder count
        if len(self.problem_stakeholders) > 5:
            requirements['stakeholder_engagement'].extend([
                'Multi-stakeholder workshops',
                'Structured consultation process',
                'Consensus building activities'
            ])

        return requirements

@dataclass
class AnalysisScope(Node):
    """Manages the scope for SFM analysis."""

    scope_complexity: ScopeComplexity = ScopeComplexity.MODERATE

    # System boundary reference
    system_boundary_id: Optional[uuid.UUID] = None
    problem_definition_id: Optional[uuid.UUID] = None

    # Scope dimensions
    institutional_scope: List[InstitutionalScope] = field(default_factory=list)
    temporal_scope_detail: Dict[str, TimeSlice] = field(default_factory=dict)
    spatial_scope_hierarchy: Dict[str, SpatialUnit] = field(default_factory=dict)

    # Analysis depth
    analysis_depth_levels: Dict[str, str] = field(default_factory=dict)  # Deep, moderate, surface
    focus_areas: List[str] = field(default_factory=list)
    analytical_priorities: Dict[str, float] = field(default_factory=dict)

    # Resource constraints
    analysis_resource_limits: Dict[str, Any] = field(default_factory=dict)
    time_constraints: Optional[timedelta] = None
    data_availability_constraints: List[str] = field(default_factory=list)

    # Scope validation
    scope_feasibility: Optional[float] = None  # 0-1 scale
    scope_adequacy: Optional[float] = None     # 0-1 scale
    scope_stakeholder_acceptance: Optional[float] = None  # 0-1 scale

    # Scope management
    scope_change_history: List[Dict[str, Any]] = field(default_factory=list)
    scope_assumptions: List[str] = field(default_factory=list)
    scope_risks: Dict[str, str] = field(default_factory=dict)

    def validate_scope_adequacy(self) -> Dict[str, Any]:
        """Validate adequacy of analysis scope for problem addressing."""
        validation_results = {
            'scope_coverage': 0.0,
            'resource_alignment': 0.0,
            'stakeholder_coverage': 0.0,
            'temporal_adequacy': 0.0,
            'overall_adequacy': 0.0,
            'adequacy_issues': [],
            'improvement_recommendations': []
        }

        # Scope coverage assessment
        coverage_factors = []
        if self.institutional_scope:
            coverage_factors.append(0.3)
        if self.focus_areas:
            coverage_factors.append(0.3)
        if self.analytical_priorities:
            coverage_factors.append(0.4)

        validation_results['scope_coverage'] = sum(coverage_factors)

        # Resource alignment
        if self.analysis_resource_limits and self.time_constraints:
            validation_results['resource_alignment'] = 0.8
        elif self.analysis_resource_limits or self.time_constraints:
            validation_results['resource_alignment'] = 0.5
        else:
            validation_results['resource_alignment'] = 0.2
            validation_results['adequacy_issues'].append('Resource constraints not defined')

        # Overall adequacy
        adequacy_scores = [
            validation_results['scope_coverage'],
            validation_results['resource_alignment']
        ]
        validation_results['overall_adequacy'] = sum(adequacy_scores) / len(adequacy_scores)
        self.scope_adequacy = validation_results['overall_adequacy']

        # Generate recommendations
        if validation_results['overall_adequacy'] < 0.6:
            validation_results['improvement_recommendations'].append(
                'Strengthen scope definition in multiple areas'
            )

        return validation_results

@dataclass
class BoundaryValidator(Node):
    """Validates system boundaries and scope definitions."""

    validation_criteria: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)

    def validate_system_boundary(self, boundary: SystemBoundary) -> Dict[str, Any]:
        """Comprehensive validation of system boundary definition."""
        validation_results = {
            'validation_passed': False,
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'feasibility_score': 0.0,
            'overall_score': 0.0,
            'validation_issues': [],
            'recommendations': []
        }

        # Completeness validation
        completeness_results = boundary.validate_boundary_completeness()
        validation_results['completeness_score'] = completeness_results['overall_completeness']

        # Consistency validation
        consistency_score = self._validate_boundary_consistency(boundary)
        validation_results['consistency_score'] = consistency_score

        # Feasibility validation
        feasibility_score = self._validate_boundary_feasibility(boundary)
        validation_results['feasibility_score'] = feasibility_score

        # Overall validation score
        scores = [
            validation_results['completeness_score'],
            validation_results['consistency_score'],
            validation_results['feasibility_score']
        ]
        validation_results['overall_score'] = sum(scores) / len(scores)

        # Pass/fail determination
        validation_results['validation_passed'] = validation_results['overall_score'] >= 0.7

        # Store results
        self.validation_results[str(boundary.id)] = validation_results

        return validation_results

    def _validate_boundary_consistency(self, boundary: SystemBoundary) -> float:
        """Validate internal consistency of boundary definition."""
        consistency_checks = []

        # Check institutional consistency
        included_set = set(boundary.included_institutions)
        excluded_set = set(boundary.excluded_institutions)
        boundary_set = set(boundary.boundary_institutions)

        # No overlap between included and excluded
        if not included_set.intersection(excluded_set):
            consistency_checks.append(1.0)
        else:
            consistency_checks.append(0.0)

        # Boundary institutions should not be in included/excluded
        if not boundary_set.intersection(included_set.union(excluded_set)):
            consistency_checks.append(1.0)
        else:
            consistency_checks.append(0.5)

        # Temporal consistency
        if boundary.analysis_time_frame and boundary.historical_context_period:
            if (boundary.historical_context_period.end_date and
                boundary.analysis_time_frame.start_date and
                boundary.historical_context_period.end_date <= boundary.analysis_time_frame.start_date):
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.5)

        return sum(consistency_checks) / len(consistency_checks) if consistency_checks else 0.0

    def _validate_boundary_feasibility(self, boundary: SystemBoundary) -> float:
        """Validate feasibility of boundary definition for analysis."""
        feasibility_factors = []

        # Institution count feasibility
        institution_count = len(boundary.included_institutions)
        if institution_count <= 20:
            feasibility_factors.append(1.0)
        elif institution_count <= 50:
            feasibility_factors.append(0.7)
        else:
            feasibility_factors.append(0.4)

        # Value category feasibility
        value_count = len(boundary.included_value_categories)
        if value_count <= 6:
            feasibility_factors.append(1.0)
        elif value_count <= 10:
            feasibility_factors.append(0.8)
        else:
            feasibility_factors.append(0.5)

        # Data collection feasibility
        if boundary.data_collection_boundaries:
            feasibility_factors.append(0.8)
        else:
            feasibility_factors.append(0.4)

        return sum(feasibility_factors) / len(feasibility_factors) if feasibility_factors else 0.0
