"""
Social indicators and statistical database framework for Social Fabric Matrix modeling.

This module implements Hayden's social indicators methodology for building
comprehensive databases of quantitative and qualitative measures that support
SFM analysis and policy evaluation.

Key Components:
- SocialIndicator: Individual social indicator with measurement capabilities
- IndicatorDatabase: Database system for managing indicator collections
- StatisticalAnalyzer: Tools for statistical analysis of indicator data
- IndicatorDashboard: Visualization and monitoring system
"""

# pylint: disable=too-many-instance-attributes,too-many-public-methods  # Complex indicator analysis requires many attributes

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import statistics
import math

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.metadata_models import ValidationRule
from models.sfm_enums import (
    IndicatorType,
    SocialFabricIndicatorType,
    ValueCategory,
    EvidenceQuality,
    MeasurementApproach,
)

class AggregationMethod(Enum):
    """Methods for aggregating indicator data."""

    MEAN = auto()
    MEDIAN = auto()
    SUM = auto()
    WEIGHTED_AVERAGE = auto()
    GEOMETRIC_MEAN = auto()
    HARMONIC_MEAN = auto()
    COMPOSITE_INDEX = auto()
    NORMALIZED_SUM = auto()

class TrendDirection(Enum):
    """Direction of trend in indicator values."""

    INCREASING = auto()
    DECREASING = auto()
    STABLE = auto()
    VOLATILE = auto()
    CYCLICAL = auto()
    UNKNOWN = auto()

@dataclass
class IndicatorMeasurement:
    """Single measurement of a social indicator."""

    value: Union[float, int, str, bool]
    timestamp: datetime
    time_slice: Optional[TimeSlice] = None
    spatial_unit: Optional[SpatialUnit] = None
    scenario: Optional[Scenario] = None

    # Measurement quality
    confidence_level: float = 1.0  # 0-1 scale
    measurement_error: Optional[float] = None
    data_source: Optional[str] = None
    collection_method: Optional[str] = None

    # Contextual information
    contextual_factors: Dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]
    measurement_conditions: Dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]

    # Metadata
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    notes: str = ""

    def get_numeric_value(self) -> Optional[float]:
        """Get numeric representation of the value."""
        if isinstance(self.value, bool):
            return 1.0 if self.value else 0.0
        elif isinstance(self.value, (int, float)):
            return float(self.value)
        elif isinstance(self.value, str):  # type: ignore[unnecessary-isinstance]
            try:
                return float(self.value)
            except ValueError:
                return None
        return None

    def is_valid(self) -> bool:
        """Check if measurement is valid."""
        return self.confidence_level > 0

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score for this measurement."""
        quality_factors = [self.confidence_level]

        if self.measurement_error is not None:
            # Lower error = higher quality
            error_quality = max(0.0, 1.0 - self.measurement_error)
            quality_factors.append(error_quality)

        if self.data_source:
            quality_factors.append(0.9)  # Bonus for having source

        if self.collection_method:
            quality_factors.append(0.8)  # Bonus for documented method

        return sum(quality_factors) / len(quality_factors)

@dataclass
class SocialIndicator(Node):
    """Social indicator for measuring aspects of the social fabric."""

    indicator_type: IndicatorType = IndicatorType.PERFORMANCE_INDICATOR
    sfm_indicator_type: Optional[SocialFabricIndicatorType] = None
    value_category: ValueCategory = ValueCategory.SOCIAL

    # Measurement specifications
    measurement_unit: str = ""
    measurement_approach: MeasurementApproach = MeasurementApproach.QUANTITATIVE
    measurement_frequency: Optional[timedelta] = None

    # Current data
    measurements: List[IndicatorMeasurement] = field(default_factory=list)
    current_value: Optional[Union[float, int, str, bool]] = None
    current_timestamp: Optional[datetime] = None

    # Targets and thresholds
    target_value: Optional[float] = None
    minimum_threshold: Optional[float] = None
    maximum_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None

    # Statistical properties
    baseline_value: Optional[float] = None
    trend_direction: TrendDirection = TrendDirection.UNKNOWN
    volatility_measure: Optional[float] = None
    seasonal_pattern: Optional[str] = None

    # SFM Matrix Integration (Enhanced)
    related_matrix_cells: List[uuid.UUID] = field(default_factory=list)
    matrix_cell_relationships: Dict[uuid.UUID, str] = field(default_factory=dict)  # Cell -> relationship type  # type: ignore[misc]
    cell_indicator_strength: Dict[uuid.UUID, float] = field(default_factory=dict)  # Cell -> strength (0-1)  # type: ignore[misc]
    matrix_measurement_dependencies: List[uuid.UUID] = field(default_factory=list)  # Dependent measurements  # type: ignore[misc]

    # Institutional Integration
    affecting_institutions: List[uuid.UUID] = field(default_factory=list)
    institutional_indicator_types: Dict[uuid.UUID, str] = field(default_factory=dict)  # Institution -> indicator type  # type: ignore[misc]
    delivery_system_indicators: Dict[uuid.UUID, str] = field(default_factory=dict)  # Delivery -> indicator role  # type: ignore[misc]

    # Policy Integration
    policy_relevance: List[uuid.UUID] = field(default_factory=list)  # Related policies
    policy_impact_measurement: Dict[uuid.UUID, float] = field(default_factory=dict)  # Policy -> impact score  # type: ignore[misc]
    policy_evaluation_role: Optional[str] = None  # Role in policy evaluation

    # Cross-Matrix Effects
    cross_matrix_influences: List[uuid.UUID] = field(default_factory=list)  # Other cells influenced  # type: ignore[misc]
    matrix_feedback_loops: List[uuid.UUID] = field(default_factory=list)  # Feedback loops  # type: ignore[misc]
    system_level_effects: List[str] = field(default_factory=list)  # System-wide effects  # type: ignore[misc]

    # Validation and quality
    validation_rules: List[ValidationRule] = field(default_factory=list)  # type: ignore[misc]
    evidence_quality: EvidenceQuality = EvidenceQuality.MEDIUM

    def add_measurement(self, measurement: IndicatorMeasurement) -> None:
        """Add a new measurement to the indicator."""
        self.measurements.append(measurement)
        self.measurements.sort(key=lambda m: m.timestamp)

        # Update current value
        if not self.current_timestamp or measurement.timestamp > self.current_timestamp:
            self.current_value = measurement.value
            self.current_timestamp = measurement.timestamp

    def get_measurements_in_period(
        self,
        start_date: datetime,
        end_date: datetime) -> List[IndicatorMeasurement]:
        """Get measurements within a specific time period."""
        return [m for m in self.measurements
                if start_date <= m.timestamp <= end_date]

    def calculate_trend(self, periods: int = 12) -> TrendDirection:
        """Calculate trend direction over recent periods."""
        if len(self.measurements) < 2:
            return TrendDirection.UNKNOWN

        recent_measurements = self.measurements[-periods:]
        numeric_values = [m.get_numeric_value() for m in recent_measurements
                         if m.get_numeric_value() is not None]

        if len(numeric_values) < 2:
            return TrendDirection.UNKNOWN

        # Simple linear trend calculation
        n = len(numeric_values)
        x_mean = (n - 1) / 2
        clean_numeric = [v for v in numeric_values if v is not None]
        y_mean = sum(clean_numeric) / len(clean_numeric) if clean_numeric else 0

        numerator = sum(
            (i - x_mean) * (v - y_mean) for i,
            v in enumerate(numeric_values) if v is not None)
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return TrendDirection.STABLE

        slope = numerator / denominator

        # Determine trend based on slope and significance
        slope_threshold = 0.01  # Adjust based on indicator scale

        if abs(slope) < slope_threshold:
            self.trend_direction = TrendDirection.STABLE
        elif slope > 0:
            self.trend_direction = TrendDirection.INCREASING
        else:
            self.trend_direction = TrendDirection.DECREASING

        return self.trend_direction

    def calculate_volatility(self, periods: int = 12) -> float:
        """Calculate volatility measure for the indicator."""
        if len(self.measurements) < 3:
            return 0.0

        recent_measurements = self.measurements[-periods:]
        numeric_values = [m.get_numeric_value() for m in recent_measurements
                         if m.get_numeric_value() is not None]

        if len(numeric_values) < 3:
            return 0.0

        # Calculate coefficient of variation
        clean_values = [v for v in numeric_values if v is not None]
        if not clean_values:
            return 0.0

        mean_value = statistics.mean(clean_values)
        if mean_value == 0:
            return 0.0

        std_dev = statistics.stdev(clean_values)
        volatility = abs(std_dev / mean_value)

        self.volatility_measure = volatility
        return volatility

    def get_current_status(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Get current status relative to thresholds and targets."""
        status = {
            'current_value': self.current_value,
            'target_achievement': None,
            'threshold_status': 'normal',
            'trend': self.trend_direction.name if self.trend_direction else 'unknown'
        }

        current_numeric = None
        if self.current_value is not None:
            try:
                current_numeric = float(self.current_value)
            except (ValueError, TypeError):
                return status

        if current_numeric is not None:
            # Check target achievement
            if self.target_value is not None:
                status['target_achievement'] = current_numeric / self.target_value

            # Check threshold status
            if self.critical_threshold is not None:
                if current_numeric <= self.critical_threshold:
                    status['threshold_status'] = 'critical'
            elif self.minimum_threshold is not None:
                if current_numeric <= self.minimum_threshold:
                    status['threshold_status'] = 'below_minimum'
            elif self.maximum_threshold is not None:
                if current_numeric >= self.maximum_threshold:
                    status['threshold_status'] = 'above_maximum'

        return status

    def validate_measurement(
        self,
        measurement: IndicatorMeasurement) -> List[str]:  # type: ignore[misc]
        """Validate a measurement against rules."""
        validation_errors = []

        for rule in self.validation_rules:
            # Simple validation - in practice, this would be more sophisticated
            if rule.rule_type.name == 'RANGE' and 'min' in rule.parameters:
                numeric_value = measurement.get_numeric_value()
                if numeric_value is not None:
                    if numeric_value < rule.parameters['min']:
                        validation_errors.append(f"Value {numeric_value} below minimum {rule.parameters['min']}")
                    if 'max' in rule.parameters and numeric_value > rule.parameters['max']:
                        validation_errors.append(f"Value {numeric_value} above maximum {rule.parameters['max']}")

        return validation_errors

    def assess_matrix_integration_strength(self) -> Dict[str, float]:  # type: ignore[misc]
        """Assess strength of integration with SFM matrix."""
        integration_assessment = {}

        # Matrix cell integration strength
        if self.related_matrix_cells:
            total_strength = sum(self.cell_indicator_strength.values())  # type: ignore[arg-type,misc]
            avg_strength = total_strength / len(self.related_matrix_cells) if self.related_matrix_cells else 0  # type: ignore[arg-type,misc]
            integration_assessment['cell_integration_strength'] = avg_strength

        # Institutional integration coverage
        if self.affecting_institutions:
            institutional_coverage = min(
                len(self.affecting_institutions) / 5.0,
                1.0)  # Normalize to 5 institutions
            integration_assessment['institutional_coverage'] = institutional_coverage

        # Policy integration coverage
        if self.policy_relevance:
            policy_coverage = min(len(self.policy_relevance) / 3.0, 1.0)  # Normalize to 3 policies
            integration_assessment['policy_coverage'] = policy_coverage

        # Cross-matrix effects
        if self.cross_matrix_influences:
            cross_effects_score = min(len(self.cross_matrix_influences) / 4.0, 1.0)
            integration_assessment['cross_matrix_effects'] = cross_effects_score

        # Feedback loop integration
        if self.matrix_feedback_loops:
            feedback_score = min(len(self.matrix_feedback_loops) / 2.0, 1.0)
            integration_assessment['feedback_integration'] = feedback_score

        # Overall matrix integration
        if integration_assessment:
            overall_integration = sum(integration_assessment.values()) / len(integration_assessment)  # type: ignore[arg-type]
            integration_assessment['overall_matrix_integration'] = overall_integration

        return integration_assessment

    def identify_matrix_measurement_gaps(self) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """Identify gaps in matrix measurement coverage."""
        gaps = []

        # Cells without sufficient measurement strength
        for cell_id in self.related_matrix_cells:
            strength = self.cell_indicator_strength.get(cell_id, 0.0)
            if strength < 0.5:
                gaps.append({
                    'type': 'weak_cell_measurement',
                    'cell_id': cell_id,
                    'current_strength': strength,
                    'description': f'Weak measurement relationship with matrix cell {cell_id}',
                    'priority': 'high' if strength < 0.3 else 'medium'
                })

        # Missing institutional coverage
        if len(self.affecting_institutions) < 2:
            gaps.append({
                'type': 'limited_institutional_coverage',
                'description': 'Indicator covers few institutional relationships',
                'current_coverage': len(self.affecting_institutions),
                'priority': 'medium'
            })

        # Missing policy relevance
        if not self.policy_relevance:
            gaps.append({
                'type': 'no_policy_relevance',
                'description': 'Indicator has no identified policy relevance',
                'priority': 'low'
            })

        # Missing feedback loop integration
        if not self.matrix_feedback_loops:
            gaps.append({
                'type': 'no_feedback_integration',
                'description': 'Indicator not integrated with matrix feedback loops',
                'priority': 'medium'
            })

        return gaps

    def calculate_policy_impact_score(self) -> Optional[float]:
        """Calculate overall policy impact score."""
        if not self.policy_impact_measurement:
            return None

        # Weighted average of policy impacts
        total_impact = sum(self.policy_impact_measurement.values())
        return total_impact / len(self.policy_impact_measurement)

    def assess_delivery_system_integration(self) -> Dict[str, float]:  # type: ignore[misc]
        """Assess integration with delivery systems."""
        delivery_assessment = {}

        if self.delivery_system_indicators:
            # Coverage assessment
            delivery_coverage = min(len(self.delivery_system_indicators) / 3.0, 1.0)
            delivery_assessment['delivery_coverage'] = delivery_coverage

            # Role diversity assessment
            unique_roles = set(self.delivery_system_indicators.values())
            role_diversity = min(len(unique_roles) / 3.0, 1.0)  # Normalize to 3 role types
            delivery_assessment['role_diversity'] = role_diversity

            # Overall delivery integration
            overall_delivery = (delivery_coverage + role_diversity) / 2
            delivery_assessment['overall_delivery_integration'] = overall_delivery

        return delivery_assessment

@dataclass
class IndicatorDatabase(Node):
    """Database system for managing collections of social indicators."""

    indicators: Dict[uuid.UUID, SocialIndicator] = field(default_factory=dict)  # type: ignore[misc]
    indicator_groups: Dict[str, List[uuid.UUID]] = field(default_factory=dict)  # type: ignore[misc]

    # Database metadata
    last_updated: Optional[datetime] = None
    update_frequency: Optional[timedelta] = None
    data_sources: List[str] = field(default_factory=list)  # type: ignore[misc]

    # Quality metrics
    overall_completeness: Optional[float] = None
    average_quality_score: Optional[float] = None
    data_coverage: Dict[str, float] = field(default_factory=dict)  # type: ignore[misc]

    # SFM Matrix Integration (Enhanced)
    matrix_coverage: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)  # Cell -> Indicators  # type: ignore[misc]
    matrix_integration_scores: Dict[uuid.UUID, float] = field(default_factory=dict)  # Cell -> integration score  # type: ignore[misc]
    institutional_coverage: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)  # Institution -> Indicators  # type: ignore[misc]

    # Policy and Delivery Integration
    policy_indicator_mapping: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)  # Policy -> Indicators  # type: ignore[misc]
    delivery_system_coverage: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)  # Delivery -> Indicators  # type: ignore[misc]

    def conduct_comprehensive_matrix_indicator_analysis(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Conduct comprehensive analysis of matrix-indicator linkages."""
        linkage_analysis = {
            'matrix_coverage_analysis': self._analyze_matrix_coverage(),
            'indicator_matrix_strength_analysis': self._analyze_indicator_matrix_strength(),
            'cross_matrix_indicator_relationships': self._analyze_cross_matrix_relationships(),
            'institutional_indicator_integration': self._analyze_institutional_integration(),
            'policy_indicator_effectiveness': self._analyze_policy_indicator_effectiveness(),
            'delivery_system_indicator_alignment': self._analyze_delivery_system_alignment(),
            'matrix_indicator_gaps': self._identify_matrix_indicator_gaps(),
            'optimization_recommendations': self._generate_optimization_recommendations()
        }

        return linkage_analysis

    def _analyze_matrix_coverage(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Analyze indicator coverage across matrix cells."""
        coverage_analysis = {
            'total_cells_covered': len(self.matrix_coverage),
            'cells_with_multiple_indicators': 0,
            'cells_with_weak_coverage': 0,
            'coverage_distribution': {},
            'coverage_quality_by_cell': {}
        }

        # Analyze coverage distribution
        indicator_counts = [len(indicators) for indicators in self.matrix_coverage.values()]
        if indicator_counts:
            coverage_analysis['coverage_distribution'] = {
                'mean_indicators_per_cell': sum(indicator_counts) / len(indicator_counts),
                'max_indicators_per_cell': max(indicator_counts),
                'min_indicators_per_cell': min(indicator_counts),
                'cells_without_indicators': len([count for count in indicator_counts if count == 0])
            }

        # Analyze coverage quality
        for cell_id, indicator_ids in self.matrix_coverage.items():
            coverage_analysis['cells_with_multiple_indicators'] += 1 if len(indicator_ids) > 1 else 0

            # Calculate cell coverage quality
            if indicator_ids:
                cell_indicators = [self.indicators[iid] for iid in indicator_ids if iid in self.indicators]
                if cell_indicators:
                    avg_strength = sum(
                        indicator.cell_indicator_strength.get(cell_id, 0.0)
                        for indicator in cell_indicators
                    ) / len(cell_indicators)

                    coverage_analysis['coverage_quality_by_cell'][str(cell_id)] = avg_strength

                    if avg_strength < 0.5:
                        coverage_analysis['cells_with_weak_coverage'] += 1

        return coverage_analysis

    def _analyze_indicator_matrix_strength(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Analyze strength of indicator-matrix relationships."""
        strength_analysis = {
            'overall_strength_distribution': {},
            'strong_relationships': [],
            'weak_relationships': [],
            'strength_by_indicator_type': {}
        }

        all_strengths = []
        strong_relationships = []
        weak_relationships = []

        for indicator in self.indicators.values():
            for cell_id, strength in indicator.cell_indicator_strength.items():
                all_strengths.append(strength)

                relationship_info = {
                    'indicator_id': indicator.id,
                    'indicator_label': indicator.label,
                    'cell_id': cell_id,
                    'strength': strength,
                    'relationship_type': indicator.matrix_cell_relationships.get(cell_id, 'unknown')
                }

                if strength > 0.7:
                    strong_relationships.append(relationship_info)
                elif strength < 0.3:
                    weak_relationships.append(relationship_info)

        if all_strengths:
            strength_analysis['overall_strength_distribution'] = {
                'mean_strength': sum(all_strengths) / len(all_strengths),
                'strong_relationships_count': len(strong_relationships),
                'weak_relationships_count': len(weak_relationships),
                'total_relationships': len(all_strengths)
            }

        strength_analysis['strong_relationships'] = strong_relationships[:10]  # Top 10
        strength_analysis['weak_relationships'] = weak_relationships[:10]  # Bottom 10

        return strength_analysis

    def _analyze_cross_matrix_relationships(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Analyze cross-matrix indicator relationships."""
        cross_matrix_analysis = {
            'indicators_with_cross_effects': 0,
            'cross_matrix_influence_patterns': {},
            'feedback_loop_indicators': 0,
            'system_level_effect_indicators': 0,
            'cross_matrix_coordination_needs': []
        }

        for indicator in self.indicators.values():
            # Count indicators with cross-matrix influences
            if indicator.cross_matrix_influences:
                cross_matrix_analysis['indicators_with_cross_effects'] += 1

                # Analyze influence patterns
                for influenced_cell in indicator.cross_matrix_influences:
                    pattern_key = f"{len(indicator.related_matrix_cells)}_to_1"
                    if pattern_key not in cross_matrix_analysis['cross_matrix_influence_patterns']:
                        cross_matrix_analysis['cross_matrix_influence_patterns'][pattern_key] = 0
                    cross_matrix_analysis['cross_matrix_influence_patterns'][pattern_key] += 1

            # Count feedback loop indicators
            if indicator.matrix_feedback_loops:
                cross_matrix_analysis['feedback_loop_indicators'] += 1

            # Count system-level effect indicators
            if indicator.system_level_effects:
                cross_matrix_analysis['system_level_effect_indicators'] += 1

                # Identify coordination needs
                if len(indicator.related_matrix_cells) > 3:
                    cross_matrix_analysis['cross_matrix_coordination_needs'].append({
                        'indicator_id': indicator.id,
                        'indicator_label': indicator.label,
                        'affected_cells_count': len(indicator.related_matrix_cells),
                        'coordination_complexity': 'high'
                    })

        return cross_matrix_analysis

    def _analyze_institutional_integration(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Analyze institutional integration of indicators."""
        institutional_analysis = {
            'institutional_coverage_analysis': {},
            'institution_indicator_density': {},
            'institutional_integration_gaps': [],
            'multi_institutional_indicators': 0
        }

        # Analyze institutional coverage
        for institution_id, indicator_ids in self.institutional_coverage.items():
            institution_indicators = [self.indicators[iid] for iid in indicator_ids if iid in self.indicators]

            if institution_indicators:
                # Calculate institution coverage metrics
                avg_integration = sum(
                    indicator.assess_matrix_integration_strength(
                        ).get('overall_matrix_integration',
                        0.0)
                    for indicator in institution_indicators
                ) / len(institution_indicators)

                institutional_analysis['institution_indicator_density'][str(institution_id)] = {
                    'indicator_count': len(institution_indicators),
                    'avg_integration_strength': avg_integration,
                    'indicator_types': list(set(indicator.indicator_type.name for indicator in institution_indicators))
                }

                # Identify integration gaps
                if avg_integration < 0.5:
                    institutional_analysis['institutional_integration_gaps'].append({
                        'institution_id': institution_id,
                        'integration_strength': avg_integration,
                        'gap_severity': 'high' if avg_integration < 0.3 else 'medium'
                    })

        # Count multi-institutional indicators
        institutional_analysis['multi_institutional_indicators'] = len([
            indicator for indicator in self.indicators.values()
            if len(indicator.affecting_institutions) > 1
        ])

        return institutional_analysis

    def _analyze_policy_indicator_effectiveness(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Analyze policy-indicator effectiveness relationships."""
        policy_analysis = {
            'policy_coverage_analysis': {},
            'policy_impact_effectiveness': {},
            'high_impact_policy_indicators': [],
            'policy_measurement_gaps': []
        }

        # Analyze policy coverage
        for policy_id, indicator_ids in self.policy_indicator_mapping.items():
            policy_indicators = [self.indicators[iid] for iid in indicator_ids if iid in self.indicators]

            if policy_indicators:
                # Calculate policy impact scores
                impact_scores = []
                for indicator in policy_indicators:
                    impact_score = indicator.policy_impact_measurement.get(policy_id)
                    if impact_score is not None:
                        impact_scores.append(impact_score)

                if impact_scores:
                    avg_impact = sum(impact_scores) / len(impact_scores)
                    policy_analysis['policy_impact_effectiveness'][str(policy_id)] = {
                        'avg_impact_score': avg_impact,
                        'indicator_count': len(policy_indicators),
                        'high_impact_indicators': len([score for score in impact_scores if score > 0.7])
                    }

                    # Identify high-impact relationships
                    if avg_impact > 0.7:
                        policy_analysis['high_impact_policy_indicators'].append({
                            'policy_id': policy_id,
                            'avg_impact_score': avg_impact,
                            'indicator_count': len(policy_indicators)
                        })
                else:
                    # Identify measurement gaps
                    policy_analysis['policy_measurement_gaps'].append({
                        'policy_id': policy_id,
                        'gap_type': 'no_impact_measurements',
                        'indicator_count': len(policy_indicators)
                    })

        return policy_analysis

    def _analyze_delivery_system_alignment(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Analyze delivery system-indicator alignment."""
        delivery_analysis = {
            'delivery_system_coverage': {},
            'delivery_indicator_roles': {},
            'delivery_alignment_strength': {},
            'delivery_optimization_opportunities': []
        }

        # Analyze delivery system coverage
        for delivery_id, indicator_ids in self.delivery_system_coverage.items():
            delivery_indicators = [self.indicators[iid] for iid in indicator_ids if iid in self.indicators]

            if delivery_indicators:
                # Analyze indicator roles in delivery system
                role_distribution = {}
                for indicator in delivery_indicators:
                    role = indicator.delivery_system_indicators.get(delivery_id, 'unknown')
                    role_distribution[role] = role_distribution.get(role, 0) + 1

                delivery_analysis['delivery_indicator_roles'][str(delivery_id)] = role_distribution

                # Calculate alignment strength
                delivery_integration_scores = [
                    indicator.assess_delivery_system_integration(
                        ).get('overall_delivery_integration',
                        0.0)
                    for indicator in delivery_indicators
                ]

                if delivery_integration_scores:
                    avg_alignment = sum(delivery_integration_scores) / len(delivery_integration_scores)
                    delivery_analysis['delivery_alignment_strength'][str(delivery_id)] = avg_alignment

                    # Identify optimization opportunities
                    if avg_alignment < 0.6:
                        delivery_analysis['delivery_optimization_opportunities'].append({
                            'delivery_id': delivery_id,
                            'current_alignment': avg_alignment,
                            'optimization_potential': 'high' if avg_alignment < 0.4 else 'medium'
                        })

        return delivery_analysis

    def _identify_matrix_indicator_gaps(self) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """Identify gaps in matrix-indicator linkages."""
        gaps = []

        # Matrix coverage gaps
        covered_cells = set(self.matrix_coverage.keys())
        total_possible_cells = len(covered_cells)  # Simplified - would use actual matrix size

        if total_possible_cells > 0:
            coverage_ratio = len(covered_cells) / total_possible_cells
            if coverage_ratio < 0.8:
                gaps.append({
                    'gap_type': 'insufficient_matrix_coverage',
                    'description': f'Only {coverage_ratio:.1%} of matrix cells have indicator coverage',
                    'severity': 'high' if coverage_ratio < 0.5 else 'medium',
                    'recommendation': 'Develop indicators for uncovered matrix cells'
                })

        # Weak relationship gaps
        weak_relationship_count = sum(
            1 for indicator in self.indicators.values()
            for strength in indicator.cell_indicator_strength.values()
            if strength < 0.3
        )

        if weak_relationship_count > len(self.indicators) * 0.2:  # More than 20% weak relationships
            gaps.append({
                'gap_type': 'weak_matrix_relationships',
                'description': f'{weak_relationship_count} weak indicator-matrix relationships found',
                'severity': 'medium',
                'recommendation': 'Strengthen weak indicator-matrix relationships'
            })

        # Cross-matrix integration gaps
        indicators_without_cross_effects = len([
            indicator for indicator in self.indicators.values()
            if not indicator.cross_matrix_influences
        ])

        if indicators_without_cross_effects > len(self.indicators) * 0.5:
            gaps.append({
                'gap_type': 'limited_cross_matrix_integration',
                'description': f'{indicators_without_cross_effects} indicators lack cross-matrix integration',
                'severity': 'medium',
                'recommendation': 'Develop cross-matrix indicator relationships'
            })

        return gaps

    def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """Generate recommendations for optimizing matrix-indicator linkages."""
        recommendations = []

        # Matrix coverage optimization
        matrix_coverage = self._analyze_matrix_coverage()
        if matrix_coverage.get('cells_with_weak_coverage', 0) > 0:
            recommendations.append({
                'type': 'strengthen_weak_coverage',
                'description': f"Strengthen indicator coverage for {matrix_coverage['cells_with_weak_coverage']} matrix cells",
                'priority': 'high',
                'implementation_approach': 'Develop targeted indicators or strengthen existing relationships'
            })

        # Relationship strength optimization
        strength_analysis = self._analyze_indicator_matrix_strength()
        weak_count = strength_analysis.get(
            'overall_strength_distribution',
            {}).get('weak_relationships_count',
            0)
        if weak_count > 0:
            recommendations.append({
                'type': 'strengthen_weak_relationships',
                'description': f'Strengthen {weak_count} weak indicator-matrix relationships',
                'priority': 'medium',
                'implementation_approach': 'Review and enhance indicator definitions and measurement methods'
            })

        # Cross-matrix integration optimization
        cross_matrix = self._analyze_cross_matrix_relationships()
        if cross_matrix.get('indicators_with_cross_effects', 0) < len(self.indicators) * 0.3:
            recommendations.append({
                'type': 'enhance_cross_matrix_integration',
                'description': 'Develop stronger cross-matrix indicator relationships',
                'priority': 'medium',
                'implementation_approach': 'Identify and model cross-matrix indicator influences'
            })

        # Institutional integration optimization
        institutional = self._analyze_institutional_integration()
        if institutional.get('institutional_integration_gaps'):
            recommendations.append({
                'type': 'improve_institutional_integration',
                'description': f"Address integration gaps for {len(institutional['institutional_integration_gaps'])} institutions",
                'priority': 'high',
                'implementation_approach': 'Develop institution-specific indicator strategies'
            })

        return recommendations

    # Matrix Analysis Capabilities
    matrix_completeness_scores: Dict[uuid.UUID, float] = field(default_factory=dict)  # Cell -> completeness  # type: ignore[misc]
    cross_matrix_relationships: Dict[str, List[uuid.UUID]] = field(default_factory=dict)  # Relationship type -> Indicators  # type: ignore[misc]

@dataclass
class StatisticalAnalysisPipeline(Node):
    """Statistical analysis pipeline for comprehensive indicator analysis."""

    target_indicators: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    analysis_scope: str = ""

    # Pipeline configuration
    analysis_methods: List[str] = field(default_factory=list)  # type: ignore[misc]
    statistical_tests: List[str] = field(default_factory=list)  # type: ignore[misc]
    regression_models: List[str] = field(default_factory=list)  # type: ignore[misc]
    time_series_methods: List[str] = field(default_factory=list)  # type: ignore[misc]

    # Real-time data integration
    data_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # type: ignore[misc]
    real_time_feeds: List[Dict[str, Any]] = field(default_factory=list)  # type: ignore[misc]
    data_update_frequency: Optional[timedelta] = None
    last_update_timestamp: Optional[datetime] = None

    # Analysis results
    statistical_results: Dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]
    model_performance: Dict[str, float] = field(default_factory=dict)  # type: ignore[misc]
    prediction_accuracy: Dict[str, float] = field(default_factory=dict)  # type: ignore[misc]

    # Quality control
    data_quality_scores: Dict[str, float] = field(default_factory=dict)  # type: ignore[misc]
    validation_results: Dict[str, bool] = field(default_factory=dict)  # type: ignore[misc]
    anomaly_detection_results: List[Dict[str, Any]] = field(default_factory=list)  # type: ignore[misc]

    def execute_comprehensive_statistical_analysis(self,
                                                 indicator_database: IndicatorDatabase) -> Dict[str, Any]:  # type: ignore[misc]
        """Execute comprehensive statistical analysis pipeline."""
        analysis_results = {
            'descriptive_statistics': self._compute_descriptive_statistics(indicator_database),
            'correlation_analysis': self._conduct_correlation_analysis(indicator_database),
            'regression_analysis': self._conduct_regression_analysis(indicator_database),
            'time_series_analysis': self._conduct_time_series_analysis(indicator_database),
            'predictive_modeling': self._conduct_predictive_modeling(indicator_database),
            'anomaly_detection': self._detect_anomalies(indicator_database),
            'trend_analysis': self._analyze_trends(indicator_database),
            'matrix_statistical_integration': self._integrate_matrix_statistics(indicator_database)
        }

        self.statistical_results = analysis_results
        return analysis_results
    matrix_feedback_coverage: Optional[float] = None  # Overall feedback loop coverage

    def add_indicator(self, indicator: SocialIndicator, group: Optional[str] = None) -> None:
        """Add an indicator to the database."""
        self.indicators[indicator.id] = indicator

        if group:
            if group not in self.indicator_groups:
                self.indicator_groups[group] = []
            self.indicator_groups[group].append(indicator.id)

        self.last_updated = datetime.now()

    def get_indicators_by_type(self, indicator_type: IndicatorType) -> List[SocialIndicator]:
        """Get all indicators of a specific type."""
        return [indicator for indicator in self.indicators.values()
                if indicator.indicator_type == indicator_type]

    def get_indicators_by_value_category(self, category: ValueCategory) -> List[SocialIndicator]:
        """Get all indicators in a specific value category."""
        return [indicator for indicator in self.indicators.values()
                if indicator.value_category == category]

    def get_indicators_for_matrix_cell(self, cell_id: uuid.UUID) -> List[SocialIndicator]:
        """Get all indicators related to a specific matrix cell."""
        return [indicator for indicator in self.indicators.values()
                if cell_id in indicator.related_matrix_cells]

    def calculate_completeness(self) -> float:
        """Calculate overall data completeness."""
        if not self.indicators:
            return 0.0

        total_expected_measurements = 0
        total_actual_measurements = 0

        for indicator in self.indicators.values():
            # Estimate expected measurements based on frequency
            if indicator.measurement_frequency:
                days_since_creation = (datetime.now() - indicator.created_at).days
                expected_measurements = max(
                    1,
                    days_since_creation // indicator.measurement_frequency.days)
                total_expected_measurements += expected_measurements
            else:
                total_expected_measurements += 1  # At least one measurement expected

            total_actual_measurements += len(indicator.measurements)

        if total_expected_measurements == 0:
            return 0.0

        completeness = min(1.0, total_actual_measurements / total_expected_measurements)
        self.overall_completeness = completeness
        return completeness

    def calculate_quality_score(self) -> float:
        """Calculate average quality score across all indicators."""
        if not self.indicators:
            return 0.0

        quality_scores = []

        for indicator in self.indicators.values():
            if indicator.measurements:
                measurement_scores = [m.calculate_quality_score() for m in indicator.measurements]
                indicator_quality = statistics.mean(measurement_scores)
                quality_scores.append(indicator_quality)

        if not quality_scores:
            return 0.0

        avg_quality = statistics.mean(quality_scores)
        self.average_quality_score = avg_quality
        return avg_quality

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of the database."""
        return {
            'total_indicators': len(self.indicators),
            'total_measurements': sum(len(ind.measurements) for ind in self.indicators.values()),
            'indicator_groups': {group: len(
                indicators) for group,
                indicators in self.indicator_groups.items()},
            'completeness': self.calculate_completeness(),
            'quality_score': self.calculate_quality_score(),
            'last_updated': self.last_updated,
            'coverage_by_type': self._calculate_type_coverage(),
            'coverage_by_category': self._calculate_category_coverage()
        }

    def _calculate_type_coverage(self) -> Dict[str, int]:  # type: ignore[misc]
        """Calculate coverage by indicator type."""
        type_counts = {}
        for indicator in self.indicators.values():
            type_name = indicator.indicator_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts

    def _calculate_category_coverage(self) -> Dict[str, int]:  # type: ignore[misc]
        """Calculate coverage by value category."""
        category_counts = {}
        for indicator in self.indicators.values():
            category_name = indicator.value_category.name
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
        return category_counts

    def update_matrix_coverage(self) -> None:
        """Update matrix coverage mapping based on current indicators."""
        # Reset coverage mappings
        self.matrix_coverage.clear()
        self.institutional_coverage.clear()
        self.policy_indicator_mapping.clear()
        self.delivery_system_coverage.clear()

        # Build coverage mappings
        for indicator in self.indicators.values():
            # Matrix cell coverage
            for cell_id in indicator.related_matrix_cells:
                if cell_id not in self.matrix_coverage:
                    self.matrix_coverage[cell_id] = []
                self.matrix_coverage[cell_id].append(indicator.id)

            # Institutional coverage
            for institution_id in indicator.affecting_institutions:
                if institution_id not in self.institutional_coverage:
                    self.institutional_coverage[institution_id] = []
                self.institutional_coverage[institution_id].append(indicator.id)

            # Policy coverage
            for policy_id in indicator.policy_relevance:
                if policy_id not in self.policy_indicator_mapping:
                    self.policy_indicator_mapping[policy_id] = []
                self.policy_indicator_mapping[policy_id].append(indicator.id)

            # Delivery system coverage
            for delivery_id in indicator.delivery_system_indicators.keys():
                if delivery_id not in self.delivery_system_coverage:
                    self.delivery_system_coverage[delivery_id] = []
                self.delivery_system_coverage[delivery_id].append(indicator.id)

    def calculate_matrix_integration_completeness(self) -> Dict[str, float]:  # type: ignore[misc]
        """Calculate completeness of matrix integration."""
        self.update_matrix_coverage()

        completeness_assessment = {}

        # Matrix cell coverage completeness
        if self.matrix_coverage:
            total_cells = len(self.matrix_coverage)
            well_covered_cells = sum(1 for indicators in self.matrix_coverage.values()
                                   if len(indicators) >= 2)  # At least 2 indicators per cell
            cell_completeness = well_covered_cells / total_cells if total_cells > 0 else 0
            completeness_assessment['matrix_cell_completeness'] = cell_completeness

        # Institutional coverage completeness
        if self.institutional_coverage:
            total_institutions = len(self.institutional_coverage)
            well_covered_institutions = sum(1 for indicators in self.institutional_coverage.values()
                                          if len(indicators) >= 3)  # At least 3 indicators per institution
            institutional_completeness = well_covered_institutions / total_institutions if total_institutions > 0 else 0
            completeness_assessment['institutional_completeness'] = institutional_completeness

        # Policy coverage completeness
        if self.policy_indicator_mapping:
            total_policies = len(self.policy_indicator_mapping)
            covered_policies = sum(1 for indicators in self.policy_indicator_mapping.values()
                                 if len(indicators) >= 1)  # At least 1 indicator per policy
            policy_completeness = covered_policies / total_policies if total_policies > 0 else 0
            completeness_assessment['policy_completeness'] = policy_completeness

        # Delivery system completeness
        if self.delivery_system_coverage:
            total_deliveries = len(self.delivery_system_coverage)
            covered_deliveries = sum(1 for indicators in self.delivery_system_coverage.values()
                                   if len(indicators) >= 1)  # At least 1 indicator per delivery
            delivery_completeness = covered_deliveries / total_deliveries if total_deliveries > 0 else 0
            completeness_assessment['delivery_completeness'] = delivery_completeness

        # Overall integration completeness
        if completeness_assessment:
            overall_completeness = sum(completeness_assessment.values()) / len(completeness_assessment)  # type: ignore[arg-type]
            completeness_assessment['overall_integration_completeness'] = overall_completeness

        return completeness_assessment

    def identify_matrix_coverage_gaps(self) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """Identify gaps in matrix coverage."""
        gaps = []
        self.update_matrix_coverage()

        # Matrix cells with insufficient coverage
        for cell_id, indicators in self.matrix_coverage.items():
            if len(indicators) < 2:
                gaps.append({
                    'type': 'insufficient_cell_coverage',
                    'cell_id': cell_id,
                    'current_indicators': len(indicators),
                    'recommended_indicators': 2,
                    'priority': 'high'
                })

        # Institutions with no indicators
        missing_institutional_coverage = []
        for institution_id, indicators in self.institutional_coverage.items():
            if len(indicators) == 0:
                missing_institutional_coverage.append(institution_id)

        if missing_institutional_coverage:
            gaps.append({
                'type': 'missing_institutional_coverage',
                'uncovered_institutions': missing_institutional_coverage,
                'count': len(missing_institutional_coverage),  # type: ignore[arg-type]
                'priority': 'medium'
            })

        # Policies without indicators
        missing_policy_coverage = []
        for policy_id, indicators in self.policy_indicator_mapping.items():
            if len(indicators) == 0:
                missing_policy_coverage.append(policy_id)

        if missing_policy_coverage:
            gaps.append({
                'type': 'missing_policy_coverage',
                'uncovered_policies': missing_policy_coverage,
                'count': len(missing_policy_coverage),  # type: ignore[arg-type]
                'priority': 'medium'
            })

        return gaps

    def calculate_matrix_integration_scores(self) -> Dict[uuid.UUID, float]:  # type: ignore[misc]
        """Calculate integration scores for each matrix cell."""
        integration_scores = {}

        for cell_id, indicators in self.matrix_coverage.items():
            if not indicators:
                integration_scores[cell_id] = 0.0
                continue

            # Calculate average integration strength for this cell
            total_strength = 0.0
            valid_indicators = 0

            for indicator_id in indicators:
                if indicator_id in self.indicators:
                    indicator = self.indicators[indicator_id]
                    cell_strength = indicator.cell_indicator_strength.get(cell_id, 0.0)
                    total_strength += cell_strength
                    valid_indicators += 1

            if valid_indicators > 0:
                avg_strength = total_strength / valid_indicators
                # Bonus for multiple indicators
                coverage_bonus = min(len(indicators) / 3.0, 1.0)  # Up to 3 indicators
                integration_score = (avg_strength * 0.7) + (coverage_bonus * 0.3)
                integration_scores[cell_id] = min(integration_score, 1.0)
            else:
                integration_scores[cell_id] = 0.0

        self.matrix_integration_scores = integration_scores
        return integration_scores

    def generate_matrix_integration_report(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Generate comprehensive matrix integration report."""
        self.update_matrix_coverage()
        completeness = self.calculate_matrix_integration_completeness()
        integration_scores = self.calculate_matrix_integration_scores()
        gaps = self.identify_matrix_coverage_gaps()

        report = {
            'overview': {
                'total_indicators': len(self.indicators),
                'matrix_cells_covered': len(self.matrix_coverage),
                'institutions_covered': len(self.institutional_coverage),
                'policies_covered': len(self.policy_indicator_mapping),
                'delivery_systems_covered': len(self.delivery_system_coverage)
            },
            'completeness_assessment': completeness,
            'integration_quality': {
                'average_cell_integration': sum(integration_scores.values()) / len(integration_scores) if integration_scores else 0,
                'high_quality_cells': sum(1 for score in integration_scores.values() if score > 0.7),
                'low_quality_cells': sum(1 for score in integration_scores.values() if score < 0.3)
            },
            'coverage_gaps': gaps,
            'recommendations': self._generate_integration_recommendations(completeness, gaps)
        }

        return report

    def _generate_integration_recommendations(self, completeness: Dict[str, float],
                                           gaps: List[Dict[str, Any]]) -> List[str]:  # type: ignore[misc]
        """Generate recommendations for improving matrix integration."""
        recommendations = []

        # Completeness-based recommendations
        cell_completeness = completeness.get('matrix_cell_completeness', 1.0)
        if cell_completeness < 0.6:
            recommendations.append("Increase indicator coverage for matrix cells - many cells have insufficient measurement")

        institutional_completeness = completeness.get('institutional_completeness', 1.0)
        if institutional_completeness < 0.5:
            recommendations.append("Expand institutional indicator coverage - many institutions lack adequate measurement")

        policy_completeness = completeness.get('policy_completeness', 1.0)
        if policy_completeness < 0.7:
            recommendations.append("Develop indicators for policy evaluation - many policies lack measurement frameworks")

        # Gap-based recommendations
        high_priority_gaps = [gap for gap in gaps if gap.get('priority') == 'high']
        if high_priority_gaps:
            recommendations.append(f"Address {len(high_priority_gaps)} high-priority coverage gaps immediately")

        # Overall integration recommendations
        overall_completeness = completeness.get('overall_integration_completeness', 1.0)
        if overall_completeness < 0.5:
            recommendations.append("Undertake comprehensive indicator development program to improve matrix integration")

        if not recommendations:
            recommendations.append("Matrix integration is well-developed - focus on maintaining data quality and coverage")

        return recommendations

@dataclass
class StatisticalAnalyzer:
    """Tools for statistical analysis of indicator data."""

    database: IndicatorDatabase

    def analyze_correlations(self, indicator_ids: List[uuid.UUID],
                           time_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, float]:  # type: ignore[misc]
        """Analyze correlations between indicators."""
        correlations = {}

        indicators = [self.database.indicators[iid] for iid in indicator_ids
                     if iid in self.database.indicators]

        if len(indicators) < 2:
            return correlations

        # Get measurement data for each indicator
        indicator_data = {}
        for indicator in indicators:
            measurements = indicator.measurements
            if time_period:
                start_date, end_date = time_period
                measurements = indicator.get_measurements_in_period(start_date, end_date)

            numeric_values = [m.get_numeric_value() for m in measurements
                            if m.get_numeric_value() is not None]

            if numeric_values:
                indicator_data[indicator.id] = numeric_values

        # Calculate pairwise correlations
        indicator_list = list(indicator_data.keys())  # type: ignore[arg-type]
        for i, ind1 in enumerate(indicator_list):  # type: ignore[arg-type]
            for ind2 in indicator_list[i+1:]:
                data1 = indicator_data[ind1]
                data2 = indicator_data[ind2]

                # Align data by length (simple approach)
                min_length = min(len(data1), len(data2))  # type: ignore[arg-type]
                if min_length > 1:
                    correlation = self._calculate_correlation(  # type: ignore[arg-type,misc]
                        data1[-min_length:], data2[-min_length:]  # type: ignore[arg-type]
                    )
                    correlations[f"{ind1}_{ind2}"] = correlation

        return correlations

    def _calculate_correlation(self, data1: List[float], data2: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(data1) != len(data2) or len(data1) < 2:
            return 0.0

        n = len(data1)
        mean1 = sum(data1) / n
        mean2 = sum(data2) / n

        numerator = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(n))

        sum_sq1 = sum((x - mean1) ** 2 for x in data1)
        sum_sq2 = sum((x - mean2) ** 2 for x in data2)

        denominator = math.sqrt(sum_sq1 * sum_sq2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def perform_trend_analysis(self, indicator_id: uuid.UUID,
                              periods: int = 12) -> Dict[str, Any]:
        """Perform comprehensive trend analysis on an indicator."""
        if indicator_id not in self.database.indicators:
            return {}

        indicator = self.database.indicators[indicator_id]
        measurements = indicator.measurements[-periods:]

        numeric_values = [m.get_numeric_value() for m in measurements
                         if m.get_numeric_value() is not None]

        if len(numeric_values) < 3:
            return {'error': 'Insufficient data for trend analysis'}

        # Calculate trend statistics
        trend_direction = indicator.calculate_trend(periods)
        volatility = indicator.calculate_volatility(periods)

        # Calculate additional statistics
        clean_values = [v for v in numeric_values if v is not None]
        if not clean_values:
            return {'error': 'No valid numeric data for analysis'}

        mean_value = statistics.mean(clean_values)
        median_value = statistics.median(clean_values)
        std_dev = statistics.stdev(clean_values) if len(clean_values) > 1 else 0

        # Calculate growth rate (if applicable)
        growth_rate = None
        if len(clean_values) >= 2 and clean_values[0] != 0:
            growth_rate = ((clean_values[-1] - clean_values[0]) / clean_values[0]) * 100

        return {
            'trend_direction': trend_direction.name,
            'volatility': volatility,
            'mean': mean_value,
            'median': median_value,
            'standard_deviation': std_dev,
            'growth_rate_percent': growth_rate,
            'coefficient_of_variation': std_dev / mean_value if mean_value != 0 else 0,
            'data_points': len(numeric_values),
            'analysis_period': periods
        }

    def create_composite_indicator(self, component_indicators: List[uuid.UUID],
                                  weights: Optional[List[float]] = None,
                                  aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE) -> SocialIndicator:
        """Create a composite indicator from multiple component indicators."""
        if not component_indicators:
            raise ValueError("No component indicators provided")

        if weights and len(weights) != len(component_indicators):
            raise ValueError("Number of weights must match number of indicators")

        if not weights:
            weights = [1.0 / len(component_indicators)] * len(component_indicators)

        # Create composite indicator
        composite = SocialIndicator(
            label=f"Composite Indicator ({len(component_indicators)} components)",
            indicator_type=IndicatorType.COMPOSITE_INDICATOR,
            measurement_unit="composite_index"
        )

        # Get all unique timestamps from component indicators
        all_timestamps = set()
        component_data = {}

        for ind_id in component_indicators:
            if ind_id in self.database.indicators:
                indicator = self.database.indicators[ind_id]
                component_data[ind_id] = {}

                for measurement in indicator.measurements:
                    timestamp = measurement.timestamp
                    all_timestamps.add(timestamp)
                    component_data[ind_id][timestamp] = measurement.get_numeric_value()

        # Calculate composite values for each timestamp
        for timestamp in sorted(all_timestamps):  # type: ignore[arg-type]
            values = []
            valid_weights = []

            for i, ind_id in enumerate(component_indicators):
                if (ind_id in component_data and
                    timestamp in component_data[ind_id] and
                    component_data[ind_id][timestamp] is not None):

                    values.append(component_data[ind_id][timestamp])  # type: ignore[arg-type]
                    valid_weights.append(weights[i])

            if values:
                composite_value = self._aggregate_values(
                    values,
                    valid_weights,
                    aggregation_method)  # type: ignore[arg-type]

                measurement = IndicatorMeasurement(  # type: ignore[arg-type,misc]
                    value=composite_value,
                    timestamp=timestamp,  # type: ignore[arg-type,misc]
                    confidence_level=min(
                        len(values) / len(component_indicators),
                        1.0)  # type: ignore[arg-type]
                )
                composite.add_measurement(measurement)

        return composite

    def _aggregate_values(self, values: List[float], weights: List[float],
                         method: AggregationMethod) -> float:
        """Aggregate values using specified method."""
        if not values:
            return 0.0

        if method == AggregationMethod.MEAN:
            return statistics.mean(values)
        elif method == AggregationMethod.MEDIAN:
            return statistics.median(values)
        elif method == AggregationMethod.SUM:
            return sum(values)
        elif method == AggregationMethod.WEIGHTED_AVERAGE:
            if len(weights) != len(values):
                return statistics.mean(values)
            total_weight = sum(weights)
            if total_weight == 0:
                return statistics.mean(values)
            return sum(v * w for v, w in zip(values, weights)) / total_weight
        elif method == AggregationMethod.GEOMETRIC_MEAN:
            if any(v <= 0 for v in values):
                return 0.0
            product = 1.0
            for v in values:
                product *= v
            return product ** (1.0 / len(values))
        else:
            return statistics.mean(values)

@dataclass
class IndicatorDashboard(Node):
    """Visualization and monitoring system for social indicators."""

    database: IndicatorDatabase
    # Dashboard state
    monitored_indicators: List[uuid.UUID] = field(default_factory=list)  # type: ignore[misc]
    alert_thresholds: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)  # type: ignore[misc]
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)  # type: ignore[misc]
    last_refresh: Optional[datetime] = None
    refresh_frequency: timedelta = timedelta(hours=1)

    def add_monitored_indicator(self, indicator_id: uuid.UUID,
                               thresholds: Optional[Dict[str, float]] = None) -> None:
        """Add an indicator to be monitored on the dashboard."""
        if indicator_id not in self.monitored_indicators:
            self.monitored_indicators.append(indicator_id)

        if thresholds:
            self.alert_thresholds[indicator_id] = thresholds

    def check_alerts(self) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """Check for alert conditions across monitored indicators."""
        alerts = []

        for indicator_id in self.monitored_indicators:
            if indicator_id not in self.database.indicators:
                continue

            indicator = self.database.indicators[indicator_id]
            current_status = indicator.get_current_status()

            # Check threshold-based alerts
            if indicator_id in self.alert_thresholds:
                thresholds = self.alert_thresholds[indicator_id]
                current_value = current_status.get('current_value')

                if current_value is not None:
                    try:
                        numeric_value = float(current_value)

                        for threshold_type, threshold_value in thresholds.items():
                            alert_triggered = False
                            severity = 'low'

                            if threshold_type == 'min' and numeric_value < threshold_value:
                                alert_triggered = True
                                severity = 'high' if numeric_value < threshold_value * 0.8 else 'medium'
                            elif threshold_type == 'max' and numeric_value > threshold_value:
                                alert_triggered = True
                                severity = 'high' if numeric_value > threshold_value * 1.2 else 'medium'

                            if alert_triggered:
                                alerts.append({
                                    'indicator_id': indicator_id,
                                    'indicator_label': indicator.label,
                                    'alert_type': f'{threshold_type}_threshold',
                                    'current_value': numeric_value,
                                    'threshold_value': threshold_value,
                                    'severity': severity,
                                    'timestamp': datetime.now()
                                })

                    except (ValueError, TypeError):
                        pass

            # Check trend-based alerts
            if current_status['trend'] in ['DECREASING', 'VOLATILE']:
                alerts.append({
                    'indicator_id': indicator_id,
                    'indicator_label': indicator.label,
                    'alert_type': 'trend_concern',
                    'trend': current_status['trend'],
                    'severity': 'low',
                    'timestamp': datetime.now()
                })

        self.active_alerts = alerts
        return alerts

    def generate_summary_dashboard(self) -> Dict[str, Any]:  # type: ignore[misc]
        """Generate summary dashboard data."""
        summary = {
            'total_indicators': len(self.monitored_indicators),
            'active_alerts': len(self.active_alerts),
            'alert_breakdown': {},
            'indicator_status': {},
            'system_health': 'good',
            'last_updated': datetime.now()
        }

        # Alert breakdown
        alert_counts = {}
        for alert in self.active_alerts:
            severity = alert['severity']
            alert_counts[severity] = alert_counts.get(severity, 0) + 1
        summary['alert_breakdown'] = alert_counts

        # Indicator status summary
        status_counts = {'normal': 0, 'warning': 0, 'critical': 0}

        for indicator_id in self.monitored_indicators:
            if indicator_id in self.database.indicators:
                indicator = self.database.indicators[indicator_id]
                status = indicator.get_current_status()

                if status['threshold_status'] == 'critical':
                    status_counts['critical'] += 1
                elif status['threshold_status'] in ['below_minimum', 'above_maximum']:
                    status_counts['warning'] += 1
                else:
                    status_counts['normal'] += 1

        summary['indicator_status'] = status_counts

        # Overall system health
        if status_counts['critical'] > 0:
            summary['system_health'] = 'critical'
        elif status_counts['warning'] > len(self.monitored_indicators) * 0.3:
            summary['system_health'] = 'warning'

        return summary
