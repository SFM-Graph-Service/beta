"""
Comprehensive Indicator Systems for Social Fabric Matrix quantification.

This module implements systematic indicator frameworks for SFM analysis,
providing structured approaches to measuring institutional performance,
social outcomes, and value realization. It integrates with the broader
SFM framework to support evidence-based analysis and decision-making.

Key Components:
- IndicatorSystem: Comprehensive indicator framework
- SocialFabricIndicator: Individual indicators for SFM analysis
- IndicatorRelationship: Relationships between indicators
- PerformanceMeasurement: Systematic measurement approaches
- IndicatorValidation: Validation and quality assurance
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum, auto
import statistics

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.sfm_enums import (
    ValueCategory,
    SocialFabricIndicatorType,
    IndicatorType,
    MeasurementApproach,
    EvidenceQuality,
    SystemLevel,
    StatisticalMethod,
    ValidationMethod,
)

class IndicatorCategory(Enum):
    """Categories of indicators in SFM analysis."""

    OUTCOME_INDICATOR = auto()          # Final outcomes/impacts
    OUTPUT_INDICATOR = auto()           # Direct outputs/deliveries
    PROCESS_INDICATOR = auto()          # Process efficiency/quality
    INPUT_INDICATOR = auto()            # Resource inputs
    CONTEXT_INDICATOR = auto()          # Contextual factors
    IMPACT_INDICATOR = auto()           # Long-term impacts

class IndicatorDataType(Enum):
    """Data types for indicator measurement."""

    QUANTITATIVE = auto()               # Numerical data
    QUALITATIVE = auto()                # Descriptive data
    BINARY = auto()                     # Yes/no data
    ORDINAL = auto()                    # Ranked data
    CATEGORICAL = auto()                # Category data
    COMPOSITE = auto()                  # Multi-dimensional index

class IndicatorFrequency(Enum):
    """Frequency of indicator measurement."""

    REAL_TIME = auto()                  # Continuous monitoring
    DAILY = auto()                      # Daily measurement
    WEEKLY = auto()                     # Weekly measurement
    MONTHLY = auto()                    # Monthly measurement
    QUARTERLY = auto()                  # Quarterly measurement
    ANNUAL = auto()                     # Annual measurement
    AD_HOC = auto()                     # As-needed measurement

class IndicatorTrend(Enum):
    """Trend patterns for indicator values."""

    IMPROVING = auto()                  # Consistently improving
    STABLE = auto()                     # Stable/no change
    DECLINING = auto()                  # Consistently declining
    VOLATILE = auto()                   # High variability
    CYCLICAL = auto()                   # Cyclical patterns
    UNKNOWN = auto()                    # Insufficient data

@dataclass
class IndicatorSpecification(Node):
    """Detailed specification for an indicator."""

    indicator_category: IndicatorCategory = IndicatorCategory.OUTCOME_INDICATOR
    data_type: IndicatorDataType = IndicatorDataType.QUANTITATIVE
    measurement_frequency: IndicatorFrequency = IndicatorFrequency.ANNUAL

    # Specification details
    measurement_unit: Optional[str] = None
    calculation_method: Optional[str] = None
    data_sources: List[str] = field(default_factory=list)
    collection_method: Optional[str] = None

    # Measurement scale
    minimum_value: Optional[float] = None
    maximum_value: Optional[float] = None
    target_value: Optional[float] = None
    threshold_values: Dict[str, float] = field(default_factory=dict)  # Performance thresholds

    # Quality specifications
    precision_requirements: Optional[str] = None
    accuracy_requirements: Optional[str] = None
    reliability_requirements: Optional[float] = None  # 0-1 scale
    validity_requirements: Optional[str] = None

    # Temporal specifications
    measurement_timing: Optional[str] = None
    reporting_timeline: Optional[str] = None
    historical_baseline_period: Optional[TimeSlice] = None

    # Spatial specifications
    geographic_aggregation: Optional[str] = None
    spatial_disaggregation: List[str] = field(default_factory=list)

    # Stakeholder specifications
    primary_users: List[uuid.UUID] = field(default_factory=list)
    data_providers: List[uuid.UUID] = field(default_factory=list)
    reporting_recipients: List[uuid.UUID] = field(default_factory=list)

    def validate_specification_completeness(self) -> Dict[str, Any]:
        """Validate completeness of indicator specification."""
        validation_results = {
            'completeness_score': 0.0,
            'missing_elements': [],
            'specification_quality': 'incomplete',
            'recommendations': []
        }

        required_elements = [
            ('measurement_unit', self.measurement_unit),
            ('calculation_method', self.calculation_method),
            ('data_sources', self.data_sources),
            ('target_value', self.target_value),
            ('primary_users', self.primary_users)
        ]

        complete_elements = sum(1 for name, value in required_elements if value)
        validation_results['completeness_score'] = complete_elements / len(required_elements)

        # Identify missing elements
        for name, value in required_elements:
            if not value:
                validation_results['missing_elements'].append(name)

        # Assess specification quality
        if validation_results['completeness_score'] >= 0.9:
            validation_results['specification_quality'] = 'excellent'
        elif validation_results['completeness_score'] >= 0.7:
            validation_results['specification_quality'] = 'good'
        elif validation_results['completeness_score'] >= 0.5:
            validation_results['specification_quality'] = 'adequate'

        # Generate recommendations
        if validation_results['completeness_score'] < 0.7:
            validation_results['recommendations'].append('Complete missing specification elements')
        if not self.threshold_values:
            validation_results['recommendations'].append('Define performance thresholds')

        return validation_results

@dataclass
class SocialFabricIndicator(Node):
    """Individual indicator for SFM analysis with measurement data."""

    indicator_type: SocialFabricIndicatorType = SocialFabricIndicatorType.INSTITUTIONAL_COHERENCE
    value_category: ValueCategory = ValueCategory.SOCIAL

    # Indicator specification
    specification: Optional[uuid.UUID] = None  # IndicatorSpecification ID
    measurement_approach: MeasurementApproach = MeasurementApproach.QUANTITATIVE

    # Current measurement
    current_value: Optional[float] = None
    current_value_date: Optional[datetime] = None
    measurement_confidence: Optional[float] = None  # 0-1 scale
    data_quality: EvidenceQuality = EvidenceQuality.MEDIUM

    # Historical data
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)
    baseline_value: Optional[float] = None
    baseline_date: Optional[datetime] = None

    # Performance assessment
    target_value: Optional[float] = None
    performance_against_target: Optional[float] = None  # Current/target ratio
    trend_direction: IndicatorTrend = IndicatorTrend.UNKNOWN
    trend_strength: Optional[float] = None  # 0-1 scale

    # Contextual information
    institutional_context: List[uuid.UUID] = field(default_factory=list)
    geographic_context: Optional[SpatialUnit] = None
    temporal_context: Optional[TimeSlice] = None

    # Relationships
    contributing_indicators: List[uuid.UUID] = field(default_factory=list)  # Indicators that feed into this
    dependent_indicators: List[uuid.UUID] = field(default_factory=list)     # Indicators that depend on this
    related_matrix_cells: List[uuid.UUID] = field(default_factory=list)    # SFM cells this measures

    # Validation and quality
    validation_methods: List[ValidationMethod] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)

    # Reporting
    last_updated: Optional[datetime] = None
    update_frequency: IndicatorFrequency = IndicatorFrequency.ANNUAL
    reporting_status: Optional[str] = None

    def calculate_performance_score(self) -> Optional[float]:
        """Calculate performance score against target."""
        if self.current_value is not None and self.target_value is not None and self.target_value != 0:
            performance_ratio = self.current_value / self.target_value
            self.performance_against_target = performance_ratio

            # Convert to 0-1 performance score (capped at 1.0 for over-achievement)
            performance_score = min(performance_ratio, 1.0)
            return performance_score

        return None

    def analyze_trend_pattern(self) -> Dict[str, Any]:
        """Analyze trend patterns in historical data."""
        trend_analysis = {
            'trend_direction': self.trend_direction.name,
            'trend_strength': 0.0,
            'volatility': 0.0,
            'recent_change': 0.0,
            'trend_consistency': 0.0,
            'trend_interpretation': ''
        }

        if len(self.historical_values) < 3:
            trend_analysis['trend_interpretation'] = 'Insufficient data for trend analysis'
            return trend_analysis

        # Extract values and calculate basic statistics
        values = [value for _, value in self.historical_values]
        dates = [date for date, _ in self.historical_values]

        # Sort by date to ensure chronological order
        sorted_data = sorted(zip(dates, values))
        sorted_values = [value for _, value in sorted_data]

        # Calculate trend strength using linear regression slope
        n = len(sorted_values)
        x_values = list(range(n))  # Time points

        # Simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(sorted_values) / n

        numerator = sum((x_values[i] - x_mean) * (sorted_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        if denominator != 0:
            slope = numerator / denominator

            # Normalize slope to get trend strength
            value_range = max(sorted_values) - min(sorted_values)
            if value_range != 0:
                normalized_slope = abs(slope) / (value_range / n)
                trend_analysis['trend_strength'] = min(normalized_slope, 1.0)

            # Determine trend direction
            if slope > 0.1:
                trend_analysis['trend_direction'] = 'IMPROVING'
                self.trend_direction = IndicatorTrend.IMPROVING
            elif slope < -0.1:
                trend_analysis['trend_direction'] = 'DECLINING'
                self.trend_direction = IndicatorTrend.DECLINING
            else:
                trend_analysis['trend_direction'] = 'STABLE'
                self.trend_direction = IndicatorTrend.STABLE

        # Calculate volatility
        if len(values) > 1:
            value_std = statistics.stdev(values)
            value_mean = statistics.mean(values)
            if value_mean != 0:
                trend_analysis['volatility'] = value_std / abs(value_mean)

        # Recent change (last vs. second-to-last value)
        if len(sorted_values) >= 2:
            recent_change = (sorted_values[-1] - sorted_values[-2]) / abs(sorted_values[-2]) if sorted_values[-2] != 0 else 0
            trend_analysis['recent_change'] = recent_change

        self.trend_strength = trend_analysis['trend_strength']

        return trend_analysis

    def assess_data_quality(self) -> Dict[str, Any]:
        """Assess quality of indicator data."""
        quality_assessment = {
            'completeness': 0.0,
            'timeliness': 0.0,
            'accuracy': 0.0,
            'reliability': 0.0,
            'overall_quality': EvidenceQuality.LOW,
            'quality_issues': self.quality_issues.copy(),
            'improvement_recommendations': []
        }

        # Completeness assessment
        if self.current_value is not None:
            quality_assessment['completeness'] += 0.4
        if self.historical_values:
            quality_assessment['completeness'] += 0.3
        if self.baseline_value is not None:
            quality_assessment['completeness'] += 0.3

        # Timeliness assessment
        if self.current_value_date and self.last_updated:
            days_old = (datetime.now() - self.last_updated).days
            expected_days = {
                IndicatorFrequency.DAILY: 2,
                IndicatorFrequency.WEEKLY: 10,
                IndicatorFrequency.MONTHLY: 35,
                IndicatorFrequency.QUARTERLY: 100,
                IndicatorFrequency.ANNUAL: 400
            }.get(self.update_frequency, 365)

            timeliness_score = max(0.0, 1.0 - (days_old / expected_days))
            quality_assessment['timeliness'] = timeliness_score

        # Confidence-based assessments
        if self.measurement_confidence is not None:
            quality_assessment['accuracy'] = self.measurement_confidence
            quality_assessment['reliability'] = self.measurement_confidence

        # Overall quality determination
        quality_scores = [
            quality_assessment['completeness'],
            quality_assessment['timeliness'],
            quality_assessment['accuracy'],
            quality_assessment['reliability']
        ]
        valid_scores = [score for score in quality_scores if score > 0]

        if valid_scores:
            overall_score = sum(valid_scores) / len(valid_scores)
            if overall_score >= 0.8:
                quality_assessment['overall_quality'] = EvidenceQuality.HIGH
            elif overall_score >= 0.6:
                quality_assessment['overall_quality'] = EvidenceQuality.MEDIUM
            else:
                quality_assessment['overall_quality'] = EvidenceQuality.LOW

            self.data_quality = quality_assessment['overall_quality']

        # Generate improvement recommendations
        if quality_assessment['completeness'] < 0.7:
            quality_assessment['improvement_recommendations'].append('Improve data completeness')
        if quality_assessment['timeliness'] < 0.6:
            quality_assessment['improvement_recommendations'].append('Update data more frequently')
        if quality_assessment['accuracy'] < 0.6:
            quality_assessment['improvement_recommendations'].append('Enhance measurement accuracy')

        return quality_assessment

@dataclass
class IndicatorRelationship(Node):
    """Relationship between indicators in the system."""

    source_indicator_id: Optional[uuid.UUID] = None
    target_indicator_id: Optional[uuid.UUID] = None
    relationship_type: str = "contributes_to"  # contributes_to, depends_on, correlates_with, etc.

    # Relationship characteristics
    relationship_strength: Optional[float] = None  # 0-1 scale
    relationship_direction: Optional[str] = None   # positive, negative, neutral
    relationship_lag: Optional[timedelta] = None   # Time lag between indicators

    # Empirical validation
    correlation_coefficient: Optional[float] = None
    statistical_significance: Optional[float] = None
    validation_method: Optional[str] = None

    # Contextual factors
    relationship_context: Optional[str] = None
    conditional_factors: List[str] = field(default_factory=list)
    relationship_stability: Optional[float] = None  # How stable over time

    # Causal inference
    causal_evidence: List[str] = field(default_factory=list)
    causal_mechanism: Optional[str] = None
    confounding_factors: List[str] = field(default_factory=list)

    def validate_relationship(self) -> Dict[str, Any]:
        """Validate the indicator relationship."""
        validation_results = {
            'statistical_validity': 0.0,
            'theoretical_validity': 0.0,
            'empirical_support': 0.0,
            'overall_validity': 0.0,
            'validation_issues': [],
            'recommendations': []
        }

        # Statistical validity
        if self.correlation_coefficient is not None and self.statistical_significance is not None:
            if abs(self.correlation_coefficient) > 0.3 and self.statistical_significance < 0.05:
                validation_results['statistical_validity'] = 0.8
            elif abs(self.correlation_coefficient) > 0.2:
                validation_results['statistical_validity'] = 0.5
            else:
                validation_results['statistical_validity'] = 0.2
                validation_results['validation_issues'].append('Weak statistical relationship')

        # Theoretical validity
        if self.causal_mechanism:
            validation_results['theoretical_validity'] = 0.7
        if self.causal_evidence:
            validation_results['theoretical_validity'] += 0.3
        validation_results['theoretical_validity'] = min(
            validation_results['theoretical_validity'],
            1.0)

        # Empirical support
        if self.relationship_strength is not None:
            validation_results['empirical_support'] = self.relationship_strength

        # Overall validity
        validity_scores = [
            validation_results['statistical_validity'],
            validation_results['theoretical_validity'],
            validation_results['empirical_support']
        ]
        valid_scores = [score for score in validity_scores if score > 0]
        if valid_scores:
            validation_results['overall_validity'] = sum(valid_scores) / len(valid_scores)

        # Generate recommendations
        if validation_results['statistical_validity'] < 0.5:
            validation_results['recommendations'].append('Strengthen statistical validation')
        if validation_results['theoretical_validity'] < 0.5:
            validation_results['recommendations'].append('Develop theoretical justification')

        return validation_results

@dataclass
class PerformanceMeasurement(Node):
    """Systematic performance measurement framework."""

    measurement_scope: Optional[str] = None
    measurement_purpose: Optional[str] = None

    # Measurement framework
    measured_indicators: List[uuid.UUID] = field(default_factory=list)
    measurement_methods: List[StatisticalMethod] = field(default_factory=list)
    measurement_schedule: Dict[uuid.UUID, IndicatorFrequency] = field(default_factory=dict)

    # Performance standards
    performance_targets: Dict[uuid.UUID, float] = field(default_factory=dict)
    performance_thresholds: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    benchmark_comparisons: Dict[uuid.UUID, float] = field(default_factory=dict)

    # Measurement results
    current_performance: Dict[uuid.UUID, float] = field(default_factory=dict)
    performance_trends: Dict[uuid.UUID, str] = field(default_factory=dict)
    variance_analysis: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)

    # Aggregate measures
    composite_scores: Dict[str, float] = field(default_factory=dict)
    performance_categories: Dict[str, List[uuid.UUID]] = field(default_factory=dict)
    overall_performance_score: Optional[float] = None

    # Quality assurance
    measurement_reliability: Dict[uuid.UUID, float] = field(default_factory=dict)
    measurement_validity: Dict[uuid.UUID, float] = field(default_factory=dict)
    data_quality_scores: Dict[uuid.UUID, float] = field(default_factory=dict)

    # Reporting
    reporting_frequency: IndicatorFrequency = IndicatorFrequency.QUARTERLY
    report_audiences: List[uuid.UUID] = field(default_factory=list)
    dashboard_indicators: List[uuid.UUID] = field(default_factory=list)

    def calculate_composite_performance(self) -> Dict[str, float]:
        """Calculate composite performance scores."""
        composite_results = {}

        # Overall performance score
        if self.current_performance:
            performance_values = list(self.current_performance.values())
            composite_results['overall_performance'] = sum(performance_values) / len(performance_values)
            self.overall_performance_score = composite_results['overall_performance']

        # Category-based composite scores
        for category, indicator_ids in self.performance_categories.items():
            category_scores = [self.current_performance.get(
                ind_id,
                0.0) for ind_id in indicator_ids]
            if category_scores:
                composite_results[category] = sum(category_scores) / len(category_scores)

        self.composite_scores.update(composite_results)
        return composite_results

    def analyze_performance_variance(self) -> Dict[str, Any]:
        """Analyze variance in performance across indicators."""
        variance_analysis = {
            'performance_spread': 0.0,
            'high_performers': [],
            'low_performers': [],
            'variance_drivers': [],
            'consistency_score': 0.0
        }

        if not self.current_performance:
            return variance_analysis

        performance_values = list(self.current_performance.values())
        performance_mean = sum(performance_values) / len(performance_values)

        # Calculate performance spread
        if len(performance_values) > 1:
            performance_std = statistics.stdev(performance_values)
            variance_analysis['performance_spread'] = performance_std

            # Consistency score (lower variance = higher consistency)
            if performance_mean > 0:
                variance_analysis['consistency_score'] = 1.0 - (performance_std / performance_mean)

        # Identify high and low performers
        for indicator_id, performance in self.current_performance.items():
            if performance > performance_mean + 0.2:  # Significantly above average
                variance_analysis['high_performers'].append(indicator_id)
            elif performance < performance_mean - 0.2:  # Significantly below average
                variance_analysis['low_performers'].append(indicator_id)

        return variance_analysis

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        performance_report = {
            'executive_summary': {},
            'indicator_performance': {},
            'trend_analysis': {},
            'variance_analysis': {},
            'recommendations': []
        }

        # Executive summary
        composite_scores = self.calculate_composite_performance()
        performance_report['executive_summary'] = {
            'overall_performance': composite_scores.get('overall_performance', 0.0),
            'total_indicators': len(self.measured_indicators),
            'performance_level': self._categorize_performance(
                composite_scores.get('overall_performance',
                0.0))
        }

        # Individual indicator performance
        for indicator_id, performance in self.current_performance.items():
            target = self.performance_targets.get(indicator_id)
            performance_report['indicator_performance'][str(indicator_id)] = {
                'current_performance': performance,
                'target_performance': target,
                'target_achievement': (performance / target) if target and target > 0 else None,
                'trend': self.performance_trends.get(indicator_id, 'unknown')
            }

        # Variance analysis
        performance_report['variance_analysis'] = self.analyze_performance_variance()

        # Generate recommendations
        variance_results = performance_report['variance_analysis']
        if len(variance_results['low_performers']) > len(self.measured_indicators) * 0.3:
            performance_report['recommendations'].append('Focus on improving consistently low-performing indicators')
        if variance_results['consistency_score'] < 0.6:
            performance_report['recommendations'].append('Address high variance in performance across indicators')

        return performance_report

    def _categorize_performance(self, score: float) -> str:
        """Categorize performance level based on score."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "very_good"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "satisfactory"
        elif score >= 0.5:
            return "needs_improvement"
        else:
            return "poor"

@dataclass
class IndicatorSystem(Node):
    """Comprehensive indicator system for SFM analysis."""

    system_scope: Optional[str] = None
    system_purpose: Optional[str] = None

    # System components
    indicators: List[uuid.UUID] = field(default_factory=list)           # SocialFabricIndicator IDs
    indicator_relationships: List[uuid.UUID] = field(default_factory=list) # IndicatorRelationship IDs
    performance_measurements: List[uuid.UUID] = field(default_factory=list) # PerformanceMeasurement IDs

    # System structure
    indicator_hierarchy: Dict[str, List[uuid.UUID]] = field(default_factory=dict)  # Level -> indicators
    indicator_categories: Dict[IndicatorCategory, List[uuid.UUID]] = field(default_factory=dict)
    value_category_mapping: Dict[ValueCategory, List[uuid.UUID]] = field(default_factory=dict)

    # System properties
    system_completeness: Optional[float] = None     # Coverage completeness (0-1)
    system_coherence: Optional[float] = None        # Internal consistency (0-1)
    system_utility: Optional[float] = None          # Usefulness for decision-making (0-1)
    system_sustainability: Optional[float] = None   # Long-term maintainability (0-1)

    # Integration with SFM
    matrix_cell_coverage: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)  # Cell -> indicators
    delivery_system_indicators: List[uuid.UUID] = field(default_factory=list)
    institutional_performance_indicators: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)

    # System governance
    indicator_stewards: Dict[uuid.UUID, uuid.UUID] = field(default_factory=dict)  # Indicator -> responsible actor
    data_governance_framework: List[str] = field(default_factory=list)
    quality_assurance_processes: List[str] = field(default_factory=list)

    # System evolution
    indicator_lifecycle_management: Dict[str, List[str]] = field(default_factory=dict)
    system_review_schedule: Optional[str] = None
    adaptation_mechanisms: List[str] = field(default_factory=list)

    def assess_system_completeness(self) -> Dict[str, Any]:
        """Assess completeness of the indicator system."""
        completeness_assessment = {
            'value_category_coverage': {},
            'institutional_coverage': 0.0,
            'process_coverage': 0.0,
            'outcome_coverage': 0.0,
            'overall_completeness': 0.0,
            'coverage_gaps': [],
            'recommendations': []
        }

        # Value category coverage
        total_value_categories = len(ValueCategory)
        covered_categories = len(self.value_category_mapping)
        completeness_assessment['value_category_coverage'] = {
            'covered_categories': covered_categories,
            'total_categories': total_value_categories,
            'coverage_ratio': covered_categories / total_value_categories
        }

        # Indicator category coverage
        category_coverage = {}
        for category in IndicatorCategory:
            category_indicators = self.indicator_categories.get(category, [])
            category_coverage[category.name] = len(category_indicators)

        # Assess coverage adequacy
        outcome_indicators = len(
            self.indicator_categories.get(IndicatorCategory.OUTCOME_INDICATOR,
            []))
        process_indicators = len(
            self.indicator_categories.get(IndicatorCategory.PROCESS_INDICATOR,
            []))

        completeness_assessment['outcome_coverage'] = min(
            outcome_indicators / 10.0,
            1.0)  # Assume 10 outcomes needed
        completeness_assessment['process_coverage'] = min(
            process_indicators / 15.0,
            1.0)  # Assume 15 processes

        # Overall completeness
        completeness_factors = [
            completeness_assessment['value_category_coverage']['coverage_ratio'],
            completeness_assessment['outcome_coverage'],
            completeness_assessment['process_coverage']
        ]
        completeness_assessment['overall_completeness'] = sum(completeness_factors) / len(completeness_factors)
        self.system_completeness = completeness_assessment['overall_completeness']

        # Identify gaps
        if completeness_assessment['outcome_coverage'] < 0.7:
            completeness_assessment['coverage_gaps'].append('Insufficient outcome indicators')
        if completeness_assessment['process_coverage'] < 0.6:
            completeness_assessment['coverage_gaps'].append('Insufficient process indicators')

        return completeness_assessment

    def validate_system_coherence(self) -> Dict[str, Any]:
        """Validate internal coherence of the indicator system."""
        coherence_assessment = {
            'relationship_consistency': 0.0,
            'hierarchy_alignment': 0.0,
            'value_alignment': 0.0,
            'overall_coherence': 0.0,
            'coherence_issues': [],
            'improvement_recommendations': []
        }

        # Relationship consistency
        if self.indicator_relationships:
            # Simplified assessment - would need detailed relationship validation
            coherence_assessment['relationship_consistency'] = 0.7  # Placeholder

        # Hierarchy alignment
        if self.indicator_hierarchy:
            # Check if indicators are appropriately categorized
            coherence_assessment['hierarchy_alignment'] = 0.8  # Placeholder

        # Value alignment
        value_coverage = len(self.value_category_mapping) / len(ValueCategory)
        coherence_assessment['value_alignment'] = value_coverage

        # Overall coherence
        coherence_factors = [
            coherence_assessment['relationship_consistency'],
            coherence_assessment['hierarchy_alignment'],
            coherence_assessment['value_alignment']
        ]
        valid_factors = [f for f in coherence_factors if f > 0]
        if valid_factors:
            coherence_assessment['overall_coherence'] = sum(valid_factors) / len(valid_factors)
            self.system_coherence = coherence_assessment['overall_coherence']

        # Generate recommendations
        if coherence_assessment['overall_coherence'] < 0.7:
            coherence_assessment['improvement_recommendations'].append('Strengthen system coherence')

        return coherence_assessment

    def generate_system_dashboard(self) -> Dict[str, Any]:
        """Generate system-level dashboard view."""
        dashboard = {
            'system_overview': {
                'total_indicators': len(self.indicators),
                'system_completeness': self.system_completeness or 0.0,
                'system_coherence': self.system_coherence or 0.0,
                'system_health': 'unknown'
            },
            'key_performance_indicators': [],
            'system_alerts': [],
            'coverage_summary': {},
            'recent_updates': []
        }

        # Assess system health
        if (self.system_completeness and self.system_coherence and
            self.system_completeness > 0.8 and self.system_coherence > 0.8):
            dashboard['system_overview']['system_health'] = 'excellent'
        elif (self.system_completeness and self.system_coherence and
              self.system_completeness > 0.6 and self.system_coherence > 0.6):
            dashboard['system_overview']['system_health'] = 'good'
        else:
            dashboard['system_overview']['system_health'] = 'needs_attention'

        # Coverage summary
        dashboard['coverage_summary'] = {
            'value_categories': len(self.value_category_mapping),
            'indicator_categories': len(self.indicator_categories),
            'matrix_cells_covered': len(self.matrix_cell_coverage)
        }

        # System alerts
        if self.system_completeness and self.system_completeness < 0.6:
            dashboard['system_alerts'].append('Low system completeness - missing indicators')
        if self.system_coherence and self.system_coherence < 0.6:
            dashboard['system_alerts'].append('Low system coherence - review indicator relationships')

        return dashboard
