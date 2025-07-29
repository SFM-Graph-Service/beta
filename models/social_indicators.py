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

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import statistics
import math

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.metadata_models import TemporalDynamics, ValidationRule
from models.sfm_enums import (
    IndicatorType,
    StatisticalMethod,
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
    contextual_factors: Dict[str, Any] = field(default_factory=dict)
    measurement_conditions: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    notes: str = ""
    
    def get_numeric_value(self) -> Optional[float]:
        """Get numeric representation of the value."""
        if isinstance(self.value, (int, float)):
            return float(self.value)
        elif isinstance(self.value, bool):
            return 1.0 if self.value else 0.0
        elif isinstance(self.value, str):
            try:
                return float(self.value)
            except ValueError:
                return None
        return None
    
    def is_valid(self) -> bool:
        """Check if measurement is valid."""
        return (self.value is not None and 
                self.confidence_level > 0 and
                self.timestamp is not None)
    
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
    sfm_indicator_type: SocialFabricIndicatorType = SocialFabricIndicatorType.INSTITUTIONAL_PERFORMANCE
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
    
    # SFM context
    related_matrix_cells: List[uuid.UUID] = field(default_factory=list)
    affecting_institutions: List[uuid.UUID] = field(default_factory=list)
    policy_relevance: List[uuid.UUID] = field(default_factory=list)  # Related policies
    
    # Validation and quality
    validation_rules: List[ValidationRule] = field(default_factory=list)
    evidence_quality: EvidenceQuality = EvidenceQuality.MEDIUM
    
    def add_measurement(self, measurement: IndicatorMeasurement) -> None:
        """Add a new measurement to the indicator."""
        self.measurements.append(measurement)
        self.measurements.sort(key=lambda m: m.timestamp)
        
        # Update current value
        if not self.current_timestamp or measurement.timestamp > self.current_timestamp:
            self.current_value = measurement.value
            self.current_timestamp = measurement.timestamp
    
    def get_measurements_in_period(self, start_date: datetime, end_date: datetime) -> List[IndicatorMeasurement]:
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
        y_mean = sum(numeric_values) / n
        
        numerator = sum((i - x_mean) * (numeric_values[i] - y_mean) for i in range(n))
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
        mean_value = statistics.mean(numeric_values)
        if mean_value == 0:
            return 0.0
        
        std_dev = statistics.stdev(numeric_values)
        volatility = std_dev / abs(mean_value)
        
        self.volatility_measure = volatility
        return volatility
    
    def get_current_status(self) -> Dict[str, Any]:
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
    
    def validate_measurement(self, measurement: IndicatorMeasurement) -> List[str]:
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


@dataclass
class IndicatorDatabase(Node):
    """Database system for managing collections of social indicators."""
    
    indicators: Dict[uuid.UUID, SocialIndicator] = field(default_factory=dict)
    indicator_groups: Dict[str, List[uuid.UUID]] = field(default_factory=dict)
    
    # Database metadata
    last_updated: Optional[datetime] = None
    update_frequency: Optional[timedelta] = None
    data_sources: List[str] = field(default_factory=list)
    
    # Quality metrics
    overall_completeness: Optional[float] = None
    average_quality_score: Optional[float] = None
    data_coverage: Dict[str, float] = field(default_factory=dict)
    
    # SFM integration
    matrix_coverage: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)  # Cell -> Indicators
    institutional_coverage: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)
    
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
                expected_measurements = max(1, days_since_creation // indicator.measurement_frequency.days)
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
                indicator_quality = statistics.mean(
                    m.calculate_quality_score() for m in indicator.measurements
                )
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
            'indicator_groups': {group: len(indicators) for group, indicators in self.indicator_groups.items()},
            'completeness': self.calculate_completeness(),
            'quality_score': self.calculate_quality_score(),
            'last_updated': self.last_updated,
            'coverage_by_type': self._calculate_type_coverage(),
            'coverage_by_category': self._calculate_category_coverage()
        }
    
    def _calculate_type_coverage(self) -> Dict[str, int]:
        """Calculate coverage by indicator type."""
        type_counts = {}
        for indicator in self.indicators.values():
            type_name = indicator.indicator_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts
    
    def _calculate_category_coverage(self) -> Dict[str, int]:
        """Calculate coverage by value category."""
        category_counts = {}
        for indicator in self.indicators.values():
            category_name = indicator.value_category.name
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
        return category_counts


@dataclass
class StatisticalAnalyzer:
    """Tools for statistical analysis of indicator data."""
    
    database: IndicatorDatabase
    
    def analyze_correlations(self, indicator_ids: List[uuid.UUID], 
                           time_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, float]:
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
        indicator_list = list(indicator_data.keys())
        for i, ind1 in enumerate(indicator_list):
            for ind2 in indicator_list[i+1:]:
                data1 = indicator_data[ind1]
                data2 = indicator_data[ind2]
                
                # Align data by length (simple approach)
                min_length = min(len(data1), len(data2))
                if min_length > 1:
                    correlation = self._calculate_correlation(
                        data1[-min_length:], data2[-min_length:]
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
        mean_value = statistics.mean(numeric_values)
        median_value = statistics.median(numeric_values)
        std_dev = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
        
        # Calculate growth rate (if applicable)
        growth_rate = None
        if len(numeric_values) >= 2 and numeric_values[0] != 0:
            growth_rate = ((numeric_values[-1] - numeric_values[0]) / numeric_values[0]) * 100
        
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
        for timestamp in sorted(all_timestamps):
            values = []
            valid_weights = []
            
            for i, ind_id in enumerate(component_indicators):
                if (ind_id in component_data and 
                    timestamp in component_data[ind_id] and 
                    component_data[ind_id][timestamp] is not None):
                    
                    values.append(component_data[ind_id][timestamp])
                    valid_weights.append(weights[i])
            
            if values:
                composite_value = self._aggregate_values(values, valid_weights, aggregation_method)
                
                measurement = IndicatorMeasurement(
                    value=composite_value,
                    timestamp=timestamp,
                    confidence_level=min(len(values) / len(component_indicators), 1.0)
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
    monitored_indicators: List[uuid.UUID] = field(default_factory=list)
    alert_thresholds: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    
    # Dashboard state
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    last_refresh: Optional[datetime] = None
    refresh_frequency: timedelta = timedelta(hours=1)
    
    def add_monitored_indicator(self, indicator_id: uuid.UUID, 
                               thresholds: Optional[Dict[str, float]] = None) -> None:
        """Add an indicator to be monitored on the dashboard."""
        if indicator_id not in self.monitored_indicators:
            self.monitored_indicators.append(indicator_id)
        
        if thresholds:
            self.alert_thresholds[indicator_id] = thresholds
    
    def check_alerts(self) -> List[Dict[str, Any]]:
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
    
    def generate_summary_dashboard(self) -> Dict[str, Any]:
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