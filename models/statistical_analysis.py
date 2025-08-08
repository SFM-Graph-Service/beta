"""
Statistical analysis tools and algorithms for Social Indicators.

This module provides statistical analysis capabilities that were previously
embedded within the social_indicators module. It separates the concerns of
data management from statistical computation.

Key Components:
- Statistical computation methods (correlation, trend analysis, volatility)
- Aggregation methods and composite indicator creation
- Time series analysis and prediction
- Data quality and validation statistics
"""

from __future__ import annotations

import uuid
import statistics
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto

# Type hints for forward references
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.social_indicators import SocialIndicator, IndicatorDatabase, IndicatorMeasurement

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
class StatisticalAnalysisTools:
    """Core statistical analysis tools for social indicators."""

    def calculate_correlation(self, data1: List[float], data2: List[float]) -> float:
        """Calculate Pearson correlation coefficient between two datasets."""
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

    def calculate_trend_direction(self, values: List[float]) -> TrendDirection:
        """Calculate trend direction using linear regression slope."""
        if len(values) < 2:
            return TrendDirection.UNKNOWN

        # Simple linear trend calculation
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return TrendDirection.STABLE

        slope = numerator / denominator

        # Determine trend based on slope and significance
        slope_threshold = 0.01  # Adjust based on indicator scale

        if abs(slope) < slope_threshold:
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING

    def calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility measure using coefficient of variation."""
        if len(values) < 3:
            return 0.0

        clean_values = [v for v in values if v is not None]
        if not clean_values:
            return 0.0

        mean_value = statistics.mean(clean_values)
        if mean_value == 0:
            return 0.0

        std_dev = statistics.stdev(clean_values)
        volatility = abs(std_dev / mean_value)

        return volatility

    def aggregate_values(self, values: List[float], weights: Optional[List[float]] = None,
                        method: AggregationMethod = AggregationMethod.MEAN) -> float:
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
            if weights is None or len(weights) != len(values):
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

    def calculate_trend_statistics(self, values: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive trend statistics."""
        if len(values) < 3:
            return {'error': 'Insufficient data for trend analysis'}

        clean_values = [v for v in values if v is not None]
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
            'trend_direction': self.calculate_trend_direction(clean_values).name,
            'volatility': self.calculate_volatility(clean_values),
            'mean': mean_value,
            'median': median_value,
            'standard_deviation': std_dev,
            'growth_rate_percent': growth_rate,
            'coefficient_of_variation': std_dev / mean_value if mean_value != 0 else 0,
            'data_points': len(values)
        }

@dataclass
class IndicatorAnalyzer:
    """High-level analyzer for working with social indicators."""
    
    tools: StatisticalAnalysisTools = field(default_factory=StatisticalAnalysisTools)

    def analyze_indicator_correlations(self, database: 'IndicatorDatabase', 
                                     indicator_ids: List[uuid.UUID],
                                     time_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, float]:
        """Analyze correlations between multiple indicators."""
        correlations = {}

        if database is None:
            return correlations

        indicators = [database.indicators[iid] for iid in indicator_ids
                     if iid in database.indicators]

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
                    correlation = self.tools.calculate_correlation(
                        data1[-min_length:], data2[-min_length:]
                    )
                    correlations[f"{ind1}_{ind2}"] = correlation

        return correlations

    def perform_comprehensive_trend_analysis(self, indicator: 'SocialIndicator',
                                           periods: int = 12) -> Dict[str, Any]:
        """Perform comprehensive trend analysis on a single indicator."""
        measurements = indicator.measurements[-periods:]

        numeric_values = [m.get_numeric_value() for m in measurements
                         if m.get_numeric_value() is not None]

        if len(numeric_values) < 3:
            return {'error': 'Insufficient data for trend analysis'}

        # Use statistical tools for comprehensive analysis
        trend_stats = self.tools.calculate_trend_statistics(numeric_values)
        trend_stats['analysis_period'] = periods
        
        return trend_stats

    def create_composite_indicator(self, database: 'IndicatorDatabase',
                                 component_indicators: List[uuid.UUID],
                                 weights: Optional[List[float]] = None,
                                 aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE) -> 'SocialIndicator':
        """Create a composite indicator from multiple component indicators."""
        # Import here to avoid circular import
        from models.social_indicators import SocialIndicator, IndicatorMeasurement
        from models.sfm_enums import IndicatorType

        if not component_indicators:
            raise ValueError("No component indicators provided")

        if weights and len(weights) != len(component_indicators):
            raise ValueError("Number of weights must match number of indicators")

        if not weights:
            weights = [1.0 / len(component_indicators)] * len(component_indicators)
        
        if database is None:
            raise ValueError("Database not available for composite indicator creation")

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
            if ind_id in database.indicators:
                indicator = database.indicators[ind_id]
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
                composite_value = self.tools.aggregate_values(
                    values,
                    valid_weights,
                    aggregation_method)

                measurement = IndicatorMeasurement(
                    value=composite_value,
                    timestamp=timestamp,
                    confidence_level=min(
                        len(values) / len(component_indicators),
                        1.0)
                )
                composite.add_measurement(measurement)

        return composite

@dataclass  
class MatrixStatisticalAnalyzer:
    """Specialized analyzer for SFM matrix-indicator statistical relationships."""
    
    tools: StatisticalAnalysisTools = field(default_factory=StatisticalAnalysisTools)

    def analyze_matrix_coverage_statistics(self, database: 'IndicatorDatabase') -> Dict[str, Any]:
        """Analyze statistical metrics for matrix coverage."""
        coverage_stats = {
            'total_cells_covered': len(database.matrix_coverage),
            'coverage_distribution': {},
            'coverage_quality_statistics': {}
        }

        # Analyze coverage distribution
        if database.matrix_coverage:
            indicator_counts = [len(indicators) for indicators in database.matrix_coverage.values()]
            coverage_stats['coverage_distribution'] = {
                'mean_indicators_per_cell': statistics.mean(indicator_counts),
                'median_indicators_per_cell': statistics.median(indicator_counts),
                'std_dev_indicators_per_cell': statistics.stdev(indicator_counts) if len(indicator_counts) > 1 else 0,
                'max_indicators_per_cell': max(indicator_counts),
                'min_indicators_per_cell': min(indicator_counts),
            }

        return coverage_stats

    def calculate_integration_strength_statistics(self, database: 'IndicatorDatabase') -> Dict[str, Any]:
        """Calculate statistical measures of matrix integration strength."""
        all_strengths = []
        
        for indicator in database.indicators.values():
            for strength in indicator.cell_indicator_strength.values():
                all_strengths.append(strength)

        if not all_strengths:
            return {'error': 'No integration strength data available'}

        return {
            'mean_strength': statistics.mean(all_strengths),
            'median_strength': statistics.median(all_strengths),
            'std_dev_strength': statistics.stdev(all_strengths) if len(all_strengths) > 1 else 0,
            'min_strength': min(all_strengths),
            'max_strength': max(all_strengths),
            'total_relationships': len(all_strengths),
            'strong_relationships': len([s for s in all_strengths if s > 0.7]),
            'weak_relationships': len([s for s in all_strengths if s < 0.3])
        }

# Factory function to create analyzers
def create_statistical_analyzer(database: Optional['IndicatorDatabase'] = None) -> IndicatorAnalyzer:
    """Create a statistical analyzer instance."""
    return IndicatorAnalyzer()

def create_matrix_analyzer() -> MatrixStatisticalAnalyzer:
    """Create a matrix statistical analyzer instance."""
    return MatrixStatisticalAnalyzer()