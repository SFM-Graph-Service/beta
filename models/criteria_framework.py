"""
Criteria Framework for systematic evaluation standards in SFM analysis.

This module implements comprehensive criteria frameworks for evaluating
institutional performance, delivery effectiveness, and value realization
within the Social Fabric Matrix. It provides structured approaches to
defining, applying, and validating evaluation criteria.

Key Components:
- CriteriaFramework: Comprehensive evaluation criteria system
- EvaluationCriterion: Individual evaluation criteria
- CriteriaApplication: Application of criteria to specific contexts
- CriteriaValidation: Validation and quality assurance of criteria
- MultiCriteriaAnalysis: Multi-criteria decision analysis tools
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from datetime import datetime
from enum import Enum, auto
import statistics

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.sfm_enums import (
    CriteriaType,
    ValueCategory,
    CriteriaPriority,
    MeasurementApproach,
    EvidenceQuality,
    ValidationMethod,
    SystemLevel,
)

class CriteriaScope(Enum):
    """Scope of criteria application."""

    INSTITUTIONAL_PERFORMANCE = auto()   # Institution-level evaluation
    DELIVERY_EFFECTIVENESS = auto()      # Delivery system evaluation
    VALUE_REALIZATION = auto()          # Value achievement evaluation
    POLICY_EFFECTIVENESS = auto()       # Policy performance evaluation
    SYSTEM_PERFORMANCE = auto()         # System-level evaluation
    PROCESS_QUALITY = auto()            # Process evaluation
    OUTCOME_ACHIEVEMENT = auto()        # Outcome-focused evaluation

class CriteriaWeightingMethod(Enum):
    """Methods for weighting criteria."""

    EQUAL_WEIGHTING = auto()            # All criteria equally weighted
    EXPERT_JUDGMENT = auto()            # Expert-determined weights
    STAKEHOLDER_PREFERENCE = auto()     # Stakeholder-determined weights
    ANALYTICAL_HIERARCHY = auto()       # AHP-based weighting
    EMPIRICAL_VALIDATION = auto()       # Data-driven weighting
    CONTEXTUAL_ADAPTATION = auto()      # Context-specific weighting

class AggregationMethod(Enum):
    """Methods for aggregating criteria scores."""

    WEIGHTED_AVERAGE = auto()           # Simple weighted average
    GEOMETRIC_MEAN = auto()             # Geometric mean aggregation
    MIN_MAX_NORMALIZATION = auto()      # Min-max normalized aggregation
    FUZZY_AGGREGATION = auto()          # Fuzzy logic aggregation
    MULTI_ATTRIBUTE_UTILITY = auto()    # Multi-attribute utility theory
    OUTRANKING = auto()                 # Outranking methods

class CriteriaValidityType(Enum):
    """Types of criteria validity."""

    CONTENT_VALIDITY = auto()           # Content appropriateness
    CONSTRUCT_VALIDITY = auto()         # Construct measurement accuracy
    CRITERION_VALIDITY = auto()         # Predictive validity
    FACE_VALIDITY = auto()              # Apparent appropriateness
    CONVERGENT_VALIDITY = auto()        # Agreement with related measures
    DISCRIMINANT_VALIDITY = auto()      # Distinction from unrelated measures

@dataclass
class EvaluationCriterion(Node):
    """Individual evaluation criterion within a criteria framework."""

    criteria_type: CriteriaType = CriteriaType.SOCIAL
    criteria_scope: CriteriaScope = CriteriaScope.INSTITUTIONAL_PERFORMANCE
    value_category: ValueCategory = ValueCategory.SOCIAL

    # Criterion definition
    criterion_description: Optional[str] = None
    measurement_definition: Optional[str] = None
    evaluation_standards: Dict[str, Any] = field(default_factory=dict)

    # Measurement specifications
    measurement_approach: MeasurementApproach = MeasurementApproach.QUANTITATIVE
    measurement_unit: Optional[str] = None
    measurement_scale: Optional[Tuple[float, float]] = None
    target_values: Dict[str, float] = field(default_factory=dict)  # Performance targets

    # Criterion importance
    priority_level: CriteriaPriority = CriteriaPriority.SECONDARY
    weight: Optional[float] = None  # Relative weight in framework (0-1)
    importance_justification: Optional[str] = None

    # Application context
    applicable_contexts: List[str] = field(default_factory=list)
    institutional_applicability: List[uuid.UUID] = field(default_factory=list)
    temporal_applicability: Optional[TimeSlice] = None
    spatial_applicability: Optional[SpatialUnit] = None

    # Measurement data
    current_scores: Dict[uuid.UUID, float] = field(default_factory=dict)  # Entity -> score
    historical_scores: Dict[uuid.UUID, List[Tuple[datetime, float]]] = field(default_factory=dict)
    benchmark_scores: Dict[str, float] = field(default_factory=dict)

    # Relationships
    related_criteria: List[uuid.UUID] = field(default_factory=list)
    supporting_indicators: List[uuid.UUID] = field(default_factory=list)
    conflicting_criteria: List[uuid.UUID] = field(default_factory=list)

    # Validation
    validity_evidence: Dict[CriteriaValidityType, str] = field(default_factory=dict)
    validation_methods: List[ValidationMethod] = field(default_factory=list)
    validation_confidence: Optional[float] = None  # 0-1 scale

    # Quality assurance
    data_quality: EvidenceQuality = EvidenceQuality.MEDIUM
    measurement_reliability: Optional[float] = None  # 0-1 scale
    inter_rater_reliability: Optional[float] = None  # For subjective criteria

    def calculate_performance_score(self, entity_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Calculate performance score for an entity against this criterion."""
        if entity_id not in self.current_scores:
            return None

        current_score = self.current_scores[entity_id]
        performance_analysis = {
            'raw_score': current_score,
            'normalized_score': 0.0,
            'performance_level': 'unknown',
            'target_achievement': 0.0,
            'benchmark_comparison': {},
            'trend_analysis': {}
        }

        # Normalize score to 0-1 scale if measurement scale is defined
        if self.measurement_scale:
            min_val, max_val = self.measurement_scale
            if max_val > min_val:
                normalized = (current_score - min_val) / (max_val - min_val)
                performance_analysis['normalized_score'] = max(0.0, min(1.0, normalized))
        else:
            performance_analysis['normalized_score'] = current_score

        # Performance level categorization
        normalized_score = performance_analysis['normalized_score']
        if normalized_score >= 0.9:
            performance_analysis['performance_level'] = 'excellent'
        elif normalized_score >= 0.8:
            performance_analysis['performance_level'] = 'very_good'
        elif normalized_score >= 0.7:
            performance_analysis['performance_level'] = 'good'
        elif normalized_score >= 0.6:
            performance_analysis['performance_level'] = 'satisfactory'
        elif normalized_score >= 0.5:
            performance_analysis['performance_level'] = 'needs_improvement'
        else:
            performance_analysis['performance_level'] = 'poor'

        # Target achievement analysis
        target_value = self.target_values.get('standard', None)
        if target_value and target_value > 0:
            performance_analysis['target_achievement'] = current_score / target_value

        # Benchmark comparison
        for benchmark_name, benchmark_value in self.benchmark_scores.items():
            if benchmark_value > 0:
                comparison_ratio = current_score / benchmark_value
                performance_analysis['benchmark_comparison'][benchmark_name] = comparison_ratio

        # Trend analysis
        if entity_id in self.historical_scores:
            historical_data = self.historical_scores[entity_id]
            if len(historical_data) >= 2:
                values = [score for _, score in historical_data]
                if len(values) >= 3:
                    recent_trend = 'improving' if values[-1] > values[-2] else 'declining' if values[-1] < values[-2] else 'stable'
                    performance_analysis['trend_analysis'] = {
                        'recent_trend': recent_trend,
                        'trend_strength': abs(values[-1] - values[-2]) / max(abs(values[-2]), 0.1)
                    }

        return performance_analysis

    def validate_criterion_quality(self) -> Dict[str, Any]:
        """Validate quality and appropriateness of the criterion."""
        validation_results = {
            'content_validity': 0.0,
            'measurement_validity': 0.0,
            'reliability_assessment': 0.0,
            'practical_utility': 0.0,
            'overall_quality': 0.0,
            'quality_issues': [],
            'improvement_recommendations': []
        }

        # Assess individual quality dimensions
        validation_results['content_validity'] = self._assess_content_validity(validation_results['quality_issues'])
        validation_results['measurement_validity'] = self._assess_measurement_validity()
        validation_results['reliability_assessment'] = self._assess_reliability(validation_results['quality_issues'])
        validation_results['practical_utility'] = self._assess_practical_utility()

        # Calculate overall quality
        validation_results['overall_quality'] = self._calculate_overall_quality(validation_results)

        # Generate improvement recommendations
        validation_results['improvement_recommendations'] = self._generate_quality_recommendations(validation_results)

        return validation_results

    def _assess_content_validity(self, quality_issues: List[str]) -> float:
        """Assess content validity of the criterion."""
        if self.criterion_description and self.measurement_definition:
            return 0.8
        elif self.criterion_description or self.measurement_definition:
            return 0.5
        else:
            quality_issues.append('Missing criterion or measurement definition')
            return 0.2

    def _assess_measurement_validity(self) -> float:
        """Assess measurement validity of the criterion."""
        validity_score = 0.0
        if self.measurement_approach and self.measurement_unit:
            validity_score += 0.4
        if self.target_values:
            validity_score += 0.3
        if self.benchmark_scores:
            validity_score += 0.3
        return min(validity_score, 1.0)

    def _assess_reliability(self, quality_issues: List[str]) -> float:
        """Assess reliability of the criterion."""
        if self.measurement_reliability is not None:
            return self.measurement_reliability
        elif self.inter_rater_reliability is not None:
            return self.inter_rater_reliability
        else:
            quality_issues.append('Reliability not assessed')
            return 0.3  # Unknown reliability

    def _assess_practical_utility(self) -> float:
        """Assess practical utility of the criterion."""
        utility_factors = []
        if self.applicable_contexts:
            utility_factors.append(0.3)
        if self.current_scores:
            utility_factors.append(0.4)
        if self.supporting_indicators:
            utility_factors.append(0.3)
        return sum(utility_factors)

    def _calculate_overall_quality(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        quality_components = [
            validation_results['content_validity'],
            validation_results['measurement_validity'],
            validation_results['reliability_assessment'],
            validation_results['practical_utility']
        ]
        return sum(quality_components) / len(quality_components)

    def _generate_quality_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on assessment."""
        recommendations = []

        if validation_results['content_validity'] < 0.7:
            recommendations.append('Improve criterion definition and clarity')
        if validation_results['measurement_validity'] < 0.6:
            recommendations.append('Strengthen measurement approach and standards')
        if validation_results['reliability_assessment'] < 0.7:
            recommendations.append('Conduct reliability assessment')

        return recommendations

@dataclass
class CriteriaApplication(Node):
    """Application of criteria framework to specific evaluation context."""

    application_context: Optional[str] = None
    evaluation_purpose: Optional[str] = None

    # Application scope
    evaluated_entities: List[uuid.UUID] = field(default_factory=list)  # What is being evaluated
    applied_criteria: List[uuid.UUID] = field(default_factory=list)    # Which criteria applied
    evaluation_timeframe: Optional[TimeSlice] = None

    # Criteria weighting
    weighting_method: CriteriaWeightingMethod = CriteriaWeightingMethod.EQUAL_WEIGHTING
    criteria_weights: Dict[uuid.UUID, float] = field(default_factory=dict)
    weight_rationale: Dict[uuid.UUID, str] = field(default_factory=dict)

    # Evaluation results
    entity_scores: Dict[uuid.UUID, Dict[uuid.UUID, float]] = field(default_factory=dict)  # Entity -> Criterion -> Score
    aggregated_scores: Dict[uuid.UUID, float] = field(default_factory=dict)  # Entity -> Overall Score
    entity_rankings: List[Tuple[uuid.UUID, float]] = field(default_factory=list)  # Ranked by score

    # Analysis results
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    robustness_assessment: Dict[str, float] = field(default_factory=dict)
    stakeholder_agreement: Dict[uuid.UUID, float] = field(default_factory=dict)

    # Quality assurance
    application_validity: Optional[float] = None  # 0-1 scale
    result_reliability: Optional[float] = None    # 0-1 scale
    consistency_checks: Dict[str, bool] = field(default_factory=dict)

    def calculate_aggregated_scores(
        self,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE) -> Dict[uuid.UUID, float]:
        """Calculate aggregated scores for all evaluated entities."""
        aggregated_results = {}

        for entity_id in self.evaluated_entities:
            if entity_id not in self.entity_scores:
                continue

            entity_criteria_scores = self.entity_scores[entity_id]

            if aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
                # Weighted average aggregation
                weighted_sum = 0.0
                total_weight = 0.0

                for criterion_id, score in entity_criteria_scores.items():
                    weight = self.criteria_weights.get(criterion_id, 1.0)
                    weighted_sum += score * weight
                    total_weight += weight

                if total_weight > 0:
                    aggregated_results[entity_id] = weighted_sum / total_weight

            elif aggregation_method == AggregationMethod.GEOMETRIC_MEAN:
                # Geometric mean aggregation
                scores = list(entity_criteria_scores.values())
                if scores and all(score > 0 for score in scores):
                    geometric_mean = 1.0
                    for score in scores:
                        geometric_mean *= score
                    aggregated_results[entity_id] = geometric_mean ** (1.0 / len(scores))

            # Add other aggregation methods as needed

        self.aggregated_scores = aggregated_results

        # Update rankings
        self.entity_rankings = sorted(
            list(aggregated_results.items()),
            key=lambda x: x[1],
            reverse=True
        )

        return aggregated_results

    def conduct_sensitivity_analysis(self) -> Dict[str, Any]:
        """Conduct sensitivity analysis on criteria weights and scores."""
        sensitivity_results = {
            'weight_sensitivity': {},
            'score_sensitivity': {},
            'ranking_stability': 0.0,
            'critical_criteria': [],
            'robust_rankings': []
        }

        if not self.entity_rankings:
            return sensitivity_results

        original_rankings = [entity_id for entity_id, _ in self.entity_rankings]

        # Weight sensitivity analysis
        for criterion_id in self.applied_criteria:
            if criterion_id in self.criteria_weights:
                original_weight = self.criteria_weights[criterion_id]

                # Test weight variations
                weight_variations = [0.5 * original_weight, 1.5 * original_weight, 2.0 * original_weight]
                ranking_changes = []

                for new_weight in weight_variations:
                    # Temporarily change weight
                    self.criteria_weights[criterion_id] = new_weight

                    # Recalculate scores
                    temp_scores = self.calculate_aggregated_scores()
                    temp_rankings = [entity_id for entity_id, _ in sorted(
                        temp_scores.items(), key=lambda x: x[1], reverse=True
                    )]

                    # Calculate ranking change
                    ranking_correlation = self._calculate_ranking_correlation(
                        original_rankings,
                        temp_rankings)
                    ranking_changes.append(1.0 - ranking_correlation)  # Higher = more sensitive

                # Restore original weight
                self.criteria_weights[criterion_id] = original_weight

                # Store sensitivity measure
                avg_sensitivity = sum(ranking_changes) / len(ranking_changes)
                sensitivity_results['weight_sensitivity'][str(criterion_id)] = avg_sensitivity

                if avg_sensitivity > 0.3:  # High sensitivity threshold
                    sensitivity_results['critical_criteria'].append(criterion_id)

        # Overall ranking stability
        if sensitivity_results['weight_sensitivity']:
            avg_sensitivity = sum(sensitivity_results['weight_sensitivity'].values()) / len(sensitivity_results['weight_sensitivity'])
            sensitivity_results['ranking_stability'] = 1.0 - avg_sensitivity

        self.sensitivity_analysis = sensitivity_results
        return sensitivity_results

    def _calculate_ranking_correlation(
        self,
        ranking1: List[uuid.UUID],
        ranking2: List[uuid.UUID]) -> float:
        """Calculate correlation between two rankings (Spearman's rank correlation)."""
        if len(ranking1) != len(ranking2) or len(ranking1) < 2:
            return 0.0

        # Create rank dictionaries
        rank1 = {entity: i for i, entity in enumerate(ranking1)}
        rank2 = {entity: i for i, entity in enumerate(ranking2)}

        # Calculate Spearman correlation
        n = len(ranking1)
        d_squared_sum = sum((rank1[entity] - rank2[entity]) ** 2 for entity in ranking1)

        correlation = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))
        return max(0.0, correlation)  # Ensure non-negative

    def validate_application_results(self) -> Dict[str, Any]:
        """Validate quality and reliability of application results."""
        validation_results = {
            'completeness_check': 0.0,
            'consistency_check': 0.0,
            'validity_assessment': 0.0,
            'reliability_assessment': 0.0,
            'overall_quality': 0.0,
            'validation_issues': [],
            'recommendations': []
        }

        # Completeness check
        expected_evaluations = len(self.evaluated_entities) * len(self.applied_criteria)
        actual_evaluations = sum(len(criteria_scores) for criteria_scores in self.entity_scores.values())

        if expected_evaluations > 0:
            validation_results['completeness_check'] = actual_evaluations / expected_evaluations

        # Consistency check
        if self.aggregated_scores and self.entity_rankings:
            # Check if rankings match aggregated scores
            score_based_ranking = sorted(
                self.aggregated_scores.items(),
                key=lambda x: x[1],
                reverse=True)
            score_ranking_ids = [entity_id for entity_id, _ in score_based_ranking]
            stated_ranking_ids = [entity_id for entity_id, _ in self.entity_rankings]

            validation_results['consistency_check'] = self._calculate_ranking_correlation(
                score_ranking_ids, stated_ranking_ids
            )

        # Overall quality assessment
        quality_components = [
            validation_results['completeness_check'],
            validation_results['consistency_check']
        ]
        valid_components = [comp for comp in quality_components if comp > 0]
        if valid_components:
            validation_results['overall_quality'] = sum(valid_components) / len(valid_components)

        # Generate recommendations
        if validation_results['completeness_check'] < 0.9:
            validation_results['recommendations'].append('Complete missing evaluations')
        if validation_results['consistency_check'] < 0.8:
            validation_results['recommendations'].append('Review ranking consistency')

        return validation_results

    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        evaluation_report = {
            'executive_summary': {},
            'detailed_results': {},
            'methodology': {},
            'quality_assessment': {},
            'recommendations': []
        }

        # Executive summary
        if self.entity_rankings:
            top_performer = self.entity_rankings[0]
            evaluation_report['executive_summary'] = {
                'entities_evaluated': len(self.evaluated_entities),
                'criteria_applied': len(self.applied_criteria),
                'top_performer': top_performer[0],
                'top_score': top_performer[1],
                'evaluation_context': self.application_context
            }

        # Detailed results
        evaluation_report['detailed_results'] = {
            'entity_scores': {str(k): v for k, v in self.aggregated_scores.items()},
            'rankings': [(str(entity_id), score) for entity_id, score in self.entity_rankings],
            'criteria_weights': {str(k): v for k, v in self.criteria_weights.items()}
        }

        # Methodology
        evaluation_report['methodology'] = {
            'weighting_method': self.weighting_method.name,
            'evaluation_timeframe': str(self.evaluation_timeframe) if self.evaluation_timeframe else None,
            'quality_assurance_applied': bool(self.consistency_checks)
        }

        # Quality assessment
        validation_results = self.validate_application_results()
        evaluation_report['quality_assessment'] = validation_results

        # Generate recommendations
        if validation_results['overall_quality'] < 0.8:
            evaluation_report['recommendations'].append('Improve evaluation quality and completeness')

        # Sensitivity analysis insights
        if self.sensitivity_analysis:
            if self.sensitivity_analysis.get('ranking_stability', 1.0) < 0.7:
                evaluation_report['recommendations'].append('Results sensitive to weight changes - consider weight validation')

        return evaluation_report

@dataclass
class MultiCriteriaAnalysis(Node):
    """Multi-criteria decision analysis tools and methods."""

    analysis_purpose: Optional[str] = None
    decision_context: Optional[str] = None

    # Analysis components
    alternatives: List[uuid.UUID] = field(default_factory=list)  # Decision alternatives
    criteria_application: Optional[uuid.UUID] = None  # CriteriaApplication ID
    stakeholder_preferences: Dict[uuid.UUID, Dict[uuid.UUID, float]] = field(default_factory=dict)

    # MCDA methods
    applied_methods: List[str] = field(default_factory=list)
    method_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    consensus_analysis: Dict[str, Any] = field(default_factory=dict)

    # Decision support
    recommendation: Optional[uuid.UUID] = None  # Recommended alternative
    recommendation_confidence: Optional[float] = None  # 0-1 scale
    decision_rationale: Optional[str] = None

    # Uncertainty analysis
    uncertainty_factors: List[str] = field(default_factory=list)
    robustness_assessment: Dict[uuid.UUID, float] = field(default_factory=dict)
    scenario_analysis: Dict[str, Dict[uuid.UUID, float]] = field(default_factory=dict)

    def conduct_multi_method_analysis(self) -> Dict[str, Any]:
        """Conduct analysis using multiple MCDA methods."""
        analysis_results = {
            'method_rankings': {},
            'consensus_ranking': [],
            'agreement_level': 0.0,
            'robust_alternatives': [],
            'method_comparison': {}
        }

        if not self.alternatives or not self.criteria_application:
            return analysis_results

        # Simplified multi-method analysis
        # In practice, would implement actual MCDA methods like TOPSIS, ELECTRE, etc.

        # Mock results for different methods
        methods = ['TOPSIS', 'ELECTRE', 'AHP', 'PROMETHEE']
        method_rankings = {}

        for method in methods:
            # Generate mock rankings (in practice, would calculate using actual method)
            if method == 'TOPSIS':
                # TOPSIS-style ranking (simplified)
                ranking = self._mock_topsis_ranking()
            elif method == 'AHP':
                # AHP-style ranking (simplified)
                ranking = self._mock_ahp_ranking()
            else:
                # Generic ranking
                ranking = list(self.alternatives)

            method_rankings[method] = ranking
            self.applied_methods.append(method)

        analysis_results['method_rankings'] = method_rankings

        # Calculate consensus ranking
        consensus_ranking = self._calculate_consensus_ranking(method_rankings)
        analysis_results['consensus_ranking'] = consensus_ranking

        # Calculate agreement level between methods
        agreement_level = self._calculate_method_agreement(method_rankings)
        analysis_results['agreement_level'] = agreement_level

        # Identify robust alternatives (consistently high-ranking)
        robust_alternatives = self._identify_robust_alternatives(method_rankings)
        analysis_results['robust_alternatives'] = robust_alternatives

        # Store results
        self.method_results['multi_method_analysis'] = analysis_results

        # Set recommendation
        if consensus_ranking:
            self.recommendation = consensus_ranking[0]
            self.recommendation_confidence = agreement_level

        return analysis_results

    def _mock_topsis_ranking(self) -> List[uuid.UUID]:
        """Mock TOPSIS ranking for demonstration."""
        # In practice, would implement actual TOPSIS method
        return list(self.alternatives)

    def _mock_ahp_ranking(self) -> List[uuid.UUID]:
        """Mock AHP ranking for demonstration."""
        # In practice, would implement actual AHP method
        return list(self.alternatives)

    def _calculate_consensus_ranking(
        self,
        method_rankings: Dict[str,
        List[uuid.UUID]]) -> List[uuid.UUID]:
        """Calculate consensus ranking across methods."""
        if not method_rankings:
            return []

        # Use Borda count method for consensus
        alternative_scores = {alt: 0 for alt in self.alternatives}

        for ranking in method_rankings.values():
            for i, alternative in enumerate(ranking):
                # Higher rank = higher score
                score = len(ranking) - i
                alternative_scores[alternative] += score

        # Sort by total score
        consensus_ranking = sorted(
            alternative_scores.keys(),
            key=lambda x: alternative_scores[x],
            reverse=True
        )

        return consensus_ranking

    def _calculate_method_agreement(self, method_rankings: Dict[str, List[uuid.UUID]]) -> float:
        """Calculate agreement level between different methods."""
        if len(method_rankings) < 2:
            return 1.0

        # Calculate pairwise correlations
        correlations = []
        method_names = list(method_rankings.keys())

        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                ranking1 = method_rankings[method_names[i]]
                ranking2 = method_rankings[method_names[j]]

                # Calculate ranking correlation
                correlation = self._calculate_ranking_correlation(ranking1, ranking2)
                correlations.append(correlation)

        # Return average correlation
        return sum(correlations) / len(correlations) if correlations else 0.0

    def _calculate_ranking_correlation(
        self,
        ranking1: List[uuid.UUID],
        ranking2: List[uuid.UUID]) -> float:
        """Calculate correlation between two rankings."""
        # Implementation similar to CriteriaApplication method
        if len(ranking1) != len(ranking2) or len(ranking1) < 2:
            return 0.0

        rank1 = {entity: i for i, entity in enumerate(ranking1)}
        rank2 = {entity: i for i, entity in enumerate(ranking2)}

        n = len(ranking1)
        d_squared_sum = sum((rank1[entity] - rank2[entity]) ** 2 for entity in ranking1)

        correlation = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))
        return max(0.0, correlation)

    def _identify_robust_alternatives(
        self,
        method_rankings: Dict[str,
        List[uuid.UUID]]) -> List[uuid.UUID]:
        """Identify alternatives that perform consistently well across methods."""
        if not method_rankings:
            return []

        # Count how often each alternative appears in top positions
        top_positions = {}  # Alternative -> count of top-3 appearances

        for ranking in method_rankings.values():
            top_3 = ranking[:3] if len(ranking) >= 3 else ranking
            for alternative in top_3:
                top_positions[alternative] = top_positions.get(alternative, 0) + 1

        # Identify robust alternatives (appear in top positions in most methods)
        threshold = len(method_rankings) * 0.7  # 70% of methods
        robust_alternatives = [
            alt for alt, count in top_positions.items()
            if count >= threshold
        ]

        return robust_alternatives

@dataclass
class CriteriaFramework(Node):
    """Comprehensive criteria framework for systematic evaluation."""

    framework_scope: Optional[str] = None
    framework_purpose: Optional[str] = None

    # Framework components
    evaluation_criteria: List[uuid.UUID] = field(default_factory=list)  # EvaluationCriterion IDs
    criteria_applications: List[uuid.UUID] = field(default_factory=list)  # CriteriaApplication IDs
    mcda_analyses: List[uuid.UUID] = field(default_factory=list)  # MultiCriteriaAnalysis IDs

    # Framework structure
    criteria_hierarchy: Dict[str, List[uuid.UUID]] = field(default_factory=dict)  # Level -> criteria
    criteria_categories: Dict[CriteriaScope, List[uuid.UUID]] = field(default_factory=dict)
    value_alignment: Dict[ValueCategory, List[uuid.UUID]] = field(default_factory=dict)

    # Framework properties
    framework_completeness: Optional[float] = None  # Coverage completeness (0-1)
    framework_coherence: Optional[float] = None     # Internal consistency (0-1)
    framework_validity: Optional[float] = None      # Overall validity (0-1)
    framework_utility: Optional[float] = None       # Practical usefulness (0-1)

    # Integration with SFM
    matrix_evaluation_coverage: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)  # Cell -> criteria
    institutional_evaluation_criteria: Dict[uuid.UUID, List[uuid.UUID]] = field(default_factory=dict)
    delivery_evaluation_criteria: List[uuid.UUID] = field(default_factory=list)

    # Framework governance
    criteria_stewardship: Dict[uuid.UUID, uuid.UUID] = field(default_factory=dict)  # Criterion -> steward
    review_schedule: Optional[str] = None
    adaptation_mechanisms: List[str] = field(default_factory=list)

    # Quality assurance
    framework_validation_history: List[Dict[str, Any]] = field(default_factory=list)
    continuous_improvement_process: List[str] = field(default_factory=list)
    stakeholder_feedback_integration: Dict[str, List[str]] = field(default_factory=dict)

    def assess_framework_completeness(self) -> Dict[str, Any]:
        """Assess completeness of the criteria framework."""
        completeness_assessment = {
            'scope_coverage': {},
            'value_coverage': {},
            'system_level_coverage': {},
            'overall_completeness': 0.0,
            'coverage_gaps': [],
            'recommendations': []
        }

        # Scope coverage assessment
        total_scopes = len(CriteriaScope)
        covered_scopes = len(self.criteria_categories)
        completeness_assessment['scope_coverage'] = {
            'covered_scopes': covered_scopes,
            'total_scopes': total_scopes,
            'coverage_ratio': covered_scopes / total_scopes
        }

        # Value category coverage
        total_values = len(ValueCategory)
        covered_values = len(self.value_alignment)
        completeness_assessment['value_coverage'] = {
            'covered_values': covered_values,
            'total_values': total_values,
            'coverage_ratio': covered_values / total_values
        }

        # Overall completeness calculation
        completeness_factors = [
            completeness_assessment['scope_coverage']['coverage_ratio'],
            completeness_assessment['value_coverage']['coverage_ratio']
        ]
        completeness_assessment['overall_completeness'] = sum(completeness_factors) / len(completeness_factors)
        self.framework_completeness = completeness_assessment['overall_completeness']

        # Identify gaps
        if completeness_assessment['scope_coverage']['coverage_ratio'] < 0.8:
            completeness_assessment['coverage_gaps'].append('Incomplete scope coverage')
        if completeness_assessment['value_coverage']['coverage_ratio'] < 0.7:
            completeness_assessment['coverage_gaps'].append('Incomplete value category coverage')

        return completeness_assessment

    def validate_framework_coherence(self) -> Dict[str, Any]:
        """Validate internal coherence of the criteria framework."""
        coherence_assessment = {
            'criteria_consistency': 0.0,
            'hierarchy_alignment': 0.0,
            'value_coherence': 0.0,
            'overall_coherence': 0.0,
            'coherence_issues': [],
            'improvement_recommendations': []
        }

        # Criteria consistency (simplified assessment)
        if self.evaluation_criteria:
            coherence_assessment['criteria_consistency'] = 0.8  # Placeholder

        # Hierarchy alignment
        if self.criteria_hierarchy:
            coherence_assessment['hierarchy_alignment'] = 0.7  # Placeholder

        # Value coherence
        value_coverage_ratio = len(self.value_alignment) / len(ValueCategory)
        coherence_assessment['value_coherence'] = value_coverage_ratio

        # Overall coherence
        coherence_factors = [
            coherence_assessment['criteria_consistency'],
            coherence_assessment['hierarchy_alignment'],
            coherence_assessment['value_coherence']
        ]
        coherence_assessment['overall_coherence'] = sum(coherence_factors) / len(coherence_factors)
        self.framework_coherence = coherence_assessment['overall_coherence']

        # Generate recommendations
        if coherence_assessment['overall_coherence'] < 0.7:
            coherence_assessment['improvement_recommendations'].append('Strengthen framework coherence')

        return coherence_assessment

    def generate_framework_dashboard(self) -> Dict[str, Any]:
        """Generate framework-level dashboard view."""
        dashboard = {
            'framework_overview': {
                'total_criteria': len(self.evaluation_criteria),
                'active_applications': len(self.criteria_applications),
                'framework_completeness': self.framework_completeness or 0.0,
                'framework_coherence': self.framework_coherence or 0.0,
                'framework_health': 'unknown'
            },
            'coverage_summary': {},
            'recent_applications': [],
            'quality_indicators': {},
            'framework_alerts': []
        }

        # Assess framework health
        if (self.framework_completeness and self.framework_coherence and
            self.framework_completeness > 0.8 and self.framework_coherence > 0.8):
            dashboard['framework_overview']['framework_health'] = 'excellent'
        elif (self.framework_completeness and self.framework_coherence and
              self.framework_completeness > 0.6 and self.framework_coherence > 0.6):
            dashboard['framework_overview']['framework_health'] = 'good'
        else:
            dashboard['framework_overview']['framework_health'] = 'needs_attention'

        # Coverage summary
        dashboard['coverage_summary'] = {
            'criteria_scopes': len(self.criteria_categories),
            'value_categories': len(self.value_alignment),
            'matrix_coverage': len(self.matrix_evaluation_coverage)
        }

        # Framework alerts
        if self.framework_completeness and self.framework_completeness < 0.6:
            dashboard['framework_alerts'].append('Low framework completeness')
        if self.framework_coherence and self.framework_coherence < 0.6:
            dashboard['framework_alerts'].append('Framework coherence issues')

        return dashboard
