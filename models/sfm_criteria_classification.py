"""
Systematic Criteria Classification Framework for Social Fabric Matrix.

This module implements Hayden's standardized criteria classification system,
providing the systematic taxonomy of evaluation criteria that is fundamental
to consistent SFM analysis. It establishes the standardized criteria categories
that Hayden emphasizes for matrix construction.

Key Components:
- StandardizedCriteria: Hayden's standard criteria taxonomy
- CriteriaClassification: Classification system for criteria
- CriteriaRelationships: Relationships between criteria
- CriteriaValidation: Validation of criteria appropriateness
- CriteriaMapping: Mapping between different criteria systems
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto

from models.base_nodes import Node
from models.sfm_enums import (
    ValueCategory,
    SystemLevel,
    ValidationMethod,
    EvidenceQuality,
)


class StandardCriteriaType(Enum):
    """Hayden's standard criteria types for SFM analysis."""
    
    # Primary Criteria (Life Process Enhancement)
    SECURITY = auto()           # Safety, stability, predictability
    EQUITY = auto()            # Fairness, justice, equal access
    LIBERTY = auto()           # Freedom, autonomy, choice
    EFFICIENCY = auto()        # Resource optimization, productivity
    
    # Secondary Criteria (Instrumental Goals)
    EFFECTIVENESS = auto()     # Goal achievement, outcomes
    SUSTAINABILITY = auto()    # Long-term viability, environmental
    ADAPTABILITY = auto()      # Flexibility, learning capacity
    PARTICIPATION = auto()     # Democratic involvement, inclusion
    
    # Supporting Criteria (Contextual Factors)
    ACCOUNTABILITY = auto()    # Transparency, responsibility
    LEGITIMACY = auto()        # Social acceptance, authority
    COORDINATION = auto()      # Integration, cooperation
    INNOVATION = auto()        # Creativity, technological advance


class CriteriaScope(Enum):
    """Scope of criteria application in SFM analysis."""
    
    INDIVIDUAL_LEVEL = auto()       # Individual outcomes
    HOUSEHOLD_LEVEL = auto()        # Household/family outcomes
    COMMUNITY_LEVEL = auto()        # Community outcomes
    ORGANIZATIONAL_LEVEL = auto()   # Organizational performance
    INSTITUTIONAL_LEVEL = auto()    # Institutional effectiveness
    SYSTEM_LEVEL = auto()          # System-wide outcomes


class CriteriaMeasurementType(Enum):
    """Types of criteria measurement approaches."""
    
    QUANTITATIVE_OBJECTIVE = auto()    # Numerical, objective measures
    QUANTITATIVE_SUBJECTIVE = auto()   # Numerical, subjective measures
    QUALITATIVE_STRUCTURED = auto()    # Structured qualitative assessment
    QUALITATIVE_NARRATIVE = auto()     # Narrative assessment
    MIXED_METHOD = auto()              # Combined approaches
    PROXY_INDICATOR = auto()           # Indirect measurement


class CriteriaWeight(Enum):
    """Standard weighting categories for criteria."""
    
    CRITICAL = auto()          # Essential, must be satisfied
    HIGH = auto()             # Very important
    MODERATE = auto()         # Important
    LOW = auto()              # Somewhat important
    CONTEXTUAL = auto()       # Depends on context


class CriteriaRelationshipType(Enum):
    """Types of relationships between criteria."""
    
    COMPLEMENTARY = auto()     # Mutually reinforcing
    COMPETING = auto()         # Trade-offs required
    HIERARCHICAL = auto()      # One builds on another
    CONDITIONAL = auto()       # Relationship depends on context
    INDEPENDENT = auto()       # No significant relationship


@dataclass
class StandardizedCriteria(Node):
    """Hayden's standardized criteria for systematic SFM analysis."""
    
    criteria_type: StandardCriteriaType = StandardCriteriaType.EFFICIENCY
    criteria_scope: CriteriaScope = CriteriaScope.INSTITUTIONAL_LEVEL
    
    # Criteria definition
    formal_definition: Optional[str] = None
    operational_definition: Optional[str] = None
    measurement_guidance: Optional[str] = None
    
    # Criteria properties
    criteria_weight: CriteriaWeight = CriteriaWeight.MODERATE
    value_alignment: List[ValueCategory] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)
    
    # Measurement specifications
    measurement_type: CriteriaMeasurementType = CriteriaMeasurementType.MIXED_METHOD
    measurement_units: List[str] = field(default_factory=list)
    measurement_scale: Optional[Tuple[float, float]] = None
    target_benchmarks: Dict[str, float] = field(default_factory=dict)
    
    # Quality standards
    minimum_data_quality: EvidenceQuality = EvidenceQuality.MEDIUM
    validation_requirements: List[ValidationMethod] = field(default_factory=list)
    inter_rater_reliability_threshold: Optional[float] = None
    
    # Relationships with other criteria
    complementary_criteria: List[uuid.UUID] = field(default_factory=list)
    competing_criteria: List[uuid.UUID] = field(default_factory=list)
    prerequisite_criteria: List[uuid.UUID] = field(default_factory=list)
    
    # Application guidance
    institutional_applicability: Dict[str, bool] = field(default_factory=dict)  # Institution type -> applicable
    sector_applicability: Dict[str, bool] = field(default_factory=dict)        # Sector -> applicable
    scale_applicability: Dict[SystemLevel, bool] = field(default_factory=dict)
    
    # Historical usage
    usage_frequency: Optional[int] = None  # How often used in analyses
    effectiveness_rating: Optional[float] = None  # How effective in practice
    modification_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def assess_applicability(self, institution_type: str, sector: str, system_level: SystemLevel) -> Dict[str, Any]:
        """Assess applicability of this criteria to specific context."""
        applicability_assessment = {
            'overall_applicable': True,
            'applicability_score': 0.0,
            'constraints': [],
            'recommendations': [],
            'confidence_level': 1.0
        }
        
        # Institution type check
        if institution_type in self.institutional_applicability:
            if not self.institutional_applicability[institution_type]:
                applicability_assessment['overall_applicable'] = False
                applicability_assessment['constraints'].append(f'Not applicable to {institution_type} institutions')
        
        # Sector check
        if sector in self.sector_applicability:
            if not self.sector_applicability[sector]:
                applicability_assessment['overall_applicable'] = False
                applicability_assessment['constraints'].append(f'Not applicable to {sector} sector')
        
        # System level check
        if system_level in self.scale_applicability:
            if not self.scale_applicability[system_level]:
                applicability_assessment['overall_applicable'] = False
                applicability_assessment['constraints'].append(f'Not applicable at {system_level.name} level')
        
        # Calculate applicability score
        if applicability_assessment['overall_applicable']:
            base_score = 1.0
            
            # Adjust based on criteria weight
            weight_multiplier = {
                CriteriaWeight.CRITICAL: 1.0,
                CriteriaWeight.HIGH: 0.9,
                CriteriaWeight.MODERATE: 0.8,
                CriteriaWeight.LOW: 0.6,
                CriteriaWeight.CONTEXTUAL: 0.7
            }.get(self.criteria_weight, 0.8)
            
            applicability_assessment['applicability_score'] = base_score * weight_multiplier
        
        # Generate recommendations
        if not applicability_assessment['overall_applicable']:
            applicability_assessment['recommendations'].append('Consider alternative criteria for this context')
        elif applicability_assessment['applicability_score'] < 0.8:
            applicability_assessment['recommendations'].append('Use with caution in this context')
        
        return applicability_assessment
    
    def validate_measurement_feasibility(self) -> Dict[str, Any]:
        """Validate feasibility of measuring this criteria."""
        feasibility_assessment = {
            'measurement_feasible': True,
            'feasibility_score': 0.0,
            'challenges': [],
            'requirements': [],
            'recommendations': []
        }
        
        # Check measurement type feasibility
        if self.measurement_type == CriteriaMeasurementType.QUANTITATIVE_OBJECTIVE:
            if not self.measurement_units:
                feasibility_assessment['challenges'].append('No measurement units defined for quantitative criteria')
                feasibility_assessment['feasibility_score'] -= 0.3
        
        # Check data quality requirements
        if self.minimum_data_quality == EvidenceQuality.HIGH:
            feasibility_assessment['requirements'].append('High-quality data collection required')
            if not self.validation_requirements:
                feasibility_assessment['challenges'].append('High data quality required but no validation methods specified')
        
        # Check reliability requirements
        if self.inter_rater_reliability_threshold and self.inter_rater_reliability_threshold > 0.8:
            feasibility_assessment['requirements'].append('High inter-rater reliability required')
            feasibility_assessment['recommendations'].append('Develop detailed measurement protocols')
        
        # Calculate overall feasibility
        base_feasibility = 1.0
        feasibility_assessment['feasibility_score'] = max(0.0, base_feasibility + sum([
            -0.2 if 'No measurement units' in challenge else 0
            for challenge in feasibility_assessment['challenges']
        ]))
        
        feasibility_assessment['measurement_feasible'] = feasibility_assessment['feasibility_score'] >= 0.6
        
        return feasibility_assessment


@dataclass
class CriteriaClassification(Node):
    """Classification system for organizing and categorizing criteria."""
    
    classification_name: Optional[str] = None
    classification_purpose: Optional[str] = None
    
    # Classification structure
    criteria_taxonomy: Dict[str, List[uuid.UUID]] = field(default_factory=dict)  # Category -> criteria
    hierarchy_levels: List[str] = field(default_factory=list)
    classification_rules: List[str] = field(default_factory=list)
    
    # Standard criteria included
    standard_criteria: List[uuid.UUID] = field(default_factory=list)  # StandardizedCriteria IDs
    custom_criteria: List[uuid.UUID] = field(default_factory=list)    # Custom criteria IDs
    
    # Classification properties
    classification_completeness: Optional[float] = None  # Coverage completeness (0-1)
    classification_consistency: Optional[float] = None   # Internal consistency (0-1)
    inter_classifier_agreement: Optional[float] = None   # Agreement between classifiers
    
    # Usage context
    applicable_domains: List[str] = field(default_factory=list)
    target_users: List[str] = field(default_factory=list)
    complexity_level: Optional[str] = None  # simple, moderate, complex
    
    # Quality assurance
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    expert_review_status: Optional[str] = None
    usage_statistics: Dict[str, int] = field(default_factory=dict)
    
    def validate_classification_completeness(self) -> Dict[str, Any]:
        """Validate completeness of the classification system."""
        completeness_assessment = {
            'coverage_assessment': {},
            'gap_analysis': [],
            'completeness_score': 0.0,
            'recommendations': []
        }
        
        # Check coverage of standard criteria types
        standard_types_covered = set()
        for _ in self.standard_criteria:
            # Would need to look up actual criteria to check type coverage
            pass  # Simplified for now
        
        # Assess coverage of key criteria categories
        required_categories = [
            StandardCriteriaType.SECURITY,
            StandardCriteriaType.EQUITY,
            StandardCriteriaType.EFFICIENCY,
            StandardCriteriaType.EFFECTIVENESS
        ]
        
        coverage_score = len(standard_types_covered) / len(required_categories) if required_categories else 1.0
        completeness_assessment['completeness_score'] = coverage_score
        self.classification_completeness = coverage_score
        
        # Generate recommendations
        if coverage_score < 0.8:
            completeness_assessment['recommendations'].append('Add missing standard criteria types')
        if not self.custom_criteria and coverage_score < 1.0:
            completeness_assessment['recommendations'].append('Consider adding domain-specific criteria')
        
        return completeness_assessment
    
    def assess_classification_consistency(self) -> Dict[str, Any]:
        """Assess internal consistency of the classification system."""
        consistency_assessment = {
            'logical_consistency': 0.0,
            'taxonomic_consistency': 0.0,
            'application_consistency': 0.0,
            'overall_consistency': 0.0,
            'consistency_issues': [],
            'improvement_recommendations': []
        }
        
        # Check logical consistency (simplified)
        logical_consistency = 0.8  # Placeholder - would check for logical contradictions
        consistency_assessment['logical_consistency'] = logical_consistency
        
        # Check taxonomic consistency
        if self.criteria_taxonomy:
            taxonomic_consistency = 0.7  # Placeholder - would check category overlaps
            consistency_assessment['taxonomic_consistency'] = taxonomic_consistency
        
        # Check application consistency
        if self.classification_rules:
            application_consistency = 0.8  # Placeholder - would check rule consistency
            consistency_assessment['application_consistency'] = application_consistency
        
        # Calculate overall consistency
        consistency_scores = [
            consistency_assessment['logical_consistency'],
            consistency_assessment['taxonomic_consistency'],
            consistency_assessment['application_consistency']
        ]
        valid_scores = [score for score in consistency_scores if score > 0]
        
        if valid_scores:
            consistency_assessment['overall_consistency'] = sum(valid_scores) / len(valid_scores)
            self.classification_consistency = consistency_assessment['overall_consistency']
        
        # Generate improvement recommendations
        if consistency_assessment['overall_consistency'] < 0.7:
            consistency_assessment['improvement_recommendations'].append('Review and improve classification consistency')
        
        return consistency_assessment


@dataclass
class CriteriaRelationship(Node):
    """Relationships between criteria in the classification system."""
    
    relationship_type: CriteriaRelationshipType = CriteriaRelationshipType.INDEPENDENT
    source_criteria_id: Optional[uuid.UUID] = None
    target_criteria_id: Optional[uuid.UUID] = None
    
    # Relationship characteristics
    relationship_strength: Optional[float] = None  # 0-1 scale
    relationship_direction: Optional[str] = None   # bidirectional, unidirectional
    context_dependency: Optional[str] = None       # When relationship applies
    
    # Empirical validation
    empirical_evidence: List[str] = field(default_factory=list)
    statistical_correlation: Optional[float] = None
    theoretical_justification: Optional[str] = None
    
    # Relationship implications  
    trade_off_nature: Optional[str] = None  # zero-sum, positive-sum, context-dependent
    optimization_strategy: Optional[str] = None  # How to handle trade-offs
    synergy_potential: Optional[float] = None  # Potential for mutual enhancement
    
    # Quality assessment
    relationship_confidence: Optional[float] = None  # Confidence in relationship (0-1)
    validation_status: Optional[str] = None
    expert_consensus: Optional[float] = None  # Expert agreement on relationship
    
    def analyze_trade_off_implications(self) -> Dict[str, Any]:
        """Analyze trade-off implications of this criteria relationship."""
        trade_off_analysis = {
            'trade_off_exists': False,
            'trade_off_intensity': 0.0,
            'optimization_approaches': [],
            'mitigation_strategies': [],
            'decision_guidance': []
        }
        
        if self.relationship_type == CriteriaRelationshipType.COMPETING:
            trade_off_analysis['trade_off_exists'] = True
            
            # Assess trade-off intensity
            if self.relationship_strength:
                trade_off_analysis['trade_off_intensity'] = self.relationship_strength
            
            # Generate optimization approaches
            if self.trade_off_nature == 'zero-sum':
                trade_off_analysis['optimization_approaches'] = [
                    'Prioritize criteria based on context',
                    'Seek efficiency gains to minimize trade-offs',
                    'Use sequential optimization approach'
                ]
            elif self.trade_off_nature == 'positive-sum':
                trade_off_analysis['optimization_approaches'] = [
                    'Look for win-win solutions',
                    'Invest in capacity building',
                    'Pursue complementary strategies'
                ]
            
            # Generate mitigation strategies
            trade_off_analysis['mitigation_strategies'] = [
                'Stakeholder engagement to build consensus',
                'Temporal sequencing of improvements',
                'Innovation to transcend trade-offs'
            ]
        
        elif self.relationship_type == CriteriaRelationshipType.COMPLEMENTARY:
            trade_off_analysis['optimization_approaches'] = [
                'Pursue joint optimization',
                'Leverage synergies',
                'Integrated improvement strategies'
            ]
        
        return trade_off_analysis
    
    def validate_relationship_evidence(self) -> Dict[str, Any]:
        """Validate empirical evidence for the criteria relationship."""
        validation_results = {
            'evidence_strength': 0.0,
            'validation_quality': 'insufficient',
            'evidence_gaps': [],
            'validation_recommendations': []
        }
        
        # Assess empirical evidence
        evidence_score = 0.0
        if self.empirical_evidence:
            evidence_score += min(len(self.empirical_evidence) * 0.2, 0.6)
        
        if self.statistical_correlation is not None:
            if abs(self.statistical_correlation) > 0.3:
                evidence_score += 0.3
            elif abs(self.statistical_correlation) > 0.1:
                evidence_score += 0.1
        
        if self.theoretical_justification:
            evidence_score += 0.2
        
        validation_results['evidence_strength'] = min(evidence_score, 1.0)
        
        # Categorize validation quality
        if validation_results['evidence_strength'] >= 0.8:
            validation_results['validation_quality'] = 'strong'
        elif validation_results['evidence_strength'] >= 0.6:
            validation_results['validation_quality'] = 'moderate'
        elif validation_results['evidence_strength'] >= 0.3:
            validation_results['validation_quality'] = 'weak'
        
        # Identify evidence gaps
        if not self.empirical_evidence:
            validation_results['evidence_gaps'].append('No empirical evidence provided')
        if self.statistical_correlation is None:
            validation_results['evidence_gaps'].append('No statistical validation')
        if not self.theoretical_justification:
            validation_results['evidence_gaps'].append('No theoretical justification')
        
        # Generate recommendations
        if validation_results['evidence_strength'] < 0.6:
            validation_results['validation_recommendations'].append('Strengthen empirical evidence base')
        if self.relationship_confidence and self.relationship_confidence < 0.7:
            validation_results['validation_recommendations'].append('Reduce relationship uncertainty through additional research')
        
        return validation_results


@dataclass
class CriteriaMapping(Node):
    """Mapping between different criteria systems and frameworks."""
    
    source_system: Optional[str] = None  # Name of source criteria system
    target_system: Optional[str] = None  # Name of target criteria system
    
    # Mapping structure
    criteria_mappings: Dict[uuid.UUID, uuid.UUID] = field(default_factory=dict)  # Source -> target
    mapping_rules: List[str] = field(default_factory=list)
    transformation_functions: Dict[str, str] = field(default_factory=dict)
    
    # Mapping quality
    mapping_completeness: Optional[float] = None  # Coverage of source system (0-1)
    mapping_accuracy: Optional[float] = None      # Accuracy of mappings (0-1)
    bidirectional_consistency: Optional[float] = None  # Consistency of reverse mapping
    
    # Validation
    expert_validation: Dict[uuid.UUID, float] = field(default_factory=dict)  # Mapping -> expert score
    empirical_validation: Dict[uuid.UUID, float] = field(default_factory=dict)  # Mapping -> correlation
    usage_validation: Dict[uuid.UUID, int] = field(default_factory=dict)  # Mapping -> usage count
    
    # Context factors
    mapping_context: Optional[str] = None
    applicable_domains: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    def validate_mapping_quality(self) -> Dict[str, Any]:
        """Validate quality of criteria mapping."""
        validation_results = {
            'completeness_assessment': 0.0,
            'accuracy_assessment': 0.0,
            'consistency_assessment': 0.0,
            'overall_quality': 0.0,
            'quality_issues': [],
            'improvement_recommendations': []
        }
        
        # Completeness assessment
        if self.mapping_completeness is not None:
            validation_results['completeness_assessment'] = self.mapping_completeness
        
        # Accuracy assessment
        if self.mapping_accuracy is not None:
            validation_results['accuracy_assessment'] = self.mapping_accuracy
        elif self.expert_validation:
            # Use expert validation scores
            expert_scores = list(self.expert_validation.values())
            validation_results['accuracy_assessment'] = sum(expert_scores) / len(expert_scores)
        
        # Consistency assessment
        if self.bidirectional_consistency is not None:
            validation_results['consistency_assessment'] = self.bidirectional_consistency
        
        # Overall quality
        quality_components = [
            validation_results['completeness_assessment'],
            validation_results['accuracy_assessment'],
            validation_results['consistency_assessment']
        ]
        valid_components = [comp for comp in quality_components if comp > 0]
        
        if valid_components:
            validation_results['overall_quality'] = sum(valid_components) / len(valid_components)
        
        # Identify quality issues
        if validation_results['completeness_assessment'] < 0.8:
            validation_results['quality_issues'].append('Incomplete mapping coverage')
        if validation_results['accuracy_assessment'] < 0.7:
            validation_results['quality_issues'].append('Low mapping accuracy')
        if validation_results['consistency_assessment'] < 0.7:
            validation_results['quality_issues'].append('Inconsistent bidirectional mapping')
        
        # Generate recommendations
        if validation_results['overall_quality'] < 0.7:
            validation_results['improvement_recommendations'].append('Improve overall mapping quality')
        if not self.expert_validation:
            validation_results['improvement_recommendations'].append('Obtain expert validation of mappings')
        
        return validation_results
    
    def generate_mapping_report(self) -> Dict[str, Any]:
        """Generate comprehensive mapping report."""
        mapping_report = {
            'mapping_overview': {
                'source_system': self.source_system,
                'target_system': self.target_system,
                'total_mappings': len(self.criteria_mappings),
                'mapping_context': self.mapping_context
            },
            'quality_assessment': self.validate_mapping_quality(),
            'usage_statistics': {},
            'limitations': self.limitations,
            'recommendations': []
        }
        
        # Usage statistics
        if self.usage_validation:
            usage_stats = list(self.usage_validation.values())
            mapping_report['usage_statistics'] = {
                'total_usage': sum(usage_stats),
                'average_usage': sum(usage_stats) / len(usage_stats),
                'most_used_mappings': len([u for u in usage_stats if u > 10])
            }
        
        # Generate recommendations based on quality assessment
        quality_assessment = mapping_report['quality_assessment']
        if quality_assessment['overall_quality'] < 0.8:
            mapping_report['recommendations'].append('Consider mapping validation and improvement')
        
        return mapping_report


@dataclass
class CriteriaValidation(Node):
    """Validation framework for criteria appropriateness and quality."""
    
    validation_purpose: Optional[str] = None
    validation_scope: Optional[str] = None
    
    # Validation targets
    validated_criteria: List[uuid.UUID] = field(default_factory=list)
    validation_methods: List[ValidationMethod] = field(default_factory=list)
    validation_standards: Dict[str, Any] = field(default_factory=dict)
    
    # Validation process
    validation_protocol: List[str] = field(default_factory=list)
    validator_qualifications: List[str] = field(default_factory=list)
    validation_timeline: Optional[str] = None
    
    # Validation results
    criteria_validation_scores: Dict[uuid.UUID, float] = field(default_factory=dict)
    validation_consensus: Dict[uuid.UUID, float] = field(default_factory=dict)
    rejected_criteria: List[uuid.UUID] = field(default_factory=list)
    
    # Quality metrics
    inter_validator_reliability: Optional[float] = None
    validation_comprehensiveness: Optional[float] = None
    validation_rigor: Optional[float] = None
    
    # Improvement recommendations
    criteria_improvement_recommendations: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    system_improvement_recommendations: List[str] = field(default_factory=list)
    
    def conduct_systematic_validation(self) -> Dict[str, Any]:
        """Conduct systematic validation of criteria."""
        validation_results = {
            'validation_summary': {},
            'criteria_results': {},
            'quality_assessment': {},
            'recommendations': []
        }
        
        if not self.validated_criteria:
            validation_results['validation_summary'] = {
                'status': 'no_criteria_to_validate',
                'message': 'No criteria specified for validation'
            }
            return validation_results
        
        # Validation summary
        total_criteria = len(self.validated_criteria)
        validated_count = len(self.criteria_validation_scores)
        rejected_count = len(self.rejected_criteria)
        
        validation_results['validation_summary'] = {
            'total_criteria': total_criteria,
            'validated_criteria': validated_count,
            'rejected_criteria': rejected_count,
            'validation_rate': validated_count / total_criteria if total_criteria > 0 else 0
        }
        
        # Individual criteria results
        for criteria_id in self.validated_criteria:
            criteria_result = {
                'validation_score': self.criteria_validation_scores.get(criteria_id, 0.0),
                'consensus_level': self.validation_consensus.get(criteria_id, 0.0),
                'status': 'accepted' if criteria_id not in self.rejected_criteria else 'rejected',
                'improvements': self.criteria_improvement_recommendations.get(criteria_id, [])
            }
            validation_results['criteria_results'][str(criteria_id)] = criteria_result
        
        # Quality assessment
        if self.criteria_validation_scores:
            avg_score = sum(self.criteria_validation_scores.values()) / len(self.criteria_validation_scores)
            validation_results['quality_assessment'] = {
                'average_validation_score': avg_score,
                'score_distribution': self._calculate_score_distribution(),
                'inter_validator_reliability': self.inter_validator_reliability or 0.0
            }
        
        # Generate recommendations
        if validation_results['validation_summary']['validation_rate'] < 0.8:
            validation_results['recommendations'].append('Review and improve rejected criteria')
        
        if self.inter_validator_reliability and self.inter_validator_reliability < 0.7:
            validation_results['recommendations'].append('Improve validator training and protocols')
        
        return validation_results
    
    def _calculate_score_distribution(self) -> Dict[str, int]:
        """Calculate distribution of validation scores."""
        distribution = {
            'excellent': 0,  # 0.9-1.0
            'good': 0,       # 0.7-0.89
            'fair': 0,       # 0.5-0.69
            'poor': 0        # 0.0-0.49
        }
        
        for score in self.criteria_validation_scores.values():
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.7:
                distribution['good'] += 1
            elif score >= 0.5:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        validation_report = {
            'executive_summary': {},
            'detailed_results': self.conduct_systematic_validation(),
            'methodology': {
                'validation_methods': [method.name for method in self.validation_methods],
                'validation_protocol': self.validation_protocol,
                'validator_qualifications': self.validator_qualifications
            },
            'quality_indicators': {
                'inter_validator_reliability': self.inter_validator_reliability,
                'validation_comprehensiveness': self.validation_comprehensiveness,
                'validation_rigor': self.validation_rigor
            },
            'recommendations': {
                'criteria_specific': self.criteria_improvement_recommendations,
                'system_wide': self.system_improvement_recommendations
            }
        }
        
        # Executive summary
        detailed_results = validation_report['detailed_results']
        if 'validation_summary' in detailed_results:
            summary = detailed_results['validation_summary']
            validation_report['executive_summary'] = {
                'validation_completion': f"{summary.get('validated_criteria', 0)} of {summary.get('total_criteria', 0)} criteria validated",
                'overall_quality': detailed_results.get('quality_assessment', {}).get('average_validation_score', 0.0),
                'key_findings': self._extract_key_findings(detailed_results),
                'priority_actions': detailed_results.get('recommendations', [])[:3]
            }
        
        return validation_report
    
    def _extract_key_findings(self, detailed_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from detailed validation results."""
        findings = []
        
        # Quality findings
        quality_assessment = detailed_results.get('quality_assessment', {})
        avg_score = quality_assessment.get('average_validation_score', 0)
        if avg_score >= 0.8:
            findings.append('High overall criteria quality')
        elif avg_score < 0.6:
            findings.append('Criteria quality needs improvement')
        
        # Reliability findings
        reliability = quality_assessment.get('inter_validator_reliability', 0)
        if reliability >= 0.8:
            findings.append('High inter-validator reliability')
        elif reliability < 0.6:
            findings.append('Low inter-validator reliability')
        
        # Coverage findings
        validation_rate = detailed_results.get('validation_summary', {}).get('validation_rate', 0)
        if validation_rate < 0.8:
            findings.append('Incomplete validation coverage')
        
        return findings