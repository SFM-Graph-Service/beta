"""
Value Inquiry Framework for Social Fabric Matrix analysis.

This module implements Hayden's instrumentalist value inquiry methodology,
which is central to the SFM approach. It provides systematic tools for
value analysis, knowledge validation, and normative evaluation within
institutional contexts following instrumentalist philosophy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

from models.base_nodes import Node
# Local enum definitions - no imports needed from sfm_enums for these

class ValueType(Enum):
    """Types of values in instrumentalist analysis."""

    INSTRUMENTAL_VALUE = auto()      # Problem-solving, efficiency-oriented values
    CEREMONIAL_VALUE = auto()        # Status, tradition-oriented values
    AESTHETIC_VALUE = auto()         # Beauty, artistic values
    MORAL_VALUE = auto()            # Ethical, moral values
    ECONOMIC_VALUE = auto()         # Economic efficiency values
    SOCIAL_VALUE = auto()           # Community, social cohesion values
    ECOLOGICAL_VALUE = auto()       # Environmental, sustainability values
    CULTURAL_VALUE = auto()         # Cultural preservation, identity values

class InquiryMethod(Enum):
    """Methods for conducting value inquiry."""

    CONSEQUENTIALIST_ANALYSIS = auto()  # Focus on consequences/outcomes
    PRAGMATIC_EVALUATION = auto()       # Practical effectiveness assessment
    PROBLEM_ORIENTED_INQUIRY = auto()   # Problem-solving focus
    CONTEXTUAL_ANALYSIS = auto()        # Context-sensitive analysis
    COMPARATIVE_EVALUATION = auto()     # Comparing alternatives
    PARTICIPATORY_INQUIRY = auto()      # Stakeholder participation
    DELIBERATIVE_PROCESS = auto()       # Group deliberation
    EXPERIMENTAL_APPROACH = auto()      # Testing and experimentation

class KnowledgeValidationType(Enum):
    """Types of knowledge validation in instrumentalist inquiry."""

    EMPIRICAL_VALIDATION = auto()       # Empirical evidence testing
    LOGICAL_VALIDATION = auto()         # Logical consistency checking
    PRAGMATIC_VALIDATION = auto()       # Practical effectiveness testing
    CONSENSUS_VALIDATION = auto()       # Stakeholder agreement
    CONSEQUENTIALIST_VALIDATION = auto() # Outcome-based validation
    CONTEXTUAL_VALIDATION = auto()      # Context-appropriate validation
    INSTRUMENTAL_VALIDATION = auto()    # Problem-solving effectiveness

class NormativeCriteria(Enum):
    """Normative criteria for value evaluation."""

    PROBLEM_SOLVING_EFFECTIVENESS = auto()  # How well it solves problems
    COMMUNITY_ENHANCEMENT = auto()          # Enhancement of community life
    DEMOCRATIC_PARTICIPATION = auto()       # Democratic inclusiveness
    ECOLOGICAL_SUSTAINABILITY = auto()     # Environmental sustainability
    SOCIAL_EQUITY = auto()                 # Social fairness and equity
    ECONOMIC_EFFICIENCY = auto()           # Economic resource efficiency
    CULTURAL_CONTINUITY = auto()           # Cultural preservation
    ADAPTIVE_CAPACITY = auto()             # Ability to adapt and learn

class ValueConflictType(Enum):
    """Types of value conflicts."""

    INSTRUMENTAL_CEREMONIAL = auto()    # Instrumental vs. ceremonial values
    EFFICIENCY_EQUITY = auto()          # Efficiency vs. equity tensions
    INDIVIDUAL_COLLECTIVE = auto()      # Individual vs. collective values
    SHORT_TERM_LONG_TERM = auto()      # Temporal value conflicts
    LOCAL_GLOBAL = auto()              # Scale-based value conflicts
    ECONOMIC_ECOLOGICAL = auto()       # Economic vs. environmental values
    TRADITIONAL_MODERN = auto()        # Traditional vs. modern values

@dataclass
class ValueInquiry(Node):  # pylint: disable=too-many-instance-attributes
    """Core value inquiry process within instrumentalist framework."""

    inquiry_focus: Optional[str] = None  # What is being inquired about
    inquiry_methods: List[InquiryMethod] = field(default_factory=list)
    inquiry_participants: List[uuid.UUID] = field(default_factory=list)

    # Inquiry context
    problem_context: Optional[str] = None  # Problem being addressed
    institutional_context: List[uuid.UUID] = field(default_factory=list)
    temporal_context: Optional[str] = None  # Time frame of inquiry
    spatial_context: Optional[str] = None  # Geographic scope

    # Value identification
    identified_values: Dict[ValueType, List[str]] = field(default_factory=dict)
    value_hierarchies: Dict[uuid.UUID, Dict[str, int]] = field(default_factory=dict)  # Stakeholder value rankings
    value_conflicts: List[ValueConflictType] = field(default_factory=list)

    # Inquiry process
    inquiry_questions: List[str] = field(default_factory=list)
    evidence_gathering: List[str] = field(default_factory=list)
    analysis_methods: List[str] = field(default_factory=list)

    # Knowledge validation
    validation_methods: List[KnowledgeValidationType] = field(default_factory=list)
    validation_criteria: List[NormativeCriteria] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)  # Validation scores

    # Inquiry outcomes
    value_synthesis: List[str] = field(default_factory=list)  # Synthesized understanding
    normative_conclusions: List[str] = field(default_factory=list)
    policy_implications: List[str] = field(default_factory=list)
    action_recommendations: List[str] = field(default_factory=list)

    # Quality assessment
    inquiry_rigor: Optional[float] = None  # Rigor of inquiry process (0-1)
    stakeholder_representation: Optional[float] = None  # Representativeness (0-1)
    conclusion_validity: Optional[float] = None  # Validity of conclusions (0-1)

    # SFM integration
    matrix_value_implications: List[uuid.UUID] = field(default_factory=list)
    delivery_value_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    institutional_value_alignment: Dict[uuid.UUID, float] = field(default_factory=dict)

    def conduct_consequentialist_analysis(self) -> Dict[str, Any]:
        """Conduct consequentialist analysis of values and actions."""
        analysis = {
            'outcome_evaluation': {},
            'consequence_mapping': {},
            'value_realization_assessment': {},
            'unintended_consequences': []
        }

        # Evaluate outcomes against normative criteria
        for criteria in self.validation_criteria:
            if criteria == NormativeCriteria.PROBLEM_SOLVING_EFFECTIVENESS:
                # Assess problem-solving outcomes
                analysis['outcome_evaluation']['problem_solving'] = {
                    'effectiveness_score': 0.7,  # Placeholder - would be calculated
                    'problems_addressed': [],
                    'problems_created': []
                }
            elif criteria == NormativeCriteria.COMMUNITY_ENHANCEMENT:
                # Assess community impacts
                analysis['outcome_evaluation']['community_impact'] = {
                    'enhancement_score': 0.6,  # Placeholder
                    'positive_impacts': [],
                    'negative_impacts': []
                }

        # Map consequences to stakeholders
        for participant in self.inquiry_participants:
            analysis['consequence_mapping'][str(participant)] = {
                'positive_consequences': [],
                'negative_consequences': [],
                'net_impact_score': 0.0  # Placeholder
            }

        # Assess value realization
        for value_type in self.identified_values:
            analysis['value_realization_assessment'][value_type.name] = {
                'realization_score': 0.5,  # Placeholder
                'supporting_factors': [],
                'constraining_factors': []
            }

        return analysis

    def assess_instrumental_effectiveness(self) -> Dict[str, float]:
        """Assess instrumental effectiveness of inquiry outcomes."""
        effectiveness_assessment = {}

        # Problem-solving effectiveness
        if self.action_recommendations:
            problem_solving_score = min(len(self.action_recommendations) / 5.0, 1.0)
            effectiveness_assessment['problem_solving_effectiveness'] = problem_solving_score

        # Knowledge application effectiveness
        if self.evidence_gathering and self.analysis_methods:
            knowledge_score = min(
                len(self.evidence_gathering) * len(self.analysis_methods) / 10.0,
                1.0)
            effectiveness_assessment['knowledge_application'] = knowledge_score

        # Stakeholder inclusion effectiveness
        if self.inquiry_participants:
            inclusion_score = min(len(self.inquiry_participants) / 8.0, 1.0)
            effectiveness_assessment['stakeholder_inclusion'] = inclusion_score

        # Policy relevance effectiveness
        if self.policy_implications:
            policy_score = min(len(self.policy_implications) / 3.0, 1.0)
            effectiveness_assessment['policy_relevance'] = policy_score

        # Overall instrumental effectiveness
        if effectiveness_assessment:
            overall_score = sum(effectiveness_assessment.values()) / len(effectiveness_assessment)
            effectiveness_assessment['overall_instrumental_effectiveness'] = overall_score

        return effectiveness_assessment

@dataclass
class KnowledgeValidation(Node):  # pylint: disable=too-many-instance-attributes
    """Knowledge validation process within value inquiry."""

    knowledge_claim: Optional[str] = None  # Knowledge being validated
    validation_type: Optional[KnowledgeValidationType] = None
    validation_context: Optional[str] = None

    # Validation process
    validation_methods: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)

    # Validation execution
    evidence_collected: Dict[str, Any] = field(default_factory=dict)
    validation_tests: List[Dict[str, Any]] = field(default_factory=list)
    expert_evaluations: Dict[uuid.UUID, float] = field(default_factory=dict)

    # Validation outcomes
    validation_score: Optional[float] = None  # Overall validation score (0-1)
    validity_confidence: Optional[float] = None  # Confidence in validity (0-1)
    validation_limitations: List[str] = field(default_factory=list)

    # Contextual factors
    contextual_applicability: Optional[float] = None  # How applicable in context (0-1)
    temporal_validity: Optional[str] = None  # Time-boundedness of validity
    spatial_validity: Optional[str] = None  # Geographic applicability

    # Instrumental assessment
    practical_utility: Optional[float] = None  # Practical usefulness (0-1)
    problem_solving_relevance: Optional[float] = None  # Relevance to problems (0-1)
    action_guidance_value: Optional[float] = None  # Value for guiding action (0-1)

    def validate_knowledge_instrumentally(self) -> Dict[str, float]:
        """Validate knowledge using instrumental criteria."""
        instrumental_validation = {}

        # Practical effectiveness validation
        if self.practical_utility is not None:
            instrumental_validation['practical_effectiveness'] = self.practical_utility

        # Problem-solving relevance validation
        if self.problem_solving_relevance is not None:
            instrumental_validation['problem_solving_relevance'] = self.problem_solving_relevance

        # Contextual appropriateness validation
        if self.contextual_applicability is not None:
            instrumental_validation['contextual_appropriateness'] = self.contextual_applicability

        # Action guidance validation
        if self.action_guidance_value is not None:
            instrumental_validation['action_guidance'] = self.action_guidance_value

        # Evidence quality validation
        if self.evidence_collected:
            evidence_quality = min(len(self.evidence_collected) / 5.0, 1.0)
            instrumental_validation['evidence_quality'] = evidence_quality

        # Overall instrumental validity
        if instrumental_validation:
            overall_validity = sum(instrumental_validation.values()) / len(instrumental_validation)
            instrumental_validation['overall_instrumental_validity'] = overall_validity
            self.validation_score = overall_validity

        return instrumental_validation

@dataclass
class NormativeEvaluation(Node):  # pylint: disable=too-many-instance-attributes
    """Normative evaluation within value inquiry framework."""

    evaluation_focus: Optional[str] = None  # What is being evaluated
    normative_criteria: List[NormativeCriteria] = field(default_factory=list)
    evaluation_participants: List[uuid.UUID] = field(default_factory=list)

    # Evaluation context
    institutional_context: List[uuid.UUID] = field(default_factory=list)
    cultural_context: List[str] = field(default_factory=list)
    temporal_scope: Optional[str] = None

    # Criteria application
    criteria_weights: Dict[NormativeCriteria, float] = field(default_factory=dict)
    criteria_scores: Dict[NormativeCriteria, float] = field(default_factory=dict)
    criteria_justifications: Dict[NormativeCriteria, str] = field(default_factory=dict)

    # Stakeholder perspectives
    stakeholder_evaluations: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    consensus_areas: List[str] = field(default_factory=list)
    disagreement_areas: List[str] = field(default_factory=list)

    # Evaluation outcomes
    overall_normative_score: Optional[float] = None  # Overall evaluation score (0-1)
    normative_conclusions: List[str] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)

    # Quality indicators
    evaluation_comprehensiveness: Optional[float] = None  # Comprehensiveness (0-1)
    stakeholder_representativeness: Optional[float] = None  # Representativeness (0-1)
    evaluation_rigor: Optional[float] = None  # Methodological rigor (0-1)

    def calculate_weighted_normative_score(self) -> Optional[float]:
        """Calculate weighted normative evaluation score."""
        if not self.criteria_weights or not self.criteria_scores:
            return None

        weighted_scores = []
        for criteria, weight in self.criteria_weights.items():
            if criteria in self.criteria_scores:
                weighted_score = self.criteria_scores[criteria] * weight
                weighted_scores.append(weighted_score)

        if weighted_scores:
            total_score = sum(weighted_scores)
            total_weight = sum(self.criteria_weights.values())

            if total_weight > 0:
                normalized_score = total_score / total_weight
                self.overall_normative_score = normalized_score
                return normalized_score

        return None

    def identify_normative_tensions(self) -> List[Dict[str, Any]]:
        """Identify tensions between normative criteria."""
        tensions = []

        criteria_list = list(self.criteria_scores.keys())

        # Check for common normative tensions
        for i, criteria1 in enumerate(criteria_list):
            for criteria2 in criteria_list[i+1:]:
                tension_type = self._identify_tension_type(criteria1, criteria2)
                if tension_type:
                    score1 = self.criteria_scores.get(criteria1, 0)
                    score2 = self.criteria_scores.get(criteria2, 0)

                    # Tension intensity based on score differences
                    tension_intensity = abs(score1 - score2)

                    tensions.append({
                        'criteria1': criteria1.name,
                        'criteria2': criteria2.name,
                        'tension_type': tension_type,
                        'tension_intensity': tension_intensity,
                        'resolution_approaches': self._suggest_tension_resolution(criteria1, criteria2)
                    })

        return tensions

    def _identify_tension_type(
        self,
        criteria1: NormativeCriteria,
        criteria2: NormativeCriteria) -> Optional[str]:
        """Identify type of tension between two criteria."""
        tension_patterns = {
            (
                NormativeCriteria.ECONOMIC_EFFICIENCY,
                NormativeCriteria.SOCIAL_EQUITY): "efficiency_equity_tension",
            (
                NormativeCriteria.ECONOMIC_EFFICIENCY,
                NormativeCriteria.ECOLOGICAL_SUSTAINABILITY): "economy_environment_tension",
            (
                NormativeCriteria.CULTURAL_CONTINUITY,
                NormativeCriteria.ADAPTIVE_CAPACITY): "tradition_change_tension",
            (
                NormativeCriteria.DEMOCRATIC_PARTICIPATION,
                NormativeCriteria.ECONOMIC_EFFICIENCY): "democracy_efficiency_tension"
        }

        # Check both directions
        tension_key = (criteria1, criteria2)
        reverse_key = (criteria2, criteria1)

        return tension_patterns.get(tension_key) or tension_patterns.get(reverse_key)

    def _suggest_tension_resolution(
        self,
        criteria1: NormativeCriteria,
        criteria2: NormativeCriteria) -> List[str]:
        """Suggest approaches for resolving normative tensions."""
        resolution_suggestions = [
            "Seek complementary solutions that serve both criteria",
            "Identify temporal sequencing that addresses both concerns",
            "Explore innovative approaches that transcend the apparent trade-off",
            "Engage stakeholders in collaborative problem-solving",
            "Consider multi-level solutions addressing different scales"
        ]
        return resolution_suggestions[:3]  # Return top 3 suggestions

@dataclass
class ValueSynthesis(Node):  # pylint: disable=too-many-instance-attributes
    """Synthesis of value inquiry processes and outcomes."""

    synthesis_scope: Optional[str] = None  # Scope of synthesis
    contributing_inquiries: List[uuid.UUID] = field(default_factory=list)
    synthesis_participants: List[uuid.UUID] = field(default_factory=list)

    # Synthesis process
    synthesis_method: Optional[str] = None  # Method used for synthesis
    integration_approach: Optional[str] = None  # How different perspectives integrated

    # Synthesized understanding
    integrated_value_framework: Dict[str, Any] = field(default_factory=dict)
    value_priorities: List[Tuple[str, float]] = field(default_factory=list)  # Value, priority score
    value_relationships: Dict[str, List[str]] = field(default_factory=dict)  # Value interdependencies

    # Normative conclusions
    synthesized_principles: List[str] = field(default_factory=list)
    action_guidelines: List[str] = field(default_factory=list)
    institutional_recommendations: List[str] = field(default_factory=list)

    # Implementation guidance
    implementation_strategies: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)
    monitoring_approaches: List[str] = field(default_factory=list)

    # Quality assessment
    synthesis_coherence: Optional[float] = None  # Internal coherence (0-1)
    practical_applicability: Optional[float] = None  # Practical usefulness (0-1)
    stakeholder_acceptance: Optional[float] = None  # Stakeholder buy-in (0-1)

    # SFM integration
    matrix_synthesis_implications: List[uuid.UUID] = field(default_factory=list)
    delivery_synthesis_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    institutional_synthesis_alignment: Dict[uuid.UUID, float] = field(default_factory=dict)

@dataclass
class InstrumentalistInquiryProcess(Node):  # pylint: disable=too-many-instance-attributes
    """Comprehensive instrumentalist inquiry process integrating value inquiry."""

    inquiry_purpose: Optional[str] = None  # Overall purpose of inquiry
    problem_focus: Optional[str] = None  # Central problem being addressed

    # Inquiry components
    value_inquiries: List[uuid.UUID] = field(default_factory=list)
    knowledge_validations: List[uuid.UUID] = field(default_factory=list)
    normative_evaluations: List[uuid.UUID] = field(default_factory=list)
    value_synthesis: Optional[uuid.UUID] = None

    # Instrumentalist principles
    problem_orientation: Optional[float] = None  # Problem-solving focus (0-1)
    consequentialist_emphasis: Optional[float] = None  # Outcome focus (0-1)
    pragmatic_orientation: Optional[float] = None  # Practical effectiveness focus (0-1)
    democratic_participation: Optional[float] = None  # Participatory approach (0-1)

    # Process characteristics
    inquiry_rigor: Optional[float] = None  # Methodological rigor (0-1)
    stakeholder_inclusiveness: Optional[float] = None  # Inclusiveness (0-1)
    contextual_sensitivity: Optional[float] = None  # Context awareness (0-1)

    # Outcomes
    policy_recommendations: List[str] = field(default_factory=list)
    institutional_adjustments: List[str] = field(default_factory=list)
    action_priorities: List[Tuple[str, float]] = field(default_factory=list)  # Action, priority

    # Learning and adaptation
    inquiry_lessons: List[str] = field(default_factory=list)
    methodological_improvements: List[str] = field(default_factory=list)
    future_inquiry_needs: List[str] = field(default_factory=list)

    # SFM integration
    matrix_inquiry_integration: Optional[float] = None  # Integration with matrix (0-1)
    delivery_inquiry_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    institutional_inquiry_effects: List[uuid.UUID] = field(default_factory=list)

    def assess_instrumentalist_quality(self) -> Dict[str, float]:
        """Assess quality of instrumentalist inquiry process."""
        quality_assessment = {}

        # Problem orientation assessment
        if self.problem_orientation is not None:
            quality_assessment['problem_orientation'] = self.problem_orientation

        # Consequentialist assessment
        if self.consequentialist_emphasis is not None:
            quality_assessment['consequentialist_approach'] = self.consequentialist_emphasis

        # Pragmatic effectiveness assessment
        if self.pragmatic_orientation is not None:
            quality_assessment['pragmatic_effectiveness'] = self.pragmatic_orientation

        # Democratic participation assessment
        if self.democratic_participation is not None:
            quality_assessment['democratic_quality'] = self.democratic_participation

        # Methodological rigor assessment
        if self.inquiry_rigor is not None:
            quality_assessment['methodological_rigor'] = self.inquiry_rigor

        # Overall instrumentalist quality
        if quality_assessment:
            overall_quality = sum(quality_assessment.values()) / len(quality_assessment)
            quality_assessment['overall_instrumentalist_quality'] = overall_quality

        return quality_assessment

    def generate_inquiry_recommendations(self) -> Dict[str, List[str]]:
        """Generate recommendations based on inquiry outcomes."""
        recommendations = {
            'immediate_actions': [],
            'institutional_changes': [],
            'policy_modifications': [],
            'further_inquiries': [],
            'monitoring_activities': []
        }

        # Immediate actions from policy recommendations
        recommendations['immediate_actions'] = self.policy_recommendations[:3]  # Top 3

        # Institutional changes
        recommendations['institutional_changes'] = self.institutional_adjustments

        # Policy modifications from action priorities
        high_priority_actions = [action for action, priority in self.action_priorities if priority > 0.7]
        recommendations['policy_modifications'] = high_priority_actions

        # Further inquiries
        recommendations['further_inquiries'] = self.future_inquiry_needs

        # Monitoring activities
        if hasattr(self, 'monitoring_needs'):
            recommendations['monitoring_activities'] = getattr(self, 'monitoring_needs', [])

        return recommendations
