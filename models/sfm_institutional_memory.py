"""
Institutional Memory and Knowledge Management Framework for Social Fabric Matrix analysis.

This module implements comprehensive institutional memory and knowledge management
systems following Hayden's approach to institutional learning and continuity.
Essential for preserving institutional knowledge, facilitating organizational
learning, and supporting continuous improvement in SFM-based institutional development.

Key Components:
- InstitutionalMemory: Comprehensive memory management system
- KnowledgeRepository: Structured knowledge storage and retrieval
- LearningSystem: Systematic learning and adaptation processes
- KnowledgeTransfer: Knowledge sharing and dissemination
- InstitutionalLearningAssessment: Evaluation of learning effectiveness
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
from models.meta_entities import TimeSlice
from models.sfm_enums import (
    InstitutionalScope,
    KnowledgeType,
    LearningMethod,
    InformationSystem,
)

class MemoryType(Enum):
    """Types of institutional memory."""

    PROCEDURAL = auto()         # How to do things
    DECLARATIVE = auto()        # Facts and information
    EXPERIENTIAL = auto()       # Lessons from experience
    RELATIONAL = auto()         # Relationship knowledge
    CONTEXTUAL = auto()         # Situational understanding
    STRATEGIC = auto()          # Strategic insights and wisdom

class KnowledgeCategory(Enum):
    """Categories of institutional knowledge."""

    BEST_PRACTICES = auto()     # Proven effective approaches
    LESSONS_LEARNED = auto()    # Learning from failures/successes
    STANDARD_PROCEDURES = auto() # Established procedures
    HISTORICAL_PRECEDENTS = auto() # Past decisions and outcomes
    STAKEHOLDER_INSIGHTS = auto() # Stakeholder knowledge
    TECHNICAL_EXPERTISE = auto()  # Technical know-how
    CULTURAL_KNOWLEDGE = auto()   # Organizational culture insights

class LearningLevel(Enum):
    """Levels of organizational learning."""

    SINGLE_LOOP = auto()        # Error correction within existing framework
    DOUBLE_LOOP = auto()        # Questioning underlying assumptions
    TRIPLE_LOOP = auto()        # Learning about learning itself
    TRANSFORMATIONAL = auto()   # Fundamental paradigm shifts

class KnowledgeQuality(Enum):
    """Quality levels of institutional knowledge."""

    EXCELLENT = auto()          # High-quality, verified knowledge
    GOOD = auto()              # Reliable, well-documented knowledge
    FAIR = auto()              # Adequate but incomplete knowledge
    POOR = auto()              # Low-quality or questionable knowledge
    UNVERIFIED = auto()        # Unverified or anecdotal knowledge

class TransferMechanism(Enum):
    """Mechanisms for knowledge transfer."""

    FORMAL_TRAINING = auto()    # Structured training programs
    MENTORING = auto()         # One-on-one mentoring
    DOCUMENTATION = auto()     # Written documentation
    COMMUNITIES_OF_PRACTICE = auto() # Professional communities
    STORYTELLING = auto()      # Narrative knowledge sharing
    APPRENTICESHIP = auto()    # Learning by doing
    TECHNOLOGY_SYSTEMS = auto() # Digital knowledge systems

@dataclass
class KnowledgeAsset(Node):
    """Individual knowledge asset within institutional memory."""

    knowledge_category: KnowledgeCategory = KnowledgeCategory.LESSONS_LEARNED
    memory_type: MemoryType = MemoryType.EXPERIENTIAL
    knowledge_quality: KnowledgeQuality = KnowledgeQuality.FAIR

    # Content description
    knowledge_title: Optional[str] = None
    knowledge_description: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)

    # Context and provenance
    creation_date: Optional[datetime] = None
    knowledge_source: Optional[uuid.UUID] = None  # Stakeholder or institution
    contextual_factors: List[str] = field(default_factory=list)
    situational_specificity: Optional[float] = None  # How context-specific (0-1)

    # Content structure
    explicit_knowledge: List[str] = field(default_factory=list)  # Codifiable knowledge
    tacit_knowledge: List[str] = field(default_factory=list)    # Implicit understanding
    procedural_steps: List[str] = field(default_factory=list)   # How-to knowledge

    # Validation and verification
    verification_methods: List[str] = field(default_factory=list)
    validation_sources: List[uuid.UUID] = field(default_factory=list)
    reliability_indicators: Dict[str, float] = field(default_factory=dict)

    # Usage and application
    application_contexts: List[str] = field(default_factory=list)
    usage_frequency: Optional[float] = None  # How often used
    success_rate: Optional[float] = None     # Success when applied (0-1)

    # Relationships
    related_knowledge: List[uuid.UUID] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    conflicting_knowledge: List[uuid.UUID] = field(default_factory=list)

    # Evolution and updates
    knowledge_updates: List[Dict[str, Any]] = field(default_factory=list)
    obsolescence_indicators: List[str] = field(default_factory=list)
    currency_assessment: Optional[float] = None  # How current (0-1)

    # Access and sharing
    access_restrictions: List[str] = field(default_factory=list)
    sharing_protocols: List[str] = field(default_factory=list)
    transfer_mechanisms: List[TransferMechanism] = field(default_factory=list)

    # SFM integration
    matrix_knowledge_relevance: List[uuid.UUID] = field(default_factory=list)
    institutional_knowledge_connections: List[uuid.UUID] = field(default_factory=list)
    delivery_system_knowledge_applications: List[uuid.UUID] = field(default_factory=list)

    def assess_knowledge_value(self) -> Dict[str, Any]:
        """Assess the value of this knowledge asset."""
        value_assessment = {
            'value_score': 0.0,
            'value_dimensions': {},
            'value_drivers': [],
            'enhancement_opportunities': []
        }

        # Value dimensions
        dimensions = {
            'quality': self._map_quality_to_score(self.knowledge_quality),
            'applicability': len(self.application_contexts) / 10.0 if self.application_contexts else 0.1,
            'reliability': sum(self.reliability_indicators.values()) / len(self.reliability_indicators) if self.reliability_indicators else 0.5,
            'currency': self.currency_assessment or 0.5,
            'usage': self.usage_frequency or 0.3
        }

        value_assessment['value_dimensions'] = dimensions
        value_assessment['value_score'] = sum(dimensions.values()) / len(dimensions)

        # Value drivers
        if dimensions['quality'] > 0.8:
            value_assessment['value_drivers'].append('High knowledge quality')
        if dimensions['applicability'] > 0.7:
            value_assessment['value_drivers'].append('Broad applicability')
        if dimensions['usage'] > 0.6:
            value_assessment['value_drivers'].append('Frequent usage')

        # Enhancement opportunities
        if dimensions['currency'] < 0.5:
            value_assessment['enhancement_opportunities'].append('Update knowledge currency')
        if dimensions['reliability'] < 0.6:
            value_assessment['enhancement_opportunities'].append('Strengthen validation')

        return value_assessment

    def _map_quality_to_score(self, quality: KnowledgeQuality) -> float:
        """Map knowledge quality enum to numeric score."""
        quality_scores = {
            KnowledgeQuality.EXCELLENT: 1.0,
            KnowledgeQuality.GOOD: 0.8,
            KnowledgeQuality.FAIR: 0.6,
            KnowledgeQuality.POOR: 0.4,
            KnowledgeQuality.UNVERIFIED: 0.2
        }
        return quality_scores.get(quality, 0.5)

@dataclass
class KnowledgeRepository(Node):
    """Structured repository for institutional knowledge storage and retrieval."""

    repository_scope: Optional[str] = None
    repository_purpose: Optional[str] = None

    # Content organization
    knowledge_assets: List[uuid.UUID] = field(default_factory=list)  # KnowledgeAsset IDs
    knowledge_taxonomy: Dict[str, List[uuid.UUID]] = field(default_factory=dict)
    thematic_collections: Dict[str, List[uuid.UUID]] = field(default_factory=dict)

    # Repository structure
    organizational_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    metadata_schema: Dict[str, str] = field(default_factory=dict)
    indexing_system: List[str] = field(default_factory=list)

    # Access and retrieval
    search_capabilities: List[str] = field(default_factory=list)
    retrieval_interfaces: List[str] = field(default_factory=list)
    user_access_patterns: Dict[uuid.UUID, List[str]] = field(default_factory=dict)

    # Quality management
    curation_standards: List[str] = field(default_factory=list)
    quality_control_processes: List[str] = field(default_factory=list)
    peer_review_mechanisms: List[str] = field(default_factory=list)

    # Repository maintenance
    content_lifecycle_management: List[str] = field(default_factory=list)
    update_procedures: List[str] = field(default_factory=list)
    archival_policies: List[str] = field(default_factory=list)

    # Usage analytics
    access_statistics: Dict[str, int] = field(default_factory=dict)
    popular_content: List[uuid.UUID] = field(default_factory=list)
    usage_patterns: Dict[str, Any] = field(default_factory=dict)

    # Integration capabilities
    external_system_connections: List[str] = field(default_factory=list)
    interoperability_standards: List[str] = field(default_factory=list)
    data_exchange_protocols: List[str] = field(default_factory=list)

    def evaluate_repository_effectiveness(self) -> Dict[str, Any]:
        """Evaluate effectiveness of knowledge repository."""
        effectiveness_evaluation = {
            'effectiveness_score': 0.0,
            'performance_dimensions': {},
            'usage_insights': {},
            'improvement_priorities': []
        }

        # Performance dimensions
        total_assets = len(self.knowledge_assets)
        if total_assets > 0:
            # Content richness
            effectiveness_evaluation['performance_dimensions']['content_richness'] = min(1.0, total_assets / 100)

            # Organization quality
            organized_content = sum(len(assets) for assets in self.knowledge_taxonomy.values())
            organization_ratio = organized_content / total_assets if total_assets > 0 else 0
            effectiveness_evaluation['performance_dimensions']['organization_quality'] = organization_ratio

            # Usage intensity
            total_access = sum(self.access_statistics.values()) if self.access_statistics else 0
            usage_per_asset = total_access / total_assets if total_assets > 0 else 0
            effectiveness_evaluation['performance_dimensions']['usage_intensity'] = min(1.0, usage_per_asset / 10)

        # Overall effectiveness
        dimensions = effectiveness_evaluation['performance_dimensions']
        if dimensions:
            effectiveness_evaluation['effectiveness_score'] = sum(dimensions.values()) / len(dimensions)

        # Usage insights
        if self.popular_content:
            effectiveness_evaluation['usage_insights']['high_value_content'] = len(self.popular_content)

        # Improvement priorities
        if effectiveness_evaluation['effectiveness_score'] < 0.6:
            effectiveness_evaluation['improvement_priorities'].extend([
                'Enhance content organization',
                'Improve search and retrieval',
                'Increase user engagement'
            ])

        return effectiveness_evaluation

@dataclass
class LearningSystem(Node):
    """Systematic learning and adaptation process for institutional development."""

    learning_scope: Optional[str] = None
    learning_objectives: List[str] = field(default_factory=list)

    # Learning design
    learning_methods: List[LearningMethod] = field(default_factory=list)
    learning_cycles: List[str] = field(default_factory=list)
    feedback_mechanisms: List[str] = field(default_factory=list)

    # Learning levels
    single_loop_learning: List[str] = field(default_factory=list)  # Error correction
    double_loop_learning: List[str] = field(default_factory=list) # Assumption questioning
    triple_loop_learning: List[str] = field(default_factory=list) # Meta-learning

    # Learning processes
    experimentation_protocols: List[str] = field(default_factory=list)
    reflection_practices: List[str] = field(default_factory=list)
    knowledge_synthesis_methods: List[str] = field(default_factory=list)

    # Learning sources
    internal_learning_sources: List[str] = field(default_factory=list)
    external_learning_sources: List[str] = field(default_factory=list)
    collaborative_learning_partnerships: List[uuid.UUID] = field(default_factory=list)

    # Learning outcomes
    learning_achievements: List[str] = field(default_factory=list)
    capability_improvements: Dict[str, float] = field(default_factory=dict)
    behavioral_changes: List[str] = field(default_factory=list)

    # Organizational context
    learning_culture_indicators: List[str] = field(default_factory=list)
    learning_barriers: List[str] = field(default_factory=list)
    learning_enablers: List[str] = field(default_factory=list)

    # Systematic improvement
    continuous_improvement_processes: List[str] = field(default_factory=list)
    innovation_initiatives: List[str] = field(default_factory=list)
    adaptation_strategies: List[str] = field(default_factory=list)

    # Performance measurement
    learning_effectiveness_indicators: Dict[str, float] = field(default_factory=dict)
    learning_ROI_assessment: Optional[float] = None  # Return on learning investment
    institutional_performance_impact: Optional[float] = None  # Impact on performance

    def assess_learning_effectiveness(self) -> Dict[str, Any]:
        """Assess effectiveness of institutional learning system."""
        learning_assessment = {
            'learning_effectiveness_score': 0.0,
            'learning_dimensions': {},
            'learning_strengths': [],
            'learning_gaps': [],
            'enhancement_recommendations': []
        }

        # Learning dimensions
        dimensions = {}

        # Learning breadth
        total_methods = len(self.learning_methods)
        dimensions['learning_breadth'] = min(1.0, total_methods / 5.0) if total_methods > 0 else 0.2

        # Learning depth
        learning_levels = [
            len(self.single_loop_learning),
            len(self.double_loop_learning),
            len(self.triple_loop_learning)
        ]
        dimensions['learning_depth'] = sum(level > 0 for level in learning_levels) / 3.0

        # Learning impact
        if self.institutional_performance_impact is not None:
            dimensions['learning_impact'] = self.institutional_performance_impact
        else:
            # Estimate from capability improvements
            if self.capability_improvements:
                avg_improvement = sum(self.capability_improvements.values()) / len(self.capability_improvements)
                dimensions['learning_impact'] = avg_improvement
            else:
                dimensions['learning_impact'] = 0.5

        learning_assessment['learning_dimensions'] = dimensions

        # Overall effectiveness
        learning_assessment['learning_effectiveness_score'] = sum(dimensions.values()) / len(dimensions)

        # Identify strengths and gaps
        for dimension, score in dimensions.items():
            if score >= 0.8:
                learning_assessment['learning_strengths'].append(
                    f"Strong {dimension.replace('_',
                    ' ')}")
            elif score <= 0.4:
                learning_assessment['learning_gaps'].append(f"Weak {dimension.replace('_', ' ')}")

        # Enhancement recommendations
        if learning_assessment['learning_effectiveness_score'] < 0.6:
            learning_assessment['enhancement_recommendations'].extend([
                'Diversify learning methods',
                'Strengthen higher-order learning',
                'Improve learning impact measurement',
                'Address learning barriers'
            ])

        return learning_assessment

@dataclass
class KnowledgeTransfer(Node):
    """Knowledge sharing and dissemination system."""

    transfer_scope: Optional[str] = None
    transfer_objectives: List[str] = field(default_factory=list)

    # Transfer design
    transfer_mechanisms: List[TransferMechanism] = field(default_factory=list)
    target_audiences: List[uuid.UUID] = field(default_factory=list)
    transfer_channels: List[str] = field(default_factory=list)

    # Content preparation
    knowledge_packaging: List[str] = field(default_factory=list)
    communication_formats: List[str] = field(default_factory=list)
    accessibility_adaptations: List[str] = field(default_factory=list)

    # Transfer processes
    formal_transfer_programs: List[str] = field(default_factory=list)
    informal_transfer_opportunities: List[str] = field(default_factory=list)
    peer_to_peer_transfer: List[str] = field(default_factory=list)

    # Quality assurance
    transfer_quality_standards: List[str] = field(default_factory=list)
    fidelity_maintenance: List[str] = field(default_factory=list)
    adaptation_guidelines: List[str] = field(default_factory=list)

    # Recipient preparation
    recipient_readiness_assessment: Dict[uuid.UUID, float] = field(default_factory=dict)
    capacity_building_support: List[str] = field(default_factory=list)
    contextual_adaptation_support: List[str] = field(default_factory=list)

    # Transfer outcomes
    transfer_success_rates: Dict[str, float] = field(default_factory=dict)
    knowledge_uptake_indicators: List[str] = field(default_factory=list)
    application_success_stories: List[str] = field(default_factory=list)

    # Feedback and improvement
    transfer_feedback_mechanisms: List[str] = field(default_factory=list)
    continuous_improvement_processes: List[str] = field(default_factory=list)
    transfer_innovation_initiatives: List[str] = field(default_factory=list)

    def evaluate_transfer_effectiveness(self) -> Dict[str, Any]:
        """Evaluate effectiveness of knowledge transfer."""
        transfer_evaluation = {
            'transfer_effectiveness_score': 0.0,
            'transfer_dimensions': {},
            'successful_transfers': [],
            'transfer_challenges': [],
            'improvement_strategies': []
        }

        # Transfer dimensions
        dimensions = {}

        # Reach effectiveness
        total_targets = len(self.target_audiences)
        if total_targets > 0 and self.recipient_readiness_assessment:
            reached_targets = len(self.recipient_readiness_assessment)
            dimensions['reach_effectiveness'] = reached_targets / total_targets
        else:
            dimensions['reach_effectiveness'] = 0.5

        # Quality effectiveness
        if self.transfer_success_rates:
            avg_success_rate = sum(self.transfer_success_rates.values()) / len(self.transfer_success_rates)
            dimensions['quality_effectiveness'] = avg_success_rate
        else:
            dimensions['quality_effectiveness'] = 0.5

        # Application effectiveness
        application_indicators = len(self.application_success_stories)
        dimensions['application_effectiveness'] = min(1.0, application_indicators / 10.0)

        transfer_evaluation['transfer_dimensions'] = dimensions

        # Overall effectiveness
        transfer_evaluation['transfer_effectiveness_score'] = sum(dimensions.values()) / len(dimensions)

        # Identify successful transfers and challenges
        for mechanism, success_rate in self.transfer_success_rates.items():
            if success_rate > 0.8:
                transfer_evaluation['successful_transfers'].append(mechanism)
            elif success_rate < 0.4:
                transfer_evaluation['transfer_challenges'].append(f"Low success in {mechanism}")

        # Improvement strategies
        if transfer_evaluation['transfer_effectiveness_score'] < 0.6:
            transfer_evaluation['improvement_strategies'].extend([
                'Enhance recipient preparation',
                'Improve transfer methods',
                'Strengthen follow-up support',
                'Better contextual adaptation'
            ])

        return transfer_evaluation

@dataclass
class InstitutionalMemory(Node):
    """Comprehensive institutional memory management system."""

    memory_scope: Optional[str] = None
    institutional_context: List[uuid.UUID] = field(default_factory=list)

    # Memory components
    knowledge_repositories: List[uuid.UUID] = field(default_factory=list)  # KnowledgeRepository IDs
    learning_systems: List[uuid.UUID] = field(default_factory=list)        # LearningSystem IDs
    transfer_systems: List[uuid.UUID] = field(default_factory=list)        # KnowledgeTransfer IDs

    # Memory architecture
    memory_domains: Dict[str, List[uuid.UUID]] = field(default_factory=dict)
    cross_domain_connections: List[Tuple[str, str]] = field(default_factory=list)
    memory_integration_mechanisms: List[str] = field(default_factory=list)

    # Organizational integration
    institutional_embedding: Dict[uuid.UUID, float] = field(default_factory=dict)
    cultural_integration: Optional[float] = None  # How well integrated into culture
    structural_integration: Optional[float] = None  # How well integrated into structure

    # Memory maintenance
    memory_curation_processes: List[str] = field(default_factory=list)
    obsolescence_management: List[str] = field(default_factory=list)
    continuous_updating_mechanisms: List[str] = field(default_factory=list)

    # Memory governance
    governance_structures: List[str] = field(default_factory=list)
    access_policies: List[str] = field(default_factory=list)
    quality_standards: List[str] = field(default_factory=list)

    # Performance metrics
    memory_utilization_rate: Optional[float] = None  # How actively used (0-1)
    memory_contribution_to_performance: Optional[float] = None  # Performance impact
    institutional_learning_velocity: Optional[float] = None  # Speed of learning

    # Strategic value
    competitive_advantage_contribution: Optional[float] = None  # Strategic value
    innovation_support_effectiveness: Optional[float] = None   # Innovation support
    resilience_enhancement: Optional[float] = None             # Organizational resilience

    # SFM integration
    matrix_memory_integration: List[uuid.UUID] = field(default_factory=list)
    institutional_memory_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_system_memory_connections: List[uuid.UUID] = field(default_factory=list)

    def assess_institutional_memory_effectiveness(self) -> Dict[str, Any]:
        """Assess overall effectiveness of institutional memory system."""
        memory_assessment = {
            'memory_effectiveness_score': 0.0,
            'effectiveness_dimensions': {},
            'memory_strengths': [],
            'memory_weaknesses': [],
            'strategic_recommendations': []
        }

        # Effectiveness dimensions
        dimensions = {
            'utilization': self.memory_utilization_rate or 0.5,
            'performance_contribution': self.memory_contribution_to_performance or 0.5,
            'learning_velocity': self.institutional_learning_velocity or 0.5,
            'cultural_integration': self.cultural_integration or 0.5,
            'structural_integration': self.structural_integration or 0.5
        }

        memory_assessment['effectiveness_dimensions'] = dimensions

        # Overall effectiveness
        memory_assessment['memory_effectiveness_score'] = sum(dimensions.values()) / len(dimensions)

        # Identify strengths and weaknesses
        for dimension, score in dimensions.items():
            if score >= 0.8:
                memory_assessment['memory_strengths'].append(
                    f"Strong {dimension.replace('_',
                    ' ')}")
            elif score <= 0.4:
                memory_assessment['memory_weaknesses'].append(f"Weak {dimension.replace('_', ' ')}")

        # Strategic recommendations
        overall_score = memory_assessment['memory_effectiveness_score']
        if overall_score >= 0.8:
            memory_assessment['strategic_recommendations'] = [
                'Maintain excellence in memory management',
                'Share best practices with other institutions',
                'Explore advanced memory technologies'
            ]
        elif overall_score >= 0.6:
            memory_assessment['strategic_recommendations'] = [
                'Strengthen weak dimensions',
                'Enhance integration mechanisms',
                'Improve utilization rates'
            ]
        else:
            memory_assessment['strategic_recommendations'] = [
                'Fundamental redesign of memory system',
                'Invest in cultural change for learning',
                'Strengthen governance and quality standards'
            ]

        return memory_assessment

@dataclass
class InstitutionalLearningAssessment(Node):
    """Assessment of institutional learning effectiveness and outcomes."""

    assessment_scope: Optional[str] = None
    assessment_timeframe: Optional[TimeSlice] = None

    # Assessment framework
    learning_indicators: Dict[str, str] = field(default_factory=dict)
    measurement_methods: List[str] = field(default_factory=list)
    assessment_standards: List[str] = field(default_factory=list)

    # Learning outcomes measurement
    knowledge_acquisition_rates: Dict[str, float] = field(default_factory=dict)
    capability_development_progress: Dict[str, float] = field(default_factory=dict)
    behavioral_change_indicators: List[str] = field(default_factory=list)

    # Organizational learning assessment
    learning_culture_maturity: Optional[float] = None  # Culture maturity (0-1)
    learning_system_effectiveness: Optional[float] = None  # System effectiveness
    knowledge_sharing_intensity: Optional[float] = None   # Sharing activity level

    # Impact assessment
    performance_improvements: Dict[str, float] = field(default_factory=dict)
    innovation_outcomes: List[str] = field(default_factory=list)
    problem_solving_enhancement: Optional[float] = None  # Problem-solving improvement

    # Comparative analysis
    peer_institution_comparisons: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    best_practice_identification: List[str] = field(default_factory=list)
    benchmark_performance: Dict[str, float] = field(default_factory=dict)

    # Learning system quality
    learning_process_effectiveness: Dict[str, float] = field(default_factory=dict)
    knowledge_quality_assessment: Optional[float] = None  # Knowledge quality
    transfer_effectiveness: Optional[float] = None        # Transfer effectiveness

    def conduct_comprehensive_learning_assessment(self) -> Dict[str, Any]:
        """Conduct comprehensive assessment of institutional learning."""
        learning_assessment = {
            'overall_learning_score': 0.0,
            'learning_performance_dimensions': {},
            'key_achievements': [],
            'improvement_areas': [],
            'strategic_priorities': []
        }

        # Learning performance dimensions
        dimensions = {
            'culture_maturity': self.learning_culture_maturity or 0.5,
            'system_effectiveness': self.learning_system_effectiveness or 0.5,
            'knowledge_sharing': self.knowledge_sharing_intensity or 0.5,
            'performance_impact': sum(self.performance_improvements.values()) / len(self.performance_improvements) if self.performance_improvements else 0.5,
            'knowledge_quality': self.knowledge_quality_assessment or 0.5
        }

        learning_assessment['learning_performance_dimensions'] = dimensions

        # Overall learning score
        learning_assessment['overall_learning_score'] = sum(dimensions.values()) / len(dimensions)

        # Key achievements
        for dimension, score in dimensions.items():
            if score >= 0.8:
                learning_assessment['key_achievements'].append(
                    f"Excellent {dimension.replace('_',
                    ' ')}")

        # Improvement areas
        for dimension, score in dimensions.items():
            if score <= 0.4:
                learning_assessment['improvement_areas'].append(
                    f"Strengthen {dimension.replace('_',
                    ' ')}")

        # Strategic priorities
        overall_score = learning_assessment['overall_learning_score']
        if overall_score >= 0.8:
            learning_assessment['strategic_priorities'] = [
                'Sustain learning excellence',
                'Mentor other institutions',
                'Pioneer advanced learning approaches'
            ]
        elif overall_score >= 0.6:
            learning_assessment['strategic_priorities'] = [
                'Address specific improvement areas',
                'Strengthen learning culture',
                'Enhance knowledge sharing'
            ]
        else:
            learning_assessment['strategic_priorities'] = [
                'Fundamental learning system reform',
                'Cultural transformation initiative',
                'Leadership development for learning'
            ]

        return learning_assessment
