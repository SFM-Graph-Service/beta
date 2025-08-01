"""
Social Beliefs-Institution Integration Framework for Social Fabric Matrix.

This module implements the integration between social beliefs, cultural values,
and institutional structures within Hayden's SFM framework. It captures how
beliefs shape institutions and how institutions reinforce or transform beliefs,
forming a critical component of the social fabric analysis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
# Local enum definitions - no imports needed from sfm_enums for these


class BeliefType(Enum):
    """Types of social beliefs within institutional contexts."""
    
    NORMATIVE_BELIEFS = auto()        # Beliefs about what ought to be
    CAUSAL_BELIEFS = auto()          # Beliefs about cause-effect relationships
    PROCEDURAL_BELIEFS = auto()      # Beliefs about appropriate procedures
    IDENTITY_BELIEFS = auto()        # Beliefs about group/individual identity
    EFFICACY_BELIEFS = auto()        # Beliefs about ability to influence outcomes
    LEGITIMACY_BELIEFS = auto()      # Beliefs about legitimate authority
    FAIRNESS_BELIEFS = auto()        # Beliefs about distributive/procedural fairness
    PROGRESS_BELIEFS = auto()        # Beliefs about social/technological progress


class InstitutionalAlignment(Enum):
    """Alignment between beliefs and institutional structures."""
    
    STRONG_ALIGNMENT = auto()        # Beliefs strongly support institutions
    MODERATE_ALIGNMENT = auto()      # Beliefs generally support institutions
    WEAK_ALIGNMENT = auto()          # Limited belief-institution alignment
    MISALIGNMENT = auto()            # Beliefs conflict with institutions
    CONTRADICTORY = auto()           # Beliefs directly oppose institutions


class BeliefTransmissionMechanism(Enum):
    """Mechanisms for transmitting beliefs through institutions."""
    
    FORMAL_EDUCATION = auto()        # Educational institutions
    SOCIALIZATION_PROCESSES = auto() # Social learning and development
    MEDIA_COMMUNICATION = auto()     # Mass media and communication
    RITUALISTIC_PRACTICES = auto()   # Ceremonies and rituals
    LEGAL_FRAMEWORKS = auto()        # Laws and regulations
    ORGANIZATIONAL_CULTURE = auto()  # Workplace and organizational norms
    PEER_NETWORKS = auto()           # Social networks and communities
    AUTHORITY_ENDORSEMENT = auto()   # Leadership and authority figures


class CulturalPersistence(Enum):
    """Levels of cultural persistence and change."""
    
    HIGHLY_PERSISTENT = auto()       # Strong cultural persistence
    MODERATELY_PERSISTENT = auto()   # Moderate resistance to change
    ADAPTIVE = auto()                # Balanced persistence and adaptation
    CHANGE_ORIENTED = auto()         # Oriented toward cultural change
    HIGHLY_VOLATILE = auto()         # Rapid cultural change patterns


class BeliefConflictType(Enum):
    """Types of conflicts between different belief systems."""
    
    GENERATIONAL_CONFLICT = auto()   # Between age cohorts
    CLASS_CONFLICT = auto()          # Between socioeconomic groups
    CULTURAL_CONFLICT = auto()       # Between cultural groups
    IDEOLOGICAL_CONFLICT = auto()    # Between political ideologies
    RELIGIOUS_CONFLICT = auto()      # Between religious beliefs
    PROFESSIONAL_CONFLICT = auto()   # Between professional cultures
    REGIONAL_CONFLICT = auto()       # Between geographic regions


@dataclass
class SocialBeliefSystem(Node):
    """Core social belief system within institutional contexts."""
    
    belief_domain: Optional[str] = None  # Domain of beliefs (economic, political, etc.)
    belief_types: List[BeliefType] = field(default_factory=list)
    core_beliefs: List[str] = field(default_factory=list)
    
    # Belief characteristics
    belief_strength: Optional[float] = None  # Strength of belief commitment (0-1)
    belief_coherence: Optional[float] = None  # Internal consistency (0-1)
    belief_salience: Optional[float] = None  # Importance to believers (0-1)
    
    # Belief holders
    belief_community: List[uuid.UUID] = field(default_factory=list)  # Who holds these beliefs
    community_size: Optional[int] = None  # Size of belief community
    demographic_distribution: Dict[str, float] = field(default_factory=dict)  # Age, class, etc.
    
    # Belief content
    normative_components: List[str] = field(default_factory=list)  # What should be
    causal_theories: List[str] = field(default_factory=list)  # How things work
    value_commitments: List[str] = field(default_factory=list)  # What matters
    
    # Institutional connections
    supporting_institutions: List[uuid.UUID] = field(default_factory=list)
    conflicting_institutions: List[uuid.UUID] = field(default_factory=list)
    institutional_alignment: Optional[InstitutionalAlignment] = None
    
    # Transmission mechanisms
    transmission_mechanisms: List[BeliefTransmissionMechanism] = field(default_factory=list)
    socialization_agents: List[uuid.UUID] = field(default_factory=list)
    reinforcement_structures: List[str] = field(default_factory=list)
    
    # Temporal dynamics
    belief_persistence: Optional[CulturalPersistence] = None
    change_drivers: List[str] = field(default_factory=list)
    resistance_factors: List[str] = field(default_factory=list)
    
    # Belief conflicts
    conflicting_beliefs: List[uuid.UUID] = field(default_factory=list)
    belief_tensions: List[BeliefConflictType] = field(default_factory=list)
    
    # SFM Integration
    matrix_belief_impacts: List[uuid.UUID] = field(default_factory=list)
    delivery_belief_effects: Dict[uuid.UUID, str] = field(default_factory=dict)
    ceremonial_instrumental_classification: Optional[float] = None  # -1 (ceremonial) to +1 (instrumental)
    
    def assess_institutional_support(self) -> Dict[str, float]:
        """Assess level of institutional support for belief system."""
        support_assessment = {}
        
        # Institutional alignment assessment
        alignment_scores = {
            InstitutionalAlignment.STRONG_ALIGNMENT: 0.9,
            InstitutionalAlignment.MODERATE_ALIGNMENT: 0.7,
            InstitutionalAlignment.WEAK_ALIGNMENT: 0.4,
            InstitutionalAlignment.MISALIGNMENT: 0.2,
            InstitutionalAlignment.CONTRADICTORY: 0.1
        }
        
        if self.institutional_alignment:
            support_assessment['institutional_alignment_score'] = alignment_scores[self.institutional_alignment]
        
        # Supporting vs conflicting institutions
        total_institutions = len(self.supporting_institutions) + len(self.conflicting_institutions)
        if total_institutions > 0:
            support_ratio = len(self.supporting_institutions) / total_institutions
            support_assessment['institutional_support_ratio'] = support_ratio
        
        # Transmission mechanism effectiveness
        if self.transmission_mechanisms:
            mechanism_score = min(len(self.transmission_mechanisms) / 4.0, 1.0)
            support_assessment['transmission_effectiveness'] = mechanism_score
        
        # Socialization agent strength
        if self.socialization_agents:
            agent_score = min(len(self.socialization_agents) / 5.0, 1.0)
            support_assessment['socialization_strength'] = agent_score
        
        # Overall institutional support
        if support_assessment:
            overall_support = sum(support_assessment.values()) / len(support_assessment)
            support_assessment['overall_institutional_support'] = overall_support
        
        return support_assessment
    
    def analyze_belief_persistence(self) -> Dict[str, Any]:
        """Analyze factors affecting belief persistence and change."""
        persistence_analysis = {
            'persistence_factors': [],
            'change_factors': [],
            'persistence_score': 0.0,
            'change_vulnerability': 0.0
        }
        
        # Persistence factors
        if self.belief_strength and self.belief_strength > 0.7:
            persistence_analysis['persistence_factors'].append('Strong belief commitment')
        
        if self.belief_coherence and self.belief_coherence > 0.8:
            persistence_analysis['persistence_factors'].append('High internal coherence')
        
        if len(self.supporting_institutions) > 3:
            persistence_analysis['persistence_factors'].append('Strong institutional support')
        
        if self.community_size and self.community_size > 1000:
            persistence_analysis['persistence_factors'].append('Large belief community')
        
        # Change factors
        if self.change_drivers:
            persistence_analysis['change_factors'].extend(self.change_drivers)
        
        if len(self.conflicting_institutions) > len(self.supporting_institutions):
            persistence_analysis['change_factors'].append('Institutional opposition')
        
        if self.belief_tensions:
            persistence_analysis['change_factors'].append('Internal belief tensions')
        
        # Calculate persistence score
        persistence_score = 0.5  # Base score
        
        if self.belief_persistence == CulturalPersistence.HIGHLY_PERSISTENT:
            persistence_score = 0.9
        elif self.belief_persistence == CulturalPersistence.MODERATELY_PERSISTENT:
            persistence_score = 0.7
        elif self.belief_persistence == CulturalPersistence.ADAPTIVE:
            persistence_score = 0.5
        elif self.belief_persistence == CulturalPersistence.CHANGE_ORIENTED:
            persistence_score = 0.3
        elif self.belief_persistence == CulturalPersistence.HIGHLY_VOLATILE:
            persistence_score = 0.1
        
        persistence_analysis['persistence_score'] = persistence_score
        persistence_analysis['change_vulnerability'] = 1.0 - persistence_score
        
        return persistence_analysis


@dataclass
class BeliefInstitutionInteraction(Node):
    """Models interactions between belief systems and institutions."""
    
    belief_system: Optional[uuid.UUID] = None
    institution: Optional[uuid.UUID] = None
    interaction_type: Optional[str] = None  # "support", "conflict", "transformation"
    
    # Interaction characteristics
    interaction_strength: Optional[float] = None  # Strength of interaction (0-1)
    interaction_direction: Optional[str] = None  # "belief_to_institution", "institution_to_belief", "bidirectional"
    interaction_mechanisms: List[str] = field(default_factory=list)
    
    # Interaction outcomes
    belief_changes: List[str] = field(default_factory=list)
    institutional_changes: List[str] = field(default_factory=list)
    interaction_effects: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal patterns
    interaction_frequency: Optional[str] = None  # How often interaction occurs
    interaction_duration: Optional[timedelta] = None  # How long interactions last
    cyclical_patterns: List[str] = field(default_factory=list)
    
    # Mediating factors
    mediating_institutions: List[uuid.UUID] = field(default_factory=list)
    facilitating_conditions: List[str] = field(default_factory=list)
    constraining_conditions: List[str] = field(default_factory=list)
    
    # Impact assessment
    social_impact: Optional[float] = None  # Broader social impact (0-1)
    stability_impact: Optional[float] = None  # Impact on social stability (-1 to +1)
    change_catalyst_potential: Optional[float] = None  # Potential for broader change (0-1)
    
    # SFM Integration
    matrix_interaction_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_interaction_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    
    def assess_interaction_impact(self) -> Dict[str, float]:
        """Assess the impact of belief-institution interaction."""
        impact_assessment = {}
        
        # Direct impact assessment
        if self.interaction_strength is not None:
            impact_assessment['interaction_intensity'] = self.interaction_strength
        
        # Change impact assessment
        total_changes = len(self.belief_changes) + len(self.institutional_changes)
        if total_changes > 0:
            change_impact = min(total_changes / 5.0, 1.0)
            impact_assessment['change_impact'] = change_impact
        
        # Social impact assessment
        if self.social_impact is not None:
            impact_assessment['social_impact'] = self.social_impact
        
        # Stability impact assessment
        if self.stability_impact is not None:
            # Convert from -1 to +1 scale to 0 to 1 scale
            stability_score = (self.stability_impact + 1.0) / 2.0
            impact_assessment['stability_contribution'] = stability_score
        
        # Catalyst potential assessment
        if self.change_catalyst_potential is not None:
            impact_assessment['change_catalyst_potential'] = self.change_catalyst_potential
        
        # Overall interaction impact
        if impact_assessment:
            overall_impact = sum(impact_assessment.values()) / len(impact_assessment)
            impact_assessment['overall_interaction_impact'] = overall_impact
        
        return impact_assessment


@dataclass
class CulturalTransmissionSystem(Node):
    """Models cultural transmission systems within institutions."""
    
    transmission_domain: Optional[str] = None  # Domain of cultural transmission
    primary_mechanisms: List[BeliefTransmissionMechanism] = field(default_factory=list)
    transmission_agents: List[uuid.UUID] = field(default_factory=list)
    
    # Transmission characteristics
    transmission_effectiveness: Optional[float] = None  # How effective transmission is (0-1)
    transmission_reach: Optional[int] = None  # Number of people reached
    transmission_fidelity: Optional[float] = None  # Accuracy of transmission (0-1)
    
    # Content being transmitted
    cultural_content: List[str] = field(default_factory=list)
    belief_components: List[str] = field(default_factory=list)
    value_systems: List[str] = field(default_factory=list)
    behavioral_norms: List[str] = field(default_factory=list)
    
    # Transmission processes
    formal_transmission: List[str] = field(default_factory=list)  # Formal education, training
    informal_transmission: List[str] = field(default_factory=list)  # Social learning, modeling
    symbolic_transmission: List[str] = field(default_factory=list)  # Rituals, ceremonies
    
    # Target populations
    primary_targets: List[str] = field(default_factory=list)  # Main target groups
    secondary_targets: List[str] = field(default_factory=list)  # Secondary audiences
    transmission_barriers: List[str] = field(default_factory=list)  # Obstacles to transmission
    
    # Institutional embedding
    embedding_institutions: List[uuid.UUID] = field(default_factory=list)
    institutional_support: Optional[float] = None  # Level of institutional support (0-1)
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Transmission outcomes
    transmission_success_rate: Optional[float] = None  # Success rate (0-1)
    cultural_persistence: Optional[float] = None  # How well culture persists (0-1)
    adaptation_capacity: Optional[float] = None  # Ability to adapt content (0-1)
    
    # SFM Integration
    matrix_transmission_effects: List[uuid.UUID] = field(default_factory=list)
    delivery_transmission_impacts: Dict[uuid.UUID, str] = field(default_factory=dict)
    
    def evaluate_transmission_effectiveness(self) -> Dict[str, float]:
        """Evaluate effectiveness of cultural transmission system."""
        effectiveness_evaluation = {}
        
        # Mechanism diversity
        if self.primary_mechanisms:
            mechanism_diversity = min(len(self.primary_mechanisms) / 4.0, 1.0)
            effectiveness_evaluation['mechanism_diversity'] = mechanism_diversity
        
        # Agent capacity
        if self.transmission_agents:
            agent_capacity = min(len(self.transmission_agents) / 6.0, 1.0)
            effectiveness_evaluation['agent_capacity'] = agent_capacity
        
        # Institutional support
        if self.institutional_support is not None:
            effectiveness_evaluation['institutional_support'] = self.institutional_support
        
        # Content comprehensiveness
        total_content = (len(self.cultural_content) + len(self.belief_components) + 
                        len(self.value_systems) + len(self.behavioral_norms))
        if total_content > 0:
            content_score = min(total_content / 12.0, 1.0)
            effectiveness_evaluation['content_comprehensiveness'] = content_score
        
        # Process integration
        total_processes = (len(self.formal_transmission) + len(self.informal_transmission) + 
                          len(self.symbolic_transmission))
        if total_processes > 0:
            process_score = min(total_processes / 9.0, 1.0)
            effectiveness_evaluation['process_integration'] = process_score
        
        # Transmission fidelity
        if self.transmission_fidelity is not None:
            effectiveness_evaluation['transmission_fidelity'] = self.transmission_fidelity
        
        # Overall transmission effectiveness
        if effectiveness_evaluation:
            overall_effectiveness = sum(effectiveness_evaluation.values()) / len(effectiveness_evaluation)
            effectiveness_evaluation['overall_transmission_effectiveness'] = overall_effectiveness
            self.transmission_effectiveness = overall_effectiveness
        
        return effectiveness_evaluation


@dataclass
class BeliefSystemEvolution(Node):
    """Models evolution and change in belief systems over time."""
    
    baseline_belief_system: Optional[uuid.UUID] = None
    evolution_timeframe: Optional[timedelta] = None
    evolution_drivers: List[str] = field(default_factory=list)
    
    # Evolution patterns
    evolution_type: Optional[str] = None  # "gradual", "punctuated", "cyclical", "revolutionary"
    change_magnitude: Optional[float] = None  # Extent of change (0-1)
    change_direction: Optional[str] = None  # Direction of change
    
    # Change processes
    internal_change_factors: List[str] = field(default_factory=list)
    external_change_pressures: List[str] = field(default_factory=list)
    institutional_influences: List[uuid.UUID] = field(default_factory=list)
    
    # Resistance and adaptation
    resistance_mechanisms: List[str] = field(default_factory=list)
    adaptation_strategies: List[str] = field(default_factory=list)
    change_facilitators: List[str] = field(default_factory=list)
    
    # Evolution outcomes
    evolved_beliefs: List[str] = field(default_factory=list)
    retained_beliefs: List[str] = field(default_factory=list)
    abandoned_beliefs: List[str] = field(default_factory=list)
    
    # Generational effects
    generational_differences: Dict[str, List[str]] = field(default_factory=dict)
    intergenerational_transmission: Optional[float] = None  # Success rate (0-1)
    generational_conflict_areas: List[str] = field(default_factory=list)
    
    # Social consequences
    social_cohesion_impact: Optional[float] = None  # Impact on cohesion (-1 to +1)
    institutional_adaptation_needs: List[str] = field(default_factory=list)
    policy_implications: List[str] = field(default_factory=list)
    
    # SFM Integration
    matrix_evolution_impacts: List[uuid.UUID] = field(default_factory=list)
    delivery_evolution_effects: Dict[uuid.UUID, str] = field(default_factory=dict)
    
    def analyze_change_dynamics(self) -> Dict[str, Any]:
        """Analyze dynamics of belief system change."""
        change_analysis = {
            'change_drivers': self.evolution_drivers,
            'change_resistors': self.resistance_mechanisms,
            'change_velocity': 0.0,
            'change_sustainability': 0.0,
            'institutional_adaptation_required': []
        }
        
        # Calculate change velocity
        if self.change_magnitude is not None and self.evolution_timeframe is not None:
            timeframe_years = self.evolution_timeframe.days / 365.25
            if timeframe_years > 0:
                change_velocity = self.change_magnitude / timeframe_years
                change_analysis['change_velocity'] = min(change_velocity, 1.0)
        
        # Assess change sustainability
        change_facilitators_count = len(self.change_facilitators)
        resistance_count = len(self.resistance_mechanisms)
        
        if change_facilitators_count + resistance_count > 0:
            sustainability = change_facilitators_count / (change_facilitators_count + resistance_count)
            change_analysis['change_sustainability'] = sustainability
        
        # Institutional adaptation requirements
        change_analysis['institutional_adaptation_required'] = self.institutional_adaptation_needs
        
        # Generational impact analysis
        if self.generational_differences:
            change_analysis['generational_impact'] = {
                'affected_generations': len(self.generational_differences),
                'conflict_areas': self.generational_conflict_areas,
                'transmission_success': self.intergenerational_transmission
            }
        
        return change_analysis


@dataclass
class SocialBeliefMatrixAnalyzer(Node):
    """Analyzer for social beliefs within the Social Fabric Matrix framework."""
    
    belief_systems: Dict[uuid.UUID, SocialBeliefSystem] = field(default_factory=dict)
    belief_interactions: Dict[uuid.UUID, BeliefInstitutionInteraction] = field(default_factory=dict)
    transmission_systems: Dict[uuid.UUID, CulturalTransmissionSystem] = field(default_factory=dict)
    belief_evolutions: Dict[uuid.UUID, BeliefSystemEvolution] = field(default_factory=dict)
    
    # Analysis results
    dominant_belief_systems: List[uuid.UUID] = field(default_factory=list)
    belief_system_stability: Optional[float] = None  # Overall stability (0-1)
    cultural_coherence_score: Optional[float] = None  # Cultural coherence (0-1)
    
    def add_belief_system(self, belief_system: SocialBeliefSystem) -> None:
        """Add a belief system to the analysis."""
        self.belief_systems[belief_system.id] = belief_system
    
    def add_belief_interaction(self, interaction: BeliefInstitutionInteraction) -> None:
        """Add a belief-institution interaction to the analysis."""
        self.belief_interactions[interaction.id] = interaction
    
    def analyze_belief_landscape(self) -> Dict[str, Any]:
        """Analyze overall belief landscape within the social fabric."""
        landscape_analysis = {
            'belief_system_count': len(self.belief_systems),
            'interaction_count': len(self.belief_interactions),
            'dominant_beliefs': [],
            'belief_conflicts': [],
            'stability_assessment': {},
            'transmission_effectiveness': {}
        }
        
        # Identify dominant belief systems
        for belief_id, belief_system in self.belief_systems.items():
            if (belief_system.community_size and belief_system.community_size > 500 and
                belief_system.belief_strength and belief_system.belief_strength > 0.7):
                landscape_analysis['dominant_beliefs'].append({
                    'belief_id': belief_id,
                    'domain': belief_system.belief_domain,
                    'community_size': belief_system.community_size,
                    'strength': belief_system.belief_strength
                })
        
        # Identify belief conflicts
        for belief_system in self.belief_systems.values():
            if belief_system.belief_tensions:
                landscape_analysis['belief_conflicts'].extend([
                    {'belief_id': belief_system.id, 'conflict_type': tension.name}
                    for tension in belief_system.belief_tensions
                ])
        
        # Overall stability assessment
        stability_scores = []
        for belief_system in self.belief_systems.values():
            if belief_system.belief_persistence:
                persistence_values = {
                    CulturalPersistence.HIGHLY_PERSISTENT: 0.9,
                    CulturalPersistence.MODERATELY_PERSISTENT: 0.7,
                    CulturalPersistence.ADAPTIVE: 0.6,
                    CulturalPersistence.CHANGE_ORIENTED: 0.4,
                    CulturalPersistence.HIGHLY_VOLATILE: 0.2
                }
                stability_scores.append(persistence_values[belief_system.belief_persistence])
        
        if stability_scores:
            self.belief_system_stability = sum(stability_scores) / len(stability_scores)
            landscape_analysis['stability_assessment']['overall_stability'] = self.belief_system_stability
        
        return landscape_analysis
    
    def generate_belief_matrix_report(self) -> Dict[str, Any]:
        """Generate comprehensive report on beliefs within Social Fabric Matrix."""
        report = {
            'executive_summary': {},
            'belief_system_analysis': {},
            'institutional_integration': {},
            'transmission_analysis': {},
            'evolution_patterns': {},
            'recommendations': []
        }
        
        # Executive summary
        report['executive_summary'] = {
            'total_belief_systems': len(self.belief_systems),
            'total_interactions': len(self.belief_interactions),
            'system_stability': self.belief_system_stability,
            'cultural_coherence': self.cultural_coherence_score
        }
        
        # Belief system analysis
        strong_beliefs = []
        weak_beliefs = []
        
        for belief_system in self.belief_systems.values():
            if belief_system.belief_strength and belief_system.belief_strength > 0.7:
                strong_beliefs.append(belief_system.belief_domain)
            elif belief_system.belief_strength and belief_system.belief_strength < 0.4:
                weak_beliefs.append(belief_system.belief_domain)
        
        report['belief_system_analysis'] = {
            'strong_belief_domains': strong_beliefs,
            'weak_belief_domains': weak_beliefs,
            'belief_diversity': len(set(bs.belief_domain for bs in self.belief_systems.values() if bs.belief_domain))
        }
        
        # Generate recommendations
        if self.belief_system_stability and self.belief_system_stability < 0.5:
            report['recommendations'].append("Address belief system instability through institutional reform")
        
        if len(weak_beliefs) > len(strong_beliefs):
            report['recommendations'].append("Strengthen cultural transmission mechanisms")
        
        if not report['recommendations']:
            report['recommendations'].append("Continue monitoring belief-institution alignment")
        
        return report