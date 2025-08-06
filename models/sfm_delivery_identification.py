"""
Systematic Delivery Identification Framework for Social Fabric Matrix.

This module implements Hayden's systematic methodology for identifying, categorizing,
and analyzing delivery systems within the SFM framework. It provides comprehensive
tools for delivery system discovery, classification, and relationship mapping that
are essential for accurate matrix construction.

Key Components:
- DeliverySystemIdentification: Systematic identification of delivery systems
- DeliveryClassification: Taxonomic classification of delivery mechanisms
- DeliverySystemMapping: Mapping relationships between delivery systems
- DeliveryCapacityAssessment: Assessment of delivery system capabilities
- DeliverySystemValidation: Validation of identified delivery systems
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit, Scenario
from models.sfm_enums import (
    DeliveryMechanism,
    ResourceType,
    FlowType,
    SystemLevel,
    InstitutionalScope,
    ValidationMethod,
    EvidenceQuality,
    ValueCategory,
)


class DeliverySystemType(Enum):
    """Types of delivery systems in SFM analysis."""
    
    DIRECT_SERVICE = auto()           # Direct service delivery
    INFRASTRUCTURE = auto()           # Infrastructure-based delivery
    REGULATORY = auto()              # Regulatory delivery mechanisms
    MARKET_BASED = auto()            # Market-mediated delivery
    NETWORK = auto()                 # Network-based delivery
    HYBRID = auto()                  # Combined delivery approaches
    EMERGENT = auto()                # Self-organizing delivery


class DeliveryScope(Enum):
    """Scope of delivery system operation."""
    
    INDIVIDUAL = auto()              # Individual-level delivery
    HOUSEHOLD = auto()               # Household-level delivery
    COMMUNITY = auto()               # Community-level delivery
    ORGANIZATIONAL = auto()          # Organizational delivery
    INSTITUTIONAL = auto()           # Institutional delivery
    SYSTEM_WIDE = auto()             # System-wide delivery


class DeliveryMode(Enum):
    """Modes of delivery system operation."""
    
    CONTINUOUS = auto()              # Continuous delivery
    PERIODIC = auto()                # Scheduled/periodic delivery
    ON_DEMAND = auto()               # On-demand delivery
    EMERGENCY = auto()               # Emergency/crisis delivery
    CONDITIONAL = auto()             # Conditional delivery
    UNIVERSAL = auto()               # Universal delivery


class DeliveryEffectiveness(Enum):
    """Effectiveness levels of delivery systems."""
    
    HIGHLY_EFFECTIVE = auto()        # Consistently exceeds targets
    EFFECTIVE = auto()               # Meets targets reliably
    MODERATELY_EFFECTIVE = auto()    # Meets some targets
    INEFFECTIVE = auto()             # Fails to meet targets
    UNKNOWN = auto()                 # Effectiveness not assessed


class IdentificationMethod(Enum):
    """Methods for identifying delivery systems."""
    
    INSTITUTIONAL_ANALYSIS = auto()   # Analysis of institutional structures
    STAKEHOLDER_MAPPING = auto()      # Stakeholder and actor mapping
    PROCESS_TRACING = auto()          # Process and flow tracing
    NETWORK_ANALYSIS = auto()         # Network structure analysis
    OUTCOME_ANALYSIS = auto()         # Working backward from outcomes
    MIXED_METHODS = auto()            # Combined identification approaches


@dataclass
class DeliverySystemIdentification(Node):
    """Systematic identification of delivery systems."""
    
    identification_scope: Optional[str] = None
    identification_purpose: Optional[str] = None
    
    # Identification methodology
    identification_method: IdentificationMethod = IdentificationMethod.MIXED_METHODS
    identification_protocol: List[str] = field(default_factory=list)
    identification_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # Target domain
    target_value_categories: List[ValueCategory] = field(default_factory=list)
    target_institutions: List[uuid.UUID] = field(default_factory=list)
    target_geographic_area: Optional[SpatialUnit] = None
    target_time_period: Optional[TimeSlice] = None
    
    # Identification process
    identification_steps: List[str] = field(default_factory=list)
    data_collection_methods: List[str] = field(default_factory=list)
    stakeholder_consultation: Dict[uuid.UUID, str] = field(default_factory=dict)
    
    # Identified delivery systems
    identified_systems: List[uuid.UUID] = field(default_factory=list)  # DeliverySystem IDs
    system_classifications: Dict[uuid.UUID, str] = field(default_factory=dict)
    system_relationships: Dict[Tuple[uuid.UUID, uuid.UUID], str] = field(default_factory=dict)
    
    # Supporting evidence
    identification_evidence: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    evidence_quality: Dict[uuid.UUID, EvidenceQuality] = field(default_factory=dict)
    validation_status: Dict[uuid.UUID, str] = field(default_factory=dict)
    
    # Quality assurance
    identification_completeness: Optional[float] = None  # Coverage completeness (0-1)
    identification_accuracy: Optional[float] = None     # Identification accuracy (0-1)
    inter_identifier_reliability: Optional[float] = None # Agreement between identifiers
    
    # Documentation
    identification_documentation: Dict[str, str] = field(default_factory=dict)
    methodology_notes: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    
    def conduct_systematic_identification(self) -> Dict[str, Any]:
        """Conduct systematic identification of delivery systems."""
        identification_results = {
            'identification_overview': {},
            'systems_discovered': [],
            'classification_results': {},
            'relationship_mapping': {},
            'quality_assessment': {},
            'coverage_analysis': {},
            'recommendations': []
        }
        
        # Identification overview
        identification_results['identification_overview'] = {
            'method_applied': self.identification_method.name,
            'scope': self.identification_scope,
            'target_institutions': len(self.target_institutions),
            'systems_identified': len(self.identified_systems),
            'identification_date': datetime.now()
        }
        
        # Systems discovered analysis
        for system_id in self.identified_systems:
            system_info = {
                'system_id': str(system_id),
                'classification': self.system_classifications.get(system_id, 'unclassified'),
                'evidence_quality': self.evidence_quality.get(system_id, EvidenceQuality.MEDIUM).name,
                'validation_status': self.validation_status.get(system_id, 'pending')
            }
            identification_results['systems_discovered'].append(system_info)
        
        # Classification results
        classification_counts = {}
        for classification in self.system_classifications.values():
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        identification_results['classification_results'] = classification_counts
        
        # Relationship mapping
        relationship_counts = {}
        for (source, target), relationship_type in self.system_relationships.items():
            relationship_counts[relationship_type] = relationship_counts.get(relationship_type, 0) + 1
        identification_results['relationship_mapping'] = {
            'total_relationships': len(self.system_relationships),
            'relationship_types': relationship_counts
        }
        
        # Quality assessment
        identification_results['quality_assessment'] = {
            'completeness': self.identification_completeness or 0.0,
            'accuracy': self.identification_accuracy or 0.0,
            'reliability': self.inter_identifier_reliability or 0.0
        }
        
        # Generate recommendations
        if (self.identification_completeness or 0.0) < 0.8:
            identification_results['recommendations'].append('Expand identification scope to improve completeness')
        if len(self.identified_systems) < 5:
            identification_results['recommendations'].append('Consider additional identification methods')
        
        return identification_results
    
    def validate_identification_quality(self) -> Dict[str, Any]:
        """Validate quality of delivery system identification."""
        validation_results = {
            'completeness_assessment': 0.0,
            'accuracy_assessment': 0.0,
            'consistency_assessment': 0.0,
            'evidence_assessment': 0.0,
            'overall_quality': 0.0,
            'quality_issues': [],
            'improvement_recommendations': []
        }
        
        # Completeness assessment
        if self.identification_completeness is not None:
            validation_results['completeness_assessment'] = self.identification_completeness
        else:
            # Estimate based on coverage indicators
            expected_systems = len(self.target_institutions) * 2  # Rough estimate
            identified_ratio = len(self.identified_systems) / max(expected_systems, 1)
            validation_results['completeness_assessment'] = min(identified_ratio, 1.0)
        
        # Accuracy assessment
        validation_results['accuracy_assessment'] = self.identification_accuracy or 0.7
        
        # Consistency assessment
        validation_results['consistency_assessment'] = self.inter_identifier_reliability or 0.6
        
        # Evidence assessment
        if self.evidence_quality:
            quality_scores = {
                EvidenceQuality.HIGH: 1.0,
                EvidenceQuality.MEDIUM: 0.7,
                EvidenceQuality.LOW: 0.4
            }
            evidence_scores = [quality_scores[quality] for quality in self.evidence_quality.values()]
            validation_results['evidence_assessment'] = sum(evidence_scores) / len(evidence_scores)
        
        # Overall quality
        quality_factors = [
            validation_results['completeness_assessment'] * 0.3,
            validation_results['accuracy_assessment'] * 0.25,
            validation_results['consistency_assessment'] * 0.25,
            validation_results['evidence_assessment'] * 0.2
        ]
        validation_results['overall_quality'] = sum(quality_factors)
        
        # Identify quality issues
        if validation_results['completeness_assessment'] < 0.7:
            validation_results['quality_issues'].append('Incomplete identification coverage')
        if validation_results['accuracy_assessment'] < 0.6:
            validation_results['quality_issues'].append('Low identification accuracy')
        if validation_results['evidence_assessment'] < 0.5:
            validation_results['quality_issues'].append('Weak supporting evidence')
        
        # Generate improvement recommendations
        if validation_results['overall_quality'] < 0.7:
            validation_results['improvement_recommendations'].append('Strengthen identification methodology')
        if not self.stakeholder_consultation:
            validation_results['improvement_recommendations'].append('Include stakeholder validation')
        
        return validation_results
    
    def generate_identification_map(self) -> Dict[str, Any]:
        """Generate comprehensive map of identified delivery systems."""
        identification_map = {
            'system_inventory': {},
            'classification_taxonomy': {},
            'relationship_network': {},
            'coverage_map': {},
            'quality_indicators': {}
        }
        
        # System inventory
        for system_id in self.identified_systems:
            identification_map['system_inventory'][str(system_id)] = {
                'classification': self.system_classifications.get(system_id, 'unclassified'),
                'evidence_quality': self.evidence_quality.get(system_id, EvidenceQuality.MEDIUM).name,
                'validation_status': self.validation_status.get(system_id, 'pending'),
                'supporting_evidence': len(self.identification_evidence.get(system_id, []))
            }
        
        # Classification taxonomy
        classification_taxonomy = {}
        for system_id, classification in self.system_classifications.items():
            if classification not in classification_taxonomy:
                classification_taxonomy[classification] = []
            classification_taxonomy[classification].append(str(system_id))
        identification_map['classification_taxonomy'] = classification_taxonomy
        
        # Relationship network
        identification_map['relationship_network'] = {
            f"{source}_{target}": relationship_type
            for (source, target), relationship_type in self.system_relationships.items()
        }
        
        return identification_map


@dataclass
class DeliveryClassification(Node):
    """Taxonomic classification of delivery mechanisms and systems."""
    
    classification_framework: Optional[str] = None
    classification_purpose: Optional[str] = None
    
    # Classification structure
    classification_taxonomy: Dict[str, List[str]] = field(default_factory=dict)  # Category -> subcategories
    classification_criteria: Dict[str, Any] = field(default_factory=dict)
    classification_rules: List[str] = field(default_factory=list)
    
    # Delivery system types
    direct_delivery_systems: List[uuid.UUID] = field(default_factory=list)
    infrastructure_systems: List[uuid.UUID] = field(default_factory=list)
    regulatory_systems: List[uuid.UUID] = field(default_factory=list)
    market_systems: List[uuid.UUID] = field(default_factory=list)
    network_systems: List[uuid.UUID] = field(default_factory=list)
    hybrid_systems: List[uuid.UUID] = field(default_factory=list)
    
    # Classification properties
    system_classifications: Dict[uuid.UUID, DeliverySystemType] = field(default_factory=dict)
    system_scopes: Dict[uuid.UUID, DeliveryScope] = field(default_factory=dict)
    system_modes: Dict[uuid.UUID, DeliveryMode] = field(default_factory=dict)
    
    # Classification quality
    classification_consistency: Optional[float] = None  # Internal consistency (0-1)
    classification_completeness: Optional[float] = None # Coverage completeness (0-1)
    inter_classifier_agreement: Optional[float] = None  # Agreement between classifiers
    
    # Validation and refinement
    classification_validation: Dict[uuid.UUID, bool] = field(default_factory=dict)
    classification_confidence: Dict[uuid.UUID, float] = field(default_factory=dict)
    reclassification_history: Dict[uuid.UUID, List[Dict[str, Any]]] = field(default_factory=dict)
    
    def apply_classification_framework(self, delivery_systems: List[uuid.UUID]) -> Dict[str, Any]:
        """Apply classification framework to delivery systems."""
        classification_results = {
            'classification_summary': {},
            'system_classifications': {},
            'classification_distribution': {},
            'quality_assessment': {},
            'recommendations': []
        }
        
        # Apply classifications
        for system_id in delivery_systems:
            system_classification = self._classify_delivery_system(system_id)
            self.system_classifications[system_id] = system_classification['type']
            self.system_scopes[system_id] = system_classification['scope']
            self.system_modes[system_id] = system_classification['mode']
            
            classification_results['system_classifications'][str(system_id)] = {
                'type': system_classification['type'].name,
                'scope': system_classification['scope'].name,
                'mode': system_classification['mode'].name,
                'confidence': system_classification['confidence']
            }
        
        # Classification distribution
        type_distribution = {}
        for system_type in self.system_classifications.values():
            type_distribution[system_type.name] = type_distribution.get(system_type.name, 0) + 1
        classification_results['classification_distribution'] = type_distribution
        
        # Classification summary
        classification_results['classification_summary'] = {
            'total_systems_classified': len(delivery_systems),
            'classification_framework': self.classification_framework,
            'primary_classification_types': len(set(self.system_classifications.values())),
            'classification_date': datetime.now()
        }
        
        return classification_results
    
    def _classify_delivery_system(self, system_id: uuid.UUID) -> Dict[str, Any]:
        """Classify individual delivery system."""
        # Simplified classification logic - in practice would use detailed criteria
        classification = {
            'type': DeliverySystemType.DIRECT_SERVICE,  # Default
            'scope': DeliveryScope.ORGANIZATIONAL,      # Default
            'mode': DeliveryMode.CONTINUOUS,           # Default
            'confidence': 0.7,                         # Default confidence
            'rationale': 'Default classification pending detailed analysis'
        }
        
        # Store confidence level
        self.classification_confidence[system_id] = classification['confidence']
        
        return classification
    
    def validate_classification_consistency(self) -> Dict[str, Any]:
        """Validate consistency of classification framework."""
        consistency_assessment = {
            'internal_consistency': 0.0,
            'rule_consistency': 0.0,
            'taxonomic_consistency': 0.0,
            'overall_consistency': 0.0,
            'consistency_issues': [],
            'improvement_recommendations': []
        }
        
        # Internal consistency check
        if self.system_classifications:
            # Check for classification conflicts or inconsistencies
            consistency_assessment['internal_consistency'] = 0.8  # Placeholder
        
        # Rule consistency
        if self.classification_rules:
            # Check if classification rules are consistently applied
            consistency_assessment['rule_consistency'] = 0.7  # Placeholder
        
        # Taxonomic consistency
        if self.classification_taxonomy:
            # Check taxonomy structure consistency
            consistency_assessment['taxonomic_consistency'] = 0.75  # Placeholder
        
        # Overall consistency
        consistency_factors = [
            consistency_assessment['internal_consistency'],
            consistency_assessment['rule_consistency'],
            consistency_assessment['taxonomic_consistency']
        ]
        valid_factors = [f for f in consistency_factors if f > 0]
        if valid_factors:
            consistency_assessment['overall_consistency'] = sum(valid_factors) / len(valid_factors)
            self.classification_consistency = consistency_assessment['overall_consistency']
        
        # Generate recommendations
        if consistency_assessment['overall_consistency'] < 0.7:
            consistency_assessment['improvement_recommendations'].append('Review and strengthen classification consistency')
        
        return consistency_assessment
    
    def generate_classification_guide(self) -> Dict[str, Any]:
        """Generate comprehensive classification guide."""
        classification_guide = {
            'framework_overview': {
                'name': self.classification_framework,
                'purpose': self.classification_purpose,
                'taxonomy_structure': self.classification_taxonomy
            },
            'classification_criteria': self.classification_criteria,
            'classification_rules': self.classification_rules,
            'system_type_definitions': self._get_system_type_definitions(),
            'classification_examples': self._generate_classification_examples(),
            'quality_standards': {
                'minimum_consistency': 0.7,
                'minimum_completeness': 0.8,
                'minimum_agreement': 0.6
            },
            'usage_guidelines': self._generate_usage_guidelines()
        }
        
        return classification_guide
    
    def _get_system_type_definitions(self) -> Dict[str, str]:
        """Get definitions for delivery system types."""
        return {
            'DIRECT_SERVICE': 'Systems that provide direct services to beneficiaries',
            'INFRASTRUCTURE': 'Physical or institutional infrastructure enabling delivery',
            'REGULATORY': 'Regulatory mechanisms that govern delivery processes',
            'MARKET_BASED': 'Market mechanisms that facilitate delivery',
            'NETWORK': 'Network-based delivery through interconnected actors',
            'HYBRID': 'Systems combining multiple delivery approaches'
        }
    
    def _generate_classification_examples(self) -> Dict[str, List[str]]:
        """Generate examples for each classification type."""
        return {
            'DIRECT_SERVICE': ['Healthcare clinics', 'Educational institutions', 'Social services'],
            'INFRASTRUCTURE': ['Transportation networks', 'Communication systems', 'Utilities'],
            'REGULATORY': ['Licensing systems', 'Standards enforcement', 'Compliance monitoring'],
            'MARKET_BASED': ['Subsidies', 'Vouchers', 'Public-private partnerships'],
            'NETWORK': ['Community networks', 'Professional associations', 'Collaborative platforms'],
            'HYBRID': ['Mixed service delivery', 'Multi-modal systems', 'Integrated approaches']
        }
    
    def _generate_usage_guidelines(self) -> List[str]:
        """Generate guidelines for using the classification framework."""
        return [
            'Start with system type classification before scope and mode',
            'Use multiple classifiers to ensure reliability',
            'Document classification rationale for all systems',
            'Regular review and validation of classifications',
            'Consider context-specific factors in classification',
            'Maintain consistency across similar systems'
        ]


@dataclass
class DeliverySystemMapping(Node):
    """Mapping relationships between delivery systems."""
    
    mapping_scope: Optional[str] = None
    mapping_purpose: Optional[str] = None
    
    # Mapped systems
    mapped_systems: List[uuid.UUID] = field(default_factory=list)
    system_relationships: Dict[Tuple[uuid.UUID, uuid.UUID], str] = field(default_factory=dict)
    relationship_strengths: Dict[Tuple[uuid.UUID, uuid.UUID], float] = field(default_factory=dict)
    
    # Network properties
    network_density: Optional[float] = None        # Connection density
    network_centralization: Optional[float] = None # Centralization measure
    network_clustering: Optional[float] = None     # Clustering coefficient
    
    # System roles
    hub_systems: List[uuid.UUID] = field(default_factory=list)        # Central hub systems
    bridge_systems: List[uuid.UUID] = field(default_factory=list)     # Bridge/connector systems
    peripheral_systems: List[uuid.UUID] = field(default_factory=list) # Peripheral systems
    
    # Relationship types
    dependency_relationships: Dict[Tuple[uuid.UUID, uuid.UUID], str] = field(default_factory=dict)
    complementary_relationships: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)
    competitive_relationships: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)
    
    # Mapping quality
    mapping_completeness: Optional[float] = None   # Coverage of relationships
    mapping_accuracy: Optional[float] = None       # Accuracy of identified relationships
    validation_status: Dict[Tuple[uuid.UUID, uuid.UUID], str] = field(default_factory=dict)
    
    def analyze_delivery_network(self) -> Dict[str, Any]:
        """Analyze the network structure of delivery systems."""
        network_analysis = {
            'network_overview': {},
            'structural_properties': {},
            'system_roles': {},
            'relationship_analysis': {},
            'network_efficiency': {},
            'recommendations': []
        }
        
        # Network overview
        network_analysis['network_overview'] = {
            'total_systems': len(self.mapped_systems),
            'total_relationships': len(self.system_relationships),
            'network_density': self.network_density or 0.0,
            'analysis_date': datetime.now()
        }
        
        # Structural properties
        network_analysis['structural_properties'] = {
            'density': self.network_density or 0.0,
            'centralization': self.network_centralization or 0.0,
            'clustering': self.network_clustering or 0.0,
            'connectivity': len(self.system_relationships) / max(len(self.mapped_systems), 1)
        }
        
        # System roles
        network_analysis['system_roles'] = {
            'hub_systems': [str(sys_id) for sys_id in self.hub_systems],
            'bridge_systems': [str(sys_id) for sys_id in self.bridge_systems],
            'peripheral_systems': [str(sys_id) for sys_id in self.peripheral_systems]
        }
        
        # Relationship analysis
        relationship_types = {}
        for relationship_type in self.system_relationships.values():
            relationship_types[relationship_type] = relationship_types.get(relationship_type, 0) + 1
        
        network_analysis['relationship_analysis'] = {
            'relationship_distribution': relationship_types,
            'dependency_relationships': len(self.dependency_relationships),
            'complementary_relationships': len(self.complementary_relationships),
            'competitive_relationships': len(self.competitive_relationships)
        }
        
        # Generate recommendations
        if (self.network_density or 0.0) < 0.3:
            network_analysis['recommendations'].append('Consider strengthening system interconnections')
        if len(self.hub_systems) < 2:
            network_analysis['recommendations'].append('Identify and develop hub systems for network resilience')
        
        return network_analysis
    
    def identify_critical_delivery_paths(self) -> Dict[str, Any]:
        """Identify critical paths in delivery system network."""
        critical_paths = {
            'primary_delivery_chains': [],
            'backup_delivery_paths': [],
            'vulnerability_points': [],
            'redundancy_analysis': {},
            'resilience_assessment': {}
        }
        
        # Simplified path analysis - in practice would use network analysis algorithms
        
        # Identify hub-dependent paths
        for hub_system in self.hub_systems:
            hub_connections = [
                (source, target) for (source, target) in self.system_relationships.keys()
                if source == hub_system or target == hub_system
            ]
            
            if len(hub_connections) > 5:  # High-connectivity threshold
                critical_paths['vulnerability_points'].append({
                    'system_id': str(hub_system),
                    'vulnerability_type': 'hub_dependency',
                    'connection_count': len(hub_connections),
                    'risk_level': 'high' if len(hub_connections) > 10 else 'moderate'
                })
        
        # Assess redundancy
        redundancy_score = len(self.bridge_systems) / max(len(self.hub_systems), 1)
        critical_paths['redundancy_analysis'] = {
            'redundancy_score': redundancy_score,
            'redundancy_level': 'high' if redundancy_score > 2 else 'low',
            'single_points_of_failure': len([sys for sys in self.hub_systems if len([
                rel for rel in self.system_relationships.keys() if sys in rel
            ]) > 8])
        }
        
        return critical_paths
    
    def validate_relationship_mapping(self) -> Dict[str, Any]:
        """Validate accuracy and completeness of relationship mapping."""
        validation_results = {
            'completeness_assessment': 0.0,
            'accuracy_assessment': 0.0,
            'consistency_assessment': 0.0,
            'validation_coverage': 0.0,
            'overall_quality': 0.0,
            'validation_issues': [],
            'recommendations': []
        }
        
        # Completeness assessment
        total_possible_relationships = len(self.mapped_systems) * (len(self.mapped_systems) - 1) // 2
        identified_relationships = len(self.system_relationships)
        if total_possible_relationships > 0:
            validation_results['completeness_assessment'] = min(
                identified_relationships / (total_possible_relationships * 0.3), 1.0  # Assume 30% connectivity is complete
            )
        
        # Accuracy assessment
        validation_results['accuracy_assessment'] = self.mapping_accuracy or 0.7
        
        # Validation coverage
        validated_relationships = sum(1 for status in self.validation_status.values() if status == 'validated')
        total_relationships = len(self.system_relationships)
        if total_relationships > 0:
            validation_results['validation_coverage'] = validated_relationships / total_relationships
        
        # Overall quality
        quality_factors = [
            validation_results['completeness_assessment'] * 0.3,
            validation_results['accuracy_assessment'] * 0.3,
            validation_results['validation_coverage'] * 0.4
        ]
        validation_results['overall_quality'] = sum(quality_factors)
        
        # Generate recommendations
        if validation_results['completeness_assessment'] < 0.6:
            validation_results['recommendations'].append('Expand relationship identification')
        if validation_results['validation_coverage'] < 0.5:
            validation_results['recommendations'].append('Increase validation of mapped relationships')
        
        return validation_results


@dataclass
class DeliveryCapacityAssessment(Node):
    """Assessment of delivery system capabilities and performance."""
    
    assessment_scope: Optional[str] = None
    assessment_purpose: Optional[str] = None
    
    # Assessed systems
    assessed_systems: List[uuid.UUID] = field(default_factory=list)
    assessment_criteria: Dict[str, Any] = field(default_factory=dict)
    assessment_timeframe: Optional[TimeSlice] = None
    
    # Capacity dimensions
    delivery_capacity: Dict[uuid.UUID, float] = field(default_factory=dict)    # Maximum delivery capacity
    current_utilization: Dict[uuid.UUID, float] = field(default_factory=dict) # Current utilization rate
    capacity_constraints: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    
    # Performance metrics
    delivery_effectiveness: Dict[uuid.UUID, DeliveryEffectiveness] = field(default_factory=dict)
    delivery_efficiency: Dict[uuid.UUID, float] = field(default_factory=dict)
    delivery_quality: Dict[uuid.UUID, float] = field(default_factory=dict)
    
    # Resource requirements
    resource_requirements: Dict[uuid.UUID, Dict[ResourceType, float]] = field(default_factory=dict)
    resource_availability: Dict[uuid.UUID, Dict[ResourceType, float]] = field(default_factory=dict)
    resource_gaps: Dict[uuid.UUID, Dict[ResourceType, float]] = field(default_factory=dict)
    
    # Stakeholder assessment
    stakeholder_satisfaction: Dict[uuid.UUID, Dict[uuid.UUID, float]] = field(default_factory=dict)
    beneficiary_feedback: Dict[uuid.UUID, Dict[str, Any]] = field(default_factory=dict)
    provider_assessment: Dict[uuid.UUID, Dict[str, float]] = field(default_factory=dict)
    
    # Capacity development
    capacity_development_needs: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    improvement_recommendations: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    capacity_building_priorities: List[Tuple[uuid.UUID, str, float]] = field(default_factory=list)
    
    def conduct_capacity_assessment(self) -> Dict[str, Any]:
        """Conduct comprehensive capacity assessment."""
        assessment_results = {
            'assessment_overview': {},
            'capacity_analysis': {},
            'performance_analysis': {},
            'resource_analysis': {},
            'stakeholder_analysis': {},
            'capacity_gaps': {},
            'recommendations': []
        }
        
        # Assessment overview
        assessment_results['assessment_overview'] = {
            'systems_assessed': len(self.assessed_systems),
            'assessment_scope': self.assessment_scope,
            'assessment_date': datetime.now(),
            'assessment_coverage': self._calculate_assessment_coverage()
        }
        
        # Capacity analysis
        if self.delivery_capacity:
            capacity_values = list(self.delivery_capacity.values())
            assessment_results['capacity_analysis'] = {
                'average_capacity': sum(capacity_values) / len(capacity_values),
                'capacity_range': (min(capacity_values), max(capacity_values)),
                'high_capacity_systems': len([c for c in capacity_values if c > 0.8]),
                'capacity_distribution': self._categorize_capacity_levels()
            }
        
        # Performance analysis
        if self.delivery_effectiveness:
            effectiveness_counts = {}
            for effectiveness in self.delivery_effectiveness.values():
                effectiveness_counts[effectiveness.name] = effectiveness_counts.get(effectiveness.name, 0) + 1
            
            assessment_results['performance_analysis'] = {
                'effectiveness_distribution': effectiveness_counts,
                'highly_effective_systems': effectiveness_counts.get('HIGHLY_EFFECTIVE', 0),
                'ineffective_systems': effectiveness_counts.get('INEFFECTIVE', 0)
            }
        
        # Resource analysis
        assessment_results['resource_analysis'] = self._analyze_resource_gaps()
        
        # Generate recommendations
        if assessment_results['performance_analysis'].get('ineffective_systems', 0) > 0:
            assessment_results['recommendations'].append('Address ineffective delivery systems')
        
        return assessment_results
    
    def _calculate_assessment_coverage(self) -> float:
        """Calculate coverage of capacity assessment."""
        if not self.assessed_systems:
            return 0.0
        
        # Simplified coverage calculation
        coverage_indicators = [
            bool(self.delivery_capacity),
            bool(self.delivery_effectiveness),
            bool(self.resource_requirements),
            bool(self.stakeholder_satisfaction)
        ]
        
        return sum(coverage_indicators) / len(coverage_indicators)
    
    def _categorize_capacity_levels(self) -> Dict[str, int]:
        """Categorize systems by capacity levels."""
        capacity_categories = {
            'high_capacity': 0,    # >0.8
            'medium_capacity': 0,  # 0.5-0.8
            'low_capacity': 0      # <0.5
        }
        
        for capacity in self.delivery_capacity.values():
            if capacity > 0.8:
                capacity_categories['high_capacity'] += 1
            elif capacity >= 0.5:
                capacity_categories['medium_capacity'] += 1
            else:
                capacity_categories['low_capacity'] += 1
        
        return capacity_categories
    
    def _analyze_resource_gaps(self) -> Dict[str, Any]:
        """Analyze resource gaps across delivery systems."""
        resource_analysis = {
            'systems_with_gaps': 0,
            'critical_resource_gaps': {},
            'total_resource_gap': 0.0,
            'resource_constraint_frequency': {}
        }
        
        # Count systems with resource gaps
        for system_id in self.assessed_systems:
            system_gaps = self.resource_gaps.get(system_id, {})
            if system_gaps:
                resource_analysis['systems_with_gaps'] += 1
                
                # Sum resource gaps
                for resource_type, gap_amount in system_gaps.items():
                    if gap_amount > 0:
                        if resource_type.name not in resource_analysis['critical_resource_gaps']:
                            resource_analysis['critical_resource_gaps'][resource_type.name] = 0
                        resource_analysis['critical_resource_gaps'][resource_type.name] += gap_amount
        
        return resource_analysis
    
    def prioritize_capacity_building(self) -> List[Dict[str, Any]]:
        """Prioritize capacity building interventions."""
        priorities = []
        
        for system_id in self.assessed_systems:
            # Calculate priority score based on multiple factors
            priority_score = 0.0
            
            # Factor 1: Current effectiveness
            effectiveness = self.delivery_effectiveness.get(system_id, DeliveryEffectiveness.UNKNOWN)
            if effectiveness == DeliveryEffectiveness.INEFFECTIVE:
                priority_score += 0.4
            elif effectiveness == DeliveryEffectiveness.MODERATELY_EFFECTIVE:
                priority_score += 0.2
            
            # Factor 2: Capacity utilization
            capacity = self.delivery_capacity.get(system_id, 0.5)
            utilization = self.current_utilization.get(system_id, 0.5)
            if capacity > 0 and utilization / capacity > 0.9:  # Over-utilized
                priority_score += 0.3
            
            # Factor 3: Resource gaps
            resource_gaps = self.resource_gaps.get(system_id, {})
            if resource_gaps:
                gap_severity = sum(resource_gaps.values()) / len(resource_gaps)
                priority_score += min(gap_severity / 10.0, 0.3)  # Normalize and cap
            
            if priority_score > 0.1:  # Only include systems with some priority
                priorities.append({
                    'system_id': str(system_id),
                    'priority_score': priority_score,
                    'priority_level': 'high' if priority_score > 0.7 else 'medium' if priority_score > 0.4 else 'low',
                    'key_issues': self.capacity_development_needs.get(system_id, []),
                    'recommendations': self.improvement_recommendations.get(system_id, [])
                })
        
        # Sort by priority score
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priorities


@dataclass
class DeliverySystemValidation(Node):
    """Validation framework for identified delivery systems."""
    
    validation_scope: Optional[str] = None
    validation_purpose: Optional[str] = None
    
    # Validation targets
    systems_to_validate: List[uuid.UUID] = field(default_factory=list)
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    validation_standards: Dict[str, float] = field(default_factory=dict)
    
    # Validation methods
    validation_methods: List[ValidationMethod] = field(default_factory=list)
    validation_protocol: List[str] = field(default_factory=list)
    validator_qualifications: List[str] = field(default_factory=list)
    
    # Validation results
    validation_outcomes: Dict[uuid.UUID, str] = field(default_factory=dict)  # System -> outcome
    validation_scores: Dict[uuid.UUID, float] = field(default_factory=dict)  # System -> score
    validation_evidence: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    
    # Quality metrics
    validation_reliability: Optional[float] = None
    validation_validity: Optional[float] = None
    inter_validator_agreement: Optional[float] = None
    
    # Improvement tracking
    validation_issues: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    improvement_actions: Dict[uuid.UUID, List[str]] = field(default_factory=dict)
    revalidation_schedule: Dict[uuid.UUID, datetime] = field(default_factory=dict)
    
    def conduct_systematic_validation(self) -> Dict[str, Any]:
        """Conduct systematic validation of delivery systems."""
        validation_results = {
            'validation_summary': {},
            'system_validation_results': {},
            'quality_assessment': {},
            'recommendations': []
        }
        
        # Validation summary
        validated_systems = sum(1 for outcome in self.validation_outcomes.values() if outcome == 'validated')
        rejected_systems = sum(1 for outcome in self.validation_outcomes.values() if outcome == 'rejected')
        pending_systems = len(self.systems_to_validate) - validated_systems - rejected_systems
        
        validation_results['validation_summary'] = {
            'total_systems': len(self.systems_to_validate),
            'validated_systems': validated_systems,
            'rejected_systems': rejected_systems,
            'pending_systems': pending_systems,
            'validation_rate': validated_systems / len(self.systems_to_validate) if self.systems_to_validate else 0
        }
        
        # Individual system results
        for system_id in self.systems_to_validate:
            system_result = {
                'validation_outcome': self.validation_outcomes.get(system_id, 'pending'),
                'validation_score': self.validation_scores.get(system_id, 0.0),
                'evidence_count': len(self.validation_evidence.get(system_id, [])),
                'issues_identified': len(self.validation_issues.get(system_id, [])),
                'improvement_actions': len(self.improvement_actions.get(system_id, []))
            }
            validation_results['system_validation_results'][str(system_id)] = system_result
        
        # Quality assessment
        if self.validation_scores:
            scores = list(self.validation_scores.values())
            validation_results['quality_assessment'] = {
                'average_validation_score': sum(scores) / len(scores),
                'validation_consistency': 1.0 - (max(scores) - min(scores)) if len(scores) > 1 else 1.0,
                'inter_validator_agreement': self.inter_validator_agreement or 0.0
            }
        
        return validation_results