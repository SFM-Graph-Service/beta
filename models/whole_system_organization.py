"""
Whole System Organization (WSO) framework for Social Fabric Matrix modeling.

This module implements Hayden's foundational Whole System Organization concept,
which defines the complete system boundary and integrates all subsystems within
the SFM analysis framework. The WSO represents the totality of institutional
arrangements and their interactions within a defined scope.

Key Components:
- WholeSystemOrganization: The complete system being analyzed
- SystemBoundary: Definition and management of system limits
- SubSystemComponent: Individual subsystems within the WSO
- SystemIntegration: Tools for integrating subsystems
- BoundaryManager: Managing system boundary dynamics
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum, auto

from models.base_nodes import Node
from models.meta_entities import TimeSlice, SpatialUnit
from models.sfm_enums import (
    SystemBoundaryType,
    SystemLevel,
    InstitutionalScope,
)


class BoundaryPermeability(Enum):
    """How permeable system boundaries are."""
    
    COMPLETELY_OPEN = auto()      # No restrictions on crossing
    SEMI_PERMEABLE = auto()       # Selective permeability
    CONTROLLED_ACCESS = auto()    # Managed boundary crossing
    RESTRICTED = auto()           # Limited boundary crossing
    CLOSED = auto()               # No boundary crossing allowed


class SystemCoherence(Enum):
    """Level of coherence within the system."""
    
    HIGHLY_INTEGRATED = auto()    # Strong integration across components
    MODERATELY_INTEGRATED = auto() # Good integration with some gaps
    LOOSELY_COUPLED = auto()      # Weak integration, autonomous components
    FRAGMENTED = auto()           # Little integration, competing components
    CHAOTIC = auto()              # No coherent organization


class SubSystemType(Enum):
    """Types of subsystems within WSO."""
    
    INSTITUTIONAL_SUBSYSTEM = auto()  # Formal institutions
    CULTURAL_SUBSYSTEM = auto()       # Values, beliefs, norms
    TECHNOLOGICAL_SUBSYSTEM = auto()  # Technology systems
    ECOLOGICAL_SUBSYSTEM = auto()     # Environmental systems
    ECONOMIC_SUBSYSTEM = auto()       # Economic arrangements
    POLITICAL_SUBSYSTEM = auto()      # Governance systems
    SOCIAL_SUBSYSTEM = auto()         # Social networks and relationships


@dataclass
class SystemBoundary(Node):
    """Defines the boundaries of the whole system organization."""
    
    boundary_type: SystemBoundaryType = SystemBoundaryType.CONCEPTUAL_BOUNDARY
    permeability: BoundaryPermeability = BoundaryPermeability.SEMI_PERMEABLE
    
    # Boundary definition
    boundary_criteria: List[str] = field(default_factory=lambda: [])
    inclusion_rules: List[str] = field(default_factory=lambda: [])
    exclusion_rules: List[str] = field(default_factory=lambda: [])
    
    # Spatial and temporal bounds
    spatial_boundaries: List[SpatialUnit] = field(default_factory=lambda: [])
    temporal_boundaries: List[TimeSlice] = field(default_factory=lambda: [])
    
    # Boundary characteristics
    boundary_strength: Optional[float] = None
    boundary_stability: Optional[float] = None
    boundary_legitimacy: Optional[float] = None
    
    # Boundary management
    boundary_enforcement: List[str] = field(default_factory=lambda: [])
    boundary_exceptions: List[str] = field(default_factory=lambda: [])
    cross_boundary_mechanisms: List[str] = field(default_factory=lambda: [])
    
    # Dynamic properties
    boundary_evolution_history: List[datetime] = field(default_factory=lambda: [])
    boundary_change_drivers: List[str] = field(default_factory=lambda: [])
    anticipated_boundary_changes: List[str] = field(default_factory=lambda: [])
    
    def assess_boundary_integrity(self) -> Dict[str, float]:
        """Assess the integrity of the system boundary."""
        integrity_factors: Dict[str, float] = {}
        
        # Clarity of definition
        if self.boundary_criteria and self.inclusion_rules:
            clarity_score = min(len(self.boundary_criteria) / 3.0, 1.0)
            integrity_factors['clarity'] = clarity_score
        else:
            integrity_factors['clarity'] = 0.0
        
        # Strength and stability
        if self.boundary_strength is not None:
            integrity_factors['strength'] = self.boundary_strength
        
        if self.boundary_stability is not None:
            integrity_factors['stability'] = self.boundary_stability
        
        # Legitimacy
        if self.boundary_legitimacy is not None:
            integrity_factors['legitimacy'] = self.boundary_legitimacy
        
        # Enforcement mechanisms
        if self.boundary_enforcement:
            enforcement_score = min(len(self.boundary_enforcement) / 2.0, 1.0)
            integrity_factors['enforcement'] = enforcement_score
        else:
            integrity_factors['enforcement'] = 0.0
        
        # Overall integrity
        if integrity_factors:
            overall_integrity = sum(integrity_factors.values()) / len(integrity_factors)
            integrity_factors['overall'] = overall_integrity
        
        return integrity_factors
    
    def identify_boundary_tensions(self) -> List[Dict[str, Any]]:
        """Identify tensions or problems with the boundary definition."""
        tensions: List[Dict[str, Any]] = []
        
        # Check for conflicting rules
        if self.inclusion_rules and self.exclusion_rules:
            inclusion_terms = set(' '.join(self.inclusion_rules).lower().split())
            exclusion_terms = set(' '.join(self.exclusion_rules).lower().split())
            conflicts = inclusion_terms.intersection(exclusion_terms)
            
            if conflicts:
                tensions.append({
                    'type': 'rule_conflict',
                    'description': f'Conflicting terms in inclusion/exclusion rules: {conflicts}',
                    'severity': 'medium'
                })
        
        # Check for weak boundary enforcement
        if not self.boundary_enforcement and self.boundary_type != SystemBoundaryType.CONCEPTUAL_BOUNDARY:
            tensions.append({
                'type': 'weak_enforcement',
                'description': 'No enforcement mechanisms defined for non-conceptual boundary',
                'severity': 'high'
            })
        
        # Check for excessive exceptions
        if len(self.boundary_exceptions) > len(self.inclusion_rules):
            tensions.append({
                'type': 'excessive_exceptions',
                'description': 'More exceptions than rules may indicate unclear boundary',
                'severity': 'medium'
            })
        
        # Check for stability issues
        if self.boundary_stability is not None and self.boundary_stability < 0.3:
            tensions.append({
                'type': 'instability',
                'description': 'Low boundary stability may cause system coherence issues',
                'severity': 'high'
            })
        
        return tensions
    
    def calculate_permeability_index(self) -> float:
        """Calculate a numeric index of boundary permeability."""
        permeability_values = {
            BoundaryPermeability.COMPLETELY_OPEN: 1.0,
            BoundaryPermeability.SEMI_PERMEABLE: 0.7,
            BoundaryPermeability.CONTROLLED_ACCESS: 0.5,
            BoundaryPermeability.RESTRICTED: 0.3,
            BoundaryPermeability.CLOSED: 0.0
        }
        
        base_permeability = permeability_values[self.permeability]
        
        # Adjust based on cross-boundary mechanisms
        if self.cross_boundary_mechanisms:
            mechanism_bonus = min(len(self.cross_boundary_mechanisms) * 0.1, 0.3)
            base_permeability = min(base_permeability + mechanism_bonus, 1.0)
        
        return base_permeability


@dataclass
class SubSystemComponent(Node):
    """Individual subsystem within the whole system organization."""
    
    subsystem_type: SubSystemType = SubSystemType.INSTITUTIONAL_SUBSYSTEM
    system_level: SystemLevel = SystemLevel.ORGANIZATIONAL
    
    # Component elements
    component_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    component_actors: List[uuid.UUID] = field(default_factory=lambda: [])
    component_processes: List[uuid.UUID] = field(default_factory=lambda: [])
    component_resources: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # Integration properties
    integration_level: Optional[float] = None
    autonomy_level: Optional[float] = None
    dependency_relationships: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # Performance characteristics
    subsystem_performance: Optional[float] = None
    contribution_to_system: Optional[float] = None
    internal_coherence: Optional[float] = None
    
    # Subsystem boundaries
    subsystem_boundaries: List[SystemBoundary] = field(default_factory=lambda: [])
    boundary_interfaces: Dict[uuid.UUID, str] = field(default_factory=lambda: {})
    
    # Evolution and adaptation
    adaptation_capacity: Optional[float] = None
    change_drivers: List[str] = field(default_factory=lambda: [])
    recent_changes: List[str] = field(default_factory=lambda: [])
    
    def calculate_system_integration_score(self) -> float:
        """Calculate how well this subsystem integrates with the overall system."""
        integration_factors: List[float] = []
        
        if self.integration_level is not None:
            integration_factors.append(self.integration_level * 0.4)
        
        # Dependency relationships indicate integration
        if self.dependency_relationships:
            dependency_score = min(len(self.dependency_relationships) / 3.0, 1.0)
            integration_factors.append(dependency_score * 0.3)
        
        # Boundary interfaces indicate integration
        if self.boundary_interfaces:
            interface_score = min(len(self.boundary_interfaces) / 4.0, 1.0)
            integration_factors.append(interface_score * 0.3)
        
        return sum(integration_factors) if integration_factors else 0.0
    
    def assess_subsystem_health(self) -> Dict[str, float]:
        """Assess overall health of the subsystem."""
        health_metrics: Dict[str, float] = {}
        
        if self.subsystem_performance is not None:
            health_metrics['performance'] = self.subsystem_performance
        
        if self.internal_coherence is not None:
            health_metrics['coherence'] = self.internal_coherence
        
        if self.adaptation_capacity is not None:
            health_metrics['adaptability'] = self.adaptation_capacity
        
        # Integration health
        integration_score = self.calculate_system_integration_score()
        health_metrics['integration'] = integration_score
        
        # Balance between autonomy and integration
        if self.autonomy_level is not None and integration_score > 0:
            balance_score = 1.0 - abs(self.autonomy_level - integration_score)
            health_metrics['autonomy_integration_balance'] = balance_score
        
        # Overall health
        if health_metrics:
            overall_health = sum(health_metrics.values()) / len(health_metrics)
            health_metrics['overall'] = overall_health
        
        return health_metrics
    
    def identify_improvement_opportunities(self) -> List[str]:
        """Identify opportunities for subsystem improvement."""
        opportunities: List[str] = []
        
        health_metrics = self.assess_subsystem_health()
        
        # Performance improvements
        if health_metrics.get('performance', 1.0) < 0.6:
            opportunities.append("Improve subsystem performance through process optimization")
        
        # Coherence improvements
        if health_metrics.get('coherence', 1.0) < 0.5:
            opportunities.append("Strengthen internal organization and coordination")
        
        # Integration improvements
        if health_metrics.get('integration', 1.0) < 0.4:
            opportunities.append("Enhance integration with other subsystems")
        
        # Adaptability improvements
        if health_metrics.get('adaptability', 1.0) < 0.5:
            opportunities.append("Build capacity for adaptation and change")
        
        # Balance improvements
        if health_metrics.get('autonomy_integration_balance', 1.0) < 0.6:
            opportunities.append("Better balance autonomy and integration needs")
        
        return opportunities


@dataclass
class WholeSystemOrganization(Node):
    """The complete system being analyzed in the Social Fabric Matrix framework."""
    
    system_level: SystemLevel = SystemLevel.ORGANIZATIONAL
    system_scope: InstitutionalScope = InstitutionalScope.LOCAL
    
    # System definition
    system_purpose: str = ""
    system_mission: Optional[str] = None
    key_functions: List[str] = field(default_factory=lambda: [])
    
    # System boundaries
    primary_boundary: Optional[SystemBoundary] = None
    secondary_boundaries: List[SystemBoundary] = field(default_factory=lambda: [])
    
    # System components
    subsystems: Dict[uuid.UUID, SubSystemComponent] = field(default_factory=lambda: {})
    core_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    supporting_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # System relationships
    internal_relationships: List[uuid.UUID] = field(default_factory=lambda: [])
    external_relationships: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # System properties
    system_coherence: SystemCoherence = SystemCoherence.MODERATELY_INTEGRATED
    coherence_score: Optional[float] = None
    system_resilience: Optional[float] = None
    system_adaptability: Optional[float] = None
    
    # Performance and effectiveness
    system_performance: Optional[float] = None
    effectiveness_indicators: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # System dynamics
    equilibrium_state: Optional[str] = None
    change_capacity: Optional[float] = None
    innovation_capacity: Optional[float] = None
    
    # Environmental context
    environmental_context: Dict[str, Any] = field(default_factory=lambda: {})
    external_pressures: List[str] = field(default_factory=lambda: [])
    environmental_constraints: List[str] = field(default_factory=lambda: [])
    
    # Matrix integration
    sfm_matrices: List[uuid.UUID] = field(default_factory=lambda: [])
    matrix_coverage: Optional[float] = None
    
    def add_subsystem(self, subsystem: SubSystemComponent) -> None:
        """Add a subsystem to the WSO."""
        self.subsystems[subsystem.id] = subsystem
    
    def remove_subsystem(self, subsystem_id: uuid.UUID) -> None:
        """Remove a subsystem from the WSO."""
        if subsystem_id in self.subsystems:
            del self.subsystems[subsystem_id]
    
    def calculate_system_coherence(self) -> float:
        """Calculate quantitative system coherence score."""
        coherence_factors: List[float] = []
        
        if not self.subsystems:
            return 0.0
        
        # Average subsystem integration
        integration_scores: List[float] = []
        for subsystem in self.subsystems.values():
            integration_score = subsystem.calculate_system_integration_score()
            integration_scores.append(integration_score)
        
        if integration_scores:
            avg_integration = sum(integration_scores) / len(integration_scores)
            coherence_factors.append(avg_integration * 0.4)
        
        # Boundary coherence
        if self.primary_boundary:
            boundary_integrity = self.primary_boundary.assess_boundary_integrity()
            boundary_score = boundary_integrity.get('overall', 0.0)
            coherence_factors.append(boundary_score * 0.3)
        
        # Functional coherence (alignment with purpose)
        if self.system_purpose and self.key_functions:
            function_coherence = min(len(self.key_functions) / 5.0, 1.0)
            coherence_factors.append(function_coherence * 0.3)
        
        coherence_score = sum(coherence_factors) if coherence_factors else 0.0
        self.coherence_score = coherence_score
        return coherence_score
    
    def assess_system_health(self) -> Dict[str, Any]:
        """Comprehensive assessment of system health."""
        health_assessment: Dict[str, Any] = {
            'overall_score': 0.0,
            'subsystem_health': {},
            'boundary_health': {},
            'integration_health': {},
            'performance_health': {},
            'recommendations': []
        }
        
        health_scores: List[float] = []
        
        # Subsystem health
        subsystem_health_scores: List[float] = []
        for subsystem_id, subsystem in self.subsystems.items():
            subsystem_health = subsystem.assess_subsystem_health()
            health_assessment['subsystem_health'][str(subsystem_id)] = subsystem_health
            if 'overall' in subsystem_health:
                subsystem_health_scores.append(subsystem_health['overall'])
        
        if subsystem_health_scores:
            avg_subsystem_health = sum(subsystem_health_scores) / len(subsystem_health_scores)
            health_scores.append(avg_subsystem_health)
            health_assessment['subsystem_health']['average'] = avg_subsystem_health
        
        # Boundary health
        if self.primary_boundary:
            boundary_integrity = self.primary_boundary.assess_boundary_integrity()
            health_assessment['boundary_health'] = boundary_integrity
            if 'overall' in boundary_integrity:
                health_scores.append(boundary_integrity['overall'])
        
        # Integration health (coherence)
        coherence_score = self.calculate_system_coherence()
        health_assessment['integration_health']['coherence_score'] = coherence_score
        health_scores.append(coherence_score)
        
        # Performance health
        if self.system_performance is not None:
            health_assessment['performance_health']['system_performance'] = self.system_performance
            health_scores.append(self.system_performance)
        
        if self.system_resilience is not None:
            health_assessment['performance_health']['resilience'] = self.system_resilience
            health_scores.append(self.system_resilience)
        
        if self.system_adaptability is not None:
            health_assessment['performance_health']['adaptability'] = self.system_adaptability
            health_scores.append(self.system_adaptability)
        
        # Overall health score
        if health_scores:
            overall_score = sum(health_scores) / len(health_scores)
            health_assessment['overall_score'] = overall_score
        
        # Generate recommendations
        health_assessment['recommendations'] = self._generate_health_recommendations(health_assessment)
        
        return health_assessment
    
    def _generate_health_recommendations(self, health_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health assessment."""
        recommendations: List[str] = []
        
        overall_score = health_assessment.get('overall_score', 0.0)
        
        if overall_score < 0.4:
            recommendations.append("System requires urgent attention - multiple critical issues identified")
        elif overall_score < 0.6:
            recommendations.append("System needs improvement in several areas")
        
        # Subsystem recommendations
        subsystem_health = health_assessment.get('subsystem_health', {})
        avg_subsystem_health = subsystem_health.get('average', 1.0)
        if avg_subsystem_health < 0.5:
            recommendations.append("Focus on improving individual subsystem performance and coherence")
        
        # Boundary recommendations
        boundary_health = health_assessment.get('boundary_health', {})
        boundary_overall = boundary_health.get('overall', 1.0)
        if boundary_overall < 0.5:
            recommendations.append("Strengthen system boundary definition and enforcement")
        
        # Integration recommendations
        integration_health = health_assessment.get('integration_health', {})
        coherence_score = integration_health.get('coherence_score', 1.0)
        if coherence_score < 0.4:
            recommendations.append("Improve integration and coordination between subsystems")
        
        # Performance recommendations
        performance_health = health_assessment.get('performance_health', {})
        if performance_health.get('system_performance', 1.0) < 0.5:
            recommendations.append("Address performance issues through process optimization")
        
        if performance_health.get('resilience', 1.0) < 0.5:
            recommendations.append("Build system resilience and capacity to handle disruptions")
        
        if performance_health.get('adaptability', 1.0) < 0.5:
            recommendations.append("Enhance system capacity for adaptation and change")
        
        return recommendations
    
    def identify_system_boundaries_issues(self) -> List[Dict[str, Any]]:
        """Identify issues with system boundary definition."""
        boundary_issues: List[Dict[str, Any]] = []
        
        # Check primary boundary
        if not self.primary_boundary:
            boundary_issues.append({
                'type': 'missing_primary_boundary',
                'description': 'No primary system boundary defined',
                'severity': 'high',
                'recommendation': 'Define clear primary system boundary'
            })
        else:
            boundary_tensions = self.primary_boundary.identify_boundary_tensions()
            boundary_issues.extend(boundary_tensions)
        
        # Check for conflicting boundaries
        all_boundaries = [self.primary_boundary] + self.secondary_boundaries
        all_boundaries_filtered = [b for b in all_boundaries if b is not None]
        
        if len(all_boundaries_filtered) > 1:
            # Simple check for conflicting boundary criteria
            all_criteria: List[str] = []
            for boundary in all_boundaries_filtered:
                all_criteria.extend(boundary.boundary_criteria)
            
            # Look for potential conflicts (simplified)
            if len(set(all_criteria)) < len(all_criteria):
                boundary_issues.append({
                    'type': 'overlapping_boundaries',
                    'description': 'Multiple boundaries may have overlapping or conflicting criteria',
                    'severity': 'medium',
                    'recommendation': 'Review boundary definitions for conflicts and overlaps'
                })
        
        return boundary_issues
    
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        system_report: Dict[str, Any] = {
            'system_overview': {
                'system_purpose': self.system_purpose,
                'system_level': self.system_level.name,
                'system_scope': self.system_scope.name,
                'coherence_level': self.system_coherence.name,
                'num_subsystems': len(self.subsystems),
                'num_core_institutions': len(self.core_institutions),
                'num_supporting_institutions': len(self.supporting_institutions)
            },
            'health_assessment': self.assess_system_health(),
            'boundary_analysis': {
                'has_primary_boundary': self.primary_boundary is not None,
                'num_secondary_boundaries': len(self.secondary_boundaries),
                'boundary_issues': self.identify_system_boundaries_issues()
            },
            'subsystem_analysis': {},
            'performance_metrics': {
                'system_performance': self.system_performance,
                'coherence_score': self.coherence_score,
                'resilience': self.system_resilience,
                'adaptability': self.system_adaptability
            },
            'integration_metrics': {},
            'recommendations': []
        }
        
        # Analyze each subsystem
        for subsystem_id, subsystem in self.subsystems.items():
            subsystem_health = subsystem.assess_subsystem_health()
            improvement_opportunities = subsystem.identify_improvement_opportunities()
            
            system_report['subsystem_analysis'][str(subsystem_id)] = {
                'type': subsystem.subsystem_type.name,
                'health_metrics': subsystem_health,
                'improvement_opportunities': improvement_opportunities
            }
        
        # Calculate integration metrics
        if self.subsystems:
            integration_scores = [
                subsystem.calculate_system_integration_score()
                for subsystem in self.subsystems.values()
            ]
            
            system_report['integration_metrics'] = {
                'average_integration': sum(integration_scores) / len(integration_scores),
                'integration_variance': sum((score - sum(integration_scores) / len(integration_scores)) ** 2 
                                          for score in integration_scores) / len(integration_scores),
                'highly_integrated_subsystems': sum(1 for score in integration_scores if score > 0.7),
                'poorly_integrated_subsystems': sum(1 for score in integration_scores if score < 0.3)
            }
        
        # Generate overall recommendations
        health_recommendations = system_report['health_assessment'].get('recommendations', [])
        system_report['recommendations'] = health_recommendations
        
        return system_report


@dataclass
class BoundaryManager(Node):
    """Manages system boundary dynamics and evolution."""
    
    managed_boundaries: List[uuid.UUID] = field(default_factory=lambda: [])
    boundary_policies: List[str] = field(default_factory=lambda: [])
    
    # Boundary monitoring
    boundary_violations: List[Dict[str, Any]] = field(default_factory=lambda: [])
    boundary_changes: List[Dict[str, Any]] = field(default_factory=lambda: [])
    
    # Management effectiveness
    management_effectiveness: Optional[float] = None
    
    def monitor_boundary_violations(self, boundary: SystemBoundary, 
                                  recent_activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Monitor for boundary violations."""
        violations: List[Dict[str, Any]] = []
        
        for activity in recent_activities:
            activity_type = activity.get('type', '')
            activity_location = activity.get('location', '')
            
            # Check against exclusion rules
            for exclusion_rule in boundary.exclusion_rules:
                if exclusion_rule.lower() in activity_type.lower():
                    violations.append({
                        'violation_type': 'exclusion_rule_violated',
                        'rule': exclusion_rule,
                        'activity': activity,
                        'severity': 'medium',
                        'timestamp': datetime.now()
                    })
            
            # Check boundary enforcement
            if boundary.boundary_type == SystemBoundaryType.PHYSICAL_BOUNDARY:
                # Check if activity violates spatial boundaries
                if activity_location and not self._check_spatial_compliance(activity_location, boundary):
                    violations.append({
                        'violation_type': 'spatial_boundary_violated',
                        'location': activity_location,
                        'activity': activity,
                        'severity': 'high',
                        'timestamp': datetime.now()
                    })
        
        # Store violations
        self.boundary_violations.extend(violations)
        
        return violations
    
    def _check_spatial_compliance(self, location: str, boundary: SystemBoundary) -> bool:
        """Check if location complies with spatial boundaries."""
        # Simplified implementation - in practice would use geographic data
        for spatial_unit in boundary.spatial_boundaries:
            if location.lower() in spatial_unit.name.lower():
                return True
        return False
    
    def recommend_boundary_adjustments(self, boundary: SystemBoundary) -> List[Dict[str, Any]]:
        """Recommend adjustments to boundary based on violations and performance."""
        recommendations: List[Dict[str, Any]] = []
        
        # Analyze violation patterns
        recent_violations = [v for v in self.boundary_violations 
                           if (datetime.now() - v['timestamp']).days <= 30]
        
        if len(recent_violations) > 10:  # Many recent violations
            recommendations.append({
                'type': 'increase_permeability',
                'rationale': 'High violation rate suggests boundary may be too restrictive',
                'priority': 'high'
            })
        
        # Check boundary integrity
        boundary_integrity = boundary.assess_boundary_integrity()
        if boundary_integrity.get('overall', 1.0) < 0.4:
            recommendations.append({
                'type': 'strengthen_definition',
                'rationale': 'Poor boundary integrity requires clearer definition',
                'priority': 'high'
            })
        
        # Check enforcement
        if boundary_integrity.get('enforcement', 1.0) < 0.3:
            recommendations.append({
                'type': 'improve_enforcement',
                'rationale': 'Weak enforcement mechanisms need strengthening',
                'priority': 'medium'
            })
        
        return recommendations
    
    def calculate_management_effectiveness(self) -> float:
        """Calculate effectiveness of boundary management."""
        effectiveness_factors: List[float] = []
        
        # Violation rate (lower is better)
        recent_violations = [v for v in self.boundary_violations 
                           if (datetime.now() - v['timestamp']).days <= 30]
        violation_rate = len(recent_violations) / 30.0  # Violations per day
        violation_score = max(0.0, 1.0 - violation_rate * 0.1)  # Penalty for violations
        effectiveness_factors.append(violation_score * 0.4)
        
        # Response time to violations (simplified)
        if recent_violations:
            # Assume we want to respond within 24 hours
            quick_responses = sum(1 for v in recent_violations 
                                if 'response_time' in v and v['response_time'] <= 24)
            response_rate = quick_responses / len(recent_violations)
            effectiveness_factors.append(response_rate * 0.3)
        else:
            effectiveness_factors.append(1.0 * 0.3)  # No violations is good
        
        # Policy compliance
        if self.boundary_policies:
            policy_score = min(len(self.boundary_policies) / 3.0, 1.0)
            effectiveness_factors.append(policy_score * 0.3)
        
        effectiveness = sum(effectiveness_factors)
        self.management_effectiveness = effectiveness
        return effectiveness