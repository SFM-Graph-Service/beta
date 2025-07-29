"""
Comprehensive Policy Evaluation Framework for Social Fabric Matrix analysis.

This module implements Hayden's systematic approach to policy evaluation using
SFM analysis. The framework provides tools for comparing policy alternatives
based on their effects on institutional deliveries, system performance, and
social provisioning outcomes.

Key Components:
- PolicyEvaluationFramework: Main evaluation orchestration
- PolicyImpactAssessment: Assessment of policy impacts on matrix relationships
- DeliveryImpactAnalysis: Analysis of policy effects on delivery systems
- PolicyComparison: Systematic comparison of policy alternatives
- ImplementationFeasibilityAnalysis: Assessment of implementation feasibility
- PolicyConsequenceAnalysis: Analysis of policy consequences through matrix effects
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum, auto
import statistics
import math

from models.base_nodes import Node
from models.core_nodes import Policy, Institution, Actor
from models.delivery_systems import DeliveryQuantification, DeliveryFlow
from models.matrix_construction import MatrixCell, DeliveryMatrix
from models.problem_solving_framework import PolicyAlternativeEvaluation
from models.sfm_enums import (
    PolicyType,
    PolicyScope,
    ImplementationComplexity,
    PolicyEffectiveness,
    EvaluationMethod,
    SystemLevel,
)


class PolicyImpactType(Enum):
    """Types of policy impacts on institutional systems."""
    
    DIRECT_IMPACT = auto()           # Direct effect on specific institutions
    INDIRECT_IMPACT = auto()         # Indirect effects through relationships
    SYSTEMIC_IMPACT = auto()         # System-wide transformational effects
    UNINTENDED_CONSEQUENCE = auto()  # Unexpected side effects
    SPILLOVER_EFFECT = auto()        # Effects on related systems
    CUMULATIVE_EFFECT = auto()       # Long-term accumulative impacts


class PolicyMechanism(Enum):
    """Mechanisms through which policies affect institutional systems."""
    
    REGULATORY_MECHANISM = auto()    # Rules, regulations, compliance
    INCENTIVE_MECHANISM = auto()     # Economic incentives, rewards
    CAPACITY_MECHANISM = auto()      # Building institutional capacity
    COORDINATION_MECHANISM = auto()  # Improving coordination
    INFORMATION_MECHANISM = auto()   # Information provision, transparency
    STRUCTURAL_MECHANISM = auto()    # Changing institutional structures


class EvaluationCriteria(Enum):
    """Criteria for evaluating policy alternatives."""
    
    EFFECTIVENESS = auto()           # Achievement of policy objectives
    EFFICIENCY = auto()              # Resource use optimization
    EQUITY = auto()                  # Fair distribution of benefits/costs
    SUSTAINABILITY = auto()          # Long-term viability
    FEASIBILITY = auto()             # Implementation practicality
    ACCEPTABILITY = auto()           # Stakeholder acceptance
    COHERENCE = auto()               # Internal consistency
    ADAPTABILITY = auto()            # Ability to adapt to change


class PolicyConsequenceType(Enum):
    """Types of consequences from policy implementation."""
    
    INTENDED_OUTCOME = auto()        # Expected positive outcomes
    UNINTENDED_BENEFIT = auto()      # Unexpected positive effects
    UNINTENDED_HARM = auto()         # Unexpected negative effects
    IMPLEMENTATION_FAILURE = auto()  # Failure to implement as designed
    ADAPTATION_RESPONSE = auto()     # System adaptation to policy
    RESISTANCE_REACTION = auto()     # Institutional resistance effects


@dataclass
class PolicyImpactAssessment(Node):
    """Assessment of policy impacts on institutional systems and matrix relationships."""
    
    policy_id: uuid.UUID
    impact_assessment_date: datetime = field(default_factory=datetime.now)
    
    # Impact identification
    identified_impacts: Dict[PolicyImpactType, List[Dict[str, Any]]] = field(default_factory=lambda: {})
    affected_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    affected_delivery_relationships: List[Tuple[uuid.UUID, uuid.UUID]] = field(default_factory=lambda: [])
    
    # Impact quantification
    impact_magnitudes: Dict[str, float] = field(default_factory=lambda: {})  # Impact ID -> magnitude
    impact_probabilities: Dict[str, float] = field(default_factory=lambda: {})  # Impact ID -> probability
    impact_timing: Dict[str, timedelta] = field(default_factory=lambda: {})  # Impact ID -> time to effect
    
    # Mechanism analysis
    policy_mechanisms: List[PolicyMechanism] = field(default_factory=lambda: [])
    mechanism_effectiveness: Dict[PolicyMechanism, float] = field(default_factory=lambda: {})
    
    # Uncertainty and confidence
    impact_confidence_levels: Dict[str, float] = field(default_factory=lambda: {})
    uncertainty_factors: List[str] = field(default_factory=lambda: [])
    sensitivity_analysis: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})
    
    def assess_direct_impacts(self, policy: Policy, 
                            affected_institutions: List[Institution]) -> Dict[str, Any]:
        """Assess direct impacts of policy on target institutions."""
        direct_impacts = {
            'institutional_changes': {},
            'capacity_effects': {},
            'performance_impacts': {},
            'relationship_changes': {}
        }
        
        for institution in affected_institutions:
            institution_impacts = {
                'operational_changes': [],
                'capacity_changes': {},
                'performance_changes': {},
                'new_relationships': [],
                'modified_relationships': []
            }
            
            # Analyze operational changes based on policy type
            if policy.policy_type == PolicyType.REGULATORY:
                institution_impacts['operational_changes'].extend([
                    'Compliance procedures implementation',
                    'Reporting requirements adoption',
                    'Process modifications for regulatory alignment'
                ])
            elif policy.policy_type == PolicyType.INCENTIVE_BASED:
                institution_impacts['operational_changes'].extend([
                    'Incentive structure adaptation',
                    'Performance measurement systems',
                    'Resource allocation adjustments'
                ])
            
            # Assess capacity effects
            if policy.resource_requirements:
                total_resources = sum(policy.resource_requirements.values())
                # Simplified capacity impact assessment
                capacity_impact = min(total_resources / 100000.0, 1.0)  # Normalize to $100k
                institution_impacts['capacity_changes'] = {
                    'resource_capacity': capacity_impact,
                    'operational_capacity': capacity_impact * 0.7,
                    'innovation_capacity': capacity_impact * 0.5
                }
            
            # Performance impact estimation
            if policy.expected_outcomes:
                # Simplified performance impact based on policy scope
                performance_multiplier = {
                    PolicyScope.INSTITUTION_SPECIFIC: 0.8,
                    PolicyScope.SECTOR_WIDE: 0.6,
                    PolicyScope.SYSTEM_WIDE: 0.4,
                    PolicyScope.CROSS_SYSTEM: 0.3
                }.get(policy.policy_scope, 0.5)
                
                institution_impacts['performance_changes'] = {
                    'efficiency_change': performance_multiplier,
                    'effectiveness_change': performance_multiplier * 0.8,
                    'service_quality_change': performance_multiplier * 0.9
                }
            
            direct_impacts['institutional_changes'][str(institution.id)] = institution_impacts
        
        return direct_impacts
    
    def assess_indirect_impacts(self, policy: Policy, 
                              delivery_matrix: DeliveryMatrix) -> Dict[str, Any]:
        """Assess indirect impacts through delivery relationship changes."""
        indirect_impacts = {
            'delivery_flow_changes': {},
            'network_effects': {},
            'cascade_effects': [],
            'equilibrium_shifts': {}
        }
        
        # Analyze delivery flow changes
        for (source_id, target_id), cell in delivery_matrix.matrix_cells.items():
            if source_id in self.affected_institutions or target_id in self.affected_institutions:
                # Estimate impact on delivery relationship
                impact_factors = []
                
                # Source institution impact
                if source_id in self.affected_institutions:
                    source_impact = self.impact_magnitudes.get(str(source_id), 0.0)
                    impact_factors.append(source_impact)
                
                # Target institution impact
                if target_id in self.affected_institutions:
                    target_impact = self.impact_magnitudes.get(str(target_id), 0.0)
                    impact_factors.append(target_impact * 0.5)  # Receiving side has less direct impact
                
                if impact_factors:
                    avg_impact = sum(impact_factors) / len(impact_factors)
                    indirect_impacts['delivery_flow_changes'][(source_id, target_id)] = {
                        'estimated_change': avg_impact,
                        'change_direction': 'increase' if avg_impact > 0 else 'decrease',
                        'confidence': 0.6  # Medium confidence for indirect effects
                    }
        
        # Network effects analysis
        affected_count = len(self.affected_institutions)
        total_institutions = len(set(delivery_matrix.row_institutions + delivery_matrix.column_institutions))
        
        if total_institutions > 0:
            network_coverage = affected_count / total_institutions
            indirect_impacts['network_effects'] = {
                'network_coverage': network_coverage,
                'systemic_impact_potential': network_coverage * 0.8,
                'adaptation_requirements': network_coverage * 0.6
            }
        
        return indirect_impacts
    
    def assess_systemic_impacts(self, policy: Policy, 
                              system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system-wide transformational impacts."""
        systemic_impacts = {
            'system_transformation_potential': 0.0,
            'institutional_paradigm_shifts': [],
            'structural_reorganization': {},
            'emergent_properties': []
        }
        
        # System transformation potential
        transformation_factors = []
        
        # Policy scope factor
        scope_weights = {
            PolicyScope.INSTITUTION_SPECIFIC: 0.2,
            PolicyScope.SECTOR_WIDE: 0.5,
            PolicyScope.SYSTEM_WIDE: 0.8,
            PolicyScope.CROSS_SYSTEM: 1.0
        }
        transformation_factors.append(scope_weights.get(policy.policy_scope, 0.3))
        
        # Mechanism strength factor
        if self.mechanism_effectiveness:
            avg_mechanism_effectiveness = sum(self.mechanism_effectiveness.values()) / len(self.mechanism_effectiveness)
            transformation_factors.append(avg_mechanism_effectiveness)
        
        # Resource scale factor
        if policy.resource_requirements:
            total_resources = sum(policy.resource_requirements.values())
            resource_scale = min(total_resources / 1000000.0, 1.0)  # Normalize to $1M
            transformation_factors.append(resource_scale)
        
        if transformation_factors:
            systemic_impacts['system_transformation_potential'] = sum(transformation_factors) / len(transformation_factors)
        
        # Identify potential paradigm shifts
        if systemic_impacts['system_transformation_potential'] > 0.7:
            systemic_impacts['institutional_paradigm_shifts'] = [
                'New coordination mechanisms',
                'Changed accountability structures',
                'Modified performance metrics',
                'Altered institutional relationships'
            ]
        
        return systemic_impacts
    
    def calculate_net_impact_score(self) -> Dict[str, float]:
        """Calculate net impact score considering all impact types."""
        net_impact = {
            'positive_impacts': 0.0,
            'negative_impacts': 0.0,
            'net_score': 0.0,
            'impact_distribution': {}
        }
        
        # Aggregate positive and negative impacts
        for impact_type, impacts in self.identified_impacts.items():
            for impact in impacts:
                impact_value = impact.get('magnitude', 0.0)
                probability = impact.get('probability', 1.0)
                weighted_impact = impact_value * probability
                
                if weighted_impact > 0:
                    net_impact['positive_impacts'] += weighted_impact
                else:
                    net_impact['negative_impacts'] += abs(weighted_impact)
                
                # Track distribution by impact type
                if impact_type.name not in net_impact['impact_distribution']:
                    net_impact['impact_distribution'][impact_type.name] = 0.0
                net_impact['impact_distribution'][impact_type.name] += weighted_impact
        
        # Calculate net score
        total_positive = net_impact['positive_impacts']
        total_negative = net_impact['negative_impacts']
        
        if total_positive + total_negative > 0:
            net_impact['net_score'] = (total_positive - total_negative) / (total_positive + total_negative)
        
        return net_impact


@dataclass
class DeliveryImpactAnalysis(Node):
    """Analysis of policy effects on delivery systems and flows."""
    
    policy_id: uuid.UUID
    baseline_delivery_matrix: uuid.UUID  # Reference to baseline matrix
    
    # Delivery impact projections
    projected_delivery_changes: Dict[Tuple[uuid.UUID, uuid.UUID], float] = field(default_factory=lambda: {})
    flow_enhancement_opportunities: List[Dict[str, Any]] = field(default_factory=lambda: [])
    bottleneck_resolution_potential: Dict[str, float] = field(default_factory=lambda: {})
    
    # Network effects
    network_efficiency_changes: Dict[str, float] = field(default_factory=lambda: {})
    connectivity_improvements: Dict[str, int] = field(default_factory=lambda: {})
    resilience_impacts: Dict[str, float] = field(default_factory=lambda: {})
    
    # Quality and performance
    delivery_quality_impacts: Dict[str, float] = field(default_factory=lambda: {})
    service_performance_changes: Dict[str, Dict[str, float]] = field(default_factory=lambda: {})
    cost_effectiveness_changes: Dict[str, float] = field(default_factory=lambda: {})
    
    def analyze_delivery_flow_changes(self, policy: Policy, 
                                    current_matrix: DeliveryMatrix) -> Dict[str, Any]:
        """Analyze how policy affects delivery flows between institutions."""
        flow_analysis = {
            'enhanced_flows': [],
            'diminished_flows': [],
            'new_flows': [],
            'eliminated_flows': [],
            'flow_quality_changes': {}
        }
        
        # Analyze existing flows
        for (source_id, target_id), cell in current_matrix.matrix_cells.items():
            current_value = cell.delivery_value or 0.0
            
            # Estimate policy impact on this flow
            impact_estimate = self._estimate_flow_impact(policy, source_id, target_id, current_value)
            
            if impact_estimate != 0:
                self.projected_delivery_changes[(source_id, target_id)] = impact_estimate
                
                if impact_estimate > 0.1:  # Significant enhancement
                    flow_analysis['enhanced_flows'].append({
                        'source': source_id,
                        'target': target_id,
                        'current_value': current_value,
                        'projected_change': impact_estimate,
                        'mechanism': self._identify_enhancement_mechanism(policy, source_id, target_id)
                    })
                elif impact_estimate < -0.1:  # Significant diminishment
                    flow_analysis['diminished_flows'].append({
                        'source': source_id,
                        'target': target_id,
                        'current_value': current_value,
                        'projected_change': impact_estimate,
                        'reason': self._identify_diminishment_reason(policy, source_id, target_id)
                    })
        
        # Identify potential new flows
        new_flows = self._identify_potential_new_flows(policy, current_matrix)
        flow_analysis['new_flows'] = new_flows
        
        return flow_analysis
    
    def assess_network_efficiency_impacts(self, policy: Policy, 
                                        current_matrix: DeliveryMatrix) -> Dict[str, float]:
        """Assess policy impacts on overall network efficiency."""
        efficiency_impacts = {}
        
        # Calculate baseline efficiency metrics
        baseline_metrics = current_matrix.calculate_matrix_metrics()
        
        # Project post-policy metrics
        projected_metrics = self._project_post_policy_metrics(policy, current_matrix, baseline_metrics)
        
        # Calculate efficiency changes
        for metric_name, baseline_value in baseline_metrics.items():
            projected_value = projected_metrics.get(metric_name, baseline_value)
            change = projected_value - baseline_value
            efficiency_impacts[f"{metric_name}_change"] = change
        
        # Overall efficiency change
        key_metrics = ['matrix_density', 'average_delivery_strength', 'delivery_symmetry']
        efficiency_changes = [efficiency_impacts.get(f"{metric}_change", 0) for metric in key_metrics]
        if efficiency_changes:
            efficiency_impacts['overall_efficiency_change'] = sum(efficiency_changes) / len(efficiency_changes)
        
        self.network_efficiency_changes = efficiency_impacts
        return efficiency_impacts
    
    def identify_bottleneck_resolution_opportunities(self, policy: Policy, 
                                                   current_bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify how policy could resolve existing delivery bottlenecks."""
        resolution_opportunities = {
            'resolvable_bottlenecks': [],
            'partial_resolutions': [],
            'unaddressed_bottlenecks': [],
            'potential_new_bottlenecks': []
        }
        
        for bottleneck in current_bottlenecks:
            bottleneck_type = bottleneck.get('type', '')
            affected_flows = bottleneck.get('affected_flows', [])
            severity = bottleneck.get('severity', 0.0)
            
            # Assess policy's potential to address this bottleneck
            resolution_potential = self._assess_bottleneck_resolution_potential(
                policy, bottleneck_type, affected_flows, severity
            )
            
            if resolution_potential > 0.7:
                resolution_opportunities['resolvable_bottlenecks'].append({
                    'bottleneck': bottleneck,
                    'resolution_potential': resolution_potential,
                    'resolution_mechanism': self._identify_resolution_mechanism(policy, bottleneck_type)
                })
            elif resolution_potential > 0.3:
                resolution_opportunities['partial_resolutions'].append({
                    'bottleneck': bottleneck,
                    'resolution_potential': resolution_potential,
                    'limitations': self._identify_resolution_limitations(policy, bottleneck_type)
                })
            else:
                resolution_opportunities['unaddressed_bottlenecks'].append(bottleneck)
        
        # Identify potential new bottlenecks
        new_bottlenecks = self._identify_potential_new_bottlenecks(policy)
        resolution_opportunities['potential_new_bottlenecks'] = new_bottlenecks
        
        return resolution_opportunities
    
    def _estimate_flow_impact(self, policy: Policy, source_id: uuid.UUID, 
                            target_id: uuid.UUID, current_value: float) -> float:
        """Estimate policy impact on specific delivery flow."""
        # Simplified impact estimation - in practice would be more sophisticated
        impact_factors = []
        
        # Policy type impact
        if policy.policy_type == PolicyType.INCENTIVE_BASED:
            impact_factors.append(0.2)  # Generally positive
        elif policy.policy_type == PolicyType.REGULATORY:
            impact_factors.append(-0.1)  # May add compliance costs
        
        # Resource availability impact
        if policy.resource_requirements:
            total_resources = sum(policy.resource_requirements.values())
            resource_impact = min(total_resources / 100000.0, 0.3)  # Cap at 30% impact
            impact_factors.append(resource_impact)
        
        # Return average impact
        return sum(impact_factors) / len(impact_factors) if impact_factors else 0.0
    
    def _identify_enhancement_mechanism(self, policy: Policy, 
                                      source_id: uuid.UUID, target_id: uuid.UUID) -> str:
        """Identify mechanism by which policy enhances delivery flow."""
        if policy.policy_type == PolicyType.INCENTIVE_BASED:
            return "Performance incentives"
        elif policy.policy_type == PolicyType.CAPACITY_BUILDING:
            return "Capacity enhancement"
        elif policy.policy_type == PolicyType.COORDINATION:
            return "Improved coordination"
        else:
            return "General policy support"
    
    def _identify_diminishment_reason(self, policy: Policy, 
                                    source_id: uuid.UUID, target_id: uuid.UUID) -> str:
        """Identify reason for delivery flow diminishment."""
        if policy.policy_type == PolicyType.REGULATORY:
            return "Regulatory compliance costs"
        else:
            return "Resource reallocation"
    
    def _identify_potential_new_flows(self, policy: Policy, 
                                    current_matrix: DeliveryMatrix) -> List[Dict[str, Any]]:
        """Identify potential new delivery flows created by policy."""
        new_flows = []
        
        # Simplified new flow identification
        if policy.policy_type == PolicyType.COORDINATION:
            # Coordination policies often create new information flows
            new_flows.append({
                'type': 'information_sharing',
                'description': 'New information sharing requirements',
                'estimated_strength': 0.3
            })
        
        return new_flows
    
    def _project_post_policy_metrics(self, policy: Policy, current_matrix: DeliveryMatrix, 
                                   baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Project matrix metrics after policy implementation."""
        projected_metrics = baseline_metrics.copy()
        
        # Apply policy-specific adjustments
        if policy.policy_type == PolicyType.INCENTIVE_BASED:
            # Incentive policies typically improve delivery strength
            current_strength = baseline_metrics.get('average_delivery_strength', 0.5)
            projected_metrics['average_delivery_strength'] = min(current_strength * 1.1, 1.0)
        
        if policy.policy_type == PolicyType.COORDINATION:
            # Coordination policies improve network density
            current_density = baseline_metrics.get('matrix_density', 0.3)
            projected_metrics['matrix_density'] = min(current_density * 1.05, 1.0)
        
        return projected_metrics
    
    def _assess_bottleneck_resolution_potential(self, policy: Policy, bottleneck_type: str, 
                                              affected_flows: List[str], severity: float) -> float:
        """Assess potential for policy to resolve specific bottleneck."""
        resolution_potential = 0.0
        
        # Match policy mechanisms to bottleneck types
        if bottleneck_type == 'capacity_bottleneck':
            if policy.policy_type == PolicyType.CAPACITY_BUILDING:
                resolution_potential = 0.8
            elif policy.policy_type == PolicyType.INCENTIVE_BASED:
                resolution_potential = 0.5
        elif bottleneck_type == 'coordination_bottleneck':
            if policy.policy_type == PolicyType.COORDINATION:
                resolution_potential = 0.9
            elif policy.policy_type == PolicyType.STRUCTURAL_REFORM:
                resolution_potential = 0.6
        elif bottleneck_type == 'resource_bottleneck':
            if policy.resource_requirements:
                total_resources = sum(policy.resource_requirements.values())
                # Higher resource availability = higher resolution potential
                resolution_potential = min(total_resources / 500000.0, 0.8)  # Normalize
        
        # Adjust for severity (higher severity is harder to resolve)
        resolution_potential *= (1.0 - severity * 0.3)
        
        return max(0.0, min(resolution_potential, 1.0))
    
    def _identify_resolution_mechanism(self, policy: Policy, bottleneck_type: str) -> str:
        """Identify mechanism by which policy resolves bottleneck."""
        mechanism_map = {
            'capacity_bottleneck': 'Capacity expansion funding',
            'coordination_bottleneck': 'Coordination structure establishment',
            'resource_bottleneck': 'Resource allocation optimization',
            'process_bottleneck': 'Process improvement requirements'
        }
        return mechanism_map.get(bottleneck_type, 'General policy intervention')
    
    def _identify_resolution_limitations(self, policy: Policy, bottleneck_type: str) -> List[str]:
        """Identify limitations in policy's ability to resolve bottleneck."""
        limitations = []
        
        if policy.implementation_complexity == ImplementationComplexity.HIGH:
            limitations.append('High implementation complexity may delay resolution')
        
        if not policy.resource_requirements and bottleneck_type == 'resource_bottleneck':
            limitations.append('Policy lacks sufficient resource allocation')
        
        return limitations
    
    def _identify_potential_new_bottlenecks(self, policy: Policy) -> List[Dict[str, Any]]:
        """Identify potential new bottlenecks created by policy."""
        new_bottlenecks = []
        
        if policy.policy_type == PolicyType.REGULATORY:
            new_bottlenecks.append({
                'type': 'compliance_bottleneck',
                'description': 'Compliance requirements may create processing delays',
                'estimated_severity': 0.3
            })
        
        return new_bottlenecks


@dataclass
class PolicyComparison(Node):
    """Systematic comparison of policy alternatives using SFM analysis."""
    
    comparison_criteria: List[EvaluationCriteria] = field(default_factory=lambda: [])
    criteria_weights: Dict[EvaluationCriteria, float] = field(default_factory=lambda: {})
    
    # Policy alternatives
    policy_alternatives: List[uuid.UUID] = field(default_factory=lambda: [])
    alternative_assessments: Dict[uuid.UUID, Dict[str, Any]] = field(default_factory=lambda: {})
    
    # Comparison results
    comparison_matrix: Dict[Tuple[uuid.UUID, EvaluationCriteria], float] = field(default_factory=lambda: {})
    overall_scores: Dict[uuid.UUID, float] = field(default_factory=lambda: {})
    ranking: List[Tuple[uuid.UUID, float]] = field(default_factory=lambda: [])
    
    # Sensitivity analysis
    sensitivity_results: Dict[str, Dict[uuid.UUID, float]] = field(default_factory=lambda: {})
    robustness_assessment: Dict[uuid.UUID, float] = field(default_factory=lambda: {})
    
    def conduct_comprehensive_comparison(self, policies: List[Policy], 
                                       delivery_matrices: Dict[uuid.UUID, DeliveryMatrix]) -> Dict[str, Any]:
        """Conduct comprehensive comparison of policy alternatives."""
        comparison_results = {
            'evaluation_overview': {},
            'criteria_analysis': {},
            'alternative_profiles': {},
            'ranking_results': {},
            'sensitivity_analysis': {},
            'recommendations': []
        }
        
        # Initialize comparison
        self.policy_alternatives = [p.id for p in policies]
        
        # Evaluate each policy against each criterion
        for policy in policies:
            policy_assessment = {}
            
            for criterion in self.comparison_criteria:
                score = self._evaluate_policy_against_criterion(policy, criterion, delivery_matrices)
                self.comparison_matrix[(policy.id, criterion)] = score
                policy_assessment[criterion.name] = score
            
            self.alternative_assessments[policy.id] = policy_assessment
        
        # Calculate overall scores
        for policy_id in self.policy_alternatives:
            overall_score = self._calculate_overall_score(policy_id)
            self.overall_scores[policy_id] = overall_score
        
        # Generate ranking
        self.ranking = sorted(self.overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Conduct sensitivity analysis
        sensitivity_results = self._conduct_sensitivity_analysis()
        
        # Compile results
        comparison_results['evaluation_overview'] = {
            'alternatives_evaluated': len(self.policy_alternatives),
            'criteria_used': [c.name for c in self.comparison_criteria],
            'top_alternative': self.ranking[0] if self.ranking else None
        }
        
        comparison_results['ranking_results'] = {
            'full_ranking': [(str(pid), score) for pid, score in self.ranking],
            'top_3': self.ranking[:3] if len(self.ranking) >= 3 else self.ranking
        }
        
        comparison_results['sensitivity_analysis'] = sensitivity_results
        comparison_results['recommendations'] = self._generate_comparison_recommendations()
        
        return comparison_results
    
    def _evaluate_policy_against_criterion(self, policy: Policy, criterion: EvaluationCriteria, 
                                         delivery_matrices: Dict[uuid.UUID, DeliveryMatrix]) -> float:
        """Evaluate a policy against a specific criterion."""
        if criterion == EvaluationCriteria.EFFECTIVENESS:
            return self._assess_effectiveness(policy, delivery_matrices)
        elif criterion == EvaluationCriteria.EFFICIENCY:
            return self._assess_efficiency(policy, delivery_matrices)
        elif criterion == EvaluationCriteria.EQUITY:
            return self._assess_equity(policy, delivery_matrices)
        elif criterion == EvaluationCriteria.SUSTAINABILITY:
            return self._assess_sustainability(policy, delivery_matrices)
        elif criterion == EvaluationCriteria.FEASIBILITY:
            return self._assess_feasibility(policy)
        elif criterion == EvaluationCriteria.ACCEPTABILITY:
            return self._assess_acceptability(policy)
        elif criterion == EvaluationCriteria.COHERENCE:
            return self._assess_coherence(policy)
        elif criterion == EvaluationCriteria.ADAPTABILITY:
            return self._assess_adaptability(policy)
        else:
            return 0.5  # Default neutral score
    
    def _assess_effectiveness(self, policy: Policy, 
                            delivery_matrices: Dict[uuid.UUID, DeliveryMatrix]) -> float:
        """Assess policy effectiveness in achieving objectives."""
        effectiveness_factors = []
        
        # Objective clarity
        if policy.expected_outcomes:
            objective_clarity = min(len(policy.expected_outcomes) / 3.0, 1.0)
            effectiveness_factors.append(objective_clarity * 0.3)
        
        # Mechanism strength
        if policy.policy_instruments:
            mechanism_strength = min(len(policy.policy_instruments) / 2.0, 1.0)
            effectiveness_factors.append(mechanism_strength * 0.3)
        
        # Delivery impact potential
        # Simplified - would use actual delivery impact analysis
        delivery_impact = 0.6  # Placeholder
        effectiveness_factors.append(delivery_impact * 0.4)
        
        return sum(effectiveness_factors) if effectiveness_factors else 0.5
    
    def _assess_efficiency(self, policy: Policy, 
                          delivery_matrices: Dict[uuid.UUID, DeliveryMatrix]) -> float:
        """Assess policy efficiency in resource utilization."""
        if not policy.resource_requirements or not policy.expected_outcomes:
            return 0.5  # Default if insufficient information
        
        # Simple cost-effectiveness calculation
        total_cost = sum(policy.resource_requirements.values())
        outcome_count = len(policy.expected_outcomes)
        
        if total_cost > 0 and outcome_count > 0:
            cost_per_outcome = total_cost / outcome_count
            # Normalize efficiency (lower cost per outcome = higher efficiency)
            efficiency = max(0.0, 1.0 - (cost_per_outcome / 100000.0))  # Normalize to $100k
            return min(efficiency, 1.0)
        
        return 0.5
    
    def _assess_equity(self, policy: Policy, 
                      delivery_matrices: Dict[uuid.UUID, DeliveryMatrix]) -> float:
        """Assess policy equity in benefit/cost distribution."""
        equity_factors = []
        
        # Beneficiary analysis
        if hasattr(policy, 'target_beneficiaries') and policy.target_beneficiaries:
            # More diverse beneficiaries = higher equity
            beneficiary_diversity = min(len(policy.target_beneficiaries) / 5.0, 1.0)
            equity_factors.append(beneficiary_diversity * 0.4)
        
        # Geographic coverage
        if policy.policy_scope in [PolicyScope.SYSTEM_WIDE, PolicyScope.CROSS_SYSTEM]:
            equity_factors.append(0.8 * 0.3)  # Broad coverage
        elif policy.policy_scope == PolicyScope.SECTOR_WIDE:
            equity_factors.append(0.6 * 0.3)  # Medium coverage
        else:
            equity_factors.append(0.3 * 0.3)  # Limited coverage
        
        # Access considerations (simplified)
        access_score = 0.6  # Placeholder
        equity_factors.append(access_score * 0.3)
        
        return sum(equity_factors) if equity_factors else 0.5
    
    def _assess_sustainability(self, policy: Policy, 
                             delivery_matrices: Dict[uuid.UUID, DeliveryMatrix]) -> float:
        """Assess long-term sustainability of policy."""
        sustainability_factors = []
        
        # Financial sustainability
        if policy.resource_requirements:
            # Simple sustainability check - ongoing vs. one-time costs
            ongoing_costs = policy.resource_requirements.get('ongoing_costs', 0)
            total_costs = sum(policy.resource_requirements.values())
            if total_costs > 0:
                sustainability_ratio = 1.0 - (ongoing_costs / total_costs)
                sustainability_factors.append(sustainability_ratio * 0.4)
        
        # Environmental sustainability (simplified)
        env_sustainability = 0.7  # Placeholder
        sustainability_factors.append(env_sustainability * 0.3)
        
        # Institutional sustainability
        if policy.implementation_complexity == ImplementationComplexity.LOW:
            inst_sustainability = 0.8
        elif policy.implementation_complexity == ImplementationComplexity.MEDIUM:
            inst_sustainability = 0.6
        else:
            inst_sustainability = 0.4
        sustainability_factors.append(inst_sustainability * 0.3)
        
        return sum(sustainability_factors) if sustainability_factors else 0.5
    
    def _assess_feasibility(self, policy: Policy) -> float:
        """Assess implementation feasibility of policy."""
        feasibility_factors = []
        
        # Technical feasibility
        if policy.implementation_complexity == ImplementationComplexity.LOW:
            feasibility_factors.append(0.9 * 0.3)
        elif policy.implementation_complexity == ImplementationComplexity.MEDIUM:
            feasibility_factors.append(0.7 * 0.3)
        else:
            feasibility_factors.append(0.4 * 0.3)
        
        # Resource feasibility
        if policy.resource_requirements:
            total_resources = sum(policy.resource_requirements.values())
            # Simplified - assume higher resource requirements = lower feasibility
            resource_feasibility = max(0.0, 1.0 - (total_resources / 1000000.0))  # Normalize to $1M
            feasibility_factors.append(resource_feasibility * 0.4)
        
        # Timeline feasibility
        if hasattr(policy, 'implementation_timeline') and policy.implementation_timeline:
            # Longer timeline = higher feasibility (more time to prepare)
            timeline_days = policy.implementation_timeline.days if hasattr(policy.implementation_timeline, 'days') else 365
            timeline_feasibility = min(timeline_days / 365.0, 1.0)  # Normalize to 1 year
            feasibility_factors.append(timeline_feasibility * 0.3)
        
        return sum(feasibility_factors) if feasibility_factors else 0.5
    
    def _assess_acceptability(self, policy: Policy) -> float:
        """Assess stakeholder acceptability of policy."""
        # Simplified acceptability assessment
        acceptability_factors = []
        
        # Policy type acceptability (some types generally more acceptable)
        type_acceptability = {
            PolicyType.INCENTIVE_BASED: 0.8,
            PolicyType.CAPACITY_BUILDING: 0.9,
            PolicyType.COORDINATION: 0.7,
            PolicyType.REGULATORY: 0.5,
            PolicyType.STRUCTURAL_REFORM: 0.4
        }
        acceptability_factors.append(type_acceptability.get(policy.policy_type, 0.6))
        
        return sum(acceptability_factors) / len(acceptability_factors) if acceptability_factors else 0.6
    
    def _assess_coherence(self, policy: Policy) -> float:
        """Assess internal coherence of policy."""
        coherence_factors = []
        
        # Objective-instrument alignment
        if policy.expected_outcomes and policy.policy_instruments:
            # Simplified alignment check
            alignment_score = min(len(policy.policy_instruments) / len(policy.expected_outcomes), 1.0)
            coherence_factors.append(alignment_score * 0.5)
        
        # Internal consistency (simplified)
        consistency_score = 0.7  # Placeholder
        coherence_factors.append(consistency_score * 0.5)
        
        return sum(coherence_factors) if coherence_factors else 0.5
    
    def _assess_adaptability(self, policy: Policy) -> float:
        """Assess policy adaptability to changing conditions."""
        adaptability_factors = []
        
        # Flexibility mechanisms
        if hasattr(policy, 'flexibility_mechanisms') and policy.flexibility_mechanisms:
            flexibility_score = min(len(policy.flexibility_mechanisms) / 3.0, 1.0)
            adaptability_factors.append(flexibility_score * 0.4)
        
        # Review and adjustment provisions
        if hasattr(policy, 'review_provisions') and policy.review_provisions:
            review_score = 0.8
            adaptability_factors.append(review_score * 0.3)
        
        # Monitoring and feedback systems
        if hasattr(policy, 'monitoring_systems') and policy.monitoring_systems:
            monitoring_score = 0.7
            adaptability_factors.append(monitoring_score * 0.3)
        
        return sum(adaptability_factors) if adaptability_factors else 0.4
    
    def _calculate_overall_score(self, policy_id: uuid.UUID) -> float:
        """Calculate overall weighted score for policy."""
        weighted_scores = []
        
        for criterion in self.comparison_criteria:
            criterion_score = self.comparison_matrix.get((policy_id, criterion), 0.5)
            weight = self.criteria_weights.get(criterion, 1.0 / len(self.comparison_criteria))
            weighted_scores.append(criterion_score * weight)
        
        return sum(weighted_scores) if weighted_scores else 0.5
    
    def _conduct_sensitivity_analysis(self) -> Dict[str, Any]:
        """Conduct sensitivity analysis on comparison results."""
        sensitivity_results = {
            'weight_sensitivity': {},
            'score_sensitivity': {},
            'ranking_stability': {}
        }
        
        # Weight sensitivity analysis
        for criterion in self.comparison_criteria:
            # Test impact of changing this criterion's weight
            original_weight = self.criteria_weights.get(criterion, 1.0 / len(self.comparison_criteria))
            
            # Test +/- 20% weight change
            test_weights = [original_weight * 0.8, original_weight * 1.2]
            
            for test_weight in test_weights:
                # Temporarily modify weight
                temp_weights = self.criteria_weights.copy()
                temp_weights[criterion] = test_weight
                
                # Recalculate scores
                temp_scores = {}
                for policy_id in self.policy_alternatives:
                    weighted_scores = []
                    for crit in self.comparison_criteria:
                        score = self.comparison_matrix.get((policy_id, crit), 0.5)
                        weight = temp_weights.get(crit, 1.0 / len(self.comparison_criteria))
                        weighted_scores.append(score * weight)
                    temp_scores[policy_id] = sum(weighted_scores)
                
                # Store sensitivity result
                weight_key = f"{criterion.name}_{'+20%' if test_weight > original_weight else '-20%'}"
                sensitivity_results['weight_sensitivity'][weight_key] = temp_scores
        
        return sensitivity_results
    
    def _generate_comparison_recommendations(self) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        if self.ranking:
            top_policy = self.ranking[0]
            recommendations.append(f"Recommended primary choice: Policy {top_policy[0]} (score: {top_policy[1]:.2f})")
            
            if len(self.ranking) > 1:
                second_policy = self.ranking[1]
                score_gap = top_policy[1] - second_policy[1]
                
                if score_gap < 0.1:  # Close competition
                    recommendations.append("Top alternatives are closely matched - consider hybrid approach")
                
                recommendations.append(f"Consider as alternative: Policy {second_policy[0]} (score: {second_policy[1]:.2f})")
        
        # Identify weak alternatives
        if self.ranking:
            weak_alternatives = [alt for alt in self.ranking if alt[1] < 0.4]
            if weak_alternatives:
                recommendations.append(f"Consider eliminating {len(weak_alternatives)} low-scoring alternatives")
        
        return recommendations


@dataclass
class PolicyEvaluationFramework(Node):
    """Main orchestrating framework for comprehensive policy evaluation using SFM analysis."""
    
    evaluation_context: Dict[str, Any] = field(default_factory=lambda: {})
    baseline_matrices: Dict[str, uuid.UUID] = field(default_factory=lambda: {})
    
    # Framework components
    impact_assessments: Dict[uuid.UUID, PolicyImpactAssessment] = field(default_factory=lambda: {})
    delivery_analyses: Dict[uuid.UUID, DeliveryImpactAnalysis] = field(default_factory=lambda: {})
    policy_comparison: Optional[PolicyComparison] = None
    
    # Evaluation results
    evaluation_results: Dict[uuid.UUID, Dict[str, Any]] = field(default_factory=lambda: {})
    comparative_analysis: Dict[str, Any] = field(default_factory=lambda: {})
    final_recommendations: List[str] = field(default_factory=lambda: [])
    
    def conduct_comprehensive_evaluation(self, policies: List[Policy], 
                                       baseline_matrices: Dict[str, DeliveryMatrix],
                                       evaluation_criteria: List[EvaluationCriteria]) -> Dict[str, Any]:
        """Conduct comprehensive evaluation of policy alternatives."""
        evaluation_report = {
            'evaluation_overview': {},
            'individual_assessments': {},
            'comparative_analysis': {},
            'implementation_considerations': {},
            'final_recommendations': []
        }
        
        # Initialize evaluation
        evaluation_report['evaluation_overview'] = {
            'policies_evaluated': len(policies),
            'evaluation_criteria': [c.name for c in evaluation_criteria],
            'baseline_matrices': list(baseline_matrices.keys()),
            'evaluation_date': datetime.now()
        }
        
        # Conduct individual policy assessments
        for policy in policies:
            # Impact assessment
            impact_assessment = PolicyImpactAssessment(
                label=f"Impact Assessment - {policy.label}",
                policy_id=policy.id
            )
            
            # Delivery impact analysis
            delivery_analysis = DeliveryImpactAnalysis(
                label=f"Delivery Analysis - {policy.label}",
                policy_id=policy.id,
                baseline_delivery_matrix=list(baseline_matrices.values())[0].id if baseline_matrices else uuid.uuid4()
            )
            
            # Store assessments
            self.impact_assessments[policy.id] = impact_assessment
            self.delivery_analyses[policy.id] = delivery_analysis
            
            # Individual evaluation results
            policy_results = {
                'impact_assessment': impact_assessment.calculate_net_impact_score(),
                'delivery_analysis': delivery_analysis.assess_network_efficiency_impacts(policy, list(baseline_matrices.values())[0] if baseline_matrices else None),
                'feasibility_assessment': self._assess_implementation_feasibility(policy)
            }
            
            self.evaluation_results[policy.id] = policy_results
            evaluation_report['individual_assessments'][str(policy.id)] = policy_results
        
        # Conduct comparative analysis
        if len(policies) > 1:
            comparison = PolicyComparison(
                label="Policy Comparison Analysis",
                comparison_criteria=evaluation_criteria,
                criteria_weights={criterion: 1.0 / len(evaluation_criteria) for criterion in evaluation_criteria}
            )
            
            comparison_results = comparison.conduct_comprehensive_comparison(policies, baseline_matrices)
            self.policy_comparison = comparison
            self.comparative_analysis = comparison_results
            evaluation_report['comparative_analysis'] = comparison_results
        
        # Implementation considerations
        implementation_considerations = self._analyze_implementation_considerations(policies)
        evaluation_report['implementation_considerations'] = implementation_considerations
        
        # Final recommendations
        final_recommendations = self._generate_final_recommendations(policies, evaluation_report)
        self.final_recommendations = final_recommendations
        evaluation_report['final_recommendations'] = final_recommendations
        
        return evaluation_report
    
    def _assess_implementation_feasibility(self, policy: Policy) -> Dict[str, float]:
        """Assess implementation feasibility across multiple dimensions."""
        feasibility_assessment = {}
        
        # Technical feasibility
        if policy.implementation_complexity == ImplementationComplexity.LOW:
            feasibility_assessment['technical_feasibility'] = 0.9
        elif policy.implementation_complexity == ImplementationComplexity.MEDIUM:
            feasibility_assessment['technical_feasibility'] = 0.7
        else:
            feasibility_assessment['technical_feasibility'] = 0.4
        
        # Resource feasibility
        if policy.resource_requirements:
            total_resources = sum(policy.resource_requirements.values())
            # Simplified resource availability check
            resource_feasibility = max(0.0, 1.0 - (total_resources / 2000000.0))  # Normalize to $2M
            feasibility_assessment['resource_feasibility'] = min(resource_feasibility, 1.0)
        else:
            feasibility_assessment['resource_feasibility'] = 0.8  # No resource requirements = high feasibility
        
        # Political feasibility (simplified)
        political_feasibility = 0.6  # Placeholder
        feasibility_assessment['political_feasibility'] = political_feasibility
        
        # Organizational feasibility
        if hasattr(policy, 'institutional_requirements') and policy.institutional_requirements:
            org_complexity = len(policy.institutional_requirements)
            organizational_feasibility = max(0.2, 1.0 - (org_complexity * 0.1))
            feasibility_assessment['organizational_feasibility'] = organizational_feasibility
        else:
            feasibility_assessment['organizational_feasibility'] = 0.7
        
        # Overall feasibility
        feasibility_assessment['overall_feasibility'] = sum(feasibility_assessment.values()) / len(feasibility_assessment)
        
        return feasibility_assessment
    
    def _analyze_implementation_considerations(self, policies: List[Policy]) -> Dict[str, Any]:
        """Analyze cross-cutting implementation considerations."""
        implementation_analysis = {
            'common_challenges': [],
            'resource_conflicts': [],
            'sequencing_considerations': [],
            'coordination_requirements': []
        }
        
        # Identify common implementation challenges
        common_challenges = [
            'Stakeholder coordination complexity',
            'Resource mobilization requirements',
            'Institutional capacity constraints',
            'Monitoring and evaluation needs'
        ]
        implementation_analysis['common_challenges'] = common_challenges
        
        # Resource conflict analysis
        total_resources_by_type = {}
        for policy in policies:
            if policy.resource_requirements:
                for resource_type, amount in policy.resource_requirements.items():
                    total_resources_by_type[resource_type] = total_resources_by_type.get(resource_type, 0) + amount
        
        # Identify potential resource conflicts
        for resource_type, total_required in total_resources_by_type.items():
            if total_required > 1000000:  # Arbitrary threshold
                implementation_analysis['resource_conflicts'].append({
                    'resource_type': resource_type,
                    'total_required': total_required,
                    'conflict_potential': 'high'
                })
        
        return implementation_analysis
    
    def _generate_final_recommendations(self, policies: List[Policy], 
                                      evaluation_report: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on comprehensive evaluation."""
        recommendations = []
        
        # Top policy recommendation
        if self.policy_comparison and self.policy_comparison.ranking:
            top_policy = self.policy_comparison.ranking[0]
            recommendations.append(f"Primary recommendation: Implement Policy {top_policy[0]} (Overall score: {top_policy[1]:.2f})")
        
        # Implementation sequence recommendations
        if len(policies) > 1:
            recommendations.append("Consider phased implementation approach to manage complexity and resource requirements")
        
        # Risk mitigation recommendations
        recommendations.extend([
            "Establish comprehensive monitoring and evaluation system",
            "Develop stakeholder engagement and communication strategy",
            "Create adaptive management mechanisms for policy adjustments"
        ])
        
        # Context-specific recommendations
        high_risk_policies = [p for p in policies if self._assess_implementation_feasibility(p)['overall_feasibility'] < 0.5]
        if high_risk_policies:
            recommendations.append(f"Address feasibility concerns for {len(high_risk_policies)} high-risk alternatives before implementation")
        
        return recommendations