"""
Enhanced delivery system quantification and flow analysis for Social Fabric Matrix.

This module implements comprehensive delivery system modeling, including quantification
methods, flow analysis, dependency mapping, and bottleneck detection. Deliveries are
central to Hayden's SFM framework as they represent how institutions serve each other
and contribute to the overall provisioning process.

Key Components:
- DeliveryQuantification: Methods for measuring delivery effectiveness
- DeliveryFlow: Individual delivery flow with performance metrics
- DeliveryNetwork: Network of interconnected deliveries
- DeliveryAnalyzer: Tools for analyzing delivery systems
- DeliveryBottleneck: Identification and analysis of delivery constraints
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
from models.meta_entities import TimeSlice, SpatialUnit
from models.sfm_enums import (
    DeliveryQuantificationMethod,
    ResourceType,
    FlowType,
    SystemLevel,
)


class DeliveryStatus(Enum):
    """Status of delivery flows."""
    
    PLANNED = auto()        # Delivery is planned but not started
    ACTIVE = auto()         # Delivery is currently active
    COMPLETED = auto()      # Delivery completed successfully
    DELAYED = auto()        # Delivery is behind schedule
    BLOCKED = auto()        # Delivery is blocked by constraints
    FAILED = auto()         # Delivery failed to complete
    SUSPENDED = auto()      # Delivery temporarily suspended


class BottleneckType(Enum):
    """Types of delivery bottlenecks."""
    
    CAPACITY_BOTTLENECK = auto()     # Limited delivery capacity
    RESOURCE_BOTTLENECK = auto()     # Insufficient resources
    PROCESS_BOTTLENECK = auto()      # Process inefficiencies  
    COORDINATION_BOTTLENECK = auto() # Poor coordination
    REGULATORY_BOTTLENECK = auto()   # Regulatory constraints
    TECHNOLOGY_BOTTLENECK = auto()   # Technology limitations
    KNOWLEDGE_BOTTLENECK = auto()    # Lack of expertise/knowledge


class FlowDirection(Enum):
    """Direction of delivery flows."""
    
    UNIDIRECTIONAL = auto()  # One-way flow
    BIDIRECTIONAL = auto()   # Two-way exchange
    MULTIDIRECTIONAL = auto() # Multiple direction flows
    CIRCULAR = auto()        # Circular flow pattern


@dataclass
class DeliveryQuantification(Node):
    """Quantification methods and metrics for institutional deliveries."""
    
    quantification_method: DeliveryQuantificationMethod = DeliveryQuantificationMethod.VOLUME_BASED
    measurement_unit: str = ""  # Unit of measurement
    
    # Quantification parameters
    base_measurement: Optional[float] = None     # Base quantitative measure
    quality_weight: Optional[float] = None       # Quality adjustment factor (0-1)
    impact_multiplier: Optional[float] = None    # Impact scaling factor
    time_normalization: Optional[str] = None     # Time unit for normalization
    
    # Value specifications
    monetary_value: Optional[float] = None       # Monetary value if applicable
    social_value: Optional[float] = None         # Social value assessment
    environmental_value: Optional[float] = None  # Environmental value
    
    # Measurement context
    measurement_context: Dict[str, Any] = field(default_factory=dict)
    measurement_assumptions: List[str] = field(default_factory=list)
    measurement_limitations: List[str] = field(default_factory=list)
    
    # Validation and quality
    measurement_reliability: Optional[float] = None  # Reliability score (0-1)
    measurement_validity: Optional[float] = None     # Validity assessment (0-1)
    confidence_intervals: Optional[Tuple[float, float]] = None
    
    def calculate_weighted_value(self) -> Optional[float]:
        """Calculate quality-weighted delivery value."""
        if self.base_measurement is None:
            return None
        
        weighted_value = self.base_measurement
        
        # Apply quality weighting
        if self.quality_weight is not None:
            weighted_value *= self.quality_weight
        
        # Apply impact multiplier
        if self.impact_multiplier is not None:
            weighted_value *= self.impact_multiplier
        
        return weighted_value
    
    def calculate_total_value(self) -> Dict[str, Optional[float]]:
        """Calculate total value across all dimensions."""
        total_values = {
            'economic_value': self.monetary_value,
            'social_value': self.social_value,
            'environmental_value': self.environmental_value,
            'weighted_quantitative': self.calculate_weighted_value()
        }
        
        # Calculate composite value if multiple dimensions available
        available_values = [v for v in total_values.values() if v is not None]
        if len(available_values) > 1:
            total_values['composite_value'] = sum(available_values) / len(available_values)
        
        return total_values
    
    def assess_measurement_quality(self) -> Dict[str, float]:
        """Assess quality of the quantification."""
        quality_metrics = {}
        
        if self.measurement_reliability is not None:
            quality_metrics['reliability'] = self.measurement_reliability
        
        if self.measurement_validity is not None:
            quality_metrics['validity'] = self.measurement_validity
        
        # Completeness based on available measures
        completeness_factors = []
        if self.base_measurement is not None:
            completeness_factors.append(1.0)
        if self.quality_weight is not None:
            completeness_factors.append(1.0)
        if self.impact_multiplier is not None:
            completeness_factors.append(1.0)
        
        if completeness_factors:
            quality_metrics['completeness'] = sum(completeness_factors) / 3.0  # Max 3 factors
        
        # Context adequacy
        if self.measurement_context:
            context_score = min(len(self.measurement_context) / 3.0, 1.0)
            quality_metrics['context_adequacy'] = context_score
        
        # Overall quality
        if quality_metrics:
            quality_metrics['overall'] = sum(quality_metrics.values()) / len(quality_metrics)
        
        return quality_metrics


@dataclass  
class DeliveryFlow(Node):
    """Individual delivery flow between institutions with performance metrics."""
    
    source_institution_id: uuid.UUID
    target_institution_id: uuid.UUID
    delivery_type: str = ""  # Type of delivery (service, resource, information, etc.)
    
    # Flow characteristics
    flow_direction: FlowDirection = FlowDirection.UNIDIRECTIONAL
    flow_type: FlowType = FlowType.MATERIAL
    resource_type: Optional[ResourceType] = None
    
    # Quantification
    quantification: Optional[DeliveryQuantification] = None
    current_flow_rate: Optional[float] = None      # Current delivery rate
    maximum_capacity: Optional[float] = None       # Maximum possible delivery rate
    minimum_required: Optional[float] = None       # Minimum required delivery level
    
    # Performance metrics
    delivery_status: DeliveryStatus = DeliveryStatus.PLANNED
    reliability_score: Optional[float] = None      # Reliability of delivery (0-1)
    timeliness_score: Optional[float] = None       # On-time delivery rate (0-1)
    quality_score: Optional[float] = None          # Quality of deliveries (0-1)
    cost_efficiency: Optional[float] = None        # Cost per unit delivered
    
    # Temporal properties
    delivery_frequency: Optional[timedelta] = None # How often delivery occurs
    delivery_duration: Optional[timedelta] = None  # How long each delivery takes
    last_delivery: Optional[datetime] = None       # When last delivery occurred
    next_scheduled: Optional[datetime] = None      # Next scheduled delivery
    
    # Dependencies and constraints
    prerequisite_deliveries: List[uuid.UUID] = field(default_factory=list)  # Required prior deliveries
    dependent_deliveries: List[uuid.UUID] = field(default_factory=list)     # Deliveries that depend on this
    delivery_constraints: List[str] = field(default_factory=list)           # Constraints affecting delivery
    
    # Historical performance
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_utilization_rate(self) -> Optional[float]:
        """Calculate capacity utilization rate."""
        if self.current_flow_rate is None or self.maximum_capacity is None:
            return None
        
        if self.maximum_capacity == 0:
            return 0.0
        
        return min(self.current_flow_rate / self.maximum_capacity, 1.0)
    
    def calculate_performance_score(self) -> float:
        """Calculate overall delivery performance score."""
        performance_factors = []
        
        if self.reliability_score is not None:
            performance_factors.append(self.reliability_score * 0.3)
        
        if self.timeliness_score is not None:
            performance_factors.append(self.timeliness_score * 0.3)
        
        if self.quality_score is not None:
            performance_factors.append(self.quality_score * 0.25)
        
        # Utilization efficiency (not too low, not maxed out)
        utilization = self.calculate_utilization_rate()
        if utilization is not None:
            # Optimal utilization around 80%
            utilization_efficiency = 1.0 - abs(utilization - 0.8) / 0.8
            performance_factors.append(max(0.0, utilization_efficiency) * 0.15)
        
        return sum(performance_factors) if performance_factors else 0.0
    
    def identify_performance_issues(self) -> List[Dict[str, Any]]:
        """Identify potential performance issues with this delivery."""
        issues = []
        
        # Low reliability
        if self.reliability_score is not None and self.reliability_score < 0.6:
            issues.append({
                'type': 'low_reliability',
                'description': f'Reliability score {self.reliability_score:.2f} below acceptable threshold',
                'severity': 'high' if self.reliability_score < 0.4 else 'medium',
                'recommendation': 'Investigate and address causes of delivery failures'
            })
        
        # Poor timeliness
        if self.timeliness_score is not None and self.timeliness_score < 0.7:
            issues.append({
                'type': 'poor_timeliness',
                'description': f'On-time delivery rate {self.timeliness_score:.2f} needs improvement',
                'severity': 'medium',
                'recommendation': 'Analyze scheduling and process bottlenecks'
            })
        
        # Quality issues
        if self.quality_score is not None and self.quality_score < 0.6:
            issues.append({
                'type': 'quality_issues',
                'description': f'Quality score {self.quality_score:.2f} indicates delivery quality problems',
                'severity': 'high',
                'recommendation': 'Implement quality improvement measures'
            })
        
        # Capacity issues
        utilization = self.calculate_utilization_rate()
        if utilization is not None:
            if utilization > 0.95:
                issues.append({
                    'type': 'capacity_constraint',
                    'description': f'Utilization rate {utilization:.2f} indicates capacity bottleneck',
                    'severity': 'high',
                    'recommendation': 'Consider capacity expansion or demand management'
                })
            elif utilization < 0.3:
                issues.append({
                    'type': 'underutilization',
                    'description': f'Low utilization rate {utilization:.2f} indicates inefficient resource use',
                    'severity': 'medium',
                    'recommendation': 'Investigate causes of low demand or capacity optimization'
                })
        
        # Overdue deliveries
        if (self.next_scheduled and datetime.now() > self.next_scheduled and 
            self.delivery_status not in [DeliveryStatus.COMPLETED, DeliveryStatus.SUSPENDED]):
            issues.append({
                'type': 'overdue_delivery',
                'description': 'Delivery is past scheduled time',
                'severity': 'high',
                'recommendation': 'Address immediate causes of delay'
            })
        
        return issues
    
    def predict_delivery_risk(self) -> Dict[str, float]:
        """Predict risk of delivery problems."""
        risk_factors = {}
        
        # Historical failure rate
        if self.failure_history:
            total_attempts = len(self.performance_history) + len(self.failure_history)
            failure_rate = len(self.failure_history) / total_attempts if total_attempts > 0 else 0
            risk_factors['historical_failure_risk'] = failure_rate
        
        # Constraint-based risk
        if self.delivery_constraints:
            constraint_risk = min(len(self.delivery_constraints) * 0.2, 1.0)
            risk_factors['constraint_risk'] = constraint_risk
        
        # Dependency risk
        if self.prerequisite_deliveries:
            dependency_risk = min(len(self.prerequisite_deliveries) * 0.15, 1.0)
            risk_factors['dependency_risk'] = dependency_risk
        
        # Capacity stress risk
        utilization = self.calculate_utilization_rate()
        if utilization is not None and utilization > 0.85:
            capacity_stress = (utilization - 0.85) / 0.15  # Scale from 0.85-1.0 to 0-1
            risk_factors['capacity_stress_risk'] = capacity_stress
        
        # Overall risk
        if risk_factors:
            overall_risk = sum(risk_factors.values()) / len(risk_factors)
            risk_factors['overall_risk'] = overall_risk
        
        return risk_factors


@dataclass
class DeliveryBottleneck(Node):
    """Identification and analysis of delivery system bottlenecks."""
    
    bottleneck_type: BottleneckType = BottleneckType.CAPACITY_BOTTLENECK
    affected_deliveries: List[uuid.UUID] = field(default_factory=list)
    
    # Bottleneck characteristics
    severity_level: Optional[float] = None         # Severity of bottleneck (0-1)
    impact_scope: Optional[str] = None             # Local, system-wide, etc.
    duration: Optional[timedelta] = None           # How long bottleneck has existed
    
    # Bottleneck metrics
    throughput_reduction: Optional[float] = None   # Reduction in delivery throughput
    delay_caused: Optional[timedelta] = None       # Average delay caused
    cost_impact: Optional[float] = None            # Additional costs due to bottleneck
    
    # Root causes
    root_causes: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    
    # Resolution approaches
    potential_solutions: List[str] = field(default_factory=list)
    implementation_difficulty: Optional[float] = None  # Difficulty to resolve (0-1)
    resolution_priority: Optional[float] = None        # Priority for resolution (0-1)
    
    def calculate_bottleneck_impact(self) -> Dict[str, float]:
        """Calculate comprehensive impact of the bottleneck."""
        impact_metrics = {}
        
        # Direct throughput impact
        if self.throughput_reduction is not None:
            impact_metrics['throughput_impact'] = self.throughput_reduction
        
        # Time impact
        if self.delay_caused is not None:
            # Convert to days for normalization
            delay_days = self.delay_caused.total_seconds() / (24 * 3600)
            time_impact = min(delay_days / 30.0, 1.0)  # Normalize to 30 days max
            impact_metrics['time_impact'] = time_impact
        
        # Cost impact
        if self.cost_impact is not None:
            # Normalize cost impact (simplified)
            cost_impact_normalized = min(self.cost_impact / 10000.0, 1.0)  # $10k as reference
            impact_metrics['cost_impact'] = cost_impact_normalized
        
        # Scope impact
        scope_impact = len(self.affected_deliveries) / 10.0  # Normalize to 10 deliveries
        impact_metrics['scope_impact'] = min(scope_impact, 1.0)
        
        # Overall impact
        if impact_metrics:
            overall_impact = sum(impact_metrics.values()) / len(impact_metrics)
            impact_metrics['overall_impact'] = overall_impact
        
        return impact_metrics
    
    def assess_resolution_feasibility(self) -> Dict[str, float]:
        """Assess feasibility of resolving this bottleneck."""
        feasibility_factors = {}
        
        # Solution availability
        if self.potential_solutions:
            solution_score = min(len(self.potential_solutions) / 3.0, 1.0)
            feasibility_factors['solution_availability'] = solution_score
        else:
            feasibility_factors['solution_availability'] = 0.0
        
        # Implementation difficulty (inverted)
        if self.implementation_difficulty is not None:
            feasibility_factors['implementation_ease'] = 1.0 - self.implementation_difficulty
        
        # Priority alignment
        if self.resolution_priority is not None:
            feasibility_factors['priority_support'] = self.resolution_priority
        
        # Impact justification
        impact_metrics = self.calculate_bottleneck_impact()
        overall_impact = impact_metrics.get('overall_impact', 0.0)
        feasibility_factors['impact_justification'] = overall_impact
        
        # Overall feasibility
        if feasibility_factors:
            overall_feasibility = sum(feasibility_factors.values()) / len(feasibility_factors)
            feasibility_factors['overall_feasibility'] = overall_feasibility
        
        return feasibility_factors
    
    def generate_resolution_plan(self) -> Dict[str, Any]:
        """Generate plan for resolving the bottleneck."""
        resolution_plan = {
            'bottleneck_summary': {
                'type': self.bottleneck_type.name,
                'severity': self.severity_level,
                'affected_deliveries': len(self.affected_deliveries)
            },
            'impact_analysis': self.calculate_bottleneck_impact(),
            'feasibility_assessment': self.assess_resolution_feasibility(),
            'recommended_solutions': [],
            'implementation_steps': [],
            'success_metrics': [],
            'timeline_estimate': None
        }
        
        # Rank solutions by feasibility and impact
        if self.potential_solutions:
            # Simplified ranking - in practice would be more sophisticated
            resolution_plan['recommended_solutions'] = self.potential_solutions[:3]  # Top 3
        
        # Generate implementation steps based on bottleneck type
        if self.bottleneck_type == BottleneckType.CAPACITY_BOTTLENECK:
            resolution_plan['implementation_steps'] = [
                '1. Conduct detailed capacity analysis',
                '2. Identify capacity expansion options',
                '3. Evaluate cost-benefit of expansion',
                '4. Implement capacity improvements',
                '5. Monitor performance improvement'
            ]
        elif self.bottleneck_type == BottleneckType.PROCESS_BOTTLENECK:
            resolution_plan['implementation_steps'] = [
                '1. Map current process flows',
                '2. Identify process inefficiencies',
                '3. Design process improvements',
                '4. Implement process changes',
                '5. Measure performance gains'
            ]
        # Add more bottleneck-specific plans as needed
        
        # Success metrics
        resolution_plan['success_metrics'] = [
            'Throughput improvement percentage',
            'Delivery time reduction',
            'Cost per delivery reduction',
            'Customer satisfaction improvement'
        ]
        
        # Timeline estimate based on implementation difficulty
        if self.implementation_difficulty is not None:
            base_timeline = 30  # 30 days base
            timeline_days = base_timeline * (1 + self.implementation_difficulty)
            resolution_plan['timeline_estimate'] = f"{timeline_days:.0f} days"
        
        return resolution_plan


@dataclass
class DeliveryNetwork(Node):
    """Network of interconnected institutional deliveries."""
    
    delivery_flows: Dict[uuid.UUID, DeliveryFlow] = field(default_factory=dict)
    network_boundaries: List[uuid.UUID] = field(default_factory=list)  # System boundaries
    
    # Network properties
    network_density: Optional[float] = None        # Density of delivery connections
    network_efficiency: Optional[float] = None     # Overall network efficiency
    network_resilience: Optional[float] = None     # Ability to handle disruptions
    
    # Performance metrics
    total_throughput: Optional[float] = None       # Total network throughput
    average_delivery_time: Optional[timedelta] = None
    network_utilization: Optional[float] = None   # Overall capacity utilization
    
    # Bottlenecks and issues
    identified_bottlenecks: List[uuid.UUID] = field(default_factory=list)
    critical_paths: List[List[uuid.UUID]] = field(default_factory=list)  # Critical delivery paths
    
    def add_delivery_flow(self, delivery_flow: DeliveryFlow) -> None:
        """Add a delivery flow to the network."""
        self.delivery_flows[delivery_flow.id] = delivery_flow
    
    def remove_delivery_flow(self, flow_id: uuid.UUID) -> None:
        """Remove a delivery flow from the network."""
        if flow_id in self.delivery_flows:
            del self.delivery_flows[flow_id]
    
    def calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive network performance metrics."""
        if not self.delivery_flows:
            return {}
        
        metrics = {}
        flows = list(self.delivery_flows.values())
        
        # Performance metrics
        performance_scores = [flow.calculate_performance_score() for flow in flows]
        metrics['average_performance'] = sum(performance_scores) / len(performance_scores)
        
        # Utilization metrics
        utilizations = [flow.calculate_utilization_rate() for flow in flows 
                       if flow.calculate_utilization_rate() is not None]
        if utilizations:
            metrics['average_utilization'] = sum(utilizations) / len(utilizations)
            self.network_utilization = metrics['average_utilization']
        
        # Reliability metrics
        reliability_scores = [flow.reliability_score for flow in flows 
                            if flow.reliability_score is not None]
        if reliability_scores:
            metrics['average_reliability'] = sum(reliability_scores) / len(reliability_scores)
        
        # Network density (simplified)
        total_possible_connections = len(flows) * (len(flows) - 1)
        if total_possible_connections > 0:
            # Count actual dependencies
            total_dependencies = sum(len(flow.prerequisite_deliveries) + len(flow.dependent_deliveries) 
                                   for flow in flows)
            metrics['network_density'] = min(total_dependencies / total_possible_connections, 1.0)
            self.network_density = metrics['network_density']
        
        return metrics
    
    def identify_bottlenecks(self) -> List[DeliveryBottleneck]:
        """Identify bottlenecks in the delivery network."""
        bottlenecks = []
        
        for flow in self.delivery_flows.values():
            # Check for capacity bottlenecks
            utilization = flow.calculate_utilization_rate()
            if utilization is not None and utilization > 0.9:
                bottleneck = DeliveryBottleneck(
                    label=f"Capacity bottleneck in {flow.label}",
                    bottleneck_type=BottleneckType.CAPACITY_BOTTLENECK,
                    affected_deliveries=[flow.id],
                    severity_level=min((utilization - 0.9) * 10, 1.0),
                    throughput_reduction=utilization - 0.8  # Assume optimal is 80%
                )
                bottlenecks.append(bottleneck)
            
            # Check for performance bottlenecks
            performance = flow.calculate_performance_score()
            if performance < 0.4:
                bottleneck = DeliveryBottleneck(
                    label=f"Performance bottleneck in {flow.label}",
                    bottleneck_type=BottleneckType.PROCESS_BOTTLENECK,
                    affected_deliveries=[flow.id],
                    severity_level=1.0 - performance,
                    root_causes=["Poor reliability", "Quality issues", "Timeliness problems"]
                )
                bottlenecks.append(bottleneck)
        
        # Store identified bottlenecks
        self.identified_bottlenecks = [b.id for b in bottlenecks]
        
        return bottlenecks
    
    def find_critical_paths(self) -> List[List[uuid.UUID]]:
        """Find critical paths through the delivery network."""
        critical_paths = []
        
        # Simplified critical path analysis
        # Find flows with many dependencies
        for flow in self.delivery_flows.values():
            if len(flow.prerequisite_deliveries) > 2:  # Flows with multiple dependencies
                # Trace back through dependencies
                path = self._trace_dependency_path(flow.id, set())
                if len(path) > 3:  # Paths with multiple steps
                    critical_paths.append(path)
        
        self.critical_paths = critical_paths
        return critical_paths
    
    def _trace_dependency_path(self, flow_id: uuid.UUID, visited: Set[uuid.UUID]) -> List[uuid.UUID]:
        """Trace dependency path for a flow."""
        if flow_id in visited or flow_id not in self.delivery_flows:
            return []
        
        visited.add(flow_id)
        flow = self.delivery_flows[flow_id]
        path = [flow_id]
        
        # Trace through prerequisites
        for prereq_id in flow.prerequisite_deliveries:
            prereq_path = self._trace_dependency_path(prereq_id, visited.copy())
            if prereq_path:
                path = prereq_path + path
                break  # Take first path found
        
        return path
    
    def analyze_network_resilience(self) -> Dict[str, Any]:
        """Analyze network resilience to disruptions."""
        resilience_analysis = {
            'single_point_failures': [],
            'cascade_risks': [],
            'redundancy_gaps': [],
            'resilience_score': 0.0
        }
        
        # Identify single points of failure
        for flow in self.delivery_flows.values():
            dependents = [f for f in self.delivery_flows.values() 
                         if flow.id in f.prerequisite_deliveries]
            if len(dependents) > 3:  # Many flows depend on this one
                resilience_analysis['single_point_failures'].append({
                    'flow_id': flow.id,
                    'flow_label': flow.label,
                    'dependent_count': len(dependents)
                })
        
        # Identify cascade risks
        for path in self.critical_paths:
            if len(path) > 5:  # Long dependency chains
                resilience_analysis['cascade_risks'].append({
                    'path': path,
                    'length': len(path),
                    'risk_level': 'high' if len(path) > 7 else 'medium'
                })
        
        # Calculate resilience score
        resilience_factors = []
        
        # Factor 1: Redundancy
        total_flows = len(self.delivery_flows)
        single_points = len(resilience_analysis['single_point_failures'])
        redundancy_score = max(0.0, 1.0 - (single_points / max(total_flows, 1)))
        resilience_factors.append(redundancy_score * 0.4)
        
        # Factor 2: Path diversity
        cascade_risks = len(resilience_analysis['cascade_risks'])
        path_diversity_score = max(0.0, 1.0 - (cascade_risks / max(len(self.critical_paths), 1)))
        resilience_factors.append(path_diversity_score * 0.3)
        
        # Factor 3: Performance buffer
        network_metrics = self.calculate_network_metrics()
        avg_utilization = network_metrics.get('average_utilization', 1.0)
        buffer_score = max(0.0, 1.0 - avg_utilization)  # Lower utilization = more buffer
        resilience_factors.append(buffer_score * 0.3)
        
        resilience_score = sum(resilience_factors)
        resilience_analysis['resilience_score'] = resilience_score
        self.network_resilience = resilience_score
        
        return resilience_analysis


@dataclass
class DeliveryAnalyzer(Node):
    """Comprehensive analyzer for delivery systems."""
    
    analyzed_networks: List[uuid.UUID] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    
    def analyze_delivery_system(self, network: DeliveryNetwork) -> Dict[str, Any]:
        """Perform comprehensive analysis of a delivery system."""
        analysis = {
            'network_overview': {
                'total_flows': len(network.delivery_flows),
                'network_type': 'delivery_network',
                'analysis_timestamp': datetime.now()
            },
            'performance_analysis': {},
            'bottleneck_analysis': {},
            'resilience_analysis': {},
            'recommendations': []
        }
        
        # Performance analysis
        network_metrics = network.calculate_network_metrics()
        analysis['performance_analysis'] = network_metrics
        
        # Bottleneck analysis
        bottlenecks = network.identify_bottlenecks()
        analysis['bottleneck_analysis'] = {
            'total_bottlenecks': len(bottlenecks),
            'bottleneck_types': {},
            'high_severity_bottlenecks': 0,
            'bottleneck_details': []
        }
        
        for bottleneck in bottlenecks:
            # Count by type
            btype = bottleneck.bottleneck_type.name
            analysis['bottleneck_analysis']['bottleneck_types'][btype] = \
                analysis['bottleneck_analysis']['bottleneck_types'].get(btype, 0) + 1
            
            # Count high severity
            if bottleneck.severity_level and bottleneck.severity_level > 0.7:
                analysis['bottleneck_analysis']['high_severity_bottlenecks'] += 1
            
            # Add details
            analysis['bottleneck_analysis']['bottleneck_details'].append({
                'id': str(bottleneck.id),
                'type': btype,
                'severity': bottleneck.severity_level,
                'affected_flows': len(bottleneck.affected_deliveries)
            })
        
        # Resilience analysis
        resilience_analysis = network.analyze_network_resilience()
        analysis['resilience_analysis'] = resilience_analysis
        
        # Generate recommendations
        recommendations = self._generate_system_recommendations(analysis, network_metrics, bottlenecks)
        analysis['recommendations'] = recommendations
        
        # Store results
        self.analysis_results[str(network.id)] = analysis
        if network.id not in self.analyzed_networks:
            self.analyzed_networks.append(network.id)
        
        return analysis
    
    def _generate_system_recommendations(self, analysis: Dict[str, Any], 
                                       network_metrics: Dict[str, float],
                                       bottlenecks: List[DeliveryBottleneck]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Performance recommendations
        avg_performance = network_metrics.get('average_performance', 1.0)
        if avg_performance < 0.6:
            recommendations.append("Improve overall delivery performance through process optimization")
        
        avg_utilization = network_metrics.get('average_utilization', 0.0)
        if avg_utilization > 0.85:
            recommendations.append("Address capacity constraints - system is operating near maximum capacity")
        elif avg_utilization < 0.4:
            recommendations.append("Investigate low utilization - system may be oversized or underused")
        
        # Bottleneck recommendations
        high_severity_bottlenecks = analysis['bottleneck_analysis']['high_severity_bottlenecks']
        if high_severity_bottlenecks > 0:
            recommendations.append(f"Priority focus on resolving {high_severity_bottlenecks} high-severity bottlenecks")
        
        # Resilience recommendations
        resilience_score = analysis['resilience_analysis']['resilience_score']
        if resilience_score < 0.5:
            recommendations.append("Strengthen system resilience by reducing single points of failure")
        
        single_points = len(analysis['resilience_analysis']['single_point_failures'])
        if single_points > 3:
            recommendations.append("Build redundancy to address single points of failure")
        
        cascade_risks = len(analysis['resilience_analysis']['cascade_risks'])
        if cascade_risks > 2:
            recommendations.append("Reduce cascade risks by shortening dependency chains")
        
        return recommendations
    
    def compare_delivery_systems(self, network_ids: List[uuid.UUID]) -> Dict[str, Any]:
        """Compare multiple delivery systems."""
        if len(network_ids) < 2:
            return {'error': 'Need at least 2 networks to compare'}
        
        comparison = {
            'networks_compared': len(network_ids),
            'performance_comparison': {},
            'bottleneck_comparison': {},
            'resilience_comparison': {},
            'best_practices': [],
            'improvement_opportunities': []
        }
        
        # Gather metrics for comparison
        network_data = {}
        for network_id in network_ids:
            if str(network_id) in self.analysis_results:
                network_data[str(network_id)] = self.analysis_results[str(network_id)]
        
        if len(network_data) < 2:
            return {'error': 'Insufficient analysis data for comparison'}
        
        # Performance comparison
        performance_metrics = {}
        for network_id, analysis in network_data.items():
            perf = analysis.get('performance_analysis', {})
            performance_metrics[network_id] = {
                'average_performance': perf.get('average_performance', 0.0),
                'average_utilization': perf.get('average_utilization', 0.0),
                'network_density': perf.get('network_density', 0.0)
            }
        
        comparison['performance_comparison'] = performance_metrics
        
        # Find best performers
        best_performance = max(performance_metrics.values(), 
                             key=lambda x: x['average_performance'])
        best_network = [k for k, v in performance_metrics.items() 
                       if v['average_performance'] == best_performance['average_performance']][0]
        
        comparison['best_practices'].append(f"Network {best_network} shows best overall performance")
        
        return comparison