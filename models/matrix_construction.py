"""
Matrix Construction Methodology for Social Fabric Matrix analysis.

This module implements Hayden's systematic approach to constructing Social Fabric
Matrices, including matrix cell modeling, systematic matrix building processes,
and validation methodologies. The framework provides the core computational
structure for SFM analysis.

Key Components:
- MatrixCell: Individual institution-criteria relationship modeling
- SFMMatrixBuilder: Systematic matrix construction methodology
- MatrixValidation: Validation and quality assurance processes
- DeliveryMatrix: Specialized matrix for delivery relationship analysis
- MatrixAnalyzer: Comprehensive matrix analysis tools
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
from models.specialized_nodes import MatrixCell
from models.meta_entities import TimeSlice, SpatialUnit
from models.delivery_systems import DeliveryQuantification, DeliveryFlow
from models.sfm_enums import (
    MatrixConstructionStage,
    DeliveryQuantificationMethod,
    InstitutionalScope,
    SystemLevel,
    AnalyticalMethod,
)


class DeliveryDirection(Enum):
    """Direction of delivery relationships in matrix cells."""
    
    UNIDIRECTIONAL = auto()      # One-way delivery
    BIDIRECTIONAL = auto()       # Two-way exchange
    MULTIDIRECTIONAL = auto()    # Multiple direction flows
    CIRCULAR = auto()            # Circular delivery pattern
    NO_DELIVERY = auto()         # No delivery relationship


class CellEvidence(Enum):
    """Types of evidence supporting matrix cell values."""
    
    QUANTITATIVE_DATA = auto()   # Statistical, measured data
    QUALITATIVE_ASSESSMENT = auto()  # Expert judgment, observation
    STAKEHOLDER_REPORT = auto()  # Stakeholder testimony
    HISTORICAL_ANALYSIS = auto() # Historical patterns
    COMPARATIVE_STUDY = auto()   # Cross-case comparison
    THEORETICAL_INFERENCE = auto()  # Logic-based inference


class MatrixDimension(Enum):
    """Dimensions of the Social Fabric Matrix."""
    
    INSTITUTION_CRITERIA = auto()    # Standard institution-criteria matrix
    INSTITUTION_INSTITUTION = auto()  # Institution interaction matrix
    CRITERIA_CRITERIA = auto()       # Criteria relationship matrix
    TEMPORAL_SEQUENCE = auto()       # Time-based relationship matrix
    DELIVERY_FLOW = auto()           # Delivery relationship matrix


class ValidationStatus(Enum):
    """Validation status for matrix components."""
    
    NOT_VALIDATED = auto()       # No validation performed
    PRELIMINARY = auto()         # Initial validation
    PEER_REVIEWED = auto()       # Expert peer review completed
    STAKEHOLDER_VALIDATED = auto()  # Stakeholder validation completed
    EMPIRICALLY_TESTED = auto()  # Empirical testing completed
    FULLY_VALIDATED = auto()     # All validation completed


    


class DeliveryMatrix(Node):
    """Specialized matrix for modeling delivery relationships between institutions."""
    
    matrix_dimension: MatrixDimension = MatrixDimension.DELIVERY_FLOW
    construction_date: datetime = field(default_factory=datetime.now)
    
    # Matrix structure
    row_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Delivering institutions
    column_institutions: List[uuid.UUID] = field(default_factory=lambda: [])  # Receiving institutions
    matrix_cells: Dict[Tuple[uuid.UUID, uuid.UUID], MatrixCell] = field(default_factory=lambda: {})
    
    # Matrix properties
    matrix_density: Optional[float] = None        # Proportion of non-zero cells
    delivery_symmetry: Optional[float] = None     # Degree of symmetric deliveries
    dominant_flows: List[Tuple[uuid.UUID, uuid.UUID, float]] = field(default_factory=lambda: [])
    
    # Aggregated metrics
    total_delivery_volume: Optional[float] = None
    average_delivery_strength: Optional[float] = None
    delivery_concentration: Optional[float] = None  # How concentrated are deliveries
    
    def calculate_matrix_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive metrics for the delivery matrix."""
        metrics = {}
        
        if not self.matrix_cells:
            return metrics
        
        # Matrix density
        total_possible_cells = len(self.row_institutions) * len(self.column_institutions)
        non_zero_cells = sum(1 for cell in self.matrix_cells.values() 
                           if cell.delivery_capacity is not None and cell.delivery_capacity > 0)
        if total_possible_cells > 0:
            metrics['matrix_density'] = non_zero_cells / total_possible_cells
            self.matrix_density = metrics['matrix_density']
        
        # Average delivery strength
        delivery_values = [cell.delivery_capacity for cell in self.matrix_cells.values() 
                          if cell.delivery_capacity is not None]
        if delivery_values:
            metrics['average_delivery_strength'] = sum(delivery_values) / len(delivery_values)
            self.average_delivery_strength = metrics['average_delivery_strength']
            
            # Total delivery volume
            metrics['total_delivery_volume'] = sum(delivery_values)
            self.total_delivery_volume = metrics['total_delivery_volume']
            
            # Delivery concentration (using Gini coefficient approximation)
            sorted_values = sorted(delivery_values)
            n = len(sorted_values)
            cumulative_sum = sum((i + 1) * value for i, value in enumerate(sorted_values))
            total_sum = sum(sorted_values)
            if total_sum > 0:
                gini = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
                metrics['delivery_concentration'] = gini
                self.delivery_concentration = gini
        
        # Delivery symmetry
        symmetric_pairs = 0
        total_pairs = 0
        for (row_id, col_id), cell in self.matrix_cells.items():
            reverse_key = (col_id, row_id)
            if reverse_key in self.matrix_cells:
                total_pairs += 1
                cell_value = cell.delivery_capacity or 0
                reverse_value = self.matrix_cells[reverse_key].delivery_capacity or 0
                # Calculate symmetry for this pair
                if cell_value + reverse_value > 0:
                    symmetry = 1 - abs(cell_value - reverse_value) / (cell_value + reverse_value)
                    symmetric_pairs += symmetry
        
        if total_pairs > 0:
            metrics['delivery_symmetry'] = symmetric_pairs / total_pairs
            self.delivery_symmetry = metrics['delivery_symmetry']
        
        return metrics
    
    def identify_dominant_flows(self, threshold: float = 0.1) -> List[Tuple[uuid.UUID, uuid.UUID, float]]:
        """Identify dominant delivery flows above threshold."""
        flows = []
        
        for (row_id, col_id), cell in self.matrix_cells.items():
            if cell.delivery_capacity is not None and cell.delivery_capacity >= threshold:
                flows.append((row_id, col_id, cell.delivery_capacity))
        
        # Sort by delivery value (descending)
        flows.sort(key=lambda x: x[2], reverse=True)
        self.dominant_flows = flows
        return flows
    
    def analyze_delivery_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in delivery relationships."""
        patterns = {
            'hub_institutions': [],      # Institutions with many outgoing deliveries
            'sink_institutions': [],     # Institutions with many incoming deliveries
            'isolated_institutions': [], # Institutions with few connections
            'reciprocal_pairs': [],      # Institution pairs with bidirectional deliveries
            'delivery_clusters': []      # Groups of highly connected institutions
        }
        
        # Calculate in-degree and out-degree for each institution
        out_degrees = {inst_id: 0 for inst_id in self.row_institutions}
        in_degrees = {inst_id: 0 for inst_id in self.column_institutions}
        
        for (row_id, col_id), cell in self.matrix_cells.items():
            if cell.delivery_capacity is not None and cell.delivery_capacity > 0:
                out_degrees[row_id] += 1
                in_degrees[col_id] += 1
        
        # Identify hubs and sinks
        avg_out_degree = sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0
        avg_in_degree = sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0
        
        for inst_id, out_degree in out_degrees.items():
            if out_degree > avg_out_degree * 1.5:  # 50% above average
                patterns['hub_institutions'].append((inst_id, out_degree))
        
        for inst_id, in_degree in in_degrees.items():
            if in_degree > avg_in_degree * 1.5:
                patterns['sink_institutions'].append((inst_id, in_degree))
        
        # Identify isolated institutions
        for inst_id in set(self.row_institutions + self.column_institutions):
            total_connections = out_degrees.get(inst_id, 0) + in_degrees.get(inst_id, 0)
            if total_connections <= 1:
                patterns['isolated_institutions'].append(inst_id)
        
        # Identify reciprocal pairs
        for (row_id, col_id), cell in self.matrix_cells.items():
            reverse_key = (col_id, row_id)
            if (reverse_key in self.matrix_cells and 
                cell.delivery_capacity is not None and cell.delivery_capacity > 0 and
                self.matrix_cells[reverse_key].delivery_capacity is not None and 
                self.matrix_cells[reverse_key].delivery_capacity > 0):
                patterns['reciprocal_pairs'].append((row_id, col_id))
        
        return patterns


@dataclass(kw_only=True)
class MatrixValidation(Node):
    """Validation and quality assurance processes for SFM matrices."""
    
    matrix_id: uuid.UUID
    validation_criteria: List[str] = field(default_factory=lambda: [])
    
    # Validation results
    cell_validation_results: Dict[uuid.UUID, Dict[str, Any]] = field(default_factory=lambda: {})
    matrix_level_validation: Dict[str, Any] = field(default_factory=lambda: {})
    
    # Validation process
    validation_stages: List[str] = field(default_factory=lambda: [])
    completed_validations: Set[str] = field(default_factory=lambda: set())
    validation_timeline: Dict[str, datetime] = field(default_factory=lambda: {})
    
    # Validators
    peer_reviewers: List[uuid.UUID] = field(default_factory=lambda: [])
    stakeholder_validators: List[uuid.UUID] = field(default_factory=lambda: [])
    expert_validators: List[uuid.UUID] = field(default_factory=lambda: [])
    
    # Quality metrics
    overall_validation_score: Optional[float] = None
    validation_confidence: Optional[float] = None
    identified_issues: List[Dict[str, Any]] = field(default_factory=lambda: [])
    
    def validate_matrix_completeness(self, matrix: DeliveryMatrix) -> Dict[str, Any]:
        """Validate completeness of matrix construction."""
        completeness_results = {
            'missing_cells': [],
            'incomplete_cells': [],
            'coverage_assessment': {},
            'recommendations': []
        }
        
        # Check for missing critical cells
        total_possible = len(matrix.row_institutions) * len(matrix.column_institutions)
        actual_cells = len(matrix.matrix_cells)
        
        completeness_results['coverage_assessment'] = {
            'total_possible_cells': total_possible,
            'populated_cells': actual_cells,
            'coverage_percentage': (actual_cells / total_possible * 100) if total_possible > 0 else 0
        }
        
        # Identify incomplete cells
        for cell_key, cell in matrix.matrix_cells.items():
            incompleteness_issues = []
            
            if cell.delivery_capacity is None:
                incompleteness_issues.append('Missing delivery capacity')
            
            if not cell.evidence_sources:
                incompleteness_issues.append('No evidence sources')
            
            if cell.confidence_level is None or cell.confidence_level < 0.3:
                incompleteness_issues.append('Low confidence level')
            
            if cell.validation_status == ValidationStatus.NOT_VALIDATED:
                incompleteness_issues.append('Not validated')
            
            if incompleteness_issues:
                completeness_results['incomplete_cells'].append({
                    'cell_key': cell_key,
                    'issues': incompleteness_issues
                })
        
        # Generate recommendations
        coverage_pct = completeness_results['coverage_assessment']['coverage_percentage']
        if coverage_pct < 50:
            completeness_results['recommendations'].append('Matrix coverage below 50% - consider expanding data collection')
        
        incomplete_count = len(completeness_results['incomplete_cells'])
        if incomplete_count > actual_cells * 0.3:
            completeness_results['recommendations'].append('High number of incomplete cells - focus on evidence gathering')
        
        return completeness_results
    
    def validate_matrix_consistency(self, matrix: DeliveryMatrix) -> Dict[str, Any]:
        """Validate internal consistency of matrix values."""
        consistency_results = {
            'logical_inconsistencies': [],
            'value_range_issues': [],
            'relationship_conflicts': [],
            'consistency_score': 0.0
        }
        
        consistency_issues = 0
        total_checks = 0
        
        for cell_key, cell in matrix.matrix_cells.items():
            total_checks += 1
            
            # Capacity range checks
            if cell.delivery_capacity is not None:
                if cell.delivery_capacity < 0 or cell.delivery_capacity > 1:
                    consistency_results['value_range_issues'].append({
                        'cell_key': cell_key,
                        'value': cell.delivery_capacity,
                        'issue': 'Capacity outside 0-1 range'
                    })
                    consistency_issues += 1
            
            # Quality vs capacity consistency
            if (cell.delivery_capacity is not None and cell.delivery_quality is not None and
                cell.delivery_capacity > 0 and cell.delivery_quality == 0):
                consistency_results['logical_inconsistencies'].append({
                    'cell_key': cell_key,
                    'issue': 'Non-zero capacity with zero quality'
                })
                consistency_issues += 1
            
            # Evidence vs confidence consistency
            if (cell.confidence_level is not None and cell.confidence_level > 0.8 and
                len(cell.evidence_sources) < 2):
                consistency_results['logical_inconsistencies'].append({
                    'cell_key': cell_key,
                    'issue': 'High confidence with limited evidence'
                })
                consistency_issues += 1
        
        # Calculate consistency score
        if total_checks > 0:
            consistency_results['consistency_score'] = 1.0 - (consistency_issues / total_checks)
        
        return consistency_results
    
    def conduct_stakeholder_validation(self, matrix: DeliveryMatrix, 
                                     stakeholders: List[uuid.UUID]) -> Dict[str, Any]:
        """Conduct stakeholder validation of matrix content."""
        stakeholder_results = {
            'participant_count': len(stakeholders),
            'validation_responses': {},
            'consensus_level': 0.0,
            'disputed_cells': [],
            'stakeholder_feedback': []
        }
        
        # Simplified stakeholder validation simulation
        # In practice, would involve actual stakeholder consultation
        
        for stakeholder_id in stakeholders:
            # Simulate stakeholder validation responses
            stakeholder_agreement = 0.7  # Placeholder - would be actual responses
            stakeholder_results['validation_responses'][str(stakeholder_id)] = {
                'overall_agreement': stakeholder_agreement,
                'specific_concerns': [],
                'suggested_modifications': []
            }
        
        # Calculate consensus level
        if stakeholder_results['validation_responses']:
            agreements = [resp['overall_agreement'] 
                         for resp in stakeholder_results['validation_responses'].values()]
            stakeholder_results['consensus_level'] = sum(agreements) / len(agreements)
        
        self.stakeholder_validators = stakeholders
        return stakeholder_results


@dataclass(kw_only=True)
class SFMMatrixBuilder(Node):
    """Systematic matrix construction methodology following Hayden's approach."""
    problem_definition_id: uuid.UUID
    system_boundary_id: uuid.UUID
    construction_stage: MatrixConstructionStage = MatrixConstructionStage.INITIALIZATION
    
    # Matrix construction components
    identified_institutions: List[uuid.UUID] = field(default_factory=lambda: [])
    evaluation_criteria: List[uuid.UUID] = field(default_factory=lambda: [])
    constructed_matrices: Dict[MatrixDimension, DeliveryMatrix] = field(default_factory=lambda: {})
    
    # Construction process management
    construction_timeline: Dict[MatrixConstructionStage, datetime] = field(default_factory=lambda: {})
    stage_validation_results: Dict[MatrixConstructionStage, Dict[str, Any]] = field(default_factory=lambda: {})
    construction_quality_metrics: Dict[str, float] = field(default_factory=lambda: {})
    
    # Data collection and evidence
    data_collection_plan: Dict[str, List[str]] = field(default_factory=lambda: {})
    evidence_inventory: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})
    data_collection_progress: Dict[str, float] = field(default_factory=lambda: {})
    
    # Stakeholder engagement
    matrix_construction_stakeholders: List[uuid.UUID] = field(default_factory=lambda: [])
    stakeholder_input_sessions: List[Dict[str, Any]] = field(default_factory=lambda: [])
    stakeholder_feedback_integration: Dict[str, Any] = field(default_factory=lambda: {})
    
    def initialize_matrix_construction(self, institutions: List[uuid.UUID], 
                                     criteria: List[uuid.UUID]) -> Dict[str, Any]:
        """Initialize the matrix construction process."""
        initialization_results = {
            'status': 'initialized',
            'institution_count': len(institutions),
            'criteria_count': len(criteria),
            'estimated_cells': len(institutions) * len(criteria),
            'construction_plan': {}
        }
        
        self.identified_institutions = institutions
        self.evaluation_criteria = criteria
        self.construction_stage = MatrixConstructionStage.INSTITUTION_MAPPING
        self.construction_timeline[MatrixConstructionStage.INITIALIZATION] = datetime.now()
        
        # Create main institution-criteria matrix
        main_matrix = DeliveryMatrix(
            label="Primary Institution-Criteria Matrix",
            matrix_dimension=MatrixDimension.INSTITUTION_CRITERIA,
            row_institutions=institutions.copy(),
            column_institutions=criteria.copy()  # In institution-criteria matrix, criteria are columns
        )
        self.constructed_matrices[MatrixDimension.INSTITUTION_CRITERIA] = main_matrix
        
        # Develop construction plan
        construction_plan = {
            'data_collection_methods': self._plan_data_collection(institutions, criteria),
            'validation_approach': self._plan_validation_approach(),
            'stakeholder_engagement': self._plan_stakeholder_engagement(),
            'quality_assurance': self._plan_quality_assurance()
        }
        initialization_results['construction_plan'] = construction_plan
        
        return initialization_results
    
    def populate_matrix_cells(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Populate matrix cells with data from various sources."""
        population_results = {
            'cells_populated': 0,
            'data_source_utilization': {},
            'quality_distribution': {},
            'population_issues': []
        }
        
        main_matrix = self.constructed_matrices.get(MatrixDimension.INSTITUTION_CRITERIA)
        if not main_matrix:
            population_results['population_issues'].append('No main matrix initialized')
            return population_results
        
        # Populate cells
        for institution_id in self.identified_institutions:
            for criteria_id in self.evaluation_criteria:
                cell_key = (institution_id, criteria_id)
                
                # Create matrix cell
                cell = MatrixCell(
                    label=f"Cell-{institution_id}-{criteria_id}",
                    row_element_id=institution_id,
                    column_element_id=criteria_id,
                    matrix_dimension=MatrixDimension.INSTITUTION_CRITERIA
                )
                
                # Populate cell data from sources
                cell_data = self._extract_cell_data(institution_id, criteria_id, data_sources)
                if cell_data:
                    cell.delivery_capacity = cell_data.get('value')
                    cell.delivery_quality = cell_data.get('quality')
                    cell.confidence_level = cell_data.get('confidence', 0.5)
                    cell.evidence_sources = cell_data.get('evidence_sources', {})
                    cell.quantification_method = cell_data.get('method')
                    
                    population_results['cells_populated'] += 1
                
                main_matrix.matrix_cells[cell_key] = cell
        
        # Calculate population metrics
        total_cells = len(self.identified_institutions) * len(self.evaluation_criteria)
        population_rate = population_results['cells_populated'] / total_cells if total_cells > 0 else 0
        
        self.construction_quality_metrics['population_rate'] = population_rate
        self.construction_stage = MatrixConstructionStage.DATA_POPULATION
        self.construction_timeline[MatrixConstructionStage.DATA_POPULATION] = datetime.now()
        
        return population_results
    
    def validate_matrix_construction(self) -> Dict[str, Any]:
        """Validate the constructed matrix."""
        validation_results = {
            'completeness_validation': {},
            'consistency_validation': {},
            'stakeholder_validation': {},
            'overall_validation_score': 0.0,
            'validation_recommendations': []
        }
        
        main_matrix = self.constructed_matrices.get(MatrixDimension.INSTITUTION_CRITERIA)
        if not main_matrix:
            validation_results['validation_recommendations'].append('No matrix to validate')
            return validation_results
        
        # Create validation instance
        validator = MatrixValidation(
            label="Matrix Construction Validation",
            matrix_id=main_matrix.id,
            validation_criteria=[
                'completeness',
                'consistency', 
                'stakeholder_acceptance',
                'evidence_quality'
            ]
        )
        
        # Completeness validation
        validation_results['completeness_validation'] = validator.validate_matrix_completeness(main_matrix)
        
        # Consistency validation
        validation_results['consistency_validation'] = validator.validate_matrix_consistency(main_matrix)
        
        # Stakeholder validation
        if self.matrix_construction_stakeholders:
            validation_results['stakeholder_validation'] = validator.conduct_stakeholder_validation(
                main_matrix, self.matrix_construction_stakeholders
            )
        
        # Calculate overall validation score
        scores = []
        completeness_score = validation_results['completeness_validation'].get('coverage_assessment', {}).get('coverage_percentage', 0) / 100
        scores.append(completeness_score)
        
        consistency_score = validation_results['consistency_validation'].get('consistency_score', 0)
        scores.append(consistency_score)
        
        if 'stakeholder_validation' in validation_results:
            stakeholder_score = validation_results['stakeholder_validation'].get('consensus_level', 0)
            scores.append(stakeholder_score)
        
        if scores:
            validation_results['overall_validation_score'] = sum(scores) / len(scores)
            self.construction_quality_metrics['validation_score'] = validation_results['overall_validation_score']
        
        self.construction_stage = MatrixConstructionStage.VALIDATION
        self.construction_timeline[MatrixConstructionStage.VALIDATION] = datetime.now()
        
        return validation_results
    
    def _plan_data_collection(self, institutions: List[uuid.UUID], 
                            criteria: List[uuid.UUID]) -> Dict[str, List[str]]:
        """Plan data collection methods for matrix population."""
        return {
            'quantitative_methods': [
                'Statistical databases',
                'Performance indicators',
                'Survey data',
                'Observational measurements'
            ],
            'qualitative_methods': [
                'Expert interviews',
                'Stakeholder consultations',
                'Document analysis',
                'Case studies'
            ],
            'mixed_methods': [
                'Multi-method validation',
                'Triangulation approaches',
                'Sequential data collection'
            ]
        }
    
    def _plan_validation_approach(self) -> Dict[str, List[str]]:
        """Plan validation approach for matrix construction."""
        return {
            'internal_validation': [
                'Logical consistency checks',
                'Data quality assessment',
                'Completeness verification'
            ],
            'external_validation': [
                'Peer review',
                'Stakeholder validation',
                'Expert panel review'
            ],
            'empirical_validation': [
                'Predictive testing',
                'Cross-case validation',
                'Longitudinal verification'
            ]
        }
    
    def _plan_stakeholder_engagement(self) -> Dict[str, List[str]]:
        """Plan stakeholder engagement for matrix construction."""
        return {
            'identification_phase': [
                'Stakeholder mapping',
                'Influence-interest analysis',
                'Engagement strategy development'
            ],
            'consultation_phase': [
                'Individual interviews',
                'Focus groups',
                'Workshop sessions'
            ],
            'validation_phase': [
                'Matrix review sessions',
                'Feedback collection',
                'Consensus building'
            ]
        }
    
    def _plan_quality_assurance(self) -> Dict[str, List[str]]:
        """Plan quality assurance for matrix construction."""
        return {
            'data_quality': [
                'Source verification',
                'Measurement validation',
                'Reliability assessment'
            ],
            'process_quality': [
                'Methodology adherence',
                'Documentation standards',
                'Peer review processes'
            ],
            'outcome_quality': [
                'Matrix completeness',
                'Logical consistency',
                'Stakeholder acceptance'
            ]
        }
    
    def _extract_cell_data(self, institution_id: uuid.UUID, criteria_id: uuid.UUID, 
                          data_sources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract cell data from available data sources."""
        # Simplified data extraction - in practice would be more sophisticated
        cell_data = {
            'value': 0.5,  # Placeholder
            'quality': 0.7,  # Placeholder
            'confidence': 0.6,  # Placeholder
            'evidence_sources': {CellEvidence.QUANTITATIVE_DATA: ['Placeholder data']},
            'method': DeliveryQuantificationMethod.VOLUME_BASED
        }
        return cell_data
    
    def generate_construction_report(self) -> Dict[str, Any]:
        """Generate comprehensive report on matrix construction process."""
        report = {
            'construction_overview': {
                'current_stage': self.construction_stage.name,
                'institutions_count': len(self.identified_institutions),
                'criteria_count': len(self.evaluation_criteria),
                'matrices_constructed': len(self.constructed_matrices),
                'construction_duration': None
            },
            'quality_metrics': self.construction_quality_metrics,
            'stage_progress': {},
            'validation_summary': {},
            'recommendations': []
        }
        
        # Calculate construction duration
        if (MatrixConstructionStage.INITIALIZATION in self.construction_timeline and
            self.construction_stage != MatrixConstructionStage.INITIALIZATION):
            start_time = self.construction_timeline[MatrixConstructionStage.INITIALIZATION]
            current_time = datetime.now()
            report['construction_overview']['construction_duration'] = current_time - start_time
        
        # Stage progress
        for stage in MatrixConstructionStage:
            stage_info = {
                'completed': stage in self.construction_timeline,
                'completion_date': self.construction_timeline.get(stage),
                'validation_results': self.stage_validation_results.get(stage, {})
            }
            report['stage_progress'][stage.name] = stage_info
        
        # Generate recommendations
        if self.construction_quality_metrics.get('population_rate', 0) < 0.7:
            report['recommendations'].append('Low population rate - consider additional data collection')
        
        if self.construction_quality_metrics.get('validation_score', 0) < 0.6:
            report['recommendations'].append('Low validation score - address identified quality issues')
        
        return report


@dataclass
class MatrixAnalyzer(Node):
    """Comprehensive analysis tools for constructed SFM matrices."""
    
    analyzed_matrices: List[uuid.UUID] = field(default_factory=lambda: [])
    analysis_methods: List[AnalyticalMethod] = field(default_factory=lambda: [])
    
    def analyze_delivery_patterns(self, matrix: DeliveryMatrix) -> Dict[str, Any]:
        """Analyze delivery patterns in the matrix."""
        analysis_results = {
            'pattern_summary': {},
            'institutional_roles': {},
            'delivery_flows': {},
            'structural_properties': {}
        }
        
        # Basic matrix metrics
        matrix_metrics = matrix.calculate_matrix_metrics()
        analysis_results['pattern_summary'] = matrix_metrics
        
        # Delivery pattern analysis
        delivery_patterns = matrix.analyze_delivery_patterns()
        analysis_results['delivery_flows'] = delivery_patterns
        
        # Institutional role analysis
        for institution_id in set(matrix.row_institutions + matrix.column_institutions):
            role_analysis = self._analyze_institutional_role(institution_id, matrix)
            analysis_results['institutional_roles'][str(institution_id)] = role_analysis
        
        return analysis_results
    
    def _analyze_institutional_role(self, institution_id: uuid.UUID, 
                                  matrix: DeliveryMatrix) -> Dict[str, Any]:
        """Analyze the role of a specific institution in the matrix."""
        role_analysis = {
            'delivery_provision': 0.0,    # How much this institution delivers
            'delivery_reception': 0.0,    # How much this institution receives
            'centrality': 0.0,            # How central in the network
            'specialization': 0.0         # How specialized the deliveries
        }
        
        # Calculate provision (out-degree strength)
        provision_total = 0.0
        provision_count = 0
        for (row_id, col_id), cell in matrix.matrix_cells.items():
            if row_id == institution_id and cell.delivery_capacity is not None:
                provision_total += cell.delivery_capacity
                provision_count += 1
        
        if provision_count > 0:
            role_analysis['delivery_provision'] = provision_total / provision_count
        
        # Calculate reception (in-degree strength)
        reception_total = 0.0
        reception_count = 0
        for (row_id, col_id), cell in matrix.matrix_cells.items():
            if col_id == institution_id and cell.delivery_capacity is not None:
                reception_total += cell.delivery_capacity
                reception_count += 1
        
        if reception_count > 0:
            role_analysis['delivery_reception'] = reception_total / reception_count
        
        # Simple centrality measure (degree centrality)
        total_connections = provision_count + reception_count
        max_possible = len(matrix.row_institutions) + len(matrix.column_institutions) - 1
        if max_possible > 0:
            role_analysis['centrality'] = total_connections / max_possible
        
        return role_analysis